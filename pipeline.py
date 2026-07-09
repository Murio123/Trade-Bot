"""Analysis pipeline: gather all market data, build a unified context, and run
the 7-level signal cascade.

This is the glue that wires the analyzers, the signal engine, risk sizing and
the Claude interpreter together. It returns a fully-formed signal dict (or a
``blocked`` result explaining where the cascade stopped).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import config
from analyzer.binance import BinanceClient
from analyzer.correlation import get_correlations
from analyzer.cvd import compute_cvd_from_klines, cvd_series
from analyzer.equilibrium import compute_equilibrium
from analyzer.reversal import detect_reversal
from analyzer.divergence import detect_divergence
from analyzer.fvg import detect_fvg
from analyzer.historical import find_analogues
from analyzer.indicators import compute_indicators
from analyzer.structure import analyze_structure
from analyzer.volatility import analyze_volatility
from analyzer.liquidation_map import get_liquidation_map, nearest_sweep_signal
from analyzer.liquidity import detect_liquidity
from analyzer.macro import get_macro
from analyzer.onchain import get_onchain
from analyzer.order_blocks import detect_order_blocks
from analyzer.session_stats import compute_session_stats
from analyzer.volume_profile import compute_volume_profile
from ai import claude
from risk.position_sizing import calculate_position
from signal_engine.confluence import (calculate_confluence_score,
                                       has_diverse_confirmation)
from signal_engine.conflict_resolver import resolve_conflicts
from signal_engine.cooldown import should_send_signal
from signal_engine.daily_limiter import within_daily_limit
from signal_engine.htf_filter import apply_htf_policy, get_htf_bias
from signal_engine.mtf_confidence import mtf_confidence_factor, trend_label
from signal_engine.no_trade_gate import (bad_risk_reward,
                                         effective_expected_move,
                                         insufficient_expected_move,
                                         invalid_tp2, low_confidence,
                                         missing_invalidation,
                                         position_conflict, tf_conflict)
from signal_engine.profiles import get_profile
from signal_engine.regime import detect_regime, weighted_total
from signal_engine.schema import STATUS_MAP, build_result
from signal_engine.vetoes import (TF_HOURS, abnormal_volatility,
                                  crowded_funding, dead_zone, stale_data)

log = logging.getLogger(__name__)


async def gather_market_context(binance: BinanceClient,
                                signal_timeframe: str | None = None,
                                profile_name: str | None = None) -> dict[str, Any]:
    """Fetch raw data from every source and compute all analyzer outputs.

    Fetches the union of timeframes the active profile needs: always 1h/4h/1d
    (HTF bias, sessions, fallbacks) plus the entry timeframe and every MTF
    timeframe (e.g. 12h for swing, 15m for intraday).
    """
    profile = get_profile(profile_name)
    signal_timeframe = signal_timeframe or profile["entry"]

    # 1h/4h/12h/1d always fetched: HTF bias, sessions, and the 4-TF reversal read.
    needed = {"1h", "4h", "12h", "1d", signal_timeframe} | set(profile["mtf"])
    tfs = sorted(needed)
    frames = await asyncio.gather(*[binance.klines(tf, limit=300) for tf in tfs])
    # Closed candles only: the exchange includes the forming bar as the last row.
    dfs = {tf: _drop_unclosed(df) for tf, df in zip(tfs, frames)}
    inds = {tf: compute_indicators(df) for tf, df in dfs.items()}

    df_1h, df_4h, df_1d = dfs["1h"], dfs["4h"], dfs["1d"]
    ind_1h, ind_4h, ind_1d = inds["1h"], inds["4h"], inds["1d"]
    df_signal = dfs[signal_timeframe]
    ind_signal = inds[signal_timeframe]

    # Crypto-specific + external sources (run concurrently, tolerate failures).
    (funding, oi, ls_ratio, macro, onchain) = await asyncio.gather(
        binance.funding_rate(),
        binance.open_interest(),
        binance.long_short_ratio(),
        get_macro(),
        get_onchain(),
        return_exceptions=True,
    )

    funding = _safe(funding, {})
    oi = _safe(oi, 0.0)
    ls_ratio = _safe(ls_ratio, {})
    macro = _safe(macro, {})
    onchain = _safe(onchain, {})

    # CVD per-candle over the signal timeframe (exact from taker volume when the
    # exchange provides it, otherwise an OHLCV-based estimate).
    cvd = compute_cvd_from_klines(df_signal)

    # Zones (Order Blocks / FVG / targets) from higher timeframes; the entry
    # timeframe supplies the liquidity sweep and the rejection-wick trigger.
    atr_value = ind_signal.get("atr") or 0.0
    zone_tfs = [t for t in (profile.get("zone_tfs") or [signal_timeframe]) if t in dfs]
    if not zone_tfs:
        zone_tfs = [signal_timeframe]
    zones = _build_htf_zones(dfs, inds, zone_tfs, ind_signal["price"], df_signal)
    order_blocks = zones["order_blocks"]
    fvg = zones["fvg"]
    htf_levels = zones["levels"]
    liquidity = detect_liquidity(df_signal, atr_value=atr_value)
    volume_profile = compute_volume_profile(dfs.get(zone_tfs[0], df_signal))
    # Is price at a key HTF level (inside an OB/FVG or near equal highs/lows)?
    at_level = _near_key_level(ind_signal["price"], htf_levels, order_blocks, fvg,
                               max(atr_value * 0.3, ind_signal["price"] * 0.002))
    reversal = detect_reversal(df_signal, ind_signal, cvd_series(df_signal),
                               at_key_level=at_level)
    # Multi-timeframe reversal read across 1H / 4H / 12H / 1D.
    reversal_mtf = _reversal_mtf(dfs, inds, ind_signal["price"],
                                 htf_levels, order_blocks, fvg)

    # Premium/Discount of the dealing range, on the trend (HTF) timeframe.
    range_tf = profile["htf"] if profile["htf"] in dfs else (
        zone_tfs[0] if zone_tfs else signal_timeframe)
    equilibrium = compute_equilibrium(dfs.get(range_tf, df_signal))

    # Divergence on both RSI(14) and MACD; combine (either one counts).
    series = ind_signal.get("_series", {})
    div_rsi = detect_divergence(df_signal, series.get("rsi"))
    div_macd = detect_divergence(df_signal, series.get("macd"))
    divergence = {
        "bullish_divergence": bool(div_rsi.get("bullish_divergence")
                                   or div_macd.get("bullish_divergence")),
        "bearish_divergence": bool(div_rsi.get("bearish_divergence")
                                   or div_macd.get("bearish_divergence")),
        "rsi": div_rsi,
        "macd": div_macd,
    }

    liq_map = await get_liquidation_map(
        ind_signal["price"], float(oi) if oi else 0.0,
        ls_ratio.get("ratio"), config.SYMBOL_DISPLAY,
    )
    sweep = nearest_sweep_signal(liq_map, ind_signal["price"], atr_value)

    sessions = compute_session_stats(df_1h)

    # Volatility read on the entry timeframe (abnormal-volatility gate) and
    # the freshness stamp of the last closed candle (stale-data gate).
    tf_hours = TF_HOURS.get(signal_timeframe, 1.0)
    volatility = analyze_volatility(df_signal, tf_per_day=24.0 / tf_hours)
    volatility_1d = analyze_volatility(df_1d, tf_per_day=1.0)
    last_close = df_signal["close_time"].iloc[-1] if len(df_signal) else None

    context = {
        "symbol": config.SYMBOL,
        "timeframe": signal_timeframe,
        "timestamp": datetime.now(timezone.utc),
        "price": ind_signal["price"],
        "atr": atr_value,
        "ind_1h": ind_1h,
        "ind_4h": ind_4h,
        "ind_1d": ind_1d,
        "ind_signal": ind_signal,
        # Raw entry-timeframe candles (for chart rendering).
        "df_signal": df_signal,
        # Indicator snapshots keyed by timeframe (incl. 12h / 15m when fetched),
        # used by the cascade to resolve a profile's HTF/MTF references.
        "inds_by_tf": inds,
        "funding": funding,
        "open_interest": oi,
        "long_short_ratio": ls_ratio,
        "cvd": cvd,
        "macro": macro,
        "onchain": onchain,
        "order_blocks": order_blocks,
        "liquidity": liquidity,
        "volume_profile": volume_profile,
        "fvg": fvg,
        "htf_levels": htf_levels,
        "zone_tfs": zone_tfs,
        "equilibrium": equilibrium,
        "reversal": reversal,
        "reversal_mtf": reversal_mtf,
        "divergence": divergence,
        "liquidation_map": liq_map,
        "sweep_signal": sweep,
        "sessions": sessions,
        "volatility": volatility,
        "volatility_1d": volatility_1d,
        "last_close_time": last_close,
    }

    # Decision snapshot: the reproducible forecast price is the CLOSE of the
    # signal candle (ctx["price"]); the executable price is the live ticker at
    # decision time. They are recorded separately and never mixed.
    executable = None
    try:
        executable = await binance.current_price()
    except Exception as exc:  # noqa: BLE001
        log.warning("current_price unavailable, executable price degraded "
                    "to signal close: %s", exc)
    context["decision_time"] = datetime.now(timezone.utc)
    context["executable_price"] = (float(executable) if executable
                                   else ind_signal["price"])
    context["executable_price_degraded"] = executable is None
    return context


def _drop_unclosed(df):
    """Drop the still-forming last candle the exchanges include in klines.

    Indicators computed on a partial candle repaint (a mid-hour "hammer" can
    close as a full bearish bar), so live analysis must see closed bars only —
    exactly what the backtest walks.
    """
    import pandas as pd
    if df is None or len(df) == 0 or "close_time" not in df.columns:
        return df
    if df["close_time"].iloc[-1] > pd.Timestamp.now(tz="UTC"):
        return df.iloc[:-1].reset_index(drop=True)
    return df


def _entry_rejection(df) -> bool:
    """Significant rejection wick on the current (entry-timeframe) candle."""
    if df is None or len(df) == 0:
        return False
    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    rng = max(h - l, 1e-9)
    lower = min(o, c) - l
    upper = h - max(o, c)
    return (lower / rng > 0.4) or (upper / rng > 0.4)


def _build_htf_zones(dfs: dict[str, Any], inds: dict[str, Any], zone_tfs: list[str],
                     price: float, entry_df) -> dict[str, Any]:
    """Detect Order Blocks / FVG and key levels on higher timeframes.

    Zones come from the HTF set (e.g. 12H+4H); the entry timeframe only
    supplies the rejection wick that confirms a reaction inside an HTF zone.
    """
    ob = {"bullish_ob": None, "bearish_ob": None, "price_in_bullish_ob": False,
          "price_in_bearish_ob": False, "rejection_wick": _entry_rejection(entry_df),
          "blocks": [], "zone_tfs": zone_tfs}
    fvg = {"bullish_fvg": None, "bearish_fvg": None, "price_in_bullish_fvg": False,
           "price_in_bearish_fvg": False, "fvgs": []}
    highs: list[float] = []
    lows: list[float] = []

    for tf in zone_tfs:
        df = dfs.get(tf)
        if df is None:
            continue
        atr = inds.get(tf, {}).get("atr")
        ob_z = detect_order_blocks(df)
        for blk in ob_z.get("blocks", []):
            tagged = {**blk, "tf": tf}
            ob["blocks"].append(tagged)
            if blk["low"] <= price <= blk["high"]:
                if blk["type"] == "bullish":
                    ob["price_in_bullish_ob"] = True
                    ob["bullish_ob"] = ob["bullish_ob"] or tagged
                else:
                    ob["price_in_bearish_ob"] = True
                    ob["bearish_ob"] = ob["bearish_ob"] or tagged
        if ob_z.get("bullish_ob") and not ob["bullish_ob"]:
            ob["bullish_ob"] = {**ob_z["bullish_ob"], "tf": tf}
        if ob_z.get("bearish_ob") and not ob["bearish_ob"]:
            ob["bearish_ob"] = {**ob_z["bearish_ob"], "tf": tf}

        fvg_z = detect_fvg(df, atr_value=atr)
        for z in fvg_z.get("fvgs", []):
            tagged = {**z, "tf": tf}
            fvg["fvgs"].append(tagged)
            if z["low"] <= price <= z["high"]:
                if z["type"] == "bullish":
                    fvg["price_in_bullish_fvg"] = True
                    fvg["bullish_fvg"] = fvg["bullish_fvg"] or tagged
                else:
                    fvg["price_in_bearish_fvg"] = True
                    fvg["bearish_fvg"] = fvg["bearish_fvg"] or tagged
        if fvg_z.get("bullish_fvg") and not fvg["bullish_fvg"]:
            fvg["bullish_fvg"] = {**fvg_z["bullish_fvg"], "tf": tf}
        if fvg_z.get("bearish_fvg") and not fvg["bearish_fvg"]:
            fvg["bearish_fvg"] = {**fvg_z["bearish_fvg"], "tf": tf}

        # Key levels for HTF targets: equal highs/lows + recent extremes.
        liq_z = detect_liquidity(df, atr_value=atr)
        highs += liq_z.get("equal_highs", [])
        lows += liq_z.get("equal_lows", [])
        recent = df.iloc[-60:]
        highs.append(float(recent["high"].max()))
        lows.append(float(recent["low"].min()))

    levels = {
        "highs": sorted({round(h, 2) for h in highs}),
        "lows": sorted({round(l, 2) for l in lows}),
    }
    return {"order_blocks": ob, "fvg": fvg, "levels": levels}


def _stop_atr(profile: dict[str, Any], inds: dict[str, Any], entry_atr: float) -> float:
    """ATR used for the stop distance.

    Profiles may anchor the stop to a higher timeframe's ATR (intraday: 1H) —
    an entry-timeframe 15m ATR produces stops so tight that round-trip fees
    consume most of an R.
    """
    tf = profile.get("stop_tf")
    if tf:
        v = (inds.get(tf) or {}).get("atr")
        if v:
            return v
    return entry_atr


def _structural_stop(ctx: dict[str, Any], direction: str, entry: float,
                     atr: float, profile: dict[str, Any]) -> float | None:
    """Stop behind the nearest HTF structure (level or OB edge) + 0.5 ATR.

    Only for profiles with structural_stop (swing). Falls back to None — i.e.
    keep the ATR stop — when no structure sits below/above, when the
    structural stop would be TIGHTER than the ATR stop (no protection), or
    when it is absurdly far (> 8 ATR, unusable sizing).
    """
    if not profile.get("structural_stop") or not atr:
        return None
    levels = ctx.get("htf_levels", {}) or {}
    ob = ctx.get("order_blocks", {}) or {}
    buf = atr * 0.5

    if direction == "long":
        cands = [l for l in levels.get("lows", []) if l < entry]
        z = ob.get("bullish_ob")
        if z and z.get("low", entry) < entry:
            cands.append(z["low"])
        if not cands:
            return None
        stop = max(cands) - buf
        risk = entry - stop
    else:
        cands = [h for h in levels.get("highs", []) if h > entry]
        z = ob.get("bearish_ob")
        if z and z.get("high", entry) > entry:
            cands.append(z["high"])
        if not cands:
            return None
        stop = min(cands) + buf
        risk = stop - entry

    if risk < atr * profile["atr_mult"] or risk > atr * 8:
        return None
    return stop


def _structure_targets(ctx: dict[str, Any], direction: str, entry: float, risk: float,
                       fb1: float, fb2: float) -> tuple[float, float, bool, str]:
    """Targets at the nearest HTF liquidity / volume nodes, else ATR fallback.

    Returns (tp1, tp2, tp1_is_structural, tp2_source). tp2_source names the
    real provenance of TP2: a single structural level still yields a
    SYNTHETIC 3R TP2, which must be labelled r_multiple_fallback — not passed
    off as structure.
    """
    levels = ctx.get("htf_levels", {})
    vp = ctx.get("volume_profile", {})
    pts: list[tuple[float, str]] = []
    if direction == "long":
        pts += [(x, "structural") for x in levels.get("highs", []) if x > entry]
        for k in ("vah", "poc"):
            v = vp.get(k)
            if v and v > entry:
                pts.append((v, "volume_profile"))
        pts = sorted({(round(p, 2), src) for p, src in pts})
    else:
        pts += [(x, "structural") for x in levels.get("lows", []) if x < entry]
        for k in ("val", "poc"):
            v = vp.get(k)
            if v and v < entry:
                pts.append((v, "volume_profile"))
        pts = sorted({(round(p, 2), src) for p, src in pts}, reverse=True)

    pts = [(p, src) for p, src in pts if abs(p - entry) >= risk]  # at least 1R away
    if not pts:
        return fb1, fb2, False, "r_multiple_fallback"
    tp1 = pts[0][0]
    if len(pts) > 1:
        tp2, tp2_source = pts[1]
    else:
        tp2 = round(entry + risk * 3 * (1 if direction == "long" else -1), 2)
        tp2_source = "r_multiple_fallback"
    return tp1, tp2, True, tp2_source


def _near_key_level(price: float, levels: dict, ob: dict, fvg: dict, tol: float) -> bool:
    """Whether price sits inside an HTF OB/FVG or near an HTF equal high/low."""
    if ob.get("price_in_bullish_ob") or ob.get("price_in_bearish_ob"):
        return True
    if fvg.get("price_in_bullish_fvg") or fvg.get("price_in_bearish_fvg"):
        return True
    for lv in (levels.get("highs", []) + levels.get("lows", [])):
        if abs(price - lv) <= tol:
            return True
    return False


REVERSAL_TFS = ["1h", "4h", "12h", "1d"]


def _reversal_mtf(dfs: dict[str, Any], inds: dict[str, Any], price: float,
                  levels: dict, ob: dict, fvg: dict) -> dict[str, Any]:
    """Run reversal detection on 1H/4H/12H/1D and combine into one verdict.

    The "at key level" tolerance scales with EACH timeframe's own ATR — a
    15m-ATR tolerance applied to a daily reversal read would call almost
    nothing "at a level" (or, with the 0.2% floor, the wrong things).
    """
    per_tf: dict[str, Any] = {}
    for tf in REVERSAL_TFS:
        if tf in dfs:
            atr_tf = (inds.get(tf) or {}).get("atr") or 0.0
            tol = max(atr_tf * 0.3, price * 0.002)
            at_lvl = _near_key_level(price, levels, ob, fvg, tol)
            per_tf[tf] = detect_reversal(dfs[tf], inds[tf], cvd_series(dfs[tf]),
                                         at_key_level=at_lvl)
    bull_tfs = [tf for tf in REVERSAL_TFS if per_tf.get(tf, {}).get("bullish_reversal")]
    bear_tfs = [tf for tf in REVERSAL_TFS if per_tf.get(tf, {}).get("bearish_reversal")]

    # Confirmation candle on 1H: don't call a bottom while the last closed
    # hourly candle is still falling (knife-catching guard), and vice versa.
    bull_candle = bear_candle = False
    df_1h = dfs.get("1h")
    if df_1h is not None and len(df_1h) > 0:
        last = df_1h.iloc[-1]
        bull_candle = float(last["close"]) > float(last["open"])
        bear_candle = float(last["close"]) < float(last["open"])

    return {
        "per_tf": per_tf,
        "bull_tfs": bull_tfs,
        "bear_tfs": bear_tfs,
        "bull_tf_count": len(bull_tfs),
        "bear_tf_count": len(bear_tfs),
        "combined_bullish": len(bull_tfs) >= 2,
        "combined_bearish": len(bear_tfs) >= 2,
        "bull_candle_confirm": bull_candle,
        "bear_candle_confirm": bear_candle,
    }


def _oi_rising(oi_hist: list[dict[str, Any]]) -> bool | None:
    """Detect whether open interest is trending up over the history window."""
    if not oi_hist or len(oi_hist) < 2:
        return None
    def _val(item: dict[str, Any]) -> float | None:
        for key in ("sumOpenInterest", "openInterest", "sumOpenInterestValue"):
            if key in item:
                try:
                    return float(item[key])
                except (TypeError, ValueError):
                    return None
        return None
    first, last = _val(oi_hist[0]), _val(oi_hist[-1])
    if first is None or last is None or first == 0:
        return None
    return last > first


async def gather_swing_context(binance: BinanceClient) -> dict[str, Any]:
    """Assemble the advanced swing context across 1D / 12H / 4H.

    Trend on 1D, zones of interest on 12H, entry confirmation on 4H, plus
    derivatives, volatility, real historical analogues and correlations.
    """
    df_4h, df_12h, df_1d = await asyncio.gather(
        binance.klines("4h", limit=500),
        binance.klines("12h", limit=400),
        binance.klines("1d", limit=400),
    )
    df_4h, df_12h, df_1d = (_drop_unclosed(df_4h), _drop_unclosed(df_12h),
                            _drop_unclosed(df_1d))
    ind_4h = compute_indicators(df_4h)
    ind_12h = compute_indicators(df_12h)
    ind_1d = compute_indicators(df_1d)
    atr_value = ind_4h.get("atr") or 0.0

    # Derivatives + correlations (tolerate failures).
    (funding, oi, oi_hist, ls_ratio, corr) = await asyncio.gather(
        binance.funding_rate(),
        binance.open_interest(),
        binance.open_interest_hist(period="4h", limit=30),
        binance.long_short_ratio(),
        get_correlations(binance, df_1d),
        return_exceptions=True,
    )
    funding = _safe(funding, {})
    oi = _safe(oi, 0.0)
    oi_hist = _safe(oi_hist, [])
    ls_ratio = _safe(ls_ratio, {})
    corr = _safe(corr, {})

    cvd = compute_cvd_from_klines(df_4h)

    # Zones of interest computed on the 12H frame.
    order_blocks = detect_order_blocks(df_12h)
    fvg = detect_fvg(df_12h, atr_value=ind_12h.get("atr") or atr_value)
    liquidity = detect_liquidity(df_12h, atr_value=ind_12h.get("atr") or atr_value)
    volume_profile = compute_volume_profile(df_12h)

    historical = find_analogues(df_4h, horizon=42, tf_hours=4.0)
    volatility = analyze_volatility(df_4h, tf_per_day=6.0)
    # Reversal read on the 4H entry frame.
    reversal = detect_reversal(df_4h, ind_4h, cvd_series(df_4h))

    return {
        "symbol": config.SYMBOL,
        "timestamp": datetime.now(timezone.utc),
        "price": ind_4h["price"],
        "atr": atr_value,
        "ind_1d": ind_1d,
        "ind_12h": ind_12h,
        "ind_4h": ind_4h,
        "structure_1d": analyze_structure(df_1d),
        "structure_12h": analyze_structure(df_12h),
        "structure_4h": analyze_structure(df_4h),
        "order_blocks": order_blocks,
        "fvg": fvg,
        "liquidity": liquidity,
        "volume_profile": volume_profile,
        "reversal": reversal,
        "funding": funding,
        "open_interest": oi,
        "oi_rising": _oi_rising(oi_hist),
        "long_short_ratio": ls_ratio,
        "cvd": cvd,
        "volatility": volatility,
        "historical": historical,
        "correlation": corr,
    }


def _flatten_for_confluence(ctx: dict[str, Any]) -> dict[str, Any]:
    """Build the flat dict the confluence scorer consumes."""
    ind = ctx["ind_signal"]
    flat = dict(ind)  # rsi, macd flags, ema alignment, volume, avg_volume, atr...
    flat.update({
        "bullish_divergence": ctx["divergence"].get("bullish_divergence"),
        "bearish_divergence": ctx["divergence"].get("bearish_divergence"),
        "price_in_bullish_ob": ctx["order_blocks"].get("price_in_bullish_ob"),
        "price_in_bearish_ob": ctx["order_blocks"].get("price_in_bearish_ob"),
        "rejection_wick": ctx["order_blocks"].get("rejection_wick"),
        "liquidity_swept_below": ctx["liquidity"].get("liquidity_swept_below"),
        "liquidity_swept_above": ctx["liquidity"].get("liquidity_swept_above"),
        "reversal_candle": ctx["liquidity"].get("reversal_candle"),
        "price_in_bullish_fvg": ctx.get("fvg", {}).get("price_in_bullish_fvg"),
        "price_in_bearish_fvg": ctx.get("fvg", {}).get("price_in_bearish_fvg"),
        "in_discount": ctx.get("equilibrium", {}).get("zone") == "discount",
        "in_premium": ctx.get("equilibrium", {}).get("zone") == "premium",
        "bullish_reversal": ctx.get("reversal", {}).get("bullish_reversal"),
        "bearish_reversal": ctx.get("reversal", {}).get("bearish_reversal"),
        "reversal_strong_bull": ctx.get("reversal", {}).get("bull_strong"),
        "reversal_strong_bear": ctx.get("reversal", {}).get("bear_strong"),
        "cvd_bullish": ctx["cvd"].get("cvd_bullish"),
        "cvd_bearish": ctx["cvd"].get("cvd_bearish"),
        "funding": (ctx["funding"] or {}).get("current"),
        "funding_z": (ctx["funding"] or {}).get("zscore"),
        "exchange_netflow": (ctx["onchain"] or {}).get("exchange_netflow"),
    })
    return flat


async def run_cascade(ctx: dict[str, Any], delivered_today: list[dict[str, Any]],
                      last_signal: dict[str, Any] | None,
                      interpret: bool = True,
                      profile_name: str | None = None,
                      open_trades: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run levels 1-7 and return a result describing the outcome."""
    profile = get_profile(profile_name)
    inds = ctx.get("inds_by_tf", {})

    # Decision snapshot attached to EVERY result (blocked ones included):
    # signal_close_price = reproducible forecast anchor (close of the signal
    # candle), executable_price_at_decision = the live ticker when the
    # decision was made. Never interchangeable.
    snapshot = _decision_snapshot(ctx)
    freshness = snapshot["data_freshness_seconds"]

    def blocked(stage: str, **extra: Any) -> dict[str, Any]:
        return _blocked(stage, **{**snapshot, **extra})

    # Hard data-quality gates BEFORE any scoring: stale klines or a
    # volatility blow-off mean NO_TRADE regardless of how good the setup
    # looks — the numbers it is built on cannot be trusted.
    if stale_data(ctx.get("last_close_time"), ctx["timeframe"]):
        return blocked("stale_data", price=ctx.get("price"),
                       last_close_time=ctx.get("last_close_time"))
    if abnormal_volatility(ctx.get("volatility")):
        return blocked("abnormal_volatility", price=ctx.get("price"),
                       atr_percentile=(ctx.get("volatility") or {}).get("atr_percentile"))

    flat = _flatten_for_confluence(ctx)
    atr_value = ctx["atr"] or 0.0

    # Level 1: HTF bias from the profile's higher timeframe.
    htf_ind = inds.get(profile["htf"], ctx["ind_1d"])
    htf_bias = get_htf_bias(htf_ind)

    # Trend + reversal linkage: a multi-TF bottom forming WITH a bullish HTF
    # trend is a pullback entry — the prime setup (mirror for tops). The flags
    # feed the confluence scorer below.
    rev_mtf = ctx.get("reversal_mtf") or {}
    flat["trend_aligned_bottom"] = bool(
        htf_bias == "bullish" and rev_mtf.get("combined_bullish"))
    flat["trend_aligned_top"] = bool(
        htf_bias == "bearish" and rev_mtf.get("combined_bearish"))

    # Score both directions with REGIME-DEPENDENT category weights (Part B):
    # the same evidence weighs differently in a trend vs a range — the 1D
    # regime is the top of the timeframe hierarchy.
    regime = detect_regime(ctx.get("ind_1d"), ctx.get("volatility_1d"))
    long_raw, long_scores, long_reasons = calculate_confluence_score(flat, "long")
    short_raw, short_scores, short_reasons = calculate_confluence_score(flat, "short")
    long_total = weighted_total(long_scores, regime)
    short_total = weighted_total(short_scores, regime)

    # Common diagnostic fields attached to every blocked result.
    diag = {
        "htf_bias": htf_bias,
        "htf_tf": profile["htf"],
        "market_regime": regime,
        "long_score": long_total,
        "short_score": short_total,
        "price": ctx["price"],
    }

    # Equal evidence for both directions is a conflicted market, not a long
    # (the old ">=" tie-break silently defaulted long). A 0/0 tie is not a
    # conflict though — it is an ordinary quiet bar with no evidence at all.
    if long_total == short_total:
        stage = "direction_conflict" if long_total > 0 else "below_threshold"
        return blocked(stage, direction=None, score=long_total, **diag)
    if long_total > short_total:
        direction, total, scores, reasons = "long", long_total, long_scores, long_reasons
        counter_reasons = short_reasons
    else:
        direction, total, scores, reasons = "short", short_total, short_scores, short_reasons
        counter_reasons = long_reasons
    counter_total = min(long_total, short_total)

    # Level 1 (blocking): drop counter-trend signals, per the profile's policy.
    allowed = apply_htf_policy(direction, htf_bias, profile, ctx)
    if allowed is None:
        return blocked("htf_filter", direction=direction, score=total, **diag)

    # Level 3 (blocking): categorical diversity.
    if not has_diverse_confirmation(scores, config.MIN_DIVERSE_CATEGORIES):
        return blocked("diversity", direction=direction, score=total,
                        category_scores=scores, reasons=reasons, **diag)

    # Below journal threshold -> ignored entirely.
    if total < config.SCORE_JOURNAL_MIN:
        return blocked("below_threshold", direction=direction, score=total,
                        category_scores=scores, reasons=reasons, **diag)

    # Quality vetoes ("when NOT to trade").
    if dead_zone(scores.get("structure", 0), ctx.get("equilibrium")):
        return blocked("dead_zone", direction=direction, score=total,
                        category_scores=scores, reasons=reasons, **diag)
    if crowded_funding(direction, ctx.get("funding")):
        return blocked("crowded_funding", direction=direction, score=total,
                        category_scores=scores, reasons=reasons, **diag)

    # Level 4: conflict resolution.
    ob_dir = "bullish" if ctx["order_blocks"].get("bullish_ob") else (
        "bearish" if ctx["order_blocks"].get("bearish_ob") else None)
    resolved = resolve_conflicts({
        "liquidation_map": ctx["sweep_signal"],
        "order_block": ob_dir,
        "primary_direction": direction,
    })
    if resolved == "wait_for_sweep":
        return blocked("wait_for_sweep", direction=direction, score=total,
                        category_scores=scores, reasons=reasons, **diag)

    # Level 5: direction-aware multi-timeframe agreement over the profile's
    # timeframes (swing: 1H/4H/12H/1D; intraday: 15m/1H/4H).
    mtf_tfs = profile["mtf"]
    mtf_trends = {tf: trend_label(inds.get(tf, ctx.get("ind_signal", {})))
                  for tf in mtf_tfs}
    base_conf = total / 10.0
    modifier, mtf_info = mtf_confidence_factor(list(mtf_trends.values()), direction)
    conflict_mod = conflict_factor(total, counter_total)
    confidence = min(base_conf * modifier * conflict_mod, 1.0)

    # Risk sizing tuned to the trade style. The stop ATR may come from a
    # higher timeframe than the entry (see _stop_atr).
    stop_atr = _stop_atr(profile, inds, atr_value)
    position = calculate_position(ctx["price"], stop_atr, direction=direction,
                                  atr_multiplier=profile["atr_mult"],
                                  targets_r=profile["targets"])

    # Swing: stop behind the HTF structure, not a 1H-ATR multiple — otherwise
    # a "swing" trade degenerates into a scalp with an hours-long horizon.
    stop_basis = "atr"
    s_stop = _structural_stop(ctx, direction, ctx["price"], stop_atr, profile)
    if s_stop is not None:
        sign = 1 if direction == "long" else -1
        dist = abs(ctx["price"] - s_stop)
        risk_amount = config.ACCOUNT_BALANCE * (config.RISK_PERCENT / 100)
        position.update({
            "stop_loss": round(s_stop, 2),
            "stop_distance": round(dist, 2),
            "position_size": round(risk_amount / dist, 4),
            "target_1": round(ctx["price"] + sign * dist * profile["targets"][0], 2),
            "target_2": round(ctx["price"] + sign * dist * profile["targets"][1], 2),
        })
        stop_basis = "structure"

    # Targets at HTF structure (nearest liquidity / volume nodes) when available,
    # otherwise the ATR-based R-multiples.
    risk = abs(ctx["price"] - position["stop_loss"])
    tp1, tp2, struct_targets, tp2_source = _structure_targets(
        ctx, direction, ctx["price"], risk, position["target_1"], position["target_2"])
    position["target_1"], position["target_2"] = tp1, tp2

    # TP2 must be validated BEFORE it feeds expected_move: a target on the
    # wrong side / colliding with the stop / of unknown origin would otherwise
    # be silently masked by abs() into a plausible-looking move.
    tp2_reason = invalid_tp2(direction, ctx["price"], position["stop_loss"],
                             position["target_2"], position["target_1"], tp2_source)

    # Expected holding time to TP1 / TP2 from ATR-based drift on this timeframe.
    # Mandatory NO_TRADE gates (Part B): a trade must have an invalidation,
    # a minimum reward for its risk, enough confidence, no hard timeframe
    # conflict, and no open position pulling the other way.
    # Expected move for the mode's horizon: a quality filter, never a target.
    expected_move = None
    if tp2_reason is None:
        expected_move = effective_expected_move(
            ctx["price"], position["target_2"], atr_value,
            profile.get("forecast_horizon_hours", 24.0),
            TF_HOURS.get(ctx["timeframe"], 1.0))
    no_trade = [r for r in (
        tp2_reason,
        missing_invalidation(position.get("stop_loss"), position.get("stop_loss")),
        bad_risk_reward(ctx["price"], position["stop_loss"], position["target_2"],
                        profile.get("minimum_risk_reward")),
        low_confidence(confidence, profile.get("minimum_confidence")),
        insufficient_expected_move(expected_move,
                                   profile.get("minimum_expected_move_points", 0),
                                   profile.get("analysis_type", ""))
        if expected_move is not None else None,
        tf_conflict(mtf_info),
        position_conflict(open_trades, direction, ctx["symbol"]),
    ) if r]
    if no_trade:
        return blocked("no_trade", direction=direction, score=total,
                        category_scores=scores, reasons=reasons,
                        no_trade_reasons=no_trade, tp2_source=tp2_source,
                        stop_loss=position.get("stop_loss"),
                        take_profit_levels=[position.get("target_1"),
                                            position.get("target_2")],
                        **diag)

    hold = estimate_holding(ctx["price"], position["target_1"],
                            position["target_2"], atr_value, ctx["timeframe"])

    signal = {
        "symbol": ctx["symbol"],
        "timeframe": ctx["timeframe"],
        "direction": direction,
        "entry_price": ctx["price"],
        "price": ctx["price"],
        "stop_loss": position["stop_loss"],
        "target_1": position["target_1"],
        "target_2": position["target_2"],
        "hold_tp1_hours": hold[0],
        "hold_tp2_hours": hold[1],
        "targets_structure": struct_targets,
        "stop_basis": stop_basis,
        "stop_atr_tf": profile.get("stop_tf") or ctx["timeframe"],
        "position_size": position["position_size"],
        "risk_amount": position["risk_amount"],
        "atr": atr_value,
        "atr_multiplier_used": position["atr_multiplier_used"],
        # signals.score is an INTEGER column; the regime-weighted float is
        # kept separately for display/diagnostics.
        "score": int(round(total)),
        "score_weighted": total,
        "category_scores": scores,
        "reasons": reasons,
        "htf_bias": htf_bias,
        "htf_tf": profile["htf"],
        "confidence": round(confidence, 3),
        "confidence_modifier": modifier,
        "counter_score": counter_total,
        "conflict_factor": conflict_mod,
        "mtf": mtf_trends,
        "mtf_agreement": mtf_info,
        "analysis_type": profile.get("analysis_type", "SWING"),
        "expected_move_points": expected_move,
        "expected_move_percent": (round(expected_move / ctx["price"] * 100, 2)
                                  if ctx["price"] else None),
        "expected_move_atr": (round(expected_move / atr_value, 2)
                              if atr_value else None),
        "forecast_horizon": profile.get("forecast_horizon"),
        "expected_holding_period": profile.get("expected_holding_period"),
        "style": profile_name or "swing",
        "style_label": profile["label"],
        "style_emoji": profile["emoji"],
        "session_best": ctx["sessions"].get("best_session"),
        "sessions": ctx["sessions"].get("sessions"),
        "funding_value": (ctx["funding"] or {}).get("current"),
        "timestamp": ctx["timestamp"],
        # Observability: both directions' evidence, TP provenance and the
        # decision snapshot (reproducible close vs executable ticker).
        "candidate_direction": direction,
        "long_score": long_total,
        "short_score": short_total,
        "raw_confidence": round(confidence, 3),
        "calibrated_confidence": None,
        "tp2_source": tp2_source,
        **snapshot,
    }

    # Structured result (Part B): regime, both sides of the evidence, entry
    # zone, invalidation, RR, confirmation/cancel conditions, freshness.
    signal.update(build_result(signal, ctx, regime, counter_reasons,
                               mtf_info).to_signal_fields())

    # Level 6: cooldown / dedup (blocking).
    if not should_send_signal(signal, last_signal, atr_value, profile["cooldown_hours"]):
        signal["status"] = "cooldown"
        signal["analysis_status"] = STATUS_MAP["cooldown"]
        signal["deliverable"] = False
        return signal

    # Level 7: daily limit (blocking for delivery) — a hard cap; over-limit
    # signals are still recorded as journal entries below.
    deliverable = within_daily_limit(delivered_today)

    # Final classification.
    if total >= config.SCORE_ALERT_MIN and deliverable:
        signal["status"] = "alert"
        signal["deliverable"] = True
    elif total >= config.SCORE_JOURNAL_MIN:
        signal["status"] = "journal"
        signal["deliverable"] = False
    else:
        signal["status"] = "ignored"
        signal["deliverable"] = False

    signal["analysis_status"] = STATUS_MAP.get(signal["status"], "NO_TRADE")

    # Freshness gate: an ENTER whose decision came too long after the entry
    # candle closed is not executable at the analysed price — downgrade it to
    # WAIT (the forecast itself is still recorded for observability). The
    # analytical snapshot stays anchored to signal_close_price.
    max_delay = profile.get("max_decision_delay_seconds")
    if (signal["status"] == "alert" and max_delay
            and freshness is not None and freshness > max_delay):
        signal["status"] = "journal"
        signal["deliverable"] = False
        signal["analysis_status"] = STATUS_MAP["journal"]
        signal["stale_decision"] = True
        reasons_list = signal.setdefault("no_trade_reasons", [])
        reasons_list.append(
            f"решение через {freshness:.0f} с после закрытия свечи — "
            f"больше лимита {max_delay:.0f} с для режима "
            f"{profile.get('analysis_type', '')}")
        log.warning("[%s] ENTER downgraded to WAIT: decision %.0fs after "
                    "candle close (limit %.0fs)",
                    profile.get("analysis_type"), freshness, max_delay)

    # AI interpretation is COMMENTARY ONLY: the numeric confidence stays a
    # deterministic function of the evidence (same input -> same output).
    # Blending in an LLM score destroyed both determinism and calibration.
    if interpret:
        ai = await claude.interpret_signal(signal)
        signal["ai_confidence"] = ai["confidence"]
        signal["ai_text"] = ai["comment"]

    return signal


# Net directional drift per candle as a fraction of ATR (range != displacement).
_DRIFT_PER_BAR = 0.5


def conflict_factor(winner_total: float, loser_total: float) -> float:
    """Confidence penalty for opposing evidence.

    1.0 for a clean setup (loser scored 0), sliding towards ~0.65 as the
    losing direction approaches the winner. The old cascade compared only
    max(long, short) — a 6/5 setup looked identical to a 6/0 one.
    """
    if winner_total <= 0:
        return 1.0
    conflict = loser_total / (winner_total + loser_total)
    return round(1.0 - 0.7 * conflict, 3)


def estimate_holding(entry: float, tp1: float, tp2: float, atr: float,
                     timeframe: str) -> tuple[float, float]:
    """Rough expected hours to reach TP1 / TP2 from ATR-based drift."""
    tf_hours = TF_HOURS.get(timeframe, 4.0)
    if not atr or atr <= 0:
        return (0.0, 0.0)
    step = atr * _DRIFT_PER_BAR
    bars_tp1 = abs(tp1 - entry) / step
    bars_tp2 = abs(tp2 - entry) / step
    return (round(bars_tp1 * tf_hours, 1), round(bars_tp2 * tf_hours, 1))


def _decision_snapshot(ctx: dict[str, Any]) -> dict[str, Any]:
    """Reproducibility fields attached to every cascade result.

    signal_close_price is the close of the last CLOSED entry candle (what the
    forecast is computed from); executable_price_at_decision is the live
    ticker at decision time (what a trade could actually get). Freshness and
    latency make any gap measurable instead of silent.
    """
    decision_time = ctx.get("decision_time") or datetime.now(timezone.utc)
    last_close = ctx.get("last_close_time")
    freshness = None
    if last_close is not None:
        try:
            freshness = max((decision_time - last_close).total_seconds(), 0.0)
        except TypeError:
            freshness = None
    latency = None
    fired = ctx.get("job_fired_at")
    if fired is not None:
        try:
            latency = max((decision_time - fired).total_seconds(), 0.0)
        except TypeError:
            latency = None
    return {
        "signal_candle_close_time": last_close,
        "decision_time": decision_time,
        "signal_close_price": ctx.get("price"),
        "executable_price_at_decision": ctx.get("executable_price", ctx.get("price")),
        "executable_price_degraded": bool(ctx.get("executable_price_degraded")),
        "data_freshness_seconds": freshness,
        "decision_latency_seconds": latency,
    }


def _blocked(stage: str, **extra: Any) -> dict[str, Any]:
    result = {"status": "blocked", "blocked_at": stage, "deliverable": False}
    result.update(extra)
    return result


def _safe(value: Any, default: Any) -> Any:
    if isinstance(value, Exception):
        log.warning("data source error: %s", value)
        return default
    return value
