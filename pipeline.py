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
from signal_engine.daily_limiter import beats_weakest, within_daily_limit
from signal_engine.htf_filter import filter_by_htf, get_htf_bias
from signal_engine.mtf_confidence import mtf_confidence_factor, trend_label
from signal_engine.profiles import get_profile

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

    needed = {"1h", "4h", "1d", signal_timeframe} | set(profile["mtf"])
    tfs = sorted(needed)
    frames = await asyncio.gather(*[binance.klines(tf, limit=300) for tf in tfs])
    dfs = dict(zip(tfs, frames))
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

    # Structural analyzers on the signal timeframe.
    atr_value = ind_signal.get("atr") or 0.0
    order_blocks = detect_order_blocks(df_signal)
    liquidity = detect_liquidity(df_signal, atr_value=atr_value)
    volume_profile = compute_volume_profile(df_signal)
    fvg = detect_fvg(df_signal, atr_value=atr_value)
    reversal = detect_reversal(df_signal, ind_signal, cvd_series(df_signal))

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
        "reversal": reversal,
        "divergence": divergence,
        "liquidation_map": liq_map,
        "sweep_signal": sweep,
        "sessions": sessions,
    }
    return context


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
        "bullish_reversal": ctx.get("reversal", {}).get("bullish_reversal"),
        "bearish_reversal": ctx.get("reversal", {}).get("bearish_reversal"),
        "reversal_strong_bull": ctx.get("reversal", {}).get("bull_strong"),
        "reversal_strong_bear": ctx.get("reversal", {}).get("bear_strong"),
        "cvd_bullish": ctx["cvd"].get("cvd_bullish"),
        "cvd_bearish": ctx["cvd"].get("cvd_bearish"),
        "funding": (ctx["funding"] or {}).get("current"),
        "exchange_netflow": (ctx["onchain"] or {}).get("exchange_netflow"),
    })
    return flat


async def run_cascade(ctx: dict[str, Any], delivered_today: list[dict[str, Any]],
                      last_signal: dict[str, Any] | None,
                      interpret: bool = True,
                      profile_name: str | None = None) -> dict[str, Any]:
    """Run levels 1-7 and return a result describing the outcome."""
    profile = get_profile(profile_name)
    inds = ctx.get("inds_by_tf", {})
    flat = _flatten_for_confluence(ctx)
    atr_value = ctx["atr"] or 0.0

    # Level 1: HTF bias from the profile's higher timeframe.
    htf_ind = inds.get(profile["htf"], ctx["ind_1d"])
    htf_bias = get_htf_bias(htf_ind)

    # Score both directions; the stronger one is the candidate.
    long_total, long_scores, long_reasons = calculate_confluence_score(flat, "long")
    short_total, short_scores, short_reasons = calculate_confluence_score(flat, "short")

    if long_total >= short_total:
        direction, total, scores, reasons = "long", long_total, long_scores, long_reasons
    else:
        direction, total, scores, reasons = "short", short_total, short_scores, short_reasons

    # Common diagnostic fields attached to every blocked result.
    diag = {
        "htf_bias": htf_bias,
        "long_score": long_total,
        "short_score": short_total,
        "price": ctx["price"],
    }

    # Level 1 (blocking): drop counter-trend signals.
    allowed = filter_by_htf(direction, htf_bias)
    if allowed is None:
        return _blocked("htf_filter", direction=direction, score=total, **diag)

    # Level 3 (blocking): categorical diversity.
    if not has_diverse_confirmation(scores, config.MIN_DIVERSE_CATEGORIES):
        return _blocked("diversity", direction=direction, score=total,
                        category_scores=scores, reasons=reasons, **diag)

    # Below journal threshold -> ignored entirely.
    if total < config.SCORE_JOURNAL_MIN:
        return _blocked("below_threshold", direction=direction, score=total,
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
        return _blocked("wait_for_sweep", direction=direction, score=total,
                        category_scores=scores, reasons=reasons, **diag)

    # Level 5: direction-aware multi-timeframe agreement over the profile's
    # timeframes (swing: 1H/4H/12H/1D; intraday: 15m/1H/4H).
    mtf_tfs = profile["mtf"]
    mtf_trends = {tf: trend_label(inds.get(tf, ctx.get("ind_signal", {})))
                  for tf in mtf_tfs}
    base_conf = total / 10.0
    modifier, mtf_info = mtf_confidence_factor(list(mtf_trends.values()), direction)
    confidence = min(base_conf * modifier, 1.0)

    # Risk sizing tuned to the trade style.
    position = calculate_position(ctx["price"], atr_value, direction=direction,
                                  atr_multiplier=profile["atr_mult"],
                                  targets_r=profile["targets"])

    # Expected holding time to TP1 / TP2 from ATR-based drift on this timeframe.
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
        "position_size": position["position_size"],
        "risk_amount": position["risk_amount"],
        "atr": atr_value,
        "atr_multiplier_used": position["atr_multiplier_used"],
        "score": total,
        "category_scores": scores,
        "reasons": reasons,
        "htf_bias": htf_bias,
        "confidence": round(confidence, 3),
        "confidence_modifier": modifier,
        "mtf": mtf_trends,
        "mtf_agreement": mtf_info,
        "style": profile_name or "swing",
        "style_label": profile["label"],
        "style_emoji": profile["emoji"],
        "session_best": ctx["sessions"].get("best_session"),
        "sessions": ctx["sessions"].get("sessions"),
        "funding_value": (ctx["funding"] or {}).get("current"),
        "timestamp": ctx["timestamp"],
    }

    # Level 6: cooldown / dedup (blocking).
    if not should_send_signal(signal, last_signal, atr_value, profile["cooldown_hours"]):
        signal["status"] = "cooldown"
        signal["deliverable"] = False
        return signal

    # Level 7: daily limit (blocking for delivery).
    deliverable = within_daily_limit(delivered_today) or beats_weakest(total, delivered_today)

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

    # AI interpretation (text + confidence comment).
    if interpret:
        ai = await claude.interpret_signal(signal)
        signal["ai_confidence"] = ai["confidence"]
        signal["ai_text"] = ai["comment"]
        # Blend AI confidence with mtf-modified confluence confidence.
        signal["confidence"] = round((signal["confidence"] + ai["confidence"]) / 2, 3)

    return signal


_TF_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "12h": 12.0, "1d": 24.0}
# Net directional drift per candle as a fraction of ATR (range != displacement).
_DRIFT_PER_BAR = 0.5


def estimate_holding(entry: float, tp1: float, tp2: float, atr: float,
                     timeframe: str) -> tuple[float, float]:
    """Rough expected hours to reach TP1 / TP2 from ATR-based drift."""
    tf_hours = _TF_HOURS.get(timeframe, 4.0)
    if not atr or atr <= 0:
        return (0.0, 0.0)
    step = atr * _DRIFT_PER_BAR
    bars_tp1 = abs(tp1 - entry) / step
    bars_tp2 = abs(tp2 - entry) / step
    return (round(bars_tp1 * tf_hours, 1), round(bars_tp2 * tf_hours, 1))


def _blocked(stage: str, **extra: Any) -> dict[str, Any]:
    result = {"status": "blocked", "blocked_at": stage, "deliverable": False}
    result.update(extra)
    return result


def _safe(value: Any, default: Any) -> Any:
    if isinstance(value, Exception):
        log.warning("data source error: %s", value)
        return default
    return value
