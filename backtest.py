"""Historical backtest that replicates the LIVE profile logic.

Walks the profile's entry timeframe (swing 1H / intraday 15m) and reconstructs
the market context as-of each bar exactly like the live engine: HTF bias from
the trend timeframe, Order Blocks / FVG from the higher zone timeframes,
premium/discount, liquidity sweep, reversal — then the same confluence score
and gates. Each qualified setup is resolved once (ATR stop vs target-1), and
results are aggregated over a score-threshold sweep with the profile cooldown.

Note: funding / on-chain history is unavailable, so those macro points are not
scored — the estimate is approximate but faithful to the price-based logic.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import config
from analyzer.binance import BinanceClient
from analyzer.cvd import compute_cvd_from_klines, cvd_series
from analyzer.divergence import detect_divergence
from analyzer.equilibrium import compute_equilibrium
from analyzer.indicators import compute_indicators
from analyzer.liquidity import detect_liquidity
from analyzer.reversal import detect_reversal
from pipeline import _build_htf_zones, _near_key_level
from risk.position_sizing import calculate_position
from signal_engine.confluence import (calculate_confluence_score,
                                      has_diverse_confirmation)
from signal_engine.htf_filter import filter_by_htf, get_htf_bias
from signal_engine.profiles import get_profile

log = logging.getLogger(__name__)

THRESHOLDS = [5, 6, 7, 8, 9, 10]
MIN_TRADES_FOR_REC = 8
MAX_BARS = 1200          # entry bars to walk (bounds runtime)
ENTRY_HISTORY = 3000     # entry candles to page back for a stable sample


async def _fetch_history(binance, interval: str, target: int) -> "pd.DataFrame":
    """Page klines backwards to assemble `target` candles for a stable sample."""
    import pandas as pd
    frames, end = [], None
    for _ in range(12):  # safety cap on pages
        df = await binance.klines(interval, limit=1000, end_time=end)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if sum(len(f) for f in frames) >= target or len(df) < 1000:
            break
        end = int(df["open_time"].iloc[0].timestamp() * 1000) - 1
    if not frames:
        return await binance.klines(interval, limit=1000)
    full = (pd.concat(frames).drop_duplicates("open_time")
            .sort_values("open_time").reset_index(drop=True))
    return full.iloc[-target:].reset_index(drop=True)
_TF_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "12h": 12.0, "1d": 24.0}


async def run_backtest(binance: BinanceClient, profile_name: str = "swing",
                       warmup: int = 210) -> str:
    profile = get_profile(profile_name)
    entry_tf = profile["entry"]
    htf = profile["htf"]
    zone_tfs = profile["zone_tfs"]

    # Entry timeframe: page back for a large, stable sample. Higher timeframes
    # already cover long history in one request.
    other_tfs = [t for t in ({htf, "1d"} | set(zone_tfs)) if t != entry_tf]
    entry_df, *other_frames = await asyncio.gather(
        _fetch_history(binance, entry_tf, ENTRY_HISTORY),
        *[binance.klines(t, limit=500) for t in other_tfs],
    )
    dfs = {entry_tf: entry_df}
    dfs.update(dict(zip(other_tfs, other_frames)))

    # The bar-by-bar walk is CPU-heavy pandas work (~10s); run it off the event
    # loop so Telegram handlers and the scheduler stay responsive.
    return await asyncio.to_thread(_walk, dfs, profile, warmup)


def _walk(dfs: dict[str, Any], profile: dict[str, Any], warmup: int) -> str:
    entry_tf = profile["entry"]
    htf = profile["htf"]
    zone_tfs = profile["zone_tfs"]
    entry_df = dfs[entry_tf]
    n = len(entry_df)

    # Round-trip trading cost as a fraction of price (entry+exit fees + slippage).
    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100

    setups: list[dict[str, Any]] = []
    start = max(warmup, n - MAX_BARS)

    for i in range(start, n - 1):
        entry_sub = entry_df.iloc[max(0, i - 250):i + 1]
        ind = compute_indicators(entry_sub)
        atr = ind.get("atr")
        if not atr:
            continue
        t = entry_df["close_time"].iloc[i]
        price = float(entry_df["close"].iloc[i])

        # HTF bias, as-of this bar.
        htf_slice = dfs[htf][dfs[htf]["close_time"] <= t].iloc[-250:]
        if len(htf_slice) < 210:
            continue
        htf_bias = get_htf_bias(compute_indicators(htf_slice))

        # HTF zones (OB/FVG/levels) as-of this bar.
        zdfs, zinds = {}, {}
        for tf in zone_tfs:
            s = dfs[tf][dfs[tf]["close_time"] <= t].iloc[-160:]
            if len(s) >= 30:
                zdfs[tf] = s
                zinds[tf] = compute_indicators(s)
        if not zdfs:
            continue
        zones = _build_htf_zones(zdfs, zinds, list(zdfs.keys()), price, entry_sub)
        ob, fvg, levels = zones["order_blocks"], zones["fvg"], zones["levels"]

        eq = compute_equilibrium(htf_slice)
        liq = detect_liquidity(entry_sub, atr_value=atr)
        at_level = _near_key_level(price, levels, ob, fvg, max(atr * 0.3, price * 0.002))
        rev = detect_reversal(entry_sub, ind, cvd_series(entry_sub), at_key_level=at_level)
        ser = ind.get("_series", {})
        drsi = detect_divergence(entry_sub, ser.get("rsi"))
        dmacd = detect_divergence(entry_sub, ser.get("macd"))
        cvd = compute_cvd_from_klines(entry_sub)

        flat = dict(ind)
        flat.update({
            "bullish_divergence": drsi["bullish_divergence"] or dmacd["bullish_divergence"],
            "bearish_divergence": drsi["bearish_divergence"] or dmacd["bearish_divergence"],
            "price_in_bullish_ob": ob["price_in_bullish_ob"],
            "price_in_bearish_ob": ob["price_in_bearish_ob"],
            "rejection_wick": ob["rejection_wick"],
            "liquidity_swept_below": liq["liquidity_swept_below"],
            "liquidity_swept_above": liq["liquidity_swept_above"],
            "reversal_candle": liq["reversal_candle"],
            "price_in_bullish_fvg": fvg["price_in_bullish_fvg"],
            "price_in_bearish_fvg": fvg["price_in_bearish_fvg"],
            "in_discount": eq["zone"] == "discount",
            "in_premium": eq["zone"] == "premium",
            "bullish_reversal": rev["bullish_reversal"],
            "bearish_reversal": rev["bearish_reversal"],
            "reversal_strong_bull": rev["bull_strong"],
            "reversal_strong_bear": rev["bear_strong"],
            "cvd_bullish": cvd["cvd_bullish"],
            "cvd_bearish": cvd["cvd_bearish"],
            "funding": None, "exchange_netflow": None,
        })

        lt, ls, _ = calculate_confluence_score(flat, "long")
        st, ss, _ = calculate_confluence_score(flat, "short")
        direction, total, scores = ("long", lt, ls) if lt >= st else ("short", st, ss)

        if filter_by_htf(direction, htf_bias) is None:
            continue
        if not has_diverse_confirmation(scores, config.MIN_DIVERSE_CATEGORIES):
            continue
        if total < min(THRESHOLDS):
            continue

        pos = calculate_position(price, atr, direction=direction,
                                 atr_multiplier=profile["atr_mult"],
                                 targets_r=profile["targets"])
        outcome = _resolve(entry_df, i, direction, pos)
        if outcome is None:
            continue
        # Net result: subtract the round-trip cost expressed in R.
        risk_dist = abs(price - pos["stop_loss"])
        cost_r = (cost_pct * price / risk_dist) if risk_dist else 0.0
        outcome["r"] = round(outcome["r"] - cost_r, 2)
        setups.append({"idx": i, "score": total, "direction": direction, **outcome})

    cooldown_bars = max(1, int(round(profile["cooldown_hours"] / _TF_HOURS.get(entry_tf, 1))))
    return _report(setups, profile, cooldown_bars, n - start)


def _resolve(df, entry_idx: int, direction: str, pos: dict[str, Any]) -> dict[str, Any] | None:
    stop, target = pos["stop_loss"], pos["target_1"]
    for j in range(entry_idx + 1, len(df)):
        high, low = float(df["high"].iloc[j]), float(df["low"].iloc[j])
        if direction == "long":
            if low <= stop:
                return {"outcome": "loss", "r": -1.0}
            if high >= target:
                return {"outcome": "win", "r": _r_multiple(pos)}
        else:
            if high >= stop:
                return {"outcome": "loss", "r": -1.0}
            if low <= target:
                return {"outcome": "win", "r": _r_multiple(pos)}
    return None


def _r_multiple(pos: dict[str, Any]) -> float:
    risk = abs(pos["entry_price"] - pos["stop_loss"])
    reward = abs(pos["target_1"] - pos["entry_price"])
    return round(reward / risk, 2) if risk else 0.0


def _aggregate(setups: list[dict[str, Any]], threshold: int, cooldown_bars: int) -> dict[str, Any]:
    taken, last_idx = [], -10 ** 9
    for s in setups:
        if s["score"] < threshold or s["idx"] - last_idx < cooldown_bars:
            continue
        taken.append(s)
        last_idx = s["idx"]
    total = len(taken)
    if total == 0:
        return {"threshold": threshold, "trades": 0}
    wins = sum(1 for t in taken if t["outcome"] == "win")
    total_r = sum(t["r"] for t in taken)
    return {"threshold": threshold, "trades": total, "winrate": wins / total * 100,
            "total_r": total_r, "avg_r": total_r / total}


def _report(setups: list[dict[str, Any]], profile: dict[str, Any],
            cooldown_bars: int, bars: int) -> str:
    label = profile["label"]
    entry = profile["entry"].upper()
    if not setups:
        return (f"📊 Бэктест {label} ({entry} вход): подходящих сетапов не найдено "
                f"на доступной истории (~{bars} свечей).")

    rows = [_aggregate(setups, t, cooldown_bars) for t in THRESHOLDS]
    lines = [
        f"📊 Бэктест {label} | вход {entry}, зоны {'/'.join(t.upper() for t in profile['zone_tfs'])}",
        f"История: ~{bars} свечей {entry} | NET: комиссии 2×{config.TAKER_FEE_PCT:g}% "
        f"+ проскальзывание {config.SLIPPAGE_PCT:g}%",
        "",
        "Порог │ Сделок │ Винрейт │   Σ R  │ Ср.R",
        "──────┼────────┼─────────┼────────┼──────",
    ]
    for r in rows:
        if r["trades"] == 0:
            lines.append(f"  {r['threshold']:>2}  │   0    │    —    │   —    │  —")
            continue
        lines.append(
            f"  {r['threshold']:>2}  │  {r['trades']:>3}   │  {r['winrate']:>4.0f}%  │ "
            f"{r['total_r']:>+5.1f} │ {r['avg_r']:>+.2f}")

    candidates = [r for r in rows if r.get("trades", 0) >= MIN_TRADES_FOR_REC
                  and r.get("avg_r", -9) > 0]
    lines.append("")
    if candidates:
        best = max(candidates, key=lambda r: r["avg_r"])
        lines.append(f"✅ Рекомендация: порог {best['threshold']} — "
                     f"ср. {best['avg_r']:+.2f}R при {best['trades']} сделках.")
        lines.append(f"Поставь SCORE_ALERT_MIN={best['threshold']} в Railway.")
        lines += _monthly_projection(best, profile, bars)
    else:
        lines.append("⚠️ Ни один порог не дал устойчивого плюса на этой истории — "
                     "цель 10%/мес пока не обоснована. Смягчи фильтры или поменяй TP.")
    lines.append("")
    lines.append("ℹ️ Без funding/on-chain истории — оценка приблизительная. "
                 "Прошлые результаты не гарантируют будущие.")
    return "\n".join(lines)


def _monthly_projection(row: dict[str, Any], profile: dict[str, Any], bars: int) -> list[str]:
    from config import RISK_PERCENT
    tf_hours = _TF_HOURS.get(profile["entry"], 1)
    period_days = max(bars * tf_hours / 24, 1)
    per_month = 30 / period_days
    trades_pm = row["trades"] * per_month
    r_pm = row["total_r"] * per_month
    ret_pm = r_pm * RISK_PERCENT   # each 1R == RISK_PERCENT of the account
    out = [
        "",
        f"📅 Проекция (риск {RISK_PERCENT:g}%/сделку, история ~{period_days:.0f} дн):",
        f"  ~{trades_pm:.0f} сделок/мес | ~{r_pm:+.1f}R/мес | ≈ {ret_pm:+.1f}%/мес",
    ]
    if r_pm > 0:
        req = 10 / r_pm
        out.append(f"  🎯 Для +10%/мес: риск ~{req:.1f}%/сделку "
                   f"(либо больше сделок/выше R).")
        if req > 3:
            out.append("  ⚠️ Нужный риск >3%/сделку — агрессивно; цель 10%/мес "
                       "на этом edge труднодостижима без роста R или частоты.")
    return out
