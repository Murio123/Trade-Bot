"""Historical backtest with a score-threshold sweep.

Walks forward over 4H candles, scoring each bar exactly like the live engine
(HTF filter + confluence incl. FVG / reversal + diversity gate). Every qualified
setup is resolved once (ATR stop vs target-1, whichever hits first), then the
results are aggregated at each score threshold so we can see which threshold is
actually profitable on history — and recommend one.
"""
from __future__ import annotations

import logging
from typing import Any

import config
from analyzer.binance import BinanceClient
from analyzer.cvd import cvd_series
from analyzer.divergence import detect_divergence
from analyzer.fvg import detect_fvg
from analyzer.indicators import compute_indicators
from analyzer.liquidity import detect_liquidity
from analyzer.order_blocks import detect_order_blocks
from analyzer.reversal import detect_reversal
from risk.position_sizing import calculate_position
from signal_engine.confluence import (calculate_confluence_score,
                                      has_diverse_confirmation)
from signal_engine.htf_filter import filter_by_htf, get_htf_bias

log = logging.getLogger(__name__)

THRESHOLDS = [5, 6, 7, 8, 9, 10]
COOLDOWN_BARS = 6        # ~24h on 4H
MIN_TRADES_FOR_REC = 8   # need a minimum sample to recommend a threshold


async def run_backtest(binance: BinanceClient, warmup: int = 210) -> str:
    df_4h = await binance.klines("4h", limit=1000)
    df_1d = await binance.klines("1d", limit=400)

    setups: list[dict[str, Any]] = []
    n = len(df_4h)

    for i in range(warmup, n - 1):
        sub = df_4h.iloc[: i + 1]
        ind = compute_indicators(sub)
        if ind.get("atr") in (None, 0):
            continue

        close_time = sub["close_time"].iloc[-1]
        daily_slice = df_1d[df_1d["close_time"] <= close_time]
        if len(daily_slice) < 200:
            continue
        htf_bias = get_htf_bias(compute_indicators(daily_slice))

        flat = dict(ind)
        osc = ind.get("_series", {}).get("rsi")
        flat.update(detect_divergence(sub, osc))
        flat.update(detect_order_blocks(sub))
        flat.update(detect_liquidity(sub, atr_value=ind["atr"]))
        flat.update(detect_fvg(sub, atr_value=ind["atr"]))
        rev = detect_reversal(sub, ind, cvd_series(sub))
        flat.update({
            "bullish_reversal": rev["bullish_reversal"],
            "bearish_reversal": rev["bearish_reversal"],
            "reversal_strong_bull": rev["bull_strong"],
            "reversal_strong_bear": rev["bear_strong"],
        })

        long_total, long_scores, _ = calculate_confluence_score(flat, "long")
        short_total, short_scores, _ = calculate_confluence_score(flat, "short")
        if long_total >= short_total:
            direction, total, scores = "long", long_total, long_scores
        else:
            direction, total, scores = "short", short_total, short_scores

        if filter_by_htf(direction, htf_bias) is None:
            continue
        if not has_diverse_confirmation(scores, config.MIN_DIVERSE_CATEGORIES):
            continue
        if total < min(THRESHOLDS):
            continue

        entry = float(sub["close"].iloc[-1])
        pos = calculate_position(entry, ind["atr"], direction=direction)
        outcome = _resolve(df_4h, i, direction, pos)
        if outcome is None:
            continue
        setups.append({"idx": i, "score": total, "direction": direction, **outcome})

    return _report(setups, n - warmup)


def _aggregate(setups: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    """Apply the threshold + cooldown sequentially and compute metrics."""
    taken = []
    last_idx = -1000
    for s in setups:
        if s["score"] < threshold:
            continue
        if s["idx"] - last_idx < COOLDOWN_BARS:
            continue
        taken.append(s)
        last_idx = s["idx"]
    total = len(taken)
    if total == 0:
        return {"threshold": threshold, "trades": 0}
    wins = sum(1 for t in taken if t["outcome"] == "win")
    total_r = sum(t["r"] for t in taken)
    return {
        "threshold": threshold,
        "trades": total,
        "winrate": wins / total * 100,
        "total_r": total_r,
        "avg_r": total_r / total,
    }


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


def _report(setups: list[dict[str, Any]], bars: int) -> str:
    if not setups:
        return "📊 Бэктест: подходящих сетапов не найдено на доступной истории."

    rows = [_aggregate(setups, t) for t in THRESHOLDS]
    lines = [
        f"📊 Бэктест порогов ({config.SYMBOL_DISPLAY} 4H, ~{bars} свечей)",
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
            f"{r['total_r']:>+5.1f} │ {r['avg_r']:>+.2f}"
        )

    # Recommend the threshold with the best expectancy among those with enough
    # trades and a positive edge.
    candidates = [r for r in rows if r.get("trades", 0) >= MIN_TRADES_FOR_REC
                  and r.get("avg_r", -9) > 0]
    lines.append("")
    if candidates:
        best = max(candidates, key=lambda r: r["avg_r"])
        lines.append(
            f"✅ Рекомендация: порог {best['threshold']} — "
            f"лучший ср. результат {best['avg_r']:+.2f}R при {best['trades']} сделках."
        )
        lines.append(f"Поставь SCORE_ALERT_MIN={best['threshold']} в Railway.")
        if best["threshold"] != config.SCORE_ALERT_MIN:
            lines.append(f"(сейчас {config.SCORE_ALERT_MIN})")
    else:
        lines.append("⚠️ Ни один порог не дал устойчивого плюса на этой истории — "
                     "снижать порог рискованно. Текущий "
                     f"{config.SCORE_ALERT_MIN} оставляем.")
    lines.append("")
    lines.append("ℹ️ Без funding/on-chain истории — оценка приблизительная.")
    return "\n".join(lines)
