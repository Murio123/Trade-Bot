"""Lightweight historical backtest of the confluence strategy.

Walks forward over 4H candles, recomputing indicators on each window and
applying the HTF filter + confluence score + diversity gate. Entries use
ATR-based stops/targets; each trade resolves to win/loss when stop or target-1
is hit first. This is a simplified estimate (no funding/on-chain history), used
to give the /backtest command a sane summary.
"""
from __future__ import annotations

import logging
from typing import Any

import config
from analyzer.binance import BinanceClient
from analyzer.divergence import detect_divergence
from analyzer.indicators import compute_indicators
from analyzer.liquidity import detect_liquidity
from analyzer.order_blocks import detect_order_blocks
from risk.position_sizing import calculate_position
from signal_engine.confluence import (calculate_confluence_score,
                                      has_diverse_confirmation)
from signal_engine.htf_filter import filter_by_htf, get_htf_bias

log = logging.getLogger(__name__)


async def run_backtest(binance: BinanceClient, window: int = 250,
                       step: int = 1, warmup: int = 210) -> str:
    df_4h = await binance.klines("4h", limit=1000)
    df_1d = await binance.klines("1d", limit=400)

    trades: list[dict[str, Any]] = []
    n = len(df_4h)
    last_entry_idx = -100

    for i in range(warmup, n - 1, step):
        sub = df_4h.iloc[: i + 1]
        ind = compute_indicators(sub)
        if ind.get("atr") in (None, 0):
            continue

        # Daily bias aligned to the candle's close time.
        close_time = sub["close_time"].iloc[-1]
        daily_slice = df_1d[df_1d["close_time"] <= close_time]
        if len(daily_slice) < 200:
            continue
        ind_1d = compute_indicators(daily_slice)
        htf_bias = get_htf_bias(ind_1d)

        flat = dict(ind)
        osc = ind.get("_series", {}).get("rsi")
        flat.update(detect_divergence(sub, osc))
        flat.update(detect_order_blocks(sub))
        flat.update(detect_liquidity(sub, atr_value=ind["atr"]))

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
        if total < config.SCORE_ALERT_MIN:
            continue
        if i - last_entry_idx < 6:  # rudimentary cooldown (~24h on 4H)
            continue

        entry = float(sub["close"].iloc[-1])
        pos = calculate_position(entry, ind["atr"], direction=direction)
        outcome = _resolve(df_4h, i, direction, pos)
        if outcome is None:
            continue
        trades.append({"direction": direction, "score": total, **outcome})
        last_entry_idx = i

    return _report(trades)


def _resolve(df, entry_idx: int, direction: str, pos: dict[str, Any]) -> dict[str, Any] | None:
    """Walk forward from entry until stop or target-1 is hit."""
    stop = pos["stop_loss"]
    target = pos["target_1"]
    for j in range(entry_idx + 1, len(df)):
        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])
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
    return None  # unresolved (still open at series end)


def _r_multiple(pos: dict[str, Any]) -> float:
    risk = abs(pos["entry_price"] - pos["stop_loss"])
    reward = abs(pos["target_1"] - pos["entry_price"])
    return round(reward / risk, 2) if risk else 0.0


def _report(trades: list[dict[str, Any]]) -> str:
    if not trades:
        return "📊 Бэктест: подходящих сделок не найдено на доступной истории."
    total = len(trades)
    wins = sum(1 for t in trades if t["outcome"] == "win")
    losses = total - wins
    winrate = wins / total * 100
    total_r = sum(t["r"] for t in trades)
    avg_r = total_r / total
    return "\n".join([
        f"📊 Бэктест стратегии ({config.SYMBOL_DISPLAY} 4H, score≥{config.SCORE_ALERT_MIN})",
        f"Сделок: {total}",
        f"Винрейт: {winrate:.1f}% ({wins}W / {losses}L)",
        f"Суммарный результат: {total_r:+.1f}R",
        f"Средний результат на сделку: {avg_r:+.2f}R",
    ])
