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
from analyzer.cvd import compute_cvd
from analyzer.divergence import detect_divergence
from analyzer.indicators import compute_indicators
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
from signal_engine.mtf_confidence import apply_mtf_confidence, trend_label

log = logging.getLogger(__name__)


async def gather_market_context(binance: BinanceClient,
                                signal_timeframe: str = "4h") -> dict[str, Any]:
    """Fetch raw data from every source and compute all analyzer outputs."""
    # Klines for the three timeframes.
    df_1h, df_4h, df_1d = await asyncio.gather(
        binance.klines("1h", limit=300),
        binance.klines("4h", limit=300),
        binance.klines("1d", limit=300),
    )

    ind_1h = compute_indicators(df_1h)
    ind_4h = compute_indicators(df_4h)
    ind_1d = compute_indicators(df_1d)

    # 1h/4h/1d are always needed for the HTF bias + MTF agreement. The signal
    # timeframe may be one of those or a separate one (e.g. 15m) fetched here.
    base_dfs = {"1h": df_1h, "4h": df_4h, "1d": df_1d}
    base_inds = {"1h": ind_1h, "4h": ind_4h, "1d": ind_1d}
    if signal_timeframe in base_dfs:
        df_signal = base_dfs[signal_timeframe]
        ind_signal = base_inds[signal_timeframe]
    else:
        df_signal = await binance.klines(signal_timeframe, limit=300)
        ind_signal = compute_indicators(df_signal)

    # Crypto-specific + external sources (run concurrently, tolerate failures).
    (funding, oi, ls_ratio, agg_trades, macro, onchain) = await asyncio.gather(
        binance.funding_rate(),
        binance.open_interest(),
        binance.long_short_ratio(),
        binance.agg_trades(limit=1000),
        get_macro(),
        get_onchain(),
        return_exceptions=True,
    )

    funding = _safe(funding, {})
    oi = _safe(oi, 0.0)
    ls_ratio = _safe(ls_ratio, {})
    agg_trades = _safe(agg_trades, [])
    macro = _safe(macro, {})
    onchain = _safe(onchain, {})

    cvd = compute_cvd(agg_trades if isinstance(agg_trades, list) else [])

    # Structural analyzers on the signal timeframe.
    atr_value = ind_signal.get("atr") or 0.0
    order_blocks = detect_order_blocks(df_signal)
    liquidity = detect_liquidity(df_signal, atr_value=atr_value)
    volume_profile = compute_volume_profile(df_signal)

    osc_series = ind_signal.get("_series", {}).get("rsi")
    divergence = detect_divergence(df_signal, osc_series)

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
        "funding": funding,
        "open_interest": oi,
        "long_short_ratio": ls_ratio,
        "cvd": cvd,
        "macro": macro,
        "onchain": onchain,
        "order_blocks": order_blocks,
        "liquidity": liquidity,
        "volume_profile": volume_profile,
        "divergence": divergence,
        "liquidation_map": liq_map,
        "sweep_signal": sweep,
        "sessions": sessions,
    }
    return context


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
        "cvd_bullish": ctx["cvd"].get("cvd_bullish"),
        "cvd_bearish": ctx["cvd"].get("cvd_bearish"),
        "funding": (ctx["funding"] or {}).get("current"),
        "exchange_netflow": (ctx["onchain"] or {}).get("exchange_netflow"),
    })
    return flat


async def run_cascade(ctx: dict[str, Any], delivered_today: list[dict[str, Any]],
                      last_signal: dict[str, Any] | None,
                      interpret: bool = True) -> dict[str, Any]:
    """Run levels 1-7 and return a result describing the outcome."""
    flat = _flatten_for_confluence(ctx)
    atr_value = ctx["atr"] or 0.0

    # Level 1: HTF bias.
    htf_bias = get_htf_bias(ctx["ind_1d"])

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

    # Level 5: multi-timeframe confidence modifier.
    t1h = trend_label(ctx["ind_1h"])
    t4h = trend_label(ctx["ind_4h"])
    t1d = trend_label(ctx["ind_1d"])
    base_conf = total / 10.0
    modifier = apply_mtf_confidence(1.0, t1h, t4h, t1d)
    confidence = min(base_conf * modifier, 1.0)

    # Risk sizing.
    position = calculate_position(ctx["price"], atr_value, direction=direction)

    signal = {
        "symbol": ctx["symbol"],
        "timeframe": ctx["timeframe"],
        "direction": direction,
        "entry_price": ctx["price"],
        "price": ctx["price"],
        "stop_loss": position["stop_loss"],
        "target_1": position["target_1"],
        "target_2": position["target_2"],
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
        "mtf": {"1h": t1h, "4h": t4h, "1d": t1d},
        "session_best": ctx["sessions"].get("best_session"),
        "sessions": ctx["sessions"].get("sessions"),
        "funding_value": (ctx["funding"] or {}).get("current"),
        "timestamp": ctx["timestamp"],
    }

    # Level 6: cooldown / dedup (blocking).
    if not should_send_signal(signal, last_signal, atr_value, config.COOLDOWN_HOURS):
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


def _blocked(stage: str, **extra: Any) -> dict[str, Any]:
    result = {"status": "blocked", "blocked_at": stage, "deliverable": False}
    result.update(extra)
    return result


def _safe(value: Any, default: Any) -> Any:
    if isinstance(value, Exception):
        log.warning("data source error: %s", value)
        return default
    return value
