"""Stage C1.8: swing edge discovery — feature attribution, not score attribution.

Read-only measurement. НЕ торгует, НЕ отправляет ордера, НЕ ходит в сеть,
НЕ пишет в БД, НЕ меняет схему, scoring, thresholds, config или runtime.
НЕ импортируется runtime-кодом — только ручной CLI-запуск. Отчёты пишутся
в reports/c18/ (артефакты, не коммитятся).

Философия (в отличие от C1.6/deep_diagnostics): цель НЕ объяснить поведение
существующего score. Цель — обнаружить, какие рыночные свойства объясняют
net expectancy после издержек, независимо от того, отражает их сегодняшний
score или нет. Score оценивается только в самом конце (score_evaluation),
как последний шаг, а не как объект исследования.

Recorder-only архитектура: deep_backtest.py НЕ модифицируется. Вместо этого
временно подменяются module-global имена, на которые deep_walk ссылается
изнутри своего цикла (тот же приём, что и в C1.6 tools.deep_diagnostics.
walk_with_positions, только шире — не один resolve, а семь функций). Каждая
обёртка захватывает то, что deep_walk и так уже посчитал как-of текущего
бара, и делегирует оригиналу без изменения результата. try/finally всегда
возвращает оригиналы. Ни порядок гейтов, ни семантика scoring не меняются.

Запуск:

    .venv/bin/python -m tools.deep_discovery \
      --dataset data/klines --exchange binance --symbol BTCUSDT \
      --profile swing --bars 70080 --outdir reports/c18
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

import config
import tools.deep_backtest as deep_backtest
from analyzer.indicators import ema
from tools.deep_backtest import DeepBacktestError, EXCHANGES, prepare
from signal_engine.confluence import CATEGORIES
from signal_engine.profiles import PROFILES

STAGE = "C1.8"

# Пороги доказательности — те же значения, что и в C1.6 tools.deep_diagnostics
# (сознательно продублированы: этот модуль не импортирует deep_diagnostics).
MIN_GROUP_N = 50
MIN_TOTAL_RESOLVED = 200
POSITIVE_EPS = 0.05

EMA_SLOPE_LOOKBACK = 10  # баров HTF-среза для наклона EMA50 (без пересчёта)

_recording = False  # non-reentrancy guard for record_walk, see below

LIMITATIONS = [
    "Discovery only: this stage finds candidate market features, it does not "
    "recommend a strategy, scoring, threshold, or config change.",
    "Feature discovery, not score decomposition: candidate features are "
    "evaluated on their own terms, independent of whether the current score "
    "already reflects them (see score_evaluation, computed last).",
    "R semantics reused verbatim from tools.deep_backtest.resolve/deep_walk "
    "(cost-adjusted, conservative intrabar stop-first).",
    "No sealed holdout is read: discovery folds come from the walk-forward "
    "region only (see fold_stability); the holdout tail is never opened.",
    "No new technical indicators are introduced; trend-strength proxies reuse "
    "already-computed structural information (HTF regime, EMA slope from "
    "already-computed EMA50, distance from equilibrium, HTF alignment flags).",
    "Single exchange / single symbol per run; unresolved setups are excluded "
    "from all R aggregates and reported as a count.",
]


# ---------------------------------------------------------------------------
# Recorder: wrap module-globals deep_walk already calls, capture, delegate.
# ---------------------------------------------------------------------------

def _ema_slope(htf_slice: pd.DataFrame, price: float) -> float | None:
    """Наклон EMA50 за EMA_SLOPE_LOOKBACK баров HTF-среза, в price/bar/price.

    Не новый индикатор: EMA50 уже считает compute_indicators (analyzer.
    indicators.ema) для того же среза; здесь лишь берётся наклон вместо
    последнего значения, на уже вычисленном же виде среза (htf_slice —
    аргумент, с которым deep_walk и так вызывает compute_equilibrium).
    """
    close = htf_slice["close"]
    if len(close) <= EMA_SLOPE_LOOKBACK or not price:
        return None
    e50 = ema(close, 50)
    now, then = e50.iloc[-1], e50.iloc[-1 - EMA_SLOPE_LOOKBACK]
    if pd.isna(now) or pd.isna(then):
        return None
    return round(float(now - then) / EMA_SLOPE_LOOKBACK / price, 6)


def record_walk(frames: dict[str, Any], profile: dict[str, Any],
                bars: int) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Один deep_walk с расширенным рекордером (не только resolve, как в C1.6).

    deep_walk обращается к этим семи именам как к module-global в своём теле
    (tools/deep_backtest.py), поэтому подмена видна изнутри без изменения
    файла. Все обёртки только читают аргументы/результат и делегируют
    оригиналу — вычисляемые числа не меняются ни на бит.

    Не реентерабельно: вложенный/перекрывающийся вызов захватил бы уже
    подменённые имена как «оригиналы» и оставил бы deep_backtest подменённым
    навсегда после выхода обоих вызовов. Этот тул — однопоточный CLI, вызовы
    всегда последовательны, но защита всё равно ставится явно, а не остаётся
    на совести вызывающего.
    """
    global _recording
    if _recording:
        raise RuntimeError(
            "record_walk is not reentrant: a call is already in progress")
    _recording = True
    originals = {
        name: getattr(deep_backtest, name)
        for name in ("calculate_confluence_score", "compute_equilibrium",
                     "_build_htf_zones", "compute_cvd_from_klines",
                     "analyze_volatility", "get_htf_bias", "resolve")
    }

    last: dict[str, Any] = {}
    last_scores: dict[str, dict[str, int]] = {}
    records: dict[int, dict[str, Any]] = {}

    def rec_confluence(data, direction):
        last["flat"] = data
        total, scores, reasons = originals["calculate_confluence_score"](
            data, direction)
        last_scores[direction] = scores
        return total, scores, reasons

    def rec_equilibrium(df):
        last["htf_slice"] = df
        result = originals["compute_equilibrium"](df)
        last["eq"] = result
        return result

    def rec_zones(dfs, inds, zone_tfs, price, entry_df):
        last["zdfs"] = dfs
        result = originals["_build_htf_zones"](dfs, inds, zone_tfs, price,
                                                entry_df)
        last["zones"] = result
        return result

    def rec_cvd(df):
        result = originals["compute_cvd_from_klines"](df)
        last["cvd"] = result
        return result

    def rec_volatility(df, tf_per_day=6.0):
        result = originals["analyze_volatility"](df, tf_per_day=tf_per_day)
        last["volatility"] = result
        return result

    def rec_htf_bias(data_1d):
        result = originals["get_htf_bias"](data_1d)
        last["htf_bias"] = result
        return result

    def rec_resolve(df, entry_idx, direction, pos, hold_bars):
        outcome = originals["resolve"](df, entry_idx, direction, pos, hold_bars)
        records[entry_idx] = _snapshot(df, entry_idx, direction, pos,
                                       hold_bars, dict(last), last_scores.get(
                                           direction, {}), outcome)
        return outcome

    deep_backtest.calculate_confluence_score = rec_confluence
    deep_backtest.compute_equilibrium = rec_equilibrium
    deep_backtest._build_htf_zones = rec_zones
    deep_backtest.compute_cvd_from_klines = rec_cvd
    deep_backtest.analyze_volatility = rec_volatility
    deep_backtest.get_htf_bias = rec_htf_bias
    deep_backtest.resolve = rec_resolve
    try:
        walk = deep_backtest.deep_walk(frames, profile, bars)
    finally:
        for name, fn in originals.items():
            setattr(deep_backtest, name, fn)
        _recording = False
    return walk, records


def _mfe_mae(df: pd.DataFrame, entry_idx: int, direction: str,
            pos: dict[str, Any], hold_bars: int) -> dict[str, Any]:
    """Max favorable/adverse excursion in R, over the exact resolve() window.

    Same forward window as tools.deep_backtest.resolve (entry_idx+1 ..
    min(entry_idx+hold_bars, len(df)-1)) — no bars are read beyond what
    resolve() already reads for this setup.
    """
    entry, stop = pos["entry_price"], pos["stop_loss"]
    risk = abs(entry - stop)
    long = direction == "long"
    horizon_idx = entry_idx + hold_bars
    last_idx = min(horizon_idx, len(df) - 1)

    mfe = mae = 0.0
    t_mfe = t_mae = None
    for j in range(entry_idx + 1, last_idx + 1):
        high, low = float(df["high"].iloc[j]), float(df["low"].iloc[j])
        fav = (high - entry) if long else (entry - low)
        adv = (entry - low) if long else (high - entry)
        fav_r = fav / risk if risk else 0.0
        adv_r = adv / risk if risk else 0.0
        if fav_r > mfe:
            mfe, t_mfe = fav_r, j - entry_idx
        if adv_r > mae:
            mae, t_mae = adv_r, j - entry_idx
    return {"mfe_r": round(mfe, 4), "mae_r": round(mae, 4),
           "time_to_mfe": t_mfe, "time_to_mae": t_mae}


def _nearest_zone_box(zones: dict[str, Any], direction: str
                      ) -> tuple[dict[str, Any] | None, str | None]:
    """OB/FVG box price currently sits in, for this direction, if any."""
    ob, fvg = zones["order_blocks"], zones["fvg"]
    bull = direction == "long"
    ob_key = "bullish_ob" if bull else "bearish_ob"
    fvg_key = "bullish_fvg" if bull else "bearish_fvg"
    ob_in = "price_in_bullish_ob" if bull else "price_in_bearish_ob"
    fvg_in = "price_in_bullish_fvg" if bull else "price_in_bearish_fvg"
    if ob.get(ob_in) and ob.get(ob_key):
        return ob[ob_key], "ob"
    if fvg.get(fvg_in) and fvg.get(fvg_key):
        return fvg[fvg_key], "fvg"
    return None, None


def _zone_age(box: dict[str, Any], zdfs: dict[str, Any]) -> int | None:
    tf = box.get("tf")
    zdf = zdfs.get(tf)
    if zdf is None or "index" not in box:
        return None
    return int(len(zdf) - 1 - box["index"])


def _nearest_level_distance(levels: dict[str, Any], price: float,
                            atr: float | None) -> float | None:
    candidates = levels.get("highs", []) + levels.get("lows", [])
    if not candidates or not atr:
        return None
    return round(min(abs(price - lv) for lv in candidates) / atr, 4)


def _snapshot(df: pd.DataFrame, entry_idx: int, direction: str,
             pos: dict[str, Any], hold_bars: int, last: dict[str, Any],
             scores: dict[str, int], outcome: dict[str, Any]) -> dict[str, Any]:
    """Assemble one setup's discovery feature record from captured context."""
    flat = last.get("flat", {})
    eq = last.get("eq", {}) or {}
    zones = last.get("zones")
    zdfs = last.get("zdfs", {})
    cvd = last.get("cvd", {}) or {}
    vol = last.get("volatility", {}) or {}
    htf_slice = last.get("htf_slice")

    price = pos["entry_price"]
    risk = abs(price - pos["stop_loss"])
    atr = flat.get("atr")
    bull = direction == "long"

    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100
    cost_r = (cost_pct * price / risk) if risk else 0.0
    gross_r = outcome["r"]

    box, zone_kind = (_nearest_zone_box(zones, direction)
                      if zones is not None else (None, None))
    zone_width = zone_age = entry_location = None
    if box is not None:
        lo, hi = box["low"], box["high"]
        zone_width = round((hi - lo) / atr, 4) if atr else None
        entry_location = (round((price - lo) / (hi - lo), 4)
                          if hi > lo else None)
        zone_age = _zone_age(box, zdfs)

    open_time = df["open_time"].iloc[entry_idx]
    ts = pd.Timestamp(open_time, tz="UTC") if not isinstance(
        open_time, pd.Timestamp) else open_time
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    mm = _mfe_mae(df, entry_idx, direction, pos, hold_bars)

    return {
        # structure
        "atr": atr,
        "volatility_atr_pct": vol.get("atr_pct"),
        "volatility_atr_percentile": vol.get("atr_percentile"),
        "htf_bias": last.get("htf_bias"),
        "eq_zone": eq.get("zone"),
        "eq_pos": eq.get("pos"),
        "ema_slope": (_ema_slope(htf_slice, price)
                     if htf_slice is not None else None),
        "trend_aligned": bool(flat.get("trend_aligned_bottom")
                             if bull else flat.get("trend_aligned_top")),
        # liquidity / geometry
        "ob_present": bool(flat.get("price_in_bullish_ob")
                          if bull else flat.get("price_in_bearish_ob")),
        "fvg_present": bool(flat.get("price_in_bullish_fvg")
                           if bull else flat.get("price_in_bearish_fvg")),
        "zone_kind": zone_kind,
        "zone_width_atr": zone_width,
        "zone_age_bars": zone_age,
        "entry_location": entry_location,
        "liquidity_swept": bool(flat.get("liquidity_swept_below")
                               if bull else flat.get("liquidity_swept_above")),
        "distance_to_level_atr": (_nearest_level_distance(
            zones["levels"], price, atr) if zones is not None else None),
        # execution
        "stop_distance_pct": round(risk / price, 6) if price else None,
        "rr_ratio": (round(abs(pos["target_2"] - price) / risk, 4)
                    if risk else None),
        "cost_r": round(cost_r, 4),
        "gross_r": round(gross_r, 4) if gross_r is not None else None,
        # orderflow
        "cvd_value": cvd.get("cvd"),
        "cvd_last_delta": cvd.get("last_delta"),
        "cvd_bullish": bool(cvd.get("cvd_bullish")),
        "cvd_bearish": bool(cvd.get("cvd_bearish")),
        # time
        "weekday": ts.weekday(),
        "month": ts.month,
        "distance_to_weekend": min(ts.weekday(), 6 - ts.weekday()),
        # score components (for later score_evaluation only — not the object
        # of discovery itself)
        "score_components": dict(scores),
        # trade path
        **mm,
    }


# ---------------------------------------------------------------------------
# Merge setups + discovery records into flat per-setup feature rows.
# ---------------------------------------------------------------------------

def build_feature_rows(walk: dict[str, Any],
                       records: dict[int, dict[str, Any]]
                       ) -> list[dict[str, Any]]:
    """One row per resolved setup: outcome fields + captured features."""
    rows = []
    for s in walk["setups"]:
        if s["outcome"] == "unresolved":
            continue
        rec = records.get(s["idx"])
        if rec is None:
            continue
        row = {
            "idx": s["idx"], "direction": s["direction"], "score": s["score"],
            "regime_current": s["regime_current"],
            "regime_live_parity": s["regime_live_parity"],
            "outcome": s["outcome"], "hit_tp1": s["hit_tp1"], "r": s["r"],
            **rec,
        }
        rows.append(row)
    return rows
