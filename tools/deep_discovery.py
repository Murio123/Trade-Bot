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
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

import config
import tools.deep_backtest as deep_backtest
from analyzer.indicators import ema
from tools.deep_backtest import DeepBacktestError, EXCHANGES, prepare
from signal_engine.profiles import PROFILES

STAGE = "C1.8"

# Пороги доказательности — те же значения, что и в C1.6 tools.deep_diagnostics
# (сознательно продублированы: этот модуль не импортирует deep_diagnostics).
MIN_GROUP_N = 50
MIN_TOTAL_RESOLVED = 200
POSITIVE_EPS = 0.05
FLAT_EPS = 0.02

# Доля фолдов/годов, согласных со знаком пула, ниже которой стабильность не
# засчитывается (не голосование большинством одного фолда — супербольшинство).
STABILITY_AGREEMENT_MIN = 0.66
N_FOLDS = 4  # временные блоки по idx для fold_stability (не WFConfig: здесь
             # проверяется устойчивость знака ассоциации, а не forward-производительность,
             # поэтому purge/embargo не нужны — это отдельная документированная граница)

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
        "year": ts.year,
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


# ---------------------------------------------------------------------------
# Part 3 (C1.8, step 4): single-feature attribution.
#
# Feature discovery, not score decomposition: every candidate below is
# evaluated on its own terms — "score" and "score_bucket" are included as ONE
# feature among many, not as the object of investigation. No pairwise or
# multivariate analysis happens here (that is redundancy/interactions, later
# stages); each feature's buckets are built and scored in isolation.
# ---------------------------------------------------------------------------

SCORE_BUCKETS: list[tuple[int, int | None]] = [
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, None)]

# name -> "categorical" | "numeric". Categorical buckets by literal value;
# numeric buckets by pooled tertile (low/mid/high, degenerate distributions
# collapse to fewer bins rather than being forced into 3).
CATEGORICAL_FEATURES = [
    "direction", "regime_current", "regime_live_parity", "regime_agreement",
    "htf_bias", "eq_zone", "trend_aligned", "ob_present", "fvg_present",
    "zone_kind", "liquidity_swept", "cvd_bullish", "cvd_bearish", "hit_tp1",
    "weekday", "month", "score_bucket",
]
NUMERIC_FEATURES = [
    "atr", "volatility_atr_pct", "volatility_atr_percentile", "ema_slope",
    "eq_pos", "distance_from_equilibrium", "zone_width_atr", "zone_age_bars",
    "entry_location", "distance_to_level_atr", "stop_distance_pct",
    "rr_ratio", "cost_r", "mfe_r", "mae_r", "time_to_mfe", "time_to_mae",
    "cvd_value", "cvd_last_delta", "distance_to_weekend",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def score_bucket(score: float) -> str:
    for lo, hi in SCORE_BUCKETS:
        if hi is None:
            if score >= lo:
                return f"[{lo},inf)"
        elif lo <= score < hi:
            return f"[{lo},{hi})"
    return "below_min"


def _feature_value(row: dict[str, Any], name: str) -> Any:
    """Direct field lookup, plus a few features derived from captured ones."""
    if name == "regime_agreement":
        return ("agree" if row["regime_current"] == row["regime_live_parity"]
               else "disagree")
    if name == "score_bucket":
        return score_bucket(row["score"])
    if name == "distance_from_equilibrium":
        pos = row.get("eq_pos")
        return round(abs(pos - 0.5), 4) if pos is not None else None
    return row.get(name)


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _numeric_buckets(rows: list[dict[str, Any]], name: str
                     ) -> dict[int, str]:
    """idx -> tertile label for one numeric feature (fewer bins if degenerate)."""
    pairs = [(r["idx"], _feature_value(r, name)) for r in rows]
    valid = [(i, v) for i, v in pairs if not _is_missing(v)]
    if len(valid) < 3:
        return {}
    values = [v for _, v in valid]
    try:
        cats = pd.qcut(values, q=3, duplicates="drop")
    except ValueError:
        return {}
    n_bins = len(cats.categories)
    if n_bins < 2:
        return {}
    names = ["low", "mid", "high"][:n_bins]
    return {i: names[code] for (i, _), code in zip(valid, cats.codes)}


def _stats_for_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    net_rs = [r["r"] for r in rows]
    gross_rs = [r["gross_r"] for r in rows if r.get("gross_r") is not None]
    wins = sum(1 for r in rows if r["outcome"] == "win")
    timeouts = sum(1 for r in rows if r["outcome"] == "timeout")
    return {
        "n": n,
        "win_rate": round(wins / n, 4),
        "gross_mean_r": round(sum(gross_rs) / len(gross_rs), 4) if gross_rs
                        else None,
        "net_mean_r": round(sum(net_rs) / n, 4),
        "median_r": round(statistics.median(net_rs), 4),
        "timeout_share": round(timeouts / n, 4),
    }


def feature_bucket_table(rows: list[dict[str, Any]], name: str
                         ) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if name in NUMERIC_FEATURES:
        labels = _numeric_buckets(rows, name)
        for r in rows:
            lbl = labels.get(r["idx"])
            if lbl is not None:
                groups[lbl].append(r)
    else:
        for r in rows:
            v = _feature_value(r, name)
            if not _is_missing(v):
                groups[str(v)].append(r)
    return {k: _stats_for_group(v) for k, v in sorted(groups.items())}


def _sign(x: float, eps: float = FLAT_EPS) -> str:
    if x > eps:
        return "positive"
    if x < -eps:
        return "negative"
    return "flat"


def feature_separation(bucket_table: dict[str, dict[str, Any]],
                       min_n: int = MIN_GROUP_N) -> dict[str, Any] | None:
    """Best-vs-worst eligible bucket by net_mean_r. None if <2 eligible."""
    eligible = {k: v for k, v in bucket_table.items() if v["n"] >= min_n}
    if len(eligible) < 2:
        return None
    best = max(eligible, key=lambda k: eligible[k]["net_mean_r"])
    worst = min(eligible, key=lambda k: eligible[k]["net_mean_r"])
    spread = round(eligible[best]["net_mean_r"] - eligible[worst]["net_mean_r"], 4)
    return {"best": best, "worst": worst, "spread": spread,
           "best_mean_r": eligible[best]["net_mean_r"],
           "best_n": eligible[best]["n"], "sign": _sign(spread)}


def _fold_assignment(rows: list[dict[str, Any]], k: int = N_FOLDS
                     ) -> dict[int, int]:
    """idx -> contiguous temporal fold id (equal-count blocks, in idx order).

    Not tools.deep_backtest.WFConfig: that machinery purges/embargoes for
    forward-performance measurement. Here the question is only "does this
    feature's sign of association replicate across independent time blocks",
    so no purge/embargo is needed — documented as a limitation, not silently
    assumed equivalent to a walk-forward performance claim.
    """
    ordered = sorted({r["idx"] for r in rows})
    n = len(ordered)
    if n == 0:
        return {}
    assign: dict[int, int] = {}
    for pos, idx in enumerate(ordered):
        fold = min(k - 1, pos * k // n)
        assign[idx] = fold
    return assign


def _stability(rows: list[dict[str, Any]], name: str,
              pooled_sep: dict[str, Any],
              split_key: Callable[[dict[str, Any]], Any],
              min_n: int = MIN_GROUP_N) -> dict[str, Any]:
    """Does the pooled best-vs-worst bucket sign replicate across splits?"""
    splits: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        splits[split_key(r)].append(r)

    agree = 0
    evaluated = 0
    details: dict[str, Any] = {}
    for key in sorted(splits, key=str):
        sub = splits[key]
        table = feature_bucket_table(sub, name)
        best_row, worst_row = table.get(pooled_sep["best"]), \
            table.get(pooled_sep["worst"])
        if not best_row or not worst_row or best_row["n"] < min_n \
                or worst_row["n"] < min_n:
            details[str(key)] = {"status": "insufficient"}
            continue
        delta = round(best_row["net_mean_r"] - worst_row["net_mean_r"], 4)
        sign = _sign(delta)
        matches = sign == pooled_sep["sign"] if pooled_sep["sign"] != "flat" \
            else None
        details[str(key)] = {"status": "evaluated", "delta": delta,
                             "sign": sign, "matches_pooled": matches}
        evaluated += 1
        if matches:
            agree += 1
    return {
        "eligible": evaluated >= 2,
        "agree": agree, "evaluated": evaluated,
        "agreement_ratio": round(agree / evaluated, 4) if evaluated else None,
        "details": details,
    }


def _is_stable(stab: dict[str, Any]) -> bool:
    return bool(stab.get("eligible")) and \
        (stab.get("agreement_ratio") or 0) >= STABILITY_AGREEMENT_MIN


def classify_feature(name: str, pooled_table: dict[str, dict[str, Any]],
                     fold_stab: dict[str, Any], year_stab: dict[str, Any],
                     total_resolved: int) -> dict[str, Any]:
    """PRELIMINARY_KEEP / PRELIMINARY_REMOVE / PRELIMINARY_UNKNOWN.

    Statistical-only gate (fold + year stability); regime_stability is
    reported but not gating here — regime-conditional re-testing of
    inconclusive features is a later stage (Part 3, "useful only under
    certain regimes"), not this pass. No feature reaches a bare KEEP: that
    label is reserved for post-economic-validation (later stage).
    """
    note = ("preliminary, statistical only; economic validation and "
            "regime-conditional re-testing happen in a later stage")
    if total_resolved < MIN_TOTAL_RESOLVED:
        return {"label": "PRELIMINARY_UNKNOWN",
                "reason": f"resolved setups {total_resolved} < "
                          f"{MIN_TOTAL_RESOLVED}", "note": note}
    sep = feature_separation(pooled_table)
    if sep is None:
        return {"label": "PRELIMINARY_UNKNOWN",
                "reason": f"fewer than 2 buckets reach n >= {MIN_GROUP_N}",
                "note": note}
    if sep["sign"] == "flat":
        return {"label": "PRELIMINARY_REMOVE",
                "reason": f"buckets do not differ (spread "
                          f"{sep['spread']:+.4f} within ±{FLAT_EPS})",
                "note": note}
    if sep["best_mean_r"] < POSITIVE_EPS:
        return {"label": "PRELIMINARY_REMOVE",
                "reason": f"even the best bucket ({sep['best']}) is not "
                          f"positive (mean_r {sep['best_mean_r']:+.4f} < "
                          f"+{POSITIVE_EPS})", "note": note}
    if _is_stable(fold_stab) and _is_stable(year_stab):
        return {"label": "PRELIMINARY_KEEP",
                "reason": f"{sep['best']} vs {sep['worst']}: mean_r "
                          f"{sep['best_mean_r']:+.4f}, spread "
                          f"{sep['spread']:+.4f}; fold agreement "
                          f"{fold_stab.get('agreement_ratio')}, year "
                          f"agreement {year_stab.get('agreement_ratio')}",
                "note": note}
    return {"label": "PRELIMINARY_UNKNOWN",
           "reason": f"positive separation exists ({sep['best']} mean_r "
                     f"{sep['best_mean_r']:+.4f}) but fold/year stability "
                     f"insufficient (fold={fold_stab.get('agreement_ratio')}, "
                     f"year={year_stab.get('agreement_ratio')})",
           "note": note}


def run_single_feature_attribution(rows: list[dict[str, Any]]
                                   ) -> dict[str, Any]:
    """One entry per feature: buckets, separation, stability, classification."""
    fold_of = _fold_assignment(rows)
    features: dict[str, Any] = {}
    for name in ALL_FEATURES:
        table = feature_bucket_table(rows, name)
        sep = feature_separation(table)
        if sep is None:
            fold_stab = {"eligible": False, "reason": "no pooled separation"}
            year_stab = dict(fold_stab)
            regime_stab = dict(fold_stab)
        else:
            fold_stab = _stability(rows, name, sep,
                                   lambda r: fold_of.get(r["idx"]))
            year_stab = _stability(rows, name, sep, lambda r: r["year"])
            regime_stab = (
                {"eligible": False, "reason": "feature is the split axis"}
                if name == "regime_current" else
                _stability(rows, name, sep, lambda r: r["regime_current"]))
        classification = classify_feature(name, table, fold_stab, year_stab,
                                          len(rows))
        features[name] = {
            "kind": "numeric" if name in NUMERIC_FEATURES else "categorical",
            "buckets": table,
            "separation": sep,
            "fold_stability": fold_stab,
            "year_stability": year_stab,
            "regime_stability": regime_stab,
            "classification": classification,
        }
    return features


def build_summary_table(features: dict[str, Any]) -> list[dict[str, Any]]:
    table = []
    for name, f in features.items():
        sep = f["separation"]
        best_bucket = f["buckets"].get(sep["best"]) if sep else None
        table.append({
            "feature": name,
            "sample": best_bucket["n"] if best_bucket else 0,
            "gross_r": best_bucket["gross_mean_r"] if best_bucket else None,
            "net_r": best_bucket["net_mean_r"] if best_bucket else None,
            "fold_stability": f["fold_stability"].get("agreement_ratio"),
            "year_stability": f["year_stability"].get("agreement_ratio"),
            "regime_stability": f["regime_stability"].get("agreement_ratio"),
            "classification": f["classification"]["label"],
        })
    return table


# ---------------------------------------------------------------------------
# Report assembly + CLI
# ---------------------------------------------------------------------------

def run_discovery(dataset: str, exchange: str, symbol: str, profile_name: str,
                  bars: int, max_gap_ratio: float = 0.001,
                  allow_estimated_cvd: bool = False) -> dict[str, Any]:
    frames, profile, _table, cvd_method = prepare(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)
    walk, records = record_walk(frames, profile, bars)
    rows = build_feature_rows(walk, records)
    features = run_single_feature_attribution(rows)
    resolved = [s for s in walk["setups"] if s["outcome"] != "unresolved"]
    return {
        "stage": STAGE,
        "kind": "single_feature_attribution",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "exchange": exchange,
        "symbol": symbol,
        "bars_requested": bars,
        "cvd_method": cvd_method,
        "counts": {"setups": len(walk["setups"]), "resolved": len(resolved),
                  "unresolved": len(walk["setups"]) - len(resolved)},
        "features": features,
        "summary_table": build_summary_table(features),
        "limitations": list(LIMITATIONS) + [
            "Single-feature attribution only (Part 3, step 4): no pairwise, "
            "multivariate, clustering, SHAP, or permutation-importance "
            "analysis is included; those are later stages.",
            "Classification is PRELIMINARY only: PRELIMINARY_KEEP requires "
            "statistical fold/year stability, not economic validation "
            "(a later stage); no feature is a trading recommendation.",
            "fold_stability uses contiguous temporal idx-blocks, not "
            "tools.deep_backtest.WFConfig purge/embargo geometry — it tests "
            "sign replication, not forward performance.",
        ],
    }


_SUMMARY_COLS = (f"{'sample':>7} {'gross_r':>8} {'net_r':>8} "
                 f"{'fold':>6} {'year':>6} {'regime':>7} classification")


def format_report(report: dict[str, Any]) -> str:
    c = report["counts"]
    lines = [
        f"single-feature attribution ({report['stage']}) — "
        f"{report['exchange']} {report['symbol']} profile={report['profile']}",
        f"  generated_at   : {report['generated_at_utc']}",
        f"  bars_requested : {report['bars_requested']}",
        f"  cvd_method     : {report['cvd_method']}",
        f"  setups         : {c['setups']} (resolved {c['resolved']}, "
        f"unresolved {c['unresolved']})",
        "",
        f"{'feature':<28}{_SUMMARY_COLS}",
    ]

    def _n(v):
        return f"{v:>+8.4f}" if isinstance(v, (int, float)) else f"{'—':>8}"

    def _r(v):
        return f"{v:>6.2f}" if isinstance(v, (int, float)) else f"{'—':>6}"

    for row in report["summary_table"]:
        lines.append(
            f"{row['feature']:<28}{row['sample']:>7} {_n(row['gross_r'])} "
            f"{_n(row['net_r'])} {_r(row['fold_stability'])} "
            f"{_r(row['year_stability'])} {_r(row['regime_stability']):>7} "
            f"{row['classification']}")

    lines.append("")
    lines.append("limitations:")
    lines.extend(f"  - {item}" for item in report["limitations"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.deep_discovery",
        description="offline swing edge discovery, single-feature "
                    "attribution (Stage C1.8, step 4). Reads cached files "
                    "only; never touches the network, the DB or the "
                    "runtime.")
    parser.add_argument("--dataset", required=True,
                        help="directory written by tools.kline_cache")
    parser.add_argument("--exchange", required=True, choices=EXCHANGES)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--bars", type=int, required=True,
                        help="entry-timeframe bars to walk")
    parser.add_argument("--max-gap-ratio", type=float, default=0.001)
    parser.add_argument("--allow-estimated-cvd", action="store_true")
    parser.add_argument("--outdir", default="reports/c18",
                        help="artifact directory for the JSON/text reports")
    parser.add_argument("--json", action="store_true",
                        help="print JSON to stdout instead of the text report")
    args = parser.parse_args(argv)

    if args.bars <= 0:
        parser.error("--bars must be positive")

    try:
        report = run_discovery(args.dataset, args.exchange, args.symbol,
                               args.profile, args.bars, args.max_gap_ratio,
                               args.allow_estimated_cvd)
    except DeepBacktestError as exc:
        print(f"deep_discovery: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, "single_feature_report")
    with open(f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    text = format_report(report)
    with open(f"{stem}.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)
          if args.json else text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
