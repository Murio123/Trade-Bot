"""Stage C2.2b: offline simulator for two independent SWING entry hypotheses.

H1 — Trend Pullback Continuation, H2 — Range Mean Reversion. Both specs were
frozen in C2.2a (see reports/c22/implementation_summary.md) and are NOT
retuned here after seeing any result.

НЕ торгует, НЕ отправляет ордера, НЕ ходит в сеть, НЕ пишет в БД, НЕ меняет
scoring/thresholds/config/runtime. НЕ импортируется runtime-кодом — только
ручной CLI-запуск. Отчёты пишутся в reports/c22/ (артефакты, не коммитятся).

Why this cannot reuse tools.deep_backtest.deep_walk's setup list: deep_walk
only reaches resolve() for bars that already pass the CURRENT confluence
score's gates (direction_conflict, htf_policy_blocked, below_min_threshold,
dead_zone, abnormal_volatility, ...). H1/H2 are new, independent entry
triggers that must be evaluated on EVERY entry bar on their own frozen
criteria — most of which the current score skips entirely. So this module
walks entry bars itself, but reuses every existing PURE, already-tested
building block it can to avoid duplicating pricing/cost/outcome logic:

  * tools.deep_backtest.resolve       — stop/TP1/breakeven/TP2/timeout
                                        mechanics, UNCHANGED, bit-for-bit.
  * tools.deep_backtest.regime_current, .max_hold_bars, .walk_start,
    ._close_ms, ._asof_end, ._IndicatorCache, .HTF_WINDOW — as-of slicing
    and regime tagging, UNCHANGED.
  * tools.deep_discovery._ema_slope   — EMA50 slope on the already-computed
                                        HTF indicator slice (C1.8 recorder).
  * analyzer.indicators.compute_indicators, analyzer.equilibrium.
    compute_equilibrium, analyzer.volatility.analyze_volatility,
    signal_engine.htf_filter.get_htf_bias — same analyzer calls deep_walk
    itself makes, called directly instead of through the score pipeline.
  * The cost formula (cost_pct * price / risk) is the same one-line
    formula already duplicated in tools/deep_backtest.py and
    tools/deep_discovery.py — a third instance here follows the same
    existing convention, not a new one.

Neither hypothesis uses tools.deep_backtest.calculate_confluence_score,
_build_htf_zones, detect_liquidity, detect_reversal, apply_htf_policy,
dead_zone, abnormal_volatility, calculate_position, _structural_stop, or
_structure_targets — those are the CURRENT strategy's scoring/positioning
machinery, not part of either frozen H1/H2 spec (no OB/FVG/liquidity-sweep
as a primary input, no shared score, own frozen stop/target rules).

Single-target quirk (documented, not a bug): tools.deep_backtest.resolve()
only understands a two-stage TP1(partial)->breakeven->TP2 lifecycle. H1's
frozen spec defines exactly ONE structural target, so H1 sets
target_1 == target_2 (both equal to the single target) rather than forking
resolve()'s logic to support a single-target trade — reusing resolve()
unchanged was preferred over duplicating its outcome mechanics. The
documented side effect: on the bar the target is first touched, resolve()
only registers hit_tp1 and continues; the "win" is only recorded once the
NEXT bar still clears target_2. If price touches the target and then closes
back at/through entry on the very next bar before reconfirming, the trade
resolves "breakeven" instead of "win" even though the target was reached.
H2 does not have this quirk: its frozen spec already defines two real
stages (TP1 = range midpoint, TP2 = opposite extreme), which map onto
resolve()'s mechanics exactly as intended.

No walk-forward / no sealed holdout in this module: this is a single-pass
discovery-region simulator (C2.2b). Purge/embargo/holdout folding is C2.2c's
job, reusing tools.deep_backtest.WFConfig/fold_windows/partition_setups
unchanged, same as C2.1. Callers of this module must pass a --bars value
that already excludes any reserved holdout tail.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

import config
import tools.deep_backtest as deep_backtest
from analyzer.equilibrium import compute_equilibrium
from analyzer.indicators import compute_indicators
from analyzer.volatility import analyze_volatility
from signal_engine.htf_filter import get_htf_bias
from signal_engine.vetoes import TF_HOURS
from tools.deep_backtest import (DeepBacktestError, EXCHANGES, HTF_WINDOW,
                                 _IndicatorCache, _asof_end, _close_ms, _iso,
                                 max_hold_bars, prepare, resolve, walk_start)
from tools.deep_discovery import _ema_slope
from tools.kline_dataset import ENTRY_WARMUP, HTF_WARMUP

STAGE = "C2.2b"
HYPOTHESES = ("H1", "H2")

# ---------------------------------------------------------------------------
# Frozen constants (C2.2a). NOT to be changed after seeing any result.
# ---------------------------------------------------------------------------
STRUCTURAL_WINDOW_BARS = 40  # shared rolling swing-extreme window, both H1/H2

H1_STOP_ATR_BUFFER = 0.25
H1_MIN_RR = 1.5
H1_MAX_VOLATILITY_PERCENTILE = 90
H1_MAX_COST_R = 0.15

H2_EQ_POS_EXTREME = 0.20          # long if eq_pos <= this, short if >= 1-this
H2_VOLATILITY_BAND = (10, 85)
H2_STOP_ATR_BUFFER = 0.30
H2_MAX_COST_R = 0.10

RANDOM_SEED = 42

LIMITATIONS = [
    "C2.2b is a single-pass discovery-region simulator: no walk-forward "
    "folds, no purge/embargo, no sealed holdout (that is C2.2c).",
    "H1 and H2 are evaluated and reported completely independently; no "
    "shared score, no blended expectancy, no combined aggregate.",
    "All frozen constants (C2.2a) are fixed before this module was run "
    "against real data and are not retuned based on results.",
    "H1 sets target_1 == target_2 (single structural target) to reuse "
    "tools.deep_backtest.resolve() unchanged rather than forking its "
    "TP1/breakeven/TP2 mechanics — see module docstring for the documented "
    "single-target quirk this implies.",
    "No OB/FVG/liquidity-sweep field is used as an input to either "
    "hypothesis, including for stop/target placement (plain rolling OHLC "
    "extremes only).",
    "CVD is not used by either frozen spec; it is not added here.",
    "R semantics reused verbatim from tools.deep_backtest.resolve "
    "(cost-adjusted, conservative intrabar stop-first).",
]


# ---------------------------------------------------------------------------
# Per-bar context (reuses only pure, already-tested analyzer/deep_backtest
# calls — never calculate_confluence_score or any of its downstream gates).
# ---------------------------------------------------------------------------

def _bar_context(entry_df, i: int, frames: dict[str, Any],
                 profile: dict[str, Any], close_ms: dict[str, np.ndarray],
                 cache: _IndicatorCache) -> tuple[dict[str, Any] | None, str | None]:
    entry_tf = profile["entry"]
    htf = profile["htf"]
    open_time = entry_df["open_time"].iloc[i]
    entry_sub = entry_df.iloc[max(0, i - (ENTRY_WARMUP - 1)):i + 1]
    ind = compute_indicators(entry_sub)
    atr = ind.get("atr")
    if not atr:
        return None, "no_atr"

    t_ms = int(close_ms[entry_tf][i])
    price = float(entry_df["close"].iloc[i])

    htf_end = _asof_end(close_ms[htf], t_ms)
    htf_slice = frames[htf].df.iloc[max(0, htf_end - HTF_WINDOW):htf_end]
    if len(htf_slice) < HTF_WARMUP:
        return None, "htf_insufficient_history"
    ind_htf = cache.get(htf, htf_end, HTF_WINDOW,
                        lambda: compute_indicators(htf_slice))
    htf_bias = get_htf_bias(ind_htf)
    eq = compute_equilibrium(htf_slice)
    ema_slope = _ema_slope(htf_slice, price)
    vol = analyze_volatility(entry_sub,
                             tf_per_day=24.0 / TF_HOURS.get(entry_tf, 1.0))
    regime_cur = deep_backtest.regime_current(profile, ind_htf)

    return {
        "open_time": open_time, "price": price, "atr": atr,
        "htf_bias": htf_bias, "eq_zone": eq.get("zone"), "eq_pos": eq.get("pos"),
        "ema_slope": ema_slope,
        "volatility_atr_percentile": vol.get("atr_percentile"),
        "regime_current": regime_cur,
    }, None


def _rolling_extremes(entry_df, i: int, window: int
                      ) -> tuple[float, float] | tuple[None, None]:
    """(low, high) over the `window` bars strictly BEFORE i (no look-ahead,
    and i's own bar never defines its own stop/target degenerate-tight)."""
    lo = i - window
    if lo < 0:
        return None, None
    sl = entry_df.iloc[lo:i]
    return float(sl["low"].min()), float(sl["high"].max())


def _cost_r(cost_pct: float, price: float, risk: float) -> float:
    return (cost_pct * price / risk) if risk else 0.0


# ---------------------------------------------------------------------------
# H1 — Trend Pullback Continuation
# ---------------------------------------------------------------------------

def h1_evaluate(ctx: dict[str, Any], entry_df, i: int, cost_pct: float,
                pullback_timing: bool = True
                ) -> tuple[str, dict[str, Any]]:
    """Returns ("reject", reason) or ("candidate", {direction, pos, risk,
    rr, cost_r}). pullback_timing=False evaluates the regime-direction
    baseline: same regime/htf_bias/volatility/RR/cost gates, but skips the
    eq_zone/ema_slope pullback-TIMING conditions (C2.2's baseline #3)."""
    regime = ctx["regime_current"]
    if regime not in ("trend_up", "trend_down"):
        return "reject", "regime_not_trend"
    direction = "long" if regime == "trend_up" else "short"
    expected_bias = "bullish" if direction == "long" else "bearish"
    if ctx["htf_bias"] != expected_bias:
        return "reject", "htf_bias_disagreement"
    vp = ctx["volatility_atr_percentile"]
    if vp is not None and vp > H1_MAX_VOLATILITY_PERCENTILE:
        return "reject", "volatility_extreme"

    if pullback_timing:
        expected_zone = "discount" if direction == "long" else "premium"
        if ctx["eq_zone"] != expected_zone:
            return "reject", "not_pullback_zone"
        slope = ctx["ema_slope"]
        if slope is None or (slope > 0) != (direction == "long"):
            return "reject", "ema_slope_mismatch"

    lo, hi = _rolling_extremes(entry_df, i, STRUCTURAL_WINDOW_BARS)
    if lo is None:
        return "reject", "insufficient_window"

    price, atr = ctx["price"], ctx["atr"]
    if direction == "long":
        if price <= lo:
            return "reject", "structure_already_broken"
        stop = lo - H1_STOP_ATR_BUFFER * atr
        target = hi
        reward, risk = target - price, price - stop
    else:
        if price >= hi:
            return "reject", "structure_already_broken"
        stop = hi + H1_STOP_ATR_BUFFER * atr
        target = lo
        reward, risk = price - target, stop - price

    if reward <= 0:
        return "reject", "no_reward_headroom"
    rr = reward / risk if risk else 0.0
    if rr < H1_MIN_RR:
        return "reject", "rr_below_floor"
    cost_r = _cost_r(cost_pct, price, risk)
    if cost_r > H1_MAX_COST_R:
        return "reject", "cost_r_exceeds_ceiling"

    pos = {"entry_price": price, "stop_loss": stop,
          "target_1": target, "target_2": target}
    return "candidate", {"direction": direction, "pos": pos, "risk": risk,
                         "rr": rr, "cost_r": cost_r}


# ---------------------------------------------------------------------------
# H2 — Range Mean Reversion
# ---------------------------------------------------------------------------

def h2_evaluate(ctx: dict[str, Any], entry_df, i: int, cost_pct: float,
                extreme_timing: bool = True
                ) -> tuple[str, dict[str, Any]]:
    """extreme_timing=False evaluates the regime-direction baseline: skip
    the eq_pos-extreme condition, fade toward the midpoint from whichever
    side of it price currently sits on, still gated by regime/volatility/
    cost."""
    if ctx["regime_current"] != "range":
        return "reject", "regime_not_range"
    vp = ctx["volatility_atr_percentile"]
    if vp is None or not (H2_VOLATILITY_BAND[0] <= vp <= H2_VOLATILITY_BAND[1]):
        return "reject", "volatility_out_of_band"

    lo, hi = _rolling_extremes(entry_df, i, STRUCTURAL_WINDOW_BARS)
    if lo is None:
        return "reject", "insufficient_window"
    price, atr = ctx["price"], ctx["atr"]
    mid = (lo + hi) / 2.0

    eq_pos = ctx["eq_pos"]
    if extreme_timing:
        if eq_pos is None:
            return "reject", "eq_pos_unavailable"
        if eq_pos <= H2_EQ_POS_EXTREME:
            direction = "long"
        elif eq_pos >= (1 - H2_EQ_POS_EXTREME):
            direction = "short"
        else:
            return "reject", "not_at_extreme"
    else:
        direction = "long" if price < mid else "short"

    if direction == "long" and ctx["htf_bias"] == "bearish":
        return "reject", "htf_bias_opposes"
    if direction == "short" and ctx["htf_bias"] == "bullish":
        return "reject", "htf_bias_opposes"

    if direction == "long":
        stop = lo - H2_STOP_ATR_BUFFER * atr
        target1, target2 = mid, hi
        reward, risk = target1 - price, price - stop
    else:
        stop = hi + H2_STOP_ATR_BUFFER * atr
        target1, target2 = mid, lo
        reward, risk = price - target1, stop - price

    if reward <= 0:
        return "reject", "no_reward_headroom"
    rr = reward / risk if risk else 0.0
    cost_r = _cost_r(cost_pct, price, risk)
    if cost_r > H2_MAX_COST_R:
        return "reject", "cost_r_exceeds_ceiling"

    pos = {"entry_price": price, "stop_loss": stop,
          "target_1": target1, "target_2": target2}
    return "candidate", {"direction": direction, "pos": pos, "risk": risk,
                         "rr": rr, "cost_r": cost_r}


_EVALUATORS = {"H1": h1_evaluate, "H2": h2_evaluate}


# ---------------------------------------------------------------------------
# Walk: one pass per hypothesis, timing on/off (full vs regime-direction).
# ---------------------------------------------------------------------------

def _record_reject(hyp: str, i: int, ctx: dict[str, Any], reason: str
                   ) -> dict[str, Any]:
    return {
        "hypothesis": hyp, "idx": i, "timestamp": _iso(ctx["open_time"]),
        "direction": None, "entry": None, "stop": None, "target": None,
        "initial_risk": None, "rr": None, "regime": ctx["regime_current"],
        "volatility_percentile": ctx["volatility_atr_percentile"],
        "cost_r": None, "gross_r": None, "net_r": None, "outcome": None,
        "exit_idx": None, "holding_bars": None,
        "rejection_reason": reason, "eligible": False,
    }


def walk_hypothesis(frames: dict[str, Any], profile: dict[str, Any],
                    bars: int, hypothesis: str, timing: bool = True
                    ) -> list[dict[str, Any]]:
    """One independent pass over entry bars for ONE hypothesis. timing=False
    runs the regime-direction baseline variant (see h1_evaluate/h2_evaluate).
    """
    if hypothesis not in _EVALUATORS:
        raise ValueError(f"unknown hypothesis: {hypothesis!r}")
    evaluator = _EVALUATORS[hypothesis]

    entry_tf = profile["entry"]
    entry_df = frames[entry_tf].df
    n = len(entry_df)
    hold_bars = max_hold_bars(profile)
    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100
    close_ms = {tf: _close_ms(f) for tf, f in frames.items()}
    cache = _IndicatorCache()
    start = walk_start(n, bars)

    records: list[dict[str, Any]] = []
    for i in range(start, n - 1):
        ctx, na_reason = _bar_context(entry_df, i, frames, profile, close_ms,
                                      cache)
        if ctx is None:
            continue  # pure data-availability gap, not a hypothesis outcome
        kind, payload = evaluator(ctx, entry_df, i, cost_pct, timing)
        if kind == "reject":
            records.append(_record_reject(hypothesis, i, ctx, payload))
            continue
        direction, pos = payload["direction"], payload["pos"]
        risk, rr, cost_r = payload["risk"], payload["rr"], payload["cost_r"]
        outcome = resolve(entry_df, i, direction, pos, hold_bars)
        gross_r = outcome["r"]
        net_r = round(gross_r - cost_r, 2) if gross_r is not None else None
        exit_idx = outcome.get("exit_idx")
        records.append({
            "hypothesis": hypothesis, "idx": i, "timestamp": _iso(ctx["open_time"]),
            "direction": direction, "entry": round(pos["entry_price"], 6),
            "stop": round(pos["stop_loss"], 6), "target": round(pos["target_2"], 6),
            "initial_risk": round(risk, 6), "rr": round(rr, 4),
            "regime": ctx["regime_current"],
            "volatility_percentile": ctx["volatility_atr_percentile"],
            "cost_r": round(cost_r, 4), "gross_r": gross_r, "net_r": net_r,
            "outcome": outcome["outcome"], "exit_idx": exit_idx,
            "holding_bars": (exit_idx - i) if exit_idx is not None else None,
            "rejection_reason": None, "eligible": True,
        })
    return records


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def current_swing_baseline(frames: dict[str, Any], profile: dict[str, Any],
                           bars: int) -> dict[str, Any]:
    """Unmodified tools.deep_backtest.deep_walk — the CURRENT swing setups,
    untouched, for comparison only."""
    return deep_backtest.deep_walk(frames, profile, bars)


def regime_direction_baseline(frames: dict[str, Any], profile: dict[str, Any],
                              bars: int, hypothesis: str
                              ) -> list[dict[str, Any]]:
    """Same regime/volatility/RR/cost gates and same stop/target
    construction as the hypothesis, but with the pullback/extreme TIMING
    condition removed — isolates whether the specific entry timing (not
    just being in the right regime) adds anything."""
    return walk_hypothesis(frames, profile, bars, hypothesis, timing=False)


def random_direction_baseline(records: list[dict[str, Any]], entry_df,
                              hold_bars: int, cost_pct: float,
                              seed: int = RANDOM_SEED) -> list[dict[str, Any]]:
    """For every ELIGIBLE trade the hypothesis actually took, resolve an
    otherwise-identical trade (same idx, same absolute stop/target
    distances) with an independently random direction. Deterministic given
    `seed`."""
    rng = np.random.RandomState(seed)
    out = []
    taken = [r for r in records if r["eligible"]]
    directions = rng.choice(["long", "short"], size=len(taken))
    for rec, direction in zip(taken, directions):
        price = rec["entry"]
        stop_dist = abs(rec["entry"] - rec["stop"])
        target_dist = abs(rec["target"] - rec["entry"])
        sign = 1 if direction == "long" else -1
        stop = price - sign * stop_dist
        target = price + sign * target_dist
        pos = {"entry_price": price, "stop_loss": stop,
              "target_1": target, "target_2": target}
        outcome = resolve(entry_df, rec["idx"], direction, pos, hold_bars)
        gross_r = outcome["r"]
        risk = abs(price - stop)
        cost_r = _cost_r(cost_pct, price, risk)
        net_r = round(gross_r - cost_r, 2) if gross_r is not None else None
        exit_idx = outcome.get("exit_idx")
        out.append({
            "hypothesis": rec["hypothesis"], "idx": rec["idx"],
            "timestamp": rec["timestamp"], "direction": direction,
            "entry": price, "stop": round(stop, 6), "target": round(target, 6),
            "initial_risk": round(risk, 6), "rr": None, "regime": rec["regime"],
            "volatility_percentile": rec["volatility_percentile"],
            "cost_r": round(cost_r, 4), "gross_r": gross_r, "net_r": net_r,
            "outcome": outcome["outcome"], "exit_idx": exit_idx,
            "holding_bars": (exit_idx - rec["idx"]) if exit_idx is not None else None,
            "rejection_reason": None, "eligible": True,
        })
    return out


# ---------------------------------------------------------------------------
# Statistics + report assembly.
# ---------------------------------------------------------------------------

def _max_losing_streak(trades: list[dict[str, Any]]) -> int | None:
    if not trades:
        return None
    streak = worst = 0
    for r in sorted(trades, key=lambda r: r["idx"]):
        if r["net_r"] < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    considered = len(records)
    eligible = [r for r in records if r["eligible"]]
    # "unresolved" (resolve() ran out of future bars, r=None) is excluded
    # from all R aggregates and from n itself — same convention as
    # tools.deep_backtest.deep_walk's own resolved/unresolved split.
    unresolved_count = sum(1 for r in eligible if r["outcome"] == "unresolved")
    taken = [r for r in eligible if r["outcome"] != "unresolved"]
    n = len(taken)
    rejection_counts: dict[str, int] = defaultdict(int)
    for r in records:
        if not r["eligible"]:
            rejection_counts[r["rejection_reason"]] += 1
    if n == 0:
        return {
            "considered": considered, "n": 0, "unresolved": unresolved_count,
            "coverage": 0.0,
            "gross_expectancy_r": None, "net_expectancy_r": None,
            "median_r": None, "win_rate": None, "timeout_share": None,
            "sum_r": None, "avg_cost_r": None, "max_losing_streak": None,
            "rejection_counts": dict(rejection_counts),
        }
    net_rs = [r["net_r"] for r in taken]
    gross_rs = [r["gross_r"] for r in taken]
    wins = sum(1 for r in taken if r["outcome"] == "win")
    timeouts = sum(1 for r in taken if r["outcome"] == "timeout")
    return {
        "considered": considered, "n": n, "unresolved": unresolved_count,
        "coverage": round(n / considered, 4) if considered else None,
        "gross_expectancy_r": round(float(np.mean(gross_rs)), 4),
        "net_expectancy_r": round(float(np.mean(net_rs)), 4),
        "median_r": round(float(np.median(net_rs)), 4),
        "win_rate": round(wins / n, 4),
        "timeout_share": round(timeouts / n, 4),
        "sum_r": round(float(np.sum(net_rs)), 4),
        "avg_cost_r": round(float(np.mean([r["cost_r"] for r in taken])), 4),
        "max_losing_streak": _max_losing_streak(taken),
        "rejection_counts": dict(rejection_counts),
    }


def run_simulation(dataset: str, exchange: str, symbol: str,
                   profile_name: str, bars: int,
                   max_gap_ratio: float = 0.001,
                   allow_estimated_cvd: bool = False) -> dict[str, Any]:
    frames, profile, _table, cvd_method = prepare(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)
    entry_df = frames[profile["entry"]].df
    hold_bars = max_hold_bars(profile)
    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100

    baseline_walk = current_swing_baseline(frames, profile, bars)
    baseline_resolved = [s for s in baseline_walk["setups"]
                        if s["outcome"] != "unresolved"]

    result: dict[str, Any] = {
        "stage": STAGE, "kind": "swing_hypothesis_simulation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name, "exchange": exchange, "symbol": symbol,
        "bars_requested": bars, "cvd_method": cvd_method,
        "frozen_constants": {
            "structural_window_bars": STRUCTURAL_WINDOW_BARS,
            "H1": {"stop_atr_buffer": H1_STOP_ATR_BUFFER,
                  "min_rr": H1_MIN_RR,
                  "max_volatility_percentile": H1_MAX_VOLATILITY_PERCENTILE,
                  "max_cost_r": H1_MAX_COST_R},
            "H2": {"eq_pos_extreme": H2_EQ_POS_EXTREME,
                  "volatility_band": list(H2_VOLATILITY_BAND),
                  "stop_atr_buffer": H2_STOP_ATR_BUFFER,
                  "max_cost_r": H2_MAX_COST_R},
        },
        "current_swing_baseline": {
            "n": len(baseline_resolved),
            "net_expectancy_r": (round(float(np.mean(
                [s["r"] for s in baseline_resolved])), 4)
                if baseline_resolved else None),
        },
        "limitations": list(LIMITATIONS),
    }

    for hyp in HYPOTHESES:
        full = walk_hypothesis(frames, profile, bars, hyp, timing=True)
        regime_only = regime_direction_baseline(frames, profile, bars, hyp)
        random_dir = random_direction_baseline(full, entry_df, hold_bars,
                                              cost_pct)
        result[hyp] = {
            "trades": full,
            "summary": summarize(full),
            "baselines": {
                "regime_direction": summarize(regime_only),
                "random_direction": summarize(random_dir),
            },
        }
    return result


# ---------------------------------------------------------------------------
# Formatting + CLI.
# ---------------------------------------------------------------------------

def format_hypothesis_report(hyp: str, report: dict[str, Any]) -> str:
    h = report[hyp]
    s = h["summary"]
    lines = [
        f"{hyp} — swing hypothesis simulation ({report['stage']}) — "
        f"{report['exchange']} {report['symbol']} profile={report['profile']}",
        f"  generated_at : {report['generated_at_utc']}",
        f"  considered   : {s['considered']}  taken : {s['n']} "
        f"coverage={s['coverage']}",
        "",
        "frozen constants:",
        f"  {json.dumps(report['frozen_constants'].get(hyp, {}), indent=2)}"
        if hyp in report["frozen_constants"] else
        f"  structural_window_bars={report['frozen_constants']['structural_window_bars']}",
        "",
        f"summary: gross={s['gross_expectancy_r']} net={s['net_expectancy_r']} "
        f"median={s['median_r']} win={s['win_rate']} timeout={s['timeout_share']} "
        f"sum_r={s['sum_r']} avg_cost_r={s['avg_cost_r']} "
        f"max_losing_streak={s['max_losing_streak']}",
        "",
        "rejection counts:",
    ]
    for reason, cnt in sorted(s["rejection_counts"].items(),
                              key=lambda kv: -kv[1]):
        lines.append(f"  {reason:28s} {cnt}")
    lines.append("")
    lines.append("baselines:")
    for name, bl in h["baselines"].items():
        lines.append(f"  {name:18s} n={bl['n']:>5} net={bl['net_expectancy_r']} "
                     f"win={bl['win_rate']}")
    lines.append("")
    lines.append(f"current_swing_baseline: n={report['current_swing_baseline']['n']} "
                f"net={report['current_swing_baseline']['net_expectancy_r']}")
    lines.append("")
    lines.append("limitations:")
    for lim in report["limitations"]:
        lines.append(f"  - {lim}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C2.2b: offline swing hypothesis simulator (H1/H2, "
                    "read-only)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exchange", required=True, choices=EXCHANGES)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bars", type=int, required=True)
    parser.add_argument("--max-gap-ratio", type=float, default=0.001)
    parser.add_argument("--allow-estimated-cvd", action="store_true")
    parser.add_argument("--outdir", default="reports/c22")
    args = parser.parse_args(argv)

    try:
        report = run_simulation(args.dataset, args.exchange, args.symbol,
                               args.profile, args.bars, args.max_gap_ratio,
                               args.allow_estimated_cvd)
    except DeepBacktestError as exc:
        import sys
        print(f"swing_hypothesis_simulator: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    names = {"H1": "h1_trend_pullback", "H2": "h2_range_reversion"}
    for hyp, fname in names.items():
        payload = {k: v for k, v in report.items() if k not in HYPOTHESES}
        payload[hyp] = report[hyp]
        with open(os.path.join(args.outdir, f"{fname}.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
        with open(os.path.join(args.outdir, f"{fname}.txt"), "w") as f:
            f.write(format_hypothesis_report(hyp, report))
        print(format_hypothesis_report(hyp, report))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
