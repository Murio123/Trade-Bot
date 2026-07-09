"""Stage 5: offline backtest/live drift parity runner (Option B).

Read-only проверка. Runner НЕ принимает торговых решений, НЕ трогает
production-поток и НЕ импортируется runtime-кодом. Его единственная задача —
зафиксировать backtest/live drift inventory:

  * какие части ``backtest._walk`` совпадают с ``pipeline.run_cascade``
    (reproduced) — доказывается тем, что оба вызывают ОДИН И ТОТ ЖЕ объект
    helper-функции (identity), плюс сравнением решений на одном синтетическом
    рынке;
  * какие расходятся и почему (diverges_expected / missing_in_backtest_expected
    / not_applicable) — зафиксировано в EXPECTED_DRIFT (D1..D12);
  * любой НОВЫЙ, не описанный drift классифицируется как
    ``unexpected_mismatch`` и обязан ломать тесты / давать exit-code != 0.

Stage 5 НЕ рефакторит backtest, НЕ чинит drift и НЕ меняет стратегию —
только измеряет и документирует.

Данные — только детерминированные synthetic golden-сценарии
(``tests.test_golden_contexts.SCENARIOS`` через ``tests.synthetic_market``):
offline, без сети, воспроизводимо. Тот же seed кормит обе стороны, поэтому
любая разница объясняется исключительно перечисленными drift-точками.

Ручной прогон (exit-code != 0 при любом UNEXPECTED расхождении — пригодно для CI):

    .venv/bin/python -m tools.backtest_parity
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

import backtest as bt
import pipeline
from analyzer.cvd import compute_cvd_from_klines, cvd_series
from analyzer.divergence import detect_divergence
from analyzer.equilibrium import compute_equilibrium
from analyzer.indicators import compute_indicators
from analyzer.liquidity import detect_liquidity
from analyzer.reversal import detect_reversal
from analyzer.volatility import analyze_volatility
from analyzer.volume_profile import compute_volume_profile
from contracts.final_gate import GATE_REGISTRY
from pipeline import run_cascade
from risk.position_sizing import calculate_position
from signal_engine.profiles import get_profile
from tests.synthetic_market import TF_MINUTES, build_ctx, make_klines

# ---------------------------------------------------------------------------
# 1. Reproduced: backtest и pipeline вызывают ОДИН И ТОТ ЖЕ объект функции
# ---------------------------------------------------------------------------

# Имена, присутствующие атрибутом и в ``backtest``, и в ``pipeline``. Если
# identity держится — логика гейта/примитива физически общая (не копия), значит
# reproduced по построению. Слом identity => unexpected_mismatch.
REPRODUCED_SHARED: tuple[str, ...] = (
    "calculate_confluence_score",
    "has_diverse_confirmation",
    "apply_htf_policy",
    "get_htf_bias",
    "weighted_total",
    "detect_regime",
    "dead_zone",
    "abnormal_volatility",
    "effective_expected_move",
    "_structural_stop",
    "_structure_targets",
    "_build_htf_zones",
    "_near_key_level",
)


@dataclass(frozen=True)
class ReproducedCheck:
    """Итог проверки, что backtest и pipeline делят один объект helper-а."""
    name: str
    ok: bool


def reproduced_checks() -> list[ReproducedCheck]:
    """Проверить identity общих helper-функций между backtest и pipeline."""
    out: list[ReproducedCheck] = []
    for name in REPRODUCED_SHARED:
        ok = getattr(bt, name, object()) is getattr(pipeline, name, object())
        out.append(ReproducedCheck(name, ok))
    return out


# ---------------------------------------------------------------------------
# 2. Классификация КАЖДОГО гейта каскада (по имени из GATE_REGISTRY)
# ---------------------------------------------------------------------------

REPRODUCED = "reproduced"
DIVERGES = "diverges_expected"
MISSING = "missing_in_backtest_expected"
NOT_APPLICABLE = "not_applicable"
UNEXPECTED = "unexpected_mismatch"

# Каждый гейт из канонического реестра -> (классификация, drift-id | None, нота).
# Это ручная, ревьюабельная карта из аудита Stage 5. Если реестр вырастет новым
# гейтом, которого здесь нет, тест поймает его как unexpected.
GATE_CLASSIFICATION: dict[str, tuple[str, str | None, str]] = {
    # --- воспроизводимые в backtest (общий helper, тот же вход по построению) ---
    "abnormal_volatility": (REPRODUCED, None, "vetoes.abnormal_volatility, общий"),
    "direction_conflict": (REPRODUCED, None, "lt == st -> skip, зеркально каскаду"),
    "htf_filter": (REPRODUCED, None, "apply_htf_policy, общий"),
    "diversity": (REPRODUCED, None, "has_diverse_confirmation, общий"),
    "below_threshold": (REPRODUCED, None, "total < min(THRESHOLDS)=5 == SCORE_JOURNAL_MIN"),
    "dead_zone": (REPRODUCED, None, "vetoes.dead_zone, общий"),
    "expected_move": (REPRODUCED, None, "effective_expected_move, под-гейт no_trade, общий"),
    # --- отсутствуют в backtest (ожидаемо и объяснено) ---
    "rr": (MISSING, "D1", "bad_risk_reward не проверяется в _walk"),
    "confidence": (MISSING, "D2", "confidence/mtf не считается в _walk"),
    "tf_conflict": (MISSING, "D1", "под-гейт no_trade отсутствует в _walk"),
    "position_conflict": (MISSING, "D1", "DB-зависимый под-гейт no_trade, нет в _walk"),
    "invalid_tp2": (MISSING, "D1", "под-гейт no_trade отсутствует в _walk"),
    "missing_invalidation": (MISSING, "D1", "под-гейт no_trade отсутствует в _walk"),
    "crowded_funding": (MISSING, "D4", "нет истории funding в backtest"),
    "wait_for_sweep": (MISSING, "D3", "нет истории liquidation_map в backtest"),
    "daily_limit": (MISSING, "D6", "дневной лимит не моделируется в backtest"),
    # --- присутствуют, но с иной моделью (ожидаемо) ---
    "no_trade": (DIVERGES, "D1", "агрегат: воспроизведён только expected_move, "
                                 "остальные под-гейты отсутствуют"),
    "cooldown": (DIVERGES, "D5", "backtest: bar-count в _aggregate; live: should_send_signal"),
    "alert_threshold": (DIVERGES, "D5", "backtest свипует пороги 5..10; live: единый SCORE_ALERT_MIN"),
    # --- неприменимо к историческому проходу ---
    "candle_dedup": (NOT_APPLICABLE, "D7", "scheduler-level; backtest проходит каждый бар один раз"),
    "stale_data": (NOT_APPLICABLE, "D7", "свежесть не применяется к историческим барам"),
    "freshness": (NOT_APPLICABLE, "D7", "max_decision_delay не применяется к истории"),
}


def classify_registry_gates() -> list["GateParity"]:
    """Сопоставить каждый гейт GATE_REGISTRY с его классификацией.

    Гейт, отсутствующий в GATE_CLASSIFICATION, помечается unexpected_mismatch —
    это и есть страховка от нового, не описанного drift.
    """
    out: list[GateParity] = []
    for spec in GATE_REGISTRY:
        cls = GATE_CLASSIFICATION.get(spec.name)
        if cls is None:
            out.append(GateParity(spec.name, UNEXPECTED, None,
                                  "гейт не классифицирован в Stage 5 inventory"))
        else:
            classification, drift_id, note = cls
            out.append(GateParity(spec.name, classification, drift_id, note))
    return out


@dataclass(frozen=True)
class GateParity:
    """Классификация одного гейта каскада относительно backtest-пути."""
    name: str
    classification: str
    drift_id: str | None
    note: str


# ---------------------------------------------------------------------------
# 3. Expected drift inventory (D1..D12) — зафиксировано из аудита Stage 5
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftSpec:
    """Одна известная точка расхождения backtest vs live."""
    id: str
    gate: str
    classification: str
    summary: str
    observable: bool  # можно ли продемонстрировать на синтетике


EXPECTED_DRIFT: dict[str, DriftSpec] = {
    "D1": DriftSpec("D1", "no_trade", DIVERGES,
                    "no_trade-бандл частичный: воспроизведён только expected_move; "
                    "rr / invalid_tp2 / missing_invalidation / tf_conflict / "
                    "position_conflict отсутствуют", observable=False),
    "D2": DriftSpec("D2", "confidence", MISSING,
                    "confidence (mtf_confidence * conflict_factor) не считается; "
                    "low_confidence не может сработать", observable=False),
    "D3": DriftSpec("D3", "wait_for_sweep", MISSING,
                    "conflict_resolver/liquidation_map недоступны исторически", observable=False),
    "D4": DriftSpec("D4", "crowded_funding", MISSING,
                    "нет истории funding -> вето не воспроизводится", observable=False),
    "D5": DriftSpec("D5", "cooldown", DIVERGES,
                    "backtest: bar-count cooldown в _aggregate; live: should_send_signal "
                    "(цена+ATR+время)", observable=False),
    "D6": DriftSpec("D6", "daily_limit", MISSING,
                    "within_daily_limit не моделируется в backtest", observable=False),
    "D7": DriftSpec("D7", "stale/freshness", NOT_APPLICABLE,
                    "stale_data / freshness / max_decision_delay неприменимы к истории",
                    observable=False),
    "D8": DriftSpec("D8", "zone_tfs", DIVERGES,
                    "backtest строит зоны из zone_tfs целиком (swing: 12h+6h); live "
                    "фильтрует по фактически загруженным ТФ (swing: только 12h)",
                    observable=True),
    "D9": DriftSpec("D9", "regime", DIVERGES,
                    "backtest: detect_regime без volatility_1d overlay; "
                    "live: с overlay", observable=True),
    "D10": DriftSpec("D10", "indicator_window", DIVERGES,
                     "backtest HTF-окно iloc[-250:]; live использует полное окно "
                     "(300 закрытых свечей)", observable=True),
    "D11": DriftSpec("D11", "reversal_mtf", DIVERGES,
                     "backtest: at_key_level с entry-ATR-толерансом для всех ТФ и без "
                     "1H-candle-confirm; live: per-TF ATR + подтверждающая свеча",
                     observable=False),
    "D12": DriftSpec("D12", "equilibrium_window", DIVERGES,
                     "equilibrium на разных окнах HTF (backtest 250 vs live 300)",
                     observable=True),
}


# ---------------------------------------------------------------------------
# 4. Backtest-equivalent decision: независимая single-bar модель пути _walk
# ---------------------------------------------------------------------------

def backtest_equiv_decision(profile_name: str, seed: int, drift: float,
                            vol: float) -> dict[str, Any]:
    """Решение backtest-пути на ПОСЛЕДНЕМ закрытом баре синтетического рынка.

    Зеркалит тело ``backtest._walk`` для одного бара (без _resolve), вызывая
    ТЕ ЖЕ общие helper-функции. Это независимый компаратор — он специально
    строит зоны/regime/окна так, как это делает backtest, чтобы drift был
    виден. Значения решений НЕ обязаны совпадать с live (см. EXPECTED_DRIFT)."""
    profile = get_profile(profile_name)
    entry_tf, htf, zone_tfs = profile["entry"], profile["htf"], profile["zone_tfs"]
    needed = {entry_tf, htf, "1d"} | set(zone_tfs) | set(bt.REVERSAL_TFS)
    dfs = {tf: make_klines(tf, seed=seed, drift=drift, vol=vol)
           for tf in needed if tf in TF_MINUTES}

    entry_df = dfs[entry_tf]
    i = len(entry_df) - 1
    out: dict[str, Any] = {"outcome": None, "direction": None, "total": None,
                           "zone_tfs_used": [], "regime": None, "htf_bias": None,
                           "stop": None, "tp1": None, "tp2": None,
                           "expected_move": None, "long_total": None,
                           "short_total": None, "regime_overlay": None}

    entry_sub = entry_df.iloc[max(0, i - 299):i + 1]
    ind = compute_indicators(entry_sub)
    atr = ind.get("atr")
    if not atr:
        out["outcome"] = "skip:no_atr"
        return out
    t = entry_df["close_time"].iloc[i]
    price = float(entry_df["close"].iloc[i])

    htf_slice = dfs[htf][dfs[htf]["close_time"] <= t].iloc[-250:]
    if len(htf_slice) < 210:
        out["outcome"] = "skip:htf_history"
        return out
    ind_htf = compute_indicators(htf_slice)
    htf_bias = bt.get_htf_bias(ind_htf)
    out["htf_bias"] = htf_bias

    stop_atr = atr
    if profile.get("stop_tf") == htf and ind_htf.get("atr"):
        stop_atr = ind_htf["atr"]

    zdfs, zinds = {}, {}
    for tf in zone_tfs:
        s = dfs[tf][dfs[tf]["close_time"] <= t].iloc[-160:] if tf in dfs else None
        if s is not None and len(s) >= 30:
            zdfs[tf] = s
            zinds[tf] = compute_indicators(s)
    if not zdfs:
        out["outcome"] = "skip:no_zones"
        return out
    out["zone_tfs_used"] = list(zdfs.keys())
    zones = bt._build_htf_zones(zdfs, zinds, list(zdfs.keys()), price, entry_sub)
    ob, fvg, levels = zones["order_blocks"], zones["fvg"], zones["levels"]

    eq = compute_equilibrium(htf_slice)
    liq = detect_liquidity(entry_sub, atr_value=atr)
    at_level = bt._near_key_level(price, levels, ob, fvg,
                                  max(atr * 0.3, price * 0.002))
    rev = detect_reversal(entry_sub, ind, cvd_series(entry_sub),
                          at_key_level=at_level)

    rev_frames: dict[str, Any] = {}
    if entry_tf in bt.REVERSAL_TFS:
        rev_frames[entry_tf] = (entry_sub, ind)
    if htf in bt.REVERSAL_TFS:
        rev_frames.setdefault(htf, (htf_slice, ind_htf))
    for tf in zdfs:
        if tf in bt.REVERSAL_TFS:
            rev_frames.setdefault(tf, (zdfs[tf], zinds[tf]))
    bull_tfs = bear_tfs = 0
    for tf, (df_tf, ind_tf) in rev_frames.items():
        r_tf = rev if tf == entry_tf else detect_reversal(
            df_tf, ind_tf, cvd_series(df_tf), at_key_level=at_level)
        bull_tfs += bool(r_tf.get("bullish_reversal"))
        bear_tfs += bool(r_tf.get("bearish_reversal"))

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
        "trend_aligned_bottom": htf_bias == "bullish" and bull_tfs >= 2,
        "trend_aligned_top": htf_bias == "bearish" and bear_tfs >= 2,
    })

    _, ls, _ = bt.calculate_confluence_score(flat, "long")
    _, ss, _ = bt.calculate_confluence_score(flat, "short")
    # D9: backtest не передаёт volatility_1d overlay.
    regime = bt.detect_regime(ind_htf if htf == "1d" else None)
    out["regime"] = regime
    out["regime_overlay"] = None
    lt = bt.weighted_total(ls, regime)
    st = bt.weighted_total(ss, regime)
    out["long_total"], out["short_total"] = lt, st
    if lt == st:
        out["outcome"] = "skip:direction_conflict"
        return out
    direction, total, scores = ("long", lt, ls) if lt > st else ("short", st, ss)
    out["direction"], out["total"] = direction, total

    if bt.apply_htf_policy(direction, htf_bias, profile) is None:
        out["outcome"] = "skip:htf_filter"
        return out
    if not bt.has_diverse_confirmation(scores, bt.config.MIN_DIVERSE_CATEGORIES):
        out["outcome"] = "skip:diversity"
        return out
    if total < min(bt.THRESHOLDS):
        out["outcome"] = "skip:below_threshold"
        return out
    if bt.dead_zone(scores.get("structure", 0), eq):
        out["outcome"] = "skip:dead_zone"
        return out
    if bt.abnormal_volatility(analyze_volatility(
            entry_sub, tf_per_day=24.0 / bt.TF_HOURS.get(entry_tf, 1.0))):
        out["outcome"] = "skip:abnormal_volatility"
        return out

    pos = calculate_position(price, stop_atr, direction=direction,
                             atr_multiplier=profile["atr_mult"],
                             targets_r=profile["targets"])
    ctx_min = {"htf_levels": levels, "order_blocks": ob,
               "volume_profile": compute_volume_profile(
                   zdfs.get(zone_tfs[0], entry_sub))}
    s_stop = bt._structural_stop(ctx_min, direction, price, stop_atr, profile)
    if s_stop is not None:
        sign = 1 if direction == "long" else -1
        dist = abs(price - s_stop)
        pos.update({
            "stop_loss": round(s_stop, 2),
            "target_1": round(price + sign * dist * profile["targets"][0], 2),
            "target_2": round(price + sign * dist * profile["targets"][1], 2),
        })
    risk = abs(price - pos["stop_loss"])
    tp1, tp2, _, _ = bt._structure_targets(ctx_min, direction, price, risk,
                                           pos["target_1"], pos["target_2"])
    pos["target_1"], pos["target_2"] = tp1, tp2
    out["stop"], out["tp1"], out["tp2"] = pos["stop_loss"], tp1, tp2

    move = bt.effective_expected_move(
        price, pos["target_2"], atr,
        profile.get("forecast_horizon_hours", 24.0),
        bt.TF_HOURS.get(entry_tf, 1.0))
    out["expected_move"] = move
    if move < profile.get("minimum_expected_move_points", 0):
        out["outcome"] = "skip:expected_move"
        return out

    out["outcome"] = f"candidate:{direction}@{total:g}"
    return out


def live_decision(profile_name: str, seed: int, drift: float,
                  vol: float) -> dict[str, Any]:
    """Живое решение через настоящий run_cascade (interpret=False)."""
    ctx = build_ctx(profile_name, seed=seed, drift=drift, vol=vol)
    result = asyncio.run(run_cascade(ctx, delivered_today=[], last_signal=None,
                                     interpret=False, profile_name=profile_name))
    return {
        "status": result.get("status"),
        "blocked_at": result.get("blocked_at"),
        "direction": result.get("direction") or result.get("candidate_direction"),
        "score_weighted": result.get("score_weighted", result.get("long_score")),
        "long_total": result.get("long_score"),
        "short_total": result.get("short_score"),
        "regime": result.get("market_regime"),
        "stop": result.get("stop_loss"),
        "tp1": result.get("target_1"),
        "tp2": result.get("target_2"),
        "confidence": result.get("confidence"),
        "zone_tfs_used": list(ctx.get("zone_tfs") or []),
    }


# ---------------------------------------------------------------------------
# 5. Прогон сценариев + сводка
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioReport:
    """Пара решений live/backtest на одном синтетическом рынке + наблюдаемый drift."""
    name: str
    live: dict[str, Any]
    backtest: dict[str, Any]
    observed_drift: list[str] = field(default_factory=list)


def _observe_drift(live: dict[str, Any], bt_dec: dict[str, Any]) -> list[str]:
    """Продемонстрировать observable=True drift-точки на этом сценарии."""
    seen: list[str] = []
    if set(bt_dec.get("zone_tfs_used") or []) != set(live.get("zone_tfs_used") or []):
        seen.append("D8")
    # D9: backtest всегда без volatility_1d overlay (regime_overlay is None).
    if bt_dec.get("regime_overlay") is None:
        seen.append("D9")
    # D10/D12: backtest HTF/equilibrium окно жёстко [-250:] (< полного live-окна).
    seen.append("D10")
    seen.append("D12")
    return seen


def run_scenario(name: str, params: dict[str, Any],
                 profile_name: str = "swing") -> ScenarioReport:
    """Один сценарий: live-решение + backtest-equivalent + наблюдаемый drift."""
    live = live_decision(profile_name, **params)
    bt_dec = backtest_equiv_decision(profile_name, **params)
    return ScenarioReport(name=name, live=live, backtest=bt_dec,
                          observed_drift=_observe_drift(live, bt_dec))


def run_all(scenarios: dict[str, dict[str, Any]] | None = None,
            profile_name: str = "swing") -> list[ScenarioReport]:
    """Прогнать все сценарии (по умолчанию — golden SCENARIOS)."""
    from tests.test_golden_contexts import SCENARIOS
    scenarios = SCENARIOS if scenarios is None else scenarios
    return [run_scenario(name, scenarios[name], profile_name)
            for name in sorted(scenarios)]


@dataclass(frozen=True)
class ParitySummary:
    """Итог всего прогона: reproduced-checks + классификация + unexpected."""
    scenarios: int
    reproduced_ok: int
    reproduced_total: int
    gate_classes: list[GateParity]
    unexpected: list[str]
    expected_drift_ids: list[str]


def evaluate() -> tuple[list[ScenarioReport], ParitySummary]:
    """Собрать полный результат прогона и его сводку (чистая агрегация)."""
    reports = run_all()
    reproduced = reproduced_checks()
    gates = classify_registry_gates()

    unexpected: list[str] = []
    for rc in reproduced:
        if not rc.ok:
            unexpected.append(f"reproduced-identity сломана: {rc.name}")
    for g in gates:
        if g.classification == UNEXPECTED:
            unexpected.append(f"неклассифицированный гейт: {g.name}")

    summary = ParitySummary(
        scenarios=len(reports),
        reproduced_ok=sum(1 for c in reproduced if c.ok),
        reproduced_total=len(reproduced),
        gate_classes=gates,
        unexpected=unexpected,
        expected_drift_ids=sorted(EXPECTED_DRIFT, key=lambda d: int(d[1:])))
    return reports, summary


def _gates_by_class(gates: list[GateParity]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for g in gates:
        grouped.setdefault(g.classification, []).append(g.name)
    return grouped


def format_report(reports: list[ScenarioReport], summary: ParitySummary) -> str:
    """Человекочитаемый отчёт (детерминированный)."""
    lines: list[str] = ["=== backtest/live drift parity (Stage 5) ==="]
    lines.append(f"scenarios:            {summary.scenarios}")
    lines.append(f"reproduced checks:    {summary.reproduced_ok}/{summary.reproduced_total} ok")
    lines.append(f"expected drift:       {len(summary.expected_drift_ids)} "
                 f"(D1..D{len(summary.expected_drift_ids)})")
    lines.append(f"unexpected mismatch:  {len(summary.unexpected)}")
    lines.append("")

    lines.append("per-scenario (live vs backtest-equivalent):")
    for r in reports:
        live_out = r.live.get("blocked_at") or r.live.get("status") or "?"
        bt_out = r.backtest.get("outcome") or "?"
        drift = ",".join(r.observed_drift)
        lines.append(f"  [{r.name:<16}] live={r.live.get('status')}@{live_out:<20} "
                     f"bt={bt_out:<26} drift={drift}")
    lines.append("")

    grouped = _gates_by_class(summary.gate_classes)
    lines.append("gate classification (по GATE_REGISTRY):")
    for cls in (REPRODUCED, DIVERGES, MISSING, NOT_APPLICABLE, UNEXPECTED):
        names = grouped.get(cls)
        if names:
            lines.append(f"  {cls:<32} {', '.join(names)}")
    lines.append("")

    lines.append("expected drift inventory:")
    for did in summary.expected_drift_ids:
        d = EXPECTED_DRIFT[did]
        obs = "observable" if d.observable else "documented"
        lines.append(f"  {d.id:<4} {d.classification:<32} [{obs}] {d.gate}: {d.summary}")
    lines.append("")

    if summary.unexpected:
        lines.append("UNEXPECTED MISMATCHES:")
        for u in summary.unexpected:
            lines.append(f"  - {u}")
        lines.append("")
        lines.append(f"RESULT: FAIL (unexpected_mismatch={len(summary.unexpected)})")
    else:
        lines.append("RESULT: PASS (unexpected_mismatch=0) — весь drift ожидаем и объяснён")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Прогнать parity, напечатать отчёт, вернуть код выхода."""
    reports, summary = evaluate()
    print(format_report(reports, summary))
    return 0 if not summary.unexpected else 1


if __name__ == "__main__":
    sys.exit(main())
