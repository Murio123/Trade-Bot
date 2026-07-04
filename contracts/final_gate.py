"""Shared final-gate слой (этап 3): read-only shadow / compatibility layer.

Единственная роль на этом этапе — читать готовый legacy-`result`
(`pipeline.run_cascade`) и представлять его финальное решение через уже
существующие контракты (`FinalDecision`, `GateCheck`). Слой НЕ принимает
самостоятельных торговых решений и НЕ трогает production-поведение: вся
логика решения делегируется в `contracts.legacy.decision_from_legacy`
(обёртка, а не перенос кода — вывод байт-идентичен, golden не меняются).

Дополнительно слой несёт `GATE_REGISTRY` — канонический реестр всех гейтов
каскада (единый источник правды об их порядке/источнике/семантике) и
`compare_legacy_and_final_decision` — независимую проверку parity между
legacy-`result` и построенным `FinalDecision`.

Границы этапа 3 (сознательно):
- no_trade-bundle остаётся ОДНИМ `GateCheck("no_trade", detail=...)`; его
  под-гейты (RR / expected_move / confidence / tf_conflict / position /
  invalid_tp2 / invalidation) только ДОКУМЕНТИРОВАНЫ в реестре
  (`documented_only=True`), но не разворачиваются в отдельные GateCheck;
- candle_dedup живёт в scheduler ДО каскада и в `result` не присутствует —
  он тоже только документируется и НЕ синтезируется из `result`;
- гейты, зависящие от БД (cooldown / daily_limit / position_conflict),
  отражаются исключительно из готового `result`, без новых запросов в БД.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts import Decision, FinalDecision, GateCheck, decision_from_legacy


# ---------------------------------------------------------------------------
# Канонический реестр гейтов каскада
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateSpec:
    """Описание одного гейта каскада (единый источник правды).

    reachable_from_result — гейт можно представить как GateCheck прямо из
    legacy-`result` (этап 3). documented_only — гейт существует в каскаде, но
    на этапе 3 в GateCheck НЕ разворачивается (под-гейт no_trade или живёт вне
    `result`, как candle_dedup).
    """
    name: str
    stage: str                 # ключ blocked_at / раздел каскада
    source: str                # модуль, где живёт логика
    blocking: bool             # блокирует ли исход (иначе — модификатор)
    db_dependent: bool         # вход гейта берётся из БД
    outcome: Decision          # к какому решению ведёт срабатывание
    reachable_from_result: bool
    documented_only: bool = False
    note: str = ""


# Порядок соответствует фактическому проходу pipeline.run_cascade. Под-гейты
# no_trade перечислены сразу после агрегирующего "no_trade"; candle_dedup —
# первым, т.к. в реальном потоке он предшествует каскаду (scheduler).
GATE_REGISTRY: tuple[GateSpec, ...] = (
    GateSpec("candle_dedup", "candle_dedup", "scheduler.py", True, True,
             Decision.NO_TRADE, reachable_from_result=False,
             documented_only=True,
             note="db.forecast_exists до run_cascade; в result отсутствует"),
    GateSpec("stale_data", "stale_data", "signal_engine/vetoes.py", True, False,
             Decision.NO_TRADE, reachable_from_result=True,
             note="hard data-quality gate до скоринга"),
    GateSpec("abnormal_volatility", "abnormal_volatility",
             "signal_engine/vetoes.py", True, False, Decision.NO_TRADE,
             reachable_from_result=True,
             note="hard data-quality gate до скоринга"),
    GateSpec("direction_conflict", "direction_conflict", "pipeline.py", True,
             False, Decision.NO_TRADE, reachable_from_result=True,
             note="long_total == short_total > 0"),
    GateSpec("htf_filter", "htf_filter", "signal_engine/htf_filter.py", True,
             False, Decision.NO_TRADE, reachable_from_result=True),
    GateSpec("diversity", "diversity", "signal_engine/schema.py", True, False,
             Decision.NO_TRADE, reachable_from_result=True,
             note="has_diverse_confirmation"),
    GateSpec("below_threshold", "below_threshold", "pipeline.py", True, False,
             Decision.NO_TRADE, reachable_from_result=True,
             note="total < SCORE_JOURNAL_MIN (и tie 0/0)"),
    GateSpec("dead_zone", "dead_zone", "signal_engine/vetoes.py", True, False,
             Decision.NO_TRADE, reachable_from_result=True),
    GateSpec("crowded_funding", "crowded_funding", "signal_engine/vetoes.py",
             True, False, Decision.NO_TRADE, reachable_from_result=True),
    GateSpec("wait_for_sweep", "wait_for_sweep",
             "signal_engine/conflict_resolver.py", True, False, Decision.WAIT,
             reachable_from_result=True),
    GateSpec("no_trade", "no_trade", "signal_engine/no_trade_gate.py", True,
             False, Decision.NO_TRADE, reachable_from_result=True,
             note="агрегат обязательных условий; сохранён свёрнутым (этап 3)"),
    # --- под-гейты no_trade: только документация, НЕ отдельные GateCheck ---
    GateSpec("rr", "no_trade", "signal_engine/no_trade_gate.py", True, False,
             Decision.NO_TRADE, reachable_from_result=False,
             documented_only=True, note="bad_risk_reward — под-гейт no_trade"),
    GateSpec("expected_move", "no_trade", "signal_engine/no_trade_gate.py",
             True, False, Decision.NO_TRADE, reachable_from_result=False,
             documented_only=True,
             note="insufficient_expected_move — под-гейт no_trade"),
    GateSpec("confidence", "no_trade", "signal_engine/no_trade_gate.py", True,
             False, Decision.NO_TRADE, reachable_from_result=False,
             documented_only=True, note="low_confidence — под-гейт no_trade"),
    GateSpec("tf_conflict", "no_trade", "signal_engine/no_trade_gate.py", True,
             False, Decision.NO_TRADE, reachable_from_result=False,
             documented_only=True, note="под-гейт no_trade"),
    GateSpec("position_conflict", "no_trade", "signal_engine/no_trade_gate.py",
             True, True, Decision.NO_TRADE, reachable_from_result=False,
             documented_only=True,
             note="db.open_trades — под-гейт no_trade, DB-зависимый"),
    GateSpec("invalid_tp2", "no_trade", "signal_engine/no_trade_gate.py", True,
             False, Decision.NO_TRADE, reachable_from_result=False,
             documented_only=True, note="под-гейт no_trade"),
    GateSpec("missing_invalidation", "no_trade",
             "signal_engine/no_trade_gate.py", True, False, Decision.NO_TRADE,
             reachable_from_result=False, documented_only=True,
             note="под-гейт no_trade"),
    # --- пост-скоринговые стадии ---
    GateSpec("cooldown", "cooldown", "signal_engine/cooldown.py", True, True,
             Decision.WAIT, reachable_from_result=True,
             note="db.last_delivered_signal; статус cooldown -> WAIT"),
    GateSpec("daily_limit", "daily_limit", "signal_engine/daily_limiter.py",
             True, True, Decision.WAIT, reachable_from_result=True,
             note="db.signals_today; over-limit -> journal (WAIT)"),
    GateSpec("alert_threshold", "alert_threshold", "pipeline.py", True, False,
             Decision.NO_TRADE, reachable_from_result=True,
             note="total < SCORE_ALERT_MIN у journal/ignored результата"),
    GateSpec("freshness", "freshness", "pipeline.py", True, False,
             Decision.WAIT, reachable_from_result=True,
             note="downgrade ENTER -> WAIT при freshness > max_decision_delay"),
)

# Быстрый доступ по имени.
GATE_BY_NAME: dict[str, GateSpec] = {g.name: g for g in GATE_REGISTRY}

# Имена, которые МОГУТ появиться как GateCheck из legacy result (этап 3).
_RESULT_GATE_NAMES: frozenset[str] = frozenset(
    g.name for g in GATE_REGISTRY if g.reachable_from_result)


# ---------------------------------------------------------------------------
# Обёртки над legacy-адаптером (делегирование, вывод идентичен)
# ---------------------------------------------------------------------------

def final_decision_from_legacy_result(result: dict[str, Any],
                                      ctx: dict[str, Any],
                                      profile: dict[str, Any]) -> FinalDecision:
    """FinalDecision из legacy-`result`. Тонкая обёртка над
    contracts.legacy.decision_from_legacy — самостоятельных решений не
    принимает, вывод байт-идентичен адаптеру."""
    return decision_from_legacy(result, ctx, profile)


def gate_checks_from_legacy_result(result: dict[str, Any],
                                   ctx: dict[str, Any],
                                   profile: dict[str, Any]) -> list[GateCheck]:
    """GateCheck[] из legacy-`result` (== FinalDecision.gates). candle_dedup и
    под-гейты no_trade сюда НЕ синтезируются — только то, что реально несёт
    result."""
    return list(final_decision_from_legacy_result(result, ctx, profile).gates)


# ---------------------------------------------------------------------------
# Сводка и проверка parity
# ---------------------------------------------------------------------------

def summarize_gate_checks(gates: list[GateCheck]) -> dict[str, Any]:
    """Детерминированная сводка трейса гейтов (чистая функция)."""
    failed = [g for g in gates if not g.passed]
    first_failed = failed[0].name if failed else None
    return {
        "total": len(gates),
        "passed": len(gates) - len(failed),
        "failed": len(failed),
        "first_failed": first_failed,
        "blocked_at": first_failed,
        "failed_names": [g.name for g in failed],
        "passed_names": [g.name for g in gates if g.passed],
    }


@dataclass(frozen=True)
class ParityReport:
    """Итог сравнения legacy-`result` и построенного FinalDecision."""
    ok: bool
    diffs: list[str] = field(default_factory=list)


def _expected_decision(status: str | None, direction: str | None) -> Decision:
    """Независимая (не через _decision_status) переформулировка маппинга
    legacy status -> Decision, зеркальная STATUS_MAP каскада."""
    if status == "alert":
        return Decision.ENTER_LONG if direction == "long" else Decision.ENTER_SHORT
    if status in ("journal", "cooldown"):
        return Decision.WAIT
    return Decision.NO_TRADE


def compare_legacy_and_final_decision(result: dict[str, Any],
                                      final: FinalDecision) -> ParityReport:
    """Доказать legacy parity: FinalDecision воспроизводит legacy-`result` без
    потери решения, направления, точки блокировки и причин.

    Проверяется независимо от того, как FinalDecision был построен, — это
    контроль над обёрткой, а не её эхо."""
    diffs: list[str] = []
    status = result.get("status")
    direction = result.get("direction") or result.get("candidate_direction")

    # 1. Решение.
    expected = _expected_decision(status, direction)
    if final.decision is not expected:
        diffs.append(
            f"decision: ожидалось {expected} для status={status!r}/"
            f"direction={direction!r}, получено {final.decision}")

    # 2. Направление.
    if final.direction != direction:
        diffs.append(f"direction: legacy={direction!r}, final={final.direction!r}")

    # 3. Точка блокировки: имя первого непройденного гейта == blocked_at.
    summary = summarize_gate_checks(final.gates)
    if status == "blocked":
        blocked_at = result.get("blocked_at")
        if summary["first_failed"] != blocked_at:
            diffs.append(
                f"blocked_at: legacy={blocked_at!r}, "
                f"first_failed_gate={summary['first_failed']!r}")
    elif status == "alert":
        if summary["failed"]:
            diffs.append(
                f"alert не должен иметь непройденных гейтов, "
                f"есть: {summary['failed_names']}")
    else:  # journal / cooldown / ignored — ровно один финальный стоп
        if summary["failed"] != 1:
            diffs.append(
                f"status={status!r}: ожидался ровно 1 непройденный гейт, "
                f"есть {summary['failed']}: {summary['failed_names']}")

    # 4. Имена гейтов принадлежат каноническому реестру (result-достижимые).
    unknown = [g.name for g in final.gates if g.name not in _RESULT_GATE_NAMES]
    if unknown:
        diffs.append(f"неизвестные гейты вне реестра: {unknown}")

    # 5. Причины не потеряны.
    legacy_reasons = list(result.get("reasons") or [])[:3]
    if final.reasons != legacy_reasons:
        diffs.append(
            f"reasons: legacy[:3]={legacy_reasons}, final={final.reasons}")

    # 6. no_trade_reasons сохранены в detail свёрнутого гейта no_trade.
    if status == "blocked" and result.get("blocked_at") == "no_trade":
        nt = result.get("no_trade_reasons") or []
        no_trade_gate = next(
            (g for g in final.gates if g.name == "no_trade" and not g.passed),
            None)
        if no_trade_gate is None:
            diffs.append("no_trade: непройденный GateCheck 'no_trade' отсутствует")
        elif nt and not all(r in (no_trade_gate.detail or "") for r in nt):
            diffs.append(
                f"no_trade_reasons потеряны в detail: {nt} vs "
                f"{no_trade_gate.detail!r}")

    return ParityReport(ok=not diffs, diffs=diffs)
