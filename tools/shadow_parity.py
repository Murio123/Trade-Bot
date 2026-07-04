"""Stage 4: offline shadow parity runner (Option B, golden-only).

Read-only проверка. Runner НЕ принимает торговых решений и НЕ трогает
production-поток: для каждого замороженного golden-сценария он прогоняет
настоящий ``pipeline.run_cascade``, строит ``FinalDecision`` через
``contracts.final_gate`` и сверяет его с legacy-``result`` независимой
проверкой parity (``compare_legacy_and_final_decision``).

Данные — только детерминированные synthetic golden-сценарии
(``tests.test_golden_contexts.SCENARIOS``): offline, без сети, воспроизводимо.
Сигналы / Telegram / scheduler / БД не задействованы.

Живёт вне guard-списка (см. tools/__init__), поэтому импорт contracts тут
легален; в runtime модуль никем не импортируется.

Ручной прогон (exit-code != 0 при любом расхождении — пригодно для CI):

    .venv/bin/python -m tools.shadow_parity
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

from contracts.final_gate import (compare_legacy_and_final_decision,
                                   final_decision_from_legacy_result)
from pipeline import run_cascade
from signal_engine.profiles import get_profile
from tests.synthetic_market import build_ctx
from tests.test_golden_contexts import SCENARIOS


@dataclass(frozen=True)
class ScenarioParity:
    """Итог shadow-сравнения по одному golden-сценарию."""
    name: str
    status: str | None
    blocked_at: str | None
    ok: bool
    diffs: list[str] = field(default_factory=list)


def run_scenario_parity(name: str, params: dict[str, Any],
                        profile_name: str = "swing") -> ScenarioParity:
    """Прогнать один сценарий: run_cascade -> FinalDecision -> parity.

    interpret=False: AI-комментарий на решение не влияет и для parity не нужен.
    """
    profile = get_profile(profile_name)
    ctx = build_ctx(profile_name, **params)
    result = asyncio.run(run_cascade(ctx, delivered_today=[], last_signal=None,
                                     interpret=False, profile_name=profile_name))
    final = final_decision_from_legacy_result(result, ctx, profile)
    report = compare_legacy_and_final_decision(result, final)
    return ScenarioParity(
        name=name,
        status=result.get("status"),
        blocked_at=result.get("blocked_at"),
        ok=report.ok,
        diffs=list(report.diffs))


def run_all(scenarios: dict[str, dict[str, Any]] | None = None,
            profile_name: str = "swing") -> list[ScenarioParity]:
    """Прогнать все сценарии (по умолчанию — golden SCENARIOS)."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    return [run_scenario_parity(name, scenarios[name], profile_name)
            for name in sorted(scenarios)]


def summarize(results: list[ScenarioParity]) -> dict[str, Any]:
    """Детерминированная сводка результатов (чистая функция)."""
    mismatched = [r for r in results if not r.ok]
    return {
        "total": len(results),
        "ok": len(results) - len(mismatched),
        "mismatched": len(mismatched),
        "mismatched_names": [r.name for r in mismatched],
    }


def format_report(results: list[ScenarioParity]) -> str:
    """Человекочитаемый отчёт: по строке на сценарий + итоговая сводка."""
    lines: list[str] = []
    for r in results:
        mark = "OK      " if r.ok else "MISMATCH"
        outcome = r.blocked_at or r.status or "?"
        lines.append(f"[{mark}] {r.name:<18} status={r.status} outcome={outcome}")
        for d in r.diffs:
            lines.append(f"           - {d}")
    s = summarize(results)
    lines.append("")
    tail = f": {s['mismatched_names']}" if s["mismatched"] else ""
    lines.append(f"parity: {s['ok']}/{s['total']} ok, "
                 f"{s['mismatched']} mismatched{tail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Прогнать golden-parity, напечатать отчёт, вернуть код выхода."""
    results = run_all()
    print(format_report(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
