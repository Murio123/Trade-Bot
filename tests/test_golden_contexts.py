"""Golden regression: legacy-каскад + адаптеры на замороженных сценариях.

Четыре синтетических рынка (тренд вверх/вниз, флэт, высокая волатильность)
прогоняются через настоящий run_cascade и адаптеры контрактов; итоговый
снапшот (контекст + решение + сырой legacy-результат) сравнивается с
закоммиченным JSON. На этапах рефакторинга 2-7 эти снапшоты обязаны
оставаться неизменными — это страховка "production-поведение не тронуто".

Обновление эталонов (ТОЛЬКО при осознанном изменении поведения):
    GOLDEN_UPDATE=1 pytest tests/test_golden_contexts.py

Времена свечей привязаны к реальному "сейчас" (иначе live-гейт stale_data
блокирует всё), поэтому все ISO-времена и сессионная статистика в снапшоте
маскируются; цены и решения зависят только от seed.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from contracts import UnifiedMarketContext, decision_from_legacy, to_jsonable
from pipeline import run_cascade
from signal_engine.profiles import get_profile
from tests.synthetic_market import build_ctx

GOLDEN_DIR = Path(__file__).parent / "golden"

# drift/vol — в долях цены в сутки (BTC-масштаб).
SCENARIOS = {
    "uptrend": dict(seed=7, drift=0.004, vol=0.015),
    "downtrend": dict(seed=11, drift=-0.004, vol=0.015),
    "range": dict(seed=13, drift=0.0, vol=0.008),
    "high_vol": dict(seed=17, drift=0.0, vol=0.035),
    # Проходит весь каскад до WAIT (journal): единственный сценарий с
    # полным сценарием сделки (stop/TP/RR) — покрывает глубокие стадии.
    "downtrend_signal": dict(seed=1, drift=-0.006, vol=0.015),
}

_ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
# Поля, зависящие от реального времени запуска (не от seed).
_TIME_DEPENDENT_KEYS = {"sessions", "session_best", "best_session"}


def _normalize(obj):
    """Маскирует времена/сессии и огрубляет float — детерминизм снапшота."""
    if isinstance(obj, dict):
        return {k: ("<TIME-DEPENDENT>" if k in _TIME_DEPENDENT_KEYS
                    else _normalize(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, str) and _ISO_TS.match(obj):
        return "<TS>"
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _snapshot(name: str) -> dict:
    params = SCENARIOS[name]
    profile = get_profile("swing")
    ctx = build_ctx("swing", **params)
    result = asyncio.run(run_cascade(ctx, delivered_today=[], last_signal=None,
                                     interpret=False, profile_name="swing"))
    context = UnifiedMarketContext.from_legacy(ctx, profile)
    decision = decision_from_legacy(result, ctx, profile)
    legacy = {k: v for k, v in result.items() if k != "df_signal"}
    return _normalize({
        "scenario": {"name": name, **params},
        "unified_context": to_jsonable(context),
        "final_decision": to_jsonable(decision),
        "legacy_result": to_jsonable(legacy),
    })


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden_scenario(name):
    snap = _snapshot(name)
    path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get("GOLDEN_UPDATE"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=1,
                                   sort_keys=True) + "\n", encoding="utf-8")
        return
    assert path.exists(), (
        f"нет эталона {path.name}: сгенерируй его командой "
        f"GOLDEN_UPDATE=1 pytest {__file__}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert snap == expected, (
        f"golden-снапшот '{name}' разошёлся: поведение legacy-каскада или "
        f"адаптеров изменилось. Если изменение ОСОЗНАННОЕ — обнови эталон "
        f"через GOLDEN_UPDATE=1.")


def test_scenarios_are_not_all_blocked_the_same_way():
    """Санити: сценарии дают разные исходы (иначе golden ничего не ловит)."""
    outcomes = set()
    for name in SCENARIOS:
        snap = _snapshot(name)
        legacy = snap["legacy_result"]
        outcomes.add((legacy.get("status"), legacy.get("blocked_at")))
    assert len(outcomes) >= 2, f"все сценарии дали одно и то же: {outcomes}"
