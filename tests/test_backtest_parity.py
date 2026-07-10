"""Stage 5: offline backtest/live drift parity (Option B).

Доказывает: runner фиксирует backtest/live drift inventory (D1..D15),
reproduced-гейты реально делят один helper с pipeline, каждый гейт каскада
классифицирован, любой НОВЫЙ drift ломает тест, а golden-снапшоты и
production-код Stage 5 не трогает. Runner read-only и в runtime не
импортируется.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pandas as pd
import pytest

import tools.backtest_parity as bp
from analyzer.bybit import BybitClient
from analyzer.cvd import compute_cvd_from_klines
from contracts.final_gate import GATE_REGISTRY
from signal_engine.profiles import get_profile
from signal_engine.vetoes import TF_HOURS
from tests.test_golden_contexts import SCENARIOS

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden"

# Замороженная ожидаемая классификация всех drift-точек аудита Stage 5.
EXPECTED_DRIFT_CLASSIFICATION = {
    "D1": bp.DIVERGES,
    "D2": bp.MISSING,
    "D3": bp.MISSING,
    "D4": bp.MISSING,
    "D5": bp.DIVERGES,
    "D6": bp.MISSING,
    "D7": bp.NOT_APPLICABLE,
    "D8": bp.DIVERGES,
    "D9": bp.DIVERGES,
    "D10": bp.DIVERGES,
    "D11": bp.DIVERGES,
    "D12": bp.DIVERGES,
    "D13": bp.DIVERGES,
    "D14": bp.DIVERGES,
    "D15": bp.DIVERGES,
}

# D13..D15 добавлены на C1-precheck: каждая точка не просто описана, а
# демонстрируется offline-тестом ниже.
NEW_DRIFT_CATEGORIES = {
    "D13": "regime",
    "D14": "data_depth",
    "D15": "data_source",
}


# --- runner offline + покрытие сценариев ------------------------------------

def test_runner_runs_offline_over_all_scenarios():
    """Runner прогоняет все golden-сценарии без сети и строит обе стороны."""
    reports = bp.run_all()
    assert {r.name for r in reports} == set(SCENARIOS)
    for r in reports:
        assert r.live.get("status") is not None
        assert r.backtest.get("outcome") is not None


def test_no_unexpected_mismatch_on_clean_run():
    _, summary = bp.evaluate()
    assert summary.unexpected == [], f"unexpected drift: {summary.unexpected}"


# --- reproduced: общий helper с pipeline ------------------------------------

def test_reproduced_identities_hold():
    """backtest и pipeline физически делят один объект каждой общей функции."""
    checks = bp.reproduced_checks()
    bad = [c.name for c in checks if not c.ok]
    assert not bad, f"reproduced-identity сломана для: {bad}"


def test_reproduced_gate_value_parity():
    """Общий helper на одном входе даёт один результат (тот же объект функции)."""
    flat = {"rsi": 45.0, "macd_bullish": True, "price": 100.0}
    assert (bp.bt.calculate_confluence_score(flat, "long")
            == bp.pipeline.calculate_confluence_score(flat, "long"))


# --- классификация каждого гейта реестра ------------------------------------

def test_every_registry_gate_is_classified():
    """Ни один гейт GATE_REGISTRY не остаётся unexpected (страховка от нового drift)."""
    gates = bp.classify_registry_gates()
    assert {g.name for g in gates} == {s.name for s in GATE_REGISTRY}
    unexpected = [g.name for g in gates if g.classification == bp.UNEXPECTED]
    assert not unexpected, f"неклассифицированные гейты: {unexpected}"


def test_reproduced_gates_are_the_expected_set():
    gates = {g.name: g for g in bp.classify_registry_gates()}
    reproduced = {n for n, g in gates.items() if g.classification == bp.REPRODUCED}
    assert reproduced == {
        "abnormal_volatility", "direction_conflict", "htf_filter", "diversity",
        "below_threshold", "dead_zone", "expected_move"}


# --- expected drift inventory D1..D12 ---------------------------------------

def test_expected_drift_inventory_matches_audit():
    assert set(bp.EXPECTED_DRIFT) == set(EXPECTED_DRIFT_CLASSIFICATION)
    for did, cls in EXPECTED_DRIFT_CLASSIFICATION.items():
        assert bp.EXPECTED_DRIFT[did].classification == cls, (
            f"{did}: классификация изменилась")


def test_observable_drift_demonstrated_per_scenario():
    """Каждый сценарий реально демонстрирует observable drift (D8..D12)."""
    for r in bp.run_all():
        assert "D8" in r.observed_drift, f"{r.name}: zone_tfs drift не виден"
        # backtest использует именно zone_tfs целиком, live — отфильтрованный набор.
        assert set(r.backtest["zone_tfs_used"]) != set(r.live["zone_tfs_used"])


# --- C1-precheck: D13..D15 зарегистрированы и наблюдаемы ---------------------

def test_new_drift_points_registered_observable_and_categorised():
    for did, category in NEW_DRIFT_CATEGORIES.items():
        spec = bp.EXPECTED_DRIFT[did]
        assert spec.observable is True, f"{did}: должен быть observable"
        assert spec.category == category, f"{did}: категория изменилась"


def test_d13_description_preserves_empirical_precheck():
    """Эмпирика C1-precheck зафиксирована в описании, а не только в чате."""
    summary = bp.EXPECTED_DRIFT["D13"].summary
    for token in ("1199/1199", "56.2%", "trend_down"):
        assert token in summary, f"D13: потеряно из описания: {token}"


def test_d13_intraday_regime_is_always_range():
    """D13 наблюдаем: у intraday htf != '1d' -> detect_regime(None) -> range.

    Прогоняем backtest-equivalent по ВСЕМ golden-сценариям (включая downtrend и
    high_vol) с профилем intraday: режим обязан быть 'range' на каждом.
    """
    assert get_profile("intraday")["htf"] != "1d"
    assert bp.bt.detect_regime(None) == "range"

    regimes = {name: bp.backtest_equiv_decision("intraday", **params)["regime"]
               for name, params in SCENARIOS.items()}
    assert set(regimes.values()) == {"range"}, (
        f"ожидался константный range на intraday, получено: {regimes}")

    # Контроль: у swing htf='1d', режим берётся с 1D и НЕ обязан быть range.
    assert get_profile("swing")["htf"] == "1d"


def test_d14_aux_frame_depth_cannot_cover_the_walk():
    """D14 наблюдаем: zone-фрейм на limit=500 короче окна, которое walk проходит.

    Чистая арифметика по константам backtest — сети не требует.
    """
    profile = get_profile("swing")
    entry_hours = TF_HOURS[profile["entry"]]
    walk_hours = bp.bt.MAX_BARS * entry_hours

    shallow = {tf: bp.AUX_KLINES_LIMIT * TF_HOURS[tf] for tf in profile["zone_tfs"]}
    deficient = {tf: h for tf, h in shallow.items() if h < walk_hours}
    assert "6h" in deficient, (
        f"ожидалось, что 6h-зона не покрывает {walk_hours}ч прогона; "
        f"покрытие: {shallow}")

    # Бары старше покрытия теряют этот zone-ТФ: len(slice) < 30 -> tf не в zdfs.
    uncovered_hours = walk_hours - shallow["6h"]
    assert uncovered_hours > 0
    assert uncovered_hours / walk_hours > 0.3, (
        "доля непокрытых баров должна быть материальной, а не краевой")


def test_d15_cvd_method_depends_on_taker_buy_base_column():
    """D15 наблюдаем: наличие taker_buy_base переключает метод CVD."""
    n = 60
    base = {"open": [100.0] * n, "high": [101.0] * n,
            "low": [99.0] * n, "close": [100.5] * n, "volume": [10.0] * n}
    bybit_shaped = pd.DataFrame(base)                                  # без колонки
    binance_shaped = pd.DataFrame({**base, "taker_buy_base": [7.0] * n})

    assert compute_cvd_from_klines(bybit_shaped)["method"] == "estimate"
    assert compute_cvd_from_klines(binance_shaped)["method"] == "exact"
    # Разные методы дают разный CVD на одних и тех же свечах.
    assert (compute_cvd_from_klines(bybit_shaped)["cvd"]
            != compute_cvd_from_klines(binance_shaped)["cvd"])

    # Колонки Bybit действительно не содержат taker_buy_base (источник drift).
    src = inspect.getsource(BybitClient.klines)
    assert "taker_buy_base" not in src


# --- unexpected mismatch ломает прогон --------------------------------------

def test_injected_identity_break_is_unexpected(monkeypatch):
    """Слом общего helper (как при скрытом форке логики) => unexpected + exit != 0."""
    monkeypatch.setattr(bp.pipeline, "apply_htf_policy",
                        lambda *a, **k: None, raising=True)
    checks = bp.reproduced_checks()
    assert any(c.name == "apply_htf_policy" and not c.ok for c in checks)
    _, summary = bp.evaluate()
    assert summary.unexpected
    assert bp.main() == 1


def test_injected_unclassified_gate_is_unexpected(monkeypatch):
    """Новый гейт в реестре без классификации => unexpected_mismatch."""
    extra = GATE_REGISTRY[0].__class__(
        "brand_new_gate", "brand_new_gate", "x.py", True, False,
        GATE_REGISTRY[0].outcome, reachable_from_result=True)
    monkeypatch.setattr(bp, "GATE_REGISTRY", GATE_REGISTRY + (extra,))
    gates = bp.classify_registry_gates()
    assert any(g.name == "brand_new_gate"
               and g.classification == bp.UNEXPECTED for g in gates)
    _, summary = bp.evaluate()
    assert any("brand_new_gate" in u for u in summary.unexpected)
    assert bp.main() == 1


# --- CLI / exit-code --------------------------------------------------------

def test_main_clean_exit_zero_and_prints(capsys):
    code = bp.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "RESULT: PASS" in out
    assert "unexpected mismatch:  0" in out
    assert "D1.." in out


# --- no behavior change: golden не тронуты, runner вне production ------------

def _golden_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(GOLDEN_DIR.glob("*.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_running_parity_does_not_touch_golden():
    before = _golden_digest()
    bp.run_all()
    bp.main()
    assert _golden_digest() == before, "golden-снапшоты изменились при прогоне runner"


def test_runner_not_imported_by_production():
    """Ни один production-модуль не импортирует tools.backtest_parity."""
    skip_parts = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip_parts & set(py.parts):
            continue
        if "backtest_parity" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"production ссылается на runner: {offenders}"
