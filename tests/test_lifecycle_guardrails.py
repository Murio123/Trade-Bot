"""Pre-Stage 15B Guardrail 3: prove setup-lifecycle metadata is isolated from
the trading decision path.

Stage 15B will add analytics-only fields to the forecast record, e.g.
``setup_lifecycle_status`` / ``setup_lifecycle_reasons`` / ``previous_forecast_id``
/ ``setup_lifecycle_comparable`` / ``setup_score_delta`` /
``setup_confidence_delta``. These are derived POST-FACTUM and must never feed
back into run_cascade / final_gate / scoring / risk / confidence / direction /
Telegram trading signal.

All checks here are static (source/AST inspection) or pure-function invariants:
no DB, no network, no Telegram, no live scheduler. They intentionally fail if a
future change wires lifecycle output into a decision module — that is the point.
"""
from __future__ import annotations

import ast
import importlib.util
from datetime import datetime, timezone

import pytest

UTC = timezone.utc

# Tokens that only exist because lifecycle metadata is involved. If any of these
# appear in a decision module, lifecycle has leaked into the decision path.
LIFECYCLE_TOKENS = (
    "setup_lifecycle",
    "classify_transition",
    "setup_lifecycle_status",
    "setup_lifecycle_reasons",
    "previous_forecast_id",
    "setup_lifecycle_comparable",
    "setup_score_delta",
    "setup_confidence_delta",
)

# Modules that COMPUTE the trading decision (status / bias / confidence / scores
# / stop / TP / RR / gating / risk). Deliberately excludes the persistence
# boundary (signal_engine.forecast_record), where lifecycle metadata may
# legitimately be carried later — it just must not drive a decision.
DECISION_MODULES = (
    "pipeline",                      # run_cascade lives here
    "contracts.final_gate",
    "signal_engine.confluence",      # scoring
    "signal_engine.quality_score",   # scoring
    "signal_engine.no_trade_gate",
    "signal_engine.conflict_resolver",
    "signal_engine.mtf_confidence",
    "signal_engine.regime",
    "signal_engine.htf_filter",
    "signal_engine.vetoes",
    "signal_engine.cooldown",
    "signal_engine.daily_limiter",
    "signal_engine.profiles",
    "signal_engine.schema",
    "risk.position_sizing",
)

# Packages the pure lifecycle leaf must never pull in (would couple it to
# runtime / decision-path / IO).
FORBIDDEN_LEAF_IMPORTS = frozenset({
    "signal_engine", "scheduler", "pipeline", "bot", "ai", "risk", "database",
})


def _module_source(modname: str) -> str:
    """Read a module's source WITHOUT importing/executing it (locate via spec).

    Reading the file text keeps the guard cheap and side-effect free — no heavy
    imports (pandas, apscheduler, Telegram) and no network at collection time.
    """
    spec = importlib.util.find_spec(modname)
    assert spec is not None and spec.origin, f"cannot locate module {modname}"
    with open(spec.origin, encoding="utf-8") as fh:
        return fh.read()


def _imported_roots(src: str) -> set[str]:
    """Top-level package names imported by a source string (AST, robust)."""
    roots: set[str] = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:      # skip relative imports
                roots.add(node.module.split(".")[0])
    return roots


def _top_level_func_block(src: str, name: str) -> str:
    """Return the source of a top-level ``def``/``async def`` by name.

    Line-number independent: slices from the header to the next unindented,
    non-blank line. Used to assert call ORDER inside a function.
    """
    lines = src.splitlines()
    start = next(
        i for i, line in enumerate(lines)
        if line.startswith(f"def {name}") or line.startswith(f"async def {name}")
    )
    body = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and not line[0].isspace():   # next top-level statement
            break
        body.append(line)
    return "\n".join(body)


# ---------------------------------------------------------------------------
# 4. Pure analyzer guard: setup_lifecycle stays a stdlib-only leaf.
# ---------------------------------------------------------------------------

def test_setup_lifecycle_is_pure_leaf():
    roots = _imported_roots(_module_source("analyzer.setup_lifecycle"))
    leaked = roots & FORBIDDEN_LEAF_IMPORTS
    assert not leaked, (
        "analyzer.setup_lifecycle must stay a pure leaf; it must not import "
        f"runtime/decision/IO packages, but imports: {sorted(leaked)}"
    )


def test_setup_lifecycle_does_not_import_analyzer_outcomes_or_tools():
    # Keeps the classifier reusable offline without dragging in producers.
    roots = _imported_roots(_module_source("analyzer.setup_lifecycle"))
    for forbidden in ("tools",):
        assert forbidden not in roots, (
            f"analyzer.setup_lifecycle must not import {forbidden}"
        )


# ---------------------------------------------------------------------------
# 2 & 5. Decision-path import + regression guard: no lifecycle in decisions.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modname", DECISION_MODULES)
def test_decision_module_has_no_lifecycle_tokens(modname):
    """A decision module referencing any lifecycle token means lifecycle output
    has leaked into decision logic. This is the Stage 15B regression trip-wire.
    """
    src = _module_source(modname)
    hits = [tok for tok in LIFECYCLE_TOKENS if tok in src]
    assert not hits, (
        f"{modname} references lifecycle tokens {hits}; lifecycle metadata must "
        "not enter run_cascade/final_gate/scoring/risk decision logic"
    )


@pytest.mark.parametrize("modname", DECISION_MODULES)
def test_decision_module_does_not_import_setup_lifecycle(modname):
    # A decision module may legitimately depend on other analyzers, but never on
    # the lifecycle classifier (by full path or bare name).
    src = _module_source(modname)
    assert "analyzer.setup_lifecycle" not in src and "setup_lifecycle" not in src, (
        f"{modname} must not import analyzer.setup_lifecycle"
    )


# ---------------------------------------------------------------------------
# 1. forecast_record isolation: injecting lifecycle fields into the engine
#    result must not change any decision-derived value in the persisted record,
#    and lifecycle keys must not be silently persisted by the record builder.
# ---------------------------------------------------------------------------

def _decision_result() -> dict:
    return {
        "status": "alert",
        "analysis_status": "ENTER",
        "direction": "long",
        "candidate_direction": "long",
        "final_bias": "LONG",
        "long_score": 8.0,
        "short_score": 2.0,
        "raw_confidence": 0.72,
        "stop_loss": 99_000.0,
        "take_profit_levels": [101_000.0, 102_000.0],
        "risk_reward": 2.1,
        "expected_move_points": 2000.0,
        "signal_candle_close_time": datetime(2026, 7, 1, 4, tzinfo=UTC),
        "decision_time": datetime(2026, 7, 1, 4, 1, tzinfo=UTC),
    }


def _lifecycle_injection() -> dict:
    return {
        "setup_lifecycle_status": "UPGRADED",
        "setup_lifecycle_reasons": ["confidence_up", "long_score_up"],
        "previous_forecast_id": 41,
        "setup_lifecycle_comparable": True,
        "setup_score_delta": 3.0,
        "setup_confidence_delta": 0.2,
    }


DECISION_DERIVED_FIELDS = (
    "analysis_status", "final_bias", "candidate_direction", "raw_confidence",
    "long_score", "short_score", "stop_loss", "take_profit_levels",
    "risk_reward", "expected_move_points",
)


def test_forecast_record_decision_fields_invariant_to_lifecycle():
    from signal_engine.forecast_record import build_forecast_record
    from signal_engine.profiles import get_profile

    profile = get_profile("swing")
    ctx = {"volatility": {"regime": "expansion"}}

    plain = build_forecast_record(_decision_result(), ctx, profile)
    injected = build_forecast_record(
        {**_decision_result(), **_lifecycle_injection()}, ctx, profile)

    for field in DECISION_DERIVED_FIELDS:
        assert plain[field] == injected[field], (
            f"lifecycle injection changed decision-derived field {field!r}: "
            f"{plain[field]!r} -> {injected[field]!r}"
        )


def test_forecast_record_does_not_silently_persist_lifecycle_fields():
    """Until Stage 15B deliberately adds lifecycle columns to the record, the
    builder must not leak injected lifecycle keys into the persisted row."""
    from signal_engine.forecast_record import build_forecast_record
    from signal_engine.profiles import get_profile

    injected = build_forecast_record(
        {**_decision_result(), **_lifecycle_injection()},
        {"volatility": {"regime": "expansion"}}, get_profile("swing"))

    for key in _lifecycle_injection():
        assert key not in injected, (
            f"forecast record silently carried lifecycle key {key!r}; add it "
            "explicitly (schema + _FORECAST_COLS) in Stage 15B instead"
        )


# ---------------------------------------------------------------------------
# 3. Producer ordering guard: the deterministic decision is built BEFORE the
#    forecast record, which is built before it is persisted. Any future
#    lifecycle wiring therefore happens at/after the record — never before the
#    decision — so it cannot influence run_cascade.
# ---------------------------------------------------------------------------

def test_scheduler_builds_decision_before_forecast_record():
    block = _top_level_func_block(_module_source("scheduler"), "analysis_job")
    i_cascade = block.find("run_cascade(")
    i_record = block.find("build_forecast_record(")
    i_insert = block.find("insert_forecast(")
    assert i_cascade != -1, "run_cascade call not found in analysis_job"
    assert i_record != -1, "build_forecast_record call not found in analysis_job"
    assert i_insert != -1, "insert_forecast call not found in analysis_job"
    assert i_cascade < i_record < i_insert, (
        "decision (run_cascade) must be built before the forecast record and "
        "its persistence; lifecycle wiring must not precede the decision"
    )


def test_scheduler_analysis_job_has_no_lifecycle_tokens_before_decision():
    """Nothing lifecycle-related may sit between job start and run_cascade."""
    block = _top_level_func_block(_module_source("scheduler"), "analysis_job")
    pre_decision = block[: block.find("run_cascade(")]
    hits = [tok for tok in LIFECYCLE_TOKENS if tok in pre_decision]
    assert not hits, (
        f"lifecycle tokens {hits} appear before run_cascade in analysis_job; "
        "lifecycle must be derived after the decision, not before it"
    )


# ---------------------------------------------------------------------------
# Stage 15B2: producer wiring guards. Lifecycle is now COMPUTED and PERSISTED
# for new forecasts — prove the wiring stays analytics-only and correctly
# ordered, and that the new lifecycle files carry no execution/order code.
# ---------------------------------------------------------------------------

# Extra decision modules must never import the wrapper either.
def test_decision_modules_do_not_import_forecast_lifecycle():
    for modname in DECISION_MODULES:
        src = _module_source(modname)
        assert "forecast_lifecycle" not in src and \
               "enrich_forecast_with_lifecycle" not in src, (
            f"{modname} must not import the lifecycle wrapper; lifecycle is "
            "analytics-only and must not enter the decision path"
        )


def test_scheduler_lifecycle_wiring_order():
    """run_cascade < build_forecast_record < enrich_forecast_with_lifecycle
    < insert_forecast — lifecycle is computed after the decision+record and
    merged strictly before persistence."""
    block = _top_level_func_block(_module_source("scheduler"), "analysis_job")
    i_cascade = block.find("run_cascade(")
    i_record = block.find("build_forecast_record(")
    i_enrich = block.find("enrich_forecast_with_lifecycle(")
    i_insert = block.find("insert_forecast(")
    assert -1 not in (i_cascade, i_record, i_enrich, i_insert), (
        "expected run_cascade, build_forecast_record, "
        "enrich_forecast_with_lifecycle and insert_forecast in analysis_job"
    )
    assert i_cascade < i_record < i_enrich < i_insert, (
        "lifecycle enrichment must sit after the decision+record and before "
        f"the insert (got {i_cascade}, {i_record}, {i_enrich}, {i_insert})"
    )


def test_scheduler_lifecycle_wiring_does_not_touch_telegram():
    """The lifecycle merge line must not reach into Telegram/alert output."""
    block = _top_level_func_block(_module_source("scheduler"), "analysis_job")
    # Line that merges lifecycle fields into the record.
    merge_line = next(
        (ln for ln in block.splitlines()
         if "enrich_forecast_with_lifecycle(" in ln), "")
    assert merge_line, "lifecycle merge line not found in analysis_job"
    for tok in ("format_signal", "send_message", "alerts.", "formatting."):
        assert tok not in merge_line, (
            f"lifecycle merge line touches Telegram token {tok!r}"
        )


def test_lifecycle_wrapper_is_not_a_decision_module():
    """forecast_lifecycle must not import the decision/scoring/risk engine."""
    roots = _imported_roots(_module_source("forecast_lifecycle"))
    forbidden = {"pipeline", "signal_engine", "risk", "contracts", "bot", "ai"}
    leaked = roots & forbidden
    assert not leaked, (
        f"forecast_lifecycle must stay analytics glue; leaked imports: {sorted(leaked)}"
    )


EXECUTION_TOKENS = (
    "create_order", "market_order", "limit_order",
    "api_key", "api_secret", "exchange.create",
)


def test_no_exchange_execution_tokens_in_lifecycle_files():
    """Manual-only principle: no order/exchange/API code in lifecycle wiring."""
    for modname in ("analyzer.setup_lifecycle", "forecast_lifecycle"):
        src = _module_source(modname)
        hits = [tok for tok in EXECUTION_TOKENS if tok in src]
        assert not hits, f"{modname} carries execution tokens {hits}"
