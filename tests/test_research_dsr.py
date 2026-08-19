"""G1.1 — M01 provenance, and the guardrails against future undercounting.

The point of these tests is narrow and worth stating: M01's arithmetic is
already covered by `tests/test_deflated_sharpe.py` and is not retested here.
What is tested here is that a project research result cannot acquire an
`n_trials` from anywhere except the registry.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from validation.deflated_sharpe import DSR_VERSION, deflated_sharpe
from validation.research_dsr import (RESEARCH_DSR_VERSION, ProvenanceError,
                                     research_deflated_sharpe,
                                     research_trial_count)
from validation.trial_registry import (ORIGIN_SYNTHETIC, REGISTRY_VERSION,
                                       Evidence, FamilyScope, TrialIdentity,
                                       TrialRecord, TrialRegistry,
                                       UndeclaredTrialError, with_parent)

SCOPE = FamilyScope(research_objective="trade_expectancy", profile="swing",
                    symbol="BTCUSDT", target_family="trade_r")

# Modules permitted to call the mathematical layer directly. Extending this
# list is a visible diff on a governed guardrail, which is the point of
# keeping it in the test rather than in the module under test.
DIRECT_CALL_ALLOWLIST = {
    "validation/negative_controls.py",   # synthetic nulls, no real data
    "validation/research_dsr.py",        # the wrapper itself
    "validation/deflated_sharpe.py",     # the definition
}

SCANNED_PACKAGES = ("validation", "labeling", "models", "research", "forecast",
                    "tools")


def _record(hypothesis: str = "h", **overrides) -> TrialRecord:
    extra = {k: overrides.pop(k) for k in list(overrides)
             if k in {"origin", "status", "evidence", "multiplicity",
                      "parent_trial_id", "target_family"}}
    identity = TrialIdentity.build(
        research_objective=overrides.pop("research_objective",
                                         "trade_expectancy"),
        research_stage="X", hypothesis=hypothesis,
        profile=overrides.pop("profile", "swing"), symbol="BTCUSDT",
        **overrides)
    return TrialRecord(identity=identity, created_at="2026-01-01",
                       target_family=extra.pop("target_family", "trade_r"),
                       **extra)


@pytest.fixture()
def registry(tmp_path) -> TrialRegistry:
    return TrialRegistry(str(tmp_path / "trials.jsonl"))


@pytest.fixture()
def returns() -> np.ndarray:
    rng = np.random.default_rng(4242)
    return rng.normal(0.05, 1.0, 400)


# --- the count comes from the registry -----------------------------------


def test_the_wrapper_takes_its_trial_count_from_the_registry(registry, returns):
    declared = registry.declare(_record())
    for i in range(4):
        registry.declare(_record(f"other_{i}"))
    result = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    assert result.dsr.n_trials == 5
    assert result.provenance.conservative == 5


def test_the_wrapper_has_no_n_trials_parameter():
    """The guarantee is structural, not documentary: there is no argument to
    pass, so no caller can supply one."""
    import inspect
    params = inspect.signature(research_deflated_sharpe).parameters
    assert "n_trials" not in params


def test_provenance_records_the_registry_state(registry, returns):
    declared = registry.declare(_record())
    result = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    prov = result.provenance
    assert prov.registry_sha256 == registry.content_sha256()
    assert prov.registry_path == registry.path
    assert prov.n_declarations == 1
    assert prov.registry_version == REGISTRY_VERSION
    assert prov.dsr_version == DSR_VERSION
    assert prov.research_dsr_version == RESEARCH_DSR_VERSION


def test_provenance_pins_a_registry_that_later_grows(registry, returns):
    """A published number must stay checkable against the registry it was
    computed from, not against whatever the registry becomes."""
    declared = registry.declare(_record())
    first = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                     trial_id=declared.trial_id)
    registry.declare(_record("later"))
    second = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    assert first.provenance.registry_sha256 != second.provenance.registry_sha256
    assert second.dsr.n_trials > first.dsr.n_trials


def test_provenance_carries_the_full_decomposition(registry, returns):
    declared = registry.declare(_record(multiplicity=(1, 4),
                                        origin="reconstructed",
                                        evidence=(Evidence("git_history", "a"),)))
    result = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    prov = result.provenance
    assert (prov.confirmed, prov.uncertain, prov.conservative) == (1, 3, 4)
    assert prov.scope["profile"] == "swing"


def test_the_conservative_count_is_what_deflates(registry, returns):
    """§8.3: the uncertain half of a reconstruction is used, not discarded."""
    declared = registry.declare(_record(multiplicity=(1, 40),
                                        origin="reconstructed",
                                        evidence=(Evidence("git_history", "a"),)))
    result = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    assert result.dsr.n_trials == 40
    reference = deflated_sharpe(returns, n_trials=40)
    assert result.dsr.dsr == pytest.approx(reference.dsr)


def test_more_trials_lower_the_deflated_sharpe(registry, returns):
    declared = registry.declare(_record())
    lonely = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    for i in range(50):
        registry.declare(_record(f"sibling_{i}"))
    crowded = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                       trial_id=declared.trial_id)
    assert crowded.dsr.dsr < lonely.dsr.dsr


def test_a_lineage_ancestor_deflates_the_descendant(registry, returns):
    parent = registry.declare(_record("parent",
                                      research_objective="volatility_forecast",
                                      target_family="volatility_ratio"))
    child = registry.declare(with_parent(_record("child"), parent))
    result = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=child.trial_id)
    assert result.dsr.n_trials == 2


# --- what the wrapper refuses --------------------------------------------


def test_an_undeclared_trial_cannot_be_deflated(registry, returns):
    with pytest.raises(UndeclaredTrialError, match="not declared"):
        research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                 trial_id="t_never_declared")


def test_a_trial_cannot_be_deflated_against_a_family_that_excludes_it(
        registry, returns):
    """The cheapest undercount available: pick a narrow family, get a small
    number, quote it."""
    declared = registry.declare(_record(profile="intraday"))
    with pytest.raises(ProvenanceError, match="not a member"):
        research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                 trial_id=declared.trial_id)


def test_a_synthetic_trial_is_refused_by_the_research_path(registry, returns):
    declared = registry.declare(_record(origin=ORIGIN_SYNTHETIC))
    with pytest.raises(ProvenanceError, match="synthetic"):
        research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                 trial_id=declared.trial_id)


def test_research_trial_count_is_the_same_object_the_wrapper_uses(registry,
                                                                  returns):
    declared = registry.declare(_record())
    snapshot, count = research_trial_count(registry, SCOPE, declared.trial_id)
    result = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    assert result.dsr.n_trials == count.n_trials
    assert result.provenance.registry_sha256 == snapshot["content_sha256"]


def test_the_result_serializes_with_its_provenance(registry, returns):
    declared = registry.declare(_record())
    result = research_deflated_sharpe(returns, registry=registry, scope=SCOPE,
                                      trial_id=declared.trial_id)
    payload = result.as_dict()
    assert set(payload) == {"dsr", "provenance"}
    assert payload["provenance"]["trial_id"] == declared.trial_id


# --- the mathematical layer stays usable ---------------------------------


def test_m01_remains_callable_as_a_pure_primitive(returns):
    """Breaking M01 as a mathematical function to enforce governance would
    trade one defect for another; the unit layer keeps its own contract."""
    assert deflated_sharpe(returns, n_trials=7).n_trials == 7


# --- the static guardrail ------------------------------------------------


def _python_files() -> list[str]:
    out = []
    for package in SCANNED_PACKAGES:
        for root, _dirs, files in os.walk(package):
            if "__pycache__" in root:
                continue
            out.extend(os.path.join(root, f) for f in files
                       if f.endswith(".py"))
    return sorted(out)


def _direct_dsr_calls(path: str) -> list[int]:
    """Lines where a module supplies its own trial count.

    The rule is deliberately about `n_trials` rather than about the callee's
    name. An earlier version matched calls named `deflated_sharpe`, which an
    alias import (`import ... as dsr`) or a `getattr` lookup walked straight
    past — a guardrail that only stops the obvious spelling of a bypass is not
    a guardrail. `n_trials` means one thing in this repository, so any module
    naming it is claiming a trial count, however it reaches M01.
    """
    tree = ast.parse(open(path).read())
    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "n_trials":
                    hits.add(kw.value.lineno)
        elif isinstance(node, ast.Dict):
            # Any dict literal naming the key, wherever it is written. An
            # earlier version looked only inside `f(**{...})`, which lifting
            # the same dict into a variable — `KW = {"n_trials": 1}` then
            # `f(**KW)` — walked straight past. Reading a count back off a
            # result stays unflagged: `payload["n_trials"]` is a subscript and
            # `.get("n_trials")` is a positional argument, neither of which is
            # a dict key.
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "n_trials":
                    hits.add(key.lineno)
    return sorted(hits)


def test_no_research_module_injects_its_own_trial_count():
    """The rule that actually prevents future undercounting.

    A new A' runner that calls the maths layer with a hand-picked n_trials
    would reintroduce exactly the failure this stage closed, and would do it
    silently. This makes it a test failure instead.
    """
    offenders = {}
    for path in _python_files():
        if path.replace(os.sep, "/") in DIRECT_CALL_ALLOWLIST:
            continue
        lines = _direct_dsr_calls(path)
        if lines:
            offenders[path] = lines
    assert offenders == {}, (
        f"these modules pass n_trials directly to M01 instead of deriving it "
        f"from the trial registry: {offenders}. Use "
        f"validation.research_dsr.research_deflated_sharpe, or add the module "
        f"to DIRECT_CALL_ALLOWLIST with a stated reason.")


def test_the_guardrail_can_actually_fail(tmp_path):
    """A guardrail that cannot fail is decoration.

    The scan is pointed at a module that does the forbidden thing, and must
    report it.
    """
    offender = tmp_path / "sneaky_runner.py"
    offender.write_text(
        "from validation.deflated_sharpe import deflated_sharpe\n"
        "def go(r):\n"
        "    return deflated_sharpe(r, n_trials=1)\n")
    assert _direct_dsr_calls(str(offender)) == [3]


@pytest.mark.parametrize("source", [
    # An alias import: the callee is no longer spelled `deflated_sharpe`.
    "from validation.deflated_sharpe import deflated_sharpe as dsr\n"
    "def go(r):\n"
    "    return dsr(r, n_trials=1)\n",
    # A module alias, same idea one level up.
    "import validation.deflated_sharpe as m\n"
    "def go(r):\n"
    "    return m.deflated_sharpe(r, n_trials=1)\n",
    # Dynamic lookup: the call's func is itself a Call node.
    "import validation.deflated_sharpe as m\n"
    "def go(r):\n"
    "    return getattr(m, 'deflated_sharpe')(r, n_trials=1)\n",
    # The argument spelled as data rather than as a keyword.
    "import validation.deflated_sharpe as m\n"
    "def go(r):\n"
    "    return m.deflated_sharpe(r, **{'n_trials': 1})\n",
])
def test_the_guardrail_survives_the_obvious_evasions(tmp_path, source):
    """Each of these walks past a scan that matches on the callee's name.

    The rule is about `n_trials`, not about how M01 was reached, precisely so
    that renaming the route does not change the answer.
    """
    offender = tmp_path / "evader.py"
    offender.write_text(source)
    assert _direct_dsr_calls(str(offender)) == [3]


def test_the_guardrail_catches_a_dict_lifted_into_a_variable(tmp_path):
    """`f(**{...})` and `KW = {...}; f(**KW)` are the same bypass.

    The second one walked past the first version of this scan, which only
    looked inside the call node. Any dict literal naming the key is flagged
    now, which is what TRIAL_REGISTRY_SPEC.md §7.3 / A2 says the rule is:
    no module outside the allowlist may name `n_trials` as a keyword argument
    or as a literal key.
    """
    offender = tmp_path / "evader.py"
    offender.write_text("import validation.deflated_sharpe as m\n"
                        "KW = {'n_trials': 1}\n"
                        "def go(r):\n"
                        "    return m.deflated_sharpe(r, **KW)\n")
    assert _direct_dsr_calls(str(offender)) == [2]


@pytest.mark.parametrize("source", [
    # Attribute access on a result.
    "def render(result):\n"
    "    return f'deflated against {result.dsr.n_trials} trials'\n",
    # A serialized result read back out of its payload.
    "def render(payload):\n"
    "    return payload['dsr']['n_trials']\n",
    # The same, defensively.
    "def render(payload):\n"
    "    return payload.get('n_trials', 0)\n",
    # A report column header.
    "COLUMNS = ['trial_id', 'n_trials', 'dsr']\n",
])
def test_the_guardrail_does_not_flag_reading_a_trial_count(tmp_path, source):
    """Reporting `n_trials` is not supplying one.

    A guard that failed here would push future reporters toward hiding the
    number — the opposite of what this stage is for. Only a count *written*
    is flagged — call keywords and dict-literal keys — rather than every
    occurrence of the string. Building `{"n_trials": count}` for output does
    trip the guard; that is deliberate, and the allowlist is where a module
    that legitimately publishes the number belongs.
    """
    reader = tmp_path / "reporter.py"
    reader.write_text(source)
    assert _direct_dsr_calls(str(reader)) == []


def test_the_allowlist_entries_all_exist():
    """An allowlist that names a deleted file is silently wider than it
    reads."""
    for path in DIRECT_CALL_ALLOWLIST:
        assert os.path.exists(path), path


def test_the_allowlisted_modules_actually_need_the_exemption():
    """If an allowlisted module stops calling the maths layer directly, the
    exemption should be removed rather than left lying around."""
    for path in sorted(DIRECT_CALL_ALLOWLIST - {"validation/deflated_sharpe.py"}):
        assert _direct_dsr_calls(path), (
            f"{path} is allowlisted but no longer calls deflated_sharpe with "
            f"n_trials; drop it from DIRECT_CALL_ALLOWLIST")


def test_research_dsr_does_not_import_offline_tools():
    tree = ast.parse(open("validation/research_dsr.py").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("tools")
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("tools") for a in node.names)


def test_no_cycle_between_the_registry_and_the_wrapper():
    tree = ast.parse(open("validation/deflated_sharpe.py").read())
    modules = {node.module for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module}
    assert not any("trial_registry" in m or "research_dsr" in m
                   for m in modules)
