"""G1 integration invariants.

These are the criteria in `reports/c50/G1_SPEC.md` §5 that no unit test of an
individual module can cover: that the new apparatus did not open a frozen
artifact, did not duplicate a statistical primitive, and did not reverse the
project's import direction.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

M01 = REPO / "validation" / "deflated_sharpe.py"
M02 = REPO / "labeling" / "triple_barrier.py"
M03 = REPO / "labeling" / "sample_weights.py"
M04 = REPO / "validation" / "cpcv.py"
NEW_MODULES = (M01, M02, M03, M04,
               REPO / "validation" / "bar_grid.py",
               REPO / "validation" / "negative_controls.py")


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# --- frozen artifacts untouched (G1_SPEC.md §5.3) -----------------------------

def test_the_m00_cost_model_still_hashes_to_its_pinned_value():
    """`reports/c45/BASELINE.md` pins the M00 cost model at 1cd4db9fe8e8. P1
    preserved it by not touching `cost_sensitivity.py`; G1 must not touch it
    either."""
    import config
    from validation.cost_sensitivity import CostModel
    model = CostModel(taker_fee_pct=config.TAKER_FEE_PCT,
                      slippage_pct=config.SLIPPAGE_PCT)
    assert model.content_sha256()[:12] == "1cd4db9fe8e8"


def test_g1_does_not_import_the_frozen_cost_model():
    """M01-M04 are apparatus. None of them has any business reading the cost
    model, and an import would be the first step towards modifying it."""
    for path in (M01, M02, M03, M04):
        assert "validation.cost_sensitivity" not in imports_of(path), path.name


def test_the_sealed_holdout_boundary_is_never_read_past():
    from validation.cpcv import SEALED_HOLDOUT_IDX_LO
    assert SEALED_HOLDOUT_IDX_LO == 8199


# --- no duplicate primitives (implementation discipline) ----------------------

def test_modal_interval_detection_has_exactly_one_implementation():
    """P1 and M02 both need it. Two copies would drift, and then two modules
    would disagree about what a gap is."""
    offenders = []
    for py in REPO.rglob("*.py"):
        if ({".venv", ".git", "__pycache__", "scratchpad", "tests"}
                & set(py.parts)):
            continue
        src = py.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"def modal_interval", src) and py.name != "bar_grid.py":
            offenders.append(str(py.relative_to(REPO)))
    # tools/funding_cache.py keeps its own detector for raw API records, which is
    # a different input (unparsed dicts, not a stamp array) and a documented
    # exception.
    assert offenders == ["tools/funding_cache.py"], offenders


def test_funding_delegates_its_bar_grid_helpers():
    """The extraction must be a delegation, not a copy left behind."""
    src = M01.parent.joinpath("funding.py").read_text(encoding="utf-8")
    assert "from validation.bar_grid import" in src
    assert "def _modal_interval_ms" not in src
    assert "def _detect_gaps" not in src


def test_there_is_only_one_sharpe_ratio_implementation():
    """One formula, in one file.

    G1.1 added `validation/research_dsr.py`, whose `research_deflated_sharpe`
    matches the name pattern without being a second implementation: it derives
    a trial count and hands the series to M01. That is checked below rather
    than assumed, so the exemption cannot quietly become a hiding place for a
    second Sharpe.
    """
    delegating = REPO / "validation" / "research_dsr.py"
    offenders = []
    for py in REPO.rglob("*.py"):
        if ({".venv", ".git", "__pycache__", "scratchpad", "tests"}
                & set(py.parts)):
            continue
        src = py.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"def \w*sharpe", src) and py not in (M01, delegating):
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, offenders

    src = delegating.read_text(encoding="utf-8")
    assert "from validation.deflated_sharpe import" in src
    # No moments, no ratio, no distribution: the wrapper does arithmetic on
    # nothing. If it ever starts to, this fails and the exemption is revisited.
    for arithmetic in ("np.mean", "np.std", "np.sqrt", "math.sqrt",
                       "skew", "kurtosis", "/ std"):
        assert arithmetic not in src, arithmetic


def test_there_is_only_one_uniqueness_weighting_implementation():
    offenders = []
    for py in REPO.rglob("*.py"):
        if ({".venv", ".git", "__pycache__", "scratchpad", "tests"}
                & set(py.parts)):
            continue
        src = py.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"def (uniqueness_weights|average_uniqueness|concurrency)\b",
                     src) and py != M03:
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, offenders


def test_the_p1_cost_layer_is_the_only_source_of_cost_r():
    """P1's invariant, re-asserted: the controls charge costs through
    `validation.trade_costs`, never with a local formula."""
    src = (REPO / "validation" / "negative_controls.py").read_text(encoding="utf-8")
    assert "from validation.trade_costs import" in src
    assert "TAKER_FEE_PCT" not in src
    assert "2 * " not in src.replace("2 * 3_600_000", "")


# --- import direction (ARCHITECTURE.md §5.9) ---------------------------------

def test_the_new_modules_do_not_import_the_offline_tools_layer():
    """`validation` and `labeling` are imported by production paths, so an
    import of `tools.*` here would drag the offline layer into the runtime. This
    is the invariant P1 tripped over once already."""
    for path in NEW_MODULES:
        for name in imports_of(path):
            assert name != "tools" and not name.startswith("tools."), \
                f"{path.name} imports {name}"


def test_labeling_does_not_import_production_modules():
    """M02 and M03 are pure primitives. An import of config, database or the
    analyzer would make them untestable in isolation and couple label geometry to
    live settings."""
    forbidden = ("config", "database", "scheduler", "analyzer", "bot",
                 "signal_engine")
    for path in (M02, M03):
        for name in imports_of(path):
            root = name.split(".")[0]
            assert root not in forbidden, f"{path.name} imports {name}"


def test_the_apparatus_creates_no_orders_and_writes_no_sql():
    """ARCHITECTURE.md §5.10 (DRY_RUN) and §5.9. A research module has no
    business writing anything."""
    for path in NEW_MODULES:
        src = path.read_text(encoding="utf-8")
        for verb in ("INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "TRUNCATE"):
            assert not re.search(rf"\b{verb}\b", src), f"{path.name}: {verb}"
        assert "create_order" not in src and "place_order" not in src


def test_production_still_does_not_import_the_research_packages():
    """The direction runs research -> production, never back. `labeling` is a new
    top-level package and is the obvious thing for a future edit to reach for
    from the wrong side."""
    # `spot` joins the research side (the S-track: universe, labels, features).
    # It is offline research and may import `labeling`; what it may not do is
    # be imported BY production, which the next test asserts.
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad",
            "validation", "labeling", "spot"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        for name in imports_of(py):
            if name == "labeling" or name.startswith("labeling."):
                offenders.append(str(py.relative_to(REPO)))
                break
    assert not offenders, offenders


def test_production_does_not_import_the_spot_research_package():
    """The S-track is research too, so the same direction rule binds it.

    Widening the skip list above without this test would have quietly bought
    `spot` an exemption in both directions.
    """
    skip = {"tests", "tools", ".venv", ".git", "__pycache__", "scratchpad",
            "validation", "labeling", "spot"}
    offenders = []
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        for name in imports_of(py):
            if name == "spot" or name.startswith("spot."):
                offenders.append(str(py.relative_to(REPO)))
                break
    assert not offenders, offenders


def test_the_spot_research_package_does_not_import_production():
    """Same rule as M02/M03: no config, no database, no live analyzer."""
    forbidden = ("config", "database", "scheduler", "analyzer", "bot",
                 "signal_engine", "pipeline", "risk")
    for path in (REPO / "spot").rglob("*.py"):
        for name in imports_of(path):
            root = name.split(".")[0]
            assert root not in forbidden, f"{path.name} imports {name}"


# --- the governed documents ---------------------------------------------------

def test_the_g1_spec_exists_and_records_its_amendments():
    spec = (REPO / "reports" / "c50" / "G1_SPEC.md").read_text(encoding="utf-8")
    assert "## 7. Amendments" in spec
    # The thresholds must still be stated as numbers, not as references.
    assert "FPR <= 0.075" in spec
    for amendment in ("### A1", "### A2", "### A3"):
        assert amendment in spec, amendment


# P1's closing commit, and therefore G1's anchor. Anything the frozen documents
# gained before it belongs to P1's record, not to G1's.
G1_ANCHOR = "7c8f37e19835d47b2e7ba842d9c115bc03f1f344"


def test_the_architecture_and_baseline_documents_are_not_modified_by_g1():
    """G1 may add documents. It may not rewrite the frozen ones."""
    import subprocess
    out = subprocess.run(
        ["git", "diff", "--name-only", G1_ANCHOR, "HEAD", "--",
         "reports/c50/ARCHITECTURE.md", "reports/c45/BASELINE.md",
         "reports/c50/P1_TASK.md", "reports/c50/P1_RESULT.md",
         "validation/cost_sensitivity.py"],
        cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:      # pragma: no cover - shallow clone or no git
        pytest.skip("git history unavailable")
    assert out.stdout.strip() == "", out.stdout
