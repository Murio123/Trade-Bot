"""G1.1 — the reconstructed history, and the honesty rules it must obey.

Reconstruction is the part of this stage that could most easily become
fiction, so the tests are mostly about what the table is not allowed to do:
claim a trial with no evidence, present an uncertainty as a number, or quietly
drop a stage.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from validation.trial_history import (ALL_TRIALS, missing_from, populate,
                                      summary)
from validation.trial_registry import (CONFIRMING_TIERS, ORIGIN_RECONSTRUCTED,
                                       TIER_INFERENCE, FamilyScope,
                                       TrialRegistry, canonical_json)
from tools.g1_1_trial_ledger import A_PRIME_SCOPE, SCOPES, build, format_report


@pytest.fixture()
def registry(tmp_path) -> TrialRegistry:
    reg = TrialRegistry(str(tmp_path / "trials.jsonl"))
    populate(reg)
    return reg


# --- the honesty rules ---------------------------------------------------


def test_every_historical_record_is_marked_reconstructed():
    for record in ALL_TRIALS:
        assert record.origin == ORIGIN_RECONSTRUCTED, record.identity.hypothesis
        assert record.reconstructed


def test_every_historical_record_cites_evidence():
    for record in ALL_TRIALS:
        assert record.evidence, record.identity.hypothesis
        for item in record.evidence:
            assert item.ref.strip()


def test_no_record_rests_on_an_empty_note_where_it_claims_a_range():
    """A band without an explanation is just an unexplained number."""
    for record in ALL_TRIALS:
        if record.multiplicity[0] != record.multiplicity[1]:
            assert record.notes.strip(), record.identity.hypothesis


def test_inferred_records_are_not_counted_as_confirmed():
    for record in ALL_TRIALS:
        if any(e.tier == TIER_INFERENCE for e in record.evidence):
            assert record.confirmed_low == 0, record.identity.hypothesis


def test_confirmed_records_all_cite_a_confirming_tier():
    for record in ALL_TRIALS:
        if record.confirmed_low > 0:
            assert any(e.tier in CONFIRMING_TIERS for e in record.evidence)


def test_the_confirmed_count_never_exceeds_the_conservative_count():
    counts = summary()
    assert counts["confirmed"] <= counts["conservative"]


def test_uncertainty_is_not_collapsed_into_one_number():
    """§8.3: where a band exists it is published, not averaged away."""
    counts = summary()
    assert counts["conservative"] > counts["confirmed"], (
        "the reconstruction claims perfect knowledge of a decade of research "
        "choices, which would be the one result this section forbids")


def test_no_two_records_collide_on_identity():
    ids = [r.trial_id for r in ALL_TRIALS]
    assert len(ids) == len(set(ids))


def test_reconstruction_is_deterministic():
    from importlib import reload

    import validation.trial_history as history
    before = [r.trial_id for r in history.ALL_TRIALS]
    reload(history)
    assert [r.trial_id for r in history.ALL_TRIALS] == before


def test_populating_twice_does_not_inflate_the_count(registry):
    first = registry.n_trials().as_dict()
    populate(registry)
    assert registry.n_trials().as_dict() == first
    assert len(registry.declarations()) == len(ALL_TRIALS)


def test_a_removed_historical_trial_is_caught_from_outside_the_file(registry):
    """The chain inside the file is not enough, and the audit was right to
    say so: remove a line, recompute `seq` and `prev_sha256`, and the file
    validates. Nothing file-local can prevent that.

    What can is that this history is code. The ids are re-derivable, so the
    deletion shows up regardless of how carefully the file was rewritten.
    """
    assert missing_from(registry) == ()

    rows = registry.read_raw()
    victim = rows[7]["trial_id"]
    kept = rows[:7] + rows[8:]
    prev = ""
    with open(registry.path, "w") as fh:
        for n, row in enumerate(kept):
            row = {k: v for k, v in row.items()
                   if k not in ("seq", "prev_sha256")}
            row["seq"] = n
            row["prev_sha256"] = prev
            line = canonical_json(row)
            fh.write(line + "\n")
            prev = hashlib.sha256(line.encode()).hexdigest()

    # The forged file passes every check the file itself can make...
    assert len(registry.declarations()) == len(ALL_TRIALS) - 1
    # ...and the deletion is caught anyway.
    assert missing_from(registry) == (victim,)


def test_the_materialized_registry_is_byte_stable(tmp_path):
    paths = []
    for name in ("a", "b"):
        reg = TrialRegistry(str(tmp_path / name / "trials.jsonl"))
        populate(reg)
        paths.append(reg.path)
    assert open(paths[0]).read() == open(paths[1]).read()


# --- what the history actually contains ----------------------------------


def test_abandoned_work_is_present_and_counted(registry):
    """H1 was rejected, LightGBM never ran. A count of survivors only would be
    counting the winners of a selection while denying one happened."""
    hypotheses = {r.identity.hypothesis for r in ALL_TRIALS}
    assert "H1_trend_pullback_continuation" in hypotheses
    assert "lightgbm_challenger" in hypotheses
    abandoned = [r for r in ALL_TRIALS if r.status in
                 ("abandoned", "insufficient_data")]
    assert len(abandoned) >= 4
    counted = registry.n_trials().trial_ids
    for record in abandoned:
        assert record.trial_id in counted


def test_the_c43_lineage_is_linked(registry):
    by_stage = {r.identity.research_stage: r for r in ALL_TRIALS
                if r.identity.research_stage.startswith("C4.3")
                or r.identity.research_stage == "C4.4"}
    assert by_stage["C4.3c"].parent_trial_id is not None
    assert by_stage["C4.3d"].parent_trial_id == by_stage["C4.3c"].trial_id
    assert by_stage["C4.3e"].parent_trial_id == by_stage["C4.3d"].trial_id
    assert by_stage["C4.4"].parent_trial_id == by_stage["C4.3e"].trial_id


def test_the_threshold_grid_is_counted_per_profile(registry):
    grid = [r for r in ALL_TRIALS if r.identity.research_stage == "C1.4a"]
    assert len(grid) == 4
    assert {r.identity.profile for r in grid} == {"swing", "position",
                                                  "intraday", "bounce"}
    assert sum(r.conservative_high for r in grid) == 48


def test_the_feature_screen_is_counted_per_feature(registry):
    screen = [r for r in ALL_TRIALS
              if r.identity.hypothesis == "single_feature_attribution"]
    assert len(screen) == 1
    assert screen[0].multiplicity == (32, 32)


def test_the_volatility_family_does_not_leak_into_the_trade_family(registry):
    trade = registry.n_trials(A_PRIME_SCOPE)
    vol = registry.n_trials(SCOPES["volatility_forecast"])
    assert set(trade.trial_ids).isdisjoint(vol.trial_ids)


def test_an_unrelated_family_is_smaller_than_everything(registry):
    everything = registry.n_trials(FamilyScope()).n_trials
    for name, scope in SCOPES.items():
        if name == "everything":
            continue
        assert registry.n_trials(scope).n_trials <= everything


def test_the_a_prime_family_is_the_conservative_count(registry):
    count = registry.n_trials(A_PRIME_SCOPE)
    assert count.conservative == count.n_trials
    assert count.confirmed < count.conservative
    # A drift guard, not a target: if the reconstruction changes, the result
    # document's headline number must change with it.
    assert count.confirmed == 83
    assert count.conservative == 100


def test_cross_profile_trials_land_in_the_swing_family(registry):
    """§6.4: a pre-C1 choice that applied to every profile is selection
    pressure on swing too, and must not vanish because its profile is null."""
    count = registry.n_trials(A_PRIME_SCOPE)
    cross = [r for r in ALL_TRIALS
             if r.identity.profile is None
             and r.identity.research_objective == "trade_expectancy"]
    assert cross
    for record in cross:
        assert record.trial_id in count.trial_ids


# --- the runner ----------------------------------------------------------


def test_the_runner_materializes_and_reports(tmp_path):
    payload = build(str(tmp_path))
    assert payload["registry"]["n_declarations"] == len(ALL_TRIALS)
    assert set(payload["counts"]) == set(SCOPES)
    report = format_report(payload)
    assert "a_prime_swing_trade_r" in report
    assert "CONSERVATIVE" in report


def test_the_runner_is_idempotent(tmp_path):
    first = build(str(tmp_path))
    second = build(str(tmp_path))
    assert first["registry"]["content_sha256"] == \
        second["registry"]["content_sha256"]
    assert first["counts"] == second["counts"]


def test_the_runner_report_is_json_serializable(tmp_path):
    payload = build(str(tmp_path))
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_the_runner_reads_no_market_data():
    """It is a bookkeeping tool. If it ever grows a data dependency, the
    sealed-holdout question comes back with it — so the check is on real
    imports, not on prose that happens to mention the holdout."""
    import ast

    tree = ast.parse(open("tools/g1_1_trial_ledger.py").read())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert modules == {"argparse", "json", "os", "typing", "__future__",
                       "validation.trial_history", "validation.trial_registry"}
