"""C5.0 governance record — the decision must stay auditable, and separate.

`reports/c50/g1_governance.json` exists so that one particular drift cannot
happen quietly: a project that moved on deciding, later, that the gate it
proceeded past had really passed all along. These tests hold the two halves
apart — the historical G1 verdict, which is a fact, and the option-3 decision,
which is dated and forward-only — and check that the canonical trial count
recorded for a reader is the one the registry actually derives.
"""
from __future__ import annotations

import json

import pytest

from tools.g1_1_trial_ledger import A_PRIME_SCOPE
from validation.trial_history import populate
from validation.trial_registry import TrialRegistry

GOVERNANCE = "reports/c50/g1_governance.json"
DECISION_DOC = "reports/c50/G1_T3_DECISION.md"
G1_RESULT = "reports/c50/G1_RESULT.md"


@pytest.fixture(scope="module")
def gov() -> dict:
    with open(GOVERNANCE) as fh:
        return json.load(fh)


# --- the historical fact -------------------------------------------------


def test_historical_g1_verdict_is_indeterminate(gov):
    fact = gov["historical_fact"]
    assert fact["gate"] == "G1"
    assert fact["verdict"] == "G1_INDETERMINATE"
    assert fact["immutable"] is True


def test_no_retroactive_pass_is_permitted_by_the_record(gov):
    fact = gov["historical_fact"]
    assert fact["retroactive_pass_permitted"] is False
    assert fact["reinterpretation_permitted"] is False
    assert fact["rerun_for_cleaner_verdict_permitted"] is False
    assert fact["a3_adopted"] is False


def test_the_g1_result_document_still_publishes_indeterminate():
    """The decision appended a note to G1_RESULT.md. It must not have edited
    the verdict, and the document must still carry exactly one verdict line."""
    lines = open(G1_RESULT).read().splitlines()
    verdicts = [ln for ln in lines if ln.startswith("## Verdict")]
    assert verdicts == ["## Verdict: **G1_INDETERMINATE**"]


def test_the_decision_documents_restate_the_historical_verdict():
    """A reader who opens only the decision record must still learn that G1
    did not pass. The prose may discuss G1_PASS - the spec's A3 weighs it as
    an option, and G1_RESULT.md narrates the wrong verdict it once published -
    but the indeterminate outcome has to be stated in each of them."""
    for path in (DECISION_DOC, "reports/c50/G1_SPEC.md", G1_RESULT):
        assert "G1_INDETERMINATE" in open(path).read(), path


# --- the prospective decision --------------------------------------------


def test_option_three_was_recorded_and_is_forward_only(gov):
    dec = gov["prospective_decision"]
    assert dec["option"] == 3
    assert dec["applies_retroactively"] is False
    assert dec["effective_from"] == "2026-08-19"
    assert dec["corrected_t3_semantics_apply"] == "prospectively only"


def test_the_t3_cutoff_is_the_frozen_one_not_the_decision_date(gov):
    """Two dates live here and an earlier draft conflated them.

    The decision was taken on 2026-08-19. The date that decides *which gates*
    the corrected T3 semantics reach was frozen a day earlier, before the
    decision existed, and taking a decision does not move it. If they are ever
    written as the same date again, one of the documents is wrong.
    """
    cutoff = "gates declared after 2026-08-18"
    assert gov["prospective_decision"]["corrected_t3_semantics_cutoff"] == cutoff
    for path in ("reports/c50/TRIAL_REGISTRY_SPEC.md",
                 "reports/c50/G1_1_RESULT.md",
                 DECISION_DOC,
                 "reports/c50/G1_SPEC.md"):
        text = open(path).read()
        assert "2026-08-18" in text, path
        assert "gates declared after 2026-08-19" not in text, path


def test_the_accepted_apparatus_properties_are_the_five_named(gov):
    assert gov["prospective_decision"]["accepted_properties"] == [
        "leakage controls",
        "purge / embargo / CPCV behaviour",
        "sample-weight and uniqueness behaviour",
        "DSR calibration behaviour",
        "negative-control false-positive-rate behaviour",
    ]


def test_t3_refreeze_is_still_required_and_may_not_use_a_prime(gov):
    dec = gov["prospective_decision"]
    assert dec["t3_refreeze_required_before"]
    assert "must not be chosen using Phase A-prime results" in \
        dec["t3_refreeze_constraint"]


# --- the trial count -----------------------------------------------------


def test_recorded_count_matches_what_the_registry_derives(tmp_path, gov):
    """The recorded number is a witness for a reader. If it ever disagrees
    with the registry, the registry is right and this test fails loudly."""
    registry = TrialRegistry(str(tmp_path / "trials.jsonl"))
    populate(registry)
    count = registry.n_trials(A_PRIME_SCOPE)

    acc = gov["trial_accounting"]
    assert acc["verdict"] == "G1_1_PASS"
    assert acc["confirmed"] == count.confirmed
    assert acc["uncertain"] == count.uncertain
    assert acc["canonical_n_trials"] == count.n_trials
    assert acc["count_basis"] == "conservative"
    assert acc["canonical_scope"] == count.scope


def test_the_canonical_count_is_the_conservative_one(tmp_path, gov):
    registry = TrialRegistry(str(tmp_path / "trials.jsonl"))
    populate(registry)
    count = registry.n_trials(A_PRIME_SCOPE)
    assert count.uncertain > 0, ("if the band ever closes, this record needs "
                                 "rewriting rather than silently reading as "
                                 "precise")
    acc = gov["trial_accounting"]
    assert acc["canonical_n_trials"] > acc["confirmed"]
    assert acc["historical_uncertainty_preserved"]


def test_the_record_says_the_number_is_a_witness_not_an_input(gov):
    note = gov["trial_accounting"]["witness_not_input"]
    assert "never from this file" in note
    assert gov["phase_a_prime"]["entry_conditions"]


# --- the consequence -----------------------------------------------------


def test_phase_a_prime_is_unblocked_but_not_started(gov):
    phase = gov["phase_a_prime"]
    assert phase["status"] == "UNBLOCKED"
    assert phase["started"] is False
