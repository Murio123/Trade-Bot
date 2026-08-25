"""The spot track's stage-close record must keep two things apart.

What was measured is fixed. What happens next is open. The failure this guards
against is the ordinary one: months later, a project that moved on reads its
own record as if the decision had already been made, or as if the target it
abandoned had never failed. Prose alone does not stop that; these do.
"""
from __future__ import annotations

import json

import pytest

GOVERNANCE = "reports/s3/spot_governance.json"
STATUS = "reports/s3/SPOT_TRACK_STATUS.md"


@pytest.fixture(scope="module")
def gov() -> dict:
    with open(GOVERNANCE) as fh:
        return json.load(fh)


# --- the measured facts ---------------------------------------------------


def test_the_stage_verdicts_are_the_published_ones(gov):
    stages = gov["established_facts"]["stages"]
    assert stages["S1"]["verdict"] == "S1_PASS"
    assert stages["S2"]["verdict"] == "S2_PASS"
    assert stages["S3"]["verdict"] == "S3_NO_SIGNAL"
    assert gov["established_facts"]["immutable"] is True


def test_the_record_matches_what_s3_actually_measured(gov):
    s3 = gov["established_facts"]["s3"]
    with open("reports/s3/s3_results.json") as fh:
        results = json.load(fh)
    assert s3["dates"] == results["dates"]
    families = results["families"]
    assert s3["independent_blocks"] == \
        families["B4"]["overall"]["rate_difference"]["blocks"]
    for fid, verdict in s3["classifications"].items():
        assert families[fid]["classification"] == verdict, fid
    assert s3["no_family_earned_keep"] is True
    assert not any(v == "PRELIMINARY_KEEP"
                   for v in s3["classifications"].values())


def test_no_rule_is_recorded_as_beating_the_universe(gov):
    low, high = gov["established_facts"]["s3"]["lift_range"]
    assert high < 1.0, "a lift at or above 1 would contradict S3_NO_SIGNAL"
    assert low <= high


def test_the_spot_family_count_is_its_own(gov):
    reg = gov["established_facts"]["trial_registry"]
    assert reg["n_trials"] == 6
    assert reg["futures_family_n_trials"] == 100
    assert reg["families_are_not_mixed"] is True
    assert reg["family"]["research_objective"] == "spot_2x_discovery"


def test_the_post_hoc_finding_is_labelled_as_one(gov):
    """The most dangerous sentence in the whole track.

    F4 ranking continuous returns is real and was found after the fact. If it
    ever reads as a validated hypothesis, someone will build on it without a
    registration.
    """
    obs = gov["established_facts"]["post_hoc_observation"]
    assert obs["status"] == "OBSERVATION_NOT_HYPOTHESIS"
    assert "fresh registration" in obs["constraint"]


def test_the_corrected_survivorship_gap_is_recorded_with_its_history(gov):
    panel = gov["established_facts"]["panel"]
    assert panel["survivorship_gap_pp"] == 0.04
    assert "2.68" in panel["survivorship_gap_note"], (
        "the superseded figure must stay visible; a corrected number that "
        "hides what it corrected is not checkable")


# --- the open decision ----------------------------------------------------


def test_the_decision_is_recorded_as_not_taken(gov):
    decision = gov["open_decision"]
    assert decision["taken"] is False
    assert decision["decided_by"] == "project owner"
    assert set(decision["options"]) == {"A", "B", "C"}


def test_the_recommendation_binds_nothing(gov):
    rec = gov["open_decision"]["implementer_recommendation"]
    assert rec["binding"] is False
    assert rec["conditions_on_C"], "re-scoping without conditions is the trap"


def test_the_trap_is_named_and_points_at_the_precedent(gov):
    trap = gov["open_decision"]["named_trap"]
    assert "G1_SPEC" in trap


def test_s4_is_not_started(gov):
    constraints = " ".join(gov["binding_constraints"])
    assert "S4 is not started" in constraints
    assert "no combined score" in constraints
    assert "Phase A-prime of the futures track remains paused" in constraints


def test_the_status_document_states_the_verdict_and_the_open_decision():
    text = open(STATUS).read()
    assert "S3_NO_SIGNAL" in text
    assert "not taken" in text
    assert "G1_INDETERMINATE" in text, (
        "the futures verdict must not quietly change while the spot track "
        "moves on")


# --- no combiner exists ---------------------------------------------------


def test_the_spot_package_still_contains_no_feature_combiner():
    """§3 constraint 2, asserted rather than promised.

    A combined score is the one thing S3 was built to avoid producing, and the
    cheapest way for it to appear is a helpful-looking utility function.
    """
    import pathlib
    forbidden = ("combine_features", "composite_score", "weighted_score",
                 "confluence", "total_score")
    for path in pathlib.Path("spot").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in src, f"{path}: {name}"


# --- operational repository state ----------------------------------------


def test_the_push_blocker_is_recorded_as_resolved(gov):
    """An administrative fact, kept honest the same way the research ones are.

    The 403 really happened and the record keeps saying so; what changed is
    that it no longer describes the present. A record that quietly deletes a
    blocker is as unreadable later as one that never notices it was lifted.
    """
    repo = gov["repository"]
    assert repo["pushed"] is True
    assert repo["commits_ahead_of_origin"] == 0
    assert repo["push_blocker"] is None
    history = repo["push_blocker_history"]
    assert history["status"] == "RESOLVED"
    assert "403" in history["what"]


def test_resolving_the_push_blocker_changed_no_research_fact(gov):
    """The correction is operational. Nothing it touched may have moved."""
    facts = gov["established_facts"]
    assert facts["immutable"] is True
    assert facts["stages"]["S3"]["verdict"] == "S3_NO_SIGNAL"
    assert facts["target"]["base_rate"] == 0.2085
    assert facts["trial_registry"]["n_trials"] == 6


def test_the_status_document_marks_the_blocker_resolved_without_erasing_it():
    text = open(STATUS).read()
    assert "RESOLVED" in text
    assert "HTTP 403" in text, (
        "the blocker that was lifted must stay legible; a record that deletes "
        "what it corrected cannot be checked")
