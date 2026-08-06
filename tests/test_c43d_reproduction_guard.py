"""C4.3d: the guard that refuses to interpret a comparison built on top of
Ridge numbers that moved.

If re-running C4.3 does not reproduce its published metrics, the new baseline
comparison is meaningless — it would be measuring a different model. These
tests pin that the guard actually catches drift rather than waving it
through, and that the sealed-holdout accounting is a real count.
"""
from __future__ import annotations

import copy

from tools import ridge_ranking_check as rrc


def _published() -> dict:
    return {
        "verdict": "RIDGE_NOT_BETTER_THAN_SIMPLE_BASELINE",
        "checks": {"beats_persistence_pooled": True, "reproducible": True},
        "holdout": {"idx_lo": 8199, "idx_hi": 9399},
        "folds": [
            {"fold": 0, "confident": True, "n_train": 2708, "n_val": 1354,
             "val_idx_range": [4122, 5476], "chosen_alpha": 10.0,
             "mae_model": 0.39316002144851697, "mae_persistence": 1.245716051986924,
             "mae_rolling_mean_60": 0.3812654962522978,
             "mae_train_mean": 0.3803803704172786,
             "rmse_model": 0.4959946813645346, "spearman": 0.3349024662530767},
            {"fold": 1, "confident": True, "n_train": 2708, "n_val": 1354,
             "val_idx_range": [5476, 6830], "chosen_alpha": 10.0,
             "mae_model": 0.4415, "mae_persistence": 1.2475,
             "mae_rolling_mean_60": 0.4066, "mae_train_mean": 0.4001,
             "rmse_model": 0.55, "spearman": 0.396},
        ],
        "pooled": {"n": 4062, "mae_model": 0.4014, "mae_persistence": 1.2459,
                   "mae_rolling_mean_60": 0.3868, "mae_train_mean": 0.3803,
                   "rmse_model": 0.50, "spearman": 0.307},
    }


def test_identical_run_reproduces():
    pub = _published()
    assert rrc.compare_to_published(copy.deepcopy(pub), pub) == []


def test_a_moved_fold_metric_is_caught():
    pub = _published()
    fresh = copy.deepcopy(pub)
    fresh["folds"][0]["spearman"] = 0.3349024662530767 + 1e-10
    diffs = rrc.compare_to_published(fresh, pub)
    assert len(diffs) == 1 and "fold 0.spearman" in diffs[0]


def test_a_moved_pooled_metric_is_caught():
    pub = _published()
    fresh = copy.deepcopy(pub)
    fresh["pooled"]["mae_model"] = 0.4015
    diffs = rrc.compare_to_published(fresh, pub)
    assert any("pooled.mae_model" in d for d in diffs)


def test_tolerance_is_tight_enough_to_catch_real_drift():
    """Ridge's solve is deterministic, so the tolerance only absorbs JSON
    float round-tripping — a change in the 12th decimal must still fail."""
    pub = _published()
    fresh = copy.deepcopy(pub)
    fresh["pooled"]["spearman"] = 0.307 + 1e-11
    assert rrc.compare_to_published(fresh, pub)

    within = copy.deepcopy(pub)
    within["pooled"]["spearman"] = 0.307 + 1e-15
    assert rrc.compare_to_published(within, pub) == []


def test_changed_verdict_or_checks_are_caught():
    pub = _published()
    fresh = copy.deepcopy(pub)
    fresh["verdict"] = "RIDGE_PASSES"
    fresh["checks"]["beats_persistence_pooled"] = False
    diffs = rrc.compare_to_published(fresh, pub)
    assert any(d.startswith("verdict:") for d in diffs)
    assert any("checks.beats_persistence_pooled" in d for d in diffs)


def test_a_vanished_fold_is_caught():
    pub = _published()
    fresh = copy.deepcopy(pub)
    fresh["folds"] = fresh["folds"][:1]
    assert any("fold set changed" in d for d in rrc.compare_to_published(fresh, pub))


def test_holdout_rows_counted_as_zero_for_sound_geometry():
    fresh = _published()
    fresh["pooled"]["c43d"] = {"pooled": {"spearman_advantage": 0.02},
                               "year_slices": {}}
    for f in fresh["folds"]:
        f["c43d"] = {"spearman_advantage": 0.01}
    assert rrc.extract_ranking_inputs(fresh)["holdout_rows_used"] == 0


def test_a_validation_window_reaching_into_the_holdout_is_counted():
    fresh = _published()
    fresh["pooled"]["c43d"] = {"pooled": {"spearman_advantage": 0.02},
                               "year_slices": {}}
    for f in fresh["folds"]:
        f["c43d"] = {"spearman_advantage": 0.01}
    fresh["folds"][1]["val_idx_range"] = [8000, 8300]  # 101 rows past 8199
    assert rrc.extract_ranking_inputs(fresh)["holdout_rows_used"] == 101


def test_unprovable_holdout_exclusion_fails_the_gate():
    fresh = _published()
    fresh["pooled"]["c43d"] = {"pooled": {"spearman_advantage": 0.02},
                               "year_slices": {}}
    for f in fresh["folds"]:
        f["c43d"] = {"spearman_advantage": 0.01}
    fresh["holdout"] = {}
    inputs = rrc.extract_ranking_inputs(fresh)
    assert inputs["holdout_rows_used"] == -1

    from tools.ridge_volatility_run import decide_ranking_verdict
    verdict, checks = decide_ranking_verdict(
        fold_advantages=[0.05, 0.05, 0.05], pooled_advantage=0.04,
        year_advantages=[0.03, 0.03, 0.03],
        holdout_rows_used=inputs["holdout_rows_used"], reproduced=True)
    assert checks["no_holdout_rows"] is False
    assert verdict == "RIDGE_RANKING_VALUE_REJECTED"


def test_only_confident_folds_and_qualifying_years_feed_the_rule():
    fresh = _published()
    fresh["folds"][0]["c43d"] = {"spearman_advantage": 0.05}
    fresh["folds"][1]["confident"] = False
    fresh["folds"][1]["c43d"] = {"spearman_advantage": 0.99}
    fresh["pooled"]["year_slices"] = {"2024": {"n": 2000}, "2025": {"n": 10}}
    fresh["pooled"]["c43d"] = {
        "pooled": {"spearman_advantage": 0.02},
        "year_slices": {"2024": {"spearman_advantage": 0.03},
                        "2025": {"spearman_advantage": 0.9}},
    }
    inputs = rrc.extract_ranking_inputs(fresh)
    assert inputs["fold_advantages"] == [0.05], "unconfident fold leaked in"
    assert inputs["year_advantages"] == [0.03], "under-sized year leaked in"
