"""G1.1 — the reconstructed history of research trials in this project.

Every record below is a research choice that was actually made, with the
evidence that establishes it. Nothing here is invented: TRIAL_REGISTRY_SPEC.md
§8.2 forbids creating a trial for a choice that only *seems* likely to have
happened, and where the evidence supports a range rather than a number, the
record carries the range (`multiplicity`) instead of a false precision.

This is a curated table, not a scraper over the working tree, and that is
deliberate (§8.4). Most of this project's research artifacts live in untracked
`reports/<stage>/` directories, so a scraper would produce a different count on
a different machine depending on which of them happen to be present — the
opposite of auditable. A table is code: it is reviewed as code, it materializes
byte-identically on any checkout, and each row states what established it.

Two things this table is honest about, because both cut against a smaller
count:

- **Abandoned work counts.** H1 was rejected, LightGBM never ran, C4.3c
  concluded Ridge was not a usable point predictor. All are trials. A research
  process that only counted its survivors would be counting the winners of a
  selection while pretending no selection took place.
- **Post-hoc variants count double in spirit.** C4.3c, C4.3d and C4.3e each
  exist because the previous stage's numbers were seen first. They are linked
  by `parent_trial_id`, and §6.2 pulls a lineage into any family that counts
  one of its members.

The gaps are recorded in `reports/c50/G1_1_RESULT.md`, not papered over here.
"""
from __future__ import annotations

from typing import Any

from validation.trial_registry import (ORIGIN_RECONSTRUCTED,
                                       STATUS_ABANDONED, STATUS_COMPLETED,
                                       STATUS_INSUFFICIENT_DATA, Evidence,
                                       TrialIdentity, TrialRecord,
                                       TrialRegistry, declare_all)

HISTORY_VERSION = "g11_trial_history_v1"

SYMBOL = "BTCUSDT"
BINANCE_4H = "binance:BTCUSDT:klines:8000bars"
BINANCE_LIVE = "binance:BTCUSDT:live_backtest_window"

OBJ_TRADE = "trade_expectancy"
OBJ_VOL = "volatility_forecast"

FAM_TRADE_R = "trade_r"
FAM_VOL_RATIO = "volatility_ratio"


def _t(*, stage: str, hypothesis: str, created_at: str,
       objective: str = OBJ_TRADE, profile: str | None = None,
       target_family: str | None = FAM_TRADE_R,
       dataset: str | None = BINANCE_LIVE,
       parameters: Any = None, features: Any = None, target: Any = None,
       horizon_bars: int | None = None, criterion: Any = None,
       spec: Any = None, evidence: tuple[Evidence, ...] = (),
       multiplicity: tuple[int, int] = (1, 1),
       status: str = STATUS_COMPLETED, parent: str | None = None,
       notes: str = "") -> TrialRecord:
    identity = TrialIdentity.build(
        research_objective=objective, research_stage=stage,
        hypothesis=hypothesis, profile=profile, symbol=SYMBOL,
        dataset_contract=dataset, parameters=parameters, features=features,
        target=target, horizon_bars=horizon_bars, criterion=criterion,
        spec=spec)
    return TrialRecord(
        identity=identity, created_at=created_at,
        origin=ORIGIN_RECONSTRUCTED, status=status,
        target_family=target_family, parent_trial_id=parent,
        evidence=evidence, multiplicity=multiplicity, notes=notes)


def _git(ref: str, note: str) -> Evidence:
    return Evidence(tier="git_history", ref=ref, note=note)


def _report(ref: str, note: str) -> Evidence:
    return Evidence(tier="local_report", ref=ref, note=note)


def _doc(ref: str, note: str) -> Evidence:
    return Evidence(tier="committed_document", ref=ref, note=note)


def _inferred(ref: str, note: str) -> Evidence:
    return Evidence(tier="inference", ref=ref, note=note)


# ---------------------------------------------------------------------------
# Era 1 — pre-C1 strategy construction (2026-06-30 .. 2026-07-10)
#
# The bot's scoring model, confluence set and profile lineup were assembled
# here, each component kept or dropped against `/backtest` output. The commit
# messages establish the choice; only some of them establish that a measurement
# decided it, so the rest are `inference` and sit in the uncertain band.
# ---------------------------------------------------------------------------

_ERA1 = [
    _t(stage="pre-C1", hypothesis="score_threshold_sweep",
       created_at="2026-06-30", profile=None,
       parameters={"sweep": "score_threshold", "selected": "most_profitable"},
       evidence=(_git("da6f12b",
                      "'Backtest: sweep score thresholds and recommend the "
                      "most profitable' — a selection over thresholds, stated "
                      "in the commit subject"),),
       multiplicity=(1, 6),
       notes="Arm count for this early sweep is not recorded; the later "
             "shipped grid is 6 wide (tools/deep_backtest.py THRESHOLDS), "
             "which is the upper bound used here."),
    _t(stage="pre-C1", hypothesis="confluence_fvg_bbw_macd_divergence",
       created_at="2026-06-30", profile=None,
       parameters={"components": ["fvg", "bbw", "macd_divergence"]},
       evidence=(_git("8b91577", "three scoring components added together"),
                 _inferred("8b91577",
                           "that each was kept on evidence is implied by the "
                           "project's backtest-then-keep workflow, not "
                           "documented per component")),
       multiplicity=(1, 3),
       notes="Three components, one commit. Low bound treats the commit as a "
             "single choice; high bound treats each component as its own."),
    _t(stage="pre-C1", hypothesis="weighted_score_model_with_weak_tier",
       created_at="2026-06-30", profile=None,
       parameters={"model": "weighted", "tiers": ["weak", "normal", "strong"]},
       evidence=(_git("a7d6c59",
                      "the deep-analysis command's scoring was reworked to a "
                      "weighted model with a WEAK tier, replacing the scoring "
                      "function outright"),)),
    _t(stage="pre-C1", hypothesis="reversal_exhaustion_detection",
       created_at="2026-06-30", profile=None,
       parameters={"component": "exhaustion"},
       evidence=(_git("368b278", "reversal/exhaustion detection added"),
                 _inferred("368b278", "retention decision not documented"))),
    _t(stage="pre-C1", hypothesis="mtf_agreement_expansion",
       created_at="2026-06-30", profile=None,
       parameters={"swing": ["1H", "4H", "12H", "1D"],
                   "intraday": ["15m", "1H", "4H", "12H"]},
       evidence=(_git("abc0094",
                      "the multi-timeframe agreement set was widened for both "
                      "profiles — a change to the confluence specification"),)),
    _t(stage="pre-C1", hypothesis="htf_zones_ltf_entry_structural_targets",
       created_at="2026-06-30", profile="swing",
       parameters={"entry": "ltf", "zones": "htf", "targets": "htf_structure"},
       evidence=(_git("aee7f48",
                      "'Swing quality: HTF zones + LTF entry, and "
                      "HTF-structure targets'"),)),
    _t(stage="pre-C1", hypothesis="equilibrium_premium_discount_confluence",
       created_at="2026-06-30", profile=None,
       parameters={"component": "equilibrium_premium_discount"},
       evidence=(_git("768783d",
                      "premium/discount confluence added to both profiles"),)),
    _t(stage="pre-C1", hypothesis="swing_hourly_cadence",
       created_at="2026-06-30", profile="swing",
       parameters={"entry_tf": "1H"},
       evidence=(_git("db4985d", "swing entry cadence moved to 1H"),
                 _inferred("db4985d", "no recorded comparison against the "
                                      "previous cadence"))),
    _t(stage="pre-C1", hypothesis="intraday_htf_filter_1h",
       created_at="2026-06-30", profile="intraday",
       parameters={"htf": "1H", "entry_tf": "15m"},
       evidence=(_git("5f7a03b",
                      "'Intraday: trend by 1H (HTF filter), entry stays 15m'"),)),
    _t(stage="pre-C1", hypothesis="per_profile_r_multiples",
       created_at="2026-07-01", profile=None,
       parameters={"tuned": ["target_r", "stop_r"]},
       evidence=(_git("cb17c14",
                      "'Make targets/stop R-multiples configurable per "
                      "profile (win-rate tuning)' — the commit subject names "
                      "the tuning objective"),),
       multiplicity=(1, 4),
       notes="Tuned per profile; the number of configurations actually "
             "compared is not recorded. Upper bound is the four profiles."),
    _t(stage="pre-C1", hypothesis="backtest_replicates_live_profiles",
       created_at="2026-07-01", profile=None,
       parameters={"backtest": "profile_replicating"},
       evidence=(_git("bd4aeb8",
                      "the backtest was rewritten to replicate live profiles, "
                      "which re-measured every prior retention decision"),)),
    _t(stage="pre-C1", hypothesis="backtest_window_800_bars",
       created_at="2026-07-01", profile="intraday",
       parameters={"bars": 800},
       evidence=(_git("1cd73c2",
                      "'Widen backtest window to 800 bars for a larger "
                      "intraday sample' — results were re-read on the wider "
                      "window"),)),
    _t(stage="pre-C1", hypothesis="dead_zone_and_crowded_funding_vetoes",
       created_at="2026-07-01", profile=None,
       parameters={"vetoes": ["dead_zone", "crowded_funding"]},
       evidence=(_git("10ed7e0",
                      "'Signal quality: closed-candle analysis + dead-zone "
                      "and crowded-funding vetoes'"),),
       multiplicity=(1, 2),
       notes="Two independent vetoes in one commit; whether they were "
             "evaluated together or separately is not recorded, so the band "
             "spans both readings."),
    _t(stage="pre-C1", hypothesis="swing_structural_stops",
       created_at="2026-07-01", profile="swing",
       parameters={"stop": "structural"},
       evidence=(_git("6fd3ce4",
                      "'Swing behaves like swing again: structural stops + "
                      "signal continuity'"),)),
    _t(stage="pre-C1", hypothesis="intraday_stops_on_1h_atr",
       created_at="2026-07-02", profile="intraday",
       parameters={"stop_anchor": "1H_ATR"},
       evidence=(_git("f4e5294",
                      "'Intraday stops anchored to 1H ATR (fee viability)' — "
                      "the parenthetical states the measured objective"),)),
    _t(stage="pre-C1", hypothesis="decorrelation_regime_weights_no_trade_gates",
       created_at="2026-07-02", profile=None,
       parameters={"changes": ["decorrelation", "regime_weights",
                               "no_trade_gates", "swing_4h"]},
       evidence=(_git("4fc8765",
                      "'Rework analysis quality: decorrelation, regime "
                      "weights, NO_TRADE gates, 4H swing' — four scoring "
                      "changes shipped after an audit"),),
       multiplicity=(1, 4),
       notes="Four scoring changes landed in one commit. The low bound treats "
             "the audit as a single decision, the high bound treats each "
             "change as its own."),
    _t(stage="pre-C1", hypothesis="min_expected_move_quality_filter",
       created_at="2026-07-02", profile=None,
       parameters={"filter": "min_expected_move"},
       evidence=(_git("76880fd",
                      "'Add per-mode analysis requirements: min expected move "
                      "as a quality filter'"),)),
    _t(stage="pre-C1", hypothesis="position_profile_4h",
       created_at="2026-07-02", profile="position",
       parameters={"entry_tf": "4H", "target_points": [2000, 5000]},
       evidence=(_git("70f1aef",
                      "'Add position stream: 4H entries targeting multi-day "
                      "2000-5000pt moves' — a fourth profile"),)),
    _t(stage="pre-C1", hypothesis="bounce_profile_countertrend",
       created_at="2026-07-10", profile="bounce",
       parameters={"direction": "counter_trend", "enabled": False},
       status=STATUS_ABANDONED,
       evidence=(_git("8843bf9", "'Add disabled bounce profile'"),
                 _report("reports/c14/C1.5_interpretation_memo.md",
                         "every threshold negative, sign_consistency 0.00; "
                         "disposition 'stay disabled'")),
       notes="Shipped disabled and later confirmed negative. Abandoned work "
             "counts (ARCHITECTURE.md §5.4)."),
]

# ---------------------------------------------------------------------------
# Era 2 — C1.4 .. C1.8, the audit of what had been built
# ---------------------------------------------------------------------------

_C14_GRID = [
    _t(stage="C1.4a", hypothesis="score_threshold_grid",
       created_at="2026-07-11", profile=profile, dataset=BINANCE_4H,
       parameters={"thresholds": [5, 6, 7, 8, 9, 10], "modes": ["rolling",
                                                                "expanding"]},
       criterion={"rule": "walk_forward_validation_expectancy"},
       evidence=(_report("reports/c14/README.md",
                         "16 walk-forward reports: 4 profiles x {rolling, "
                         "expanding} x {json, txt}"),
                 _report("reports/c14/C1.5_interpretation_memo.md",
                         "§3 tabulates validation expectancy for every "
                         "(profile, threshold) cell"),
                 _doc("tools/deep_backtest.py:123",
                      "THRESHOLDS = [5, 6, 7, 8, 9, 10] — the grid is six "
                      "wide and committed")),
       multiplicity=(12, 12),
       notes="Six thresholds x two fold modes, per profile. Both modes are "
             "counted: the memo's finding that mode does not change any "
             "verdict is a result, and results do not retroactively make the "
             "arms that produced them free.")
    for profile in ("swing", "position", "intraday", "bounce")
]

_ERA2 = _C14_GRID + [
    _t(stage="C1.6", hypothesis="profile_root_cause_disposition",
       created_at="2026-07-13", profile=profile, dataset=BINANCE_4H,
       parameters={"analysis": "root_cause_diagnostics"},
       criterion={"outcomes": ["targeted_fix_candidate",
                               "structural_negative"]},
       evidence=(_report("reports/c16/C1.6_consolidated_summary.md",
                         "headline table assigns each profile a disposition "
                         "that decided whether it stayed in scope"),),
       status=STATUS_COMPLETED)
    for profile in ("swing", "position", "intraday", "bounce")
] + [
    _t(stage="C1.8", hypothesis="single_feature_attribution",
       created_at="2026-07-21", profile="swing", dataset=BINANCE_4H,
       parameters={"screen": "single_feature", "n_features": 32},
       criterion={"labels": ["PRELIMINARY_KEEP", "PRELIMINARY_REMOVE",
                             "PRELIMINARY_UNKNOWN", "DIAGNOSTIC_ONLY"]},
       evidence=(_report("reports/c18/single_feature_report.json",
                         "'features' holds exactly 32 entries and "
                         "'summary_table' 32 rows"),
                 _report("reports/c18/SWING_SHORTLIST.md",
                         "every one of the 32 is assigned a keep/remove/"
                         "unknown status — a selection decision per feature"),
                 _git("9a27a0a", "'Add C1.8 single-feature attribution'")),
       multiplicity=(32, 32),
       notes="Each feature was screened and labelled, so each is an arm of "
             "the same selection. This is the single largest confirmed block "
             "in the reconstruction and the one most often left uncounted."),
    _t(stage="C1.8", hypothesis="feature_redundancy_analysis",
       created_at="2026-07-21", profile="swing", dataset=BINANCE_4H,
       parameters={"screen": "redundancy"},
       evidence=(_report("reports/c18/redundancy_report.json", "committed run"),
                 _git("7eade81", "'Add C1.8 redundancy analysis and swing "
                                 "candidate shortlist (step 5)'"))),
]

# ---------------------------------------------------------------------------
# Era 3 — C2.x, discovery and the two frozen hypotheses
# ---------------------------------------------------------------------------

_C2A = [
    _t(stage="C2a", hypothesis="unsupervised_archetype_clustering",
       created_at="2026-07-21", profile="swing", dataset=BINANCE_4H,
       parameters={"method": "kmeans", "k_range": [2, 8],
                   "restarts": 8, "selection": "silhouette"},
       evidence=(_report("reports/c20/archetype_discovery.md",
                         "'manual k-means + silhouette, k=2..8, 8 random "
                         "restarts each'; §2 records two independent "
                         "clusterings"),),
       multiplicity=(14, 14),
       notes="Two clusterings x seven k values = 14 arms. The 8 restarts per "
             "k are reruns of one specification under different seeds and are "
             "NOT counted (TRIAL_REGISTRY_SPEC.md §3.3) — the seed was not "
             "selected over, the silhouette across k was."),
]

_C21_GATES = [
    _t(stage="C2.1", hypothesis=name, created_at="2026-07-21",
       profile="swing", dataset=BINANCE_4H,
       parameters={"gate": name},
       criterion={"checks": ["beats_unfiltered", "net_positive",
                             "fold_majority", "year_independence",
                             "retains_trades", "beats_random_p95"]},
       evidence=(_report("reports/c21/DECISION.md",
                         "three named gate variants tabulated side by side "
                         "with full-sample results for each"),
                 _git("69977f1", "'Add C2.1 swing regime gate confirmation'")))
    for name in ("A_baseline", "B_range_gate", "C_trend_up_exclusion_only")
]

_HYPOTHESES = [
    _t(stage="C2.2b", hypothesis="H1_trend_pullback_continuation",
       created_at="2026-07-21", profile="swing", dataset=BINANCE_4H,
       parameters={"structural_window": 40, "stop_atr_buffer": 0.25,
                   "min_rr": 1.5, "volatility_reject_above_pct": 90,
                   "cost_r_ceiling": 0.15},
       spec={"stage": "C2.2a", "frozen": True},
       evidence=(_report("reports/c22/implementation_summary.md",
                         "H1's frozen constants tabulated; 'No constant was "
                         "changed after seeing any result'"),
                 _git("f8c1e2a", "'Add C2.2b offline simulator for swing "
                                 "hypotheses H1/H2'")),
       status=STATUS_INSUFFICIENT_DATA,
       notes="0 of 7999 bars taken in the smoke run. Counts in full: a "
             "hypothesis that produced no trades still consumed a look."),
    _t(stage="C2.2b", hypothesis="H2_range_mean_reversion",
       created_at="2026-07-21", profile="swing", dataset=BINANCE_4H,
       parameters={"structural_window": 40, "stop_atr_buffer": 0.30,
                   "volatility_band_pct": [10, 85], "cost_r_ceiling": 0.10},
       spec={"stage": "C2.2a", "frozen": True},
       evidence=(_report("reports/c22/implementation_summary.md",
                         "H2's frozen constants tabulated"),
                 _git("f8c1e2a", "same commit as H1"))),
]

_ERA3 = _C2A + _C21_GATES + _HYPOTHESES

_H1_WF = _t(
    stage="C2.2c", hypothesis="H1_trend_pullback_continuation",
    created_at="2026-07-22", profile="swing", dataset=BINANCE_4H,
    parameters={"structural_window": 40, "stop_atr_buffer": 0.25,
                "min_rr": 1.5, "volatility_reject_above_pct": 90,
                "cost_r_ceiling": 0.15},
    spec={"stage": "C2.2a", "frozen": True},
    criterion={"geometry": "walk_forward_purge_embargo",
               "verdict_space": ["proceed", "insufficient_evidence",
                                 "reject"]},
    status=STATUS_ABANDONED,
    evidence=(_report("reports/c23/DECISION.md",
                      "verdict REJECT_FROZEN_SPECIFICATION"),
              _git("76648ea", "'Add C2.2c walk-forward evaluation of frozen "
                              "swing hypotheses H1/H2'")),
    notes="Same frozen spec as the C2.2b arm, but a different acceptance "
          "criterion that could and did move the decision, so §3.1's "
          "criterion_hash makes it a separate trial rather than a rerun.")

_H2_WF = _t(
    stage="C2.2c", hypothesis="H2_range_mean_reversion",
    created_at="2026-07-22", profile="swing", dataset=BINANCE_4H,
    parameters={"structural_window": 40, "stop_atr_buffer": 0.30,
                "volatility_band_pct": [10, 85], "cost_r_ceiling": 0.10},
    spec={"stage": "C2.2a", "frozen": True},
    criterion={"geometry": "walk_forward_purge_embargo",
               "checks": 12, "threshold": "stability"},
    status=STATUS_ABANDONED,
    evidence=(_report("reports/c23/DECISION.md",
                      "verdict REJECT_HYPOTHESIS on 7/12 stability checks"),
              _git("76648ea", "same commit as the H1 walk-forward")))

_C23_BASELINES = [
    _t(stage="C2.2c", hypothesis=name, created_at="2026-07-22",
       profile="swing", dataset=BINANCE_4H,
       parameters={"baseline": name},
       evidence=(_report("reports/c23/baseline_comparison.json",
                         "each baseline was computed and H2 was compared "
                         "against it as a pass/fail check"),
                 _report("reports/c22/implementation_summary.md",
                         "the smoke run reports regime_direction and "
                         "random_direction baseline expectancies")),
       notes="A baseline that a hypothesis must beat is part of the selection "
             "apparatus, and choosing which baselines to require is itself a "
             "research choice.")
    for name in ("current_swing_baseline", "regime_direction_baseline",
                 "random_direction_baseline", "matched_coverage_random")
]

_ERA4 = [_H1_WF, _H2_WF] + _C23_BASELINES + [
    _t(stage="C3.0", hypothesis="position_profile_feasibility",
       created_at="2026-07-22", profile="position", dataset=BINANCE_4H,
       parameters={"analysis": "feasibility"},
       evidence=(_report("reports/c30/position_feasibility.md",
                         "a standalone feasibility verdict for the position "
                         "stream"),)),
]

# ---------------------------------------------------------------------------
# Era 5 — C4.x, the volatility model. A different objective and a different
# target family, so these do not enter a trade-expectancy count except through
# the lineage rule.
# ---------------------------------------------------------------------------

_RIDGE = _t(
    stage="C4.3", hypothesis="ridge_volatility_range_model",
    created_at="2026-07-23", objective=OBJ_VOL, profile=None,
    target_family=FAM_VOL_RATIO, dataset=BINANCE_4H,
    target={"quantity": "atr_normalized_realized_range"}, horizon_bars=12,
    features={"spec": "C4.1 frozen feature set"},
    parameters={"estimator": "ridge", "alpha_candidates": [0.1, 1.0, 10.0]},
    spec={"doc": "reports/c41/volatility_model_frozen_spec.md"},
    criterion={"checks": ["beats_persistence_pooled", "fold_majority",
                          "no_single_fold_dominance", "year_independence",
                          "residual_bias_ok", "reproducible"]},
    evidence=(_report("reports/c43/RIDGE_DECISION.md",
                      "verdict RIDGE_NEEDS_MORE_EVIDENCE against the six "
                      "frozen go/no-go checks"),
              _git("f806c0d", "'Add C4.3 ridge volatility/range model and "
                              "walk-forward run'")))

_ALPHA = _t(
    stage="C4.3", hypothesis="ridge_alpha_selection",
    created_at="2026-07-23", objective=OBJ_VOL, profile=None,
    target_family=FAM_VOL_RATIO, dataset=BINANCE_4H,
    target={"quantity": "atr_normalized_realized_range"}, horizon_bars=12,
    parameters={"alpha_candidates": [0.1, 1.0, 10.0],
                "selection": "inner_train_only_split"},
    evidence=(_doc("reports/c41/volatility_model_frozen_spec.md:305",
                   "'nested alpha selection (frozen procedure)': three frozen "
                   "alpha candidates, lowest inner-validation MAE wins"),),
    multiplicity=(3, 3),
    notes="Counted even though selection happened on an inner train-only "
          "split that never touched the outer validation fold. §2.4 resolves "
          "the ambiguity upward; the mitigation is real and is recorded here "
          "rather than used to zero the arms.")

_LIGHTGBM = _t(
    stage="C4.3b", hypothesis="lightgbm_challenger",
    created_at="2026-07-23", objective=OBJ_VOL, profile=None,
    target_family=FAM_VOL_RATIO, dataset=BINANCE_4H,
    target={"quantity": "atr_normalized_realized_range"}, horizon_bars=12,
    parameters={"estimator": "lightgbm", "early_stopping_rounds": 20},
    spec={"doc": "reports/c41/volatility_model_frozen_spec.md"},
    status=STATUS_ABANDONED,
    evidence=(_report("reports/c43/RIDGE_DECISION.md",
                      "'LightGBM status: DEFERRED_NOT_RUN_THIS_STAGE'"),
              _report("reports/c43b/implementation_summary.md",
                      "the challenger stage, blocked on libomp")),
    notes="Declared in the frozen C4.1 spec as ridge's challenger, then "
          "blocked and never run. A declared-but-unrun arm still consumed a "
          "look: had it run and won, it would have been selected.")

_C43C = _t(
    stage="C4.3c", hypothesis="ridge_vs_simple_baselines",
    created_at="2026-08-09", objective=OBJ_VOL, profile=None,
    target_family=FAM_VOL_RATIO, dataset=BINANCE_4H,
    target={"quantity": "atr_normalized_realized_range"}, horizon_bars=12,
    parameters={"baselines": ["persistence", "rolling_mean_60", "train_mean"]},
    criterion={"question": "is_ridge_useful_as_point_predictor"},
    evidence=(_report("reports/c43c/ridge_utility_decision.md",
                      "'Every number below is read directly from the existing "
                      "C4.3 artifacts' — a new question asked of results "
                      "already seen"),),
    parent=_RIDGE.trial_id,
    notes="Post-hoc variant under §2.1: it exists because C4.3's numbers were "
          "seen first.")

_C43D = _t(
    stage="C4.3d", hypothesis="ridge_ranking_vs_time_varying_baseline",
    created_at="2026-08-10", objective=OBJ_VOL, profile=None,
    target_family=FAM_VOL_RATIO, dataset=BINANCE_4H,
    target={"quantity": "atr_normalized_realized_range"}, horizon_bars=12,
    parameters={"baseline": "trailing_mean_60_lagged_12",
                "statistic": "spearman"},
    criterion={"verdict_space": ["RIDGE_RANKING_VALUE_CONFIRMED", "other"]},
    evidence=(_report("reports/c43d/DECISION.md",
                      "verdict RIDGE_RANKING_VALUE_CONFIRMED; §'Why this "
                      "comparison was needed' states it follows from C4.3c's "
                      "rejection"),
              _git("ea5d459", "'C4.3d: measure Ridge's ranking against a "
                              "baseline that can actually rank'")),
    parent=_C43C.trial_id,
    notes="Post-hoc variant of a post-hoc variant. The statistic was changed "
          "to one Ridge could win on after MAE was lost — legitimate as "
          "analysis, but it is exactly the selection pressure DSR exists to "
          "deflate, and the lineage rule (§6.2) makes it inescapable.")

_C43E = _t(
    stage="C4.3e", hypothesis="ridge_ranking_significance",
    created_at="2026-08-10", objective=OBJ_VOL, profile=None,
    target_family=FAM_VOL_RATIO, dataset=BINANCE_4H,
    target={"quantity": "atr_normalized_realized_range"}, horizon_bars=12,
    parameters={"tests": ["bootstrap", "permutation"],
                "effective_sample": True},
    evidence=(_report("reports/c43e/ranking_significance.json",
                      "bootstrap and permutation blocks with an explicit "
                      "decision_rule and verdict"),
              _git("8583679", "'C4.3e: test whether Ridge's +0.13 ranking "
                              "edge survives a null'")),
    parent=_C43D.trial_id)

_C44 = _t(
    stage="C4.4", hypothesis="frozen_ranker_forward_forecast",
    created_at="2026-08-10", objective=OBJ_VOL, profile=None,
    target_family=FAM_VOL_RATIO, dataset=BINANCE_4H,
    target={"quantity": "volatility_rank_percentile"}, horizon_bars=12,
    parameters={"mode": "report_only", "horizon_hours": 48},
    evidence=(_doc("reports/c44/STATE.md",
                   "the frozen ranker artifact and its registry pin"),
              _git("8ad489a", "'C4.4 core: report-only 48h volatility ranking "
                              "— ranker, reference, ledger'")),
    parent=_C43E.trial_id,
    notes="Operationalising the surviving signal is the decision the whole "
          "C4.3x lineage was selecting toward, so it carries that lineage.")

_ERA5 = [_RIDGE, _ALPHA, _LIGHTGBM, _C43C, _C43D, _C43E, _C44]

# ---------------------------------------------------------------------------
# Era 6 — C5.0 gates. M00/P1/G1 are apparatus, not strategy selection, with one
# exception recorded below.
# ---------------------------------------------------------------------------

_ERA6 = [
    _t(stage="P1", hypothesis="g0_reevaluation_under_funding_costs",
       created_at="2026-08-17", profile="swing", dataset=BINANCE_4H,
       parameters={"cost_model": "funding_aware",
                   "net_r": "gross - fee - slippage - funding"},
       criterion={"gate": "G0", "verdict_space": ["G0_STILL_PASSES",
                                                  "G0_FAILS"]},
       evidence=(_doc("reports/c50/P1_RESULT.md",
                      "G0_STILL_PASSES, re-read after the cost definition "
                      "changed"),
                 _doc("reports/c50/ARCHITECTURE.md",
                      "§3.1 defines G0 as a gate whose outcome decides "
                      "whether the swing programme continues")),
       notes="The cost correction itself is a definition fix, not a trial "
             "(§2.3). Re-reading the G0 verdict against both the old and the "
             "new number is the decision §2.3's second clause makes a trial."),
]

ALL_TRIALS: tuple[TrialRecord, ...] = tuple(
    _ERA1 + _ERA2 + _ERA3 + _ERA4 + _ERA5 + _ERA6)


def _assert_unique(records: tuple[TrialRecord, ...]) -> None:
    seen: dict[str, TrialRecord] = {}
    for record in records:
        prior = seen.get(record.trial_id)
        if prior is not None:
            raise ValueError(
                f"two history records collide on {record.trial_id}: "
                f"{prior.identity.hypothesis} and {record.identity.hypothesis}")
        seen[record.trial_id] = record


_assert_unique(ALL_TRIALS)


def populate(registry: TrialRegistry) -> list[TrialRecord]:
    """Write the reconstructed history into `registry`, idempotently.

    Safe to run twice: identical declarations are no-ops (§4.2), so replaying
    the table never inflates a count and never rewrites a record.
    """
    return declare_all(registry, ALL_TRIALS)


def missing_from(registry: TrialRegistry) -> tuple[str, ...]:
    """Historical trials that are absent from `registry`.

    The registry's own `seq`/`prev_sha256` chain catches accidental damage and
    naive edits, but it is self-contained: anyone who removes a line and
    recomputes the chain produces a file that validates. No file-local format
    can defend against that, so the defence has to come from outside the file.

    For the reconstructed history it does: this table is code, so the 50
    trial ids are re-derivable, and a removed historical record shows up here
    no matter how carefully the file was rewritten around it.
    """
    present = set(registry.declarations())
    return tuple(r.trial_id for r in ALL_TRIALS if r.trial_id not in present)


def summary() -> dict[str, Any]:
    """Counts by era and by evidence quality, for the result document."""
    by_stage: dict[str, dict[str, int]] = {}
    for record in ALL_TRIALS:
        stage = record.identity.research_stage
        bucket = by_stage.setdefault(stage, {"records": 0, "confirmed": 0,
                                             "conservative": 0})
        bucket["records"] += 1
        bucket["confirmed"] += record.confirmed_low
        bucket["conservative"] += record.conservative_high
    return {
        "history_version": HISTORY_VERSION,
        "n_records": len(ALL_TRIALS),
        "confirmed": sum(r.confirmed_low for r in ALL_TRIALS),
        "conservative": sum(r.conservative_high for r in ALL_TRIALS),
        "by_stage": dict(sorted(by_stage.items())),
    }
