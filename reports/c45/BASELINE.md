# C4.5 — Frozen Research Baseline

**Frozen: 2026-08-11. Immutable.**

This document is never edited. If any fact below changes, a new baseline is
written under a new name and this one is left standing. Its value comes
entirely from not moving.

**Purpose.** P1 makes the cost model funding-aware, which will change every
net number the project has produced. This snapshot exists so that any
difference observed after P1 can be attributed to the funding change and not
to some unrelated drift in the repository, the caches, or the artifacts.

Every number below is **read** from existing artifacts. Nothing was recomputed,
retrained, or re-derived while writing this. No code changed.

---

## 0. Integrity warning — read before using this baseline for attribution

**The working tree is dirty at the moment of freezing.** Five tracked files
carry uncommitted modifications and two new paths are untracked (§1.2). The
baseline therefore describes a state that exists on one machine and in no
commit.

This weakens the stated purpose. Strict attribution of post-P1 differences
requires the pre-P1 state to be reconstructible, and right now it is not: the
uncommitted C4.5 three-category change and the C5.0 M00 module are both part of
what is being frozen.

**Recommended before P1 begins:** commit the tree, then record the resulting
commit hash in the P1 task as the true attribution anchor. This document then
becomes the descriptive companion to that commit rather than the anchor itself.
This limitation is recorded here rather than resolved, because resolving it
would require a commit, and this task is documentation-only.

---

## 1. Repository state

### 1.1 Commit

| | |
|---|---|
| HEAD | `719f8622f655a49804ffefa3179b3487b7cd8e1a` |
| Date | 2026-08-11 12:39:51 +0300 |
| Subject | Route every ledger entry point through one validator, and test the class |
| Branch | `claude/btc-telegram-trading-bot-1avo98` |

### 1.2 Uncommitted at freeze time

Modified (tracked):

- `tools/ridge_freeze_artifact.py` — frozen-write guard (`_write_frozen`,
  `registered_sha256`, `FrozenArtifactError`); registry now read before write
- `tests/test_ridge_freeze_artifact.py` — +9 guard tests
- `volatility/percentile.py` — four categories reduced to three
  (`CATEGORY_SCHEME_VERSION = "c45_three_v1"`)
- `volatility/__init__.py` — docstring updated to three categories
- `tests/test_volatility_core.py` — updated for three categories

Untracked (code):

- `validation/` — `__init__.py`, `cost_sensitivity.py` (C5.0 M00)
- `tests/test_cost_sensitivity.py` — 37 tests

Untracked (reports, per project convention generated reports stay untracked):
`c14 c16 c18 c20 c21 c22 c23 c30 c40 c41 c41a c42 c43 c43b c43c c43d c43e c44
c50 d10 d11 d12 research`

---

## 2. Datasets

All caches: Binance BTCUSDT, single pinned source, `has_taker_buy_base = true`,
`has_gaps = false`, `gap_count = 0`, `missing_bars = 0`,
`duplicate_count_removed = 0`, refreshed 2026-08-11T13:19Z.

| TF | bars | first open (UTC) | last open (UTC) | CSV sha256 |
|---|---|---|---|---|
| 4h | 9400 | 2022-04-28T00:00 | 2026-08-11T12:00 | `7a5bf36d59e55104…` |
| 6h | 6100 | 2022-06-08T18:00 | 2026-08-11T12:00 | `cca8c047ae92c387…` |
| 12h | 3100 | 2022-05-15T00:00 | 2026-08-11T12:00 | `7b86764e26e31a7f…` |
| 1d | 1750 | 2021-10-27T00:00 | 2026-08-11T00:00 | `e55ac385848817d1…` |

Full hashes:

```
7a5bf36d59e551042c21173215829ac3eb229e217ee6defaed215c79f5aa622b  4h
cca8c047ae92c387c694e723f12171a336789979df955a73fff68f30e4fc8dad  6h
7b86764e26e31a7fbc6e231326bfc8218193bd4573f83b6832774c8eeed4c175  12h
e55ac385848817d1869197b86d69fd4863f96844a24d704b2fe4d6ed6785aacb  1d
```

**The 4h cache's last bar was still forming when it was written** (fetched
13:19Z, bar opens 12:00Z and closes 15:59Z). Any point-in-time consumer must
exclude it. This is the condition C4.5 Phase 2 was specified to handle.

### 2.1 Dataset versions

Two distinct `dataset_version` values are live in the artifacts, and the
difference is the whole reason §3 is complicated:

| Value | Origin |
|---|---|
| `98790821553d99286914fb18da20aac02df5c9f273bad25a82344da18a1d5cd4` | 4h cache **before** the 2026-08-11 refresh; the registered v1 artifact was trained on it |
| `7dbef2ab2a1f45a5ad126b623ffea13048bd155cbd12974461d444a48fa3cbf4` | 4h cache **after** the refresh; the quarantined artifact was trained on it |

The pre-refresh 4h CSV no longer exists on disk. `98790821…` is recoverable
only as a recorded value, not as data.

---

## 3. Frozen artifacts

### 3.1 Registered and consistent

| | |
|---|---|
| model_id | `c44_ridge_frozen_v1` |
| Path | `reports/c44/c44_ridge_frozen_v1.json` |
| File sha256 | `a5013922c67248403d681d7307ff511dbc9e72ffcffd1b2a5b95c17efb1f05fd` |
| Registry pin | identical (`evaluation_hash`) |
| code_commit | `8ad489ae46398a91ff44ca5af529f44e26c47f92` |
| dataset_version | `98790821…` (pre-refresh) |
| feature_version | `c41_v1` |
| label_version | `c41_volatility_range_v1` |
| seed | 42 |
| chosen_alpha | 10.0 (candidates 0.1, 1.0, 10.0) |
| training_window | `[1400, 8187]` |
| created | 2026-08-10T10:40:53Z |

### 3.2 Reference distributions — **lost**

`reference_ranker.json` and `reference_ridge.json` matching artifact
`a5013922…` **no longer exist**. They were overwritten by a re-run on the
refreshed dataset and were never backed up; only the artifact itself was.

There is no recovery path. Regenerating them would require the pre-refresh 4h
CSV, which is gone.

**Consequence:** C4.5 cannot produce a forecast from v1. `load_reference`
requires a pinned hash, and artifact + ranker reference + ridge reference must
all originate from one dataset.

### 3.3 Quarantined — `reports/c44/stale_refresh_20260811/`

Internally consistent with each other, inconsistent with the registered v1.

| File | File sha256 | Internal `content_sha256` |
|---|---|---|
| `c44_ridge_frozen_v1.json` | `1d9bbe7cb6796204…` | — |
| `reference_ranker.json` | `c62f57889ff21e5f…` | `26268991da9be6d1…` |
| `reference_ridge.json` | `e38b35dcf35c3ebb…` | `bd2f8d62f03561a3…` |
| `freeze_report.json` | `101cac1ecef16667…` | — |

Both references carry `version = c44_ref_v1`; the version string cannot
distinguish them from the lost originals, which is precisely why
`content_sha256` and mandatory pinning exist.

Quarantined metadata: code_commit `719f8622…`, dataset_version `7dbef2ab…`,
chosen_alpha 10.0, n_train_rows 6787, train_idx_range `[1400, 8187]`,
holdout_idx_lo 8199, spec_hash `e8a76108e5861e0e…`, generated
2026-08-11T13:25:10Z, registry_status `not_registered` (the registry refused —
correctly, but after the file had already been replaced).

**`train_idx_range` reads identically `[1400, 8187]` in both versions** because
`idx` is positional inside a sliding 9400-bar window. That identity is why the
overwrite was not obvious from the reports.

### 3.4 Three-category scenario table (quarantined artifact, refreshed data)

Median realized 48h range, in multiples of ATR at the prediction bar:

| Category | n | p10 | median | p90 |
|---|---|---|---|---|
| LOW | 1694 | 1.601 | 2.862 | 5.626 |
| NORMAL | 4065 | 1.871 | 3.474 | 6.730 |
| HIGH | 1016 | 2.345 | 4.008 | 8.240 |

Recorded as a measured fact about the quarantined artifact. It is **not** the
scenario table for any registered artifact.

---

## 4. Cost model at baseline

| Parameter | Value | Source |
|---|---|---|
| `TAKER_FEE_PCT` | 0.05 % per side | `config.py:135` |
| `SLIPPAGE_PCT` | 0.03 % per round trip | `config.py:136` |
| Funding | **not modelled** | — |
| M00 model version | `c50_m00_v1` | `validation/cost_sensitivity.py` |
| M00 model sha256 | `1cd4db9fe8e8…` (12-char prefix as reported by the CLI) | — |
| M00 funding default | 0.01 % / 8h | `DEFAULT_FUNDING_PCT_PER_8H` |
| M00 reference notional | $10,000 | `DEFAULT_REFERENCE_NOTIONAL_USD` |

### 4.1 The omission P1 will fix

Seven call sites compute round-trip cost as
`(2 * TAKER_FEE_PCT + SLIPPAGE_PCT) / 100 = 0.0013` and hold positions for
hours without paying funding:

- `backtest.py:133`
- `tools/deep_backtest.py:432`
- `tools/deep_discovery.py:276`
- `tools/deep_diagnostics.py:267`
- `tools/swing_hypothesis_simulator.py:345`
- `tools/swing_hypothesis_simulator.py:509`
- `tools/swing_hypothesis_walkforward.py:440`

`tools/forecast_metrics.py:34-35` keeps its own copies of the same two
constants (0.05 / 0.03) and reports
`modeled_round_trip_cost_pct = 0.13` at line 366.

`analyzer/outcomes.py:100` computes `net_after_costs` as
`r72 - (2 * taker_fee_pct + slippage_pct)` over a 72-hour horizon — nine
funding intervals, none charged.

**Magnitude of the omission at M00's default rate:** +0.06pp per trade at the
48h swing horizon, +0.09pp at the 72h outcome horizon. Every net number in the
artifacts below is overstated by approximately that amount, and the bias is
one-directional (optimistic).

---

## 5. Validation results at baseline

### 5.1 C4.3c — Ridge as a point predictor

Verdict: **`RIDGE_NEEDS_MORE_EVIDENCE`**

| | pooled | fold 0 | fold 1 | fold 2 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| MAE persistence | 1.2459 | 1.2457 | 1.2475 | 1.2445 | 1.2483 | 1.2437 |
| MAE ridge | 0.4014 | 0.3932 | 0.4415 | 0.3695 | 0.4060 | 0.3971 |
| relative improvement | 67.8% | 68.4% | 64.6% | 70.3% | 67.5% | 68.1% |

Ridge beat persistence by a wide, stable margin and **lost on MAE to both
rolling-mean-60 and the plain train mean**. It was rejected as a point
predictor of volatility magnitude. That rejection stands and is not reopened by
anything in C4.3d/e.

### 5.2 C4.3d — ranking against a time-varying baseline

Verdict: **`RIDGE_RANKING_VALUE_CONFIRMED`**. Reproduction guard: reproduced
`True` at tolerance 1e-12; every previously published C4.3 metric returned
unchanged. Generated at commit `c64a8189e95c0b55…`.

Pooled, n = 4062, rows dropped for no observable history: 0.

| metric | ridge | rolling-series baseline |
|---|---|---|
| Spearman | 0.30664 | −0.17601 |
| MAE | 0.40138 | 0.42401 |
| RMSE | 0.49873 | 0.53575 |

Per fold (all confident, n = 1354 each): advantage 0.5569 / 0.6170 / 0.3976.
Per year: 2024 n=1943 advantage 0.5754; 2025 n=2119 advantage 0.4175.

### 5.3 C4.3e — significance of the ranking advantage

Verdict: **`ADVANTAGE_SIGNIFICANT`**, under a rule fixed before the test ran.

| quantity | value |
|---|---|
| `advantage_signed` | 0.48264 |
| `advantage_vs_abs` (honest) | **0.13063** |
| n rows | 4062 |
| naive effective sample | **338** |
| label autocorrelation, lag 1 | 0.93029 |

Moving-block bootstrap, 5000 resamples, seed 42:

| block | CI low | CI high | mean | frac ≤ 0 |
|---|---|---|---|---|
| 12 | 0.05343 | 0.21412 | 0.13071 | 0.0006 |
| 24 | 0.04731 | 0.21444 | 0.13109 | 0.0006 |
| 60 | 0.05054 | 0.21069 | 0.12952 | 0.0002 |
| 120 | 0.04730 | 0.20498 | 0.12840 | 0.0008 |

Block permutation p-values: 0.00360 (block 12), 0.00020 (24), 0.00020 (60).
Worst CI low 0.04730; worst p 0.00360.

Forward-confirmation requirement: **236 effective observations ≈ 2830 matured
forecasts** for a 2.0 SE ratio.

**The distinction that matters:** the headline +0.48 is the *signed* difference
and is inflated, because the baseline is anti-correlated and an observer could
simply invert it. The honest figure is **+0.13** against `|rho|`, worth roughly
1.5 percentage points of quartile accuracy (31.3% vs 29.8%, random 25%).

---

## 6. Hypotheses

### Rejected

| ID | Verdict | Evidence |
|---|---|---|
| **H1** Trend Pullback Continuation | `REJECT_FROZEN_SPECIFICATION` (C2.2c, `reports/c23/DECISION.md`) | 4 bars in validation folds reached the RR gate; 100% failed the frozen RR ≥ 1.5 floor or the cost ceiling. Exercised at its final gates and never survived. |
| **H2** Range Mean Reversion | `REJECT_HYPOTHESIS` (C2.2c) | Passed 7/12 stability checks. `positive_in_2_of_3_folds_or_more`: False. `not_dependent_on_one_calendar_year`: False. `net_positive_after_costs`: True. |
| Regime gate | `REJECT_REGIME_GATE` (C2.1, `reports/c21/DECISION.md`) | — |
| Ridge as point predictor | `RIDGE_NEEDS_MORE_EVIDENCE` (C4.3c) | Lost on MAE to rolling-mean-60 and to the train mean. |

**Direction of bias:** H1, H2 and the regime gate were rejected using the
funding-free cost model, i.e. against costs that were **too low**. Correcting
costs upward makes these rejections stronger. None reopens as a result of P1.

### Accepted

| ID | Verdict | Caveat |
|---|---|---|
| Ridge ranking value | `RIDGE_RANKING_VALUE_CONFIRMED` (C4.3d) | Against a time-varying baseline, not a constant. |
| Ranking advantage significance | `ADVANTAGE_SIGNIFICANT` (C4.3e) | +0.13 honest, not +0.48. Forward confirmation needs ~2830 matured forecasts. |
| Cost viability of the swing horizon | **G0 PASS** (C5.0 M00) | Measured *with* funding; see §7. |

### Never validated

**The confluence score.** The project's primary signal generator has no
validation record of any kind. Its unvalidated numbers carry the same
funding-free optimistic bias as everything else.

### Shipped without a model

The C4.4 ranker is an **inverted trailing mean** (`c44_inverted_trailing_v1`,
window 60, horizon 12), not Ridge. Chosen because it captures roughly 57% of
Ridge's ranking skill with no features and no model. Ridge rides along as a
recorded shadow in the ledger. The inversion is required because the label's
autocorrelation flips sign at the horizon: +0.93 at lag 1, +0.38 at lag 6,
**−0.13 at lag 12**.

---

## 7. Gate status (C5.0 `reports/c50/ARCHITECTURE.md`)

| Gate | Status | Detail |
|---|---|---|
| **G0** viability | **PASS** (2026-08-11) | 4h swing: 0.190% round trip at 48h, cost 0.127 R at a 1.5% stop, required win rate 37.6% at 2R vs 33.3% at zero cost. Phases A–E proceed. |
| **G1** apparatus trustworthy | not started | Requires M01–M04 and the negative-control suite. |
| **G2** meta-labeling | not started | Requires M05. |
| **G3** any structural edge | not started | Requires M06–M08. |
| **G4** new data justified | not started | Requires G3. |

Prerequisite **P1** (funding retrofit) was added by Amendment 1 and blocks
phase A′. No research module has been implemented apart from M00.

### 7.1 M00 numbers as frozen

Break-even, 1.5% stop, 2R payoff:

| holding | fees | slippage | funding | total | cost/R | win rate needed |
|---|---|---|---|---|---|---|
| 1 bar (4h) | 0.100 | 0.030 | 0.005 | 0.135% | 0.090 | 36.3% |
| 3 bars (12h) | 0.100 | 0.030 | 0.015 | 0.145% | 0.097 | 36.6% |
| 6 bars (24h) | 0.100 | 0.030 | 0.030 | 0.160% | 0.107 | 36.9% |
| 12 bars (48h) | 0.100 | 0.030 | 0.060 | 0.190% | 0.127 | 37.6% |
| 24 bars (96h) | 0.100 | 0.030 | 0.120 | 0.250% | 0.167 | 38.9% |

Stop-distance sensitivity at 48h:

| stop | cost/R | need @2R | need @1.5R |
|---|---|---|---|
| 0.3% | 0.633 | 54.4% | 65.3% |
| 0.5% | 0.380 | 46.0% | 55.2% |
| 0.8% | 0.237 | 41.2% | 49.5% |
| 1.0% | 0.190 | 39.7% | 47.6% |
| 1.5% | 0.127 | 37.6% | 45.1% |
| 2.0% | 0.095 | 36.5% | 43.8% |
| 3.0% | 0.063 | 35.4% | 42.5% |

Intraday (3h hold): stop 0.3% → cost 0.446 R, needs 48.2% at 2R.

---

## 8. Test suite

| | |
|---|---|
| Total | **1559 passed**, 0 failed, 14 warnings |
| Runtime | 239.87 s |
| Runner | `.venv/bin/python -m pytest -q` (Python 3.9) |
| `git diff --check` | clean |

New at this baseline: 37 in `tests/test_cost_sensitivity.py`, 9 added to
`tests/test_ridge_freeze_artifact.py` (15 total in that file).

The frozen-write guard was **mutation-verified**: reverting `_write_frozen` to
the original `render(path)` behaviour caused exactly 5 tests to fail; the
mutation was then reverted.

---

## 9. Performance metrics

**There are no live trading performance metrics, and none are expected.** The
bot is DRY_RUN only, has never placed an order, and no module in C5.0 changes
that. Any figure resembling live P&L in a future document is a defect.

Model-level metrics at baseline are those in §5. No strategy-level Sharpe,
expectancy, drawdown, or win rate exists for the current engine, because the
confluence score has never been validated (§6).

---

## 10. Known limitations

1. **Funding is not modelled** anywhere in production or backtest code (§4.1).
   This is what P1 fixes; every net figure in §5 and every backtest artifact is
   optimistically biased.
2. **Effective sample is ~338**, not 4062. Overlapping 12-bar windows.
   Everything downstream inherits this.
3. **The v1 reference distributions are unrecoverable** (§3.2). C4.5 is blocked
   until a new artifact id is frozen on the refreshed dataset.
4. **Training uses uniform sample weights** on overlapping labels — the model
   believes it has ~12× more information than it does. C5.0 M03 addresses this;
   it is not addressed today.
5. **No correction for multiple testing.** H1, H2, confluence variants, four
   profiles, a `/backtest` threshold grid and C4.3a–e were all tried. No trial
   ledger exists; the count is not even known. C5.0 M01 addresses this.
6. **Labels are fixed-horizon, not path-dependent.** The current label answers
   "what was the range 12 bars later, unconditionally" — a question nobody
   trades. C5.0 M02 addresses this.
7. **The sealed holdout (idx 8199+, 1200 bars) has never been read** and must
   stay that way. At ~100 effective observations against the 236 needed, it
   cannot settle the +0.13 question even if spent.
8. **BTC hardcoding** blocks multi-asset work; catalogued in
   `reports/d13/multi_asset_blockers.md`. `db.open_trades()`
   (`database.py:391`) becomes a correctness bug the moment a second asset
   exists.
9. **Two regime notions coexist** — the heuristic `signal_engine/regime.py` and
   the planned M06 HMM. Neither has been reconciled with the other.
10. **The three-category change is uncommitted** and its scenario table (§3.4)
    derives from an unregistered artifact.

---

## 11. Open assumptions

Assumptions in force at baseline, each of which could be wrong without any test
currently detecting it.

| # | Assumption | Status |
|---|---|---|
| A1 | Funding cost is zero | **Known false.** P1 fixes it. Magnitude: 0.06pp per swing trade. |
| A2 | Slippage is a constant 0.03% regardless of size | Unverified. M00 offers a square-root alternative; the reference notional ($10k) is itself unverified. |
| A3 | Execution is instantaneous at bar close | Known false. M00 can price the delay; no backtest applies it. |
| A4 | Taker execution, 0.05% per side | Matches Binance futures taker at the base tier. Fee tier not verified against the actual account. |
| A5 | Funding averages 0.01% per 8h and is always paid | M00's default. Not measured from the funding series; the real series is variable and occasionally negative. |
| A6 | A 1.5% stop distance is representative of the swing profile | Used for §7.1. Plausible for BTC 4h ATR stops; not measured from the actual signal history. |
| A7 | The 19 frozen features carry information | Never established. Ridge lost to a constant on MAE. |
| A8 | Overlapping observations can be corrected for at evaluation time alone | False in training (§10.4). |
| A9 | The pre-refresh 4h data (`98790821…`) was equivalent to the current cache apart from the added bars | Unverifiable — that CSV no longer exists. |
| A10 | Binance klines are an accurate record of traded prices | Assumed throughout. No cross-venue reconciliation has been performed. |
| A11 | The label's inverse autocorrelation at lag 12 is structural, not an artifact of this sample | The C4.4 ranker's inversion depends on it. Explanation (ATR catching up) is plausible and untested. |

---

## 12. Attribution protocol for P1

For a post-P1 comparison to be valid against this baseline:

1. Cite this document by its own hash (`BASELINE.sha256`, alongside).
2. Change **only** the cost model. No retraining, no feature change, no label
   change, no walk-forward geometry change, no cache refresh.
3. Re-run the same artifacts listed in §5 and report old vs new side by side.
4. Expect every net figure to move in one direction — down. A net figure that
   improves after funding is charged indicates a defect in the retrofit, not a
   discovery.
5. If a verdict flips, it can only flip from accepted to rejected. A rejection
   reversing to acceptance is impossible under a strictly increasing cost and
   would signal that something other than cost changed.
