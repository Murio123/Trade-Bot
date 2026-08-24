# S3 — independent feature families and strong baselines: frozen criteria

Written **before any S3 outcome was computed**. Everything below is a choice
that could be tuned toward a preferred answer, which is why it is fixed first.
Changes require an amendment section and a new trial registration, never an
edit.

Anchors, all frozen and not reinterpreted here: S2 (`reports/s2/S2_RESULT.md`,
`S2_PASS`) — `CLEAN_2X(180d)`, +100% / −40%, **7,324 resolved events, 1,334
positives, base rate 18.21%**, 98 monthly dates, ≈16.3 independent windows,
point-in-time universe mandatory, delisted assets retained, death inside the
horizon = failure, survivorship-only rate 20.90% **must not be used**.

---

## 1. The question S3 answers

> Can any simple, pre-declared, point-in-time observable rank future spot
> opportunities better than strong trivial baselines?

Not: can we build a scanner. Not: what is the best achievable number. Each
family is measured **alone**, and the verdict decides only whether S4 is
justified.

## 2. Evaluation dates

- Candidate dates: the 98 monthly decision dates of S2.
- **Included iff the point-in-time eligible universe has ≥ 25 symbols.** That
  is **67 dates, 2020-08-01 → 2026-02-01**.
- Rationale, stated before results: the primary selection is a top quintile,
  and a quintile of fewer than 25 names is 4 assets or less — a rate estimated
  from 4 draws is not a rate. The cost is real and is recorded as a limitation:
  **2018-01 → 2020-07 is discarded entirely**, including the 2020 window whose
  base rate S2 measured at 52%.
- No secondary threshold is run. A sensitivity sweep over the minimum universe
  size would be a parameter search, and it is not registered, so it will not
  happen.

## 3. Selection policy

- **Primary: the top quintile** — `K = ceil(0.20 × N)` of the eligible
  universe, ranked by the feature. With `N ≥ 25`, `K ≥ 5`.
- Percentile rather than fixed Top-N because the universe ranges from 25 to 238
  names; a fixed Top-5 would be the top 20% early and the top 2% late, which
  makes the time series of a metric incomparable with itself.
- **Secondary, reported for product intuition only and explicitly not part of
  any verdict: fixed Top-10.**
- Ties in the feature are broken by symbol name ascending, so a ranking is
  deterministic.

## 4. Baselines

| id | definition | note |
|---|---|---|
| **B0** | random eligible selection: `K` symbols drawn uniformly without replacement per date, **1,000 draws**, seed `20260824` mixed with the date index | deterministic; the matched null for a top-quintile rule |
| **B1** | hold BTC over the same 180 days from each date | the economic benchmark; not a cross-sectional ranking |
| **B2** | equal-weight eligible universe | the "just buy the market" reference; equals the per-date universe rate |
| **B3** | market-cap-weighted eligible universe | **UNAVAILABLE.** No trustworthy point-in-time market cap exists (S0 §6); today's cap applied historically is a leak. Reported as unavailable, never approximated. |
| **B4** | **momentum**: `close[−31] / close[−181] − 1`, ranked descending | one definition, no grid |
| **B5** | **BTC-relative strength**: `log(close[−1]/close[−91]) − log(btc[−1]/btc[−91])`, ranked descending | one definition, no grid |

Indices count backwards over **visible** bars — those that closed strictly
before the decision instant.

**Why B4 is 5 months of formation skipped by 1 month**, chosen from first
principles rather than from a sweep: cross-sectional momentum is conventionally
formed over 6–12 months and skips the most recent month to avoid short-term
reversal contaminating the signal. The formation window is set to the scale of
the 180-day horizon being predicted. The skip is one month because the decision
grid is monthly.

**Why B5 is 90 days:** relative strength is the shorter-lived of the two
effects and is measured against the asset the user would otherwise hold. 90
days is one quarter — long enough to be a trend, short enough to be current.

## 5. Feature families

One primary implementation each. No family may use another family's inputs, no
family is combined with any other, and no parameter is tuned.

| id | family | frozen definition (visible bars only) | direction |
|---|---|---|---|
| **F1** | relative strength | **persistence**: the fraction of the last **120** days on which `close/btc_close` sat above its own **60**-day trailing mean | descending |
| **F2** | momentum / trend quality | **distance above a medium-term average**: `close[−1] / mean(close[−200:]) − 1` | descending |
| **F3** | volume / participation | **abnormal participation**: `log( median(quote_volume[−30:]) / median(quote_volume[−180:]) )` | descending |
| **F4** | volatility / drawdown state | **distance from the 180-day high**: `close[−1] / max(close[−180:]) − 1` | descending |

Notes that are part of the freeze:

- **F1 is not B5.** B5 is a point-to-point relative return; F1 is how *often*
  the ratio held above its own trend. One is a level, the other is persistence.
- **F2 is not B4.** B4 is a past return with a skip; F2 is the current position
  relative to a long average. One is what happened, the other is where price
  now stands.
- **F3 is normalised within each asset** (a ratio of an asset's own volume to
  its own history), so it cannot reward permanently large coins. It uses no
  future liquidity and no present-day survivor information.
- **F4's primary direction is trend continuation** — least drawdown ranked
  first. The opposite hypothesis (deep drawdown = cheap) is plausible and is
  *not* being tested as a second arm, because running both and reporting the
  better one is a two-arm search reported as one result. Instead the **decile
  table is published in full**, and if the relationship is non-monotonic that
  is reported as the finding. **No threshold may be manufactured from the
  observed shape.**

Missing feature values (insufficient history) exclude a symbol from that date's
ranking and are counted as missingness. Eligibility already requires ≥200 bars,
so this should be near zero; a non-trivial rate is a defect report.

## 6. Metrics

Six, frozen. Every one is computed **per date**, then aggregated.

| id | metric |
|---|---|
| **M-A** | `CLEAN_2X` rate within the selection |
| **M-B** | **lift** = selection rate ÷ eligible-universe rate |
| **M-C** | median 180-day terminal return of the selection |
| **M-D** | median BTC-relative forward return (`excess_vs_btc`, log) of the selection |
| **M-E** | median MAE of the selection |
| **M-F** | per-date Spearman rank correlation between the feature and forward terminal return, over all eligible resolved symbols |

Aggregation across dates is the **mean of per-date values, equally weighted per
date** — not a pooled average over 7,324 rows, which would weight 2024 (238
names) twenty times more than 2020 (25 names) and would treat overlapping
events as independent. The pooled figure is reported alongside for reference
and is not the primary.

No other metric may be introduced after results. Nothing is selected on the
basis of which metric flatters a family.

## 7. Dependence and uncertainty

- **Blocks**: consecutive, non-overlapping runs of **6 monthly dates** = one
  180-day horizon. 67 dates ⇒ **≈11 blocks**.
- **Procedure**: block bootstrap over whole blocks, **2,000 resamples**, seed
  `20260824`, **90% percentile interval**. Frozen.
- Three sample counts are reported together, always: raw event count, number of
  decision dates, and the effective independent block count. **No confidence
  claim may rest on n = 7,324.**

## 8. Regime breakdown

The S2 regime classifier is reused unchanged: BTC close versus its 200-day SMA
plus the SMA's own 20-day slope, computable at the decision instant from past
BTC bars only. Categories: **bull / range / bear**. Every family reports every
metric split by regime.

The split is diagnostic. **No feature threshold may differ per regime in S3.**

## 9. BTC common-factor control

For every family, the absolute outcome (`CLEAN_2X`, raw return) and the
relative outcome (`excess_vs_btc`) are reported separately, overall and within
regime. If a family's apparent lift disappears once measured against BTC, or
survives only in the bull regime, **that is the finding and it is stated
plainly** rather than averaged away.

## 10. Go / no-go rule

Frozen. May be tightened before running; may **not** be loosened afterwards.

**`PRELIMINARY_KEEP`** requires **all seven**:

1. **lift > 1.0** — the top-quintile `CLEAN_2X` rate exceeds the eligible
   universe rate (mean of per-date values);
2. **beats matched random** — the selection rate exceeds the **95th percentile**
   of the B0 distribution for the same dates and the same `K`;
3. **improves the BTC-relative outcome** — median `excess_vs_btc` of the
   selection exceeds that of the eligible universe;
4. **temporal stability** — per-date lift > 1 in **at least 60% of blocks**;
5. **not one regime** — lift > 1 in **at least two of the three** regimes;
6. **uncertainty** — the block-bootstrap 90% interval for
   (selection rate − universe rate) **excludes zero**;
7. **no point-in-time or survivorship violation**, asserted by tests rather
   than by inspection.

**`PRELIMINARY_REMOVE`**: lift < 1.0 **and** the 90% interval for
(selection rate − universe rate) lies entirely below zero.

**`PRELIMINARY_UNKNOWN`**: anything else — including a positive point estimate
whose interval contains zero. Most honest outcomes at ≈11 blocks will land
here, and that is expected rather than disappointing.

## 11. Multiple testing and the Trial Registry

- **Family scope**: `research_objective = "spot_2x_discovery"`,
  `target_family = "clean_2x_180d"`, `symbol = None` (multi-asset),
  `profile = None`. The futures family's `n_trials = 100` is **historical
  governance context for a different family** and is neither transferred nor
  mixed; the spot family starts at zero.
- **Six trials are registered before any outcome is read**: B4, B5, F1, F2, F3,
  F4 — every rule that ranks assets and could therefore influence selection.
- **B0–B3 are not trials.** B0 is a null distribution, B1 and B2 are reference
  portfolios with no cross-sectional choice in them, and B3 is unavailable.
  None of them selects among assets on evidence, so none consumes a look. This
  is recorded here rather than left implicit, because §2.4 of the registry spec
  resolves ambiguity upward and a reader is entitled to disagree.
- Each trial's identity pins the exact definition. **Changing a definition
  produces a different `trial_id` and therefore a new trial** — that is the
  mechanism, and a test asserts it.
- Any diagnostic variant inspected after primary results is registered as its
  own trial before it is looked at. There is no unregistered experimentation.

## 12. What S3 must not do

No combined score, no weighting, no sum of features, no confluence, no
regression over several features, no ML, no hand-written "2x score", no
BUY/AVOID rule, no threshold tuning, no S4, no Telegram, no change to any
runtime, futures, database or scheduler code, and no reinterpretation of any S2
number.

## 13. Verdict vocabulary

- **`S3_PASS`** — at least one family earns `PRELIMINARY_KEEP`. It means S4 is
  justified. It does **not** mean a profitable strategy exists.
- **`S3_NO_SIGNAL`** — no family shows credible selection value.
- **`S3_INDETERMINATE`** — the sample cannot support a defensible conclusion.
- **`S3_FAIL`** — the measurement apparatus itself is not trustworthy.
