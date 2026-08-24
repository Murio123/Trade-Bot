# S3 — independent feature families and strong baselines: result

Criteria: `reports/s3/S3_SPEC.md`, frozen before any outcome was computed.
Trials declared before any label was read. No S2 definition was changed.

## Verdict: `S3_NO_SIGNAL`

> **No simple pre-declared feature family ranks `CLEAN_2X(180d)` better than
> the eligible universe.** All six rules select a top quintile whose clean-2x
> rate is *below* the universe rate; none beats matched random selection; two
> are actively harmful with intervals excluding zero.

`S3_PASS` required at least one `PRELIMINARY_KEEP` under the seven frozen
conditions. **None was earned, and none was close** — every rule failed
condition 1 (lift > 1) and condition 2 (beats matched random).

One qualification is material and is stated up front rather than buried: **two
families do carry credible cross-sectional information about *continuous*
forward returns** — just not about the extreme-outcome target this stage was
built around. §5 sets it out; §8 states what may and may not be concluded from
it.

---

## 1. Scope actually evaluated

| | |
|---|---|
| decision dates | **67** of S2's 98 — those with ≥25 eligible symbols |
| span | 2020-08-01 → 2026-02-01 |
| events | 7,029 (symbol, date) resolved outcomes |
| **independent blocks** | **12** (six monthly dates each) |
| feature coverage | **1.000** for all six rules — no missingness |
| regime dates | bull 41 · bear 16 · range 10 |

The ≥25 threshold discarded 2018-01 → 2020-07, as `S3_SPEC.md` §2 said it
would. **The universe rate over these 67 dates is 26.8%, not S2's 18.21%** —
different date set, and date-weighted rather than pooled. S2's number is
unchanged and is not restated by this; the two are simply not the same
quantity, and every comparison below uses the 26.8% figure computed on the same
dates as the rules it judges.

## 2. Baselines first

| baseline | CLEAN_2X | median 180d | vs BTC (log) | median MAE |
|---|---|---|---|---|
| **B2 eligible universe (equal weight)** | **0.268** | −0.034 | **−0.365** | −0.477 |
| **B1 BTC hold** | 0.134 | **+0.159** | 0.000 | **−0.206** |
| **B0 random eligible** (1,000 seeded draws/date) | 0.268 | — | — | — |
| **B3 cap-weighted** | **UNAVAILABLE** — no trustworthy point-in-time market cap (S0 §6); today's cap applied historically is a leak and was not approximated | | | |

This table is the honest framing for everything after it. **The eligible
universe doubles twice as often as BTC (26.8% vs 13.4%) and still loses to it**
— median terminal −3.4% against BTC's +15.9%, a −0.365 log gap, with more than
twice the drawdown. Chasing the 2x outcome is not the same as making money, and
the baseline that matters is the boring one.

## 3. The six rules

Top quintile, `K = ceil(0.20 × N)`, aggregated as the mean of per-date values.

| id | family | rate | universe | **lift** | vs BTC | MAE | Spearman ρ | ρ 90% CI | rate-diff 90% CI | class |
|---|---|---|---|---|---|---|---|---|---|---|
| B4 | momentum 6-1 | 0.254 | 0.268 | **0.947** | −0.317 | −0.467 | **+0.096** | [0.034, 0.163] | [−0.043, 0.014] | UNKNOWN |
| B5 | relative strength 90d | 0.244 | 0.268 | 0.909 | −0.389 | −0.495 | +0.003 | [−0.035, 0.045] | [−0.065, 0.013] | UNKNOWN |
| **F1** | RS persistence | 0.251 | 0.268 | 0.937 | −0.307 | −0.464 | −0.027 | [−0.061, 0.013] | **[−0.033, −0.002]** | **REMOVE** |
| **F2** | distance above 200d MA | 0.246 | 0.268 | 0.919 | −0.361 | −0.478 | +0.042 | [−0.016, 0.106] | [−0.060, 0.014] | UNKNOWN |
| **F3** | abnormal participation | 0.233 | 0.268 | **0.868** | **−0.464** | **−0.513** | **−0.086** | [−0.125, −0.045] | **[−0.068, −0.003]** | **REMOVE** |
| **F4** | drawdown from 180d high | 0.248 | 0.268 | 0.926 | **−0.252** | **−0.413** | **+0.136** | [0.064, 0.206] | [−0.046, 0.003] | UNKNOWN |

Secondary fixed Top-10 (product intuition only, not part of any verdict):
0.251–0.283, i.e. the same picture. **Dates on which a rule beat the 95th
percentile of matched random: 0% to 10%** — the frozen condition wanted a
majority.

Per-family classification with the full seven-condition check is in
`reports/s3/s3_results.json`; every rule failed conditions 1 and 2, and none
achieved lift > 1 in two of three regimes.

## 4. Trial registry

Six trials declared in family
`(spot_2x_discovery, ·, ·, clean_2x_180d)` **before any outcome was read** —
the runner declares first and opens the labels afterwards, and a test asserts a
rerun adds nothing while an edited definition produces a different `trial_id`.

| id | trial_id | id | trial_id |
|---|---|---|---|
| B4 | `t_974d0dead1b1996a` | F2 | `t_91ec376f90e6d974` |
| B5 | `t_82eceae674fe0c9e` | F3 | `t_8ace183d042bfa38` |
| F1 | `t_2aafc60f8e8228fe` | F4 | `t_068c060b2fa91abe` |

**Spot family `n_trials` is now 6.** The futures family's 100 is another
family's number and was neither transferred nor mixed. Any future S4 hypothesis
starts on top of 6, not on top of zero.

## 5. The one result that is not negative

**F4 (distance from the 180-day high) and B4 (6-1 momentum) rank continuous
forward returns better than chance**, with bootstrap intervals over 12 blocks
that exclude zero: ρ = +0.136 [0.064, 0.206] and +0.096 [0.034, 0.163]. F4's
top quintile also has the best BTC-relative outcome of any rule (−0.252 against
the universe's −0.365) and the shallowest drawdown (−0.413 against −0.477).

And yet **F4's top quintile doubles *less* often than the universe** (0.248 vs
0.268). Both statements are measured, and the tension between them is the most
informative thing S3 produced:

> Ranking well on the *average* coin and ranking well on the *tail* are not the
> same problem here — and on this evidence they point in opposite directions.
> Coins near their highs behave better on the whole and reach +100% slightly
> less often; the 2x outcome disproportionately comes from beaten-down,
> high-variance names, which are also the names that die.

**What may not be concluded from that.** It is an observation about two
frozen features, not a validated hypothesis, and "invert F4" is a hypothesis
that has not been tested. Testing it is a new trial and it is not run here
(`S3_SPEC.md` §11). It is written down so that S4's scoping can consider it,
and so that nobody later reads the direction back into S3 as if it had been
demonstrated.

## 6. What went wrong in the first run, and how it was caught

The first evaluation reported **lift > 1 for all six rules** — 1.14 to 1.40 —
while their selection rates sat *below* the universe rate on the same lines.
The two numbers contradicted each other on the same row.

Cause: lift was implemented as the mean of per-date ratios. A date where the
universe rate is 2% and the selection rate is 10% contributes a ratio of 5, so
a handful of quiet months dominate the average and a rule that is worse
throughout reads as better.

`S3_SPEC.md` §10 condition 1 is worded as *"the top-quintile `CLEAN_2X` rate
exceeds the eligible universe rate"* — a comparison of **rates**. So the
mean-of-ratios implementation was a defect **against the frozen spec**, not a
choice the results made inconvenient. It was corrected to a ratio of means, the
mean-of-ratios figure is still reported as a diagnostic, and a test now pins
the exact pathology.

**The correction made the verdict worse, not better**: F1 and F3 moved from
`UNKNOWN` to `PRELIMINARY_REMOVE`. That direction is worth stating explicitly,
because a fix applied after seeing results deserves the question "did it help
the author?" — here it did the opposite.

## 7. Limitations, stated rather than discovered later

1. **Twelve blocks.** Every interval rests on 12 independent 180-day windows,
   not on 7,029 rows. A true lift of 1.1 would not be reliably detectable at
   this sample size, so `NO_SIGNAL` means *these rules did not show value*, not
   *no simple rule can have value*.
2. **2018 → mid-2020 is absent**, including the window S2 measured at a 52%
   base rate. The rules were never tested on the most favourable era.
3. **Bull-heavy**: 41 of 67 dates. The bear and range subsamples are 16 and 10
   dates — one to two blocks each, which is why the per-regime lifts in the
   artifact are diagnostic only and no verdict rests on them.
4. **B3 is missing entirely.** Without point-in-time market cap there is no
   size-weighted benchmark, and size is the most commonly claimed
   cross-sectional effect in this asset class.
5. **One definition per family, by design.** A different momentum lookback
   might behave differently. Finding that out costs a registered trial, and
   the registry is what stops that from becoming a silent search.

## 8. Verification

- **50 targeted tests**: `test_spot_features.py` (30), `test_spot_evaluate.py`
  (20), covering point-in-time construction, that appending future bars cannot
  change a computed value, determinism, tie-breaking, percentile Top-K under a
  changing universe, BTC alignment on timestamps rather than positions,
  block-resampled uncertainty, the frozen go/no-go rule in all four outcomes,
  trial declaration, rerun idempotence, and definition-change ⇒ new trial.
- **Full suite: 2051 passed, 0 failed.** `git diff --check` clean.
- No runtime, futures, database, scheduler or Telegram code touched; the S2
  import-direction guards still pass. Phase A′ remains paused.
- No combined score, no weighting, no regression over features, no ML, no
  BUY/AVOID rule was built. `spot/` contains no function that combines two
  families.

## 9. Product answer

> *If I had pressed "🔎 Найти монеты" on any of these 67 months, would the
> shortlist have been better?*

**No.** Against the eligible universe it would have doubled slightly less
often. Against matched random selection it was indistinguishable on 90%+ of
dates. Against simply holding BTC it would have been much worse on return and
much worse on drawdown — the universe itself loses to BTC by ~36% in log terms
over 180 days, and none of these rules closes that gap.

The nearest thing to good news is F4: its shortlist would have lost to BTC by
less, and drawn down less, than a random eligible pick. That is a smaller loss,
not a gain.

## 10. Next allowed stage

**S4 is not automatically justified**, and this is the owner's call, not the
implementer's. Three options, stated without a recommendation being acted on:

1. **Stop the discovery track.** Ship the screener as a facts tool — liquidity,
   drawdown, relative strength percentiles — with no selection claim. S0 §18
   pre-registered exactly this outcome.
2. **Re-scope the target.** §5's finding suggests the tail and the average
   diverge. Changing the target is a governance decision on the scale of S0 §3,
   requires re-freezing, and must not be chosen because it is the branch that
   keeps the project alive.
3. **Register new families and run S3b.** Each on top of `n_trials = 6`, each
   declared before its outcome is read.

Nothing further is started. S3 stops here.
