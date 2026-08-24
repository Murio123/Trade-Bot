# S3 — independent feature families and strong baselines: result

Criteria: `reports/s3/S3_SPEC.md`, frozen before any outcome was computed.
Trials declared before any label was read. No S2 definition was changed.

## Amendment A1 — three audit defects fixed, all numbers recomputed (2026-08-24)

An independent audit returned `NOT_SAFE` on the first S3 run with three real
defects. All are fixed, everything below is recomputed, and **the verdict is
unchanged**.

1. **Eligibility admitted coins that had already stopped trading.** Rule E8
   checked that a symbol's last 30 *observed* bars were consecutive rather than
   that they were the last 30 days *before the decision date*, so a coin
   delisted in 2022 stayed eligible at every later date and then resolved as a
   failure. 938 such rows. This is S2's defect; `S2_RESULT.md` Amendment A1
   carries the correction, and it moved the universe base rate from 18.21% to
   **20.85%** and the survivorship gap from +2.68 pp to **+0.04 pp**.
2. **The ranking set was filtered by label availability.** Symbols whose
   outcome was censored or gap-unresolved were removed *before* `K` was
   computed, letting information from after the decision reshape the shortlist.
   The ranking is now over every eligible symbol with a feature value;
   unresolved symbols are excluded from the rates only.
3. **A partial trailing bootstrap block was resampled as a whole one.** 67
   dates read as 12 blocks where `S3_SPEC.md` §7 defines 11. Partial tails are
   now dropped.

A follow-up audit found a fourth, smaller defect and it is fixed too: the
per-date event count reported the size of the *ranking* rather than the number
of outcomes the rates were measured over, overstating the sample by every
unresolved row (6,125 against the true 6,107). Ranked symbols and resolved
outcomes are different counts — `K` must come from the ranking, the sample size
must not.

Net effect on the verdict: none. Every lift moved further below 1
(0.847–0.939, previously 0.868–0.947), the same two families are `REMOVE`, and
the same four are `UNKNOWN`. **The corrections made the result more negative,
not less** — worth stating, because a fix applied after seeing results invites
the question of whom it helped.

---

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
| decision dates | **66** of S2's 98 — those with ≥25 eligible symbols |
| span | 2020-08-01 → 2026-02-01 |
| events | 6,107 (symbol, date) resolved outcomes |
| **independent blocks** | **11** (six monthly dates each; a partial tail is dropped) |
| feature coverage | **1.000** for all six rules — no missingness |
| regime dates | bull 40 · bear 16 · range 10 |

The ≥25 threshold discarded 2018-01 → 2020-07, as `S3_SPEC.md` §2 said it
would. **The universe rate over these 66 dates is 25.7%, not S2's 20.85%** —
a different date set, and date-weighted rather than pooled. Both numbers are
correct and they are not the same quantity; every comparison below uses the
25.7% figure, computed on exactly the dates that judge the rules.

## 2. Baselines first

| baseline | CLEAN_2X | median 180d | vs BTC (log) | median MAE |
|---|---|---|---|---|
| **B2 eligible universe (equal weight)** | **0.257** | −0.091 | **−0.374** | −0.483 |
| **B1 BTC hold** | 0.121 | **+0.159** | 0.000 | **−0.213** |
| **B0 random eligible** (1,000 seeded draws/date) | 0.257 | — | — | — |
| **B3 cap-weighted** | **UNAVAILABLE** — no trustworthy point-in-time market cap (S0 §6); today's cap applied historically is a leak and was not approximated | | | |

This table is the honest framing for everything after it. **The eligible
universe doubles twice as often as BTC (25.7% vs 12.1%) and still loses to it**
— median terminal −9.1% against BTC's +15.9%, a −0.374 log gap, with more than
twice the drawdown. Chasing the 2x outcome is not the same as making money, and
the baseline that matters is the boring one.

## 3. The six rules

Top quintile, `K = ceil(0.20 × N)`, aggregated as the mean of per-date values.

| id | family | rate | universe | **lift** | vs BTC | MAE | Spearman ρ | ρ 90% CI | rate-diff 90% CI | class |
|---|---|---|---|---|---|---|---|---|---|---|
| B4 | momentum 6-1 | 0.230 | 0.257 | 0.896 | −0.330 | −0.474 | **+0.093** | [0.029, 0.167] | [−0.055, 0.000] | UNKNOWN |
| B5 | relative strength 90d | 0.229 | 0.257 | 0.891 | −0.389 | −0.503 | +0.005 | [−0.035, 0.048] | [−0.067, 0.007] | UNKNOWN |
| **F1** | RS persistence | 0.237 | 0.257 | 0.922 | −0.328 | −0.472 | −0.030 | [−0.071, 0.011] | **[−0.034, −0.007]** | **REMOVE** |
| **F2** | distance above 200d MA | 0.230 | 0.257 | 0.895 | −0.357 | −0.484 | +0.043 | [−0.018, 0.113] | [−0.066, 0.007] | UNKNOWN |
| **F3** | abnormal participation | 0.218 | 0.257 | **0.847** | **−0.441** | **−0.517** | **−0.084** | [−0.124, −0.044] | **[−0.073, −0.011]** | **REMOVE** |
| **F4** | drawdown from 180d high | 0.241 | 0.257 | **0.939** | **−0.261** | **−0.418** | **+0.136** | [0.071, 0.205] | [−0.036, 0.002] | UNKNOWN |

Universe reference on the same dates: median vs BTC −0.374, median MAE −0.483.

Secondary fixed Top-10 (product intuition only, not part of any verdict):
0.230–0.262, i.e. the same picture. **Dates on which a rule beat the 95th
percentile of matched random: 1.5% to 9.1%** — the frozen condition wanted a
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
forward returns better than chance**, with bootstrap intervals over 11 blocks
that exclude zero: ρ = +0.136 [0.071, 0.205] and +0.093 [0.029, 0.167]. F4's
top quintile also has the best BTC-relative outcome of any rule (−0.261 against
the universe's −0.374) and the shallowest drawdown (−0.418 against −0.483).

And yet **F4's top quintile doubles *less* often than the universe** (0.241 vs
0.257). Both statements are measured, and the tension between them is the most
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

1. **Eleven blocks.** Every interval rests on 11 independent 180-day windows,
   not on 6,107 rows. A true lift of 1.1 would not be reliably detectable at
   this sample size, so `NO_SIGNAL` means *these rules did not show value*, not
   *no simple rule can have value*.
2. **2018 → mid-2020 is absent**, including the window S2 measured at a 52%
   base rate. The rules were never tested on the most favourable era.
3. **Bull-heavy**: 40 of 66 dates. The bear and range subsamples are 16 and 10
   dates — one to two blocks each, which is why the per-regime lifts in the
   artifact are diagnostic only and no verdict rests on them.
4. **B3 is missing entirely.** Without point-in-time market cap there is no
   size-weighted benchmark, and size is the most commonly claimed
   cross-sectional effect in this asset class.
5. **One definition per family, by design.** A different momentum lookback
   might behave differently. Finding that out costs a registered trial, and
   the registry is what stops that from becoming a silent search.

## 8. Verification

- **56 targeted tests**: `test_spot_features.py` (30), `test_spot_evaluate.py`
  (20) plus 6 new regression tests for the audit defects, covering point-in-time construction, that appending future bars cannot
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

> *If I had pressed "🔎 Найти монеты" on any of these 66 months, would the
> shortlist have been better?*

**No.** Against the eligible universe it would have doubled less often — every
rule, in every configuration tested. Against matched random selection it was
indistinguishable on more than 90% of dates. Against simply holding BTC it
would have been much worse on return and on drawdown: the universe itself
loses to BTC by ~37% in log terms over 180 days, and none of these rules closes
that gap.

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
