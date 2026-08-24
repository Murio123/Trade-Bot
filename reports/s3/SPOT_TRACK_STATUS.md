# Spot discovery track — stage close after S3

Recorded 2026-08-24. This document freezes **where the track stands**, not what
will be done next. It exists because the gap between a finished measurement and
the next decision is exactly where a project quietly rewrites what it found.

Machine-readable form: `reports/s3/spot_governance.json`, asserted by
`tests/test_spot_governance.py`.

---

## 1. Established facts — measured, not to be reinterpreted

| stage | verdict | artifact |
|---|---|---|
| S0 — architecture | `SPOT_ARCHITECTURE_READY` | `reports/s0/SPOT_DISCOVERY_ARCHITECTURE.md` |
| S1 — point-in-time data | `S1_PASS` | `reports/s1/S1_RESULT.md` |
| S2 — universe + labels | `S2_PASS` | `reports/s2/S2_RESULT.md` |
| **S3 — feature families** | **`S3_NO_SIGNAL`** | `reports/s3/S3_RESULT.md` |

The numbers that carry forward:

- **`CLEAN_2X(180d)` base rate: 20.85%** (1,334 of 6,397 resolved events,
  98 monthly dates, 2018-01 → 2026-02).
- **734 USDT pairs, 826,301 daily bars, 250 of them delisted.** Point-in-time
  throughout; delisted assets retained; death inside the horizon is a failure.
- **Survivorship gap: +0.04 pp.** Near zero for this metric on this horizon —
  see S2 Amendment A1, which corrects an earlier +2.68 pp figure that was an
  artifact of a defect rather than a property of the market.
- **All six pre-declared ranking rules underperform the eligible universe**:
  lifts 0.847–0.939; none beats matched random on more than 9% of dates. F1 and
  F3 are `PRELIMINARY_REMOVE`; B4, B5, F2, F4 are `PRELIMINARY_UNKNOWN`.
- **Spot family `n_trials` = 6**, family `(spot_2x_discovery, ·, ·,
  clean_2x_180d)`. The futures family's 100 is a different family and does not
  mix.
- **Effective sample: 11 independent 180-day blocks** in S3, ≈16 in S2. No
  claim rests on the row count.

### 1.1 The two findings worth more than the verdict

1. **The eligible universe doubles twice as often as BTC (25.7% vs 12.1%) and
   still loses to it** — median −9.1% against +15.9%, −0.374 in logs, with
   double the drawdown. Reaching 2x and making money are not the same event.
2. **Ranking the average coin and ranking the tail point in opposite
   directions.** F4 ranks continuous forward returns with ρ = +0.136, CI
   [0.071, 0.205] excluding zero, and has the best BTC-relative outcome of any
   rule — while doubling *less* often than the universe.

**Finding 2 is an observation, not a validated hypothesis.** It was seen after
the fact, on the same data, against a target the rule was not declared for.
Nothing may be built on it without a fresh registration.

## 2. The open decision — not taken

The owner has not chosen, and no option below has been acted on. Recorded so
that a later reader can see the decision was open rather than assumed.

| option | what it means | cost |
|---|---|---|
| **A — ship a facts screener** | PIT universe, liquidity, drawdown, relative strength, BTC regime, as percentiles. **No selection claim, no probability, no Buy/Avoid.** S0 §18 pre-registered this as an admissible outcome. | 1–2 sessions |
| **B — S3b, new families, same target** | More features against `CLEAN_2X`, each registered on top of `n_trials = 6`. | 3–4 sessions; low power at 11 blocks |
| **C — re-scope the target** | e.g. BTC-relative outperformance over 180 days. A governance decision on the scale of S0 §3. | re-freeze + sealed holdout |

### 2.1 The implementer's recommendation, which binds nothing

A + C, in that order, with C conditional: ship the facts screener first, and
attempt a re-scoped target **only** under a fresh freeze that seals the most
recent 18 months of data before any feature is looked at. B is not recommended:
at 11 blocks against a tail target, the sample cannot separate a real small
effect from search.

This is a recommendation from the party that ran the measurement. It is
recorded as such and decides nothing.

### 2.2 The trap this record exists to name

The tempting move is C without the freeze: *the 2x target failed, so let us
measure the thing that worked instead*. That is a frozen criterion being
replaced by a passing one after the results were seen — the same shape as
`G1_SPEC.md` §7 A3, one stage later, and the reason G1's verdict is still
`G1_INDETERMINATE`.

C is legitimate **only** with: a target frozen before inspection, trials
registered before outcomes, a sealed holdout, and the post-hoc origin of the
idea stated in the spec.

## 3. Constraints binding until a decision is taken

1. **S4 is not started.** No scanner, no combined score, no ranking model.
2. **No feature may be combined with another.** `spot/` contains no combiner
   and that absence is the mechanism.
3. **No unregistered experimentation.** Any new family is a trial declared
   before its outcome is read.
4. **The S2 target is not edited.** `CLEAN_2X`, +100% / −40% / 180 days stands
   as frozen; a different target is a new spec, not an amendment to this one.
5. **Phase A′ of the futures track remains paused**, not cancelled and not
   deleted. G1 stays `G1_INDETERMINATE`; G1.1 stays `G1_1_PASS` with
   `n_trials = 100` in its own family.
6. **No runtime, futures, database, scheduler or Telegram code is touched** by
   the spot track. The import-direction guards enforce both directions.

## 4. Repository state

- Branch `claude/btc-telegram-trading-bot-1avo98`, **12 commits ahead of
  origin and not pushed**: `git push` returns HTTP 403, the local credential
  (`umutmost-svg`) is not the repository owner (`Murio123/Trade-Bot`). This is
  the only external blocker and it needs the owner's authentication.
- Full suite: **2054 passed**. Codex audit of the S3 work: **SAFE_TO_PUSH**
  after four rounds that found seven real defects between them.
- `data/` is gitignored; the 73 MB panel and all generated evaluation
  artifacts are regenerable from the committed tools.

## 5. What the four audit rounds found

Recorded because the pattern is worth keeping, not to tally errors:

| round | defect | direction |
|---|---|---|
| 1 | eligibility admitted coins that had already stopped trading (938 rows) | inflated failures, depressed the base rate |
| 1 | the ranking set was filtered by label availability | outcome information reached the selection |
| 1 | a partial bootstrap block was resampled as a whole one | overstated independence |
| 2 | the event count reported ranked symbols, not measured outcomes | overstated the sample |
| 2 | a stale table header | report disagreed with its artifact |
| 3 | two medians reported the upper-middle value | report disagreed with its artifact |
| 4 | S1 kept pre-re-ingestion panel totals | report disagreed with its artifact |

**Every correction moved the result in the unfavourable direction**, and the
first one reversed a headline claim the project had already published. Three of
the seven were reports drifting from the artifacts they describe, which is the
failure mode of a stage that ships documents and data separately.
