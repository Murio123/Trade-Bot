# P1 — Funding-aware cost model: result

Closes the prerequisite inserted by `ARCHITECTURE.md` Amendment 1 and specified
in `P1_TASK.md`. Attribution anchor, as required:
`b3eca107f3d0f76928daf73448f00f3c294fe5c7`.

---

## 1. Verdict

**`G0_STILL_PASSES`**, and by a wider margin than when it was frozen.

The retrofit did what it was supposed to — it removed a real omission — but the
omission was **not** one-directional, and the headline number moved the opposite
way from the prediction. Both findings are in §4 and §5.

---

## 2. What was built

One canonical layer, two new modules, no second implementation anywhere:

| Module | Responsibility |
|---|---|
| `validation/funding.py` | `FundingSeries`: observed settlements, coverage, gap detection, `realized_funding_pct(entry, exit, side)`. Signed, fail-closed. |
| `validation/trade_costs.py` | `trade_costs(...)` → `TradeCosts`: the decomposition and the single accounting identity. `transaction_cost_r` for admission gates. `funding_summary` for aggregates. |
| `tools/funding_cache.py` | Offline, source-pinned settlement cache with a sidecar manifest (range, count, gap report, sha256). |

The accounting identity, asserted by `TradeCosts.assert_identity` and by tests:

```
net_r = gross_r - fee_r - slippage_r - funding_r
```

All three cost terms are charged on notional, so all three normalise the same
way: a cost of `c` percent is `c/100 * entry_price / stop_distance` in R.

`validation/cost_sensitivity.py` (M00) was **not modified**. Its frozen pin is
still `1cd4db9fe8e8` and G0 stays comparable to its own baseline; the measured
rate is supplied as an override, which produces its own hash `f0163eee67b1`.

### 2.1 Four decisions that could have gone wrong

**Funding is signed.** A long pays a positive rate; a short is credited. 15.15%
of measured BTCUSDT settlements are negative. Clamping funding to a
non-negative cost would have been a falsification, and it would have produced
the answer the task predicted (§5, expectation 1) for the wrong reason.

**Boundary convention `(entry, exit]`.** A settlement at the entry stamp is not
charged (the position was not held when the snapshot was taken); one at the
exit stamp is. Both edges are tested.

**Funding never enters the admission gate.** `H1_MAX_COST_R` / `H2_MAX_COST_R`
reject candidates before resolution. Folding funding in would have changed
*which* trades were taken — forbidden by P1. Funding is a post-resolution term,
which is also the honest ordering: the holding interval is unknown at entry.
The same reasoning keeps `cost_r` in `deep_discovery` transaction-only: it is a
feature, and a feature must be observable at the decision bar.

**Missing data fails closed.** Outside the observed window, or across a known
gap, the answer is `FundingDataUnavailable`. Zero is returned only when the
series proves no settlement was crossed. The one route to a zero on a real hold
is the `FUNDING_NOT_MODELLED` sentinel, which must be named at the call site and
propagates as `funding_modelled=False` into every record and report.

### 2.2 Documented approximation

Funding is charged on notional **at entry price**. Binance's historical
`fundingRate` records carry `markPrice` for only 63.8% of the measured window,
so per-settlement mark-to-market notional is unavailable for the whole history.
The residual error is a few percent of a term worth ~0.06pp per 48h trade —
below 0.005pp — and it normalises the funding term exactly like the fee term.

---

## 3. Coverage: cost paths found and migrated

The task listed nine sites. An independent search found **twelve**, plus one
more downstream caller. All are migrated; none is left inconsistent.

| # | Site | Now |
|---|---|---|
| 1 | `backtest.py` (live `/backtest`) | `trade_costs`; funding fetched per run; `_resolve` now returns `exit_idx` |
| 2 | `tools/deep_backtest.py` `deep_walk` | `trade_costs`, `funding` required keyword-only |
| 3 | `tools/deep_discovery.py` `_snapshot` | `transaction_cost_r` (feature — see §2.1) |
| 4 | `tools/deep_diagnostics.py` `_cost_r` | `transaction_cost_r`; the counterfactual re-charges funding on the extended horizon |
| 5 | `tools/swing_hypothesis_simulator.py` `_cost_r` | adapter over `cost_pct_to_r` |
| 6 | `…simulator.py` `walk_hypothesis` | `trade_costs`, `funding` required |
| 7 | `…simulator.py` `run_simulation` | loads the real series; sentinel unreachable |
| 8 | `tools/swing_hypothesis_walkforward.py` | `transaction_cost_pct` + threaded `funding` |
| 9 | `tools/forecast_metrics.py` | constants from `config`; `modeled_round_trip_cost_pct` → `modeled_transaction_cost_pct`; new `realized_funding_pct` block |
| 10 | `analyzer/outcomes.py` `net_after_costs` | `trade_costs` over the measured 72h interval; new `funding_pct` column |
| 11 | `analyzer/counterfactual.py` | constants from `config`; `FUNDING_NOT_MODELLED` (consumes touches, not net) |
| 12 | `validation/cost_sensitivity.py` (M00) | unchanged, by design |
| +1 | `tools/regime_gate_confirmation.py` | threaded `funding` (found via `record_walk`) |

`scheduler.py` fetches the series once per outcome-tracking cycle and injects
it. `database.py` gains a nullable `funding_pct` on `forecast_outcomes`, with no
backfill — a NULL is exactly how a pre-P1 `net_after_costs` (fees + slippage
only) stays distinguishable from a post-P1 one.

Post-implementation re-search for the duplicated formula
`(2 * TAKER_FEE_PCT + SLIPPAGE_PCT)`: **zero occurrences** outside
`validation/trade_costs.py`, which owns it. A test enforces this going forward,
as does a test that `funding` is keyword-only with no default on every walk.

---

## 4. Data provenance

| | |
|---|---|
| Source | Binance `/fapi/v1/fundingRate`, BTCUSDT, exchange pinned (no failover) |
| Tool | `tools/funding_cache.py` |
| Dataset | `data/funding/binance_BTCUSDT_funding.csv` (untracked, regenerable) |
| sha256 | `83b160384fe816c427ac48c34d148cf1cc8cf3ba7f96df9bc50930ad75e9a592` |
| Settlements | 4799 |
| Range | 2022-04-01T00:00:00Z … 2026-08-17T08:00:00Z |
| Modal interval | 8h, **measured from the data**, not assumed |
| Gaps | none |
| Coverage vs 4h klines (2022-04-28 … 2026-08-11) | complete, with margin at both ends |

Measured rates over that window:

| quantity | value |
|---|---|
| mean | **0.006095 %/8h** |
| median | 0.005879 %/8h |
| min / max | −0.11917% / +0.08815% |
| negative share | **15.15%** |

M00 assumed 0.01 %/8h. The measured mean is **0.61×** that: the frozen
assumption was conservative, not optimistic.

---

## 5. PRE_P1 → POST_P1

PRE was produced by running the frozen simulator at the anchor commit in a
separate git worktree; POST by the same command on the migrated code. Same
dataset, same `--bars 8000`, same profile, same frozen constants, same WF
geometry. Nothing on `P1_TASK.md`'s out-of-scope list was touched.

### 5.1 The invariant P1 must not break

| quantity | PRE | POST |
|---|---|---|
| bars considered (H1 / H2) | 7999 / 7999 | 7999 / 7999 |
| trades (H1 / H2 / swing baseline) | 3 / 497 / 859 | 3 / 497 / 859 |
| gross expectancy (H1 / H2) | 1.74 / 0.2095 | 1.74 / 0.2095 |
| win rate (H1 / H2) | 1.0 / 0.1288 | 1.0 / 0.1288 |
| timeout share (H2) | 0.2153 | 0.2153 |
| max losing streak (H2) | 45 | 45 |
| `avg_cost_r` (H1 / H2) | 0.07 / 0.0303 | 0.07 / 0.0303 |
| unresolved | 0 | 0 |

**Trade population, directions and gross outcomes are identical.** Only net
figures moved, which is the entire permitted effect of P1. `avg_cost_r` is
unchanged because it reports the transaction cost the admission gate saw —
funding is reported separately, not folded in.

### 5.2 What moved

| | PRE net | POST net | Δ attributable solely to funding |
|---|---|---|---|
| current swing baseline (n=859) | 0.0122 | **0.0108** | −0.0014 R/trade |
| H2 (n=497) | 0.1792 | **0.1752** | −0.0040 R/trade |
| H1 (n=3) | 1.67 | **1.6933** | **+0.0233 R/trade** |
| H2 total R | 89.07 | 87.05 | −2.02 R |
| H1 total R | 5.01 | 5.08 | +0.07 R |

### 5.3 Funding, itemised

| | swing baseline | H2 | H1 |
|---|---|---|---|
| total funding R | +0.9717 | +1.9582 | −0.0676 |
| mean funding R/trade | +0.001131 | +0.003940 | −0.022533 |
| median funding R/trade | +0.000583 | +0.003429 | −0.023123 |
| funding as share of total costs | **37.4%** | 18.2% | 47.4% |
| trades crossing ≥1 settlement | 99.65% | 96.98% | 100% |
| longs: n, mean funding R | 404, +0.013695 | 300, +0.011021 | 0, — |
| shorts: n, mean funding R | 455, **−0.010024** | 197, **−0.006844** | 3, −0.022533 |
| largest debit | +0.057484 R | +0.071540 R | −0.019950 R |
| largest credit | −0.039058 R | −0.067130 R | −0.024533 R |

**The finding that matters.** Funding is 37% of the swing baseline's total cost
and yet costs it only 0.0011 R per trade, because the strategy is close to
direction-balanced (404 long / 455 short) and the two sides very nearly cancel:
longs pay +0.0137 R, shorts collect −0.0100 R. The omission P1 corrected was
large in gross terms and small in net terms — and it would have been neither if
the funding term had been clamped to a cost.

---

## 6. The four falsifiable expectations

Recorded in `P1_TASK.md` §"Falsifiable expectations" before the work started.

1. **"Every net figure moves down." — FALSIFIED.** H1's net expectancy *rose*
   from 1.67 to 1.6933. Not a defect: all three H1 trades were shorts, and a
   short collects funding when the rate is positive. The expectation assumed
   funding is always a cost, which the data contradicts 15% of the time and
   which direction contradicts systematically. The prediction, not the
   retrofit, was wrong.
2. **"Verdicts can only flip from accepted to rejected." — HELD**, and no
   verdict flipped in either direction.
3. **"H1, H2 and the regime gate stay rejected." — HELD.** H2's net fell; H1
   rose by 0.023 R on a sample of three trades, which reverses nothing (its
   rejection never rested on expectancy at n=3).
4. **"The +0.13 ranking edge is unaffected." — HELD.** The Ridge ranking path
   carries no direction and opens no position; the full suite reproduces every
   C4.3c/d/e figure unchanged.

Expectation 1 failing is the informative outcome of this task: the project's
own prior was that correcting the omission would only ever hurt.

---

## 7. G0, re-evaluated

Re-run under the exact frozen M00 rule (4h bars, 1.5% stop, 2R payoff), with
only the funding rate substituted from assumed to measured.

| holding | fund (assumed 0.01) | fund (measured) | total | cost/R | need WR |
|---|---|---|---|---|---|
| 1 bar (4h) | 0.005 | 0.003 | 0.135 → **0.133** | 0.090 → **0.089** | 36.3% → 36.3% |
| 3 bars (12h) | 0.015 | 0.009 | 0.145 → **0.139** | 0.097 → **0.093** | 36.6% → 36.4% |
| 6 bars (24h) | 0.030 | 0.018 | 0.160 → **0.148** | 0.107 → **0.099** | 36.9% → 36.6% |
| **12 bars (48h)** | 0.060 | 0.037 | 0.190 → **0.167** | 0.127 → **0.111** | 37.6% → **37.0%** |
| 24 bars (96h) | 0.120 | 0.073 | 0.250 → **0.203** | 0.167 → **0.135** | 38.9% → 37.8% |

**`G0_STILL_PASSES`.** At the 4h swing horizon the round trip costs 0.167%
rather than 0.190%, needing a 37.0% win rate at 2R against 33.3% at zero cost.
No threshold and no strategy parameter was changed to obtain this; the gate
moved because the assumption it was built on was pessimistic.

The binding variable is still stop distance, not holding period, and that
conclusion is unchanged.

---

## 8. Verification

| check | result |
|---|---|
| targeted suite (`tests/test_funding.py`) | 59 passed |
| full suite | **1618 passed, 0 failed** (1559 at the anchor + 59 new) |
| `git diff --check` | clean |
| duplicated legacy formula re-search | 0 occurrences outside its owner |
| frozen M00 pin | `1cd4db9fe8e8`, unchanged |
| `BASELINE.md` / `P1_TASK.md` | not modified |
| generated artifacts | `reports/p1/`, untracked |

The targeted suite covers the sign in all four direction × rate combinations,
zero funding, zero/one/many settlements, a sign change inside a trade, both
timestamp boundaries, millisecond jitter, the accounting identity, R
normalisation, no double counting, fail-closed on missing data and on gaps,
absence of look-ahead, deterministic replay, rate-selection by settlement
stamp, and the integration guards.

---

## 9. Independent audit

Codex audited commit `90bcf60` against the seven risk areas P1 defines. Clean on
five: funding sign, the half-open boundary, look-ahead, the sentinel's
reachability from research entry points, and the invariant that trade
population, entries, exits, directions and gross R are untouched. Two findings:

**Accepted and fixed — a resolved row could be stranded without its costs.**
P1 split two things that used to be one: resolution is about price (stop,
target, or the 72h horizon), while `net_after_costs` additionally needs funding,
which comes from a fetch that can fail. A row resolved during such a failure
kept `resolved = TRUE` with a NULL net, and `forecasts_pending_outcomes`
excluded it from every later cycle — so one transient outage permanently cost
that forecast its cost accounting. Fixed in `database.py` by retrying rows where
`return_72h IS NOT NULL AND net_after_costs IS NULL`, a predicate that cannot
select a pre-P1 row (those always carry a non-NULL net once the horizon
elapsed), so it is a retry and not the backfill P1 refuses.

**Accepted and fixed — the retry itself could starve new work.** A re-audit of
that fix found the regression it introduced: pending rows are served
oldest-first under `LIMIT 200`, so rows whose funding is permanently out of the
live fetch's reach (it reads the last 1000 settlements, ~333 days) would refill
the window on every cycle and block forecasts that had never been measured at
all. The retry is now bounded to `FUNDING_RETRY_WINDOW_DAYS = 30`, far inside
the fetch reach and far beyond any transient outage the retry exists to absorb.
Both fixes carry regression tests.

**Raised, then withdrawn by the reviewer — the strict right edge of `covers()`.**
The reviewer notes that a trade exiting one hour after the last observed
settlement is refused even though the settlement inside its interval is known.
On re-audit the reviewer accepted this as intended fail-closed behaviour rather
than a defect. Concluding that no settlement exists between
`last_ms` and the exit requires assuming the cadence continued, and this module
refuses that assumption at both edges for the same reason. The alternative —
charging only what happens to be in the snapshot — is silent undercharging,
which is precisely the defect P1 exists to remove. The practical cost is nil:
research datasets are built with margin at both ends (funding from 2022-04-01
against klines from 2022-04-28), and on the live path the affected rows are
those whose horizon ended within the last settlement interval, which the retry
above picks up on the next cycle once that settlement lands.

---

## 10. Next blocker

P1 is closed. Phase A′ (`ARCHITECTURE.md` §3.2) is now unblocked: re-evaluate
H1, H2, the confluence score and C4.3e under the corrected apparatus. Per §3.2
it is mandatory before phase B, and per §9 revisions happen only at a gate.

The next gate is **G1 — apparatus trustworthy**, which requires M01–M04
(`deflated_sharpe`, `triple_barrier`, `sample_weights`, `cpcv`) and a clean
negative-control suite. Nothing downstream is admissible until G1 passes.
