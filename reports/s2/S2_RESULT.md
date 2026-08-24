# S2 — universe and labels: result

Criteria: `reports/s2/S2_SPEC.md`, written **before a single label was
computed**. Anchor: S1 (`reports/s1/S1_RESULT.md`), 734 USDT pairs.

## Verdict: `S2_PASS`

The number the whole stage existed to produce:

> **The base rate of `CLEAN_2X(180d)` is 18.2%** — 1,334 clean doubles out of
> 7,324 resolved (symbol, decision-date) events across 98 monthly dates,
> 2018-01 → 2026-02.

Neither pre-registered branch of `S2_SPEC.md` §9 is triggered. 18% is far
above "too rare to measure precision@K" and far below "most eligible assets
double". **The target defined in S0 §3 stands unchanged**, and so does the
+100% / −40% / 180-day geometry.

---

## 1. What the panel says

| quantity | value |
|---|---|
| decision dates | **98** (first of each month, 2018-01-01 → 2026-02-01) |
| **independent 180-day windows** | **16.3** — the honest N |
| events | 7,353 · resolved **7,324** · gap-unresolved 29 · **censored 0** |
| **base rate** | **0.1821** (1,334 hits) |
| eligible symbols per date | min 0 · median **84** · max 238 |
| same-bar ties | **1** |

`censored = 0` is the spec's own self-check passing: §1 stops the decision grid
a full horizon before the panel ends, so no living coin can run off the edge.
A non-zero count here would have been a bug report.

The median eligible universe of **84** is below S0 §5.3's 150–250 estimate. The
binding screen is the $5M liquidity floor, not the size cap — the cap of 250
almost never binds. Worth knowing before S3 builds cross-sectional ranks on
thin cross-sections.

## 2. The survivorship gap, as a number

S0 §7.4 asked for this to be published rather than assumed. Same labels, same
dates, same rules — only the universe differs:

| universe | base rate | n |
|---|---|---|
| **full (point-in-time)** | **0.1821** | 7,324 |
| survivors only — symbols still trading today | 0.2090 | 5,159 |
| symbols that were later delisted | 0.1182 | 2,165 |

> **Survivorship gap: +2.68 percentage points, +14.7% relative.**

A naive universe built from today's symbol list would report a 21% success rate
where the truth is 18%. That is the size of the bias for *this* metric on
*this* horizon. It is smaller than folklore suggests and entirely real: 2,165
of 7,324 outcomes — **30%** — belong to coins that no longer exist, and they
succeed at barely half the rate of the survivors.

**981 of those events are deaths inside the horizon**, resolved as failures by
§4.1 rather than dropped. Every one of them would have been silently missing
from a survivors-only panel.

## 3. The finding that matters most for the product

The base rate is the headline, but it is not the most useful number here.

| across all 7,324 resolved events | value |
|---|---|
| median MAE (deepest drawdown in the window) | **−54.8%** |
| median terminal return at 180 days | **−33.1%** |
| **median excess return vs BTC (log)** | **−0.409** |
| median MFE | +40.3% |
| median days to 2x, among hits | **56** |
| median MAE of the hits | −23.9% |

> **The median liquid altcoin loses a third of its value over 180 days and
> underperforms simply holding BTC by roughly 34%.**

Picking at random from the eligible universe is not a neutral act — it is a
substantially losing one. This sets the bar S3 and S4 must clear: a scanner
does not have to beat zero, it has to beat BTC, from a starting distribution
whose middle is deeply negative. S0 §9 already required BTC as a baseline;
this is the number that makes it the *hard* one.

The hits are reachable, though: a median drawdown of −24% before doubling,
and a median 56 days to get there. Winners do not, typically, require sitting
through a −39% hole.

## 4. Regime dependence is severe

| regime (point-in-time, BTC 200d) | events | base rate |
|---|---|---|
| bull | 4,547 | 0.171 |
| range | 1,133 | **0.314** |
| bear | 1,644 | 0.121 |

By year, the same story louder:

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| 0.133 | 0.250 | **0.523** | 0.313 | 0.053 | 0.259 | 0.195 | **0.050** | 0.030 |

A tenfold spread between 2020 and 2025. **S0 §21 item 4 is confirmed, not
avoided**: most of the outcome is the calendar. Two consequences carry into
S3/S4, and both were pre-registered:

1. every metric must be reported **split by regime**, never only pooled;
2. the honest N is ~16 windows, and those windows are not exchangeable — a
   scanner validated mostly on 2020–2021 has been validated on two years.

`range` scoring above `bull` is not what one would guess. It is reported as
measured, not explained away: the §6 definition puts early recovery phases —
price crossing back over a still-falling average — into `range`, and those are
exactly the windows where beaten-down alts double. Whether that is a real
effect or an artifact of the regime definition is an S3 question, and it must
not be settled by trying definitions until one flatters a result.

## 5. Method notes

- **M02 was reused unmodified.** `sigma = 1.0` with `upper_mult = 1.0`,
  `lower_mult = 0.4`, `vertical_bars = 180` turns volatility-scaled barriers
  into fixed ±ratios, exactly as S0 §6 predicted. Frozen config hash:
  `f2346ef283a5`.
- **The tie rule cost nothing here.** One event in 7,353 had a single daily bar
  spanning both barriers. The pessimistic convention is inherited and its
  frequency is now measured rather than assumed.
- **Death is an outcome.** 981 resolved deaths; 0 censored; the two are never
  merged (§4.1).
- **29 gap-unresolved events** (0.4%), each one an event whose window crossed a
  hole in the bar grid. M02 refuses to guess at a first touch across missing
  bars, and the count is small enough to be a footnote rather than a threat.
- **Snapshots are write-once.** 98 snapshots, each carrying every screen value
  for every symbol — including the rejected ones and the reason — plus a
  content hash.

### 5.1 Two data defects found, both by existing guards

1. **A bar that closed before it opened.** `KLAYUSDT`'s final bar carried a
   `close_time` earlier than its own `open_time`. M02's timestamp guard refused
   it — the guard doing precisely its job. A scan then found **300 malformed
   bars across 265 symbols**: 249 are the last bar of a delisted pair, a
   partial day closing the moment trading stopped (1INCHDOWNUSDT's is three
   hours long), and 48 sit mid-history. S1's loader checked `open_time`
   ordering, which stays perfectly monotonic through all of them. The panel was
   re-ingested with per-bar consistency validation; the loader now re-checks it
   on every load, since files outlive the run that wrote them. Symbols
   reporting gaps rose from 22 to 41 — dropping a mid-history partial bar
   *creates* a visible gap where a corrupt bar used to hide one.
2. **The write-once guard fired on a real mistake.** Three snapshots from a
   smoke run had been built on the pre-fix panel. The rerun refused to
   overwrite them, and they were deleted deliberately rather than silently
   replaced. That is the guard working as designed.

### 5.2 One rule that would have deleted real assets

The leveraged-token screen cannot be a suffix match. **JUP** (Jupiter) and
**SYRUP** (Maple) both end in `UP`, and a naive rule removes them from every
universe, permanently and invisibly. The implemented rule requires the prefix
to itself be an asset the venue lists — `ADAUP` strips to `ADA`, `JUP` strips
to `J`, which is nothing. Both cases are pinned by tests.

## 6. Verification

- **59 targeted tests**: `test_spot_universe.py` (25), `test_spot_labels.py`
  (13), `test_spot_ingestion.py` (21).
- **Full suite: 2001 passed, 0 failed.**
- The pre-existing G1 import-direction guard caught `spot/` as a new top-level
  package importing `labeling`. Rather than widen its skip list silently, two
  guards were added: production may not import `spot`, and `spot` may not
  import config, database, scheduler, analyzer, bot, signal_engine, pipeline or
  risk. An exemption in one direction now costs a test in the other.
- No runtime file, no futures logic, no database, no Telegram code touched.
  Phase A′ remains paused. Nothing committed.

## 7. Next stage: S3 — baselines and the first feature families

S2 answers *what happened*. S3 asks *what could have been known beforehand*,
and it must do so under two constraints this stage just made concrete:

- **the baseline to beat is BTC**, and the median eligible coin loses to it by
  ~34% over 180 days;
- **~16 independent windows, dominated by regime.** Every feature family is
  declared in the trial registry before evaluation, evaluated **alone** (S0
  §8.2), and reported split by regime.

S3 builds the seven baselines of S0 §9 and feature families 1–4, and nothing
else: no blend, no score, no model.
