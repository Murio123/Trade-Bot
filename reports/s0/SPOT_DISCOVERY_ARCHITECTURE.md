# S0 — Spot Opportunity Discovery: architecture

Stage S0. **Read-only.** No production code, tests, database, Telegram UI,
dependencies or futures logic were touched, nothing was committed or pushed,
and no sealed holdout was opened. What follows is a specification.

---

## 1. Executive decision

The project pivots from *"does BTC futures have a directional edge"* to
*"which liquid spot assets have an unusually attractive upside profile over the
next several months"*. That is a different research question on a different
data shape: **cross-sectional** rather than single-series, **long-horizon**
rather than intraday, **spot** rather than levered.

Three consequences follow, and they set the whole of S0:

1. **The measurement apparatus survives the pivot; the strategy work does
   not.** Triple barrier, sample uniqueness, purge/embargo, the trial registry,
   the append-only ledger and the manifest-checked data loader are all
   problem-agnostic. H1, H2, the confluence score and the profile thresholds
   are BTC-futures artifacts and are frozen where they lie.
2. **The binding constraint is not modelling, it is sample size.** A 180-day
   label over ~7 years of usable altcoin history yields roughly **14
   non-overlapping windows**. That is the honest N of this problem. It is small
   enough that complex models cannot be validated, which is why this
   architecture puts an interpretable rule layer first and defers ML.
3. **The dominant failure mode is not overfitting, it is point-in-time
   leakage.** Today's top-100, today's circulating supply and today's symbol
   list all encode the future. §7 makes the defence architectural rather than
   procedural.

Verdict: **`SPOT_ARCHITECTURE_READY`**. Next stage: **S1 — multi-asset
point-in-time spot data**.

---

## 2. Product objective

> Rank liquid spot crypto assets by the credibility of a large medium-term
> appreciation, and keep a falsifiable record of every ranking.

The system does not promise 2x. It produces an **ordering under uncertainty**
plus the evidence for it, and it records each ordering before the outcome is
knowable so that the ordering can be proven wrong later.

The eventual user question — *"what is worth accumulating now?"* — is answered
by a shortlist with stated reasons, stated risks and a stated invalidation
condition. Anything the system cannot currently estimate is **omitted**, not
approximated (§12).

---

## 3. Definition of a 2x opportunity

Vague labels ("gem", "high potential") are unusable because they cannot be
scored against history. The target below can.

### 3.1 Why "+100% within H" alone is wrong

A coin that falls −70% and then reaches +100% from the original reference price
satisfies "hit 2x" while being an asset almost nobody could hold to the payoff.
A target that cannot distinguish those two paths trains the system to find
lottery tickets.

### 3.2 The primary label: a path-aware clean 2x

**`CLEAN_2X(H)` — binary.** From reference price `P0` at decision date `t`:

| barrier | value | meaning |
|---|---|---|
| upper | `2.00 × P0` | the outcome of interest |
| lower | `0.60 × P0` | −40%: the drawdown beyond which the path is disqualified |
| vertical | `t + H` | the horizon |

`CLEAN_2X = 1` iff the **upper barrier is touched first**. Lower first, or
neither by `H`, is 0.

Two properties make this the right primary:

- It is **exactly a triple-barrier problem**, so `labeling/triple_barrier.py`
  (M02) applies unchanged, including its audited same-bar tie rule — a bar that
  touches both barriers resolves to the adverse one (G1 amendment A1). That
  rule is worth more here than it was in futures: 180-day daily bars on a
  small-cap can straddle both barriers.
- The −40% floor is a **held-position** criterion, not a stop-loss
  recommendation. It encodes "would a human plausibly still be holding when the
  payoff arrived".

`−40%` is a parameter and must be frozen before labels are inspected (S2), with
`−50%` and `−30%` reported as sensitivity only — never selected on outcome.

### 3.3 The record vector — five quantities, no more

`CLEAN_2X` discards information that later analysis needs, so every
`(asset, decision_date)` row also stores:

| field | why it is in the minimum set |
|---|---|
| `mfe` — max favourable excursion over H | the upside that was actually available |
| `mae` — max adverse excursion over H | how much pain preceded it |
| `days_to_2x` | a 2x on day 170 is not a 2x on day 20 |
| `terminal_return` at H | what a buy-and-hold to horizon returned |
| `excess_vs_btc` at H (log) | whether the pick beat simply holding BTC |

Everything else — "opportunity quality", risk-adjusted upside — is **derived**
from these five (e.g. `mfe / |mae|`), not stored. No sixth field is added
without a stated question it answers.

### 3.4 Not a probability, yet

The MVP produces a **rank**, not `P(2x)`. A probability requires calibration
against enough resolved outcomes to be meaningful, and at N≈14 independent
windows the first calibration curve would be noise wearing a decimal point.
`tools/forecast_platform/calibration_engine.py` already exists for the day the
sample supports it (§12, §16 S7).

---

## 4. Primary horizon: 180 days

Recommended: **H = 180 days**, single primary. 90d and 365d are reported as
secondary labels only where they cost nothing to compute.

| horizon | argument for | argument against | verdict |
|---|---|---|---|
| 90d | fastest falsification; ~28 non-overlapping windows | a 2x in 90 days is overwhelmingly a momentum/listing burst, not a re-rating; base rate outside bull phases is near zero, so the positive class nearly vanishes | secondary |
| **180d** | long enough for a re-rating thesis to play out; ~14 independent windows; a live forecast matures within one product cycle | still cycle-sensitive | **primary** |
| 365d | matches the "accumulate" framing best | ~7 independent windows over usable history, and every window spans a regime change; you wait a year to learn you were wrong | secondary, later |

The deciding argument is falsifiability speed against sample size. 180d is the
longest horizon that still leaves a double-digit number of independent
observations, and the shortest that measures a re-rating rather than a squeeze.

---

## 5. The investable universe

The scanner must not "discover" illiquid garbage because microcaps sometimes
10x. Eligibility at decision date `t` is a **filter, computed only from data
knowable at `t`**.

### 5.1 MVP eligibility rules

| rule | threshold (MVP) | why |
|---|---|---|
| venue | listed spot on Binance, quoted in USDT | one venue keeps PIT tractable; expansion is S7 |
| not a stablecoin | peg-list + realized 90d vol floor | a stablecoin cannot 2x; it pollutes the denominator |
| not wrapped/staked/derivative | WBTC, stETH, WBETH, BTCB… | duplicates an underlying already in the universe |
| not a leveraged token | `*UP`, `*DOWN`, `*BULL`, `*BEAR` | path-dependent decay, not an asset |
| listing age | ≥ 180 days at `t` | features need history; new listings are a separate problem with its own dynamics |
| price history | ≥ 200 daily bars ending at `t` | warmup for medium-term features |
| liquidity | median 30d USDT volume ≥ **$5M** at `t` | the floor that separates "investable" from "exit is the problem" |
| activity | traded in each of the last 30 days | filters dead assets without using future knowledge |
| size cap | top **250** eligible by trailing 30d dollar volume | keeps the panel computable and the tail honest |

Market cap and FDV are **deliberately absent from MVP eligibility** — see §8.2.
Liquidity is used as the size proxy because it is derivable from data we can
actually obtain point-in-time.

### 5.2 Point-in-time membership, mechanically

1. For each decision date, an eligibility snapshot is computed and written
   **once** to `data/universe/YYYY-MM-DD.json`, then never recomputed.
2. Snapshots are append-only and content-hashed, exactly as the trial registry
   is; a rerun that produces different bytes is a defect, not an update.
3. A snapshot may only read bars with `close_time ≤ t`.
4. **Delisted assets remain in every snapshot they qualified for**, with their
   subsequent (often terminal) price path. They are the entire reason
   survivorship bias exists, so they are the one thing that may never be
   dropped.
5. Membership is never derived from a present-day symbol list, screener export
   or ranking page.

### 5.3 Recommended initial universe

**Binance spot USDT pairs, daily bars, 2019-01-01 → present, including
delisted symbols**, filtered by §5.1. Expected size: roughly 150–250 eligible
names per date, and materially fewer before 2021. Expansion (other venues,
other quote assets, longer history) is S7 and changes no interface.

---

## 6. Existing data inventory

Determined by inspecting the repository, not from memory.

### ALREADY_AVAILABLE

| asset | location | note |
|---|---|---|
| BTCUSDT OHLCV, 7 timeframes (15m…1d) | `data/klines/` (20 MB, manifest per file) | **BTC only** |
| BTCUSDT funding history | `data/funding/` | futures-only concept |
| Source-pinned ingestion with gap reports | `tools/kline_cache.py` | exchange pinned, manifest sidecar, pagination without cap |
| Fail-closed dataset loader | `tools/kline_dataset.py` | manifest read first, every claim re-verified on the data |
| Indicator/analysis library | `analyzer/*` (30 modules) | volatility, correlation, structure, volume profile |
| Full validation stack | `validation/`, `labeling/` | M01–M04, negative controls, trial registry |
| Append-only forecast ledger | `volatility/ledger.py` | **already symbol-keyed** — "a second asset needs no migration" |
| Storage with symbol columns | `database.py` | signals/forecasts/journal all carry `symbol` |
| Telegram bot + scheduler + Railway | `bot/`, `scheduler.py`, `Procfile` | single-symbol at runtime (§19) |

Dependencies are `pandas`, `numpy`, `httpx`, `asyncpg`, `APScheduler`,
`matplotlib`, `anthropic` — **no sklearn, no scipy**. Everything in
`tools/forecast_platform/` was written under that constraint and stays valid.

### REUSABLE_WITH_CHANGES

| component | change needed |
|---|---|
| `analyzer/binance.py` | **it is a futures client** — every path is `/fapi/v1/*`. Spot needs `/api/v3/klines`, `/api/v3/exchangeInfo`. New methods, same shape. |
| `tools/kline_cache.py` | one symbol per invocation → iterate a symbol list; add `market=spot`; needs a bulk path (§6 MISSING) |
| `labeling/triple_barrier.py` | barriers are `mult × sigma` where sigma is volatility as a fraction of price. Fixed ratios are expressible by passing a **constant sigma of 1.0** with `upper_mult=1.0, lower_mult=0.4`, so this is plausibly a zero-code-change reuse — to be confirmed in S2, not assumed. |
| `validation/cpcv.py` | event-grouped → must become **date-grouped panel** geometry (§13) |
| `labeling/sample_weights.py` | uniqueness over overlapping event spans → same maths, spans now 180d and shared across assets |
| `validation/trial_registry.py` | usable as-is; needs a new family scope, no code change |
| `volatility/ledger.py` | usable as-is for spot forecasts; horizon in days rather than bars |

### MISSING — required before anything can be measured

1. **Spot daily OHLCV for the full universe, including delisted symbols.**
   This is the whole of S1 and the project's single biggest data risk.
2. **Listing and delisting dates per symbol** (drives PIT eligibility).
3. **Point-in-time dollar volume** — free once (1) exists.
4. **Immutable universe snapshots** — computed, not sourced.

### OPTIONAL_LATER — ranked by expected value against cost

| rank | data | value | cost / risk | stage |
|---|---|---|---|---|
| 1 | point-in-time circulating supply → market cap | high: valuation context, size effects | **high** — historical supply is restated by every provider; a naive pull imports future knowledge | S7 |
| 2 | token unlock schedules | high: a known-in-advance supply shock is one of the few genuinely forward-looking features | high — sparse, unstandardised, often only known retrospectively | S7 |
| 3 | TVL / fees / revenue (DefiLlama) | medium: fundamental growth for a subset | medium — restatement risk, covers maybe 30% of universe | S7 |
| 4 | derivatives context (OI, funding, liquidations) | medium | low — partially built already | S7 |
| 5 | on-chain actives, dev activity | low-medium | high — per-chain plumbing | later |
| 6 | narrative / sector labels | **negative as usually built** | retrospective sector labelling is a leak, not a feature | not needed |

MVP takes items 1–4 of MISSING and **nothing** from OPTIONAL_LATER. A scanner
built on price, volume and liquidity alone is honest and testable; one built on
restated fundamentals is neither.

---

## 7. Point-in-time correctness (first-class requirement)

This problem leaks in ways the futures problem could not. The defence is
structural.

### 7.1 The rule

> A feature value used at decision date `t` must have been **observable at
> `t`**, at the value it had then.

### 7.2 The mechanism

- **As-of store.** Every stored record carries both `value_date` (what period
  it describes) and `observed_at` (when it became knowable). Readers take an
  `asof` argument and may only see rows with `observed_at ≤ asof`. Restatements
  append a new vintage; they never overwrite. This generalises the as-of
  alignment already shipped for aux frames in C1.3c.
- **Immutable universe snapshots** (§5.2).
- **No present-tense source may enter historical evaluation.** Today's screener,
  today's supply, today's symbol list are inputs to *live* operation only.

### 7.3 The named leaks, and what stops each

| leak | defence |
|---|---|
| today's market cap used historically | mcap excluded from MVP entirely (§6) |
| today's circulating supply used historically | same |
| today's token universe used historically | PIT snapshots (§5.2) |
| delisted/dead tokens vanish from history | snapshots retain them; ingestion explicitly targets delisted symbols |
| future exchange listings | listing-age rule uses listing date ≤ t |
| revised fundamental metrics | vintage-keyed as-of store |
| future unlock knowledge | unlocks excluded from MVP |
| retrospective sector/narrative winners | excluded permanently as usually constructed |
| future volume deciding historical eligibility | eligibility computed from trailing 30d at `t` only |

### 7.4 Leakage is tested, not asserted

Mirroring the G1 negative-control suite, S4 ships **PIT controls** that must
fail loudly:

- a deliberately leaked feature (forward 30d return) must be detected by the
  harness as anomalously predictive;
- date-shuffled labels must destroy all measured skill;
- a universe built from the present-day symbol list must produce a visibly
  better result than the PIT universe — the size of that gap **is** the
  survivorship bias, and it gets published.

The third control is the one that matters. It converts survivorship bias from a
worry into a number.

---

## 8. Candidate-generation architecture

### 8.1 The layering

```
    ingestion (spot OHLCV, incl. delisted)
            ↓
    as-of store  (value_date + observed_at)
            ↓
    PIT universe filter  →  immutable daily snapshot
            ↓
    feature families  (independent, versioned, each testable alone)
            ↓
    candidate generator  (transparent screens → a SET, not a score)
            ↓
    ranking rule  (ONE statistic, validated against baselines)
            ↓
    opportunity report  →  append-only forecast ledger
```

### 8.2 The rule that this project learned the hard way

**No composite weighted score in the MVP.** The previous cycle produced an
impressive-looking confluence score that was never validated, and the trial
registry now records what that search cost. The architectural answer is not
discipline, it is structure:

1. Every feature family is computed and stored **independently**, versioned.
2. Every family is evaluated **alone** against the baselines of §9, on the
   metrics of §10, with its trial declared in the registry beforehand.
3. A family may enter the ranking only after it has beaten the baselines by
   itself.
4. The first shipped ranking is **a single statistic**, not a blend.
5. Weights are earned later, from a fitted model whose lift over the single
   statistic is itself measured — never assigned by judgement.

### 8.3 Feature families (MVP: 1–4 only)

| # | family | MVP? | notes |
|---|---|---|---|
| 1 | liquidity / investability | **yes** | already a filter; also a feature (volume trend) |
| 2 | market regime (BTC trend, breadth) | **yes** | a conditioner, not a ranker: 2x base rate is regime-dominated |
| 3 | relative strength vs BTC / vs universe | **yes** | the strongest prior from the cross-sectional literature |
| 4 | medium-term momentum (3–12m, skip-month) | **yes** | the classic cross-sectional effect; also the baseline to beat |
| 5 | trend quality (path smoothness, MAE of the trend) | later | |
| 6 | volatility / drawdown state | later | plausibly the best partner to (3)/(4) |
| 7 | volume participation (volume-price divergence) | later | |
| 8 | valuation / mcap context | S7 | blocked on PIT supply |
| 9 | supply / unlock risk | S7 | blocked on unlock data |
| 10 | fundamental growth (TVL/fees) | S7 | blocked on coverage |
| 11 | catalyst / event | not planned | unfalsifiable as usually built |

Families 3 and 4 overlap by construction; that is deliberate — 4 is the
baseline, 3 is the candidate, and the question is whether 3 adds anything to 4.

---

## 9. Baselines

Defined **before** any model exists, run on the identical PIT universe,
identical decision dates and identical costs.

| baseline | why it is here |
|---|---|
| random eligible asset (bootstrapped) | the true null; gives the 2x base rate |
| buy-and-hold BTC | the opportunity cost the user actually has |
| buy-and-hold ETH | the second thing they would otherwise do |
| equal-weight eligible universe | what "just buy the market" returns |
| cap-weighted universe | S7 (needs PIT mcap); until then, volume-weighted |
| top-K by 6-month momentum | the simplest known cross-sectional effect |
| top-K by 90-day relative strength vs BTC | the simplest thing the scanner will resemble |

The C4.3 lesson is explicit here: a weak baseline made a model look impressive,
and the honest baseline later recovered 57% of its skill for none of its
machinery. **The scanner's headline number is its lift over the *best*
baseline, never over random.**

---

## 10. Evaluation metrics — the smallest useful set

The primary question, stated so it can be answered "no":

> **Do the assets ranked highest subsequently produce clean-2x outcomes more
> often, and better forward returns, than eligible alternatives and than the
> best simple baseline?**

Four metrics. Not more.

| # | metric | reads |
|---|---|---|
| 1 | **precision@K for `CLEAN_2X`** (K = 5, 10, 20) vs universe base rate | does the top of the list actually contain the outcomes |
| 2 | **median 180d forward return of Top-K** vs universe median, vs BTC | does it pay, including when it misses 2x |
| 3 | **MAE distribution of Top-K** (median, 90th pct) | is the payoff reachable, or does it come through unholdable drawdown |
| 4 | **lift over the best baseline of §9**, with a block-bootstrap interval over decision dates and deflation for the number of rules tried | is the difference distinguishable from search |

Reported alongside, always: the number of independent date blocks, and results
split by BTC regime (§13.5). Calibration (Brier/ECE) enters only when
probabilities do (S7).

Not used as headline: total return of a simulated portfolio. It hides selection
skill behind position sizing and rebalancing choices that are not what is being
tested.

---

## 11. Validation design

The futures geometry must not be forced onto this problem. What changes is the
grouping; what stays is the leakage property.

### 11.1 Panel shape

Rows are `(asset, decision_date)`. Decision dates are a **monthly grid**, not
daily: with a 180-day label, daily dates give 180× redundancy and a false sense
of sample size.

### 11.2 The blocking rule

Folds are blocked **by date, never by asset**. All assets sharing a decision
date resolve together and must sit in the same fold; splitting them puts an
asset's neighbours-in-time on both sides of the boundary and leaks the common
factor.

### 11.3 Purge and embargo

Purge = the full label span (180d) on **both** sides of every test block —
this is exactly the two-sided purge M04 introduced for combinatorial splits.
Embargo of one further month absorbs the monthly grid's edge effects.

### 11.4 Overlap and effective sample size

Overlapping 180d labels are handled by M03's uniqueness weighting, unchanged in
mechanism. But the number that governs conclusions is the count of
**non-overlapping windows**: ~14 over 2019–2026. Every interval is computed by
block bootstrap over those blocks — never over the ~30 000 asset-date rows,
which would understate uncertainty by more than an order of magnitude.

**This is the project's binding constraint and it is stated up front so it
cannot be argued away later:** at N≈14, a model with many free parameters
cannot be distinguished from search. That is the reason §8.2 exists.

### 11.5 Cross-sectional dependence and regime

Crypto assets share one dominant factor. Two mitigations, both mandatory:

- every metric reported **raw and BTC-relative**;
- every metric reported **split by BTC regime** (bull / bear / range, defined
  point-in-time from BTC's own trend). A scanner that works only in 2021-style
  conditions is a valid finding, but it must be labelled as one rather than
  averaged into a flattering total.

### 11.6 Delistings and new listings

A delisted asset's label resolves along its actual path — usually to a loss —
and it stays in the panel. Dropping it is the survivorship bias. New listings
enter only when they pass the 180-day age rule.

### 11.7 What is reused, and what is adapted

| module | status | note |
|---|---|---|
| **M02** triple barrier | **reused directly** | the 2.00/0.60/180d label *is* a triple barrier, and fixed ratios fall out of a constant sigma; inherits the audited same-bar tie rule and the frozen-multiplier discipline (`content_sha256`) |
| **M03** sample weights / uniqueness | **reused directly** | overlapping spans are the same problem; spans now cross assets |
| **M04** CPCV | **adapted** | two-sided purge logic reused; grouping changes from events to dates; leakage-pair assertion kept |
| **M01** DSR | **reused, narrowed** | correct for return-series baselines; **not** the right statistic for precision@K, which uses block bootstrap + explicit multiple-testing accounting |
| **G1.1** trial registry | **reused as-is** | new family scope `(spot_2x_discovery, *, multi, clean_2x_180d)`; the spot family starts at `n_trials = 0` — the futures 100 belong to a different family and do not transfer, but §6.2 ancestry still binds any rule descended from futures work |
| negative controls | **adapted** | become the PIT controls of §7.4 |

---

## 12. What a recommendation may contain

A field ships only if it is defensible today.

**Defensible at MVP**

- symbol, current price, quote venue;
- eligibility: liquidity percentile, 30d median dollar volume, listing age;
- rank and rank percentile within the eligible universe;
- the feature values driving the rank, each with its universe percentile;
- historical context: current drawdown from ATH, realized volatility, 180d
  return vs BTC;
- **invalidation**: an explicit, checkable condition (falls out of the eligible
  universe; relative strength drops below the Kth percentile);
- evidence tier: which of the claims rest on validated evidence and which are
  descriptive.

**Not shipped until earned**

| field | blocked on |
|---|---|
| `2x probability` | calibration against enough resolved outcomes (S7) |
| `expected upside` | a fitted conditional distribution, not a point guess |
| `downside risk` as a number | same; the MAE distribution is descriptive, not predictive |
| tokenomics risk | unlock data (S7) |
| fundamental commentary | TVL/fees coverage (S7) |
| `Buy gradually / Wait for pullback` | an execution study nobody has run |

Status vocabulary at MVP is deliberately coarse — **`Candidate` / `Watch` /
`Not eligible`** — with `Accumulate` and `Avoid` withheld until the evaluation
of §10 justifies a stronger verb.

> If a probability cannot be estimated, no probability is displayed. A
> plausible-looking number is the most expensive thing this system could ship.

---

## 13. Daily workflow (eventual)

```
  1. update spot OHLCV for the tracked universe          (idempotent, manifested)
  2. build the PIT eligibility snapshot for today        (write-once, hashed)
  3. compute market regime from BTC + breadth
  4. compute feature families for every eligible asset
  5. rank; take Top-N
  6. WRITE FORECASTS TO THE APPEND-ONLY LEDGER           ← before any display
  7. render the Telegram report
  8. mature yesterday's due forecasts, append outcomes
```

Step 6 precedes step 7 deliberately: the record is created before anyone sees
the recommendation, which is what makes the history falsifiable rather than
curated. `volatility/ledger.py` already enforces write-once plus
maturation-by-horizon and is symbol-keyed, so it is reused rather than rebuilt.

---

## 14. MVP scope

**MVP — the target is ~1 week of working sessions**

- daily spot OHLCV for ~150–250 Binance USDT assets, 2019→now, **including
  delisted**;
- as-of store + immutable PIT universe snapshots;
- `CLEAN_2X(180d)` labels via M02 + the five-field record vector;
- feature families 1–4 (§8.3), each computed independently;
- all seven baselines of §9;
- the four metrics of §10 over 2019–2026, date-blocked, regime-split;
- PIT leakage controls (§7.4), including the published survivorship gap;
- one offline CLI runner and one report. **No Telegram, no ML, no database
  changes, no probabilities.**

**LATER** — PIT market cap and supply; unlock data; TVL/fees; probability +
calibration; learned ranker; Telegram integration; daily automation; multi-venue.

**NOT NEEDED** — narrative/sector labels; microcaps below the liquidity floor;
intraday data for a 180-day question; leveraged tokens; any port of the futures
strategy logic.

---

## 15. Components reused from the existing bot

| component | disposition | note |
|---|---|---|
| `tools/kline_cache.py`, `tools/kline_dataset.py` | **ADAPT** | the manifest + gap-report + fail-closed-loader design is exactly right for a 250-asset panel; needs spot endpoints and a symbol loop |
| `analyzer/binance.py` | **ADAPT** | futures-only today; add `/api/v3` spot klines + exchangeInfo |
| `analyzer/exchange.py` (failover) | **KEEP** | venue failover is more valuable multi-asset, not less |
| `labeling/triple_barrier.py` | **KEEP** | fixed ratios via constant sigma; extension only if S2 disproves that |
| `labeling/sample_weights.py` | **KEEP** | |
| `validation/cpcv.py` | **ADAPT** | date-grouped panel geometry |
| `validation/deflated_sharpe.py`, `research_dsr.py` | **KEEP** | narrower role (§11.7) |
| `validation/trial_registry.py`, `trial_history.py` | **KEEP** | governance is unchanged by the pivot |
| `volatility/ledger.py` | **KEEP** | already symbol-keyed |
| `tools/forecast_platform/*` | **KEEP, dormant** | feature store, model registry, calibration engine all become relevant at S7 |
| `analyzer/indicators.py`, `volatility.py`, `correlation.py` | **KEEP** | pure functions, asset-agnostic |
| `database.py` | **KEEP** | schema already carries `symbol` |
| `bot/*`, `scheduler.py` | **FREEZE for now** | works; single-symbol (§19); untouched until S5 |
| `signal_engine/*`, `risk/`, `pipeline.py`, `backtest.py` | **FREEZE** | futures decision logic; not deleted, not used |
| `analyzer/onchain.py`, `macro.py`, `news.py` | **FREEZE** | BTC-context oriented; revisit at S7 |
| `volatility/ranker.py`, Ridge line | **FREEZE** | limited validated ranking information on a volatility target; not a directional spot selector |
| `analyzer/liquidation_map.py`, `fvg.py`, `order_blocks.py`, `divergence.py` | **REMOVE_LATER** | intraday futures microstructure, no role in a 180-day spot question |

**Nothing is deleted in S0.**

---

## 16. Components frozen from the futures track

Phase A′ is **PAUSED, not cancelled, not deleted**. Frozen and out of scope:
H1/H2 re-evaluation, any new BTC futures entry, futures thresholds, stops,
leverage, funding work, intraday revival, any new confluence score, Ridge
development, LightGBM, and every sealed futures holdout.

Historical facts stand unchanged and are not rewritten by this pivot:

- H1 rejected; H2 rejected; confluence score never validated;
- no directional BTC edge established;
- Ridge has limited validated ranking information on a volatility target and is
  not a spot-selection model;
- **G1 = `G1_INDETERMINATE`**; the option-3 governance decision of 2026-08-19
  stands as recorded;
- **G1.1 = `G1_1_PASS`**, `n_trials = 100` for the futures family, and the
  trial registry remains the governance mechanism for *this* track too.

---

## 17. Multi-asset blockers

`reports/d13/multi_asset_blockers.md` already catalogued these during D1.3 and
the list survives inspection. In dependency order:

1. **`config.SYMBOL` is process-global** (`config.py:76-77`), read once at
   import. 22 runtime call sites reach for the global (`scheduler.py` ×9,
   `bot/handlers.py` ×10, `pipeline.py` ×3) plus display strings in
   `bot/formatting.py` and `bot/charts.py`. **One process = one asset.** This is
   the single biggest blocker to a multi-asset *product*.
2. **`db.open_trades()` is symbol-blind** (`database.py:414`) — a correctness
   bug the moment a second asset exists, not merely a limitation.
3. **The exchange client is per-process and futures-only** — and for spot it is
   also the wrong API surface (§6).
4. **Profile thresholds are BTC-price-scaled** (`signal_engine/profiles.py`,
   e.g. `POSITION_MIN_MOVE_PTS = 3000`) — nonsense on any other asset. Frozen
   with the rest of the futures logic; the spot track never touches them.
5. Product copy names BTC directly in `HELP_TEXT` / `WELCOME_TEXT`.

**The mitigating fact that makes this tractable:** the storage layer is already
symbol-aware end to end — `signals`, `forecasts`, `trades_journal` and
`price_alerts` all carry `symbol`, and the forecast ledger states that "a second
asset needs no migration". The blocker is plumbing in the runtime path, not a
schema redesign.

**And the decisive scoping fact:** blockers 1–5 all sit in the *runtime*
product, and the S1–S4 research pipeline is offline. **None of them blocks the
MVP.** They must be cleared before S5 (Telegram integration), not before the
scanner exists.

---

## 18. Implementation roadmap

| stage | goal | modules | complexity | depends on | go / no-go |
|---|---|---|---|---|---|
| **S1** | multi-asset PIT spot data | `tools/spot_cache.py` (new), `analyzer/binance.py` (+spot), `data/spot/` | **M–L** — the delisted-symbol path is the risk | — | ≥150 symbols ingested with manifests, **including ≥20 delisted**; gap ratio within threshold; loader fail-closed on manifest mismatch |
| **S2** | universe + labels | `spot/universe.py`, `labeling/triple_barrier.py` (+ratio mode), `data/universe/` | **M** | S1 | PIT snapshots write-once and reproducible byte-for-byte; `CLEAN_2X` base rate computed and published per regime; parameters frozen **before** inspection |
| **S3** | baselines + first features | `spot/features/`, `spot/baselines.py` | **M** | S2 | all seven baselines run on identical universe/dates; each feature family evaluated **alone**; every trial declared in the registry first |
| **S4** | historical validation | `validation/panel_cv.py` (adapted M04), `spot/evaluate.py`, PIT controls | **L** — the real gate | S3 | **the go/no-go of the whole pivot**: does any rule beat the best baseline on precision@K and Top-K return, with a block-bootstrap interval over ~14 date blocks, after multiple-testing deflation, and do the PIT controls all fire correctly? |
| **S5** | Telegram integration | `bot/*`, blockers 1–3 of §17 | **M** | S4 = pass | a shortlist renders with no fabricated fields; forecasts written to the ledger before display |
| **S6** | daily automation | `scheduler.py`, ledger maturation | **S–M** | S5 | a full day runs unattended; snapshots immutable; maturation appends outcomes on schedule |
| **S7** | advanced data & modelling | PIT mcap/supply, unlocks, TVL, calibration, learned ranker | **L** | S6 + a stated hypothesis per source | each new source earns its place against S4's metrics; probabilities ship only when calibrated |

**S4 is a stop condition, not a checkpoint.** If no rule beats the best simple
baseline, the honest outcome is that the scanner does not ship as a
recommendation engine — it ships as a screener that reports facts (liquidity,
relative strength, drawdown) without claiming selection skill. That outcome is
written here, in advance, so that it cannot later be argued away.

---

## 19. Risks and unknowns

1. **Delisted-symbol history may be awkward to obtain.** Everything about
   survivorship control depends on it. If bulk history for delisted pairs
   proves unavailable, the fallback is to publish the survivorship gap (§7.4)
   as a stated bias bound rather than pretend it is closed. *Highest risk item.*
2. **Effective N ≈ 14.** Every conclusion lives inside that. It is why ML is
   deferred and why lift intervals will be wide.
3. **The 2x base rate is unknown until S2.** If it is ~2%, precision@K is
   nearly unmeasurable at these sample sizes and the target may need to move to
   +50%. If it is ~30%, the interesting question changes to *which* 2x survives
   its drawdown. **S2 may force a redesign of §3, and that is expected.**
4. **Regime dominance.** Most 2x outcomes will cluster in a handful of months.
   A scanner may be measuring "is it a bull market" — which is useful, but is a
   different product than "which coin".
5. **No PIT market cap in MVP** — a real feature gap, accepted deliberately
   over a leaky one.
6. **Binance-only is itself a survivorship channel** (the venue selects
   listings). Bounded by S7 expansion, disclosed until then.
7. **Product expectation risk.** A ranked shortlist without probabilities will
   feel weaker than the confident-looking bot that preceded it. It will also be
   the first version whose claims are checkable.

---

## 20. Final recommendation

Proceed. Build S1 as specified, with the delisted-symbol path treated as the
acceptance criterion rather than a detail, and hold the line on two things that
the previous cycle proves are worth holding: **no composite score before its
components are individually validated**, and **no number displayed that cannot
be estimated**.

**Verdict: `SPOT_ARCHITECTURE_READY`.**

**Next stage: S1 — multi-asset point-in-time spot data.**
