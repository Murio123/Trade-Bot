# S2 — universe and labels: frozen criteria

Written and committed **before a single label was computed**. Everything here
is a choice that could be tuned to a preferred answer, which is why it is fixed
first and why changes require an amendment section rather than an edit.

Anchor: S1 (`reports/s1/S1_RESULT.md`, `S1_PASS`) — 734 USDT pairs, 826,601
daily bars, 2017-08-17 → 2026-08-24, 250 of them delisted.

---

## 1. The decision grid

- Decision dates: the **first calendar day of each month**, UTC.
- At decision date `t`, only bars with `close_time < t` may be read. The
  reference price `P0` is the **close of the last such bar**.
- First decision date: **2018-01-01**. Last: **panel end − 180 days**, so every
  labelled event has a full horizon available in calendar time.

Monthly rather than daily is deliberate. With a 180-day label, daily decision
dates would produce 180× overlapping rows and a sample that looks 180 times
larger than it is (S0 §11.1).

## 2. Eligibility (S0 §5.1, restated as the binding version)

At `t`, a symbol is eligible iff **all** hold, computed only from bars with
`close_time < t`:

| # | rule | threshold |
|---|---|---|
| E1 | quoted in USDT on Binance spot | — |
| E2 | not a stablecoin | curated base-asset list, §2.1 |
| E3 | not a wrapped / staked / duplicate representation | curated list, §2.1 |
| E4 | not a leveraged token | base asset matches `*UP`, `*DOWN`, `*BULL`, `*BEAR` |
| E5 | listing age | ≥ **180** days since first bar |
| E6 | price history | ≥ **200** daily bars |
| E7 | liquidity | median daily `quote_volume` over the last **30** bars ≥ **$5,000,000** |
| E8 | activity | a bar exists on **each** of the last 30 calendar days |
| E9 | size cap | top **250** eligible by 30-day median dollar volume |

### 2.1 Curated exclusion lists

Stablecoins and wrapped assets are excluded by **base asset**, listed
explicitly in `spot/universe.py`. This is a judgement call and it is recorded
as one. Two properties keep it honest:

- membership is a **property of the asset's design**, not of its outcome — a
  stablecoin was a stablecoin before it depegged, so UST/USTC is excluded on
  every date including the ones where it moved violently;
- the lists may not be extended after labels are inspected. Any later addition
  is an amendment with a stated reason.

### 2.2 What eligibility deliberately does not use

Market cap, FDV, circulating supply, unlock schedules, TVL, sector labels. All
are either unavailable point-in-time or restated by their sources (S0 §6).
Dollar volume is the size proxy.

## 3. Point-in-time snapshots

One snapshot per decision date, written **once** to
`data/spot/universe/YYYY-MM-DD.json`, containing the eligible symbol list, the
value of every screen per symbol, and a content hash. A rerun that produces
different bytes for an existing date is a defect, not an update.

Delisted symbols appear in every snapshot they qualified for. Nothing removes
them retroactively.

## 4. The label: `CLEAN_2X(180d)`

Triple barrier via M02 (`labeling/triple_barrier.py`), **unmodified**:

| barrier | value | M02 configuration |
|---|---|---|
| upper | `2.00 × P0` | `upper_mult = 1.0` |
| lower | `0.60 × P0` | `lower_mult = 0.4` |
| vertical | 180 bars | `vertical_bars = 180` |

with `sigma = 1.0` for every bar, which turns M02's volatility-scaled barriers
into fixed ratios exactly as S0 §6 predicted. The frozen configuration hashes
to a stable `content_sha256`, recorded in every artifact.

`CLEAN_2X = 1` iff the upper barrier is touched **first**. The same-bar tie —
one daily bar whose range spans both barriers — resolves to the **adverse**
barrier under M02's audited convention (G1 amendment A1), so a tie is a 0. Tie
frequency is reported, because a bias whose frequency is unmeasured is
indistinguishable from a bug.

### 4.1 Death is an outcome, not missing data

A delisted symbol's price path ends. M02 reports `TRUNCATED` for an event whose
horizon extends past the available bars, and the reinterpretation is **ours,
not M02's**:

| situation | treatment |
|---|---|
| symbol still trading, horizon extends past panel end | **CENSORED** — excluded from every rate, counted separately |
| symbol delisted, path ends inside the horizon, no barrier touched | **DEAD** — a resolved outcome, `CLEAN_2X = 0` |
| a barrier was touched before the path ended | resolved normally |

Treating death as missing data would drop precisely the failures and rebuild
the survivorship bias S1 exists to prevent. The decision grid already excludes
genuine right-censoring (§1), so `CENSORED` should be empty by construction; it
is still counted, and a non-zero count is a bug report.

### 4.2 Events spanning a gap in the bar grid

M02 refuses an event whose window crosses a hole in the bar grid, because the
first touch may have happened in the missing bars. Those events are recorded as
`GAP_UNRESOLVED` and excluded from rates. Their count is published: 0.18% of
the panel has gaps (S1), so this must stay small, and if it does not, the
labels are not trustworthy.

## 5. The record vector (S0 §3.3)

Per resolved `(symbol, decision_date)`, over the same window:

| field | definition |
|---|---|
| `mfe` | max high / `P0` − 1 |
| `mae` | min low / `P0` − 1 |
| `days_to_2x` | days from `t` to first touch of the upper barrier; null if never |
| `terminal_return` | last close in the window / `P0` − 1 |
| `excess_vs_btc` | `log(1 + terminal_return) − log(1 + BTC terminal return)` over the identical window |

## 6. BTC regime, point-in-time

At `t`, from BTCUSDT daily closes only, using bars before `t`:

- **bull** — close > 200-day SMA **and** the SMA is higher than 20 bars ago;
- **bear** — close < 200-day SMA **and** the SMA is lower than 20 bars ago;
- **range** — otherwise.

Used only to split results (S0 §11.5). It is not a feature and not a filter.

## 7. What S2 publishes

1. The base rate of `CLEAN_2X`, overall and per regime.
2. The distribution of the record vector.
3. Counts: eligible symbols per date, resolved / dead / censored / gap events,
   tie frequency.
4. The number of **independent** 180-day windows — the honest N.

## 8. What S2 does not do

No features, no ranking, no baselines, no model, no evaluation of anything.
Those are S3 and S4. S2 answers one question: *what actually happened, to whom,
and how often.*

## 9. The pre-registered consequence (S0 §21 item 3)

If the base rate is so low that precision@K cannot be measured at this sample
size, **the target moves to +50% and S0 §3 is rewritten**. That is recorded
here, before the number is known, so that moving the target later cannot be
mistaken for — or become — fitting the target to the data.

A base rate that is *high* has a consequence too: if most eligible assets reach
2x in a bull regime, the interesting question is not "which coins 2x" but
"which 2x survive their drawdown", and the primary metric shifts to the
`mae`-conditioned rate. Both branches are written down in advance; neither is
chosen after seeing the number.
