# S1 — multi-asset point-in-time spot data: result

Criteria: `reports/s0/SPOT_DISCOVERY_ARCHITECTURE.md` §18, stage S1, written
before any spot data was fetched.

## Verdict: `S1_PASS`

All four go/no-go conditions are met, and the one that mattered is met with
room to spare.

| condition (S0 §18) | required | actual |
|---|---|---|
| symbols ingested with manifests | ≥ 150 | **734** |
| **delisted symbols among them** | **≥ 20** | **250 (34.1%)** |
| gap ratio within threshold | honest reporting | **0.17%** missing bars; 41 symbols affected |
| loader fail-closed on manifest mismatch | must refuse | refuses, with tests for six distinct corruptions |

The panel: **734 USDT pairs, 826,301 daily bars, 2017-08-17 → 2026-08-24**,
73 MB under `data/spot/` (gitignored, regenerable).

**Figures updated 2026-08-24.** S2 found 300 malformed bars — almost all the
partial final bar of a delisted pair — and the panel was re-ingested with
per-bar consistency validation (`S2_RESULT.md` §5.1). Every count in this
document is the post-re-ingestion one. The gap-symbol count *rose* from 22 to
41 as a result, which reads like a regression and is the opposite: dropping a
corrupt mid-history bar turns a hidden defect into a visible hole.

---

## 1. The finding that decides the whole spot track

**The live spot API serves history for delisted pairs.** `/api/v3/klines`
answers for BCCUSDT (dead since 2018-11-20) and MITHUSDT (dead since
2022-12-22) with their complete price path up to the day they stopped trading.

S0 rated the delisted-symbol path as the project's single biggest data risk and
budgeted half of S1 for it. It cost one API call to disprove. There is no bulk
download, no dump parsing, and no reconstruction: dead pairs come down the same
pipe as live ones.

What remained was the harder half of the problem — not *fetching* dead pairs
but *knowing which ones exist*, since `exchangeInfo` is the present tense.

## 2. The bug that would have quietly reintroduced survivorship bias

The first version of `tools/spot_symbols.py` derived "delisted" the obvious
way: present in the published data-dump index, absent from `exchangeInfo`. It
reported **1 dead pair out of 734** — a number that looks like a clean answer
and is completely wrong.

**A pair that stops trading is not removed from `exchangeInfo`.** It stays,
with `status = "BREAK"`. BCCUSDT, delisted in 2018, is still listed today. The
correct test is `status != "TRADING"`, and it finds **250**.

Had that gone unnoticed, every downstream universe would have been built from
survivors while carrying a field named `trading_now` that asserted otherwise.
The test `test_a_break_pair_counts_as_dead_not_as_listed` pins the venue's
actual behaviour so the derivation cannot quietly revert.

**The scale of what this protects.** Deaths are spread across the whole
history, not concentrated in one crash:

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| 2 | 3 | 19 | 20 | 38 | 24 | 46 | 51 | 47 |

A universe built from today's symbol list would silently omit a third of the
assets that were investable at any past decision date, and the omitted third is
disproportionately the losers. That is the bias S0 §7 named, and it is now
measurable rather than hypothetical.

---

## 3. What was built

| module | role |
|---|---|
| `tools/spot_client.py` | offline spot client, venue pinned, no failover; paginated symbol index; forward-walking history |
| `tools/spot_symbols.py` | the survivorship-free symbol spine → `data/spot/symbols.json` |
| `tools/spot_cache.py` | daily OHLCV per symbol → CSV + content-hashed manifest |
| `tools/spot_dataset.py` | fail-closed panel loader |
| `tests/test_spot_ingestion.py` | 19 tests |

### 3.1 One deliberate deviation from S0

S0 §6 said to add spot endpoints to `analyzer/binance.py`. They went into a new
`tools/spot_client.py` instead, because `analyzer/binance.py` is live runtime
code on the futures API reached through a failover wrapper, and both properties
are wrong here: spot is a different API, and a panel assembled from two venues
after a silent failover is not one panel — the C1.2 lesson (drift point D15)
verbatim. The side effect is that **S1 changed no runtime file at all**: no bot,
scheduler, pipeline, database, futures logic or Telegram code was touched.

### 3.2 Conventions inherited rather than reinvented

Manifest-first loading, source pinning, uncapped pagination and honest gap
reporting all come from `tools/kline_cache.py` / `kline_dataset.py`;
`gap_report` is imported, not copied. Two things are new, and both because this
is a panel rather than one series:

- **Every dataset is content-hashed.** With one symbol, "the manifest was
  written next to the file a second ago" is nearly good enough. Across 734
  files written over minutes, with reruns and partial refreshes, it is not.
- **Dead symbols may not go missing.** `load_panel()` refuses a panel whose
  delisted share has collapsed below a floor (default 15%; observed 34.1%).

### 3.3 The survivorship floor, and why it is in the loader

A panel filtered down to survivors loads cleanly, produces plausible numbers,
and is wrong in the direction that flatters every result. Nothing downstream
can detect it. So it fails at load time, and a deliberate subset must say so
explicitly (`require_dead_share=0.0`). One test builds exactly the survivors-only
panel and asserts it cannot be read.

### 3.4 Columns

`taker_buy_base`, `taker_buy_quote` and Binance's `ignore` are dropped: CVD is
a microstructure concept with no role in a 180-day question, and keeping them
would roughly double the panel on disk for data nothing downstream reads.
`quote_volume` is kept — dollar volume is the liquidity screen S2 depends on.
The dropped columns are recorded in every manifest rather than silently absent.

---

## 4. What the panel looks like

- **734** USDT pairs ever; **484** trading now, **250** not.
- **826,301** daily bars; median history per symbol **985.5 bars**.
- **630 symbols have ≥ 200 bars** — the warmup S0 §5.1 requires. That is the
  realistic ceiling on universe size before eligibility filters, and it is
  comfortably above the 150–250 per-date estimate.
- Gaps: 41 symbols, **1,395 missing bars of 827,696 expected (0.17%)**. The
  largest is FTTUSDT's 311-bar hole — the FTX collapse and the halt that
  followed. It is reported, not filled.

---

## 5. Verification

- **19 targeted tests**, `tests/test_spot_ingestion.py`. Six distinct
  corruptions are tested as refusals: edited data file, truncated file,
  manifest naming a missing file, row count disagreeing after a partial
  refresh, out-of-order bars, and a stray CSV with no manifest.
- **Full suite: 1959 passed, 0 failed.**
- Every one of the 734 symbols loads through the fail-closed loader (1.6 s).
- Network calls are confined to `tools/spot_client.py`; the tests reach the
  network nowhere and drive pagination through a fake pages client.
- `data/` is gitignored; the panel is regenerable and idempotent — a rerun
  skips cached symbols and produces the same bytes.
- No runtime file, no futures logic, no sealed holdout, no database and no
  Telegram code was touched. Phase A′ remains paused.

### 5.1 One test was weaker than it read

`test_full_history_refuses_to_return_a_truncated_history` first fed the fake
client the *same* page repeatedly. The overlap filter dropped every duplicate
bar, the walk ended naturally, and the test passed for the wrong reason —
it never reached the page budget it claimed to test. The pages are now strictly
increasing, so the walk genuinely cannot terminate. This is the vacuous-guard
shape G1 shipped once and G1.1 was audited for twice; it is cheaper to catch
here.

---

## 6. Next stage: S2 — universe + labels

S1 delivers data and nothing else: no eligibility rule, no filter, no judgement
about what is investable. That is S2's job, and keeping it out of the data layer
is deliberate — an eligibility decision baked into ingestion could never be
revisited without re-fetching.

S2 does three things, in this order:

1. **Point-in-time universe snapshots** (S0 §5): apply the eligibility rules at
   each monthly decision date using only bars with `close_time ≤ t`; write each
   snapshot once, content-hashed, never recomputed.
2. **`CLEAN_2X(180d)` labels** via M02's triple barrier — upper 2.00, lower
   0.60, vertical 180 days — plus the five-field record vector. **The barrier
   parameters are frozen before any label is inspected.**
3. **Publish the base rate**, overall and per BTC regime.

The base rate is the number that governs everything after it. S0 §21 states the
consequence in advance and it stands: **if clean-2x turns out to be rare enough
that precision@K cannot be measured at this sample size, the target moves to
+50% and §3 of the architecture is rewritten.** That is an expected outcome of
S2, not a failure of it.

Not started. Nothing is committed.
