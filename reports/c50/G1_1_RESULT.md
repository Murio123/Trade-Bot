# C5.0 G1.1 — trial registry: result

Criteria: `reports/c50/TRIAL_REGISTRY_SPEC.md`, committed at `dd45a9b`
**before** a single historical trial was reconstructed. Anchor: `b2d6f0d`
(G1 closed, 1798 tests green).

## Verdict: `G1_1_PASS`

All eleven conditions of §10 are met. The number that matters:

> **Phase A′, evaluating swing trade expectancy on BTCUSDT, must deflate
> against `n_trials = 100`** — 83 confirmed, plus an uncertain band of 17.

That is not a comfortable number, and it is not supposed to be. It is what
this project's own history establishes when abandoned experiments, screened
features, swept thresholds and post-hoc variants are all counted, which is
what ARCHITECTURE.md §5.4 has required since C5.0 was frozen.

---

## 1. What counts as a trial (frozen §2)

> A trial is one distinct research choice whose result could influence
> selection.

The binding half is "could influence". The counterfactual decides, not the
outcome: a hypothesis that was rejected, that produced zero trades, or whose
result was disliked and set aside still consumed a look. H1 took 0 of 7999
bars and counts in full. LightGBM never ran and counts in full — had it run
and won, it would have been selected.

Ambiguity resolves upward (§2.4). This is stated as a rule rather than left to
judgement because the temptation is one-sided: every undercount makes every
deflated Sharpe more favourable, and no amount of care inside M01 can detect
it.

---

## 2. Trial vs execution (§4)

Two record kinds, and the split is the whole mechanism:

| | Declaration | Execution |
|---|---|---|
| Written | before the result exists | per run |
| Identity | §3.1 selection-relevant fields | + commit, dataset, seed, timestamp |
| Count | moves `n_trials` | never |
| Mutability | immutable; conflict fails closed | append-only |

So re-running a frozen specification is free and choosing something new is
not. `code_commit` is deliberately outside identity (§3.2) — a refactor that
leaves the research question intact must not manufacture a trial — but every
execution records it, so a number is still traceable to the code that made it.

Seeds follow the same logic (§3.3): not part of identity, unless results across
seeds were compared and one was chosen, in which case the seed enters
`parameter_hash` and the sweep's arms become separate trials. C2a's 8 k-means
restarts per k are therefore **not** counted; its 7 k values across two
clusterings **are**.

An execution against an undeclared trial raises. That is ARCHITECTURE.md
§6.1's rule verbatim, and it is what stops a count from being assembled
favourably once the outcomes are known.

---

## 3. The multiple-testing family (§6)

Family key: `(research_objective, profile, symbol, target_family)`.
Unspecified scope components are wildcards, and — separately — a record whose
own field is `None` matches any scope value. Both widen the count, which is
the direction §6.4 requires: a pre-C1 change that applied to every profile is
selection pressure on swing too, and must not vanish because its profile is
null.

Lineage is pulled in unconditionally, **even across families** (§6.2). A
post-hoc variant inherits the selection pressure of what it descends from, and
relabelling its objective would otherwise be the cheapest available
undercount. The C4.3 → C4.3c → C4.3d → C4.3e → C4.4 chain is linked for
exactly this reason.

Two exclusions only: `origin = synthetic` (negative controls, which never
touched real data) and `status = withdrawn_duplicate`. There is no "it was
exploratory" exclusion and no "it didn't work out" exclusion.

---

## 4. Historical counts

Reconstruction: **50 records** standing for **137 confirmed / 154 conservative**
trials, each citing its evidence.

| Family | confirmed | uncertain | **conservative** |
|---|---|---|---|
| **A′ — swing trade_r, BTCUSDT** | 83 | 17 | **100** |
| trade expectancy, all profiles | 128 | 17 | **145** |
| volatility forecast | 9 | 0 | **9** |
| everything | 137 | 17 | **154** |

By stage:

| Stage | records | confirmed | conservative |
|---|---|---|---|
| pre-C1 | 19 | 16 | 33 |
| C1.4a | 4 | 48 | 48 |
| C1.6 | 4 | 4 | 4 |
| C1.8 | 2 | 33 | 33 |
| C2a | 1 | 14 | 14 |
| C2.1 | 3 | 3 | 3 |
| C2.2b | 2 | 2 | 2 |
| C2.2c | 6 | 6 | 6 |
| C3.0 | 1 | 1 | 1 |
| C4.3 | 2 | 4 | 4 |
| C4.3b / c / d / e | 4 | 4 | 4 |
| C4.4 | 1 | 1 | 1 |
| P1 | 1 | 1 | 1 |

### 4.1 Where the count comes from

Three blocks dominate, and all three are routinely left uncounted:

- **C1.4a — 48.** Six thresholds (`tools/deep_backtest.py:123`) × two fold
  modes × four profiles. The memo's finding that fold mode changes no verdict
  is a *result*; results do not retroactively make the arms that produced them
  free.
- **C1.8 — 32.** Every one of the 32 features in
  `reports/c18/single_feature_report.json` was screened and labelled
  keep/remove/unknown. Each label is a selection decision.
- **C2a — 14.** Two independent clusterings × seven k values, selected on
  silhouette.

The uncertain band of 17 is entirely pre-C1: commits that changed the scoring
or confluence specification where the *choice* is established by the commit
but the *number of arms compared* is not. Those records carry a band —
`(1, 3)`, `(1, 4)`, `(1, 6)` — rather than a number.

### 4.2 What was NOT counted, and why

- **Reruns, reproduction guards, determinism checks.** §2.2.
- **The 8 k-means restarts per k.** Seeds, not choices (§3.3).
- **Apparatus construction** — the walk-forward geometry, the C4.2 platform,
  M00–M04, the negative controls. Building a measuring instrument is not
  selecting a strategy. The controls are `origin = synthetic` and excluded by
  §6.3.
- **P1's cost correction itself.** A definition fix, not a trial (§2.3). The
  *re-reading of the G0 verdict* against both the old and new number is the
  decision §2.3's second clause makes a trial, and that one is counted.
- **Telegram/UX and product work.** No evaluation against data.

### 4.3 Stated gaps

Recorded here rather than papered over, per §8.2:

1. **Pre-C1 evaluation depth is unknowable.** The commit messages establish
   that a confluence component was added; only some establish that a
   measurement decided it stayed. Those records rest on `inference` evidence
   and contribute 0 to confirmed — which is why the pre-C1 band is wide
   (16 → 33).
2. **`/backtest` was interactive.** Threshold sweeps run from Telegram left no
   artifact. `da6f12b` proves at least one selection over thresholds happened;
   its arm count is a band `(1, 6)`.
3. **C2a's clustering script was never committed** (the report says so
   explicitly). The k range and restart count come from the report's own prose.
4. **No trial was invented to cover any of the above.** A suspected but
   unevidenced choice is a gap, not a record.

---

## 5. Implementation

| Component | Path | Lines |
|---|---|---|
| Registry | `validation/trial_registry.py` | append-only store, identity, family counting |
| Research DSR | `validation/research_dsr.py` | the only project path to a published DSR |
| History | `validation/trial_history.py` | 50 evidence-cited records |
| Runner | `tools/g1_1_trial_ledger.py` | materialize + report, offline |

Storage reuses `volatility/ledger.py`'s proven conventions rather than
inventing a second philosophy: JSON Lines, exclusive `fcntl` lock held across
the whole check-then-append, `fsync` per write, and one `_validated()` entry
point every public reader goes through. That last rule is not stylistic — four
audit rounds on the C4.4 ledger found the same defect shape each time, a check
added to some read paths and not others.

Corruption detected: unparseable line, unknown `kind`, duplicate declaration,
duplicate `run_id`, execution without declaration, identity payload that does
not hash to its own `trial_id`, a truncated final line, and — after amendment
A1 — any removed, reordered or in-place-edited record, via a `seq` +
`prev_sha256` chain on every line.

One bound is stated rather than papered over, and the second audit round was
right to force it wider than I first wrote it: the chain is **self-contained**.
Anyone who deletes a line and recomputes `seq`/`prev_sha256` produces a file
that validates, and truncating complete lines from the end needs no
recomputation at all. No file-local format can prevent either. What the chain
buys is that accidental damage and casual edits fail closed instead of reading
as a smaller registry — worth having, and not the same as tamper-proof.

Three external defences cover the deliberate case, and all three exist:
provenance pins the registry's content hash into every published DSR; the
reconstructed history is code, so `trial_history.missing_from()` reports any
historical record absent from a registry however carefully the file was
rewritten around it (a test forges exactly that file and catches the
deletion); and the runner is deterministic, so re-materializing reproduces the
same bytes.

There is no delete, no amend, no compact. The absence is the mechanism, and a
test asserts it.

### 5.1 Reconstruction is a table, not a scraper

Deliberate (§8.4). Most of this project's research artifacts live in untracked
`reports/<stage>/` directories, so a scraper would produce different counts on
different machines depending on which of them happen to be present — the
opposite of auditable. A table is code: reviewed as code, byte-identical on any
checkout, and every row states what established it.

---

## 6. M01 provenance

`validation/deflated_sharpe.py` is unchanged. It remains a mathematical
primitive taking `n_trials` as an argument, which is the correct shape for a
formula, and remains directly unit-testable.

Above it, `research_deflated_sharpe(returns, registry=, scope=, trial_id=)`
**has no `n_trials` parameter**. There is no argument to pass. It:

- refuses a trial that is not declared;
- refuses a trial that is not a member of the family it is being deflated
  against — the cheapest available undercount is to pick a narrow family, get
  a small number and quote it;
- refuses a `synthetic` trial, directing it to the maths layer;
- uses the **conservative** count, never the confirmed one;
- records the scope, the confirmed/uncertain/conservative decomposition, the
  registry path, its record count and its **content hash**, so a published
  number stays checkable against the registry it was computed from rather than
  against whatever the registry later becomes.

The count and the hash that pins it come from a **single read** of the
registry. Reading twice — once to count, once to hash — is a race the audit
caught: a declaration landing between the two reads produces provenance
advertising a registry state whose family count is larger than the number
actually used.

Layering has no cycle: `trial_registry` does not import `deflated_sharpe`,
`deflated_sharpe` imports neither, and both facts are asserted by tests
walking the import AST rather than grepping prose.

---

## 7. Guardrails against future undercounting

1. **The static scan.** A test walks the AST of every module under
   `validation/`, `labeling/`, `models/`, `research/`, `forecast/` and
   `tools/` and fails on any module outside a three-entry allowlist that names
   `n_trials` — as a keyword argument or as a literal key. A future A′ runner
   that hand-picks a trial count becomes a test failure instead of a silent
   regression. The rule keys on `n_trials`, not on the callee's name
   (amendment A2), so an alias import, a module alias or a `getattr` lookup
   does not change the answer; four such evasions are tested explicitly.
2. **The allowlist is checked in both directions.** Entries must exist, and
   must still need the exemption — an allowlist naming a file that no longer
   makes the call is silently wider than it reads.
3. **The guardrail is proven able to fail.** The scan is pointed at a module
   that does the forbidden thing and must report it. G1 shipped an anti-grep
   test that turned out to be vacuous; that is not repeated here.
4. **Declaration before interpretation.** An execution for an undeclared trial
   raises.
5. **Frozen specs cannot be edited.** A second declaration with different
   content raises; a byte-identical one is a no-op.
6. **Post-hoc variants carry lineage**, and lineage is inescapable across
   families.

A′ itself is **not** wired to the registry. The interface and the guardrail
exist; integrating a runner would be starting A′, which §11 puts out of scope.

---

## 8. Historical G1 handling

Unchanged and not revisited:

1. G1's verdict remains **`G1_INDETERMINATE`**.
2. M01–M04 implementation and the negative-control findings remain historical
   facts at the numbers G1 published.
3. The corrected T3 semantics of `G1_SPEC.md` §7 A3 apply **prospectively
   only**, to gates declared after 2026-08-18.
4. The registry resolves the `n_trials` blocker **prospectively**. It does not
   retroactively validate any number published against a hand-counted trial
   count.
5. No G1 control was re-run, re-scored, or re-interpreted with a
   registry-derived count.

There is no retroactive PASS, and the mechanism that would produce one was not
built.

---

## 9. Verification

- **Targeted tests: 125.** `tests/test_trial_registry.py` (73),
  `tests/test_research_dsr.py` (28), `tests/test_trial_history.py` (24).
- **Full suite: 1919 passed, 0 failed** — run twice, 246s and 255s.
- Two pre-existing guards fired on the new code and were addressed without
  weakening either:
  - the single-Sharpe-implementation guard matched
    `research_deflated_sharpe`. The exemption is granted **with a check** that
    the module imports M01 and contains no moment arithmetic of its own, so it
    cannot become a hiding place for a second formula.
  - the removed-UX guard matched a commit message quoted verbatim in
    `trial_history.py`. The evidence note was reworded; the guard was not
    touched.
- `git diff --check` clean.
- **Independent audit: three real defects, all fixed** (§9.1).
- Sealed holdout (idx 8199+): not read. The runner imports only `argparse`,
  `json`, `os`, `typing` and the two registry modules, asserted by test.
- No strategy rule, threshold, model, gate or runtime behaviour changed.
- Generated artifacts in `reports/g1_1/` are untracked.

### 9.1 What the audit found

The first submission returned **FAIL**, and it was right to. Three defects,
all of them in the direction that matters:

1. **The provenance hash did not pin the count it travelled with.** The count
   and the snapshot came from two separate reads with no lock between them.
   Fixed: `snapshot_and_count()` derives both from one read of the bytes.
2. **A registry that lost a whole line read as a valid, smaller registry.**
   §5 claimed corruption was detectable; for the case that matters most —
   silent undercount — it was not. Fixed by the A1 record chain.
3. **The static guardrail was evadable by aliasing.** It matched calls named
   `deflated_sharpe`; `import ... as dsr`, `import ... as m`, and
   `getattr(m, "deflated_sharpe")` all walked past it. A guardrail that only
   stops the obvious spelling of a bypass is not a guardrail. Fixed by A2, and
   each named evasion now has a test.

A second round returned **FAIL** again, on two counts, and both stand:

4. **Amendment A1 overclaimed.** It said removal and reordering "all fail
   closed"; they do not, if the chain is recomputed. Corrected to what is
   actually true, with the three external defences named — including
   `missing_from()`, which is new and which catches the exact forged file the
   audit described.
5. **The strengthened guardrail had a false positive.** Flagging every string
   constant `"n_trials"` would have failed a future reporter reading
   `payload["n_trials"]` back out of a result — pushing it toward hiding the
   number, the opposite of the intent. The scan now looks only at call
   keywords and `**{...}` literals; four legitimate read patterns are tested
   as *not* flagged, alongside the four evasions that are.

Across both rounds the audit found no defect in family counting, in the
ancestry closure, or in the reconstruction arithmetic, and confirmed no
sealed-holdout access, no frozen-artifact modification, and no change to the
G1 verdict.

The pattern in my own errors is worth naming: every one was a claim that a
protection was **broader than it was** — a hash that pinned less than it
implied, a chain that detected less than A1 said, a guard whose name-matching
looked complete. That is the same shape as the failure this whole stage exists
to close, one level up.

Two tests I had written were also weaker than they read: the tampering tests
mutated the file in ways the new chain check catches *first*, which would have
left the unknown-kind, duplicate-declaration and orphaned-execution guards
looking covered while never being reached. They now re-chain the file so each
guard is genuinely exercised.

---

## 10. Next allowed step under C5.0

**Phase A′ is now eligible, and eligibility is not permission.** Two things
stand between here and an A′ run, and only one of them is mine:

1. **The T3 decision** (`G1_SPEC.md` §7 A3) is still open and still the
   owner's. Three options are recorded there; the recommendation remains
   option 3 — re-freeze T3 for a future gate, take the apparatus properties as
   demonstrated.
2. **When A′ does start, it declares its trials first.** Every hypothesis it
   evaluates is a new declaration on top of the 100, before its result is
   looked at. A′ starting at `n_trials = 100` and ending at 100 would mean it
   evaluated nothing.

The infrastructure blocker named in ARCHITECTURE.md §6.1 — "the single largest
failure mode of M01" — is closed.
