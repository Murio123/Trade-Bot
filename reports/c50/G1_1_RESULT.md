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
not hash to its own `trial_id`, and a truncated final line. The last is
treated as corruption rather than skipped — a registry that quietly drops its
newest trial is precisely the failure this stage exists to prevent.

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

Layering has no cycle: `trial_registry` does not import `deflated_sharpe`,
`deflated_sharpe` imports neither, and both facts are asserted by tests
walking the import AST rather than grepping prose.

---

## 7. Guardrails against future undercounting

1. **The static scan.** A test walks the AST of every module under
   `validation/`, `labeling/`, `models/`, `research/`, `forecast/` and
   `tools/` and fails on any call to `deflated_sharpe(...)` passing
   `n_trials`, outside a three-entry allowlist with stated reasons. A future
   A′ runner that hand-picks a trial count becomes a test failure instead of a
   silent regression.
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

- **Targeted tests: 110.** `tests/test_trial_registry.py` (67),
  `tests/test_research_dsr.py` (20), `tests/test_trial_history.py` (23).
- **Full suite: 1908 passed, 0 failed.**
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
- Sealed holdout (idx 8199+): not read. The runner imports only `argparse`,
  `json`, `os`, `typing` and the two registry modules, asserted by test.
- No strategy rule, threshold, model, gate or runtime behaviour changed.
- Generated artifacts in `reports/g1_1/` are untracked.

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
