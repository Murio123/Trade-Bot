# C5.0 G1.1 — trial registry: frozen specification

Declared 2026-08-18, **before a single historical trial was reconstructed and
before any trial count was computed**. Anchor commit: `b2d6f0d` (G1 closed,
full suite 1798 green). This document exists so that the rule deciding what
counts as a research trial cannot be chosen after seeing how large the count
turns out to be.

The direction of the temptation is known in advance and is one-sided: a smaller
`n_trials` makes every deflated Sharpe more favourable. Every rule below is
therefore written to round **up** when it is unsure, and the tie-breaks are
stated here rather than left to the implementation.

G1.1 asks one question: **can a research result state where its `n_trials`
came from, and is that number defensible?** It does not evaluate any strategy,
does not re-run any hypothesis, and does not re-open G1.

---

## 0. Relationship to G1

`G1_INDETERMINATE` (2026-08-18, `reports/c50/G1_RESULT.md`) is a historical
fact and is **not** revisited here. Nothing in this stage may be used to turn
it into a pass. See §9.

---

## 1. Scope

| Component | Path | Purpose |
|---|---|---|
| Registry | `validation/trial_registry.py` | canonical append-only trial store |
| Research DSR | `validation/research_dsr.py` | the only project path to a published DSR |
| History | `validation/trial_history.py` | curated, evidence-cited reconstruction |
| Runner | `tools/g1_1_trial_ledger.py` | materialize + report, offline only |

`validation/deflated_sharpe.py` (M01) is **not modified as a mathematical
primitive**. It keeps `deflated_sharpe(returns, n_trials=...)`. The registry
sits above it, never below: `trial_registry` must not import
`deflated_sharpe`, and `deflated_sharpe` must not import `trial_registry`.
`research_dsr` imports both and is imported by neither.

Storage conventions are taken from `volatility/ledger.py` (C4.4), which is
already proven in this project — exclusive `fcntl` lock across the whole
check-then-append, `fsync` on every write, duplicate detection on read, one
validation entry point that every public function goes through. A second
storage philosophy is not invented here.

---

## 2. What a trial is

> **A trial is one distinct research choice whose result could influence
> selection.**

Three words carry the whole definition and each is binding.

**Distinct** — it differs from every previously declared trial in at least one
selection-relevant field (§3.1). Two evaluations that differ in nothing but
when they were run are one trial.

**Research choice** — a human decision about *what to evaluate*: which
hypothesis, which features, which target, which horizon, which thresholds,
which model family, which acceptance criterion. Not a decision about *how to
execute* an evaluation already chosen.

**Could influence selection** — the result was visible to a decision about
what to keep, drop, tune, publish or carry forward. The counterfactual is what
matters, not the outcome: a trial that was abandoned, that returned
`INSUFFICIENT_DATA`, or whose result was disliked and set aside **still
counts**. ARCHITECTURE.md §5.4 states this directly ("abandoned experiments
count"), and it is the entire reason the count is not reconstructible
favourably after the fact.

### 2.1 Counts as a new trial

Derived from this project's actual workflow, not from a generic list:

| Case | Evidence in this project |
|---|---|
| A new hypothesis with its own frozen spec | H1, H2 (C2.2a) |
| A different entry/exit rule set under one hypothesis | C2.1 gates A/B/C |
| A different feature set offered to a model | C4.1 frozen feature spec |
| A different target/label definition | C4.1 12-bar range ratio |
| A different horizon | any change to the 12-bar label window |
| A different model family | Ridge vs LightGBM (C4.3/C4.3b) |
| A hyperparameter configuration **evaluated for selection** | ridge alpha grid |
| A different threshold or gate value screened for keeping | `/backtest` grid |
| A different profile evaluated for the same objective | swing/position/intraday/bounce (C1.4a) |
| A different acceptance criterion, when the result could move | C4.3 go/no-go rule |
| A post-hoc variant created **after** seeing a previous result | C4.3d after C4.3c |
| A different dataset contract (exchange, symbol, bar span) | Binance vs Bybit |

The last row of §2.1 deserves emphasis because it is the case that inflates
counts fastest and is the easiest to rationalise away: **a variant proposed
after seeing a result is a new trial, and it is the most expensive kind**, since
the prior result informed it. C4.3d exists precisely because C4.3c's numbers
were seen first. It counts.

### 2.2 Does NOT count as a new trial

| Case | Why |
|---|---|
| Re-running a frozen spec unchanged | no new choice was made |
| Re-running on a newer code commit with identical research spec | §3.2 |
| A reproduction guard / determinism check | its purpose is to confirm sameness |
| A different random seed, when the seed is not being selected over | §3.3 |
| Reformatting, re-reporting or re-plotting an existing result | no evaluation |
| A bug fix that changes a number without changing the question | §2.3 |
| Unit tests and synthetic negative controls | no real-data selection |

### 2.3 The bug-fix case, stated explicitly

A defect fix that changes a previously published number is **not** a new trial
by default: the research question is unchanged. But if the fix is followed by a
decision that used both the old and the new number — for example keeping a
specification that the corrected number now favours — the *decision* is a
post-hoc variant under §2.1 and a trial is declared for it. The registry
records the distinction in `origin`, so the judgement is visible rather than
implicit.

### 2.4 Fail-closed default

Where §2.1 and §2.2 both plausibly apply, **§2.1 wins and a trial is
declared.** The registry never resolves an ambiguity in the direction that
lowers the count.

---

## 3. Trial identity

### 3.1 Selection-relevant fields

`trial_id` is a deterministic hash over exactly these fields, canonicalised as
described in §3.4:

| Field | Why it is selection-relevant |
|---|---|
| `research_objective` | what is being selected for |
| `research_stage` | the stage that owned the choice |
| `hypothesis` | the named specification under test |
| `profile` | a per-profile result drives a per-profile decision |
| `symbol` | a per-asset result drives a per-asset decision |
| `dataset_contract` | exchange, timeframe and bar span the claim is about |
| `target_hash` | the label being predicted |
| `horizon_bars` | the window the label spans |
| `feature_set_hash` | the inputs the model was allowed |
| `parameter_hash` | thresholds, gates, alphas, barrier multipliers |
| `spec_hash` | the frozen specification document/config |
| `criterion_hash` | the acceptance rule, when it could move the decision |

Any of these may be `None` where the concept does not apply (a threshold screen
has no `feature_set_hash`). `None` is hashed distinctly from an empty string, so
"no feature set" and "the empty feature set" are different trials.

### 3.2 Deliberately NOT part of identity

| Field | Why excluded |
|---|---|
| `code_commit` | a refactor that leaves the research question intact must not manufacture a trial |
| `created_at` | wall-clock time is not a research choice |
| `run_seed` | see §3.3 |
| `runner`/`host`/`cli_args` | execution detail |
| `status`, `outcome`, `notes` | recorded *after*; identity must be fixable before results exist |

`code_commit` being excluded from identity is not the same as it being
unrecorded: every execution records it (§4), so a number can still be traced
to the code that produced it.

### 3.3 The seed rule

A seed is **not** part of trial identity by default. Reproducibility runs and
bootstrap resamples under one frozen spec are one trial.

A seed **becomes** selection-relevant, and enters `parameter_hash`, when the
declaration sets `seed_is_selective = True` — meaning results across seeds were
compared and one was chosen or reported preferentially. Declaring a seed sweep
after the fact is a post-hoc variant under §2.1.

### 3.4 Canonicalisation

`trial_id = "t_" + sha256(canonical_json(identity_fields))[:16]`.

Canonical JSON is `json.dumps(obj, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)` over a dict containing exactly the §3.1 keys, with `None`
preserved. Component hashes (`target_hash`, `feature_set_hash`,
`parameter_hash`, `spec_hash`, `criterion_hash`) are themselves sha256 over the
canonical JSON of the underlying object, so a caller may pass either a
pre-computed hash or the object.

Sequences are order-sensitive **except** feature sets, which are sorted before
hashing: offering the same features in a different order is not a different
research choice.

The 16-hex-character truncation gives a collision probability below `1e-9` for
any trial population this project could ever reach, and keeps ids legible in
reports. A collision would be detected, not silently absorbed: §4.2's conflict
rule fires when the same `trial_id` arrives with different identity fields.

---

## 4. Trial identity vs execution identity

Two records, two lifetimes:

**Declaration** (`kind = "declaration"`) — one per `trial_id`, written
**before** the result exists. Carries the §3.1 identity fields, the full
identity payload, `created_at`, `origin`, `reconstructed`, `parent_trial_id`,
`evidence`, `notes`. Immutable.

**Execution** (`kind = "execution"`) — one per run of a declared trial. Carries
`trial_id`, `run_id`, `code_commit`, `dataset_version`, `run_seed`,
`started_at`, `status`, `outcome_summary`. Many per trial; never affects
`n_trials`.

`run_id = "r_" + sha256(canonical_json({trial_id, code_commit,
dataset_version, run_seed, started_at}))[:16]`.

This split is the mechanism behind §2.2's first four rows. A rerun is an
execution; only a new declaration moves the count.

### 4.1 Registration before interpretation

An execution whose `trial_id` has no declaration is **inadmissible** and raises.
This is ARCHITECTURE.md §6.1's rule verbatim ("a result whose hypothesis has no
prior declaration is inadmissible") and is what stops a count from being
assembled favourably once the outcomes are known.

### 4.2 Conflict rule

A second declaration for an existing `trial_id` is:

- **accepted silently as a no-op** when its identity payload is byte-identical
  (idempotent replay of the same declaration);
- **rejected** (`ImmutableRecordError`) when any field differs.

The second case is a genuine hash collision or a caller mutating a frozen
specification in place. Both are errors, and both fail closed.

---

## 5. Registry storage

JSON Lines, one record per line, sorted keys, at a caller-supplied path.

| Requirement | Mechanism |
|---|---|
| Append-only | no code path opens the file in `"w"`; only `"a"` |
| No silent overwrite | §4.2 conflict rule, under an exclusive lock |
| Idempotent replay | byte-identical declaration is a no-op |
| Corruption detectable | unparseable line, unknown `kind`, duplicate `run_id`, execution without declaration, identity payload that does not hash to its own `trial_id` |
| Deterministic serialization | `json.dumps(..., sort_keys=True)`; re-serializing a read record reproduces the line |
| Atomic-enough writes | single `write` of one line + `flush` + `fsync`, as `volatility/ledger.py` |
| Concurrency | exclusive `fcntl` lock on a `.lock` sidecar held across check-then-append; refuses to append at all where `fcntl` is unavailable |
| No deletion, no compaction | not implemented, deliberately |

A truncated final line (interrupted write) is a corrupt registry and raises on
read. It is not silently skipped: a registry that quietly drops its last trial
is exactly the failure this stage exists to prevent.

---

## 6. Multiple-testing family semantics

The question a DSR needs answered is not "how many experiments has this
repository ever run" but **"how many attempts was this result selected from"**.

### 6.1 Family key

A trial belongs to a family, keyed by:

`(research_objective, profile, symbol, target_family)`

`target_family` is a coarse, declared label (`trade_r`, `volatility_ratio`,
...), not `target_hash`: two different target *definitions* for the same
underlying quantity are competing attempts at one question and must not each
get their own private null.

### 6.2 Membership rule

`n_trials(scope)` counts distinct `trial_id`s that:

1. match every **specified** component of the scope, treating an unspecified
   component as a wildcard; **and**
2. are not excluded by §6.3.

Plus, unconditionally, the transitive `parent_trial_id` ancestry of any counted
trial, even where an ancestor sits in a different family. A post-hoc variant
inherits the selection pressure of what it descends from; letting a lineage
escape by relabelling its objective would be the cheapest possible way to
undercount.

### 6.3 The only exclusions

A trial is excluded from every family count when, and only when:

- `origin = "synthetic"` — a negative control or unit-test fixture, which
  never touched real project data and could not have influenced selection; or
- `status = "withdrawn_duplicate"` — declared, then found to be an exact
  identity duplicate of an earlier trial. `trial_id` equality already prevents
  this; the status exists for reconstruction, where two evidence sources
  describe one historical trial.

There is no "it didn't work out" exclusion, no "it was exploratory"
exclusion, and no "different researcher" exclusion.

### 6.4 Conservatism

Where family membership is ambiguous, the trial is **included**. Specifically:

- a reconstructed trial with an unknown profile is counted in every profile's
  family, not in none;
- an unknown `target_family` is counted in the family being queried;
- an uncertain historical count contributes its **upper** bound (§8.3).

`n_trials()` returns the conservative number. A narrower number is available
only through `n_trials_confirmed()`, which is explicitly labelled and may not
be used to deflate a published result.

---

## 7. M01 integration and guardrails

### 7.1 The research path

`research_dsr.research_deflated_sharpe(returns, registry=..., scope=...,
trial_id=...)` returns the M01 result **plus** a provenance record:

- the family scope queried;
- `n_trials` and how it decomposed (confirmed / uncertain / conservative);
- the registry path, record count, and a content hash of the registry file at
  the moment of the call;
- the registry version and the DSR version.

It has no `n_trials` parameter. There is no way to pass one.

### 7.2 The declaration precondition

`research_deflated_sharpe` requires a `trial_id` that is already declared, and
raises otherwise. A result cannot be deflated before its own trial exists.

### 7.3 The static guardrail

A test walks the AST of every module under `validation/`, `labeling/`,
`models/`, `research/`, `forecast/` and `tools/` and fails if it calls
`deflated_sharpe(...)` with an `n_trials` argument, except for an explicit
allowlist:

- `validation/negative_controls.py` — synthetic nulls, no real data, permitted
  by §6.3;
- `validation/research_dsr.py` — the wrapper itself, which supplies the
  registry-derived count.

The allowlist is in the test, so extending it is a visible diff on a governed
guardrail rather than a new import nobody notices. Tests themselves are exempt:
M01 remains directly unit-testable as a mathematical primitive.

---

## 8. Historical reconstruction

Frozen **before** reconstruction begins, which is the point of this section's
position in the document.

### 8.1 Evidence tiers

| Tier | Meaning | Effect |
|---|---|---|
| `committed_document` | a governed report or spec in git | confirmed |
| `git_history` | a commit message or diff establishing the choice | confirmed |
| `local_report` | an untracked `reports/<stage>/` artifact | confirmed, evidence path recorded |
| `inference` | the choice must have existed for a documented result to exist | **uncertain** |

Every reconstructed record carries `reconstructed = true` and a non-empty
`evidence` list of `{tier, ref, note}`. A record without evidence cannot be
written — the constructor rejects it.

### 8.2 What may not be invented

No trial is created for a choice that has no evidence, however likely it seems
that one was made. Suspected-but-unevidenced work is reported as a stated gap
in `G1_1_RESULT.md`, not as records.

### 8.3 Three counts, always reported together

- **confirmed** — trials whose evidence is tier 1–3.
- **uncertain** — a `[low, high]` band for choices that evidence supports only
  as a range (a threshold grid whose size is documented approximately; a sweep
  whose arm count is implied but not enumerated).
- **conservative** — `confirmed + high(uncertain)`. This is the number DSR
  uses.

A single precise-looking number is never published on its own. Where a band
exists, the band is published with it.

### 8.4 Determinism

Reconstruction is a curated table in `validation/trial_history.py`, with each
record citing its evidence — not a scraper over the working tree. A scraper
would produce different counts on different machines depending on which
untracked `reports/` directories happen to be present, which is the opposite of
auditable. The table is code, is reviewed as code, and materializes byte
identically on any checkout.

---

## 9. Historical G1 handling

Binding, and stated here so no later document can quietly differ:

1. G1's verdict remains **`G1_INDETERMINATE`**. Nothing in G1.1 changes it.
2. M01–M04 implementation and the negative-control findings remain historical
   facts, at the numbers G1 published.
3. The corrected T3 semantics proposed in `G1_SPEC.md` §7 A3 apply
   **prospectively only**, to gates declared after 2026-08-18. They are not
   applied to G1's own controls.
4. The trial registry resolves the `n_trials` infrastructure blocker
   **prospectively**. It does not retroactively validate any number that was
   published against a hand-counted `n_trials`.
5. Phase A′ becomes eligible only after G1.1 passes audit — and eligibility is
   not permission; the T3 decision of `G1_SPEC.md` §7 remains open and is the
   owner's.

Recomputing G1's negative controls with a registry-derived count and reporting
a different verdict is explicitly forbidden.

---

## 10. Decision rule

`G1_1_PASS` requires all eleven:

1. Trial definition frozen before reconstruction (this document, committed
   first).
2. Stable trial identity implemented, with the §3 tests passing.
3. Trial vs execution distinction implemented and tested.
4. Registry append-only and corruption-detectable.
5. Family semantics frozen (§6) and implemented.
6. M01 research path derives `n_trials` from registry provenance.
7. Historical reconstruction explicitly marked, evidence-cited, auditable.
8. Guardrails against future undercounting exist and are tested.
9. Targeted tests pass.
10. Full suite passes.
11. Independent audit finds no unresolved correctness defect.

`G1_1_INDETERMINATE` if reconstruction uncertainty is wide enough that no
defensible conservative count can be stated. `G1_1_FAIL` if the registry can be
made to undercount — if any path lets a research result publish a `n_trials`
lower than the registry's conservative count for its family.

A count that is *large* is not a failure. A count that is *unjustifiable* is.

---

## 11. Out of scope

- **Phase A′.** No strategy is evaluated here. The registry is built and
  populated; it is not consumed by an edge search.
- **Wiring A′ runners into the registry.** The interface and the guardrail
  exist; integrating a runner would be starting A′.
- **The T3 decision** (`G1_SPEC.md` §7 A3). Still open, still the owner's.
- **PBO over the real trial population.** Needs real result series per trial,
  which is A′ work.
- **Deleting or amending historical trials.** Not implemented; the registry has
  no such operation by design.
