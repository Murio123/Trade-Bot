# C5.0 — governed decision record: T3 / option 3

Owner decision, recorded 2026-08-19. Machine-readable form:
`reports/c50/g1_governance.json`, asserted by `tests/test_g1_governance.py`.

This document exists to keep two things apart that are easy to blur once a
project moves on: **what was measured** and **what was then decided**. The
first is history and cannot change. The second carries a date and applies only
forwards.

---

## 1. Historical fact — not amended by this decision

> **G1 verdict: `G1_INDETERMINATE`.**

That is what `G1_RESULT.md` published and what it still says. T3 as frozen in
`G1_SPEC.md` §4.1 required the share of replications with positive mean gross
expectancy to fall in `[0.394, 0.606]`; measured at the frozen `R = 200` it
read 0.0350 after the A1 fix. **It failed, and it is recorded as failed.**

This decision does **not**:

- change G1 to `G1_PASS`, on this line or any other;
- adopt A3 (the long/short-gap statistic) for the historical gate;
- reinterpret the historical T3 measurement;
- re-run G1 in order to obtain a cleaner verdict.

A3's status in `G1_SPEC.md` §7 stays **PROPOSED, NOT ADOPTED** for G1. The
argument recorded there against adopting it — that the ordering was run,
fail, fix, fail, then replace the statistic — was accepted, and option 1 was
therefore declined.

Nobody reading the C5.0 documents afterwards should be able to come away with
the impression that G1 originally passed. If this record ever appears to
create that impression, this record is wrong, not `G1_RESULT.md`.

---

## 2. Prospective decision — option 3

Of the three resolutions `G1_SPEC.md` §7 A3 put to the owner, **option 3** was
chosen: re-freeze T3 for a future gate, record G1 as indeterminate on that one
line, and take the apparatus properties as demonstrated.

**Recorded and effective 2026-08-19, forwards only.** The separate cutoff
that governs which gates the corrected T3 semantics reach — gates declared
after **2026-08-18** — is the frozen one and is unchanged (§2.2).

### 2.1 What is accepted

The substantive properties demonstrated by M01–M04 and the negative-control
suite are accepted as sufficient to proceed to Phase A′:

1. leakage controls;
2. purge / embargo / CPCV behaviour;
3. sample-weight and uniqueness behaviour;
4. DSR calibration behaviour;
5. negative-control false-positive-rate behaviour.

These were demonstrated at the numbers `G1_RESULT.md` published. Accepting
them is a judgement about sufficiency, not a re-scoring: no control was
re-run, re-scored, or re-interpreted to produce this record.

### 2.2 What is deferred

Historical T3 is treated as a **specification problem for that historical
gate** — a criterion that turned out to lack power for the property its own
closing sentence names. Treated, not erased: the failing measurement stays in
the record with its evidence.

**T3 must be re-frozen prospectively before any future gate that depends on
the directional-symmetry property.** Two constraints bind that re-freeze:

- the replacement criterion must be frozen **before** the controls it will
  judge are run, as `G1_SPEC.md` §1 requires of any criterion;
- **the new criterion must not be chosen using Phase A′ results.** A statistic
  selected after seeing A′ is the same defect as a statistic selected after
  seeing G1, one stage later.

The corrected T3 semantics of `G1_SPEC.md` §7 A3 therefore apply **to gates
declared after 2026-08-18 and to no earlier one**. That cutoff is not this
decision's date and is not moved by it: it was fixed by
`TRIAL_REGISTRY_SPEC.md` §9 item 3 and repeated in `G1_1_RESULT.md` §8 item 3,
both written before the decision was taken. This record is dated 2026-08-19
and changes nothing about which gates the corrected semantics reach. An
earlier draft of this document wrote 2026-08-19 for both and claimed the two
agreed; the audit caught it, and the frozen cutoff governs.

---

## 3. Trial accounting — unchanged by this decision

G1.1 remains the trial-accounting gate and stands on its own criteria
(`TRIAL_REGISTRY_SPEC.md`, frozen at `dd45a9b` before any trial was
reconstructed). Its verdict is **`G1_1_PASS`**.

The canonical count entering Phase A′, for the family
`(trade_expectancy, swing, BTCUSDT, trade_r)`:

> **`n_trials = 100`** — 83 confirmed plus an uncertain band of 17,
> conservative basis.

Two properties of that number are part of this decision, not decoration:

- **It comes from the registry machinery.** `research_deflated_sharpe()` has
  no trial-count parameter; the count is derived from the declared registry
  under an explicit family scope, with the registry's content hash pinned into
  the provenance of every published DSR. The figure quoted in this document
  and in `g1_governance.json` is a **witness for the reader**, not an input.
  A′ that reads 100 from a document instead of from the registry has
  reintroduced the free parameter the gate exists to remove.
- **The uncertainty is preserved as stated.** The band of 17 is pre-C1 and
  stays a band; the gaps in `G1_1_RESULT.md` §4.3 stay open. Nothing here may
  present the reconstruction as more precise than it is.

---

## 4. Consequence

**Phase A′ is UNBLOCKED as of 2026-08-19.** Both blockers that stood in front
of it are resolved: the `n_trials` infrastructure blocker by G1.1, and the T3
decision by this record.

Unblocked is not started. A′ has not begun, and when it does it declares each
of its hypotheses in the registry — on top of the 100 — before their results
are looked at.
