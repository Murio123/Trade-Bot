# P1 — Funding-aware cost model

Prerequisite inserted by `ARCHITECTURE.md` Amendment 1. Blocks phase A′.
Not a research module: a correctness fix to existing code that changes
historical results.

---

## Attribution anchor

```
b3eca107f3d0f76928daf73448f00f3c294fe5c7
2026-08-11 18:47:23 +0300
research: freeze pre-funding baseline before P1 cost model
```

**Every P1 evaluation compares against this exact commit.** Not against the
working tree, not against a later commit, not against `main`.

This commit is immutable. It is never amended, rebased, or rewritten. If
something in it turns out to be wrong, the fix is a new commit on top —
because a rewritten anchor silently invalidates every comparison already made
against it.

Descriptive companion: `reports/c45/BASELINE.md`, sealed
`762026aa7ce1264353412c044a2665fb2df4c739d33c8edd25ef4f9269ecf90e`
(`BASELINE.sha256`). The baseline document was written against a dirty tree;
the commit above is what resolved that, and the commit — not the document —
is the anchor.

### Verified at anchor time

| Check | Result |
|---|---|
| Test suite | 1559 passed, 0 failed |
| `git diff --check` | clean |
| `BASELINE.sha256` | OK |
| Artifact `c44_ridge_frozen_v1` vs registry pin | match, `a5013922…` |
| Quarantined hashes vs BASELINE §3.3 | all match |
| Cost model `c50_m00_v1` hash | `1cd4db9fe8e8` |

---

## Scope

Charge funding wherever a position is held. Nine known sites:

| Site | Current |
|---|---|
| `backtest.py:133` | `cost_pct = (2*TAKER_FEE_PCT + SLIPPAGE_PCT)/100` |
| `tools/deep_backtest.py:432` | same |
| `tools/deep_discovery.py:276` | same |
| `tools/deep_diagnostics.py:267` | same |
| `tools/swing_hypothesis_simulator.py:345` | same |
| `tools/swing_hypothesis_simulator.py:509` | same |
| `tools/swing_hypothesis_walkforward.py:440` | same |
| `tools/forecast_metrics.py:34-35,366` | private copies of the two constants; reports `modeled_round_trip_cost_pct = 0.13` |
| `analyzer/outcomes.py:100` | `net_after_costs = r72 - (2*taker_fee_pct + slippage_pct)` over 72h — nine funding intervals, none charged |

Seven of the nine duplicate the same expression. A single shared helper is the
point of the task; nine independently patched call sites would recreate the
defect the next time a cost component is added.

Expected magnitude: **+0.06pp per trade at the 48h swing horizon, +0.09pp at
the 72h outcome horizon**, at M00's default 0.01%/8h.

---

## Out of scope

No retraining. No feature change. No label change. No walk-forward geometry
change. No cache refresh. No new hypothesis. No change to
`SCORE_ALERT_MIN`. DRY_RUN unchanged.

Changing anything on this list breaks attribution: a post-P1 difference would
no longer be traceable to funding alone, and the anchor would be spent for
nothing.

---

## Falsifiable expectations

From `BASELINE.md` §12. These are predictions, recorded before the work, and
each one can fail:

1. **Every net figure moves down.** Costs strictly increase, so a net number
   that improves indicates a defect in the retrofit — not a discovery.
2. **Verdicts can only flip from accepted to rejected.** A rejection reversing
   to acceptance is impossible under strictly higher costs and would mean
   something other than cost changed.
3. **H1, H2 and the regime gate stay rejected.** They were rejected against
   costs that were too low; correcting the bias strengthens those rejections.
   If any reopens, the retrofit is wrong.
4. **The +0.13 ranking edge is unaffected.** Volatility ranking carries no
   direction and generates no trades, so it has no round trip to pay. A change
   there means funding leaked into a path that has no position.

---

## Definition of done

- One shared funding-aware cost helper; all nine sites route through it.
- Old vs new reported side by side for every artifact in `BASELINE.md` §5.
- Tests for the funding term, including the zero-holding and long-hold edges.
- Full suite green.
- Its own audit — this task changes historical results, so it does not ride
  along on someone else's review.
