# C5.0 M00 — cost sensitivity: result

Run: 2026-08-11. Cost model `c50_m00_v1`, sha `1cd4db9fe8e8`.
Inputs: `config.TAKER_FEE_PCT = 0.05`, `config.SLIPPAGE_PCT = 0.03`,
funding 0.01%/8h (new — previously not modelled anywhere in the project).

## Gate G0 verdict: **PASS** for the swing profile

The 4h swing horizon is not killed by execution cost. This contradicts the
expectation stated when M00 was proposed, and the correction is recorded here
rather than quietly dropped.

| holding | fees | slippage | funding | total | cost/R @1.5% stop | win rate needed @2R |
|---|---|---|---|---|---|---|
| 1 bar (4h) | 0.100 | 0.030 | 0.005 | **0.135%** | 0.090 | 36.3% |
| 6 bars (24h) | 0.100 | 0.030 | 0.030 | **0.160%** | 0.107 | 36.9% |
| 12 bars (48h) | 0.100 | 0.030 | 0.060 | **0.190%** | 0.127 | **37.6%** |
| 24 bars (96h) | 0.100 | 0.030 | 0.120 | **0.250%** | 0.167 | 38.9% |

At the swing horizon the bar is a **37.6% win rate at 2R**, against 33.3% at
zero cost. Execution takes 4.3 percentage points. That is a real handicap and
a survivable one.

## The decisive variable is stop distance, not holding period

| stop distance | cost/R | win rate needed @2R | @1.5R |
|---|---|---|---|
| 0.3% | 0.633 | 54.4% | 65.3% |
| 0.5% | 0.380 | 46.0% | 55.2% |
| 1.0% | 0.190 | 39.7% | 47.6% |
| 1.5% | 0.127 | 37.6% | 45.1% |
| 3.0% | 0.063 | 35.4% | 42.5% |

Below roughly a 0.8% stop the required win rate passes 41% and the strategy is
paying more than a fifth of every R to execution before it has an edge.

This confirms quantitatively what `pipeline.py:310` already asserted
qualitatively — 15m-ATR stops are eaten by fees. The intraday profile at a
0.3% stop needs a 48.2% win rate at 2R, which is why leaving intraday disabled
remains correct.

## Consequences

1. **G0 passes → phases A–E are not cancelled.** The C5.0 programme proceeds
   as specified.
2. **Funding must be added to every existing backtest.** `backtest.py:133`,
   `tools/deep_backtest.py:432`, `tools/deep_discovery.py:276`,
   `tools/deep_diagnostics.py:267`, `tools/swing_hypothesis_simulator.py:345,509`
   and `tools/swing_hypothesis_walkforward.py:440` all compute
   `cost_pct = (2*TAKER_FEE_PCT + SLIPPAGE_PCT)/100` and hold positions for
   hours without paying funding. Every net number they have produced is
   overstated — by 0.06pp per trade at the swing horizon, more for longer
   holds. Not fixed in this change: it alters historical results and belongs
   in its own task with its own audit.
3. **A minimum stop distance is a live constraint**, not a preference. Any
   future setup filter should treat sub-0.8% stops as structurally
   disadvantaged.
4. **The +0.13 ranking edge is not addressed by this.** M00 measures the bar,
   not whether anything clears it. Volatility ranking carries no direction and
   generates no trades, so it has no round trip to pay — the cost question
   applies to the swing engine, not to C4.4/C4.5.

## What this does not establish

Slippage remains a modelled constant scaled by a square root, calibrated at a
notional nobody has verified. Funding uses a long-run average, not the realized
series. Both should be replaced with measured data before the numbers above are
treated as precise rather than indicative. The ordering of the conclusions —
swing survives, tight stops do not — is robust to plausible errors in either.
