"""P1 regression suite: realized funding, and the accounting identity it enters.

Every test here exists because getting the corresponding detail wrong would
produce a plausible-looking net number that is quietly biased. The sign, the
boundary convention and the fail-closed behaviour are the three that would
survive code review by being invisible.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation.funding import (FundingDataUnavailable, FundingError,
                                FundingSeries, dataset_stem,
                                load_funding_series, series_from_binance_records,
                                to_ms)
from validation.trade_costs import (FUNDING_NOT_MODELLED, TradeCosts,
                                    cost_pct_to_r, funding_summary,
                                    net_pct_after_costs, trade_costs,
                                    transaction_cost_pct, transaction_cost_r)

REPO = Path(__file__).resolve().parents[1]

H8 = 8 * 3_600_000          # one settlement interval, ms
T0 = 1_700_000_000_000      # arbitrary epoch ms, aligned to nothing in particular

# A canonical position: entry 100, stop 99 -> one R is 1.0 of price, so a cost of
# 1% of notional is exactly 1.0 R. That makes every R assertion below readable by
# inspection rather than by arithmetic.
ENTRY_PRICE = 100.0
RISK_DISTANCE = 1.0


def series(rates, *, start=T0, step=H8, pad=True) -> FundingSeries:
    """A settlement series at `start`, `start+step`, ... with the given rates.

    Padded with one zero-rate settlement on each side by default. Coverage is
    strict at both edges — the series says nothing before its first observation
    and nothing after its last — so a test that wants to charge the settlement
    at `start` has to enter inside the observed window. The real dataset is
    built with the same margin (funding from 2022-04-01, klines from 04-28),
    which is why strictness costs nothing in production and everything here.
    """
    times = [start + i * step for i in range(len(rates))]
    rates = list(rates)
    if pad:
        times = [start - step] + times + [times[-1] + step]
        rates = [0.0] + rates + [0.0]
    return FundingSeries.from_records(times, rates, symbol="BTCUSDT",
                                      exchange="binance")


def costs_for(rates, *, side, entry, exit, gross_r=1.0, **kw) -> TradeCosts:
    return trade_costs(entry_price=ENTRY_PRICE, risk_distance=RISK_DISTANCE,
                       side=side, entry_time=entry, exit_time=exit,
                       funding=series(rates), gross_r=gross_r,
                       taker_fee_pct=0.0, slippage_pct=0.0, **kw)


# ---------------------------------------------------------------------------
# Economics: the sign is the thing most likely to be wrong.
# ---------------------------------------------------------------------------

def test_long_pays_when_funding_is_positive():
    c = costs_for([0.0001], side="long", entry=T0 - 1, exit=T0 + 1)
    assert c.funding_settlements == 1
    assert c.funding_pct == pytest.approx(0.01)   # 0.0001 -> 0.01%
    assert c.funding_r > 0                        # a cost


def test_short_is_credited_when_funding_is_positive():
    c = costs_for([0.0001], side="short", entry=T0 - 1, exit=T0 + 1)
    assert c.funding_pct == pytest.approx(-0.01)
    assert c.funding_r < 0                        # a credit, not a cost


def test_long_is_credited_when_funding_is_negative():
    c = costs_for([-0.0001], side="long", entry=T0 - 1, exit=T0 + 1)
    assert c.funding_pct == pytest.approx(-0.01)
    assert c.funding_r < 0


def test_short_pays_when_funding_is_negative():
    c = costs_for([-0.0001], side="short", entry=T0 - 1, exit=T0 + 1)
    assert c.funding_pct == pytest.approx(0.01)
    assert c.funding_r > 0


def test_funding_is_never_clamped_to_a_cost():
    """A credit must reach net_r as a credit. Clamping at zero would be a
    one-directional bias dressed up as conservatism."""
    long_c = costs_for([-0.0005], side="long", entry=T0 - 1, exit=T0 + 1)
    assert long_c.total_r < 0
    assert long_c.net_r > long_c.gross_r


def test_zero_rate_settlement_is_crossed_but_free():
    c = costs_for([0.0], side="long", entry=T0 - 1, exit=T0 + 1)
    assert c.funding_settlements == 1
    assert c.funding_pct == 0.0
    assert c.net_r == pytest.approx(c.gross_r)


def test_no_settlement_crossed_is_a_valid_zero():
    """Between two settlements nothing is owed — and that zero is legitimate,
    unlike the zero produced by missing data."""
    c = costs_for([0.0001, 0.0001], side="long",
                  entry=T0 + 1, exit=T0 + H8 - 1)
    assert c.funding_settlements == 0
    assert c.funding_pct == 0.0
    assert c.funding_modelled is True


def test_exactly_one_settlement():
    c = costs_for([0.0002, 0.0003, 0.0004], side="long",
                  entry=T0 + H8 - 1, exit=T0 + H8 + 1)
    assert c.funding_settlements == 1
    assert c.funding_pct == pytest.approx(0.03)   # only the middle rate


def test_multiple_settlements_sum_the_actual_rates():
    """Not `latest_rate x holding_time`: the rates differ, and the sum of the
    real ones is the only correct answer."""
    rates = [0.0001, 0.0002, 0.0003, 0.0004]
    c = costs_for(rates, side="long", entry=T0 - 1, exit=T0 + 3 * H8 + 1)
    assert c.funding_settlements == 4
    assert c.funding_pct == pytest.approx(sum(rates) * 100)
    # The extrapolation P1 forbids would have given 4 x the LAST rate.
    assert c.funding_pct != pytest.approx(4 * 0.0004 * 100)


def test_sign_change_inside_the_trade_nets_out():
    rates = [0.0005, -0.0004]
    c = costs_for(rates, side="long", entry=T0 - 1, exit=T0 + H8 + 1)
    assert c.funding_settlements == 2
    assert c.funding_pct == pytest.approx(0.01)   # 0.05% - 0.04%


def test_credit_and_debit_can_cancel_exactly():
    c = costs_for([0.0003, -0.0003], side="long", entry=T0 - 1,
                  exit=T0 + H8 + 1)
    assert c.funding_settlements == 2
    assert c.funding_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The boundary convention: (entry, exit]. Both halves are asserted, because
# either one alone is satisfied by an off-by-one implementation.
# ---------------------------------------------------------------------------

def test_settlement_exactly_at_entry_is_not_charged():
    c = costs_for([0.0001], side="long", entry=T0, exit=T0 + 1)
    assert c.funding_settlements == 0


def test_settlement_exactly_at_exit_is_charged():
    c = costs_for([0.0001], side="long", entry=T0 - 1, exit=T0)
    assert c.funding_settlements == 1


def test_zero_length_hold_crosses_nothing_even_on_a_stamp():
    c = costs_for([0.0001], side="long", entry=T0, exit=T0)
    assert c.funding_settlements == 0
    assert c.holding_hours == 0.0


def test_millisecond_jitter_in_settlement_stamps_is_not_a_gap():
    """Binance stamps funding times a millisecond or two late (…400002). Naive
    gap detection would declare a gap and refuse the whole interval."""
    times = [T0, T0 + H8 + 2, T0 + 2 * H8 + 1, T0 + 3 * H8]
    s = FundingSeries.from_records(times, [0.0001] * 4, symbol="BTCUSDT",
                                   exchange="binance")
    assert s.gaps == ()
    assert s.covers(T0, T0 + 3 * H8)


def test_exit_before_entry_is_rejected():
    with pytest.raises(FundingError):
        trade_costs(entry_price=100.0, risk_distance=1.0, side="long",
                    entry_time=T0 + H8, exit_time=T0,
                    funding=series([0.0001]))


# ---------------------------------------------------------------------------
# Accounting identity and R normalisation.
# ---------------------------------------------------------------------------

def test_net_equals_gross_minus_every_cost_component():
    c = trade_costs(entry_price=ENTRY_PRICE, risk_distance=RISK_DISTANCE,
                    side="long", entry_time=T0 - 1, exit_time=T0 + H8 + 1,
                    funding=series([0.0001, 0.0002]), gross_r=2.0,
                    taker_fee_pct=0.05, slippage_pct=0.03)
    c.assert_identity()
    assert c.net_r == pytest.approx(
        c.gross_r - c.fee_r - c.slippage_r - c.funding_r)
    assert c.total_pct == pytest.approx(
        c.fee_pct + c.slippage_pct + c.funding_pct)


def test_r_normalisation_scales_with_stop_distance():
    """A cost is a bigger share of a tighter stop — the whole reason cost is
    reported in R and not in percent."""
    tight = trade_costs(entry_price=100.0, risk_distance=1.0, side="long",
                        entry_time=T0 - 1, exit_time=T0 + 1,
                        funding=series([0.0001]), gross_r=1.0,
                        taker_fee_pct=0.05, slippage_pct=0.03)
    wide = trade_costs(entry_price=100.0, risk_distance=2.0, side="long",
                       entry_time=T0 - 1, exit_time=T0 + 1,
                       funding=series([0.0001]), gross_r=1.0,
                       taker_fee_pct=0.05, slippage_pct=0.03)
    assert tight.funding_r == pytest.approx(2 * wide.funding_r)
    assert tight.total_r == pytest.approx(2 * wide.total_r)


def test_funding_r_is_the_cash_bill_over_one_r_of_risk():
    """The identity behind the normalisation: funding cash is rate x notional,
    one R is stop_distance x size, so funding_r = pct/100 x price / distance."""
    c = costs_for([0.0001], side="long", entry=T0 - 1, exit=T0 + 1)
    assert c.funding_r == pytest.approx(
        c.funding_pct / 100.0 * ENTRY_PRICE / RISK_DISTANCE)


def test_funding_cost_reduces_net_r_and_credit_improves_it():
    debit = costs_for([0.0010], side="long", entry=T0 - 1, exit=T0 + 1,
                      gross_r=2.0)
    credit = costs_for([-0.0010], side="long", entry=T0 - 1, exit=T0 + 1,
                       gross_r=2.0)
    assert debit.net_r < 2.0 < credit.net_r
    assert debit.net_r == pytest.approx(4.0 - credit.net_r)  # symmetric


def test_transaction_cost_is_not_double_counted_in_total():
    c = trade_costs(entry_price=100.0, risk_distance=1.0, side="long",
                    entry_time=T0 - 1, exit_time=T0 + 1,
                    funding=series([0.0001]), gross_r=1.0,
                    taker_fee_pct=0.05, slippage_pct=0.03)
    # transaction_r appears once in total_r, alongside funding_r — not twice.
    assert c.total_r == pytest.approx(c.transaction_r + c.funding_r)
    assert c.transaction_r == pytest.approx(c.fee_r + c.slippage_r)


def test_funding_does_not_leak_into_the_admission_gate_quantity():
    """`transaction_cost_r` gates candidates before the hold is known. If
    funding entered it, P1 would change which trades were taken — forbidden."""
    assert transaction_cost_r(100.0, 1.0, taker_fee_pct=0.05,
                              slippage_pct=0.03) == pytest.approx(0.13)
    assert transaction_cost_pct(0.05, 0.03) == pytest.approx(0.13)


def test_zero_risk_distance_yields_zero_r_not_a_division_error():
    """Legacy behaviour (`if risk_dist else 0.0`) preserved verbatim: changing
    it would move results for a reason unrelated to funding."""
    assert cost_pct_to_r(0.13, 100.0, 0.0) == 0.0
    c = costs_for([0.0001], side="long", entry=T0 - 1, exit=T0 + 1)
    zero = trade_costs(entry_price=100.0, risk_distance=0.0, side="long",
                       entry_time=T0 - 1, exit_time=T0 + 1,
                       funding=series([0.0001]), gross_r=1.0)
    assert zero.funding_r == 0.0 and c.funding_r != 0.0
    assert zero.funding_pct != 0.0  # the percent term is still reported


def test_pre_p1_arithmetic_is_reproduced_when_no_settlement_is_crossed():
    """The compatibility guarantee: with no settlement in the interval, net is
    exactly the old `gross - (2*taker + slippage)/100 * price / risk`."""
    legacy = (2 * 0.05 + 0.03) / 100 * 100.0 / 1.0
    c = trade_costs(entry_price=100.0, risk_distance=1.0, side="long",
                    entry_time=T0 + 1, exit_time=T0 + H8 - 1,
                    funding=series([0.0001, 0.0001]), gross_r=1.0,
                    taker_fee_pct=0.05, slippage_pct=0.03)
    assert c.total_r == pytest.approx(legacy)
    assert c.net_r == pytest.approx(1.0 - legacy)


def test_net_pct_helper_matches_the_percent_decomposition():
    net = net_pct_after_costs(gross_pct=5.0, side="long", entry_time=T0 - 1,
                              exit_time=T0 + 1, funding=series([0.0001]),
                              taker_fee_pct=0.05, slippage_pct=0.03)
    assert net == pytest.approx(5.0 - 0.13 - 0.01)


# ---------------------------------------------------------------------------
# Data integrity: unavailable is not zero, and nothing may be read from the
# future.
# ---------------------------------------------------------------------------

def test_interval_past_the_last_observation_fails_closed():
    s = series([0.0001, 0.0001], pad=False)
    with pytest.raises(FundingDataUnavailable):
        s.settlement_rates(T0, T0 + 5 * H8)


def test_interval_before_the_first_observation_fails_closed():
    s = series([0.0001, 0.0001], pad=False)
    with pytest.raises(FundingDataUnavailable):
        s.settlement_rates(T0 - 5 * H8, T0 + H8)


def test_a_gap_overlapping_the_trade_fails_closed():
    """A missing settlement inside the hold is indistinguishable from a cheap
    trade unless it is refused."""
    times = [T0, T0 + H8, T0 + 6 * H8, T0 + 7 * H8]
    s = FundingSeries.from_records(times, [0.0001] * 4, symbol="BTCUSDT",
                                   exchange="binance")
    assert len(s.gaps) == 1
    with pytest.raises(FundingDataUnavailable):
        s.settlement_rates(T0 + H8, T0 + 6 * H8)


def test_a_gap_outside_the_trade_does_not_block_it():
    times = [T0, T0 + H8, T0 + 6 * H8, T0 + 7 * H8, T0 + 8 * H8]
    s = FundingSeries.from_records(times, [0.0001] * 5, symbol="BTCUSDT",
                                   exchange="binance")
    assert len(s.gaps) == 1
    # Entirely after the gap: the missing settlements cannot have been part of
    # this hold, so refusing it would be over-strict rather than honest.
    assert s.settlement_rates(T0 + 6 * H8, T0 + 8 * H8).size == 2


def test_missing_series_fails_closed_for_a_real_hold():
    with pytest.raises(FundingDataUnavailable):
        trade_costs(entry_price=100.0, risk_distance=1.0, side="long",
                    entry_time=T0, exit_time=T0 + H8, funding=None)


def test_missing_series_is_accepted_only_for_a_zero_length_hold():
    c = trade_costs(entry_price=100.0, risk_distance=1.0, side="long",
                    entry_time=T0, exit_time=T0, funding=None, gross_r=1.0)
    assert c.funding_pct == 0.0 and c.funding_settlements == 0


def test_not_modelled_sentinel_is_recorded_not_hidden():
    """The one route to a zero on a real hold announces itself, so a report
    built from it cannot claim to have charged funding."""
    c = trade_costs(entry_price=100.0, risk_distance=1.0, side="long",
                    entry_time=T0, exit_time=T0 + 3 * H8,
                    funding=FUNDING_NOT_MODELLED, gross_r=1.0)
    assert c.funding_pct == 0.0
    assert c.funding_modelled is False
    assert funding_summary([{"funding_r": 0.0, "cost_r": 0.13,
                             "direction": "long",
                             "funding_modelled": False}])["funding_modelled"] \
        is False


def test_no_look_ahead_a_rate_settling_after_the_exit_is_never_charged():
    """The decisive look-ahead test: a large rate one settlement after the exit
    must not touch the bill."""
    early = costs_for([0.0001, 0.0500], side="long", entry=T0 - 1, exit=T0 + 1)
    assert early.funding_settlements == 1
    assert early.funding_pct == pytest.approx(0.01)


def test_rate_is_selected_by_settlement_timestamp_not_by_position():
    """Rates are matched to their own stamps, so an unsorted or duplicated feed
    cannot shift the assignment."""
    times = [T0 + 2 * H8, T0, T0 + H8, T0]          # unsorted, one duplicate
    rates = [0.0003, 0.0001, 0.0002, 0.0001]
    s = FundingSeries.from_records(times, rates, symbol="BTCUSDT",
                                   exchange="binance")
    assert len(s) == 3
    assert list(s.settlement_rates(T0, T0 + 2 * H8)) == pytest.approx(
        [0.0002, 0.0003])


def test_replay_is_deterministic():
    a = costs_for([0.0001, -0.0002, 0.0003], side="short", entry=T0 - 1,
                  exit=T0 + 2 * H8 + 1)
    b = costs_for([0.0001, -0.0002, 0.0003], side="short", entry=T0 - 1,
                  exit=T0 + 2 * H8 + 1)
    assert a == b


def test_empty_series_is_refused_at_construction():
    with pytest.raises(FundingError):
        FundingSeries.from_records([], [], symbol="BTCUSDT", exchange="binance")


def test_non_finite_rate_is_refused():
    with pytest.raises(FundingError):
        FundingSeries.from_records([T0], [float("nan")], symbol="BTCUSDT",
                                   exchange="binance")


def test_bad_side_is_refused():
    with pytest.raises(FundingError):
        costs_for([0.0001], side="flat", entry=T0 - 1, exit=T0 + 1)
    with pytest.raises(FundingError):
        costs_for([0.0001], side=True, entry=T0 - 1, exit=T0 + 1)


def test_modal_interval_is_measured_not_assumed():
    """Binance moved some symbols to 4h funding. A hardcoded 8h would
    mis-detect every gap on those."""
    four_h = series([0.0001] * 5, step=4 * 3_600_000, pad=False)
    assert four_h.modal_interval_ms == 4 * 3_600_000
    assert four_h.gaps == ()


def test_timestamps_accept_datetimes_and_naive_stamps_as_utc():
    aware = pd.Timestamp("2023-11-14 22:13:20+00:00")
    naive = pd.Timestamp("2023-11-14 22:13:20")
    assert to_ms(aware) == to_ms(naive) == T0


def test_binance_records_build_the_same_series_as_the_cache():
    records = [{"fundingTime": T0, "fundingRate": "0.00010000"},
               {"fundingTime": T0 + H8, "fundingRate": "-0.00020000"},
               {"fundingTime": T0 + 2 * H8, "fundingRate": "0.00030000"}]
    s = series_from_binance_records(records)
    assert len(s) == 3
    # (T0, T0+2*H8] -> the -0.02% and the +0.03% settlements, not the first.
    assert s.realized_funding_pct(T0, T0 + 2 * H8, "long") == pytest.approx(0.01)


def test_empty_binance_payload_fails_closed():
    with pytest.raises(FundingDataUnavailable):
        series_from_binance_records([])


# ---------------------------------------------------------------------------
# Aggregation, as reported by every migrated tool.
# ---------------------------------------------------------------------------

def test_funding_summary_reports_both_tails_and_the_direction_split():
    trades = [
        {"funding_r": 0.04, "cost_r": 0.13, "direction": "long",
         "funding_settlements": 6, "funding_modelled": True},
        {"funding_r": -0.02, "cost_r": 0.13, "direction": "short",
         "funding_settlements": 3, "funding_modelled": True},
        {"funding_r": 0.0, "cost_r": 0.13, "direction": "long",
         "funding_settlements": 0, "funding_modelled": True},
    ]
    s = funding_summary(trades)
    assert s["n"] == 3
    assert s["total_funding_r"] == pytest.approx(0.02)
    assert s["largest_debit_r"] == pytest.approx(0.04)
    assert s["largest_credit_r"] == pytest.approx(-0.02)
    assert s["share_crossing_settlement"] == pytest.approx(2 / 3)
    assert s["by_direction"]["long"]["n"] == 2
    assert s["by_direction"]["short"]["total_funding_r"] == pytest.approx(-0.02)
    assert s["funding_modelled"] is True


def test_funding_summary_share_of_costs_uses_absolute_values():
    """With a credit and a debit of equal size, a signed ratio would report
    funding as ~0% of costs even though it moved both trades."""
    trades = [{"funding_r": 0.10, "cost_r": 0.10, "direction": "long"},
              {"funding_r": -0.10, "cost_r": 0.10, "direction": "short"}]
    s = funding_summary(trades)
    assert s["funding_share_of_costs"] == pytest.approx(1.0)


def test_funding_summary_of_nothing_is_empty_not_zero():
    s = funding_summary([])
    assert s["n"] == 0 and s["total_funding_r"] is None


# ---------------------------------------------------------------------------
# Integration: one implementation, no survivors of the old one.
# ---------------------------------------------------------------------------

LEGACY_FORMULA = re.compile(
    r"2\s*\*\s*(config\.)?TAKER_FEE_PCT\s*\+\s*(config\.)?SLIPPAGE_PCT")

# The canonical layer is allowed to state the formula — it owns it. Tests may
# quote it to assert parity.
FORMULA_OWNERS = {"validation/trade_costs.py"}


def _production_and_tools_sources():
    skip = {".venv", ".git", "__pycache__", "scratchpad", "tests", "reports"}
    for py in REPO.rglob("*.py"):
        if skip & set(py.parts):
            continue
        yield py, py.read_text(encoding="utf-8", errors="ignore")


def test_the_legacy_cost_formula_survives_nowhere_but_its_owner():
    offenders = []
    for path, src in _production_and_tools_sources():
        rel = str(path.relative_to(REPO))
        if rel in FORMULA_OWNERS:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue  # prose about the old formula is documentation
            if LEGACY_FORMULA.search(line):
                offenders.append(f"{rel}:{i}")
    assert not offenders, (
        "the duplicated cost formula P1 removed has reappeared: " +
        ", ".join(offenders))


def test_every_cost_bearing_module_imports_the_canonical_layer():
    """Named explicitly rather than inferred: a module dropping the import is
    the exact regression that would reintroduce funding-free accounting."""
    required = [
        "backtest.py",
        "analyzer/outcomes.py",
        "analyzer/counterfactual.py",
        "tools/deep_backtest.py",
        "tools/deep_discovery.py",
        "tools/deep_diagnostics.py",
        "tools/swing_hypothesis_simulator.py",
        "tools/swing_hypothesis_walkforward.py",
        "tools/forecast_metrics.py",
        "tools/regime_gate_confirmation.py",
    ]
    missing = []
    for rel in required:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        modules = {n.module for n in ast.walk(tree)
                   if isinstance(n, ast.ImportFrom) and n.module}
        if not any(m.startswith("validation.") for m in modules):
            missing.append(rel)
    assert not missing, f"no canonical cost import in: {missing}"


def test_the_walks_refuse_to_run_without_an_explicit_funding_decision():
    """`funding` is keyword-only and has no default in every walk. A default
    would let a caller forget it, which is how the defect existed for so long."""
    import inspect

    from tools.deep_backtest import deep_walk
    from tools.swing_hypothesis_simulator import (random_direction_baseline,
                                                  walk_hypothesis)

    for fn in (deep_walk, walk_hypothesis, random_direction_baseline):
        param = inspect.signature(fn).parameters["funding"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert param.default is inspect.Parameter.empty, fn.__name__


def test_outcomes_leaves_net_null_when_funding_is_unavailable():
    """The live column fails closed: a 72h horizon crosses nine settlements, so
    an unknown bill must not be published as a cost-free net figure."""
    from datetime import datetime, timedelta, timezone

    from analyzer.outcomes import measure_outcome

    anchor = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = []
    for h in range(90):
        t = anchor + timedelta(hours=h)
        rows.append({"open_time": t, "close_time": t + timedelta(hours=1),
                     "high": 100.0 + h, "low": 99.0 + h, "close": 100.0 + h})
    df = pd.DataFrame(rows)
    fc = {"id": 1, "decision_time": anchor, "candidate_direction": "long",
          "executable_price_at_decision": 100.0, "signal_close_price": 100.0,
          "stop_loss": 50.0, "take_profit_levels": [1e9, 1e9]}

    # A series that stops well before the 72h horizon: coverage is genuinely
    # incomplete, so the net figure is unknown rather than free.
    short = series([0.0001], start=to_ms(anchor) + H8)
    out = measure_outcome(fc, df, anchor + timedelta(hours=100), 0.05, 0.03,
                          short)
    assert out["return_72h"] is not None
    assert out["net_after_costs"] is None
    assert out["funding_pct"] is None


def test_outcomes_charges_funding_when_the_horizon_is_covered():
    from datetime import datetime, timedelta, timezone

    from analyzer.outcomes import measure_outcome

    anchor = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = []
    for h in range(90):
        t = anchor + timedelta(hours=h)
        rows.append({"open_time": t, "close_time": t + timedelta(hours=1),
                     "high": 100.0 + h, "low": 99.0 + h, "close": 100.0 + h})
    df = pd.DataFrame(rows)
    fc = {"id": 1, "decision_time": anchor, "candidate_direction": "long",
          "executable_price_at_decision": 100.0, "signal_close_price": 100.0,
          "stop_loss": 50.0, "take_profit_levels": [1e9, 1e9]}

    covered = series([0.0001] * 20, start=to_ms(anchor) - H8)
    out = measure_outcome(fc, df, anchor + timedelta(hours=100), 0.05, 0.03,
                          covered)
    # Nine settlements over 72h at 0.01% each = 0.09pp, the magnitude P1
    # predicted for this horizon before the work started.
    assert out["funding_pct"] == pytest.approx(0.09, abs=1e-6)
    assert out["net_after_costs"] == pytest.approx(
        out["return_72h"] - 0.13 - 0.09, abs=1e-4)


# ---------------------------------------------------------------------------
# The shipped dataset, when it is present. Skipped rather than failed off-box:
# data/ is gitignored and regenerable.
# ---------------------------------------------------------------------------

def _dataset_paths():
    stem = dataset_stem("binance", "BTCUSDT")
    d = REPO / "data" / "funding"
    return d / f"{stem}.csv", d / f"{stem}.manifest.json"


@pytest.mark.skipif(not _dataset_paths()[0].exists(),
                    reason="funding cache not built in this environment")
def test_shipped_dataset_covers_the_kline_window_without_gaps():
    s = load_funding_series()
    manifest = json.loads(_dataset_paths()[1].read_text(encoding="utf-8"))
    assert s.gaps == (), "the shipped funding series has gaps"
    assert s.modal_interval_ms == H8
    assert s.source_sha256 == manifest["data_sha256"]

    klines = pd.read_csv(REPO / "data" / "klines" / "binance_BTCUSDT_4h.csv",
                         usecols=["close_time"])
    first = to_ms(pd.Timestamp(klines["close_time"].iloc[0]))
    last = to_ms(pd.Timestamp(klines["close_time"].iloc[-1]))
    assert s.first_ms <= first and s.last_ms >= last


@pytest.mark.skipif(not _dataset_paths()[0].exists(),
                    reason="funding cache not built in this environment")
def test_shipped_dataset_sha_mismatch_is_refused():
    """The provenance guard: a silently edited dataset must not load."""
    import shutil
    import tempfile

    csv, manifest = _dataset_paths()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(csv, Path(tmp) / csv.name)
        shutil.copy(manifest, Path(tmp) / manifest.name)
        with open(Path(tmp) / csv.name, "a", encoding="utf-8") as fh:
            fh.write(f"{T0},0.0001,,Regular\n")
        with pytest.raises(FundingDataUnavailable):
            load_funding_series(tmp)


@pytest.mark.skipif(not _dataset_paths()[0].exists(),
                    reason="funding cache not built in this environment")
def test_measured_funding_is_two_sided_on_real_data():
    """The assumption M00 froze (a constant positive rate) is not what the
    history looks like — which is why the retrofit charges realized rates."""
    s = load_funding_series()
    assert (s.rates < 0).mean() > 0.05
    assert (s.rates > 0).mean() > 0.5


def test_a_row_whose_funding_fetch_failed_is_retried_not_stranded():
    """P1 split resolution from cost availability: a row can resolve on price
    while its funding fetch fails. Without a retry it would keep `resolved` and
    a NULL net forever, so one transient outage would permanently cost that
    forecast its cost accounting.
    """
    from datetime import timedelta

    from database import _MemoryStore, utcnow

    mem = _MemoryStore()
    base = {"symbol": "BTCUSDT", "candidate_direction": "long",
            "analysis_status": "ENTER",
            "decision_time": utcnow() - timedelta(days=4)}
    mem.forecasts = [dict(base, id=1), dict(base, id=2), dict(base, id=3)]
    mem.outcomes = {
        # Resolved, funding fetch failed -> must come back.
        1: {"forecast_id": 1, "resolved": True, "return_72h": 4.2,
            "net_after_costs": None, "funding_pct": None},
        # Fully accounted -> must not come back.
        2: {"forecast_id": 2, "resolved": True, "return_72h": 4.2,
            "net_after_costs": 4.07, "funding_pct": 0.0},
        # Pre-P1 shape: stopped out before the 72h horizon elapsed, so the net
        # was never due. This is the row the retry must NOT mistake for a
        # funding failure — recomputing it would be the backfill P1 refuses.
        3: {"forecast_id": 3, "resolved": True, "return_72h": None,
            "net_after_costs": None},
    }
    pending = {f["id"] for f in mem.forecasts_pending_outcomes("BTCUSDT")}
    assert pending == {1}


def test_the_retry_cannot_starve_forecasts_that_were_never_measured():
    """Pending rows are served oldest-first under a LIMIT. A row whose funding is
    permanently out of the live fetch's reach must drop out of the retry set, or
    it refills that window every cycle and newer forecasts are never measured.
    """
    from datetime import timedelta

    from database import FUNDING_RETRY_WINDOW_DAYS, _MemoryStore, utcnow

    mem = _MemoryStore()
    base = {"symbol": "BTCUSDT", "candidate_direction": "long",
            "analysis_status": "ENTER"}
    stale = utcnow() - timedelta(days=FUNDING_RETRY_WINDOW_DAYS + 1)
    fresh = utcnow() - timedelta(days=1)
    mem.forecasts = [dict(base, id=1, decision_time=stale),
                     dict(base, id=2, decision_time=fresh),
                     dict(base, id=3, decision_time=fresh)]
    failed = {"resolved": True, "return_72h": 4.2, "net_after_costs": None}
    mem.outcomes = {1: dict(failed, forecast_id=1),
                    2: dict(failed, forecast_id=2)}
    pending = {f["id"] for f in mem.forecasts_pending_outcomes("BTCUSDT")}
    # 1 is past repair and abandoned; 2 is still repairable; 3 has never been
    # measured and must never be blocked by either.
    assert pending == {2, 3}


def test_abandoned_rows_are_counted_so_the_loss_is_not_silent():
    """The retry bound gives up on some rows. That is a deliberate trade-off, so
    it has to be reportable — an unmeasured cost that nobody can see is the kind
    of quiet loss P1 exists to remove."""
    from datetime import timedelta

    from database import FUNDING_RETRY_WINDOW_DAYS, _MemoryStore, utcnow

    mem = _MemoryStore()
    base = {"symbol": "BTCUSDT", "candidate_direction": "long",
            "analysis_status": "ENTER"}
    retry_after = utcnow() - timedelta(days=FUNDING_RETRY_WINDOW_DAYS)
    stale = retry_after - timedelta(days=1)
    fresh = utcnow() - timedelta(days=1)
    mem.forecasts = [dict(base, id=1, decision_time=stale),
                     dict(base, id=2, decision_time=fresh),
                     dict(base, id=3, decision_time=stale)]
    mem.outcomes = {
        1: {"forecast_id": 1, "resolved": True, "return_72h": 4.2,
            "net_after_costs": None},                      # abandoned
        2: {"forecast_id": 2, "resolved": True, "return_72h": 4.2,
            "net_after_costs": None},                      # still retryable
        3: {"forecast_id": 3, "resolved": True, "return_72h": 4.2,
            "net_after_costs": 4.07},                      # fully accounted
    }
    assert mem.count_abandoned_funding_outcomes("BTCUSDT", retry_after) == 1


def test_both_pending_implementations_use_one_cutoff_per_scan():
    """The SQL path computes the retry cutoff once per query. The in-memory path
    must too, or a forecast sitting on the boundary is classified differently by
    the two implementations depending on microsecond timing."""
    import inspect

    from database import _MemoryStore

    src = inspect.getsource(_MemoryStore.forecasts_pending_outcomes)
    body = src.split("for f in self.forecasts:", 1)[1]
    assert "utcnow() - timedelta" not in body, (
        "the retry cutoff is recomputed inside the row loop")
