"""C5.0 M00: cost model and break-even arithmetic.

The module's job is to be pessimistic correctly. Most of these tests pin the
places where an optimistic shortcut would be easy and invisible: funding
silently dropped, slippage not scaling, a delayed entry quietly discarded
because it overtook its exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.cost_sensitivity import (COST_MODEL_VERSION,
                                         FUNDING_INTERVAL_HOURS, CostModel,
                                         CostModelError, bootstrap_ci,
                                         break_even_edge_pct, break_even_r,
                                         break_even_table, net_returns,
                                         required_win_rate,
                                         sensitivity_surface)


@pytest.fixture
def cost():
    return CostModel(taker_fee_pct=0.05, slippage_pct=0.03,
                     funding_pct_per_8h=0.01,
                     reference_notional_usd=10_000.0)


class TestCostModel:
    def test_round_trip_is_fees_plus_slippage_plus_funding(self, cost):
        # 48h held = 6 funding intervals at 0.01% = 0.06%
        total = cost.round_trip_pct(holding_hours=48.0)
        assert total == pytest.approx(0.05 * 2 + 0.03 + 0.06)

    def test_funding_is_not_silently_dropped(self, cost):
        """The defect this module exists to fix: every backtest in the project
        applied fees and slippage and ignored funding entirely."""
        short = cost.round_trip_pct(holding_hours=0.0)
        long = cost.round_trip_pct(holding_hours=48.0)
        assert long > short, "holding longer must cost more"
        assert long - short == pytest.approx(0.06)

    def test_funding_dominates_at_long_holds(self, cost):
        """At the swing horizon funding is the same order as the fees, which
        is why ignoring it flattered every past net number."""
        parts = cost.breakdown(holding_hours=12 * 4.0)  # 12 bars of 4h
        assert parts["funding_pct"] == pytest.approx(0.06)
        assert parts["funding_pct"] > parts["slippage_pct"]

    def test_funding_is_continuous_not_stepped(self, cost):
        half = cost.funding_cost(FUNDING_INTERVAL_HOURS / 2)
        assert half == pytest.approx(cost.funding_pct_per_8h / 2)

    def test_slippage_scales_as_sqrt_of_size(self, cost):
        assert cost.slippage_at(10_000.0) == pytest.approx(0.03)
        assert cost.slippage_at(40_000.0) == pytest.approx(0.06)  # 2x for 4x
        assert cost.slippage_at(2_500.0) == pytest.approx(0.015)

    def test_breakdown_components_sum_to_total(self, cost):
        p = cost.breakdown(holding_hours=30.0, notional_usd=25_000.0)
        assert (p["fees_pct"] + p["slippage_pct"] + p["funding_pct"]
                == pytest.approx(p["total_pct"]))

    @pytest.mark.parametrize("field", ["taker_fee_pct", "slippage_pct",
                                       "funding_pct_per_8h"])
    def test_negative_costs_are_rejected(self, field):
        kwargs = {"taker_fee_pct": 0.05, "slippage_pct": 0.03, field: -1.0}
        with pytest.raises(CostModelError):
            CostModel(**kwargs)

    def test_negative_holding_is_rejected(self, cost):
        with pytest.raises(CostModelError):
            cost.funding_cost(-1.0)

    def test_hash_changes_with_any_component(self, cost):
        """A cost model that drifted between runs makes two backtests
        incomparable, so it is pinned like any other frozen artifact."""
        base = cost.content_sha256()
        from dataclasses import replace
        assert replace(cost, funding_pct_per_8h=0.02).content_sha256() != base
        assert replace(cost, slippage_pct=0.04).content_sha256() != base
        assert cost.content_sha256() == base, "hashing must be stable"

    def test_version_is_recorded(self, cost):
        assert cost.version == COST_MODEL_VERSION


class TestBreakEven:
    def test_required_win_rate_at_zero_cost_matches_the_payoff(self):
        assert required_win_rate(2.0, 0.0) == pytest.approx(1 / 3)
        assert required_win_rate(1.0, 0.0) == pytest.approx(0.5)
        assert required_win_rate(3.0, 0.0) == pytest.approx(0.25)

    def test_cost_raises_the_required_win_rate(self):
        assert required_win_rate(2.0, 0.2) > required_win_rate(2.0, 0.0)

    def test_cost_hits_winners_and_losers_both(self):
        """p = (1+c)/(W+1): cost does not merely subtract from the payoff,
        because it also deepens every loss."""
        assert required_win_rate(2.0, 0.1) == pytest.approx(1.1 / 3.0)

    def test_unreachable_payoff_reports_infinity(self):
        assert required_win_rate(0.5, 1.0) == float("inf")

    def test_break_even_r_is_cost_over_stop_distance(self, cost):
        r = break_even_r(cost, stop_distance_pct=1.0, holding_hours=48.0)
        assert r == pytest.approx(break_even_edge_pct(cost, holding_hours=48.0))
        tighter = break_even_r(cost, stop_distance_pct=0.5, holding_hours=48.0)
        assert tighter == pytest.approx(2 * r), "tighter stops surrender more R"

    def test_zero_stop_distance_is_rejected(self, cost):
        with pytest.raises(CostModelError):
            break_even_r(cost, stop_distance_pct=0.0, holding_hours=1.0)

    def test_table_needs_no_trade_history(self, cost):
        df = break_even_table(cost, bar_hours=4.0, holding_bars=[1, 12],
                              stop_distance_pct=1.5)
        assert list(df["holding_bars"]) == [1, 12]
        assert df["break_even_pct"].is_monotonic_increasing
        assert (df["required_win_rate"] > 1 / 3).all()


@pytest.fixture
def trades_and_prices():
    # a clean +10% ramp; every long is a winner before costs
    prices = list(np.linspace(100.0, 110.0, 21))
    trades = pd.DataFrame({"entry_idx": [0, 5, 10],
                           "exit_idx": [4, 9, 14],
                           "side": [1, 1, 1]})
    return trades, prices


class TestNetReturns:
    def test_costs_are_subtracted(self, trades_and_prices, cost):
        trades, prices = trades_and_prices
        net = net_returns(trades, prices, cost, delay_bars=0, bar_hours=4.0)
        assert net.size == 3
        assert (net > 0).all(), "a 10% ramp survives 0.1% of cost"

        gross_first = (prices[4] - prices[0]) / prices[0] * 100
        expected_cost = cost.round_trip_pct(holding_hours=4 * 4.0)
        assert net[0] == pytest.approx(gross_first - expected_cost)

    def test_delay_moves_the_entry(self, trades_and_prices, cost):
        trades, prices = trades_and_prices
        immediate = net_returns(trades, prices, cost, delay_bars=0, bar_hours=4.0)
        delayed = net_returns(trades, prices, cost, delay_bars=2, bar_hours=4.0)
        assert (delayed < immediate).all(), "entering late into a ramp costs"

    def test_shorts_are_signed_correctly(self, trades_and_prices, cost):
        trades, prices = trades_and_prices
        shorts = trades.assign(side=[-1, -1, -1])
        net = net_returns(shorts, prices, cost, delay_bars=0, bar_hours=4.0)
        assert (net < 0).all(), "shorting a ramp loses"

    def test_overtaken_trades_keep_full_cost_and_no_gain(self, cost):
        """A delay long enough to pass the exit must not drop the trade.
        Those are the fast fills, and they are disproportionately winners —
        discarding them would bias the result upward."""
        prices = [100.0, 101.0, 102.0, 103.0]
        trades = pd.DataFrame({"entry_idx": [0], "exit_idx": [1], "side": [1]})
        net = net_returns(trades, prices, cost, delay_bars=3, bar_hours=4.0)
        assert net.size == 1, "the trade is kept, not dropped"
        assert net[0] < 0
        assert net[0] == pytest.approx(-cost.round_trip_pct(holding_hours=0.0))

    def test_holding_cost_tracks_actual_bars_held(self, cost):
        prices = [100.0] * 30
        short_hold = pd.DataFrame({"entry_idx": [0], "exit_idx": [1], "side": [1]})
        long_hold = pd.DataFrame({"entry_idx": [0], "exit_idx": [24], "side": [1]})
        a = net_returns(short_hold, prices, cost, delay_bars=0, bar_hours=4.0)
        b = net_returns(long_hold, prices, cost, delay_bars=0, bar_hours=4.0)
        assert b[0] < a[0], "flat price, longer hold, more funding paid"

    def test_missing_columns_are_rejected(self, cost):
        bad = pd.DataFrame({"entry_idx": [0], "side": [1]})
        with pytest.raises(CostModelError, match="exit_idx"):
            net_returns(bad, [1.0, 2.0], cost, delay_bars=0, bar_hours=4.0)

    def test_bad_side_is_rejected(self, cost):
        bad = pd.DataFrame({"entry_idx": [0], "exit_idx": [1], "side": [0]})
        with pytest.raises(CostModelError, match="side must be"):
            net_returns(bad, [1.0, 2.0], cost, delay_bars=0, bar_hours=4.0)

    def test_negative_delay_is_rejected(self, trades_and_prices, cost):
        trades, prices = trades_and_prices
        with pytest.raises(CostModelError):
            net_returns(trades, prices, cost, delay_bars=-1, bar_hours=4.0)

    def test_there_is_no_cost_free_path(self, cost):
        """Structural: net_returns takes a CostModel positionally and has no
        flag to skip it. §5.6 forbids reporting a gross number."""
        import inspect
        sig = inspect.signature(net_returns)
        assert sig.parameters["cost"].default is inspect.Parameter.empty


class TestSurface:
    def test_surface_covers_the_grid(self, trades_and_prices, cost):
        trades, prices = trades_and_prices
        df = sensitivity_surface(trades, prices, cost,
                                 cost_multipliers=[1.0, 2.0, 5.0],
                                 delay_grid=[0, 1], bar_hours=4.0)
        assert len(df) == 6
        assert set(df["delay_bars"]) == {0, 1}

    def test_higher_costs_never_improve_expectancy(self, trades_and_prices, cost):
        trades, prices = trades_and_prices
        df = sensitivity_surface(trades, prices, cost,
                                 cost_multipliers=[1.0, 3.0, 10.0],
                                 delay_grid=[0], bar_hours=4.0)
        assert df.sort_values("cost_multiplier")["mean_net_pct"].is_monotonic_decreasing

    def test_multiplier_scales_all_three_components(self, cost):
        """A stressed market widens fees, slippage and funding together;
        scaling only one would understate the tail."""
        prices = [100.0] * 20
        trades = pd.DataFrame({"entry_idx": [0], "exit_idx": [12], "side": [1]})
        df = sensitivity_surface(trades, prices, cost, cost_multipliers=[1.0, 2.0],
                                 delay_grid=[0], bar_hours=4.0)
        base, doubled = df["mean_net_pct"].tolist()
        assert doubled == pytest.approx(2 * base)

    def test_negative_multiplier_is_rejected(self, trades_and_prices, cost):
        trades, prices = trades_and_prices
        with pytest.raises(CostModelError):
            sensitivity_surface(trades, prices, cost, cost_multipliers=[-1.0],
                                delay_grid=[0], bar_hours=4.0)


class TestBootstrap:
    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(0.5, 1.0, size=500)
        lo, hi = bootstrap_ci(sample, n_resamples=500)
        assert lo < sample.mean() < hi

    def test_deterministic_under_a_fixed_seed(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(size=100)
        assert (bootstrap_ci(sample, n_resamples=200, seed=7)
                == bootstrap_ci(sample, n_resamples=200, seed=7))

    def test_blocks_widen_the_interval_on_dependent_data(self):
        """C4.3e's finding, reapplied: an i.i.d. bootstrap on autocorrelated
        observations reports an interval that is too narrow."""
        rng = np.random.default_rng(2)
        walk = np.cumsum(rng.normal(size=600)) / 10.0
        iid_lo, iid_hi = bootstrap_ci(walk, block=1, n_resamples=800)
        blk_lo, blk_hi = bootstrap_ci(walk, block=40, n_resamples=800)
        assert (blk_hi - blk_lo) > (iid_hi - iid_lo)

    def test_empty_sample_is_rejected(self):
        with pytest.raises(CostModelError):
            bootstrap_ci([])

    def test_block_larger_than_sample_is_rejected(self):
        with pytest.raises(CostModelError, match="exceeds sample size"):
            bootstrap_ci([1.0, 2.0], block=5)
