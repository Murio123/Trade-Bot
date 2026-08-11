"""C5.0 M00: what gross edge must a strategy produce to break even?

This module is deliberately the first thing built and the cheapest. It does
not measure whether the engine has an edge — it measures the height of the bar
that any edge has to clear. If no plausible edge clears it, the remaining 14
research modules are cancelled and the project stays report-only.

Three things the project got wrong before, which this module fixes:

1. **Funding was never modelled.** `config.SLIPPAGE_PCT` and
   `config.TAKER_FEE_PCT` are applied by every backtest, but a perpetual
   position pays funding every 8h for as long as it is held. On a 12-bar 4h
   swing that is six funding payments, and at typical rates it is the same
   order of magnitude as the fees. Ignoring it flattered every net number the
   project has ever produced.

2. **Slippage was a constant.** It scales with order size; a flat 0.03% is
   only true at one notional. The square-root form here is the standard
   approximation and is stated as an approximation.

3. **Execution delay was assumed to be zero.** The engine decides on bar close
   and would fill at the next bar at best. `net_returns` re-anchors entry so
   the delay is priced rather than wished away.

Everything is versioned and hashed: the cost model is a frozen artifact under
ARCHITECTURE §6.2, because a backtest whose costs quietly changed between runs
is not comparable to itself.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

COST_MODEL_VERSION = "c50_m00_v1"

# Funding on BTC perps is not constant, but the long-run average is well
# documented as slightly positive (longs pay). A conservative default is used
# because the alternative — assuming zero — is the error this module exists to
# correct. Override with a measured series when one is available.
DEFAULT_FUNDING_PCT_PER_8H = 0.01

# Square-root impact: slippage grows as sqrt(notional / reference). The
# reference notional is the size at which `slippage_pct` was calibrated.
DEFAULT_REFERENCE_NOTIONAL_USD = 10_000.0

FUNDING_INTERVAL_HOURS = 8.0


class CostModelError(Exception):
    pass


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost of holding a perpetual position.

    All percentages are in percent (0.05 == 0.05%), matching the convention
    already used by config.TAKER_FEE_PCT.
    """
    taker_fee_pct: float
    slippage_pct: float
    funding_pct_per_8h: float = DEFAULT_FUNDING_PCT_PER_8H
    reference_notional_usd: float = DEFAULT_REFERENCE_NOTIONAL_USD
    always_pays_funding: bool = True
    version: str = COST_MODEL_VERSION

    def __post_init__(self) -> None:
        for name in ("taker_fee_pct", "slippage_pct", "funding_pct_per_8h"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise CostModelError(
                    f"{name} must be finite and non-negative, got {value!r}")
        if self.reference_notional_usd <= 0:
            raise CostModelError("reference_notional_usd must be positive")

    @classmethod
    def from_config(cls, **overrides: Any) -> "CostModel":
        """Defaults from the project config, so research and backtests agree.

        config is imported lazily: this package must stay importable without
        the bot's environment.
        """
        import config
        base = cls(taker_fee_pct=config.TAKER_FEE_PCT,
                   slippage_pct=config.SLIPPAGE_PCT)
        return replace(base, **overrides) if overrides else base

    def content_sha256(self) -> str:
        """Pin for the trial ledger: a cost model that changed silently
        between runs makes two backtests incomparable."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def slippage_at(self, notional_usd: float) -> float:
        """Round-trip slippage at a given size, in percent."""
        if notional_usd <= 0:
            raise CostModelError(f"notional must be positive, got {notional_usd!r}")
        scale = float(np.sqrt(notional_usd / self.reference_notional_usd))
        return self.slippage_pct * scale

    def funding_cost(self, holding_hours: float) -> float:
        """Funding paid over the holding period, in percent.

        Charged continuously rather than at discrete 8h marks: which side of a
        funding stamp a position lands on is not predictable at decision time,
        so the expectation is the honest quantity.
        """
        if holding_hours < 0:
            raise CostModelError("holding_hours cannot be negative")
        if not self.always_pays_funding:
            return 0.0
        return self.funding_pct_per_8h * (holding_hours / FUNDING_INTERVAL_HOURS)

    def round_trip_pct(self, *, holding_hours: float,
                       notional_usd: float | None = None) -> float:
        """Total cost of one complete trade, in percent of notional."""
        notional = (self.reference_notional_usd if notional_usd is None
                    else notional_usd)
        return (2.0 * self.taker_fee_pct
                + self.slippage_at(notional)
                + self.funding_cost(holding_hours))

    def breakdown(self, *, holding_hours: float,
                  notional_usd: float | None = None) -> dict[str, float]:
        """The same number, itemised — which component dominates is the
        actionable part, and it is not always the fees."""
        notional = (self.reference_notional_usd if notional_usd is None
                    else notional_usd)
        return {
            "fees_pct": 2.0 * self.taker_fee_pct,
            "slippage_pct": self.slippage_at(notional),
            "funding_pct": self.funding_cost(holding_hours),
            "total_pct": self.round_trip_pct(holding_hours=holding_hours,
                                             notional_usd=notional),
        }


def break_even_edge_pct(cost: CostModel, *, holding_hours: float,
                        notional_usd: float | None = None) -> float:
    """Gross move, in percent, that a trade must capture to break even."""
    return cost.round_trip_pct(holding_hours=holding_hours,
                               notional_usd=notional_usd)


def break_even_r(cost: CostModel, *, stop_distance_pct: float,
                 holding_hours: float,
                 notional_usd: float | None = None) -> float:
    """Cost expressed in R — the share of one stop-distance eaten per trade.

    This is the number that decides whether a strategy is viable. A setup
    risking 1% with 0.2% of cost surrenders 20% of every R before it starts.
    """
    if stop_distance_pct <= 0:
        raise CostModelError("stop_distance_pct must be positive")
    return break_even_edge_pct(cost, holding_hours=holding_hours,
                               notional_usd=notional_usd) / stop_distance_pct


def required_win_rate(reward_r: float, cost_r: float) -> float:
    """Win rate needed for zero expectancy at a given payoff and cost.

    Solving  p(W - c) - (1 - p)(1 + c) = 0  gives  p = (1 + c) / (W + 1).
    Cost enters both the winners (less reward) and the losers (bigger loss),
    which is why it does not simply subtract from the payoff ratio.
    """
    if reward_r <= 0:
        raise CostModelError("reward_r must be positive")
    if cost_r < 0:
        raise CostModelError("cost_r cannot be negative")
    p = (1.0 + cost_r) / (reward_r + 1.0)
    if p >= 1.0:
        return float("inf")  # unreachable: no win rate makes this profitable
    return float(p)


def net_returns(trades: pd.DataFrame, prices: Sequence[float],
                cost: CostModel, *, delay_bars: int, bar_hours: float,
                notional_usd: float | None = None) -> np.ndarray:
    """Realized net return per trade, in percent, with entry delayed.

    `trades` needs entry_idx, exit_idx and side (+1 long, -1 short);
    `prices` is the close series those indices point into. Entry is re-anchored
    at `entry_idx + delay_bars`, which is what actually happens: the engine
    decides on a closed bar and cannot fill inside it.

    There is no cost-free path through this function, by design. Once M00
    exists, a gross number reported without costs is a specification
    violation (ARCHITECTURE §5.6).
    """
    required = {"entry_idx", "exit_idx", "side"}
    missing = required - set(trades.columns)
    if missing:
        raise CostModelError(f"trades is missing columns: {sorted(missing)}")
    if delay_bars < 0:
        raise CostModelError("delay_bars cannot be negative")
    if bar_hours <= 0:
        raise CostModelError("bar_hours must be positive")

    px = np.asarray(prices, dtype=float)
    if px.ndim != 1 or px.size == 0:
        raise CostModelError("prices must be a non-empty 1-D series")

    entry_idx = trades["entry_idx"].to_numpy(dtype=np.int64) + delay_bars
    exit_idx = trades["exit_idx"].to_numpy(dtype=np.int64)
    side = trades["side"].to_numpy(dtype=float)

    if not np.isin(side, (-1.0, 1.0)).all():
        raise CostModelError("side must be +1 (long) or -1 (short)")

    # A delayed entry can overtake its own exit. Dropping such trades would
    # bias the result — they are the fast ones, and they are usually the
    # winners. They are held to a zero-length round trip instead: full cost,
    # no move captured.
    overtaken = entry_idx >= exit_idx
    entry_idx = np.clip(entry_idx, 0, px.size - 1)
    exit_idx = np.clip(exit_idx, 0, px.size - 1)

    entry_px = px[entry_idx]
    exit_px = px[exit_idx]
    if np.any(entry_px <= 0):
        raise CostModelError("non-positive entry price in the series")

    gross = side * (exit_px - entry_px) / entry_px * 100.0
    gross = np.where(overtaken, 0.0, gross)

    holding_hours = np.maximum(exit_idx - entry_idx, 0) * bar_hours
    costs = np.array([cost.round_trip_pct(holding_hours=float(h),
                                          notional_usd=notional_usd)
                      for h in holding_hours])
    return gross - costs


def sensitivity_surface(trades: pd.DataFrame, prices: Sequence[float],
                        cost: CostModel, *,
                        cost_multipliers: Iterable[float],
                        delay_grid: Iterable[int],
                        bar_hours: float,
                        notional_usd: float | None = None) -> pd.DataFrame:
    """Net expectancy across a grid of cost levels and execution delays.

    A curve, not a pass/fail: where the strategy dies is more informative than
    whether it is alive at one assumed cost. The multiplier scales fees,
    slippage and funding together, which is the realistic joint move — a
    stressed market widens all three at once.
    """
    rows: list[dict[str, Any]] = []
    for mult in cost_multipliers:
        if mult < 0:
            raise CostModelError("cost multiplier cannot be negative")
        scaled = replace(cost,
                         taker_fee_pct=cost.taker_fee_pct * mult,
                         slippage_pct=cost.slippage_pct * mult,
                         funding_pct_per_8h=cost.funding_pct_per_8h * mult)
        for delay in delay_grid:
            net = net_returns(trades, prices, scaled, delay_bars=delay,
                              bar_hours=bar_hours, notional_usd=notional_usd)
            rows.append({
                "cost_multiplier": float(mult),
                "delay_bars": int(delay),
                "n_trades": int(net.size),
                "mean_net_pct": float(net.mean()) if net.size else float("nan"),
                "median_net_pct": (float(np.median(net)) if net.size
                                   else float("nan")),
                "share_positive": (float((net > 0).mean()) if net.size
                                   else float("nan")),
                "total_net_pct": float(net.sum()),
            })
    return pd.DataFrame(rows)


def bootstrap_ci(net: Sequence[float], *, block: int = 1, n_resamples: int = 5000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Moving-block bootstrap CI on mean net return.

    `block` defaults to 1 because trades, unlike bars, are often already
    non-overlapping. When they do overlap — concurrent positions — pass a
    block at least as long as the overlap, for the reason C4.3e established:
    an i.i.d. bootstrap on dependent observations understates the interval.
    """
    arr = np.asarray(net, dtype=float)
    n = arr.size
    if n == 0:
        raise CostModelError("cannot bootstrap an empty sample")
    if block < 1:
        raise CostModelError("block must be >= 1")
    if block > n:
        raise CostModelError(f"block {block} exceeds sample size {n}")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block + 1
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([arr[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def break_even_table(cost: CostModel, *, bar_hours: float,
                     holding_bars: Iterable[int],
                     stop_distance_pct: float,
                     reward_r: float = 2.0,
                     notional_usd: float | None = None) -> pd.DataFrame:
    """The headline output: the bar, at several holding periods.

    Needs no trade history at all, which is why it can run today.
    """
    rows: list[dict[str, Any]] = []
    for bars in holding_bars:
        hours = bars * bar_hours
        parts = cost.breakdown(holding_hours=hours, notional_usd=notional_usd)
        cost_r = break_even_r(cost, stop_distance_pct=stop_distance_pct,
                              holding_hours=hours, notional_usd=notional_usd)
        rows.append({
            "holding_bars": int(bars),
            "holding_hours": float(hours),
            "fees_pct": parts["fees_pct"],
            "slippage_pct": parts["slippage_pct"],
            "funding_pct": parts["funding_pct"],
            "break_even_pct": parts["total_pct"],
            "cost_in_r": cost_r,
            "required_win_rate": required_win_rate(reward_r, cost_r),
        })
    return pd.DataFrame(rows)


def format_break_even(df: pd.DataFrame, cost: CostModel, *,
                      stop_distance_pct: float, reward_r: float) -> str:
    lines = [
        f"C5.0 M00 — break-even cost ({cost.version}, "
        f"sha {cost.content_sha256()[:12]})",
        f"  taker {cost.taker_fee_pct}%/side, slippage {cost.slippage_pct}% "
        f"@ ${cost.reference_notional_usd:,.0f}, "
        f"funding {cost.funding_pct_per_8h}%/8h",
        f"  stop distance {stop_distance_pct}%, payoff {reward_r}R",
        "",
        f"  {'bars':>5} {'hours':>6} {'fees':>7} {'slip':>7} {'fund':>7} "
        f"{'total':>8} {'cost/R':>8} {'need WR':>8}",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"  {int(r['holding_bars']):>5} {r['holding_hours']:>6.0f} "
            f"{r['fees_pct']:>7.3f} {r['slippage_pct']:>7.3f} "
            f"{r['funding_pct']:>7.3f} {r['break_even_pct']:>8.3f} "
            f"{r['cost_in_r']:>8.3f} {r['required_win_rate'] * 100:>7.1f}%")
    lines.append("")
    lines.append("A win rate at 50% payoff-neutral is 33.3% at zero cost; the "
                 "gap above is what execution takes before any edge exists.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="C5.0 M00: break-even cost for the swing profile")
    p.add_argument("--bar-hours", type=float, default=4.0)
    p.add_argument("--holding-bars", type=int, nargs="+",
                   default=[1, 3, 6, 12, 24])
    p.add_argument("--stop-distance-pct", type=float, default=1.5,
                   help="stop distance as %% of price; swing ATR stops on BTC "
                        "4h typically land near this")
    p.add_argument("--reward-r", type=float, default=2.0)
    p.add_argument("--notional-usd", type=float, default=None)
    p.add_argument("--funding-pct-per-8h", type=float,
                   default=DEFAULT_FUNDING_PCT_PER_8H)
    args = p.parse_args(argv)

    cost = CostModel.from_config(funding_pct_per_8h=args.funding_pct_per_8h)
    table = break_even_table(cost, bar_hours=args.bar_hours,
                             holding_bars=args.holding_bars,
                             stop_distance_pct=args.stop_distance_pct,
                             reward_r=args.reward_r,
                             notional_usd=args.notional_usd)
    print(format_break_even(table, cost,
                            stop_distance_pct=args.stop_distance_pct,
                            reward_r=args.reward_r))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
