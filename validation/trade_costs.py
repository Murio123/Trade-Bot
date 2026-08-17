"""P1: the one place that turns a gross R into a net R.

Before this module, seven call sites computed

    cost_pct = (2 * TAKER_FEE_PCT + SLIPPAGE_PCT) / 100
    cost_r   = cost_pct * price / risk_distance

independently, two more kept private copies of the two constants, and none of
them paid funding. Patching nine sites in place would have recreated the defect
the next time a cost component was added, so the sites now call `trade_costs`
and the economics live here.

The accounting identity, and the only one this project has:

    net_r = gross_r - fee_r - slippage_r - funding_r

with every term normalised the same way — cash divided by the cash risked, one
stop distance. For a position of `q` contracts entered at `P` with a stop
`D` away, one R is `q * D`, and a cost of `c` percent of notional is
`c/100 * q * P` in cash, hence `c/100 * P / D` in R. Fees, slippage and funding
all share that form because all three are charged on notional.

**Funding is not part of the admission gate.** `H1_MAX_COST_R` and
`H2_MAX_COST_R` reject a candidate before it is resolved, using
`transaction_cost_r`. Folding funding into that number would change *which*
historical trades were taken, and P1 is forbidden from changing the trade
population — it may only change what those trades cost. Funding is therefore a
post-resolution term, which is also the honest ordering: the holding interval
is not known at entry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from validation.funding import (FundingDataUnavailable, FundingError,
                                FundingSeries, to_ms)

TRADE_COSTS_VERSION = "p1_trade_costs_v1"

MS_PER_HOUR = 3_600_000.0


class _FundingNotModelled:
    """Explicit opt-out for runs whose funding is genuinely unknowable.

    Synthetic frames in the test suite have invented timestamps, so there is no
    real funding bill to charge. Passing zero silently is the bias P1 removes,
    so the opt-out is a sentinel that must be named at the call site and that
    propagates as `funding_modelled=False` into every record and report built
    from it. The tools' CLI entry points refuse it; only in-process callers
    (tests, parity harnesses) can reach it.
    """
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "FUNDING_NOT_MODELLED"

    def __bool__(self) -> bool:
        return False


FUNDING_NOT_MODELLED = _FundingNotModelled()


def _fees_from_config() -> tuple[float, float]:
    """config is imported lazily: `validation` must stay importable without the
    bot's environment (same rule as CostModel.from_config)."""
    import config
    return float(config.TAKER_FEE_PCT), float(config.SLIPPAGE_PCT)


def transaction_cost_pct(taker_fee_pct: float, slippage_pct: float) -> float:
    """Round-trip transaction cost in percent of notional: two fills plus
    slippage. Funding is excluded by design — see the module docstring."""
    return 2.0 * taker_fee_pct + slippage_pct


def cost_pct_to_r(cost_pct: float, entry_price: float,
                  risk_distance: float) -> float:
    """Percent of notional -> share of one stop distance.

    A zero or missing stop distance yields 0.0 rather than an exception: that
    is the behaviour every legacy site had (`if risk_dist else 0.0`), and
    changing it here would change results for reasons unrelated to funding.
    """
    if not risk_distance:
        return 0.0
    return cost_pct / 100.0 * entry_price / risk_distance


def transaction_cost_r(entry_price: float, risk_distance: float, *,
                       taker_fee_pct: float | None = None,
                       slippage_pct: float | None = None) -> float:
    """Pre-trade transaction cost in R — the admission-gate quantity.

    Bit-for-bit the legacy `(2 * taker + slip) / 100 * price / risk`, so
    migrating a gate site cannot move it.
    """
    if taker_fee_pct is None or slippage_pct is None:
        cfg_taker, cfg_slip = _fees_from_config()
        taker_fee_pct = cfg_taker if taker_fee_pct is None else taker_fee_pct
        slippage_pct = cfg_slip if slippage_pct is None else slippage_pct
    return cost_pct_to_r(transaction_cost_pct(taker_fee_pct, slippage_pct),
                         entry_price, risk_distance)


@dataclass(frozen=True)
class TradeCosts:
    """The economic decomposition of one round trip, in percent and in R.

    Unrounded on purpose: rounding belongs at the reporting boundary, and a
    component rounded here would break the identity checked by
    `assert_identity`.
    """
    fee_pct: float
    slippage_pct: float
    funding_pct: float
    total_pct: float

    fee_r: float
    slippage_r: float
    funding_r: float
    transaction_r: float
    total_r: float

    gross_r: float | None
    net_r: float | None

    funding_settlements: int
    holding_hours: float | None
    funding_modelled: bool = True
    version: str = TRADE_COSTS_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_identity(self, tol: float = 1e-9) -> None:
        """The invariant P1 exists to establish. Cheap enough to assert in
        tests and in any report that publishes a net number."""
        if abs(self.transaction_r - (self.fee_r + self.slippage_r)) > tol:
            raise FundingError("transaction_r != fee_r + slippage_r")
        if abs(self.total_r - (self.transaction_r + self.funding_r)) > tol:
            raise FundingError("total_r != transaction_r + funding_r")
        if abs(self.total_pct - (self.fee_pct + self.slippage_pct
                                 + self.funding_pct)) > tol:
            raise FundingError("total_pct != fee_pct + slippage_pct + funding_pct")
        if self.gross_r is not None and self.net_r is not None:
            if abs(self.net_r - (self.gross_r - self.total_r)) > tol:
                raise FundingError("net_r != gross_r - total_r")


def trade_costs(*, entry_price: float, risk_distance: float, side: Any,
                entry_time: Any, exit_time: Any,
                funding: FundingSeries | _FundingNotModelled | None,
                gross_r: float | None = None,
                taker_fee_pct: float | None = None,
                slippage_pct: float | None = None) -> TradeCosts:
    """Full cost decomposition for one closed position.

    `funding=None` is accepted only for a zero-length hold, where no settlement
    can have been crossed. Any other missing-data case raises
    `FundingDataUnavailable`: treating an unknown funding bill as zero is the
    exact bias P1 removes, so it is not available as a fallback. The one way to
    get a zero on a real hold is `FUNDING_NOT_MODELLED`, which says so in the
    result.
    """
    if taker_fee_pct is None or slippage_pct is None:
        cfg_taker, cfg_slip = _fees_from_config()
        taker_fee_pct = cfg_taker if taker_fee_pct is None else taker_fee_pct
        slippage_pct = cfg_slip if slippage_pct is None else slippage_pct

    entry_ms, exit_ms = to_ms(entry_time), to_ms(exit_time)
    if exit_ms < entry_ms:
        raise FundingError(f"exit {exit_ms} precedes entry {entry_ms}")
    holding_hours = (exit_ms - entry_ms) / MS_PER_HOUR

    funding_modelled = True
    if isinstance(funding, _FundingNotModelled):
        funding_pct, settlements, funding_modelled = 0.0, 0, False
    elif funding is None:
        if exit_ms != entry_ms:
            raise FundingDataUnavailable(
                "no funding series supplied for a position held from "
                f"{entry_ms} to {exit_ms}; funding cannot be assumed zero")
        funding_pct, settlements = 0.0, 0
    else:
        rates = funding.settlement_rates(entry_ms, exit_ms)
        settlements = int(rates.size)
        funding_pct = funding.realized_funding_pct(entry_ms, exit_ms, side)

    fee_pct = 2.0 * taker_fee_pct
    total_pct = fee_pct + slippage_pct + funding_pct

    fee_r = cost_pct_to_r(fee_pct, entry_price, risk_distance)
    slippage_r = cost_pct_to_r(slippage_pct, entry_price, risk_distance)
    funding_r = cost_pct_to_r(funding_pct, entry_price, risk_distance)
    transaction_r = fee_r + slippage_r
    total_r = transaction_r + funding_r

    return TradeCosts(
        fee_pct=fee_pct, slippage_pct=float(slippage_pct),
        funding_pct=funding_pct, total_pct=total_pct,
        fee_r=fee_r, slippage_r=slippage_r, funding_r=funding_r,
        transaction_r=transaction_r, total_r=total_r,
        gross_r=gross_r,
        net_r=None if gross_r is None else gross_r - total_r,
        funding_settlements=settlements, holding_hours=holding_hours,
        funding_modelled=funding_modelled,
    )


def funding_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the funding term over a set of resolved trades.

    Lives here rather than in each tool because P1's whole point is that this
    arithmetic exists once. Each trade must carry `funding_r`, `cost_r` and
    `direction`; `funding_settlements` and `funding_modelled` are read when
    present.

    `funding_share_of_costs` is computed on ABSOLUTE funding against absolute
    total cost: with credits and debits both present, a signed ratio would
    report a small share for a set where funding dominated in both directions.
    """
    import numpy as np

    usable = [t for t in trades if t.get("funding_r") is not None]
    if not usable:
        return {"n": 0, "total_funding_r": None, "mean_funding_r": None,
                "median_funding_r": None, "funding_share_of_costs": None,
                "share_crossing_settlement": None, "largest_debit_r": None,
                "largest_credit_r": None, "by_direction": {},
                "funding_modelled": (all(t.get("funding_modelled") is True
                                         for t in trades) if trades else None)}

    fr = np.array([float(t["funding_r"]) for t in usable], dtype=float)
    cost = np.array([float(t.get("cost_r") or 0.0) for t in usable], dtype=float)
    settled = [t.get("funding_settlements") for t in usable]
    denom = float(np.abs(cost + fr).sum())

    by_direction: dict[str, Any] = {}
    for side in ("long", "short"):
        sel = fr[[t.get("direction") == side for t in usable]]
        by_direction[side] = {
            "n": int(sel.size),
            "total_funding_r": float(sel.sum()) if sel.size else None,
            "mean_funding_r": float(sel.mean()) if sel.size else None,
        }

    return {
        "n": int(fr.size),
        "total_funding_r": float(fr.sum()),
        "mean_funding_r": float(fr.mean()),
        "median_funding_r": float(np.median(fr)),
        "funding_share_of_costs": (float(np.abs(fr).sum() / denom)
                                   if denom else None),
        "share_crossing_settlement": (
            float(np.mean([bool(s) for s in settled if s is not None]))
            if any(s is not None for s in settled) else None),
        "largest_debit_r": float(fr.max()),
        "largest_credit_r": float(fr.min()),
        "by_direction": by_direction,
        "funding_modelled": all(t.get("funding_modelled") is True
                                for t in usable),
    }


def net_pct_after_costs(*, gross_pct: float, side: Any, entry_time: Any,
                        exit_time: Any, funding: FundingSeries | None,
                        taker_fee_pct: float | None = None,
                        slippage_pct: float | None = None) -> float:
    """Net percentage return, for paths that measure in percent rather than R.

    `analyzer.outcomes` reports a directional 72h return, not an R multiple, so
    it needs the same economics without the stop-distance normalisation.
    """
    costs = trade_costs(entry_price=1.0, risk_distance=0.0, side=side,
                        entry_time=entry_time, exit_time=exit_time,
                        funding=funding, taker_fee_pct=taker_fee_pct,
                        slippage_pct=slippage_pct)
    return gross_pct - costs.total_pct
