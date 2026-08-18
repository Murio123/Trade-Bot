"""G1.1 — the only path from a project research result to a published DSR.

M01 is a mathematical primitive and stays one: `deflated_sharpe(returns,
n_trials=...)` still takes the count as an argument, because that is the
correct shape for a formula. The problem was never the formula. It was that a
published number could carry a hand-counted `n_trials` with no record of where
that count came from, and the direction of the error is one-sided — a smaller
count always flatters the result.

This wrapper closes that by construction rather than by convention: it has no
`n_trials` parameter. The count comes from the registry, the family it was
counted over is recorded, and the registry's content hash is stored alongside
the result so the number can be recomputed later against the same registry
state and shown to be the same number.

Layering, per TRIAL_REGISTRY_SPEC.md §1: this module imports both
`deflated_sharpe` and `trial_registry`, and neither imports it. There is no
cycle, and M01 remains usable — and unit-testable — on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from validation.deflated_sharpe import (DSR_VERSION, DeflatedSharpeResult,
                                        deflated_sharpe)
from validation.trial_registry import (ORIGIN_SYNTHETIC, REGISTRY_VERSION,
                                       FamilyScope, TrialCount, TrialRegistry,
                                       TrialRegistryError,
                                       UndeclaredTrialError)

RESEARCH_DSR_VERSION = "g11_research_dsr_v1"


class ProvenanceError(TrialRegistryError):
    """The result cannot state where its trial count came from."""


@dataclass(frozen=True)
class DSRProvenance:
    """Where `n_trials` came from, in enough detail to re-derive it."""
    research_dsr_version: str
    dsr_version: str
    registry_version: str
    registry_path: str
    registry_sha256: str
    n_declarations: int
    trial_id: str
    scope: dict[str, Any]
    confirmed: int
    uncertain: int
    conservative: int
    n_records: int
    trial_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items()}
        payload["trial_ids"] = list(self.trial_ids)
        return payload


@dataclass(frozen=True)
class ResearchDSRResult:
    """A deflated Sharpe that knows what it was deflated against."""
    dsr: DeflatedSharpeResult
    provenance: DSRProvenance

    @property
    def significant(self) -> bool:
        return self.dsr.significant

    def as_dict(self) -> dict[str, Any]:
        return {"dsr": self.dsr.as_dict(),
                "provenance": self.provenance.as_dict()}


def research_trial_count(registry: TrialRegistry, scope: FamilyScope,
                         trial_id: str) -> TrialCount:
    """The family count for `trial_id`, refusing a family it is not part of.

    Deflating against a scope that excludes the very trial being deflated
    would be the easiest possible way to undercount: pick a narrow family, get
    a small number, quote it. So the trial must be inside its own family.
    """
    declarations = registry.declarations()
    record = declarations.get(trial_id)
    if record is None:
        raise UndeclaredTrialError(
            f"trial {trial_id} is not declared in {registry.path}; a result "
            f"cannot be deflated before its own trial exists "
            f"(TRIAL_REGISTRY_SPEC.md §7.2)")
    if record.origin == ORIGIN_SYNTHETIC:
        raise ProvenanceError(
            f"trial {trial_id} is synthetic and is excluded from every family "
            f"count (§6.3); synthetic analysis uses the mathematical layer "
            f"`validation.deflated_sharpe.deflated_sharpe` directly")
    count = registry.n_trials(scope)
    if trial_id not in count.trial_ids:
        raise ProvenanceError(
            f"trial {trial_id} is not a member of the family it is being "
            f"deflated against ({scope.as_dict()}); a result may not be "
            f"measured against a family that excludes it")
    return count


def research_deflated_sharpe(returns: Sequence[float] | np.ndarray, *,
                             registry: TrialRegistry, scope: FamilyScope,
                             trial_id: str,
                             trial_sharpes: Sequence[float] | np.ndarray | None
                             = None) -> ResearchDSRResult:
    """Deflate a project research result against its registered trial family.

    There is deliberately no `n_trials` argument. The conservative count of
    TRIAL_REGISTRY_SPEC.md §8.3 is used — `confirmed + high(uncertain)` — and
    the decomposition travels with the result so the choice is visible rather
    than buried.
    """
    count = research_trial_count(registry, scope, trial_id)
    snapshot = registry.snapshot()
    result = deflated_sharpe(returns, n_trials=count.n_trials,
                             trial_sharpes=trial_sharpes)
    provenance = DSRProvenance(
        research_dsr_version=RESEARCH_DSR_VERSION,
        dsr_version=DSR_VERSION,
        registry_version=REGISTRY_VERSION,
        registry_path=registry.path,
        registry_sha256=snapshot["content_sha256"],
        n_declarations=snapshot["n_declarations"],
        trial_id=trial_id,
        scope=count.scope,
        confirmed=count.confirmed,
        uncertain=count.uncertain,
        conservative=count.conservative,
        n_records=count.n_records,
        trial_ids=count.trial_ids,
    )
    return ResearchDSRResult(dsr=result, provenance=provenance)
