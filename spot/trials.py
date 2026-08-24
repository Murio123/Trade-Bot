"""S3: the six trial declarations, written before any outcome is read.

Every rule that ranks assets could influence selection, so every one of them
consumes a look and is declared here first (`TRIAL_REGISTRY_SPEC.md` §2, §4).
The declaration carries the exact definition, so **changing a definition
changes the `trial_id` and produces a new trial** — that is the whole
mechanism, and a test asserts it rather than trusting anyone to remember.

`B0`-`B3` are deliberately absent. B0 is a null distribution, B1 and B2 are
reference portfolios with no cross-sectional choice in them, and B3 is
unavailable for want of point-in-time market cap. None of them selects among
assets on evidence. That reading is recorded in `S3_SPEC.md` §11 so a reader
can disagree with it explicitly instead of discovering it by absence.

The family is **`spot_2x_discovery` / `clean_2x_180d`**, and it starts at zero.
The futures family's `n_trials = 100` is historical governance context for a
different research question; the registry's family key keeps them apart, and
nothing here transfers or mixes them.
"""
from __future__ import annotations

from typing import Any

from spot.features import FEATURES
from validation.trial_registry import (ORIGIN_DECLARED, STATUS_DECLARED,
                                       FamilyScope, TrialIdentity, TrialRecord,
                                       TrialRegistry)

RESEARCH_OBJECTIVE = "spot_2x_discovery"
RESEARCH_STAGE = "S3"
TARGET_FAMILY = "clean_2x_180d"
HORIZON_DAYS = 180
SPEC = "reports/s3/S3_SPEC.md"
DECLARED_AT = "2026-08-24"

SCOPE = FamilyScope(research_objective=RESEARCH_OBJECTIVE,
                    target_family=TARGET_FAMILY)

# The go/no-go rule the trials will be judged by, pinned into identity so that
# judging them by a different rule later is a different trial.
CRITERION = {
    "rule": "S3_SPEC.md §10",
    "keep_requires": ["lift>1", "beats_random_p95", "better_excess_vs_btc",
                      "stable_in_60pct_blocks", "positive_in_2_of_3_regimes",
                      "bootstrap90_excludes_zero", "no_pit_violation"],
    "selection": "top quintile, K=ceil(0.20*N), N>=25",
}


def _record(feature) -> TrialRecord:
    identity = TrialIdentity.build(
        research_objective=RESEARCH_OBJECTIVE,
        research_stage=RESEARCH_STAGE,
        hypothesis=(f"{feature.fid}: rank the eligible universe by "
                    f"{feature.name}, descending"),
        profile=None,
        symbol=None,
        dataset_contract="binance_spot_usdt_daily_s1",
        target=TARGET_FAMILY,
        horizon_bars=HORIZON_DAYS,
        features=[feature.name],
        parameters={"definition": feature.definition,
                    "min_bars": feature.min_bars,
                    "descending": feature.descending},
        spec=SPEC,
        criterion=CRITERION,
    )
    return TrialRecord(
        identity=identity,
        created_at=DECLARED_AT,
        origin=ORIGIN_DECLARED,
        status=STATUS_DECLARED,
        target_family=TARGET_FAMILY,
        notes=f"S3 {feature.family}: {feature.definition}",
    )


ALL_TRIALS: tuple[TrialRecord, ...] = tuple(_record(f) for f in FEATURES)
BY_FEATURE: dict[str, TrialRecord] = {
    f.fid: rec for f, rec in zip(FEATURES, ALL_TRIALS)
}


def declare_all(registry: TrialRegistry) -> dict[str, str]:
    """Declare every S3 trial. Idempotent: a rerun adds nothing.

    Declaration happens **before** the evaluation reads a single outcome, which
    is the rule that stops a trial count from being assembled favourably once
    the results are known.
    """
    out: dict[str, str] = {}
    for fid, record in BY_FEATURE.items():
        registry.declare(record)
        out[fid] = record.trial_id
    return out


def counts(registry: TrialRegistry) -> dict[str, Any]:
    return registry.n_trials(SCOPE).as_dict()
