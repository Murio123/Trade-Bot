"""G1.1 — the append-only registry of research trials.

M01 deflates a Sharpe by the number of times we looked. It takes that number
as an argument and cannot check it, which ARCHITECTURE.md §6.1 names as the
single largest failure mode of the whole correction: an understated trial
count makes every deflated result more favourable, and no amount of care
inside M01 can detect it. This module is the structural answer.

Two records with two lifetimes, and the split is the entire design:

- a **declaration** is written before the result exists, is immutable, and is
  the only thing that moves the trial count;
- an **execution** is written per run, references a declaration, and never
  moves the count.

So re-running a frozen specification is free, and choosing something new is
not. That is the distinction ARCHITECTURE.md §6.1 asks for, made structural
rather than left to the discipline of whoever runs the next experiment.

Storage follows `volatility/ledger.py` rather than inventing a second
philosophy: JSON Lines, exclusive lock held across the whole check-then-append,
fsync per write, duplicate detection on read, and one validation entry point
that every public reader goes through. That last rule is not stylistic — four
audit rounds on the C4.4 ledger found the same defect shape each time, a check
added to some read paths and not others.

Policy is frozen in `reports/c50/TRIAL_REGISTRY_SPEC.md` and this module
implements it without reinterpretation. Where the spec says an ambiguity
resolves upward, the code resolves it upward.

Layering: this module must not import `validation.deflated_sharpe`. The
registry knows nothing about Sharpe ratios; `validation/research_dsr.py` sits
above both.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore

REGISTRY_VERSION = "g11_trial_registry_v1"

KIND_DECLARATION = "declaration"
KIND_EXECUTION = "execution"
KINDS = (KIND_DECLARATION, KIND_EXECUTION)

LOCK_SUFFIX = ".lock"

# TRIAL_REGISTRY_SPEC.md §3.1. Order is fixed here and must not be changed:
# the canonical payload is a sorted-key dict, so order does not affect the
# hash, but the tuple is the authoritative list of what identity means.
IDENTITY_FIELDS = (
    "research_objective",
    "research_stage",
    "hypothesis",
    "profile",
    "symbol",
    "dataset_contract",
    "target_hash",
    "horizon_bars",
    "feature_set_hash",
    "parameter_hash",
    "spec_hash",
    "criterion_hash",
)

ORIGIN_DECLARED = "declared"
ORIGIN_RECONSTRUCTED = "reconstructed"
ORIGIN_SYNTHETIC = "synthetic"
ORIGINS = (ORIGIN_DECLARED, ORIGIN_RECONSTRUCTED, ORIGIN_SYNTHETIC)

STATUS_DECLARED = "declared"
STATUS_COMPLETED = "completed"
STATUS_ABANDONED = "abandoned"
STATUS_INSUFFICIENT_DATA = "insufficient_data"
STATUS_WITHDRAWN_DUPLICATE = "withdrawn_duplicate"
STATUSES = (STATUS_DECLARED, STATUS_COMPLETED, STATUS_ABANDONED,
            STATUS_INSUFFICIENT_DATA, STATUS_WITHDRAWN_DUPLICATE)

# §8.1. The first three establish that a choice was made; `inference` only
# establishes that one probably was, and is what the uncertain band is made of.
TIER_COMMITTED_DOCUMENT = "committed_document"
TIER_GIT_HISTORY = "git_history"
TIER_LOCAL_REPORT = "local_report"
TIER_INFERENCE = "inference"
EVIDENCE_TIERS = (TIER_COMMITTED_DOCUMENT, TIER_GIT_HISTORY,
                  TIER_LOCAL_REPORT, TIER_INFERENCE)
CONFIRMING_TIERS = (TIER_COMMITTED_DOCUMENT, TIER_GIT_HISTORY,
                    TIER_LOCAL_REPORT)


class TrialRegistryError(Exception):
    pass


class ImmutableRecordError(TrialRegistryError):
    """Raised on any attempt to rewrite a declared trial."""


class UndeclaredTrialError(TrialRegistryError):
    """A result arrived for a trial that was never declared (§4.1)."""


def canonical_json(obj: Any) -> str:
    """The one serialization used for every hash and every stored line.

    Sorted keys and no incidental whitespace, so the same content produces the
    same bytes on any machine and a stored line round-trips exactly.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def feature_set_hash(features: Iterable[str]) -> str:
    """Order-insensitive by design: offering the same features in a different
    order is not a different research choice (§3.4)."""
    return content_hash(sorted(features))


def _component(value: Any, *, sort_sequence: bool = False) -> str | None:
    """A component hash from either a precomputed hash or the object itself.

    `None` stays `None` and is hashed distinctly from `""`, so "no feature set"
    and "the empty feature set" are different trials.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if sort_sequence and isinstance(value, (list, tuple, set, frozenset)):
        return feature_set_hash(value)  # type: ignore[arg-type]
    return content_hash(value)


@dataclass(frozen=True)
class TrialIdentity:
    """Exactly the selection-relevant fields of §3.1, and nothing else.

    Code commit, timestamps, seeds and CLI arguments are deliberately absent:
    a refactor that leaves the research question intact must not manufacture a
    trial, and a trial must be identifiable before any result exists.
    """
    research_objective: str
    research_stage: str
    hypothesis: str
    profile: str | None = None
    symbol: str | None = None
    dataset_contract: str | None = None
    target_hash: str | None = None
    horizon_bars: int | None = None
    feature_set_hash: str | None = None
    parameter_hash: str | None = None
    spec_hash: str | None = None
    criterion_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("research_objective", "research_stage", "hypothesis"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TrialRegistryError(
                    f"{name} is required and must be a non-empty string; "
                    f"got {value!r}")
        if self.horizon_bars is not None:
            if not isinstance(self.horizon_bars, int) or isinstance(
                    self.horizon_bars, bool):
                raise TrialRegistryError(
                    f"horizon_bars must be an int or None, got "
                    f"{self.horizon_bars!r}")
        for name in ("profile", "symbol", "dataset_contract", "target_hash",
                     "feature_set_hash", "parameter_hash", "spec_hash",
                     "criterion_hash"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TrialRegistryError(
                    f"{name} must be a string hash or None by the time the "
                    f"identity is constructed, got {type(value).__name__}; "
                    f"use TrialIdentity.build() to hash objects")

    @classmethod
    def build(cls, *, research_objective: str, research_stage: str,
              hypothesis: str, profile: str | None = None,
              symbol: str | None = None, dataset_contract: str | None = None,
              target: Any = None, horizon_bars: int | None = None,
              features: Any = None, parameters: Any = None,
              spec: Any = None, criterion: Any = None,
              selective_seed: int | None = None) -> "TrialIdentity":
        """Construct an identity from objects, hashing each component.

        `selective_seed` implements §3.3: a seed is not part of identity unless
        results across seeds were compared and one was chosen, in which case it
        folds into the parameter hash and the sweep's arms become distinct
        trials — which is what a seed sweep used for selection actually is.
        """
        parameter = _component(parameters)
        if selective_seed is not None:
            parameter = content_hash({"parameters": parameter,
                                      "selective_seed": int(selective_seed)})
        return cls(
            research_objective=research_objective,
            research_stage=research_stage,
            hypothesis=hypothesis,
            profile=profile,
            symbol=symbol,
            dataset_contract=dataset_contract,
            target_hash=_component(target),
            horizon_bars=horizon_bars,
            feature_set_hash=_component(features, sort_sequence=True),
            parameter_hash=parameter,
            spec_hash=_component(spec),
            criterion_hash=_component(criterion),
        )

    def payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in IDENTITY_FIELDS}

    @property
    def trial_id(self) -> str:
        return "t_" + content_hash(self.payload())[:16]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TrialIdentity":
        unknown = set(payload) - set(IDENTITY_FIELDS)
        if unknown:
            raise TrialRegistryError(
                f"identity payload has fields outside §3.1: {sorted(unknown)}")
        missing = set(IDENTITY_FIELDS) - set(payload)
        if missing:
            raise TrialRegistryError(
                f"identity payload is missing §3.1 fields: {sorted(missing)}")
        return cls(**payload)


@dataclass(frozen=True)
class Evidence:
    """What established that this trial happened (§8.1)."""
    tier: str
    ref: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.tier not in EVIDENCE_TIERS:
            raise TrialRegistryError(
                f"unknown evidence tier {self.tier!r}; expected one of "
                f"{list(EVIDENCE_TIERS)}")
        if not self.ref.strip():
            raise TrialRegistryError("evidence ref must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {"tier": self.tier, "ref": self.ref, "note": self.note}


@dataclass(frozen=True)
class TrialRecord:
    """One declared research choice.

    `multiplicity` is how many trials this one record stands for. It is
    `(1, 1)` for anything declared in the normal way. Reconstruction uses a
    band — `(4, 12)` for a threshold grid whose documented size is
    approximate — so that an unknown never has to be rounded to a single
    precise-looking number (§8.3).
    """
    identity: TrialIdentity
    created_at: str
    origin: str = ORIGIN_DECLARED
    status: str = STATUS_DECLARED
    target_family: str | None = None
    parent_trial_id: str | None = None
    evidence: tuple[Evidence, ...] = ()
    multiplicity: tuple[int, int] = (1, 1)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise TrialRegistryError(
                f"unknown origin {self.origin!r}; expected one of "
                f"{list(ORIGINS)}")
        if self.status not in STATUSES:
            raise TrialRegistryError(
                f"unknown status {self.status!r}; expected one of "
                f"{list(STATUSES)}")
        if not str(self.created_at).strip():
            raise TrialRegistryError("created_at must not be empty")
        lo, hi = self.multiplicity
        if not isinstance(lo, int) or not isinstance(hi, int):
            raise TrialRegistryError(
                f"multiplicity must be a pair of ints, got {self.multiplicity!r}")
        if lo < 1 or hi < lo:
            raise TrialRegistryError(
                f"multiplicity must satisfy 1 <= low <= high, got "
                f"{self.multiplicity!r}")
        if self.origin == ORIGIN_RECONSTRUCTED and not self.evidence:
            raise TrialRegistryError(
                f"reconstructed trial {self.identity.trial_id} has no "
                "evidence; §8.2 forbids inventing a trial that nothing "
                "establishes")

    @property
    def trial_id(self) -> str:
        return self.identity.trial_id

    @property
    def reconstructed(self) -> bool:
        return self.origin == ORIGIN_RECONSTRUCTED

    @property
    def confirmed_low(self) -> int:
        """How many of this record's trials are established, not inferred.

        A record resting on any `inference` evidence contributes nothing to the
        confirmed count — its whole multiplicity sits in the uncertain band.
        """
        if any(e.tier == TIER_INFERENCE for e in self.evidence):
            return 0
        return self.multiplicity[0]

    @property
    def conservative_high(self) -> int:
        return self.multiplicity[1]

    def as_record(self) -> dict[str, Any]:
        return {
            "registry_version": REGISTRY_VERSION,
            "kind": KIND_DECLARATION,
            "trial_id": self.trial_id,
            "identity": self.identity.payload(),
            "created_at": self.created_at,
            "origin": self.origin,
            "reconstructed": self.reconstructed,
            "status": self.status,
            "target_family": self.target_family,
            "parent_trial_id": self.parent_trial_id,
            "evidence": [e.as_dict() for e in self.evidence],
            "multiplicity_low": self.multiplicity[0],
            "multiplicity_high": self.multiplicity[1],
            "notes": self.notes,
        }

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "TrialRecord":
        identity = TrialIdentity.from_payload(rec["identity"])
        return cls(
            identity=identity,
            created_at=rec["created_at"],
            origin=rec["origin"],
            status=rec["status"],
            target_family=rec.get("target_family"),
            parent_trial_id=rec.get("parent_trial_id"),
            evidence=tuple(Evidence(**e) for e in rec.get("evidence", ())),
            multiplicity=(int(rec.get("multiplicity_low", 1)),
                          int(rec.get("multiplicity_high", 1))),
            notes=rec.get("notes", ""),
        )


def run_id(trial_id: str, code_commit: str, dataset_version: str,
           run_seed: int | None, started_at: str) -> str:
    """Derived, not random, so an identical replay is detectable as one."""
    return "r_" + content_hash({
        "trial_id": trial_id, "code_commit": code_commit,
        "dataset_version": dataset_version, "run_seed": run_seed,
        "started_at": started_at})[:16]


@dataclass(frozen=True)
class FamilyScope:
    """Which multiple-testing family a DSR is being deflated against (§6.1).

    Every unspecified component is a wildcard, and — separately — a record
    whose own field is `None` matches any scope value. Both directions widen
    the count, which is the direction §6.4 requires.
    """
    research_objective: str | None = None
    profile: str | None = None
    symbol: str | None = None
    target_family: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"research_objective": self.research_objective,
                "profile": self.profile, "symbol": self.symbol,
                "target_family": self.target_family}

    def matches(self, record: TrialRecord) -> bool:
        pairs = (
            (self.research_objective, record.identity.research_objective),
            (self.profile, record.identity.profile),
            (self.symbol, record.identity.symbol),
            (self.target_family, record.target_family),
        )
        for wanted, got in pairs:
            if wanted is None or got is None:
                continue
            if wanted != got:
                return False
        return True


@dataclass(frozen=True)
class TrialCount:
    """Three numbers, always together (§8.3).

    Publishing `conservative` alone would hide the reconstruction uncertainty
    behind a precise-looking figure, which §8.3 forbids; publishing
    `confirmed` alone would understate selection pressure, which is the whole
    failure mode this stage exists to close.
    """
    scope: dict[str, Any]
    confirmed: int
    conservative: int
    n_records: int
    trial_ids: tuple[str, ...] = field(default=(), repr=False)

    @property
    def uncertain(self) -> int:
        return self.conservative - self.confirmed

    @property
    def n_trials(self) -> int:
        """The number DSR uses. Conservative by construction."""
        return self.conservative

    def as_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "confirmed": self.confirmed,
                "uncertain": self.uncertain,
                "conservative": self.conservative,
                "n_records": self.n_records,
                "trial_ids": list(self.trial_ids)}


def _read_lines(path: str) -> Iterator[dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            if not line.endswith("\n"):
                raise TrialRegistryError(
                    f"{path}:{n} has no terminating newline — the registry was "
                    "truncated mid-write and its last trial may be incomplete; "
                    "a silently dropped trial is exactly what this registry "
                    "exists to prevent")
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TrialRegistryError(
                    f"{path}:{n} is not valid JSON: {exc}") from exc


@contextlib.contextmanager
def _exclusive(path: str):
    """Hold an exclusive lock across the whole check-then-append.

    Without it the conflict guard is a read-then-write race: two runners
    declaring the same trial would both find nothing and both append.
    """
    if fcntl is None:  # pragma: no cover - POSIX only in practice
        raise TrialRegistryError(
            "file locking is unavailable on this platform (no fcntl); "
            "refusing to append, because without the lock the conflict guard "
            "is a read-then-write race and the registry could gain two rows "
            "for one trial")
    if path.endswith(LOCK_SUFFIX):
        raise TrialRegistryError(
            f"registry path must not end in {LOCK_SUFFIX!r}: its sidecar lock "
            "would collide with another registry's")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + LOCK_SUFFIX, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class TrialRegistry:
    """Append-only store of trial declarations and executions."""

    def __init__(self, path: str) -> None:
        self.path = path

    # -- reading ---------------------------------------------------------

    def read_raw(self) -> list[dict[str, Any]]:
        """Every line as written, with no validation.

        For inspecting a registry you already suspect is damaged. Anything
        that draws a conclusion uses the validated readers instead.
        """
        return list(_read_lines(self.path))

    def _validated(self) -> tuple[dict[str, TrialRecord], dict[str, dict[str, Any]]]:
        """The single entry point every public reader goes through.

        Routing everything through one function removes a class of defect
        rather than an instance of it: a public method either calls this, or
        is deliberately raw (`read_raw`) and says so.
        """
        declarations: dict[str, TrialRecord] = {}
        executions: dict[str, dict[str, Any]] = {}
        for n, rec in enumerate(_read_lines(self.path), 1):
            kind = rec.get("kind")
            if kind not in KINDS:
                raise TrialRegistryError(
                    f"{self.path}:{n} has unknown kind {kind!r}; the registry "
                    f"is corrupt")
            if kind == KIND_DECLARATION:
                trial = TrialRecord.from_record(rec)
                if trial.trial_id != rec["trial_id"]:
                    raise TrialRegistryError(
                        f"{self.path}:{n} stores trial_id {rec['trial_id']!r} "
                        f"but its identity hashes to {trial.trial_id!r} — the "
                        f"record was edited after it was written")
                if trial.trial_id in declarations:
                    raise TrialRegistryError(
                        f"{self.path}:{n} declares {trial.trial_id} a second "
                        f"time; the registry is corrupt, a trial is declared "
                        f"exactly once")
                declarations[trial.trial_id] = trial
            else:
                rid = rec.get("run_id")
                if rid in executions:
                    raise TrialRegistryError(
                        f"{self.path}:{n} repeats run {rid}; the registry is "
                        f"corrupt")
                executions[rid] = rec
        for rid, rec in executions.items():
            if rec.get("trial_id") not in declarations:
                raise TrialRegistryError(
                    f"execution {rid} references undeclared trial "
                    f"{rec.get('trial_id')!r} — the registry is corrupt")
        return declarations, executions

    def declarations(self) -> dict[str, TrialRecord]:
        return self._validated()[0]

    def executions(self, trial_id: str | None = None) -> list[dict[str, Any]]:
        _, executions = self._validated()
        rows = list(executions.values())
        if trial_id is not None:
            rows = [r for r in rows if r["trial_id"] == trial_id]
        return rows

    def content_sha256(self) -> str:
        """Hash of the registry file, for pinning a count to a registry state."""
        if not os.path.exists(self.path):
            return hashlib.sha256(b"").hexdigest()
        with open(self.path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def snapshot(self) -> dict[str, Any]:
        declarations, executions = self._validated()
        return {"registry_version": REGISTRY_VERSION, "path": self.path,
                "content_sha256": self.content_sha256(),
                "n_declarations": len(declarations),
                "n_executions": len(executions)}

    # -- writing ---------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(canonical_json(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def declare(self, record: TrialRecord) -> TrialRecord:
        """Record a research choice. Refuses to change an existing one.

        An exactly identical re-declaration is a no-op rather than an error:
        reconstruction and idempotent setup both replay the same declarations,
        and making that fail would push callers toward a "clear it first"
        habit, which is how an append-only store stops being one.
        """
        with _exclusive(self.path):
            declarations, _ = self._validated()
            existing = declarations.get(record.trial_id)
            if existing is not None:
                if existing.as_record() == record.as_record():
                    return existing
                raise ImmutableRecordError(
                    f"trial {record.trial_id} is already declared with "
                    f"different content; declarations are immutable. A "
                    f"changed research choice is a new trial, and a changed "
                    f"record for an unchanged choice is a mutation of frozen "
                    f"history. Existing: {existing.as_record()!r}")
            self._append(record.as_record())
            return record

    def record_execution(self, trial_id: str, *, code_commit: str,
                         dataset_version: str, started_at: str,
                         run_seed: int | None = None,
                         status: str = STATUS_COMPLETED,
                         outcome_summary: dict[str, Any] | None = None,
                         ) -> dict[str, Any]:
        """Record one run of an already-declared trial.

        Never moves the trial count. An execution against an undeclared trial
        is inadmissible (§4.1): that rule is what stops a count from being
        assembled favourably once the outcomes are known.
        """
        if status not in STATUSES:
            raise TrialRegistryError(
                f"unknown status {status!r}; expected one of {list(STATUSES)}")
        with _exclusive(self.path):
            declarations, executions = self._validated()
            if trial_id not in declarations:
                raise UndeclaredTrialError(
                    f"no declaration for trial {trial_id}; a result whose "
                    f"trial was never declared is inadmissible "
                    f"(ARCHITECTURE.md §6.1)")
            rid = run_id(trial_id, code_commit, dataset_version, run_seed,
                         started_at)
            existing = executions.get(rid)
            record = {
                "registry_version": REGISTRY_VERSION,
                "kind": KIND_EXECUTION,
                "run_id": rid,
                "trial_id": trial_id,
                "code_commit": code_commit,
                "dataset_version": dataset_version,
                "run_seed": run_seed,
                "started_at": started_at,
                "status": status,
                "outcome_summary": outcome_summary or {},
            }
            if existing is not None:
                if existing == record:
                    return existing
                raise ImmutableRecordError(
                    f"run {rid} is already recorded with different content; "
                    f"executions are append-only")
            self._append(record)
            return record

    # -- counting --------------------------------------------------------

    def _counted_records(self, scope: FamilyScope) -> list[TrialRecord]:
        declarations = self.declarations()
        selected: dict[str, TrialRecord] = {}
        for trial_id, record in declarations.items():
            if record.origin == ORIGIN_SYNTHETIC:
                continue
            if record.status == STATUS_WITHDRAWN_DUPLICATE:
                continue
            if scope.matches(record):
                selected[trial_id] = record

        # §6.2: ancestry is pulled in unconditionally, even across families. A
        # post-hoc variant inherits the selection pressure of what it descends
        # from, and relabelling its objective would otherwise be the cheapest
        # way to shed that history.
        frontier = list(selected.values())
        while frontier:
            record = frontier.pop()
            parent_id = record.parent_trial_id
            if parent_id is None or parent_id in selected:
                continue
            parent = declarations.get(parent_id)
            if parent is None:
                raise TrialRegistryError(
                    f"trial {record.trial_id} names parent {parent_id}, which "
                    f"is not declared; the lineage cannot be counted")
            if parent.origin == ORIGIN_SYNTHETIC:
                continue
            selected[parent_id] = parent
            frontier.append(parent)
        return sorted(selected.values(), key=lambda r: r.trial_id)

    def n_trials(self, scope: FamilyScope | None = None) -> TrialCount:
        """The conservative trial count for a family, with its decomposition."""
        scope = scope or FamilyScope()
        records = self._counted_records(scope)
        return TrialCount(
            scope=scope.as_dict(),
            confirmed=sum(r.confirmed_low for r in records),
            conservative=sum(r.conservative_high for r in records),
            n_records=len(records),
            trial_ids=tuple(r.trial_id for r in records),
        )

    def n_trials_confirmed(self, scope: FamilyScope | None = None) -> int:
        """The count resting only on tier 1-3 evidence.

        Reported for transparency and explicitly not for deflating a published
        result (§6.4): using it would understate selection pressure by exactly
        the amount that is uncertain.
        """
        return self.n_trials(scope).confirmed


def declare_all(registry: TrialRegistry,
                records: Sequence[TrialRecord]) -> list[TrialRecord]:
    """Declare a batch, in order, idempotently."""
    return [registry.declare(r) for r in records]


def with_parent(record: TrialRecord, parent: TrialRecord) -> TrialRecord:
    """A post-hoc variant, linked to what it descends from (§2.1)."""
    return replace(record, parent_trial_id=parent.trial_id)
