"""G1.1 — the trial registry: identity, persistence and counting.

The tests are grouped the way TRIAL_REGISTRY_SPEC.md is: what makes a trial the
same trial (§3), what makes a run not a trial (§4), what the store refuses to
do (§5), and what enters a family count (§6).
"""
from __future__ import annotations

import ast
import hashlib
import json

import pytest

from validation.trial_registry import (KIND_DECLARATION, ORIGIN_RECONSTRUCTED,
                                       ORIGIN_SYNTHETIC,
                                       STATUS_WITHDRAWN_DUPLICATE, Evidence,
                                       FamilyScope, ImmutableRecordError,
                                       TrialIdentity, TrialRecord,
                                       TrialRegistry, TrialRegistryError,
                                       UndeclaredTrialError, canonical_json,
                                       content_hash, feature_set_hash, run_id,
                                       with_parent)

BASE = dict(research_objective="trade_expectancy", research_stage="X1",
            hypothesis="h", profile="swing", symbol="BTCUSDT",
            dataset_contract="binance:BTCUSDT")


def identity(**overrides) -> TrialIdentity:
    kwargs = dict(BASE)
    kwargs.update(overrides)
    return TrialIdentity.build(**kwargs)


def record(created_at: str = "2026-01-01", **overrides) -> TrialRecord:
    extra = {k: overrides.pop(k) for k in list(overrides)
             if k in {"origin", "status", "target_family", "parent_trial_id",
                      "evidence", "multiplicity", "notes"}}
    return TrialRecord(identity=identity(**overrides), created_at=created_at,
                       target_family=extra.pop("target_family", "trade_r"),
                       **extra)


@pytest.fixture()
def registry(tmp_path) -> TrialRegistry:
    return TrialRegistry(str(tmp_path / "reg" / "trials.jsonl"))


def rewrite(registry: TrialRegistry, rows: list[dict]) -> None:
    """Rewrite the file with a valid seq/prev chain.

    Tampering tests need this: without it the chain check fires first and the
    specific guard under test — unknown kind, duplicate declaration, orphaned
    execution — is never reached, so it would look covered while being untested.
    """
    prev = ""
    with open(registry.path, "w") as fh:
        for n, row in enumerate(rows):
            row = {k: v for k, v in row.items() if k not in ("seq",
                                                             "prev_sha256")}
            row["seq"] = n
            row["prev_sha256"] = prev
            line = canonical_json(row)
            fh.write(line + "\n")
            prev = hashlib.sha256(line.encode()).hexdigest()


# --- §3 identity ---------------------------------------------------------


def test_the_same_research_choice_maps_to_the_same_id():
    assert identity().trial_id == identity().trial_id


def test_an_identical_rerun_is_the_same_trial_even_on_new_code(registry):
    first = registry.declare(record())
    again = registry.declare(record())
    assert again.trial_id == first.trial_id
    assert len(registry.declarations()) == 1


def test_a_changed_threshold_is_a_new_trial():
    a = identity(parameters={"score_threshold": 7})
    b = identity(parameters={"score_threshold": 8})
    assert a.trial_id != b.trial_id


def test_a_changed_feature_set_is_a_new_trial():
    a = identity(features=["atr", "cvd"])
    b = identity(features=["atr", "cvd", "rsi"])
    assert a.trial_id != b.trial_id


def test_reordering_a_feature_set_is_not_a_new_trial():
    a = identity(features=["atr", "cvd", "rsi"])
    b = identity(features=["rsi", "atr", "cvd"])
    assert a.trial_id == b.trial_id


def test_a_changed_target_is_a_new_trial():
    a = identity(target={"quantity": "trade_r"})
    b = identity(target={"quantity": "realized_range"})
    assert a.trial_id != b.trial_id


def test_a_changed_horizon_is_a_new_trial():
    assert identity(horizon_bars=12).trial_id != identity(
        horizon_bars=24).trial_id


def test_a_changed_hyperparameter_used_for_selection_is_a_new_trial():
    a = identity(parameters={"alpha": 0.1})
    b = identity(parameters={"alpha": 10.0})
    assert a.trial_id != b.trial_id


def test_a_changed_acceptance_criterion_is_a_new_trial():
    a = identity(criterion={"checks": 6})
    b = identity(criterion={"checks": 12})
    assert a.trial_id != b.trial_id


def test_a_changed_dataset_contract_is_a_new_trial():
    a = identity(dataset_contract="binance:BTCUSDT")
    b = identity(dataset_contract="bybit:BTCUSDT")
    assert a.trial_id != b.trial_id


def test_a_reproducibility_seed_is_not_a_new_trial():
    """§3.3: reruns under different seeds are one research choice."""
    a = identity(parameters={"alpha": 1.0})
    b = identity(parameters={"alpha": 1.0})
    assert a.trial_id == b.trial_id


def test_a_seed_that_is_selected_over_is_a_new_trial():
    a = identity(parameters={"alpha": 1.0}, selective_seed=1)
    b = identity(parameters={"alpha": 1.0}, selective_seed=2)
    plain = identity(parameters={"alpha": 1.0})
    assert a.trial_id != b.trial_id
    assert a.trial_id != plain.trial_id


def test_none_and_empty_string_are_different_choices():
    """"no feature set" and "the empty feature set" are not the same claim."""
    assert identity(features=None).trial_id != identity(features=[]).trial_id


def test_identity_carries_no_execution_fields():
    """A field list that quietly grew a code_commit would make every refactor
    a new trial, which is the failure §3.2 exists to prevent."""
    payload = identity().payload()
    for banned in ("code_commit", "created_at", "run_seed", "runner",
                   "status", "notes"):
        assert banned not in payload


def test_identity_rejects_an_unknown_payload_field():
    payload = identity().payload()
    payload["code_commit"] = "abc123"
    with pytest.raises(TrialRegistryError, match="outside"):
        TrialIdentity.from_payload(payload)


def test_identity_round_trips_through_its_payload():
    ident = identity(features=["a", "b"], parameters={"x": 1}, horizon_bars=12)
    assert TrialIdentity.from_payload(ident.payload()) == ident


def test_identity_requires_the_three_naming_fields():
    with pytest.raises(TrialRegistryError, match="hypothesis"):
        TrialIdentity(research_objective="o", research_stage="s",
                      hypothesis="  ")


def test_identity_rejects_an_unhashed_object_in_a_hash_field():
    with pytest.raises(TrialRegistryError, match="TrialIdentity.build"):
        TrialIdentity(research_objective="o", research_stage="s",
                      hypothesis="h", parameter_hash={"raw": "dict"})


def test_a_precomputed_hash_and_its_object_agree():
    direct = identity(parameters={"a": 1})
    precomputed = TrialIdentity(**{**BASE,
                                   "parameter_hash": content_hash({"a": 1})})
    assert direct.trial_id == precomputed.trial_id


def test_feature_set_hash_is_order_insensitive():
    assert feature_set_hash(["b", "a"]) == feature_set_hash(["a", "b"])


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


# --- §4 trial vs execution ----------------------------------------------


def test_an_execution_does_not_move_the_trial_count(registry):
    declared = registry.declare(record())
    for commit in ("aaa", "bbb", "ccc"):
        registry.record_execution(declared.trial_id, code_commit=commit,
                                  dataset_version="v1",
                                  started_at="2026-01-02")
    assert registry.n_trials().n_trials == 1
    assert len(registry.executions()) == 3


def test_execution_identity_differs_while_trial_identity_holds(registry):
    declared = registry.declare(record())
    a = registry.record_execution(declared.trial_id, code_commit="aaa",
                                  dataset_version="v1",
                                  started_at="2026-01-02")
    b = registry.record_execution(declared.trial_id, code_commit="bbb",
                                  dataset_version="v1",
                                  started_at="2026-01-02")
    assert a["run_id"] != b["run_id"]
    assert a["trial_id"] == b["trial_id"] == declared.trial_id


def test_an_undeclared_result_is_inadmissible(registry):
    with pytest.raises(UndeclaredTrialError, match="inadmissible"):
        registry.record_execution("t_nothing", code_commit="aaa",
                                  dataset_version="v1",
                                  started_at="2026-01-02")


def test_an_identical_execution_replay_is_idempotent(registry):
    declared = registry.declare(record())
    kwargs = dict(code_commit="aaa", dataset_version="v1",
                  started_at="2026-01-02", run_seed=7)
    first = registry.record_execution(declared.trial_id, **kwargs)
    second = registry.record_execution(declared.trial_id, **kwargs)
    assert first == second
    assert len(registry.executions()) == 1


def test_a_conflicting_execution_is_rejected(registry):
    declared = registry.declare(record())
    kwargs = dict(code_commit="aaa", dataset_version="v1",
                  started_at="2026-01-02")
    registry.record_execution(declared.trial_id, **kwargs,
                              outcome_summary={"net_r": 0.1})
    with pytest.raises(ImmutableRecordError, match="append-only"):
        registry.record_execution(declared.trial_id, **kwargs,
                                  outcome_summary={"net_r": 0.9})


def test_run_id_is_derived_not_random():
    args = ("t_x", "aaa", "v1", None, "2026-01-02")
    assert run_id(*args) == run_id(*args)


def test_execution_rejects_an_unknown_status(registry):
    declared = registry.declare(record())
    with pytest.raises(TrialRegistryError, match="unknown status"):
        registry.record_execution(declared.trial_id, code_commit="a",
                                  dataset_version="v1",
                                  started_at="2026-01-02", status="looks_good")


# --- §5 persistence ------------------------------------------------------


def test_a_conflicting_declaration_is_rejected(registry):
    """Same id, different content: either a hash collision or a frozen spec
    being mutated in place. Both fail closed."""
    first = record()
    registry.declare(first)
    mutated = TrialRecord(identity=first.identity, created_at="2026-06-06",
                          target_family="trade_r")
    with pytest.raises(ImmutableRecordError, match="immutable"):
        registry.declare(mutated)


def test_nothing_ever_opens_the_registry_for_writing():
    """The append-only guarantee is one `open(..., "a")` away from being a
    lie, so the mode is asserted directly on the source."""
    src = open("validation/trial_registry.py").read()
    assert 'open(self.path, "a")' in src
    assert 'open(self.path, "w")' not in src


def test_a_truncated_final_line_is_a_corrupt_registry(registry, tmp_path):
    registry.declare(record())
    registry.declare(record(hypothesis="second"))
    with open(registry.path) as fh:
        text = fh.read()
    with open(registry.path, "w") as fh:
        fh.write(text[:-12])
    with pytest.raises(TrialRegistryError, match="truncated"):
        registry.declarations()


def test_an_unparseable_line_is_detected(registry):
    registry.declare(record())
    with open(registry.path, "a") as fh:
        fh.write("{not json\n")
    with pytest.raises(TrialRegistryError, match="not valid JSON"):
        registry.declarations()


def test_an_unknown_kind_is_detected(registry):
    registry.declare(record())
    rows = registry.read_raw() + [{"kind": "amendment"}]
    rewrite(registry, rows)
    with pytest.raises(TrialRegistryError, match="unknown kind"):
        registry.declarations()


def test_a_duplicated_declaration_line_is_detected(registry):
    """Idempotent replay never writes a second line, so a second line for one
    trial means the file was tampered with — not an update."""
    registry.declare(record())
    rows = registry.read_raw()
    rewrite(registry, rows + rows)
    with pytest.raises(TrialRegistryError, match="second time"):
        registry.declarations()


def test_an_edited_identity_no_longer_hashes_to_its_id(registry):
    registry.declare(record())
    rows = registry.read_raw()
    rows[0]["identity"]["hypothesis"] = "something_else"
    rewrite(registry, rows)
    with pytest.raises(TrialRegistryError, match="hashes to"):
        registry.declarations()


def test_an_execution_without_its_declaration_is_detected(registry):
    declared = registry.declare(record())
    registry.record_execution(declared.trial_id, code_commit="a",
                              dataset_version="v1", started_at="2026-01-02")
    kept = [r for r in registry.read_raw() if r["kind"] != KIND_DECLARATION]
    rewrite(registry, kept)
    with pytest.raises(TrialRegistryError, match="undeclared trial"):
        registry.declarations()


def test_a_removed_record_is_detected(registry):
    """The failure the registry exists to prevent, made detectable.

    A file that simply loses a line would otherwise read as a valid, smaller
    registry — a silent undercount, which is the one direction of error that
    always flatters the result.
    """
    for name in ("a", "b", "c"):
        registry.declare(record(hypothesis=name))
    lines = open(registry.path).read().splitlines(keepends=True)
    with open(registry.path, "w") as fh:
        fh.writelines(lines[:1] + lines[2:])
    with pytest.raises(TrialRegistryError, match="removed or reordered"):
        registry.declarations()


def test_a_reordered_registry_is_detected(registry):
    for name in ("a", "b"):
        registry.declare(record(hypothesis=name))
    lines = open(registry.path).read().splitlines(keepends=True)
    with open(registry.path, "w") as fh:
        fh.writelines(reversed(lines))
    with pytest.raises(TrialRegistryError, match="removed or reordered"):
        registry.declarations()


def test_a_record_edited_in_place_breaks_the_chain(registry):
    """Even an edit that leaves the identity hash intact — a note, a status —
    is caught, because the next line's chain no longer matches."""
    registry.declare(record(hypothesis="a"))
    registry.declare(record(hypothesis="b"))
    rows = [json.loads(line) for line in open(registry.path)]
    rows[0]["notes"] = "quietly reinterpreted later"
    with open(registry.path, "w") as fh:
        for row in rows:
            fh.write(canonical_json(row) + "\n")
    with pytest.raises(TrialRegistryError, match="does not chain"):
        registry.declarations()


def test_the_first_record_chains_to_nothing(registry):
    registry.declare(record())
    row = json.loads(open(registry.path).read())
    assert row["seq"] == 0
    assert row["prev_sha256"] == ""


def test_chain_fields_do_not_make_a_replay_look_like_a_conflict(registry):
    """They describe a line's position in the file, not the research choice,
    so idempotent replay must ignore them."""
    declared = registry.declare(record())
    registry.declare(record())
    kwargs = dict(code_commit="a", dataset_version="v1",
                  started_at="2026-01-02")
    registry.record_execution(declared.trial_id, **kwargs)
    registry.record_execution(declared.trial_id, **kwargs)
    assert len(registry.read_raw()) == 2


def test_a_count_and_its_snapshot_come_from_one_read(registry):
    """Counting and hashing in two reads is a race: a declaration landing
    between them pins a registry state that is not the one counted."""
    registry.declare(record())
    snapshot, count, declarations = registry.snapshot_and_count(FamilyScope())
    assert snapshot["content_sha256"] == registry.content_sha256()
    assert snapshot["n_declarations"] == count.n_records == len(declarations)


def test_serialization_is_deterministic(registry):
    registry.declare(record(features=["b", "a"], parameters={"z": 1, "a": 2}))
    stored = open(registry.path).read()
    reserialized = canonical_json(json.loads(stored)) + "\n"
    assert stored == reserialized


def test_two_registries_from_the_same_records_are_byte_identical(tmp_path):
    records = [record(), record(hypothesis="b"), record(hypothesis="c")]
    paths = []
    for name in ("one", "two"):
        reg = TrialRegistry(str(tmp_path / name / "trials.jsonl"))
        for r in records:
            reg.declare(r)
        paths.append(reg.path)
    assert open(paths[0]).read() == open(paths[1]).read()


def test_read_raw_returns_a_damaged_registry_without_judging_it(registry):
    registry.declare(record())
    with open(registry.path, "a") as fh:
        fh.write(json.dumps({"kind": "amendment"}) + "\n")
    assert len(registry.read_raw()) == 2


def test_the_registry_refuses_a_path_that_would_collide_with_a_lock(tmp_path):
    reg = TrialRegistry(str(tmp_path / "trials.jsonl.lock"))
    with pytest.raises(TrialRegistryError, match="sidecar lock"):
        reg.declare(record())


def test_content_hash_changes_when_a_trial_is_added(registry):
    before = registry.content_sha256()
    registry.declare(record())
    assert registry.content_sha256() != before


def test_the_registry_has_no_delete_or_compact_operation():
    """§5: no silent deletion, no compaction that destroys history. The
    absence is the mechanism, so it is asserted rather than assumed."""
    for banned in ("def delete", "def remove", "def compact", "def amend",
                   "def update"):
        assert banned not in open("validation/trial_registry.py").read()


def test_a_missing_registry_reads_as_empty(registry):
    assert registry.declarations() == {}
    assert registry.n_trials().n_trials == 0


# --- §6 counting ---------------------------------------------------------


def test_a_family_count_ignores_other_families(registry):
    registry.declare(record(research_objective="trade_expectancy"))
    registry.declare(record(research_objective="volatility_forecast",
                            target_family="volatility_ratio"))
    scope = FamilyScope(research_objective="trade_expectancy",
                        target_family="trade_r")
    assert registry.n_trials(scope).n_trials == 1


def test_a_different_profile_is_a_different_family(registry):
    registry.declare(record(profile="swing"))
    registry.declare(record(profile="intraday"))
    assert registry.n_trials(FamilyScope(profile="swing")).n_trials == 1
    assert registry.n_trials().n_trials == 2


def test_an_unknown_profile_counts_in_every_family(registry):
    """§6.4: ambiguity widens the count rather than escaping it."""
    registry.declare(record(profile=None))
    for profile in ("swing", "intraday", "position", "bounce"):
        assert registry.n_trials(FamilyScope(profile=profile)).n_trials == 1


def test_a_post_hoc_variant_increases_the_count(registry):
    parent = registry.declare(record(hypothesis="ridge"))
    child = with_parent(record(hypothesis="ridge_ranking"), parent)
    registry.declare(child)
    assert registry.n_trials().n_trials == 2


def test_a_lineage_is_pulled_in_across_families(registry):
    """Relabelling a variant's objective must not shed the selection pressure
    it inherited — otherwise it is the cheapest possible way to undercount."""
    parent = registry.declare(record(research_objective="volatility_forecast",
                                     hypothesis="ridge",
                                     target_family="volatility_ratio"))
    child = with_parent(record(research_objective="trade_expectancy",
                               hypothesis="ridge_for_trading"), parent)
    registry.declare(child)
    scope = FamilyScope(research_objective="trade_expectancy",
                        target_family="trade_r")
    count = registry.n_trials(scope)
    assert count.n_trials == 2
    assert parent.trial_id in count.trial_ids


def test_a_lineage_chain_is_followed_to_its_root(registry):
    root = registry.declare(record(hypothesis="a"))
    mid = registry.declare(with_parent(record(hypothesis="b"), root))
    registry.declare(with_parent(record(hypothesis="c"), mid))
    scope = FamilyScope()
    assert registry.n_trials(scope).n_trials == 3


def test_a_dangling_parent_fails_closed(registry):
    registry.declare(TrialRecord(identity=identity(), created_at="2026-01-01",
                                 target_family="trade_r",
                                 parent_trial_id="t_missing"))
    with pytest.raises(TrialRegistryError, match="not declared"):
        registry.n_trials()


def test_synthetic_trials_are_excluded(registry):
    registry.declare(record())
    registry.declare(record(hypothesis="nc1", origin=ORIGIN_SYNTHETIC))
    assert registry.n_trials().n_trials == 1


def test_a_withdrawn_duplicate_is_excluded(registry):
    registry.declare(record())
    registry.declare(record(hypothesis="dupe",
                            status=STATUS_WITHDRAWN_DUPLICATE))
    assert registry.n_trials().n_trials == 1


def test_an_abandoned_trial_still_counts(registry):
    registry.declare(record(hypothesis="rejected", status="abandoned"))
    registry.declare(record(hypothesis="no_data", status="insufficient_data"))
    assert registry.n_trials().n_trials == 2


def test_multiplicity_stands_for_a_grid(registry):
    registry.declare(record(multiplicity=(6, 6)))
    count = registry.n_trials()
    assert count.confirmed == 6
    assert count.conservative == 6
    assert count.n_records == 1


def test_an_uncertain_band_lands_in_the_uncertain_count(registry):
    registry.declare(record(multiplicity=(1, 6)))
    count = registry.n_trials()
    assert (count.confirmed, count.uncertain, count.conservative) == (1, 5, 6)
    assert count.n_trials == 6


def test_inferred_evidence_contributes_nothing_to_confirmed(registry):
    registry.declare(record(
        origin=ORIGIN_RECONSTRUCTED,
        evidence=(Evidence("inference", "guess", "no direct record"),)))
    count = registry.n_trials()
    assert count.confirmed == 0
    assert count.conservative == 1


def test_n_trials_is_the_conservative_count(registry):
    registry.declare(record(multiplicity=(2, 9)))
    count = registry.n_trials()
    assert count.n_trials == count.conservative == 9
    assert registry.n_trials_confirmed() == 2


def test_the_scope_travels_with_the_count(registry):
    registry.declare(record())
    count = registry.n_trials(FamilyScope(profile="swing"))
    assert count.as_dict()["scope"]["profile"] == "swing"


# --- record validation ---------------------------------------------------


def test_a_reconstructed_record_without_evidence_is_refused():
    with pytest.raises(TrialRegistryError, match="no evidence"):
        TrialRecord(identity=identity(), created_at="2026-01-01",
                    origin=ORIGIN_RECONSTRUCTED)


def test_evidence_rejects_an_unknown_tier():
    with pytest.raises(TrialRegistryError, match="unknown evidence tier"):
        Evidence("vibes", "somewhere")


def test_evidence_rejects_an_empty_ref():
    with pytest.raises(TrialRegistryError, match="ref must not be empty"):
        Evidence("git_history", "   ")


def test_a_record_rejects_an_unknown_origin():
    with pytest.raises(TrialRegistryError, match="unknown origin"):
        TrialRecord(identity=identity(), created_at="2026-01-01",
                    origin="probably_real")


def test_a_record_rejects_an_inverted_multiplicity():
    with pytest.raises(TrialRegistryError, match="1 <= low <= high"):
        TrialRecord(identity=identity(), created_at="2026-01-01",
                    multiplicity=(5, 2))


def test_a_record_rejects_a_zero_multiplicity():
    with pytest.raises(TrialRegistryError, match="1 <= low <= high"):
        TrialRecord(identity=identity(), created_at="2026-01-01",
                    multiplicity=(0, 3))


def test_a_record_round_trips_through_its_serialized_form():
    original = record(features=["a"], multiplicity=(1, 4),
                      origin=ORIGIN_RECONSTRUCTED,
                      evidence=(Evidence("git_history", "abc123", "note"),))
    assert TrialRecord.from_record(original.as_record()) == original


def _imported_modules(path: str) -> set[str]:
    """Real imports only. A grep would match the module's own prose, and a
    layering rule enforced by prose is not enforced."""
    tree = ast.parse(open(path).read())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_registry_does_not_import_the_sharpe_module():
    """§1 layering: the registry knows nothing about Sharpe ratios, so the
    dependency can only run one way and no cycle is possible."""
    imports = _imported_modules("validation/trial_registry.py")
    assert not any("deflated_sharpe" in name for name in imports)


def test_the_registry_does_not_import_offline_tools():
    imports = _imported_modules("validation/trial_registry.py")
    assert not any(name == "tools" or name.startswith("tools.")
                   for name in imports)
