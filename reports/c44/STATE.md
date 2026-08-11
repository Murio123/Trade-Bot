# reports/c44 — current state (2026-08-11)

## Consistent

- `c44_ridge_frozen_v1.json` — sha256 `a5013922…`, matches the pin in
  `model_registry/c44_ridge_frozen_v1.json`. Trained on dataset_version
  `98790821…`.

## Missing

`reference_ranker.json` and `reference_ridge.json` for **this** artifact no
longer exist. They were overwritten by a re-run on the refreshed dataset and
were never backed up; only the artifact itself was.

C4.5 cannot produce a forecast without them, because `load_reference` requires
a pinned hash and the artifact, the ranker reference, and the ridge reference
must all come from one dataset.

## Quarantined

`stale_refresh_20260811/` holds the outputs of the re-run on the refreshed
dataset (4h/9400 through 2026-08-11): a different artifact (sha `1d9bbe7c…`,
dataset_version `7dbef2ab…`) plus its two references and freeze report. They
are internally consistent with each other but not with the registered v1, and
`train_idx_range` reads identically `[1400, 8187]` in both because `idx` is
positional inside a sliding window — which is precisely why the overwrite was
not obvious.

Kept rather than deleted: they are the only surviving evidence of what the
re-run produced.

## Required before C4.5 resumes

Freeze a **new** artifact id (`c45_ridge_frozen_v2`) on the refreshed dataset,
together with its two references, in one run — now safe, because
`tools/ridge_freeze_artifact.py` writes through `_write_frozen` and refuses to
replace a frozen file with different content.

Do **not** re-point v1 at the refreshed data. Its registry entry is immutable
and correct as it stands.
