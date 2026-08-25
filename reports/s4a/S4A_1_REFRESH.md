# S4A.1 — automated spot snapshot refresh

Recorded 2026-08-25. **Operational stage.** No trial, no hypothesis, no
threshold moved. S3 is still `S3_NO_SIGNAL`, `CLEAN_2X` is still frozen, the
four factual views and every eligibility rule are byte-for-byte what S4A
shipped, and `test_this_stage_changed_no_ranking_or_eligibility_rule` asserts
it.

The problem: the snapshot was built by hand. After 24 hours the reader's
fail-closed rule turns the scanner off, correctly, and nobody rebuilds it.

---

## 1. Chosen architecture

**An in-process APScheduler job that shells out to the existing CLI.**

```
main.py  ─ one Railway service, one process, one filesystem
  ├─ Telegram Application (long polling)
  └─ AsyncIOScheduler
       ├─ analysis / alerts / trades / outcomes   (existing)
       └─ spot_snapshot  ── every 2h ──▶ subprocess:
                                        python -m tools.spot_snapshot --source live
                                          └─ writes data/spot/snapshot/current.json
                                                          ▲
                              bot/spot_screener.py ───────┘ reads the same file
```

Three files changed in production (`scheduler.py`, `main.py`, `config.py`),
one new module (`spot_market/refresh.py`), one new capability in the existing
builder (a lock).

## 2. Why this and not a Railway cron service

**Because a separate service would have its own filesystem.** The deployment is
`Procfile: worker: python main.py` — a single Nixpacks service, no
`railway.json`, no `Dockerfile`, and an in-process scheduler that already runs
five jobs. A Railway cron service is a *different container*: it would build a
perfectly good snapshot into a filesystem the bot process cannot see, and the
scanner would stay dark forever while the cron logs reported success. That is
exactly the failure the stage brief warned about, and it is the deciding
argument. `test_the_deployment_runs_one_service_that_owns_the_snapshot` pins
the Procfile so a second process type has to come back through a test.

Given one process, two sub-decisions:

* **The builder is a subprocess, not an import.** `tools/spot_snapshot.py`
  imports `spot.features` and `spot.universe`. Importing it from `scheduler.py`
  would pull offline research code into the live runtime — the direction the
  S-track's import rule exists to forbid. A module name in a string is not an
  import, and `test_the_research_package_never_enters_the_bot_process` checks
  that no module reachable from `main.py` imports `spot` or the builder. It
  also means a builder that segfaults or gets OOM-killed is an exit code, not
  an exception in the event loop the futures streams are sharing.
* **`spot_market/refresh.py` owns the decision, not the work.** It imports only
  stdlib and its own package, so the `spot_market` boundary is unchanged.

`scheduler.py` is now the second module allowed to import `spot_market`.
That widening is explicit — `SPOT_MARKET_CONSUMERS` in
`tests/test_spot_screener.py` lists both files and says what each is for — and
it is not a weakening of the research guards in `tests/test_g1_integration.py`,
which are untouched.

## 3. Cadence

**Every 2 hours, at minute 7**: `CronTrigger(hour="*/2", minute=7)` → 00:07,
02:07, … Configurable via `SPOT_SNAPSHOT_REFRESH_HOURS`, clamped to **1–12**.
The ceiling has two independent reasons and the stricter one wins: APScheduler
rejects a step above 23 in the hour field, and a cadence past 12 h could not
hold the snapshot inside the reader's 24 h limit with any margin — the scanner
would go dark between refreshes by design.

Cron rather than interval so the cadence survives a redeploy instead of
restarting from "now" each time the container comes back. Minute 7 puts the
midnight run just after the daily bar closes at 00:00 UTC, with margin for the
venue to finalise it.

Two hours is deliberate against a **24 h** fail-closed limit: twelve
consecutive failures are survivable before the scanner goes dark.
`test_the_refresh_window_sits_well_inside_the_fail_closed_limit` asserts the
gap is at least 4×. The source is daily OHLCV, so anything faster buys nothing.

`coalesce=True` collapses slots missed during downtime into one run;
`max_instances=1` is the in-process half of the overlap guard;
`misfire_grace_time=1800`.

## 4. Command

Identical in both invocation paths — there is one builder:

```
python -m tools.spot_snapshot --source live --snapshot-dir data/spot/snapshot
```

Manual operation is unchanged and still supported:

```
python -m tools.spot_snapshot --source live
python -m tools.spot_snapshot --source panel     # offline, from the S1 panel
```

`test_the_automated_path_invokes_that_same_cli` captures the argv the
scheduler actually builds. Exit codes are part of the contract: `0` built,
`1` failed, `2` another build holds the lock.

## 5. Snapshot path

`data/spot/snapshot/current.json`, and both sides derive it from the same
constants — `test_the_snapshot_path_is_the_same_one_on_both_sides` asserts
`spot_market.snapshot.DEFAULT_SNAPSHOT_PATH ==
os.path.join(tools.spot_snapshot.DEFAULT_SNAPSHOT_DIR, SNAPSHOT_NAME)`.

The refresher passes `--snapshot-dir` **explicitly** rather than letting the
subprocess inherit a working directory. The bot and the builder must not be
able to disagree about which file is canonical, and a working directory is
exactly the kind of thing that differs between a Railway container and a
developer's shell.

## 6. Persistence analysis

| question | answer |
|---|---|
| Does the path survive a redeploy? | **No.** Railway's container filesystem is ephemeral and `data/` is gitignored, so it is not in the image either. |
| Do the bot and the refresher see the same file? | **Yes** — same process, same container. This is the whole basis of the design. |
| Would two Railway services share it? | **No.** Which is why there is only one. |
| Is that acceptable? | **Yes**, and this is the part worth stating plainly. The snapshot is a *cache of the present*, fully regenerable from public data in ~22–33 s. Nothing in it is a record of anything. Losing it on redeploy costs one startup rebuild, not information. |

No Railway volume is required, and none is used. Startup covers the empty-disk
case (§9).

## 7. Atomic and durable writes

`write_snapshot` writes to a unique temp file in the destination directory,
**fsyncs it**, `os.replace`s it into position, and then **fsyncs the
directory**. Same filesystem, so the rename is atomic: a handler reading
during a replace sees the whole old file or the whole new one, never a half.

The two fsyncs were added after the audit. Without the first, a host that dies
just after the rename can come back with a directory entry pointing at
unflushed content — an atomic rename to nothing. Without the second, the
rename itself is not persisted on most filesystems.

The two are treated differently, on purpose. **A file fsync that fails aborts
the publish**: data the kernel has not committed must not replace a good
snapshot, so the scratch file is removed, the previous snapshot stays
byte-identical and the build exits 1. **A directory fsync that fails does
not** — not every filesystem permits it, and refusing there would disable the
scanner over a durability nicety rather than a correctness one. It is
best-effort but **never silent**: `_fsync_dir` returns whether it happened and
prints why it did not, so "durable publish" cannot quietly become "atomic
publish" with nobody able to tell which one they have.

The ordering is asserted against the real syscalls, not just their presence —
`fsync(file) → replace → fsync(dir)`; an fsync after the rename would be
useless.

The temp name is unique (`tempfile.mkstemp`) rather than fixed and is removed
on any exception including `KeyboardInterrupt`. A successful write sweeps
orphans left by a build that was SIGKILLed between opening its scratch file
and renaming it — but only those **older than an hour**. `write_snapshot` is
an ordinary function and does not hold the build lock itself, so a sweep that
deleted every scratch file it found could delete one another writer was in the
middle of; age is what separates "abandoned" from "in flight".

The stronger guarantee is **ordering**, and it is what `main()` was
restructured around: the lock is taken, the snapshot is built and validated
entirely in memory, and only a complete payload reaches the filesystem. There
is no instant at which the old snapshot is gone and the new one does not yet
exist. Nothing is ever truncated in place, and the only files the write path
unlinks are scratch files — the canonical path is replaced, never removed and
rewritten. Both are asserted on the AST.

`test_an_interrupted_build_leaves_the_previous_snapshot_intact` kills a real
builder process mid-run and compares SHA-256 before and after. That property
comes from ordering, not from a `finally`, so it has to hold when the process
simply stops existing.

## 8. Concurrency

Two mechanisms, each doing one thing:

* **`max_instances=1`** on the APScheduler job — two scheduled runs cannot
  overlap in-process.
* **`flock` on `data/spot/snapshot/.refresh.lock`** in the builder — covers
  every invocation path, including the overlap most likely to actually happen:
  an operator running the CLI by hand while the two-hourly job is mid-build.
  A second build exits `2` immediately rather than waiting.

`flock` rather than a pid file for one reason: **the kernel releases it when
the holder dies, however it dies.** A pid file survives SIGKILL and then jams
every later refresh until a human notices — a worse failure than the overlap
it prevents. `test_the_lock_is_released_when_its_holder_exits` kills a holder
and shows the next build succeeds.

No Redis, no database lock. Neither would be correct here anyway: the thing
being protected is a file on one container.

## 9. Startup behaviour

`main.py` adds a one-shot `initial_spot_snapshot` job after
`scheduler.start()`, alongside the existing `initial_analysis`:

* **fresh snapshot survived the restart** → `needs_refresh` is false, the job
  returns in milliseconds, no rebuild;
* **absent (the normal post-redeploy state) or stale or unreadable** → one
  controlled rebuild;
* **the rebuild fails** → logged, the bot carries on.

Scheduled rather than awaited: startup must not block ~22 s on Binance's spot
API, and `test_startup_schedules_one_refresh_and_does_not_await_it` walks
`main.py`'s AST to confirm nothing awaits it.

`needs_refresh` returns true for *any* reason the reader refuses the file, not
only age — a malformed or self-contradictory snapshot will not repair itself,
and leaving it would keep the scanner dark until somebody looked.

## 10. Failure behaviour

`spot_market.refresh.refresh()` **never raises.** It returns a `RefreshResult`
with a status: `built`, `skipped_fresh`, `locked`, `failed`, `timeout`.

| failure | handling |
|---|---|
| network / timeout / HTTP error | the client's own backoff, then a non-zero exit; previous snapshot preserved |
| malformed venue response | build raises inside the builder → exit 1, one legible line |
| partial asset failure | the symbol is dropped, the build continues; BTC missing is fatal (it is the benchmark) |
| rate limiting | `SpotClient` retries with growing delay; exhaustion is an ordinary failure |
| empty universe | `SnapshotBuildError` — refused rather than published empty |
| build exceeds `SPOT_SNAPSHOT_BUILD_TIMEOUT_SECONDS` (600) | killed **and reaped**; an unreaped child would hold the lock for the life of the bot and every later slot would skip |
| builder cannot even start | logged, preserved |
| **builder exits 0 with an unreadable result** | treated as a failure. Exit 0 is the builder's opinion; the refresher reads the file back through the same fail-closed loader the Telegram screens use |
| another build running | exit 2, skipped, no error |

The builder's own `main()` catches broadly on purpose. This is an operational
entry point, and every way a build can fail has to end the same way: exit 1,
one legible line, previous snapshot untouched. (Found while writing the tests
— a `SpotDatasetError` from the panel loader was escaping as a traceback. The
file was still preserved, but a caller had to parse a stack trace to learn it.)

**A refresh failure never escalates to the user and never reaches unrelated
features.** No Telegram broadcast, no exception into the shared event loop. If
the snapshot eventually ages past the reader's limits, the spot scanner goes
dark on its own while everything else keeps running. **No freshness threshold
was relaxed** — `test_freshness_thresholds_were_not_relaxed_by_this_stage`
pins 48 h / 24 h.

## 11. Observability

The builder logs `refresh started` / `refresh finished` with duration,
`data_asof`, eligible and screened counts, the top exclusion reasons, the BTC
regime and the destination path. Failures print `FAILED — <type>: <detail>;
previous snapshot preserved` to stderr; skips print `SKIPPED`. No per-symbol
logging.

The runtime logs one line per outcome and records the last attempt in
`bot_data["spot_snapshot_refresh"]` (status, duration, detail, asset count)
plus `spot_snapshot_last_success`.

Real output, from a live run during verification:

```
INFO spot_market.refresh: spot snapshot: still fresh, no rebuild (assets=34)
INFO spot_market.refresh: spot snapshot: refresh started (source=live, dest=…/data/spot/snapshot)
INFO spot_market.refresh: spot snapshot: refresh finished in 22.0s (assets=34, data_asof=1787615999999, …)
ERROR spot_market.refresh: spot snapshot: could not start the builder: [Errno 2] …; previous snapshot preserved
```

## 12. `/status`

One line, two at most, added without redesigning anything:

```
🪙 Спот-снимок: OK · 34 монеты · обновлён 1 ч 12 мин назад
```
```
🪙 Спот-снимок: НЕДОСТУПЕН (данные устарели) — сканер отключён
   └ последнее обновление: failed (18 мин назад)
```

Answered through `bot/spot_screener.snapshot_health()`, i.e. the same
fail-closed reader the screens use, so /status cannot call a snapshot healthy
that the scanner is refusing to show. Wrapped so that a broken snapshot can
never break `/status` — the spot section is one feature among several. The
line is omitted entirely when there is nothing to say. No research internals.

## 13. Railway configuration

**Nothing to configure.** No new service, no cron entry, no volume, no
`railway.json`. The existing `worker: python main.py` picks the job up on the
next deploy.

Optional environment variables, documented by name in `.env.example`, all with
working defaults:

| variable | default |
|---|---|
| `ENABLE_SPOT_SNAPSHOT_REFRESH` | `true` |
| `SPOT_SNAPSHOT_REFRESH_HOURS` | `2` |
| `SPOT_SNAPSHOT_BUILD_TIMEOUT_SECONDS` | `600` |

**No credentials.** Binance's public spot endpoints only — no API key, no
secret, no signed request. `test_no_credentials_are_needed_or_referenced_by_the_refresh_path`
scans the refresh path for credential-shaped names. No secret value appears in
any file or log line.

Memory: the builder holds ~470 frames of 400 daily bars — on the order of tens
of MB, transient, in a subprocess that then exits. It does not accumulate in
the bot process.

## 14. Tests

**2203 passed** (2151 before S4A.1; +52). New file
`tests/test_spot_refresh.py` (51); `tests/test_spot_screener.py` gained the
import-boundary tests. Every pre-existing test still passes.

Failure paths are tested with **real subprocesses against real files** — a
mock cannot demonstrate that a killed builder left a byte-identical snapshot.
Nothing in the suite touches the network: the builder runs in `--source panel`
mode against a synthetic six-symbol panel, two of whose pairs are delisted
(the S1 loader refuses a survivors-only panel, and a fixture that dodged that
rule would be testing a loader production does not use).

| requirement | test |
|---|---|
| successful refresh publishes | `test_the_manual_cli_builds_and_publishes_a_snapshot` |
| **failed refresh preserves the old snapshot** | `…_leaves_the_previous_snapshot_byte_identical`, `…_an_interrupted_build_…` (real SIGKILL) |
| invalid result is never published | `test_a_builder_that_exits_zero_with_an_unreadable_result_is_a_failure` |
| atomic **and durable** replacement | `test_the_publish_is_an_atomic_rename_and_a_durable_one` (AST: replace + fsync + dir fsync) |
| no scratch file survives | `test_a_killed_build_leaves_no_scratch_file_behind`, `…_an_interrupted_publish_removes_its_own_…` |
| a corrupt snapshot raises one type only | `…_cannot_raise_anything_but_snapshot_unavailable` (5 shapes) |
| a corrupt snapshot does not escape the job | `test_a_corrupt_snapshot_does_not_escape_the_scheduler_job` |
| a mistyped cadence cannot stop startup | `test_a_mistyped_cadence_cannot_stop_the_bot_from_starting`, `…_is_clamped_…` |
| overlapping refresh prevented | `test_two_builds_cannot_run_at_once`, `…_the_lock_is_released_when_its_holder_exits` |
| stale / missing stays fail-closed | `test_freshness_thresholds_were_not_relaxed_by_this_stage` + S4A's suite |
| startup with a fresh snapshot does not rebuild | `test_startup_with_a_fresh_snapshot_does_not_rebuild` |
| startup recovery when stale/missing | `test_startup_schedules_one_refresh_and_does_not_await_it` |
| refresh failure does not crash startup | `test_a_failed_refresh_does_not_raise_into_the_event_loop` |
| manual CLI still works | `test_the_manual_cli_builds_and_publishes_a_snapshot` |
| automated path uses the same builder | `test_the_automated_path_invokes_that_same_cli` |
| no S4A logic changed | `test_this_stage_changed_no_ranking_or_eligibility_rule` |
| no predictive language introduced | `test_the_refresh_layer_introduced_no_predictive_language` |
| no research in the runtime | `test_the_research_package_never_enters_the_bot_process` |
| deployment points at the right command | `test_the_deployment_runs_one_service_that_owns_the_snapshot`, `…_the_same_one_on_both_sides` |
| no secrets | `test_no_credentials_are_needed_or_referenced_by_the_refresh_path` |
| timeout is killed **and reaped** | `test_a_build_that_exceeds_its_timeout_is_killed_and_reaped` |

Verified live, outside the suite: skip-when-fresh (0.0 s), a real live rebuild
(22.0 s, 34 assets), and a simulated failure leaving the snapshot
byte-identical.

## 15. Manual actions still required

**None.** Deploying the branch is sufficient.

## 16. Known limitations

1. **The snapshot does not survive a redeploy** — by design, see §6. Cost: one
   ~22–33 s rebuild at startup, during which the scanner shows its
   fail-closed screen.
2. **One process means one failure domain.** A bot restart loses the snapshot
   and the in-flight refresh. Acceptable because both are cheap to redo; a
   second service would trade this for a much worse problem.
3. **The refresh is invisible until `/status` is asked.** Nothing alerts the
   owner that refreshes have been failing for ten hours — the first visible
   sign is the scanner going dark at the 24 h mark. Deliberate: a Telegram
   alert for every spot API hiccup would be noise. If it should be louder,
   that is a product decision.
4. **`max_instances=1` and the file lock guard overlap, not duration.** A
   build that consistently takes longer than the cadence would skip every
   other slot rather than queueing. At 22 s against 2 h this is theoretical.
5. **No jitter.** Every deployment of this bot would fetch at the same minute.
   Irrelevant at one instance.
6. **The snapshot is not shared between environments.** A staging service
   builds its own. Correct, but worth knowing before assuming one refresh
   serves everything.

## 17. Audit round 1 — what it found

Codex returned `NOT_SAFE` on the first pass. Four real defects, all fixed:

1. **The reader's exception contract was not total.** `load_snapshot` promised
   `SnapshotUnavailable`, but a coin entry that was a string reached
   `raw.get(...)` and left an `AttributeError` — which sails past every
   caller: into a Telegram handler as a silent failure, and out of the
   scheduler's job into the shared event loop. Valid JSON of the wrong shape
   is exactly what a half-finished or hand-edited file looks like. Fixed with
   an explicit type check plus a `_guard` backstop, so the one exception type
   the callers catch really is the only one they can receive. Five shapes are
   now regression-tested.
2. **The publish was atomic but not durable** — no fsync on the file, none on
   the directory, and a fixed temp name with no cleanup. §7.
3. **A mistyped cadence could stop the whole bot from starting.**
   `SPOT_SNAPSHOT_REFRESH_HOURS=0` becomes `hour="*/0"`, which APScheduler
   rejects — inside `build_scheduler()`, which runs before the bot finishes
   starting. A spot setting would have taken the futures streams, the alerts
   and the journal down with it. Now clamped to 1–24 in config, and the job
   registration is guarded so an unreachable failure still costs only the
   scanner.
4. **Three tests asserted less than they claimed** — the atomicity test would
   have passed against the missing fsyncs, and the event-loop test mocked the
   very function whose real failure it was supposed to catch. Replaced with
   tests that exercise the real paths against real files.

## 18. Audit round 2

Round 2 returned `NOT_SAFE` again, on three narrower points. All fixed:

1. **The cadence ceiling was itself invalid.** Round 1's clamp allowed 24, and
   APScheduler rejects `hour="*/24"` — so a documented, permitted value would
   have been caught by the round-1 guard and silently disabled the refresh
   entirely. Ceiling is now 12, and the test constructs the real
   `CronTrigger` for every allowed value instead of asserting a number. The
   earlier test was named "clamped to something cron accepts" and never asked
   cron; that is the kind of test that is worse than none.
2. **`_fsync_dir` swallowed its failure silently.** The behaviour is right —
   publishing anyway is correct on a filesystem that cannot fsync a directory
   — but the silence was not. It now reports, and returns whether it happened.
3. **The orphan sweep could delete a scratch file still being written.**
   `write_snapshot` does not hold the build lock, so "delete every temp file
   you find" was unsafe for a case the docstring itself had raised. Only files
   older than an hour are swept now.

And two tests were renamed or replaced because they claimed more than they
did: the scratch-file test never killed anything (the kill case lives in the
interrupted-build test, which now checks scratch files too), and the cadence
test described above.

`parse_snapshot` also now type-checks its own argument, so the
single-exception contract holds when it is called directly and not only
through `load_snapshot`.

Two smaller things came out of round 1: `refresh()` and `needs_refresh()` now
resolve the canonical path at call time instead of binding it as a default
argument (correct in production, untestable everywhere else), and the
builder's `main()` catches broadly so no failure mode escapes as a traceback.

---

Unchanged and not started: predictive S4, S3b, `CLEAN_2X`, futures Phase A′
(still paused, G1 still `G1_INDETERMINATE`).
