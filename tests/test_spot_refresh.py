"""S4A.1 — the automated snapshot refresh.

One property matters more than the rest and most of this file is about it:
**a refresh that fails must leave the previous snapshot exactly where it was.**
The scanner already knows how to go dark when its data ages out. What it
cannot survive is a refresher that truncates the file it is about to rewrite
and then loses its network connection — that turns a temporary outage into a
permanent one, and it does so silently.

So the failure paths are tested with real subprocesses against real files,
not with mocks of them. A mock cannot demonstrate that an interrupted build
left a byte-identical snapshot behind.

Nothing here reaches the network: every subprocess runs the builder in
`--source panel` mode against a synthetic panel on disk, or is made to fail
before it gets that far.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import scheduler as sched
from spot_market import refresh as spot_refresh
from spot_market.snapshot import MAX_SNAPSHOT_AGE_HOURS
from tools.spot_snapshot import (EXIT_FAILED, EXIT_LOCKED, EXIT_OK, LOCK_NAME,
                                 SNAPSHOT_NAME, SnapshotLocked, build_lock)

REPO = Path(__file__).resolve().parents[1]
HOUR_MS = 3_600_000
DAY = 86_400_000


# --- a tiny real panel, so the builder can run for real ----------------------

def _frame(bars: int, ends_ms: int, *, close: float, drift: float,
           quote_volume: float) -> pd.DataFrame:
    rows = []
    first = ends_ms - bars * DAY
    price = close
    for i in range(bars):
        ot = first + i * DAY
        price *= (1.0 + drift)
        rows.append({"open_time": ot, "open": price, "high": price * 1.01,
                     "low": price * 0.99, "close": price, "volume": 1.0,
                     "close_time": ot + DAY - 1,
                     "quote_volume": quote_volume, "trades": 10})
    return pd.DataFrame(rows)


@pytest.fixture
def panel(tmp_path):
    """A four-symbol S1-shaped panel the builder's `--source panel` accepts."""
    from tools.spot_cache import write_symbol

    outdir = tmp_path / "data"
    klines = outdir / "klines"
    klines.mkdir(parents=True)
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    # Bars end at the last midnight so the snapshot is fresh by construction.
    ends = (now_ms // DAY) * DAY

    # Two of the six no longer trade. Not decoration: `load_panel` refuses a
    # panel whose delisted share has collapsed, because a survivors-only panel
    # is what a survivorship-biased universe looks like from the inside. A
    # fixture that dodged that rule would be testing a loader production does
    # not use.
    symbols = [("BTCUSDT", "BTC", 0.0005, 6e8, True),
               ("AAAUSDT", "AAA", 0.0002, 2e7, True),
               ("BBBUSDT", "BBB", -0.0003, 3e7, True),
               ("CCCUSDT", "CCC", 0.0001, 4e7, True),
               ("DEADUSDT", "DEAD", 0.0001, 5e7, False),
               ("GONEUSDT", "GONE", 0.0001, 5e7, False)]
    records = []
    for symbol, base, drift, volume, alive in symbols:
        status = "TRADING" if alive else "BREAK"
        spine = {"symbol": symbol, "status": status, "trading_now": alive}
        # The dead pairs stop a year ago, exactly as a delisted pair does.
        last = ends if alive else ends - 365 * DAY
        write_symbol(_frame(400, last, close=100.0, drift=drift,
                            quote_volume=volume),
                     str(klines), symbol=symbol, spine=spine,
                     duplicates_removed=0)
        records.append({"symbol": symbol, "base_asset": base,
                        "quote_asset": "USDT", "status": status,
                        "trading_now": alive})
    (outdir / "symbols.json").write_text(
        json.dumps({"stage": "S1", "symbols": records}), encoding="utf-8")
    return outdir


@pytest.fixture
def snapdir(tmp_path):
    d = tmp_path / "snapshot"
    d.mkdir()
    return d


def run_builder(panel_dir, snap_dir, *extra) -> subprocess.CompletedProcess:
    """The CLI, exactly as an operator or the scheduler would invoke it."""
    return subprocess.run(
        [sys.executable, "-m", "tools.spot_snapshot", "--source", "panel",
         "--outdir", str(panel_dir), "--snapshot-dir", str(snap_dir), *extra],
        cwd=str(REPO), capture_output=True, text=True, timeout=300)


def digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# --- the manual CLI still works, and it is the same one ----------------------

def test_the_manual_cli_builds_and_publishes_a_snapshot(panel, snapdir):
    proc = run_builder(panel, snapdir)
    assert proc.returncode == EXIT_OK, proc.stderr
    payload = json.loads((snapdir / SNAPSHOT_NAME).read_text())
    assert payload["schema"] == "spot.snapshot/1"
    assert payload["counts"]["eligible"] == 4
    assert "refresh started" in proc.stdout
    assert "refresh finished" in proc.stdout


def test_the_automated_path_invokes_that_same_cli(snapdir, monkeypatch):
    """One builder, two invocation paths — asserted, not asserted about.

    The refresher names the builder module; it does not reimplement it. If a
    second implementation ever appears, this is the test that has to be
    deleted to make it pass.
    """
    assert spot_refresh.BUILDER_MODULE == "tools.spot_snapshot"
    seen: dict = {}

    async def capture(*argv, **kw):
        seen["argv"] = list(argv)
        seen["cwd"] = kw.get("cwd")

        class _P:
            returncode = EXIT_OK

            async def communicate(self):
                return b"", b""
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    asyncio.run(spot_refresh.refresh(str(snapdir / SNAPSHOT_NAME), force=True))

    assert seen["argv"][1:3] == ["-m", "tools.spot_snapshot"]
    assert "--source" in seen["argv"] and "live" in seen["argv"]
    # The destination is passed explicitly rather than inherited from the
    # working directory: the bot and the builder must not be able to disagree
    # about which file is the canonical one.
    assert seen["argv"][seen["argv"].index("--snapshot-dir") + 1] == \
        str(snapdir)
    assert seen["cwd"] == spot_refresh.REPO_ROOT


# --- a successful refresh publishes -----------------------------------------

def test_a_successful_refresh_publishes_a_readable_snapshot(panel, snapdir):
    proc = run_builder(panel, snapdir)
    assert proc.returncode == EXIT_OK, proc.stderr

    result = asyncio.run(spot_refresh.refresh(str(snapdir / SNAPSHOT_NAME)))
    assert result.status == "skipped_fresh"
    assert result.ok
    assert result.eligible == 4  # the two delisted pairs are excluded


# --- a failed refresh preserves the previous snapshot ------------------------

def test_a_failed_refresh_leaves_the_previous_snapshot_byte_identical(
        panel, snapdir, tmp_path):
    """The defining requirement of this stage.

    The build is made to fail the way a real one does — the data it needs is
    not there — after the previous snapshot has already been published. The
    old file must be untouched, not merely present: a truncated or partially
    rewritten file would still "exist".
    """
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    before = digest(snapdir / SNAPSHOT_NAME)
    assert before is not None

    empty = tmp_path / "empty"
    (empty / "klines").mkdir(parents=True)
    (empty / "symbols.json").write_text(json.dumps({"symbols": []}))
    proc = run_builder(empty, snapdir)

    assert proc.returncode == EXIT_FAILED
    assert "FAILED" in proc.stderr
    assert "previous snapshot preserved" in proc.stderr
    assert digest(snapdir / SNAPSHOT_NAME) == before


def test_an_interrupted_build_leaves_the_previous_snapshot_intact(panel,
                                                                  snapdir):
    """A killed process, not a raised exception. There is no `finally` here.

    The guarantee comes from ordering — the write is the last thing that
    happens inside the lock — so it has to hold when the process simply stops
    existing mid-build.
    """
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    before = digest(snapdir / SNAPSHOT_NAME)

    proc = subprocess.Popen(
        [sys.executable, "-m", "tools.spot_snapshot", "--source", "panel",
         "--outdir", str(panel), "--snapshot-dir", str(snapdir)],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.kill()
    proc.communicate()

    assert digest(snapdir / SNAPSHOT_NAME) == before
    assert not list(snapdir.glob("*.tmp"))
    assert not [p for p in snapdir.iterdir()
                if p.name.startswith(".current.json.")]


def test_a_refresh_whose_builder_cannot_start_reports_it_and_preserves(
        panel, snapdir):
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    before = digest(snapdir / SNAPSHOT_NAME)
    result = asyncio.run(spot_refresh.refresh(
        str(snapdir / SNAPSHOT_NAME), force=True,
        python="/nonexistent/python"))
    assert result.status == "failed"
    assert result.previous_preserved
    assert digest(snapdir / SNAPSHOT_NAME) == before


def test_a_build_that_exceeds_its_timeout_is_killed_and_reaped(panel, snapdir,
                                                               monkeypatch):
    """A hung socket must cost one cycle, not every cycle after it.

    The kill alone is not enough: an unreaped child keeps the build lock for
    the life of the bot process, and every later slot would then skip. So the
    test asserts the refresher waits for the corpse as well as producing one.
    """
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    before = digest(snapdir / SNAPSHOT_NAME)
    events: list[str] = []

    async def hang(*argv, **kw):
        class _P:
            returncode = None

            async def communicate(self):
                if "killed" in events:
                    return b"", b"killed"
                await asyncio.sleep(60)
                return b"", b""       # pragma: no cover

            def kill(self):
                events.append("killed")
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", hang)
    result = asyncio.run(spot_refresh.refresh(
        str(snapdir / SNAPSHOT_NAME), force=True, timeout_s=1))

    assert result.status == "timeout"
    assert events == ["killed"]
    assert result.previous_preserved
    assert digest(snapdir / SNAPSHOT_NAME) == before


# --- an invalid result is never published -----------------------------------

def test_a_builder_that_exits_zero_with_an_unreadable_result_is_a_failure(
        panel, snapdir, monkeypatch):
    """Exit code 0 is the builder's opinion, not a fact.

    The refresher reads the result back through the same fail-closed loader
    the Telegram screens use. A snapshot the product cannot read is not a
    successful refresh, whatever the process claimed on its way out.
    """
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    (snapdir / SNAPSHOT_NAME).write_text("{ not json", encoding="utf-8")

    async def fake_exec(*argv, **kw):
        class _P:
            returncode = EXIT_OK

            async def communicate(self):
                return b"", b""
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(spot_refresh.refresh(
        str(snapdir / SNAPSHOT_NAME), force=True))
    assert result.status == "failed"
    assert "unreadable" in result.detail


def test_the_publish_is_an_atomic_rename_and_a_durable_one():
    """Asserted on the code, because the races it prevents cannot be staged.

    Two separate properties, and an audit caught the second missing:

      * a handler reading at the instant of the replace must see the whole old
        file or the whole new one — `os.replace`, never a rewrite in place;
      * a host that dies just after the rename must not come back to a
        directory entry pointing at unflushed content. That needs fsync on the
        file, and fsync on the directory for the rename itself.
    """
    src = (REPO / "tools" / "spot_snapshot.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "write_snapshot")
    calls = [n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "replace" in calls
    assert "fsync" in calls, "the new snapshot is renamed into place unflushed"
    assert "flush" in calls
    # The directory fsync lives in its own helper; the rename is not durable
    # without it on most filesystems.
    assert "_fsync_dir" in src
    dir_fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_fsync_dir")
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", None)
               == "fsync" for n in ast.walk(dir_fn))
    # Cleanup unlinks scratch files. It must never touch the live one: the
    # canonical path is only ever *replaced*, never removed and rewritten.
    assert "truncate" not in calls
    unlinked = [ast.dump(n.args[0]) for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "unlink" and n.args]
    assert all("tmp" in u for u in unlinked), unlinked
    sweep = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "_sweep_stale_temps")
    assert "TMP_PREFIX" in ast.dump(sweep), (
        "the sweep must be scoped to scratch files by prefix")


def test_a_successful_build_leaves_no_scratch_file_and_sweeps_old_orphans(
        panel, snapdir):
    """Two claims, and the earlier version of this test named a third it did
    not test — it never killed anything. The kill case is
    `test_an_interrupted_build_leaves_the_previous_snapshot_intact`, which now
    checks scratch files too.
    """
    import os
    import time

    assert run_builder(panel, snapdir).returncode == EXIT_OK
    assert [p for p in snapdir.iterdir()
            if p.name.startswith(".current.json.")] == []

    orphan = snapdir / ".current.json.orphan"
    orphan.write_text("half a json", encoding="utf-8")
    old_enough = time.time() - 2 * builder_module().TMP_ORPHAN_AGE_S
    os.utime(orphan, (old_enough, old_enough))
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    assert not orphan.exists()


def builder_module():
    from tools import spot_snapshot

    return spot_snapshot


def test_the_sweep_never_deletes_a_scratch_file_still_being_written(snapdir):
    """`write_snapshot` does not hold the build lock itself.

    The lock makes a second writer unlikely, not impossible, and a sweep that
    deleted every scratch file it found would delete one another writer was in
    the middle of. Age is what separates "abandoned" from "in flight".
    """
    in_flight = snapdir / ".current.json.someone-elses"
    in_flight.write_text("being written right now", encoding="utf-8")
    builder_module()._sweep_stale_temps(str(snapdir))
    assert in_flight.exists()


def test_an_interrupted_publish_removes_its_own_scratch_file(snapdir,
                                                             monkeypatch):
    from tools import spot_snapshot as builder

    def boom(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(builder.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        builder.write_snapshot({"schema": "x"}, str(snapdir))
    assert not [p for p in snapdir.iterdir()
                if p.name.startswith(".current.json.")]


# --- concurrency -------------------------------------------------------------

def test_two_builds_cannot_run_at_once(panel, snapdir):
    """The lock covers every invocation path, including a manual one.

    The overlap that will actually happen is not two scheduler slots — those
    are two hours apart — but an operator running the CLI while the job is
    mid-build.
    """
    with build_lock(str(snapdir)):
        proc = run_builder(panel, snapdir)
    assert proc.returncode == EXIT_LOCKED
    assert "SKIPPED" in proc.stderr
    assert "previous snapshot preserved" in proc.stderr


def test_the_lock_is_released_when_its_holder_exits(panel, snapdir, tmp_path):
    """`flock`, not a pid file: a killed builder must not block every later run.

    This is the whole reason for the mechanism. A pid file survives SIGKILL
    and then jams every subsequent refresh until a human notices — a worse
    failure than the overlap it was meant to prevent.
    """
    import time

    holder = tmp_path / "holder.py"
    holder.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from tools.spot_snapshot import build_lock\n"
        f"with build_lock({str(snapdir)!r}):\n"
        "    print('held', flush=True)\n"
        "    time.sleep(60)\n", encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(holder)], cwd=str(REPO),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == b"held"
        # While it is held, a build refuses instead of waiting.
        assert run_builder(panel, snapdir).returncode == EXIT_LOCKED
    finally:
        proc.kill()
        proc.communicate(timeout=10)

    # The kernel released it when the holder died.
    for _ in range(100):
        if run_builder(panel, snapdir).returncode == EXIT_OK:
            break
        time.sleep(0.05)
    else:  # pragma: no cover
        pytest.fail("the lock outlived the process that held it")


def test_the_lock_raises_rather_than_waiting():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with build_lock(d):
            with pytest.raises(SnapshotLocked):
                with build_lock(d):
                    pass  # pragma: no cover


# --- freshness decisions -----------------------------------------------------

def test_a_fresh_snapshot_is_not_rebuilt(panel, snapdir):
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    assert spot_refresh.needs_refresh(str(snapdir / SNAPSHOT_NAME)) is False
    result = asyncio.run(spot_refresh.refresh(str(snapdir / SNAPSHOT_NAME)))
    assert result.status == "skipped_fresh"


def test_a_missing_snapshot_needs_a_refresh(snapdir):
    assert spot_refresh.needs_refresh(str(snapdir / SNAPSHOT_NAME)) is True


def test_a_snapshot_past_the_refresh_window_needs_one(panel, snapdir):
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    path = str(snapdir / SNAPSHOT_NAME)
    payload = json.loads((snapdir / SNAPSHOT_NAME).read_text())
    later = payload["generated_at_ms"] + spot_refresh.REFRESH_AFTER_HOURS * HOUR_MS
    assert spot_refresh.needs_refresh(path, now_ms=later) is True
    assert spot_refresh.needs_refresh(path, now_ms=later - HOUR_MS) is False


def test_an_unreadable_snapshot_needs_a_refresh_whatever_the_reason(snapdir):
    """Not only age. A malformed file will not repair itself, and leaving it
    in place would keep the scanner dark until somebody looked."""
    (snapdir / SNAPSHOT_NAME).write_text("{}", encoding="utf-8")
    assert spot_refresh.needs_refresh(str(snapdir / SNAPSHOT_NAME)) is True


def test_the_refresh_window_sits_well_inside_the_fail_closed_limit():
    """The refresher must act long before the scanner would go dark.

    Several consecutive failures have to be survivable, which is only true if
    the two thresholds are far apart.
    """
    assert spot_refresh.REFRESH_AFTER_HOURS < MAX_SNAPSHOT_AGE_HOURS
    assert MAX_SNAPSHOT_AGE_HOURS / spot_refresh.REFRESH_AFTER_HOURS >= 4


def test_freshness_thresholds_were_not_relaxed_by_this_stage():
    """S4A.1 is infrastructure. Weakening a limit to hide a refresh that did
    not happen would make the scanner lie instead of going quiet."""
    from spot_market.snapshot import MAX_BAR_AGE_HOURS
    assert MAX_BAR_AGE_HOURS == 48
    assert MAX_SNAPSHOT_AGE_HOURS == 24


# --- scheduler and startup wiring -------------------------------------------

def test_the_refresh_job_is_registered_on_the_expected_cadence(monkeypatch):
    import config

    monkeypatch.setattr(config, "ENABLE_SPOT_SNAPSHOT_REFRESH", True)
    monkeypatch.setattr(config, "SPOT_SNAPSHOT_REFRESH_HOURS", 2)
    built = sched.build_scheduler(_FakeApp())
    job = built.get_job("spot_snapshot")
    assert job is not None
    assert job.func is sched.spot_snapshot_job
    # Cron, not interval: the cadence must survive a redeploy rather than
    # restarting from "now" every time the container comes back.
    assert str(job.trigger).startswith("cron")
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["hour"] == "*/2"
    assert fields["minute"] == "7"
    assert job.coalesce is True
    assert job.max_instances == 1


def test_a_mistyped_cadence_cannot_stop_the_bot_from_starting(monkeypatch):
    """`hour="*/0"` is rejected by APScheduler, and build_scheduler() runs
    before the bot finishes starting. A typo in a spot setting must cost the
    spot scanner, not the futures streams, the alerts and the journal."""
    import config

    monkeypatch.setattr(config, "ENABLE_SPOT_SNAPSHOT_REFRESH", True)
    monkeypatch.setattr(config, "SPOT_SNAPSHOT_REFRESH_HOURS", 0)
    built = sched.build_scheduler(_FakeApp())      # must not raise
    assert built.get_job("spot_snapshot") is None
    assert built.get_job("price_alerts") is not None


def test_the_configured_cadence_is_clamped_to_something_cron_accepts():
    """The clamp is only worth having if the clamped value actually works.

    An audit caught the earlier version asserting a ceiling of 24 — which
    APScheduler rejects, because a step above 23 exceeds the hour field's
    range. The test said "something cron accepts" and never asked cron.
    """
    import importlib
    import os

    from apscheduler.triggers.cron import CronTrigger

    import config

    for raw, expected in (("0", 1), ("-3", 1), ("99", 12), ("24", 12),
                          ("3", 3), ("12", 12)):
        os.environ["SPOT_SNAPSHOT_REFRESH_HOURS"] = raw
        try:
            importlib.reload(config)
            hours = config.SPOT_SNAPSHOT_REFRESH_HOURS
            assert hours == expected, raw
            # The whole point: this must not raise.
            CronTrigger(timezone="UTC", minute=7, hour=f"*/{hours}")
        finally:
            os.environ.pop("SPOT_SNAPSHOT_REFRESH_HOURS", None)
    importlib.reload(config)


def test_the_cadence_ceiling_keeps_the_scanner_inside_its_own_limit():
    """A cadence past 12h could not hold the snapshot under the reader's 24h
    limit with any margin — the scanner would go dark between refreshes by
    design. The ceiling is that constraint, not only cron's."""
    import config

    assert config.SPOT_SNAPSHOT_REFRESH_MAX_HOURS == 12
    assert config.SPOT_SNAPSHOT_REFRESH_MAX_HOURS * 2 <= MAX_SNAPSHOT_AGE_HOURS


def test_the_refresh_job_can_be_switched_off(monkeypatch):
    import config

    monkeypatch.setattr(config, "ENABLE_SPOT_SNAPSHOT_REFRESH", False)
    assert sched.build_scheduler(_FakeApp()).get_job("spot_snapshot") is None


class _FakeApp:
    def __init__(self) -> None:
        self.bot_data: dict = {}


def test_a_failed_refresh_does_not_raise_into_the_event_loop(monkeypatch):
    """The futures streams share this loop. A spot outage is not their problem."""
    async def failing(*a, **kw):
        return spot_refresh.RefreshResult(
            status="failed", detail="binance unreachable", duration_s=1.0,
            previous_preserved=True)

    monkeypatch.setattr(sched.spot_refresh, "refresh", failing)
    app = _FakeApp()
    asyncio.run(sched.spot_snapshot_job(app))
    state = app.bot_data["spot_snapshot_refresh"]
    assert state["status"] == "failed"
    assert "spot_snapshot_last_success" not in app.bot_data


@pytest.mark.parametrize("garbage", [
    '{"schema": "spot.snapshot/1", "generated_at_ms": 1, "data_asof_ms": 1, '
    '"coins": ["oops"], "btc": {}}',
    '{"schema": "spot.snapshot/1", "generated_at_ms": "x", '
    '"data_asof_ms": 1, "coins": [{}], "btc": {}}',
    '{"coins": null}',
    'null',
    'not json at all',
])
def test_a_corrupt_snapshot_cannot_raise_anything_but_snapshot_unavailable(
        snapdir, garbage):
    """The hole an audit found: the reader caught only what it had thought of.

    A coin entry that is a string reached `raw.get` and left an AttributeError,
    which sails straight past every caller — into a Telegram handler as a
    silent failure, and out of the scheduler's job into the shared event loop.
    Valid JSON of the wrong shape is exactly what a half-finished or
    hand-edited file looks like.
    """
    from spot_market.snapshot import SnapshotUnavailable

    from spot_market.snapshot import load_snapshot

    path = snapdir / SNAPSHOT_NAME
    path.write_text(garbage, encoding="utf-8")
    # The loader itself must already convert it: everything downstream catches
    # this one type and nothing else.
    with pytest.raises(SnapshotUnavailable):
        load_snapshot(str(path))
    # And every consumer stays quiet.
    assert spot_refresh.needs_refresh(str(path)) is True
    snap, reason = spot_refresh.read_snapshot(str(path))
    assert snap is None and reason


def test_a_corrupt_snapshot_does_not_escape_the_scheduler_job(snapdir,
                                                              monkeypatch):
    """The real path, not a mocked one: a corrupt file on disk, the real job."""
    path = snapdir / SNAPSHOT_NAME
    path.write_text('{"schema": "spot.snapshot/1", "coins": ["oops"], '
                    '"generated_at_ms": 1, "data_asof_ms": 1, "btc": {}}',
                    encoding="utf-8")
    monkeypatch.setattr(spot_refresh, "DEFAULT_SNAPSHOT_PATH", str(path))

    async def no_build(*argv, **kw):
        class _P:
            returncode = EXIT_FAILED

            async def communicate(self):
                return b"", b"no network"
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_build)
    app = _FakeApp()
    asyncio.run(sched.spot_snapshot_job(app))   # must not raise
    assert app.bot_data["spot_snapshot_refresh"]["status"] == "failed"


def test_a_refresh_that_somehow_raises_is_still_contained(monkeypatch):
    async def explode(*a, **kw):
        raise RuntimeError("something nobody enumerated")

    monkeypatch.setattr(sched.spot_refresh, "refresh", explode)
    app = _FakeApp()
    asyncio.run(sched.spot_snapshot_job(app))   # must not raise
    assert app.bot_data["spot_snapshot_refresh"]["status"] == "failed"


def test_a_successful_refresh_records_its_success(monkeypatch):
    async def ok(*a, **kw):
        return spot_refresh.RefreshResult(
            status="built", detail="ok", duration_s=33.0,
            previous_preserved=True, eligible=34)

    monkeypatch.setattr(sched.spot_refresh, "refresh", ok)
    app = _FakeApp()
    asyncio.run(sched.spot_snapshot_job(app))
    assert app.bot_data["spot_snapshot_refresh"]["assets"] == 34
    assert "spot_snapshot_last_success" in app.bot_data


def test_startup_schedules_one_refresh_and_does_not_await_it():
    """Startup must not block on Binance, and must not skip the gap.

    Railway's filesystem is ephemeral, so a redeploy begins with no snapshot
    at all; without this the scanner would stay dark until the next slot.
    """
    src = (REPO / "main.py").read_text(encoding="utf-8")
    assert "initial_spot_snapshot" in src
    assert "ENABLE_SPOT_SNAPSHOT_REFRESH" in src
    tree = ast.parse(src)
    awaited = [n for n in ast.walk(tree)
               if isinstance(n, ast.Await)
               and "spot" in ast.dump(n.value).lower()]
    assert not awaited, "startup must not await the spot refresh"


def test_startup_with_a_fresh_snapshot_does_not_rebuild(panel, snapdir):
    """The startup job is the same job; freshness is what makes it cheap."""
    assert run_builder(panel, snapdir).returncode == EXIT_OK
    result = asyncio.run(spot_refresh.refresh(str(snapdir / SNAPSHOT_NAME)))
    assert result.status == "skipped_fresh"
    assert result.duration_s < 5


# --- /status -----------------------------------------------------------------

def test_status_reports_the_snapshot_through_the_same_fail_closed_reader(
        tmp_path, monkeypatch):
    from bot import spot_screener as ss

    path = tmp_path / "current.json"
    monkeypatch.setattr(ss.snap_mod, "DEFAULT_SNAPSHOT_PATH", str(path))
    health = ss.snapshot_health()
    assert health["ok"] is False
    assert health["reason"] == "missing"


def test_the_status_line_says_the_scanner_is_off_when_the_snapshot_is_not_ok():
    from bot import formatting

    text = formatting.format_status({
        "dry_run": False, "db_connected": True, "alert_chats": 1,
        "signal_tf": "4H", "fast_tf": "15m",
        "spot_snapshot": {"ok": False, "reason": "stale_data"},
        "spot_refresh": {"status": "failed", "at": None},
    })
    assert "НЕДОСТУПЕН" in text and "сканер отключён" in text
    assert "последнее обновление: failed" in text


def test_the_status_line_is_absent_when_there_is_nothing_to_say():
    from bot import formatting

    text = formatting.format_status({
        "dry_run": False, "db_connected": True, "alert_chats": 1,
        "signal_tf": "4H", "fast_tf": "15m"})
    assert "Спот-снимок" not in text


def test_a_broken_snapshot_never_breaks_status(monkeypatch):
    import bot.handlers as h

    def boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(h.spot_screener, "snapshot_health", boom)
    assert h._spot_health() is None


# --- deployment configuration ------------------------------------------------

def test_the_deployment_runs_one_service_that_owns_the_snapshot():
    """The persistence argument, asserted where it can rot.

    A second Railway service would have its own filesystem and would write a
    snapshot this bot could never read. One `worker` process is what makes the
    shared-file design correct, so a Procfile that grows a second process type
    has to come back through this test.
    """
    procfile = (REPO / "Procfile").read_text(encoding="utf-8").strip()
    assert procfile == "worker: python main.py"
    assert not (REPO / "railway.json").exists()
    assert not (REPO / "Dockerfile").exists()


def test_the_snapshot_path_is_the_same_one_on_both_sides():
    from spot_market.snapshot import DEFAULT_SNAPSHOT_PATH
    from tools.spot_snapshot import DEFAULT_SNAPSHOT_DIR, SNAPSHOT_NAME

    assert DEFAULT_SNAPSHOT_PATH == os.path.join(DEFAULT_SNAPSHOT_DIR,
                                                 SNAPSHOT_NAME)


def test_no_credentials_are_needed_or_referenced_by_the_refresh_path():
    """Binance's public spot endpoints only. No key, no secret, no signature."""
    for rel in ("tools/spot_snapshot.py", "tools/spot_client.py",
                "spot_market/refresh.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        low = src.lower()
        # Credential-shaped names only. "token" is excluded on purpose:
        # `is_leveraged_token` is a universe rule, not a secret.
        for banned in ("api_key", "apikey", "api_secret", "secret",
                       "x-mbx-apikey", "bearer", "signature="):
            assert banned not in low, f"{rel}: {banned}"


def test_the_new_settings_are_documented_by_name_only():
    env = (REPO / ".env.example").read_text(encoding="utf-8")
    for name in ("ENABLE_SPOT_SNAPSHOT_REFRESH", "SPOT_SNAPSHOT_REFRESH_HOURS",
                 "SPOT_SNAPSHOT_BUILD_TIMEOUT_SECONDS"):
        assert name in env, name


# --- nothing about S4A's product logic moved ---------------------------------

def test_this_stage_changed_no_ranking_or_eligibility_rule():
    """S4A.1 is infrastructure. The screener's semantics are frozen here."""
    from spot_market.views import SHORTLIST_SIZE, VIEWS
    from tools.spot_snapshot import (ACTIVITY_WINDOW_DAYS, MAX_LAST_BAR_AGE_MS,
                                     MIN_BARS, MIN_LISTING_AGE_DAYS,
                                     MIN_MEDIAN_DOLLAR_VOLUME)

    assert [v.field for v in VIEWS] == [
        "rel_btc_90d", "median_quote_volume_30d", "drawdown_180d",
        "relative_volume"]
    assert SHORTLIST_SIZE == 8
    assert (MIN_BARS, MIN_LISTING_AGE_DAYS) == (200, 180)
    assert MIN_MEDIAN_DOLLAR_VOLUME == 5_000_000.0
    assert MAX_LAST_BAR_AGE_MS == 48 * 3_600_000
    assert ACTIVITY_WINDOW_DAYS == 30


def test_the_refresh_layer_introduced_no_predictive_language():
    for rel in ("spot_market/refresh.py", "scheduler.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        low = src.lower()
        for banned in ("probability of", "2x", "expected return",
                       "alpha_score", "composite_score"):
            assert banned not in low, f"{rel}: {banned}"


# --- durability, verified against the syscalls themselves --------------------

def test_the_data_is_flushed_before_the_rename_and_the_rename_after(tmp_path):
    """Ordering, not just presence. `fsync` after `replace` would be useless.

    The file's contents must be on disk before the directory entry points at
    them, and the directory entry must be persisted after it moves. Asserted
    on the real call order because an AST check can only see that both appear.
    """
    import os
    import stat

    from tools import spot_snapshot as builder

    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        order.append(f"fsync-{kind}")
        return real_fsync(fd)

    def replace(a, b):
        order.append("replace")
        return real_replace(a, b)

    os.fsync, os.replace = fsync, replace
    try:
        builder.write_snapshot({"schema": "x"}, str(tmp_path))
    finally:
        os.fsync, os.replace = real_fsync, real_replace
    assert order == ["fsync-file", "replace", "fsync-dir"], order


def test_data_that_cannot_be_flushed_is_never_published(tmp_path):
    """A snapshot that cannot be made durable must not replace a good one.

    Refusing costs one cycle. Publishing content the kernel has not committed
    would mean a host that dies leaves a directory entry pointing at nothing —
    an atomic rename to an empty file.
    """
    import os
    import stat

    from tools import spot_snapshot as builder

    builder.write_snapshot({"schema": "old"}, str(tmp_path))
    before = digest(tmp_path / SNAPSHOT_NAME)
    real_fsync = os.fsync

    def file_fsync_fails(fd):
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("EIO")
        return real_fsync(fd)

    os.fsync = file_fsync_fails
    try:
        with pytest.raises(OSError):
            builder.write_snapshot({"schema": "new"}, str(tmp_path))
    finally:
        os.fsync = real_fsync

    assert digest(tmp_path / SNAPSHOT_NAME) == before
    assert not [p for p in tmp_path.iterdir()
                if p.name.startswith(".current.json.")]


def test_a_filesystem_that_cannot_fsync_a_directory_publishes_and_says_so(
        tmp_path, capsys):
    """Directory fsync is best-effort: not every filesystem permits it, and
    refusing to publish there would disable the scanner for a durability
    nicety rather than a correctness one."""
    import os
    import stat

    from tools import spot_snapshot as builder

    real_fsync = os.fsync

    def dir_fsync_fails(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("EINVAL")
        return real_fsync(fd)

    os.fsync = dir_fsync_fails
    try:
        path = builder.write_snapshot({"schema": "x"}, str(tmp_path))
    finally:
        os.fsync = real_fsync
    assert os.path.exists(path)
    # Best-effort, but never silent: "durable publish" must not quietly become
    # "atomic publish" with nobody able to tell which one they have. The
    # return value is half of that; the operator-visible line is the half that
    # actually reaches a human, so it is asserted rather than assumed.
    assert "not fsynced" in capsys.readouterr().err
    assert builder._fsync_dir("/nonexistent-directory-for-this-test") is False
    assert "could not open" in capsys.readouterr().err


def test_a_coins_field_of_the_wrong_type_does_not_report_an_empty_market():
    """"Пусто" and "сломано" are different sentences to a user."""
    from spot_market.snapshot import SnapshotUnavailable, parse_snapshot

    with pytest.raises(SnapshotUnavailable) as exc:
        parse_snapshot({"schema": "spot.snapshot/1", "generated_at_ms": 1,
                        "data_asof_ms": 1, "coins": {"a": 1}, "btc": {}})
    assert exc.value.reason == "malformed"
