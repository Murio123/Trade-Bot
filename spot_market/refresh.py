"""S4A.1: keeping the snapshot fresh, from inside the running bot.

The bot and the refresher are the **same process on the same container**, so
they see the same file. That is not an incidental convenience — it is the
reason this design was chosen over a separate Railway cron service, which
would have its own filesystem and would write a snapshot the bot could never
read.

The builder is invoked as a **subprocess**, not imported:

    python -m tools.spot_snapshot --source live --snapshot-dir <dir>

Three things follow from that, and all three are the point:

  * **The research package never enters the bot's process.**
    `tools/spot_snapshot.py` imports `spot.features` and `spot.universe`; a
    module-level import here would drag offline research code into the live
    runtime, which the S-track's import-direction rule exists to prevent. A
    module name in a string is not an import.
  * **One builder, one code path.** What the scheduler runs at 02:07 is
    character-for-character what an operator runs by hand. There is no second
    implementation to drift.
  * **A builder that dies cannot take the bot with it.** A segfault, an OOM
    kill or a hung socket is an exit code here, not an exception in the event
    loop.

Nothing in this module raises. A refresh that fails returns a result saying
so, the previous snapshot stays exactly where it was, and the existing
fail-closed reader decides — on its own unchanged thresholds — whether what
remains is still fresh enough to show. Freshness limits are never relaxed to
paper over a refresh that did not happen.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass

from spot_market.snapshot import (DEFAULT_SNAPSHOT_PATH, MarketSnapshot,
                                  SnapshotUnavailable, load_snapshot)

log = logging.getLogger(__name__)

# The builder, named rather than imported. See the module docstring.
BUILDER_MODULE = "tools.spot_snapshot"

# The repository root, so the subprocess can resolve `-m tools.spot_snapshot`
# regardless of what the working directory happens to be.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rebuild once the snapshot is older than this. Deliberately far below the
# reader's 24 h / 48 h fail-closed limits: the refresher's job is to keep the
# scanner from ever reaching them, and a threshold that only fires once the
# scanner has already gone dark would be a monitor, not a refresher.
REFRESH_AFTER_HOURS = 2

# A live build measured ~33 s over 467 symbols. Ten minutes is slack enough
# for a bad network day and short enough that a hung socket cannot hold the
# lock until the next cycle.
BUILD_TIMEOUT_SECONDS = 600

_MAX_LOG_OUTPUT = 2000

# Exit codes, mirroring tools/spot_snapshot.py.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_LOCKED = 2


@dataclass(frozen=True)
class RefreshResult:
    """What one refresh attempt did. Never an exception, always a record."""
    status: str            # built | skipped_fresh | locked | failed | timeout
    detail: str
    duration_s: float
    previous_preserved: bool
    eligible: int | None = None
    data_asof_ms: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("built", "skipped_fresh")


def _digest(path: str) -> str | None:
    """Fingerprint of the snapshot on disk, or None if there is none."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def read_snapshot(path: str = DEFAULT_SNAPSHOT_PATH
                  ) -> tuple[MarketSnapshot | None, str]:
    """(snapshot, reason). The reader's own rules decide; nothing is bypassed."""
    try:
        return load_snapshot(path), "ok"
    except SnapshotUnavailable as exc:
        return None, exc.reason


def needs_refresh(path: str = DEFAULT_SNAPSHOT_PATH, *,
                  max_age_hours: int = REFRESH_AFTER_HOURS,
                  now_ms: int | None = None) -> bool:
    """Is the snapshot missing, unusable, or simply due?

    A snapshot the reader refuses for ANY reason needs rebuilding, including
    the ones that are not about age — a malformed or self-contradictory file
    is not going to fix itself, and leaving it in place would keep the scanner
    dark until someone looked.
    """
    snap, reason = read_snapshot(path)
    if snap is None:
        log.debug("spot snapshot needs a refresh: %s", reason)
        return True
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    age_h = (now - snap.generated_at_ms) / 3_600_000
    return age_h >= max_age_hours


def _tail(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip()
    return text[-_MAX_LOG_OUTPUT:] if len(text) > _MAX_LOG_OUTPUT else text


async def refresh(path: str = DEFAULT_SNAPSHOT_PATH, *,
                  force: bool = False,
                  timeout_s: int = BUILD_TIMEOUT_SECONDS,
                  source: str = "live",
                  python: str | None = None,
                  cwd: str | None = None) -> RefreshResult:
    """Run one refresh. Returns a result; never raises, never leaves a ruin.

    `force` skips the freshness check only. It does not skip the lock, and it
    cannot make a partial build reach the canonical path.
    """
    started = time.monotonic()

    def done(status: str, detail: str, *, preserved: bool = True,
             snap: MarketSnapshot | None = None) -> RefreshResult:
        return RefreshResult(
            status=status, detail=detail,
            duration_s=time.monotonic() - started,
            previous_preserved=preserved,
            eligible=None if snap is None else snap.universe_size,
            data_asof_ms=None if snap is None else snap.data_asof_ms)

    if not force and not needs_refresh(path):
        snap, _ = read_snapshot(path)
        log.info("spot snapshot: still fresh, no rebuild "
                 "(assets=%s)", None if snap is None else snap.universe_size)
        return done("skipped_fresh", "snapshot is within the refresh window",
                    snap=snap)

    before = _digest(path)
    outdir = os.path.dirname(os.path.abspath(path)) or "."
    argv = [python or sys.executable, "-m", BUILDER_MODULE,
            "--source", source, "--snapshot-dir", outdir]
    log.info("spot snapshot: refresh started (source=%s, dest=%s)",
             source, outdir)

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd or REPO_ROOT,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except OSError as exc:
        log.error("spot snapshot: could not start the builder: %s; previous "
                  "snapshot preserved", exc)
        return done("failed", f"spawn failed: {exc}",
                    preserved=_digest(path) == before)

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(),
                                                timeout=timeout_s)
    except asyncio.TimeoutError:
        # Kill it, then reap it: an unreaped child would hold the build lock
        # for the whole life of the bot process and every later cycle would
        # skip.
        proc.kill()
        with_output = await proc.communicate()
        log.error("spot snapshot: build exceeded %ss and was killed; previous "
                  "snapshot preserved. tail=%s",
                  timeout_s, _tail(with_output[1]))
        return done("timeout", f"builder killed after {timeout_s}s",
                    preserved=_digest(path) == before)

    if proc.returncode == EXIT_LOCKED:
        log.info("spot snapshot: another build is already running — skipped")
        return done("locked", _tail(stderr) or "another build holds the lock",
                    preserved=_digest(path) == before)

    if proc.returncode != EXIT_OK:
        log.error("spot snapshot: builder exited %s; previous snapshot "
                  "preserved. stderr=%s", proc.returncode, _tail(stderr))
        return done("failed", f"exit {proc.returncode}: {_tail(stderr)}",
                    preserved=_digest(path) == before)

    # The builder says it succeeded. Verify by reading the file back through
    # the same fail-closed reader the Telegram screens use — a snapshot the
    # product cannot read is not a successful refresh, whatever the exit code
    # claimed.
    snap, reason = read_snapshot(path)
    if snap is None:
        log.error("spot snapshot: builder exited 0 but the result is "
                  "unreadable (%s)", reason)
        return done("failed", f"builder wrote an unreadable snapshot: {reason}",
                    preserved=False)

    log.info("spot snapshot: refresh finished in %.1fs "
             "(assets=%s, data_asof=%s, path=%s)",
             time.monotonic() - started, snap.universe_size,
             snap.data_asof_ms, path)
    return done("built", "snapshot replaced", snap=snap)
