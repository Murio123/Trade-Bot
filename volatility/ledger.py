"""C4.4: append-only forecast ledger.

Two rules make the ledger worth keeping, and both are enforced here rather
than left to caller discipline:

1. A forecast, once written, is never edited. Maturation appends a separate
   record that references it. If the prediction could be rewritten after the
   outcome was known, the whole ledger would be worthless as evidence.
2. A forecast matures only after `horizon_bars` fully closed bars. Not
   "roughly 48 hours later" — a specific bar count, checked against the bar
   index the forecast was made at.

Storage is JSON Lines: append is a single write, a truncated tail damages
one line rather than the file, and the history stays greppable.

Schema is symbol-keyed throughout so a second asset needs no migration.
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore

LEDGER_VERSION = "c44_ledger_v1"
KIND_FORECAST = "forecast"
KIND_MATURATION = "maturation"
LOCK_SUFFIX = ".lock"


class LedgerError(Exception):
    pass


class ImmutableRecordError(LedgerError):
    """Raised on any attempt to rewrite history."""


def _read_lines(path: str) -> Iterator[dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"{path}:{n} is not valid JSON: {exc}") from exc


def read_all(path: str) -> list[dict[str, Any]]:
    return list(_read_lines(path))


@contextlib.contextmanager
def _exclusive(path: str):
    """Hold an exclusive lock across the whole check-then-append.

    Without it the duplicate guards are read-then-write races: two producers
    firing on the same bar would both find nothing and both append. The lock
    lives on a sidecar so it is unaffected by how the ledger itself is opened.
    """
    if fcntl is None:  # pragma: no cover - POSIX only in practice
        raise LedgerError(
            "file locking is unavailable on this platform (no fcntl); "
            "refusing to append, because without the lock the duplicate "
            "guards are read-then-write races and the ledger could silently "
            "gain two rows for the same forecast")
    if path.endswith(LOCK_SUFFIX):
        raise LedgerError(
            f"ledger path must not end in {LOCK_SUFFIX!r}: its sidecar lock "
            f"would collide with another ledger's")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + LOCK_SUFFIX, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _append(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _forecast_index(path: str) -> dict[str, dict[str, Any]]:
    """forecast_id -> row, refusing to collapse duplicates.

    A dict comprehension would let the last duplicate win, which is exactly
    how an append-only ledger stops being append-only: a second row for the
    same id would silently override the first.
    """
    out: dict[str, dict[str, Any]] = {}
    for rec in _read_lines(path):
        if rec.get("kind") != KIND_FORECAST:
            continue
        fid = rec.get("forecast_id")
        if fid in out:
            raise LedgerError(
                f"duplicate forecast {fid} in {path} — the ledger is "
                "corrupt; it must contain each forecast exactly once")
        out[fid] = rec
    return out


def _matured_ids(path: str) -> set[str]:
    seen: set[str] = set()
    for rec in _read_lines(path):
        if rec.get("kind") != KIND_MATURATION:
            continue
        fid = rec.get("forecast_id")
        if fid in seen:
            raise LedgerError(
                f"duplicate maturation for {fid} in {path} — the ledger is "
                "corrupt; an outcome is recorded exactly once")
        seen.add(fid)
    return seen


def forecast_id(symbol: str, bar_idx: int) -> str:
    """One forecast per (symbol, bar). The id is derived, not random, so a
    duplicate run is detectable instead of silently adding a second row."""
    return f"{symbol}:{bar_idx}"


def append_forecast(path: str, *, symbol: str, bar_idx: int,
                    bar_close_utc: str, horizon_bars: int,
                    ranker_version: str, distribution_version: str,
                    score: float, percentile: float, category: str,
                    shadow: dict[str, Any] | None = None,
                    baselines: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record one forecast. Refuses to overwrite an existing (symbol, bar)."""
    fid = forecast_id(symbol, bar_idx)
    with _exclusive(path):
        existing = _forecast_index(path).get(fid)
        if existing is not None:
            raise ImmutableRecordError(
                f"forecast {fid} already recorded at bar "
                f"{existing.get('bar_idx')}; the ledger is append-only and a "
                "forecast is never rewritten")
        return _write_forecast(path, fid, symbol, bar_idx, bar_close_utc,
                               horizon_bars, ranker_version,
                               distribution_version, score, percentile,
                               category, shadow, baselines)


def _write_forecast(path, fid, symbol, bar_idx, bar_close_utc, horizon_bars,
                    ranker_version, distribution_version, score, percentile,
                    category, shadow, baselines) -> dict[str, Any]:
    record = {
        "ledger_version": LEDGER_VERSION,
        "kind": KIND_FORECAST,
        "forecast_id": fid,
        "symbol": symbol,
        "bar_idx": bar_idx,
        "bar_close_utc": bar_close_utc,
        "horizon_bars": horizon_bars,
        "matures_at_bar_idx": bar_idx + horizon_bars,
        "ranker_version": ranker_version,
        "distribution_version": distribution_version,
        "score": score,
        "percentile": percentile,
        "category": category,
        # Recorded, never acted on. Present from the first forecast so the
        # comparison can be made later without a backfill.
        "shadow": shadow or {},
        "baselines": baselines or {},
        "status": "pending",
    }
    _append(path, record)
    return record


def append_maturation(path: str, *, forecast_id_: str, at_bar_idx: int,
                      realized_score: float, realized_percentile: float,
                      realized_category: str,
                      shadow_errors: dict[str, Any] | None = None,
                      baseline_errors: dict[str, Any] | None = None
                      ) -> dict[str, Any]:
    """Record an outcome. The forecast row itself is left untouched."""
    with _exclusive(path):
        forecasts = _forecast_index(path)
        matured = _matured_ids(path)
        if forecast_id_ not in forecasts:
            raise LedgerError(f"no forecast {forecast_id_} to mature")
        if forecast_id_ in matured:
            raise ImmutableRecordError(
                f"forecast {forecast_id_} is already matured; an outcome is "
                "recorded exactly once")

        fc = forecasts[forecast_id_]
        if at_bar_idx < fc["matures_at_bar_idx"]:
            raise LedgerError(
                f"forecast {forecast_id_} matures at bar "
                f"{fc['matures_at_bar_idx']}, not {at_bar_idx} — an outcome "
                f"read before the horizon closes would be measuring an "
                f"unfinished window")

        record = {
            "ledger_version": LEDGER_VERSION,
            "kind": KIND_MATURATION,
            "forecast_id": forecast_id_,
            "symbol": fc["symbol"],
            "at_bar_idx": at_bar_idx,
            "realized_score": realized_score,
            "realized_percentile": realized_percentile,
            "realized_category": realized_category,
            "rank_error": abs(realized_percentile - fc["percentile"]),
            "category_hit": realized_category == fc["category"],
            "shadow_errors": shadow_errors or {},
            "baseline_errors": baseline_errors or {},
        }
        _append(path, record)
        return record


def pending(path: str, symbol: str | None = None) -> list[dict[str, Any]]:
    matured = _matured_ids(path)
    return [r for r in _forecast_index(path).values()
            if r["forecast_id"] not in matured
            and (symbol is None or r["symbol"] == symbol)]


def matured_pairs(path: str, symbol: str | None = None
                  ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(forecast, maturation) for every settled forecast, in ledger order."""
    forecasts = _forecast_index(path)
    _matured_ids(path)  # rejects a duplicated outcome on this read path too
    out = []
    for rec in _read_lines(path):
        if rec.get("kind") != KIND_MATURATION:
            continue
        fc = forecasts.get(rec["forecast_id"])
        if fc is None:
            raise LedgerError(f"maturation without forecast: {rec['forecast_id']}")
        if symbol is None or fc["symbol"] == symbol:
            out.append((fc, rec))
    return out


def latest_forecast(path: str, symbol: str) -> dict[str, Any] | None:
    """Most recent forecast for `symbol`.

    Goes through _forecast_index so a duplicated row raises here as well —
    scanning raw records and keeping the last match would have made a
    duplicate look like a legitimate update.
    """
    found = None
    for rec in _forecast_index(path).values():
        if rec.get("symbol") == symbol:
            found = rec
    return found
