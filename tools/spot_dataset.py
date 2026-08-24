"""S1: fail-closed loader for the spot panel.

Read-only, offline. Never fetches; it only reads what `tools/spot_cache.py`
wrote.

The loading discipline is `tools/kline_dataset.py`'s, for the same reason it
was built there: **the manifest is read first, the data file is opened only by
the reference the manifest gives, and every claim the manifest makes is
re-verified against the data itself.** A dataset that does not match its own
manifest is not loaded at all, because a silently wrong frame is worse than a
missing one.

What is re-verified per symbol:

    content_sha256   the bytes are the bytes the manifest describes
    rows             the row count
    first/last       the boundary timestamps
    columns          the exact column set, in order
    monotonicity     strictly increasing open_time AND close_time
    bar consistency  close_time == open_time + one day - 1ms

The hash check is the one that earns its place in a 700-file panel. In a
one-symbol cache "the manifest was written next to the file a second ago" is
nearly good enough; across hundreds of files written over many minutes, with
reruns and partial refreshes, it is not.

**One panel-level rule beyond the per-symbol checks: dead symbols may not go
missing.** `require_dead_share` refuses to hand back a panel whose delisted
fraction has collapsed, because that is what a survivorship-biased universe
looks like from the inside — everything loads cleanly, and the answer is
quietly wrong. It is cheap to check and it is the failure this whole stage
exists to prevent.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd

from tools.spot_cache import (COLUMNS, DAY_MS, INGESTION_NAME, KLINES_SUBDIR,
                              TIMEFRAME, dataset_stem, sha256_file)
from tools.spot_symbols import DEFAULT_OUTDIR


class SpotDatasetError(Exception):
    """The stored panel does not match what its manifests claim."""


@dataclass(frozen=True)
class SpotSeries:
    """One symbol's verified daily history."""
    symbol: str
    df: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def trading_now(self) -> bool:
        return bool(self.manifest.get("trading_now"))

    @property
    def rows(self) -> int:
        return len(self.df)


def _manifest_path(klines_dir: str, symbol: str) -> str:
    return os.path.join(klines_dir, f"{dataset_stem(symbol)}.manifest.json")


def klines_dir(outdir: str = DEFAULT_OUTDIR) -> str:
    return os.path.join(outdir, KLINES_SUBDIR)


def available_symbols(outdir: str = DEFAULT_OUTDIR) -> list[str]:
    """Symbols with a manifest on disk. The manifest is the index, not the
    directory listing: a stray CSV without one is not a dataset."""
    d = klines_dir(outdir)
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        if name.endswith(".manifest.json") and not name.startswith("_"):
            with open(os.path.join(d, name), encoding="utf-8") as fh:
                out.append(json.load(fh)["symbol"])
    return sorted(out)


def load_symbol(symbol: str, outdir: str = DEFAULT_OUTDIR) -> SpotSeries:
    """Load one symbol, or raise. Manifest first, then the data it points to."""
    d = klines_dir(outdir)
    mpath = _manifest_path(d, symbol)
    if not os.path.exists(mpath):
        raise SpotDatasetError(f"{symbol}: no manifest at {mpath}")
    with open(mpath, encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("timeframe") != TIMEFRAME:
        raise SpotDatasetError(
            f"{symbol}: manifest timeframe {manifest.get('timeframe')!r} is "
            f"not {TIMEFRAME!r}")

    data_path = os.path.join(d, manifest["data_file"])
    if not os.path.exists(data_path):
        raise SpotDatasetError(f"{symbol}: manifest names {data_path}, which "
                               f"does not exist")

    digest = sha256_file(data_path)
    if digest != manifest.get("content_sha256"):
        raise SpotDatasetError(
            f"{symbol}: content hash mismatch — the data file is not the file "
            f"this manifest describes (manifest "
            f"{str(manifest.get('content_sha256'))[:12]}, file {digest[:12]})")

    df = pd.read_csv(data_path)

    if tuple(df.columns) != tuple(manifest.get("columns", ())):
        raise SpotDatasetError(
            f"{symbol}: columns {tuple(df.columns)} do not match the manifest")
    if tuple(df.columns) != COLUMNS:
        raise SpotDatasetError(
            f"{symbol}: columns {tuple(df.columns)} are not the S1 schema")
    if len(df) != manifest.get("rows"):
        raise SpotDatasetError(
            f"{symbol}: {len(df)} rows on disk, manifest claims "
            f"{manifest.get('rows')}")
    if len(df):
        if int(df["open_time"].iloc[0]) != manifest.get("first_open_time"):
            raise SpotDatasetError(f"{symbol}: first open_time disagrees with "
                                   f"the manifest")
        if int(df["open_time"].iloc[-1]) != manifest.get("last_open_time"):
            raise SpotDatasetError(f"{symbol}: last open_time disagrees with "
                                   f"the manifest")
        deltas = df["open_time"].diff().dropna()
        if not (deltas > 0).all():
            raise SpotDatasetError(
                f"{symbol}: open_time is not strictly increasing; the frame "
                f"has duplicate or out-of-order bars")
        # close_time gets its own checks because open_time can be perfectly
        # ordered while close_time is not: KLAYUSDT's final bar closed before
        # it opened. A frame like that passes an open_time check and then
        # breaks every scan keyed on close_time — which is most of them.
        if not (df["close_time"].diff().dropna() > 0).all():
            raise SpotDatasetError(
                f"{symbol}: close_time is not strictly increasing")
        if not (df["close_time"] == df["open_time"] + DAY_MS - 1).all():
            bad = int((df["close_time"] != df["open_time"] + DAY_MS - 1).sum())
            raise SpotDatasetError(
                f"{symbol}: {bad} bar(s) do not close exactly one day after "
                f"they open; the frame is internally inconsistent")
    return SpotSeries(symbol=symbol, df=df, manifest=manifest)


def ingestion_summary(outdir: str = DEFAULT_OUTDIR) -> dict[str, Any]:
    path = os.path.join(klines_dir(outdir), INGESTION_NAME)
    if not os.path.exists(path):
        raise SpotDatasetError(f"no ingestion summary at {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_panel(outdir: str = DEFAULT_OUTDIR, *, symbols: list[str] | None = None,
               require_dead_share: float = 0.15) -> Iterator[SpotSeries]:
    """Yield every verified symbol, refusing a panel that lost its dead.

    `require_dead_share` is a floor on the fraction of loaded symbols that no
    longer trade. The observed share is about a third; the default sits well
    below that so ordinary drift does not trip it, while a panel that has been
    filtered down to survivors — the one error that silently flatters every
    downstream result — cannot be read at all. Pass 0.0 to load a deliberate
    subset.
    """
    names = symbols if symbols is not None else available_symbols(outdir)
    if not names:
        raise SpotDatasetError(f"no symbols available under {outdir}")

    loaded = [load_symbol(s, outdir) for s in names]
    dead = sum(1 for s in loaded if not s.trading_now)
    share = dead / len(loaded)
    if share < require_dead_share:
        raise SpotDatasetError(
            f"only {dead}/{len(loaded)} loaded symbols ({share:.1%}) are "
            f"delisted, below the required {require_dead_share:.0%}. A panel "
            f"of survivors is exactly what point-in-time work must not use; "
            f"pass require_dead_share=0.0 if this subset is deliberate")
    yield from loaded
