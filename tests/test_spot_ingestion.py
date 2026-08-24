"""S1 — the spot panel's ingestion layer.

The tests are mostly about what the layer must refuse. A research panel fails
in one direction that matters: it loads cleanly and is quietly missing the
assets that died. Every check here exists to make that failure loud.
"""
from __future__ import annotations

import asyncio
import json
import os

import pandas as pd
import pytest

from tools.spot_cache import COLUMNS, dataset_stem, to_frame, write_symbol
from tools.spot_client import SpotClient, SpotClientError, usdt_pairs
from tools.spot_dataset import (SpotDatasetError, available_symbols,
                                load_panel, load_symbol)
from tools.spot_symbols import STATUS_GONE, build_table


def kline_row(open_time: int, close: float = 100.0) -> list:
    """One Binance kline row, all twelve fields."""
    return [open_time, "1.0", "2.0", "0.5", str(close), "10.0",
            open_time + 86_399_999, "1000.0", 42, "5.0", "500.0", "0"]


def info(symbol: str, status: str = "TRADING", quote: str = "USDT") -> dict:
    return {"symbol": symbol, "baseAsset": symbol[:-len(quote)],
            "quoteAsset": quote, "status": status}


# --- the symbol spine ----------------------------------------------------


def test_a_break_pair_counts_as_dead_not_as_listed():
    """The finding that made the first run of this module wrong.

    A delisted pair keeps its exchangeInfo entry with status BREAK rather than
    disappearing. Deriving "delisted" from absence found 1 dead pair out of
    734; the venue's own status finds 250.
    """
    table = build_table(["BTCUSDT", "BCCUSDT"],
                        [info("BTCUSDT"), info("BCCUSDT", status="BREAK")])
    by_symbol = {r["symbol"]: r for r in table["symbols"]}
    assert by_symbol["BCCUSDT"]["trading_now"] is False
    assert by_symbol["BTCUSDT"]["trading_now"] is True
    assert table["counts"]["not_trading"] == 1
    assert table["counts"]["trading_now"] == 1


def test_a_pair_only_the_dump_index_remembers_is_kept():
    table = build_table(["BTCUSDT", "DEADUSDT"], [info("BTCUSDT")])
    by_symbol = {r["symbol"]: r for r in table["symbols"]}
    assert by_symbol["DEADUSDT"]["status"] == STATUS_GONE
    assert by_symbol["DEADUSDT"]["trading_now"] is False
    assert by_symbol["DEADUSDT"]["in_exchange_info"] is False
    assert table["counts"]["usdt_pairs_ever"] == 2


def test_non_usdt_pairs_are_not_in_the_spine():
    table = build_table(["BTCUSDT", "ETHBTC"],
                        [info("BTCUSDT"), info("ETHBTC", quote="BTC")])
    assert [r["symbol"] for r in table["symbols"]] == ["BTCUSDT"]


def test_usdt_pairs_helper_deduplicates_and_orders():
    assert usdt_pairs(["BUSDT", "AUSDT", "AUSDT", "XBTC"]) == ["AUSDT", "BUSDT"]


# --- pagination ----------------------------------------------------------


class FakePages(SpotClient):
    """A client whose network is a list of pages."""

    def __init__(self, pages):
        super().__init__()
        self.pages = list(pages)
        self.calls = []

    async def klines(self, symbol, interval, *, start_ms=None, end_ms=None,
                     limit=1000):
        self.calls.append(start_ms)
        return self.pages.pop(0) if self.pages else []


def test_full_history_walks_forward_until_the_pages_run_out():
    full = [kline_row(i * 86_400_000) for i in range(1, 2101)]
    pages = [full[:1000], full[1000:2000], full[2000:]]
    client = FakePages(pages)
    rows = asyncio.run(client.full_history("XUSDT", "1d"))
    assert len(rows) == 2100
    assert client.calls[0] == 0 and client.calls[1] is not None


def test_full_history_refuses_to_return_a_truncated_history():
    """A short answer is worse than an error: it looks like a young coin.

    Every page here is full and strictly newer than the last, so the walk
    never reaches a natural end — exactly the shape of a history deeper than
    the page budget allows.
    """
    pages = [[kline_row((p * 1000 + i) * 86_400_000) for i in range(1000)]
             for p in range(5)]
    client = FakePages(pages)
    with pytest.raises(SpotClientError, match="truncated"):
        asyncio.run(client.full_history("XUSDT", "1d", max_pages=3))


def test_full_history_drops_the_overlap_between_pages():
    a = [kline_row(i * 86_400_000) for i in range(1000)]
    b = [a[-1]] + [kline_row((1000 + i) * 86_400_000) for i in range(5)]
    rows = asyncio.run(FakePages([a, b]).full_history("XUSDT", "1d"))
    times = [r[0] for r in rows]
    assert len(times) == len(set(times)) == 1005


# --- frame construction --------------------------------------------------


def test_to_frame_keeps_the_s1_schema_and_drops_microstructure():
    df, dupes, malformed = to_frame([kline_row(0), kline_row(86_400_000)])
    assert tuple(df.columns) == COLUMNS
    assert "taker_buy_base" not in df.columns
    assert dupes == 0 and malformed == 0


def test_a_bar_that_closes_before_it_opens_is_dropped():
    """KLAYUSDT's real final bar, and 299 others like it.

    Almost all of them are the last bar of a delisted pair: a partial day that
    closes the moment trading stopped — 1INCHDOWNUSDT's is three hours long.
    A partial bar's high, low and close are not a day's, so a barrier scan over
    it is measuring something else. open_time stays perfectly ordered
    throughout, which is why an ordering check on open_time alone missed it.
    """
    good = kline_row(0)
    broken = kline_row(86_400_000)
    broken[6] = 43_200_000            # closes before its own bar opens
    df, _, malformed = to_frame([good, broken])
    assert malformed == 1
    assert list(df["open_time"]) == [0]


def test_to_frame_deduplicates_and_sorts():
    df, dupes, malformed = to_frame([kline_row(86_400_000), kline_row(0), kline_row(0)])
    assert dupes == 1
    assert list(df["open_time"]) == [0, 86_400_000]


# --- the fail-closed loader ----------------------------------------------


@pytest.fixture()
def panel(tmp_path):
    """Two symbols on disk: one alive, one delisted."""
    d = tmp_path / "klines"
    for symbol, trading in (("BTCUSDT", True), ("DEADUSDT", False)):
        df, _, _ = to_frame([kline_row(i * 86_400_000) for i in range(10)])
        write_symbol(df, str(d), symbol=symbol,
                     spine={"status": "TRADING" if trading else "BREAK",
                            "trading_now": trading},
                     duplicates_removed=0)
    return tmp_path


def manifest_path(root, symbol):
    return root / "klines" / f"{dataset_stem(symbol)}.manifest.json"


def data_path(root, symbol):
    return root / "klines" / f"{dataset_stem(symbol)}.csv"


def test_a_clean_panel_loads(panel):
    assert available_symbols(str(panel)) == ["BTCUSDT", "DEADUSDT"]
    series = load_symbol("BTCUSDT", str(panel))
    assert series.rows == 10 and series.trading_now is True


def test_an_edited_data_file_does_not_load(panel):
    """The check that earns its place across hundreds of files."""
    path = data_path(panel, "BTCUSDT")
    rows = path.read_text().splitlines()
    rows[3] = rows[3].replace(",100.0,", ",999.0,")
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(SpotDatasetError, match="content hash"):
        load_symbol("BTCUSDT", str(panel))


def test_a_truncated_data_file_does_not_load(panel):
    path = data_path(panel, "BTCUSDT")
    path.write_text("\n".join(path.read_text().splitlines()[:-2]) + "\n")
    with pytest.raises(SpotDatasetError):
        load_symbol("BTCUSDT", str(panel))


def test_a_manifest_pointing_at_nothing_does_not_load(panel):
    mpath = manifest_path(panel, "BTCUSDT")
    m = json.loads(mpath.read_text())
    m["data_file"] = "not_here.csv"
    mpath.write_text(json.dumps(m))
    with pytest.raises(SpotDatasetError, match="does not exist"):
        load_symbol("BTCUSDT", str(panel))


def test_a_row_count_that_disagrees_does_not_load(panel):
    """Rewriting both file and manifest is the shape of a partial refresh."""
    df, _, _ = to_frame([kline_row(i * 86_400_000) for i in range(4)])
    df.to_csv(data_path(panel, "BTCUSDT"), index=False)
    mpath = manifest_path(panel, "BTCUSDT")
    m = json.loads(mpath.read_text())
    from tools.spot_cache import sha256_file
    m["content_sha256"] = sha256_file(str(data_path(panel, "BTCUSDT")))
    mpath.write_text(json.dumps(m))
    with pytest.raises(SpotDatasetError, match="rows on disk"):
        load_symbol("BTCUSDT", str(panel))


def test_an_inconsistent_bar_does_not_load(panel):
    """The loader repeats the check rather than trusting the manifest: the
    files outlive the run that wrote them."""
    from tools.spot_cache import sha256_file
    df = pd.read_csv(data_path(panel, "BTCUSDT"))
    df.loc[3, "close_time"] = int(df.loc[3, "open_time"]) + 3_600_000
    df.to_csv(data_path(panel, "BTCUSDT"), index=False)
    mpath = manifest_path(panel, "BTCUSDT")
    m = json.loads(mpath.read_text())
    m["content_sha256"] = sha256_file(str(data_path(panel, "BTCUSDT")))
    mpath.write_text(json.dumps(m))
    with pytest.raises(SpotDatasetError, match="one day after"):
        load_symbol("BTCUSDT", str(panel))


def test_out_of_order_bars_do_not_load(panel):
    from tools.spot_cache import sha256_file
    df = pd.read_csv(data_path(panel, "BTCUSDT"))
    df = df.iloc[::-1]
    df.to_csv(data_path(panel, "BTCUSDT"), index=False)
    mpath = manifest_path(panel, "BTCUSDT")
    m = json.loads(mpath.read_text())
    m["content_sha256"] = sha256_file(str(data_path(panel, "BTCUSDT")))
    m["first_open_time"] = int(df["open_time"].iloc[0])
    m["last_open_time"] = int(df["open_time"].iloc[-1])
    mpath.write_text(json.dumps(m))
    with pytest.raises(SpotDatasetError, match="strictly increasing"):
        load_symbol("BTCUSDT", str(panel))


def test_a_stray_csv_without_a_manifest_is_not_a_dataset(panel):
    (panel / "klines" / "binance_spot_XUSDT_1d.csv").write_text("open_time\n1\n")
    assert "XUSDT" not in available_symbols(str(panel))


# --- the survivorship floor ----------------------------------------------


def test_a_panel_of_survivors_refuses_to_load(panel):
    """The one failure this whole stage exists to prevent.

    Loading only the living is not a smaller panel; it is a different and
    flattering answer, and nothing downstream can detect it. So it fails here.
    """
    with pytest.raises(SpotDatasetError, match="delisted"):
        list(load_panel(str(panel), symbols=["BTCUSDT"]))


def test_a_deliberate_subset_is_still_allowed(panel):
    loaded = list(load_panel(str(panel), symbols=["BTCUSDT"],
                             require_dead_share=0.0))
    assert [s.symbol for s in loaded] == ["BTCUSDT"]


def test_the_mixed_panel_passes_the_floor(panel):
    loaded = list(load_panel(str(panel)))
    assert len(loaded) == 2
    assert sum(1 for s in loaded if not s.trading_now) == 1
