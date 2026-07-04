"""US10Y hotfix: правильный Stooq-тикер, каскад Stooq -> FRED DGS10,
валидация данных и graceful degradation. Все тесты без сети."""
from __future__ import annotations

import asyncio

import httpx
import pytest

import config
from analyzer import macro

STOOQ_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2026-06-30,4.20,4.25,4.18,4.21,0\n"
    "2026-07-01,4.21,4.30,4.20,4.28,0\n"
    "2026-07-02,4.28,4.33,4.25,4.31,0\n"
    "2026-07-03,4.31,4.35,4.28,4.25,0\n"
    "2026-07-04,4.25,4.29,4.22,4.24,0\n"
)

FRED_DATA = (
    "Title:               10-Year Treasury Constant Maturity Rate\n"
    "Series ID:           DGS10\n"
    "Frequency:           Daily\n"
    "\n"
    "DATE        VALUE\n"
    "2026-06-30  4.20\n"
    "2026-07-01  4.28\n"
    "2026-07-02  .\n"
    "2026-07-03  4.31\n"
    "2026-07-04  4.35\n"
    "2026-07-05  .\n"
    "\n"
)


@pytest.fixture(autouse=True)
def _no_fred_api_key(monkeypatch):
    """Keyless-режим: каскад Stooq -> FRED plain (как на Railway)."""
    monkeypatch.setattr(config, "FRED_API_KEY", None)


def _http_404() -> httpx.HTTPStatusError:
    req = httpx.Request("GET", macro.STOOQ_BASE)
    return httpx.HTTPStatusError("404", request=req,
                                 response=httpx.Response(404, request=req))


def test_stooq_symbol_is_fixed():
    assert macro.STOOQ_SYMBOLS["us10y"] == "10yusy.b"


def test_stooq_success_is_used_without_fallback(monkeypatch):
    calls = []

    async def fake_stooq(symbol):
        calls.append(symbol)
        return macro._parse_stooq_csv(STOOQ_CSV)

    async def fred_must_not_be_called(series_id):
        raise AssertionError("FRED fallback не должен вызываться при живом Stooq")

    monkeypatch.setattr(macro, "_stooq_series", fake_stooq)
    monkeypatch.setattr(macro, "_fred_data_series", fred_must_not_be_called)
    result = asyncio.run(macro.get_macro())
    assert calls == ["10yusy.b"]
    assert result["us10y"] == 4.24  # последний close из Stooq
    assert result["us10y_trend"] == "up"  # 4.21 -> 4.24 за 5 наблюдений


@pytest.mark.parametrize("stooq_behavior", ["http_404", "invalid_html"])
def test_stooq_failure_falls_back_to_fred(monkeypatch, stooq_behavior):
    async def fake_stooq(symbol):
        if stooq_behavior == "http_404":
            raise _http_404()
        return macro._parse_stooq_csv("<!DOCTYPE html><html>challenge</html>")

    async def fake_fred(series_id):
        assert series_id == "DGS10"
        return macro._parse_fred_data(FRED_DATA)

    monkeypatch.setattr(macro, "_stooq_series", fake_stooq)
    monkeypatch.setattr(macro, "_fred_data_series", fake_fred)
    result = asyncio.run(macro.get_macro())
    assert result["us10y"] == 4.35  # последнее валидное значение FRED
    assert result["us10y_trend"] == "unknown"  # 4 валидных точки < 5


def test_both_providers_down_degrades_gracefully(monkeypatch):
    async def fake_stooq(symbol):
        raise _http_404()

    async def fake_fred(series_id):
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(macro, "_stooq_series", fake_stooq)
    monkeypatch.setattr(macro, "_fred_data_series", fake_fred)
    result = asyncio.run(macro.get_macro())  # не должен бросать
    assert result == {"us10y": None, "us10y_trend": "unknown"}


def test_fred_parser_skips_placeholders_and_headers():
    values = macro._parse_fred_data(FRED_DATA)
    # "." и заголовки пропущены, порядок хронологический, последнее — 4.35.
    assert values == [4.20, 4.28, 4.31, 4.35]
    # CSV-вариант (fredgraph) парсится тем же кодом.
    assert macro._parse_fred_data("DATE,DGS10\n2026-07-03,4.31\n2026-07-04,.\n") == [4.31]
    # NaN/мусор не проходят валидацию.
    assert macro._parse_fred_data("2026-07-03  nan\n2026-07-04  n/a\n") == []


def test_stooq_parser_rejects_html_and_junk():
    assert macro._parse_stooq_csv("<!DOCTYPE html><body>anti-bot</body>") == []
    assert macro._parse_stooq_csv(STOOQ_CSV) == [4.21, 4.28, 4.31, 4.25, 4.24]
    junk = "Date,Open,High,Low,Close,Volume\n2026-07-04,a,b,c,not_a_number,0\n"
    assert macro._parse_stooq_csv(junk) == []


def test_get_macro_result_shape_is_unchanged(monkeypatch):
    async def fake_stooq(symbol):
        return []

    async def fake_fred(series_id):
        return []

    monkeypatch.setattr(macro, "_stooq_series", fake_stooq)
    monkeypatch.setattr(macro, "_fred_data_series", fake_fred)
    result = asyncio.run(macro.get_macro())
    assert set(result) == {"us10y", "us10y_trend"}  # формат для production не менялся
