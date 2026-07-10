"""Stage B2: the bounce profile ships DISABLED and OBSERVATION-ONLY.

Three independent locks, because any one of them failing would put a
counter-trend alert in front of the user:

1. the profile is declared correctly (require_exhaustion, observation_only);
2. no scheduler job exists for it unless ENABLE_BOUNCE_PROFILE is true;
3. even when it does run, the job stops after the forecast row — no signal,
   no Telegram, no delivery mark, no journal trade, DRY_RUN alerts or not.
"""
from __future__ import annotations

import asyncio
import re

import pytest

import backtest as bt
import config
import scheduler
from tools import backtest_parity
from signal_engine.htf_filter import DEFAULT_HTF_POLICY, apply_htf_policy
from signal_engine.profiles import PROFILES, get_profile
from signal_engine.vetoes import TF_HOURS

TREND_PROFILES = ("swing", "position", "intraday")


# --- 1. The profile ---------------------------------------------------------

def test_bounce_profile_is_declared():
    p = PROFILES["bounce"]
    assert p["analysis_type"] == "BOUNCE"
    assert p["htf_policy"] == "require_exhaustion"
    assert p["observation_only"] is True
    assert p["entry"] == "1h" and p["htf"] == "1d"
    assert p["label"] and p["emoji"]


def test_bounce_timeframes_are_known_and_ordered():
    p = PROFILES["bounce"]
    for tf in [p["entry"], p["htf"], p["stop_tf"], *p["mtf"], *p["zone_tfs"]]:
        assert tf in TF_HOURS, tf
    # The entry TF must be part of the multi-TF grid, and the stop must be
    # taken from a slower frame than the entry.
    assert p["entry"] in p["mtf"]
    assert TF_HOURS[p["stop_tf"]] > TF_HOURS[p["entry"]]


def test_bounce_targets_and_cooldown_are_conservative():
    p = PROFILES["bounce"]
    assert 0 < p["targets"][0] < p["targets"][1]
    assert p["cooldown_hours"] > 0
    assert p["minimum_confidence"] > 0 and p["minimum_risk_reward"] > 0


def test_analysis_type_is_unique_across_profiles():
    types = [p["analysis_type"] for p in PROFILES.values()]
    assert len(types) == len(set(types)), types


def test_bounce_is_the_only_observation_only_profile():
    observation = {n for n, p in PROFILES.items() if p.get("observation_only")}
    assert observation == {"bounce"}


# --- 2. Existing profiles are untouched -------------------------------------

@pytest.mark.parametrize("name", TREND_PROFILES)
def test_trend_profiles_still_block_counter_trend(name):
    p = get_profile(name)
    assert p["htf_policy"] == DEFAULT_HTF_POLICY
    assert not p.get("observation_only")
    assert apply_htf_policy("long", "bearish", p) is None
    assert apply_htf_policy("short", "bullish", p) is None


# --- 3. Disabled by default -------------------------------------------------

def _job_ids(monkeypatch, *, bounce: bool) -> set[str]:
    monkeypatch.setattr(scheduler.config, "ENABLE_BOUNCE_PROFILE", bounce)
    monkeypatch.setattr(scheduler.config, "ENABLE_POSITION_ANALYSIS", True)
    monkeypatch.setattr(scheduler.config, "ENABLE_FAST_ANALYSIS", True)
    app = type("App", (), {"bot": object(), "bot_data": {}})()
    return {job.id for job in scheduler.build_scheduler(app).get_jobs()}


def test_bounce_flag_defaults_to_false():
    assert config.ENABLE_BOUNCE_PROFILE is False


def test_scheduler_registers_no_bounce_job_by_default(monkeypatch):
    assert "analysis_bounce" not in _job_ids(monkeypatch, bounce=False)


def test_scheduler_registers_bounce_job_only_when_enabled(monkeypatch):
    off = _job_ids(monkeypatch, bounce=False)
    on = _job_ids(monkeypatch, bounce=True)
    assert on - off == {"analysis_bounce"}   # nothing else appears
    assert off - on == set()                 # and nothing else disappears
    assert {"analysis_swing", "analysis_position", "analysis_intraday"} <= off


# --- 4. observation_only: forecast in, nothing out --------------------------

class _ObservationDb:
    """Records every write the job attempts."""

    def __init__(self) -> None:
        self.forecasts: list[dict] = []
        self.signals: list[dict] = []
        self.delivered: list[int] = []

    async def forecast_exists(self, *args, **kwargs) -> bool:
        return False

    async def insert_forecast(self, fc: dict) -> int:
        self.forecasts.append(fc)
        return 7

    async def signals_today(self, *args, **kwargs) -> list[dict]:
        return []

    async def last_delivered_signal(self, *args, **kwargs):
        return None

    async def open_trades(self) -> list[dict]:
        return []

    async def insert_signal(self, record: dict) -> int:  # must never be called
        self.signals.append(record)
        return 1

    async def mark_delivered(self, signal_id: int) -> None:  # must never fire
        self.delivered.append(signal_id)

    async def link_forecast_signal(self, forecast_id: int, signal_id: int) -> None:
        pass


async def _async(value):
    return value


def _run_observation_job(monkeypatch, profile_name: str) -> tuple[_ObservationDb, dict]:
    """Drive analysis_job on a profile with a would-be ENTER signal."""
    db = _ObservationDb()
    called: dict[str, bool] = {}
    app = type("App", (), {"bot": object(), "bot_data": {"binance": object()}})()
    # A score well above SCORE_ALERT_MIN: without the guard this delivers.
    result = {"status": "alert", "score": 9, "symbol": "BTCUSDT"}

    # Delivery must be blocked by the profile, not by DRY_RUN.
    monkeypatch.setattr(scheduler.config, "SEND_DRY_RUN_ALERTS", True)
    monkeypatch.setattr(scheduler.config, "ENABLE_FORECAST_LEDGER", True)
    monkeypatch.setattr(scheduler, "db", db)
    monkeypatch.setattr(scheduler, "_last_closed_candle_time",
                        lambda *a, **k: _async(None))
    monkeypatch.setattr(scheduler, "gather_market_context", lambda *a, **k: _async({}))
    monkeypatch.setattr(scheduler, "run_cascade", lambda *a, **k: _async(result))
    monkeypatch.setattr(scheduler, "enrich_forecast_with_lifecycle",
                        lambda *a, **k: _async({}))
    import signal_engine.forecast_record as fr
    monkeypatch.setattr(fr, "build_forecast_record",
                        lambda *a, **k: {"analysis_type": "BOUNCE"})

    def _spy(name):
        def fn(*a, **k):
            called[name] = True
            return _async(True)
        return fn

    monkeypatch.setattr(scheduler, "_maybe_reversal_alert", _spy("reversal_alert"))
    monkeypatch.setattr(scheduler.alerts, "send_signal_alert", _spy("telegram"))
    monkeypatch.setattr(scheduler, "_send_signal_chart", _spy("chart"))
    monkeypatch.setattr(scheduler.journal, "has_active_trade",
                        lambda *a, **k: _async(False))
    monkeypatch.setattr(scheduler.journal, "can_open_new_trade",
                        lambda *a, **k: _async(True))
    monkeypatch.setattr(scheduler.journal, "record_signal_as_trade",
                        _spy("journal_trade"))
    monkeypatch.setattr(scheduler.formatting, "format_signal", lambda r: "body")

    asyncio.run(scheduler.analysis_job(app, profile_name))
    return db, called


def test_observation_only_writes_forecast_and_nothing_else(monkeypatch):
    db, called = _run_observation_job(monkeypatch, "bounce")
    assert len(db.forecasts) == 1        # the ledger row IS written
    assert db.signals == []              # no signal row
    assert db.delivered == []            # never marked delivered
    assert called == {}                  # no Telegram, no chart, no journal trade


def test_non_observation_profile_still_delivers(monkeypatch):
    """Control: the guard is what stops bounce, not the fixture."""
    db, called = _run_observation_job(monkeypatch, "swing")
    assert len(db.forecasts) == 1
    assert len(db.signals) == 1
    assert db.delivered == [1]
    assert called.get("telegram") and called.get("journal_trade")


# --- 5. The backtest passes ctx into the policy -----------------------------

def test_backtest_policy_ctx_has_the_live_nested_shape():
    ctx = bt._htf_policy_ctx(
        rev={"bull_strong": True}, bull_tfs=3, bear_tfs=0,
        eq={"zone": "discount"}, liq={"liquidity_swept_below": True},
        ob={}, fvg={}, cvd={}, bullish_div=True, bearish_div=False)
    assert ctx["reversal_mtf"]["bull_tf_count"] == 3
    assert ctx["reversal_mtf"]["bear_tf_count"] == 0
    assert ctx["equilibrium"]["zone"] == "discount"
    assert ctx["divergence"]["bullish_divergence"] is True
    # funding history does not exist in the walk: the key is absent, so the
    # exhaustion predicates fail closed instead of passing vacuously.
    assert "funding" not in ctx
    assert apply_htf_policy("long", "bearish", PROFILES["bounce"], ctx) is None


@pytest.mark.parametrize("module", [bt, backtest_parity])
def test_no_policy_call_bypasses_ctx(module):
    """Every apply_htf_policy call site in the backtest path passes a ctx."""
    import inspect
    calls = [line.strip() for line in inspect.getsource(module).splitlines()
             if re.search(r"\bapply_htf_policy\(", line)]
    assert calls, "no apply_htf_policy call found"
    for call in calls:
        assert "policy_ctx" in call, call
