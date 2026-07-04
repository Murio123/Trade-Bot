"""Адаптеры legacy -> контракты: маппинг полей, деградация, кластеризация."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import config
from contracts import (ContractError, Decision, GateCheck, ReversalEffect,
                       UnifiedMarketContext, VolatilityRegime,
                       analysis_from_legacy, decision_from_legacy,
                       to_jsonable)
from contracts.legacy import _cluster_items, _vol_regime
from signal_engine.profiles import get_profile

NOW = datetime(2026, 7, 1, 12, 2, tzinfo=timezone.utc)
LAST_CLOSE = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
PROFILE = get_profile("swing")


def _ind(price: float = 100_000.0, **over):
    base = {
        "price": price, "ema20": 99_500.0, "ema50": 99_000.0,
        "ema100": 98_500.0, "ema200": 97_000.0, "rsi": 55.0,
        "macd_hist": 10.0, "atr": 800.0, "volume": 120.0, "avg_volume": 100.0,
        "ema_aligned_bullish": True, "ema_aligned_bearish": False,
        "_series": {"rsi": object()},  # не должна попадать в контракт
    }
    base.update(over)
    return base


def _ctx(**over):
    ob_zone = {"low": 98_000.0, "high": 98_600.0, "type": "bullish", "tf": "12h"}
    fvg_zone = {"low": 99_900.0, "high": 100_300.0, "type": "bullish", "tf": "4h"}
    ctx = {
        "symbol": "BTCUSDT", "timeframe": "4h", "timestamp": NOW,
        "price": 100_000.0, "atr": 800.0,
        "ind_1h": _ind(), "ind_4h": _ind(), "ind_1d": _ind(),
        "ind_signal": _ind(),
        "inds_by_tf": {"1h": _ind(), "4h": _ind(), "12h": _ind(),
                       "1d": _ind()},
        "funding": {"current": 0.0003, "zscore": 2.5},
        "open_interest": 5e9,
        "long_short_ratio": {"ratio": 1.4},
        "cvd": {"cvd_bullish": True, "cvd_bearish": False},
        "macro": {"us10y": 4.2, "us10y_trend": "down"},
        "onchain": {"exchange_netflow": -1000.0},
        "order_blocks": {"blocks": [ob_zone], "bullish_ob": ob_zone,
                         "bearish_ob": None, "price_in_bullish_ob": False,
                         "price_in_bearish_ob": False, "rejection_wick": True,
                         "zone_tfs": ["12h"]},
        "liquidity": {"equal_highs": [101_500.0], "equal_lows": [98_200.0],
                      "liquidity_swept_below": False,
                      "liquidity_swept_above": False,
                      "reversal_candle": False},
        "volume_profile": {"poc": 99_800.0, "vah": 101_200.0, "val": 98_400.0},
        "fvg": {"fvgs": [fvg_zone], "bullish_fvg": fvg_zone,
                "bearish_fvg": None, "price_in_bullish_fvg": True,
                "price_in_bearish_fvg": False},
        "htf_levels": {"highs": [101_500.0, 103_000.0],
                       "lows": [96_500.0, 98_200.0]},
        "zone_tfs": ["12h"],
        "equilibrium": {"high": 103_000.0, "low": 96_500.0, "eq": 99_750.0,
                        "zone": "equilibrium", "pos": 0.54},
        "reversal": {"bullish_reversal": False, "bearish_reversal": False},
        "reversal_mtf": {
            "per_tf": {"1h": {"bullish_reversal": True, "bull_strong": True,
                              "factors_bull": ["sweep", "divergence"],
                              "bull_score": 3},
                       "4h": {"bullish_reversal": True, "bull_score": 2,
                              "factors_bull": ["exhaustion"]},
                       "12h": {}, "1d": {}},
            "bull_tfs": ["1h", "4h"], "bear_tfs": [],
            "bull_tf_count": 2, "bear_tf_count": 0,
            "combined_bullish": True, "combined_bearish": False,
            "bull_candle_confirm": True, "bear_candle_confirm": False},
        "divergence": {"bullish_divergence": False,
                       "bearish_divergence": False},
        "liquidation_map": None, "sweep_signal": None,
        "sessions": {"best_session": "NY"},
        "volatility": {"atr": 800.0, "atr_pct": 0.8, "atr_percentile": 60.0,
                       "hv_percentile": 55.0, "regime": "normal",
                       "expected_move": {"24h": {"abs": 1960.0, "pct": 1.96}}},
        "volatility_1d": {"atr_percentile": 50.0},
        "last_close_time": LAST_CLOSE,
        "decision_time": NOW,
        "executable_price": 100_050.0,
        "executable_price_degraded": False,
    }
    ctx.update(over)
    return ctx


def _signal(**over):
    sig = {
        "symbol": "BTCUSDT", "timeframe": "4h", "status": "alert",
        "direction": "long", "entry_price": 100_000.0, "price": 100_000.0,
        "stop_loss": 98_800.0, "target_1": 101_800.0, "target_2": 103_600.0,
        "take_profit_levels": [101_800.0, 103_600.0],
        "targets_structure": True, "stop_basis": "structure",
        "tp2_source": "structural", "atr": 800.0,
        "score": 8, "score_weighted": 8.4,
        "category_scores": {"trend": 2, "momentum": 2, "volume": 1,
                            "structure": 3, "macro": 0},
        "reasons": ["EMA бычий порядок", "MACD бычий кроссовер",
                    "Реакция от бычьего Order Block", "Объём подтверждает"],
        "contradicting_factors": ["RSI перекуплен"],
        "long_score": 8.4, "short_score": 2.0, "counter_score": 2.0,
        "confidence": 0.71, "market_regime": "trend_up",
        "mtf": {"1h": "bullish", "4h": "bullish", "12h": "bullish",
                "1d": "bullish"},
        "mtf_agreement": {"agree": 4, "total": 4, "ratio": 1.0},
        "risk_reward": 3.0, "expected_move_points": 2400.0,
        "expected_move_atr": 3.0,
        "entry_zone": {"low": 99_800.0, "high": 100_200.0},
        "cancel_conditions": ["закрытие свечи за уровнем стопа"],
        "analysis_type": "SWING",
        "data_freshness_seconds": 120.0, "signal_close_price": 100_000.0,
        "executable_price_at_decision": 100_050.0,
        "executable_price_degraded": False,
    }
    sig.update(over)
    return sig


# --- контекст-адаптер -------------------------------------------------------

def test_full_context_maps_every_block():
    uc = UnifiedMarketContext.from_legacy(_ctx(), PROFILE)

    assert uc.meta.symbol == "BTCUSDT"
    assert uc.meta.analysis_type == "SWING"
    assert uc.meta.signal_close_price == 100_000.0
    assert uc.meta.executable_price == 100_050.0
    assert uc.meta.engine == "legacy_v1"

    snaps = uc.technical.snapshots
    assert set(snaps) == {"1h", "4h", "12h", "1d"}
    assert snaps["1d"].role == "bias"
    assert snaps["4h"].role == "setup"
    assert snaps["1h"].role == "execution"
    assert snaps["12h"].role == "phase"
    assert snaps["4h"].trend == "bullish"
    assert "_series" not in snaps["4h"].indicators
    assert uc.technical.market_regime in ("trend_up", "range")

    assert uc.volatility.regime == VolatilityRegime.NORMAL_TREND
    assert uc.volatility.atr_percent == 0.8
    # abnormal_volatility — честный источник (ATR-перцентиль окна);
    # abnormal_candle появится только с candle_expansion_ratio (этап 2).
    assert uc.volatility.abnormal_volatility is False
    assert uc.volatility.abnormal_candle is None

    assert uc.derivatives.funding == 0.0003
    assert uc.derivatives.overheated is True  # |z|=2.5 >= 2.0

    assert uc.spot_flow.exchange_netflow == -1000.0
    assert uc.macro.us10y == 4.2

    assert uc.reversal.combined_side == "bull"
    assert uc.reversal.tf_count == 2
    assert uc.reversal.bull_tf_count == 2 and uc.reversal.bear_tf_count == 0
    assert uc.reversal.candle_confirmed is True
    assert uc.reversal.trend_alignment == "aligned"  # бычий bias + дно
    assert uc.reversal.effect == ReversalEffect.AMPLIFY
    assert uc.reversal.per_tf["1h"].confirmed_side == "bull"
    assert uc.reversal.per_tf["1h"].strong is True
    assert uc.reversal.per_tf["1h"].bull_factors == ["sweep", "divergence"]

    assert uc.data_quality.stale is False
    assert uc.data_quality.data_freshness_seconds == 120.0
    # Отсутствующие на этапе 1 блоки честно перечислены.
    assert "news" in uc.data_quality.missing_blocks
    assert "intermarket" in uc.data_quality.missing_blocks


def test_levels_block_unifies_all_sources():
    uc = UnifiedMarketContext.from_legacy(_ctx(), PROFILE)
    lv = uc.levels

    all_sources = {s for c in lv.clusters for s in c.sources}
    # Каждый legacy-источник представлен в едином наборе кластеров.
    for expected in ("htf_level_high", "htf_level_low", "ob_bullish",
                     "fvg_bullish", "poc", "equal_highs", "equal_lows",
                     "ema50", "ema200"):
        assert expected in all_sources, expected

    assert lv.nearest_resistance is not None
    assert lv.nearest_resistance.lo > 100_000.0
    assert lv.nearest_support is not None
    assert lv.nearest_support.hi < 100_000.0

    # Цена внутри бычьего FVG -> он кандидат entry-зоны.
    assert any("fvg_bullish" in c.sources for c in lv.entry_zone_candidates)

    # Кандидаты стопов — структура за ценой, с provenance и ATR-нормировкой.
    assert lv.stop_candidates
    assert all(p.atr_distance is not None for p in lv.stop_candidates)
    assert lv.distance_to_invalidation_atr == min(
        p.atr_distance for p in lv.stop_candidates)


def test_stop_tp_candidates_are_mirrored_for_long_and_short():
    """M1: lows ниже цены — цели SHORT, highs выше цены — цели LONG;
    уровни «не на своей стороне» кандидатами не являются."""
    ctx = _ctx(htf_levels={"highs": [101_500.0, 103_000.0],
                           "lows": [96_500.0, 98_200.0]},
               volume_profile={})  # без VP: только структурные уровни
    uc = UnifiedMarketContext.from_legacy(ctx, PROFILE)
    lv = uc.levels

    structural_tps = {p.price for p in lv.tp_candidates
                      if p.provenance == "structural"}
    # Обе стороны представлены зеркально.
    assert {101_500.0, 103_000.0} <= structural_tps      # цели LONG
    assert {96_500.0, 98_200.0} <= structural_tps        # цели SHORT
    stop_prices = {p.price for p in lv.stop_candidates
                   if p.provenance == "structural"}
    assert stop_prices == structural_tps  # тот же структурный набор
    # Provenance сохранён и у OB-краёв.
    assert any(p.provenance == "ob_edge" for p in lv.stop_candidates)


def test_cluster_merging_within_quarter_atr():
    items = [
        {"lo": 101_000.0, "hi": 101_000.0, "source": "htf_level_high", "tf": None},
        {"lo": 101_100.0, "hi": 101_100.0, "source": "equal_highs", "tf": None},
        {"lo": 105_000.0, "hi": 105_000.0, "source": "poc", "tf": None},
    ]
    clusters = _cluster_items(items, price=100_000.0, tol=200.0)  # 0.25*ATR(800)
    assert len(clusters) == 2
    merged = clusters[0]
    assert merged.lo == 101_000.0 and merged.hi == 101_100.0
    assert merged.sources == ["equal_highs", "htf_level_high"]
    assert merged.kind == "resistance"
    # 2 источника -> 0.5; одиночный уровень -> 0.25.
    assert merged.strength == 0.5
    assert clusters[1].strength == 0.25


def test_degraded_sources_yield_none_plus_missing_blocks():
    ctx = _ctx(funding={}, macro={}, onchain={}, volatility={},
               reversal_mtf={})
    uc = UnifiedMarketContext.from_legacy(ctx, PROFILE)
    assert uc.derivatives.funding is None
    assert uc.derivatives.overheated is None
    assert uc.macro.us10y is None
    assert uc.spot_flow.exchange_netflow is None
    assert uc.volatility.regime is None
    assert uc.reversal.combined_side is None
    assert uc.reversal.effect == ReversalEffect.NEUTRAL
    mb = uc.data_quality.missing_blocks
    for name in ("derivatives.funding", "macro", "spot_flow.exchange_netflow",
                 "volatility", "reversal.mtf"):
        assert name in mb, name


def test_missing_required_field_raises_contract_error():
    ctx = _ctx()
    del ctx["price"]
    with pytest.raises(ContractError):
        UnifiedMarketContext.from_legacy(ctx, PROFILE)


@pytest.mark.parametrize("vol,expected", [
    ({"regime": "normal", "atr_percentile": 50.0}, VolatilityRegime.NORMAL_TREND),
    ({"regime": "compression", "atr_percentile": 10.0}, VolatilityRegime.COMPRESSION),
    ({"regime": "expansion", "atr_percentile": 85.0}, VolatilityRegime.HIGH_VOLATILITY),
    ({"regime": "normal", "atr_percentile": 96.0}, VolatilityRegime.HIGH_VOLATILITY),
    ({"regime": "unknown"}, None),
    ({}, None),
])
def test_volatility_regime_mapping(vol, expected):
    assert _vol_regime(vol) == expected


def test_weak_reversal_signals_are_preserved():
    """M3: partial-сигналы (score < порога подтверждения) не теряются —
    они нужны раннему предупреждению и display-parity с legacy /reversal."""
    ctx = _ctx(reversal_mtf={
        "per_tf": {"1h": {"bull_score": 1, "factors_bull": ["divergence"]},
                   "4h": {"bear_score": 1, "factors_bear": ["exhaustion"]},
                   "12h": {}, "1d": {}},
        "bull_tfs": [], "bear_tfs": [],
        "bull_tf_count": 0, "bear_tf_count": 0,
        "combined_bullish": False, "combined_bearish": False,
        "bull_candle_confirm": False, "bear_candle_confirm": False})
    uc = UnifiedMarketContext.from_legacy(ctx, PROFILE)
    rv = uc.reversal
    assert rv.combined_side is None and rv.tf_count == 0
    assert rv.bull_tf_count == 0 and rv.bear_tf_count == 0
    weak_1h = rv.per_tf["1h"]
    assert weak_1h.confirmed_side is None and not weak_1h.bullish
    assert weak_1h.bull_score == 1
    assert weak_1h.bull_factors == ["divergence"]
    weak_4h = rv.per_tf["4h"]
    assert weak_4h.bear_score == 1
    assert weak_4h.bear_factors == ["exhaustion"]


def test_confirmed_reversal_keeps_both_sides_data():
    """M3: подтверждённое чтение маппится как раньше и не теряет счёт
    противоположной стороны."""
    ctx = _ctx()
    ctx["reversal_mtf"]["per_tf"]["1h"]["bear_score"] = 1
    ctx["reversal_mtf"]["per_tf"]["1h"]["factors_bear"] = ["wick"]
    uc = UnifiedMarketContext.from_legacy(ctx, PROFILE)
    read = uc.reversal.per_tf["1h"]
    assert read.bullish is True and read.confirmed_side == "bull"
    assert read.strong is True
    assert read.bull_score == 3 and read.bear_score == 1
    assert read.bear_factors == ["wick"]


def test_reversal_effect_counter_trend_dampens():
    bear_ind = _ind(ema20=101_000.0, ema50=102_000.0, ema200=103_000.0,
                    price=100_000.0, ema_aligned_bullish=False,
                    ema_aligned_bearish=True)
    ctx = _ctx(inds_by_tf={"1h": _ind(), "4h": _ind(), "12h": _ind(),
                           "1d": bear_ind},
               ind_1d=bear_ind)
    uc = UnifiedMarketContext.from_legacy(ctx, PROFILE)
    # Дно (bull) при медвежьем 1D-bias — контртренд, эффект DAMPEN.
    assert uc.technical.htf_bias == "bearish"
    assert uc.reversal.trend_alignment == "counter"
    assert uc.reversal.effect == ReversalEffect.DAMPEN


def test_deep_context_adapter_degrades_honestly():
    deep_ctx = {
        "symbol": "BTCUSDT", "timestamp": NOW, "price": 100_000.0,
        "atr": 800.0,
        "ind_1d": _ind(), "ind_12h": _ind(), "ind_4h": _ind(),
        "structure_1d": {"trend": "bullish", "last_event": "BOS_up"},
        "structure_12h": {"trend": "bullish", "last_event": None},
        "structure_4h": {"trend": "range", "last_event": None},
        "order_blocks": {"bullish_ob": {"low": 98_000.0, "high": 98_600.0,
                                        "type": "bullish"},
                         "bearish_ob": None, "price_in_bullish_ob": False,
                         "price_in_bearish_ob": False},
        "fvg": {"fvgs": [], "bullish_fvg": None, "bearish_fvg": None,
                "price_in_bullish_fvg": False, "price_in_bearish_fvg": False},
        "liquidity": {"equal_highs": [101_500.0], "equal_lows": [98_200.0]},
        "volume_profile": {"poc": 99_800.0, "vah": 101_200.0, "val": 98_400.0},
        "reversal": {"bullish_reversal": True, "bull_strong": False,
                     "factors_bull": ["exhaustion"], "bull_score": 2},
        "funding": {"current": 0.0001, "zscore": 0.5},
        "open_interest": 5e9, "oi_rising": True,
        "long_short_ratio": {"ratio": 1.1},
        "cvd": {"cvd_bullish": True},
        "volatility": {"regime": "normal", "atr_percentile": 40.0,
                       "hv_percentile": 45.0},
        "correlation": {"verdict": "bullish", "supportive": 2, "counted": 3},
    }
    uc = UnifiedMarketContext.from_legacy_deep(deep_ctx)
    assert uc.meta.executable_price_degraded is True
    assert uc.meta.signal_candle_close_time is None
    assert uc.data_quality.stale is True
    assert uc.technical.snapshots["1d"].structure["last_event"] == "BOS_up"
    assert uc.derivatives.oi_rising is True
    assert uc.intermarket.verdict == "bullish"
    assert uc.reversal.per_tf["4h"].bullish is True
    assert uc.reversal.combined_side is None  # MTF-вердикта в deep нет
    assert "reversal.mtf" in uc.data_quality.missing_blocks
    json.dumps(uc.to_json(), sort_keys=True)  # сериализуемость


# --- адаптеры решений -------------------------------------------------------

def test_analysis_separates_direction_and_quality():
    a = analysis_from_legacy(_signal(), PROFILE)
    assert a.direction == "long"
    assert a.direction_probability == pytest.approx(8.4 / 10.4, abs=1e-3)
    assert a.setup_quality == pytest.approx(0.84, abs=1e-3)
    assert a.contradiction_score == pytest.approx(2.0 / 10.4, abs=1e-3)
    # freshness 120/1800 и смещение 50/800 ATR.
    assert a.execution_quality == pytest.approx(
        (1 - 120 / 1800) * (1 - 50 / 800), abs=1e-3)
    assert a.scenario is not None
    assert a.scenario.stop.provenance == "structural"
    assert a.scenario.tp2.provenance == "structural"
    assert a.scenario.setup_type == "continuation"
    assert a.evidence["long_total"] == 8.4


def test_decision_alert_maps_to_enter_with_passed_gates():
    d = decision_from_legacy(_signal(), _ctx(), PROFILE)
    assert d.decision == Decision.ENTER_LONG
    assert d.vetoes_triggered == []
    assert all(g.passed for g in d.gates)
    assert len(d.reasons) == 3
    assert d.invalidation_note == "закрытие свечи за уровнем стопа"
    assert d.meta.engine == "legacy_v1"


def test_decision_statuses_map_to_wait_and_no_trade():
    d = decision_from_legacy(_signal(status="cooldown"), _ctx(), PROFILE)
    assert d.decision == Decision.WAIT
    assert d.vetoes_triggered == ["cooldown"]

    blocked = {"status": "blocked", "blocked_at": "stale_data",
               "price": 100_000.0}
    d2 = decision_from_legacy(blocked, _ctx(), PROFILE)
    assert d2.decision == Decision.NO_TRADE
    assert d2.gates == [GateCheck(name="stale_data", passed=False)]
    assert d2.scenario is None

    blocked_nt = {"status": "blocked", "blocked_at": "no_trade",
                  "direction": "long", "long_score": 6.0, "short_score": 1.0,
                  "no_trade_reasons": ["R:R ниже минимума", "TF conflict"]}
    d3 = decision_from_legacy(blocked_nt, _ctx(), PROFILE)
    failed = [g for g in d3.gates if not g.passed]
    assert [g.name for g in failed] == ["no_trade"]
    assert "R:R ниже минимума" in failed[0].detail
    # До no_trade каскад дошёл — предыдущие стадии помечены пройденными.
    assert {g.name for g in d3.gates if g.passed} == {
        "stale_data", "abnormal_volatility", "direction_conflict",
        "htf_filter", "diversity", "below_threshold", "dead_zone",
        "crowded_funding", "wait_for_sweep"}


def test_decision_journal_distinguishes_block_reason():
    # score ниже порога алерта -> alert_threshold.
    d = decision_from_legacy(_signal(status="journal", score=6,
                                     score_weighted=6.0), _ctx(), PROFILE)
    assert d.decision == Decision.WAIT
    assert d.vetoes_triggered == ["alert_threshold"]
    # score выше порога, но решение опоздало -> freshness.
    d2 = decision_from_legacy(_signal(status="journal", stale_decision=True),
                              _ctx(), PROFILE)
    assert d2.vetoes_triggered == ["freshness"]
    assert any("WAIT" in w for w in d2.warnings)
    # score выше порога, свежесть в норме -> дневной лимит.
    d3 = decision_from_legacy(_signal(status="journal"), _ctx(), PROFILE)
    assert d3.vetoes_triggered == ["daily_limit"]


def test_direction_probability_needs_both_scores():
    a = analysis_from_legacy(_signal(long_score=None, short_score=None),
                             PROFILE)
    assert a.direction_probability is None
    b = analysis_from_legacy(_signal(long_score=0.0, short_score=0.0,
                                     direction=None,
                                     candidate_direction=None), PROFILE)
    assert b.direction_probability is None


def test_decision_serializes_to_stable_json():
    d = decision_from_legacy(_signal(), _ctx(), PROFILE)
    one = json.dumps(to_jsonable(d), sort_keys=True)
    two = json.dumps(to_jsonable(
        decision_from_legacy(_signal(), _ctx(), PROFILE)), sort_keys=True)
    assert one == two
