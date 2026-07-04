"""Unified Context Builder (этап 2): enrichment, иерархия ТФ, изоляция legacy.

Проверяем shadow/compatibility-слой:
- enrich_swing_context не мутирует исходный legacy ctx;
- snapshots содержат 1W/1D/12H/4H/1H с корректными ролями и structure;
- stage-2 volatility и OI-поля заполнены и читаются контрактом;
- from_legacy БЕЗ enrichment остаётся идентичным (missing_blocks не съезжают);
- новый golden unified_swing.json стабилен (offline, seed-детерминизм).

Обновление эталона (ТОЛЬКО при осознанном изменении):
    GOLDEN_UPDATE=1 pytest tests/test_unified_builder.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from contracts import UnifiedMarketContext, to_jsonable
from signal_engine.profiles import get_profile
from tests.synthetic_market import build_ctx, build_unified_ctx
from tests.test_golden_contexts import _normalize
from unified_context import (SWING_TF_HIERARCHY, enrich_swing_context,
                             oi_enrichment, stage2_volatility)

GOLDEN_DIR = Path(__file__).parent / "golden"
PROFILE = get_profile("swing")

# Параметры golden-сценария (совпадают с восходящим рынком golden-контекстов).
UNIFIED_PARAMS = dict(seed=7, drift=0.004, vol=0.015)

_EXPECTED_ROLES = {"1w": "regime", "1d": "bias", "12h": "phase",
                   "4h": "setup", "1h": "execution"}


def test_enrich_does_not_mutate_legacy_ctx():
    ctx = build_ctx("swing", **UNIFIED_PARAMS)
    keys_before = set(ctx)
    inds_before = ctx["inds_by_tf"]
    inds_keys_before = set(inds_before)

    extra = {
        "inds_1w": {"price": 1.0}, "structure_by_tf": {"1w": {"trend": "bullish"}},
        "volatility_stage2": {"realized_volatility": 0.01},
        "oi_change_pct": 1.5, "price_oi_relation": "rising_price_rising_oi",
        "oi_rising": True, "correlation": {"verdict": "risk_on"},
    }
    enriched = enrich_swing_context(ctx, extra)

    # Исходный ctx не тронут: ни новых ключей, ни 1W в inds_by_tf.
    assert set(ctx) == keys_before
    assert "structure_by_tf" not in ctx
    assert "volatility_stage2" not in ctx
    assert ctx["inds_by_tf"] is inds_before
    assert set(inds_before) == inds_keys_before  # 1W не подмешан в оригинал

    # Enriched — новый словарь с enrichment-данными.
    assert enriched is not ctx
    assert enriched["inds_by_tf"] is not inds_before
    assert "1w" in enriched["inds_by_tf"]
    assert enriched["structure_by_tf"]["1w"]["trend"] == "bullish"


def test_unified_context_has_full_tf_hierarchy():
    ctx = build_unified_ctx(**UNIFIED_PARAMS)
    context = UnifiedMarketContext.from_legacy(ctx, PROFILE)
    snaps = context.technical.snapshots

    for tf in SWING_TF_HIERARCHY:
        assert tf in snaps, f"нет снапшота {tf}"
        assert snaps[tf].role == _EXPECTED_ROLES[tf], tf
        assert snaps[tf].structure is not None, f"нет structure для {tf}"
        assert "last_event" in snaps[tf].structure


def test_enrichment_fields_read_by_contract():
    ctx = build_unified_ctx(**UNIFIED_PARAMS)
    context = UnifiedMarketContext.from_legacy(ctx, PROFILE)

    vol = context.volatility
    assert vol.realized_volatility is not None
    assert vol.range_width_atr is not None
    assert vol.candle_expansion_ratio is not None
    assert vol.compression_score is not None
    assert vol.abnormal_candle is not None

    deriv = context.derivatives
    assert deriv.oi_change_pct is not None
    assert deriv.price_oi_relation is not None
    assert deriv.oi_rising is not None

    assert context.intermarket.correlations is not None
    assert context.intermarket.verdict in ("risk_on", "risk_off")


def test_enrichment_clears_stage2_missing_blocks():
    plain = UnifiedMarketContext.from_legacy(build_ctx("swing", **UNIFIED_PARAMS),
                                             PROFILE)
    unified = UnifiedMarketContext.from_legacy(build_unified_ctx(**UNIFIED_PARAMS),
                                               PROFILE)
    plain_missing = set(plain.data_quality.missing_blocks)
    unified_missing = set(unified.data_quality.missing_blocks)

    stage2 = {"technical.structure", "technical.1w",
              "volatility.realized_volatility", "volatility.range_width_atr",
              "volatility.candle_expansion_ratio", "volatility.compression_score",
              "volatility.abnormal_candle",
              "derivatives.oi_change", "derivatives.price_oi_relation"}
    # В обычном ctx все stage-2 блоки — в missing.
    assert stage2 <= plain_missing
    # Enrichment их снимает.
    assert not (stage2 & unified_missing), stage2 & unified_missing


def test_from_legacy_without_enrichment_unchanged():
    """Регрессия: market ctx без enrichment даёт structure=None и полный
    stage-2 missing-набор (поведение этапа 1 не тронуто)."""
    context = UnifiedMarketContext.from_legacy(build_ctx("swing", **UNIFIED_PARAMS),
                                               PROFILE)
    assert all(s.structure is None for s in context.technical.snapshots.values())
    assert context.volatility.realized_volatility is None
    assert context.derivatives.oi_change_pct is None


def test_stage2_volatility_degrades_gracefully():
    out = stage2_volatility(None, 100.0)
    assert set(out) == {"realized_volatility", "range_width_atr",
                        "candle_expansion_ratio", "compression_score",
                        "abnormal_candle", "liquidation_event"}
    assert all(v is None for v in out.values())


def test_oi_enrichment_relation_classification():
    rising = [{"sumOpenInterest": 100.0}, {"sumOpenInterest": 110.0}]
    import pandas as pd
    df_up = pd.DataFrame({"close": [100.0, 105.0]})
    change, relation, is_rising = oi_enrichment(rising, df_up)
    assert change == 10.0
    assert relation == "rising_price_rising_oi"
    assert is_rising is True

    change, relation, is_rising = oi_enrichment([], df_up)
    assert change is None and relation is None and is_rising is None


def _unified_snapshot() -> dict:
    ctx = build_unified_ctx(**UNIFIED_PARAMS)
    context = UnifiedMarketContext.from_legacy(ctx, PROFILE)
    return _normalize({
        "scenario": {"name": "unified_swing", **UNIFIED_PARAMS},
        "unified_context": to_jsonable(context),
    })


def test_golden_unified_swing():
    snap = _unified_snapshot()
    path = GOLDEN_DIR / "unified_swing.json"
    if os.environ.get("GOLDEN_UPDATE"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=1,
                                   sort_keys=True) + "\n", encoding="utf-8")
        return
    assert path.exists(), (
        f"нет эталона {path.name}: сгенерируй его командой "
        f"GOLDEN_UPDATE=1 pytest {__file__}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert snap == expected, (
        "golden-снапшот 'unified_swing' разошёлся: enrichment или адаптер "
        "изменились. Если изменение ОСОЗНАННОЕ — обнови через GOLDEN_UPDATE=1.")
