"""Адаптеры legacy -> контракты (этап 1).

Единственное место, где живёт знание о форме legacy-словарей
(gather_market_context / gather_swing_context / run_cascade result).
Адаптеры ничего не меняют в production-поведении: они только читают.

Провизорные формулы качества (direction_probability, setup_quality,
execution_quality, contradiction_score) — детерминированные преобразования
существующих чисел каскада; помечены как shadow-only и не участвуют в
production-решении до калибровки (этап 10).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import config
from contracts.base import (ContractError, Decision, PricePoint,
                            ReversalEffect, VolatilityRegime, price_point)
from contracts.context import (BlockMeta, DataQualityContext,
                               DerivativesContext, IntermarketContext,
                               LevelCluster, LevelsContext, MacroContext,
                               MetaContext, NewsContext, ReversalContext,
                               ReversalRead, SpotFlowContext, TechnicalContext,
                               TFSnapshot, UnifiedMarketContext,
                               VolatilityContext)
from contracts.decision import (FinalDecision, GateCheck, Scenario,
                                SwingAnalysisResult)
from signal_engine.htf_filter import get_htf_bias
from signal_engine.mtf_confidence import trend_label
from signal_engine.regime import detect_regime
from signal_engine.vetoes import (FUNDING_Z_EXTREME, TF_HOURS,
                                  abnormal_volatility,
                                  reversal_trend_alignment, stale_data)

_LEGACY_SOURCE = "legacy_ctx"

# Сколько кандидатов уровней держим в контексте (снапшоты и промпты не
# должны раздуваться хвостом дальних уровней).
_MAX_CANDIDATES = 10


# ---------------------------------------------------------------------------
# Контекст: основной каскадный ctx (gather_market_context)
# ---------------------------------------------------------------------------

def context_from_legacy(ctx: dict[str, Any],
                        profile: dict[str, Any]) -> UnifiedMarketContext:
    missing: set[str] = set()
    meta = _meta_from_legacy(ctx, profile)

    technical = _technical_from_legacy(ctx, profile, meta, missing)
    volatility = _volatility_from_legacy(ctx, meta, missing)
    levels = _levels_from_legacy(ctx, meta, missing)
    reversal = _reversal_from_legacy(ctx, technical.htf_bias, meta, missing)
    derivatives = _derivatives_from_legacy(ctx, meta, missing)
    spot_flow = _spot_flow_from_legacy(ctx, meta, missing)
    intermarket = _intermarket_from_legacy(ctx, meta, missing)
    macro = _macro_from_legacy(ctx, meta, missing)
    news = _news_from_legacy(meta, missing)
    data_quality = _data_quality_from_legacy(ctx, meta, missing)

    return UnifiedMarketContext(
        meta=meta, technical=technical, volatility=volatility, levels=levels,
        reversal=reversal, derivatives=derivatives, spot_flow=spot_flow,
        intermarket=intermarket, macro=macro, news=news,
        data_quality=data_quality)


def context_from_legacy_deep(ctx: dict[str, Any]) -> UnifiedMarketContext:
    """gather_swing_context: нет decision snapshot и MTF-reversal — эти блоки
    помечаются degraded, executable price деградирует до close."""
    missing: set[str] = {"meta.executable_price", "meta.signal_candle_close_time",
                         "reversal.mtf", "levels.htf_levels"}
    ts = ctx.get("timestamp")
    price = ctx.get("price")
    if ts is None or price is None or ctx.get("symbol") is None:
        raise ContractError("deep ctx: нет symbol/timestamp/price")
    meta = MetaContext(
        symbol=ctx["symbol"], analysis_type="SWING", entry_timeframe="4h",
        gathered_at=ts, decision_time=ts, signal_candle_close_time=None,
        signal_close_price=float(price), executable_price=float(price),
        executable_price_degraded=True, engine="legacy_v1",
        prompt_version=config.PROMPT_VERSION)

    inds = {tf: ctx.get(f"ind_{tf}") for tf in ("1d", "12h", "4h")
            if ctx.get(f"ind_{tf}")}
    snapshots = {}
    for tf, ind in inds.items():
        snapshots[tf] = TFSnapshot(
            tf=tf, role=_tf_role(tf, {"htf": "1d", "entry": "4h"}),
            indicators=_strip_series(ind),
            structure=ctx.get(f"structure_{tf}"), trend=trend_label(ind))
    htf_bias = get_htf_bias(inds.get("1d") or {})
    block = BlockMeta(source=_LEGACY_SOURCE, as_of=ts)
    technical = TechnicalContext(
        meta=block, snapshots=snapshots, htf_bias=htf_bias,
        market_regime=detect_regime(inds.get("1d")),
        divergence=None, sessions=None, equilibrium=None)

    volatility = _volatility_from_legacy(ctx, meta, missing)
    levels = _levels_from_legacy(ctx, meta, missing)

    rev = ctx.get("reversal") or {}
    read = _reversal_read("4h", rev)
    reversal = ReversalContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=ts, degraded=True),
        per_tf={"4h": read} if rev else {}, combined_side=None, tf_count=0,
        candle_confirmed=None, trend_alignment=None,
        effect=ReversalEffect.NEUTRAL)

    derivatives = _derivatives_from_legacy(ctx, meta, missing)
    spot_flow = _spot_flow_from_legacy(ctx, meta, missing)
    intermarket = _intermarket_from_legacy(ctx, meta, missing)
    macro = _macro_from_legacy(ctx, meta, missing)
    news = _news_from_legacy(meta, missing)
    data_quality = DataQualityContext(
        meta=block, data_freshness_seconds=None, decision_latency_seconds=None,
        stale=True, executable_price_degraded=True,
        missing_blocks=sorted(missing))

    return UnifiedMarketContext(
        meta=meta, technical=technical, volatility=volatility, levels=levels,
        reversal=reversal, derivatives=derivatives, spot_flow=spot_flow,
        intermarket=intermarket, macro=macro, news=news,
        data_quality=data_quality)


def _meta_from_legacy(ctx: dict[str, Any], profile: dict[str, Any],
                      engine: str = "legacy_v1") -> MetaContext:
    for key in ("symbol", "timeframe", "price", "timestamp"):
        if ctx.get(key) is None:
            raise ContractError(f"legacy ctx: обязательное поле '{key}' отсутствует")
    executable = ctx.get("executable_price")
    return MetaContext(
        symbol=ctx["symbol"],
        analysis_type=profile.get("analysis_type", "SWING"),
        entry_timeframe=ctx["timeframe"],
        gathered_at=ctx["timestamp"],
        decision_time=ctx.get("decision_time") or ctx["timestamp"],
        signal_candle_close_time=_to_dt(ctx.get("last_close_time")),
        signal_close_price=float(ctx["price"]),
        executable_price=float(executable if executable is not None else ctx["price"]),
        executable_price_degraded=bool(ctx.get("executable_price_degraded")
                                       or executable is None),
        engine=engine,
        prompt_version=config.PROMPT_VERSION)


def _technical_from_legacy(ctx: dict[str, Any], profile: dict[str, Any],
                           meta: MetaContext, missing: set[str]) -> TechnicalContext:
    inds_by_tf = ctx.get("inds_by_tf") or {}
    if not inds_by_tf:
        raise ContractError("legacy ctx: inds_by_tf отсутствует")
    snapshots = {tf: TFSnapshot(tf=tf, role=_tf_role(tf, profile),
                                indicators=_strip_series(ind),
                                structure=None, trend=trend_label(ind))
                 for tf, ind in sorted(inds_by_tf.items())}
    missing.add("technical.structure")  # BOS/CHoCH есть только в deep-ctx (этап 2)
    if "1w" not in snapshots:
        missing.add("technical.1w")     # недельный режим — этап 8
    htf_ind = inds_by_tf.get(profile.get("htf"), ctx.get("ind_1d") or {})
    return TechnicalContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at),
        snapshots=snapshots,
        htf_bias=get_htf_bias(htf_ind),
        market_regime=detect_regime(ctx.get("ind_1d"), ctx.get("volatility_1d")),
        divergence=ctx.get("divergence"),
        sessions=ctx.get("sessions"),
        equilibrium=ctx.get("equilibrium"))


def _tf_role(tf: str, profile: dict[str, Any]) -> str:
    """Роль ТФ в иерархии режима. 1W -> regime (появится на этапе 8)."""
    if tf == "1w":
        return "regime"
    if tf == profile.get("htf"):
        return "bias"
    if tf == profile.get("entry"):
        return "setup"
    tf_h = TF_HOURS.get(tf, 1.0)
    entry_h = TF_HOURS.get(profile.get("entry", ""), 1.0)
    return "execution" if tf_h < entry_h else "phase"


def _strip_series(ind: dict[str, Any]) -> dict[str, Any]:
    """Индикаторный снапшот без несериализуемых pandas-серий."""
    return {k: v for k, v in ind.items() if not k.startswith("_")}


def _volatility_from_legacy(ctx: dict[str, Any], meta: MetaContext,
                            missing: set[str]) -> VolatilityContext:
    vol = ctx.get("volatility") or {}
    atr = ctx.get("atr")
    price = meta.signal_close_price
    if not vol:
        missing.add("volatility")
    for f in ("realized_volatility", "range_width_atr",
              "candle_expansion_ratio", "compression_score",
              "abnormal_candle", "liquidation_event"):
        missing.add(f"volatility.{f}")  # источники появятся на этапе 2
    return VolatilityContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=not vol),
        atr=atr,
        atr_percent=round(atr / price * 100, 4) if (atr and price) else None,
        atr_percentile=vol.get("atr_percentile"),
        hv_percentile=vol.get("hv_percentile"),
        realized_volatility=None,
        range_width_atr=None,
        candle_expansion_ratio=None,
        compression_score=None,
        abnormal_volatility=abnormal_volatility(vol) if vol else None,
        abnormal_candle=None,
        liquidation_event=None,
        regime=_vol_regime(vol),
        expected_move=vol.get("expected_move") or None)


def _vol_regime(vol: dict[str, Any]) -> VolatilityRegime | None:
    """Маппинг legacy-режима в 5-значный enum.

    LOW_VOLATILITY и LIQUIDATION_EVENT на этапе 1 недостижимы: legacy
    различает только expansion/compression/normal (их источники — этап 2).
    """
    regime = (vol or {}).get("regime")
    if regime in (None, "unknown"):
        return None
    p = vol.get("atr_percentile")
    if p is not None and p >= config.ABNORMAL_VOL_PERCENTILE:
        return VolatilityRegime.HIGH_VOLATILITY
    if regime == "expansion":
        return VolatilityRegime.HIGH_VOLATILITY
    if regime == "compression":
        return VolatilityRegime.COMPRESSION
    return VolatilityRegime.NORMAL_TREND


# ---------------------------------------------------------------------------
# Levels: единый сбор уровней (канон для этапа 5)
# ---------------------------------------------------------------------------

def _levels_from_legacy(ctx: dict[str, Any], meta: MetaContext,
                        missing: set[str]) -> LevelsContext:
    price = meta.signal_close_price
    atr = ctx.get("atr") or price * 0.005
    items = _collect_level_items(ctx)
    clusters = _cluster_items(items, price, tol=atr * 0.25)

    supports = [c for c in clusters if c.kind == "support" and c.hi < price * 0.9995]
    resistances = [c for c in clusters if c.kind == "resistance" and c.lo > price * 1.0005]
    nearest_support = max(supports, key=lambda c: c.hi) if supports else None
    nearest_resistance = min(resistances, key=lambda c: c.lo) if resistances else None

    liq = ctx.get("liquidity") or {}
    liquidity_above = [price_point(h, "liquidity", price, atr)
                       for h in (liq.get("equal_highs") or []) if h > price]
    liquidity_below = [price_point(l, "liquidity", price, atr)
                       for l in (liq.get("equal_lows") or []) if l < price]

    ob = ctx.get("order_blocks") or {}
    fvg = ctx.get("fvg") or {}
    entry_zones = _entry_zone_candidates(ob, fvg, price)

    stop_cands, tp_cands = _stop_tp_candidates(ctx, price, atr)

    eq = ctx.get("equilibrium")
    dist_target = (min(pp.atr_distance for pp in tp_cands
                       if pp.atr_distance is not None)
                   if tp_cands else None)
    dist_inv = (min(pp.atr_distance for pp in stop_cands
                    if pp.atr_distance is not None)
                if stop_cands else None)
    return LevelsContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=not items),
        clusters=clusters,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        liquidity_above=liquidity_above[:_MAX_CANDIDATES],
        liquidity_below=liquidity_below[:_MAX_CANDIDATES],
        entry_zone_candidates=entry_zones,
        stop_candidates=stop_cands[:_MAX_CANDIDATES],
        invalidation_candidates=stop_cands[:_MAX_CANDIDATES],
        tp_candidates=tp_cands[:_MAX_CANDIDATES],
        range_position=eq,
        distance_to_next_target_atr=dist_target,
        distance_to_invalidation_atr=dist_inv)


def _collect_level_items(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Union всех legacy-источников уровней: htf_levels + OB + FVG +
    volume profile + equal highs/lows + ключевые EMA (1D/4H).

    Тот же состав, что использовали formatting._nearest_sr и
    _collect_level_items, — теперь в одном месте."""
    items: list[dict[str, Any]] = []

    def add(lo: Any, hi: Any, source: str, tf: str | None = None) -> None:
        if lo is None:
            return
        hi = hi if hi is not None else lo
        items.append({"lo": float(lo), "hi": float(hi), "source": source, "tf": tf})

    levels = ctx.get("htf_levels") or {}
    for h in levels.get("highs") or []:
        add(h, h, "htf_level_high")
    for l in levels.get("lows") or []:
        add(l, l, "htf_level_low")

    ob = ctx.get("order_blocks") or {}
    blocks = ob.get("blocks")
    if not blocks:  # deep-ctx: только ближайшие OB без общего списка
        blocks = [z for z in (ob.get("bullish_ob"), ob.get("bearish_ob")) if z]
    for blk in blocks:
        add(blk.get("low"), blk.get("high"),
            f"ob_{blk.get('type', '?')}", blk.get("tf"))

    fvg = ctx.get("fvg") or {}
    zones = fvg.get("fvgs")
    if not zones:
        zones = [z for z in (fvg.get("bullish_fvg"), fvg.get("bearish_fvg")) if z]
    for z in zones or []:
        add(z.get("low"), z.get("high"), f"fvg_{z.get('type', '?')}", z.get("tf"))

    vp = ctx.get("volume_profile") or {}
    for key in ("poc", "vah", "val"):
        add(vp.get(key), vp.get(key), key)

    liq = ctx.get("liquidity") or {}
    for h in liq.get("equal_highs") or []:
        add(h, h, "equal_highs")
    for l in liq.get("equal_lows") or []:
        add(l, l, "equal_lows")

    inds = ctx.get("inds_by_tf") or {tf: ctx.get(f"ind_{tf}") or {}
                                     for tf in ("1d", "4h")}
    for tf in ("1d", "4h"):
        ind = inds.get(tf) or {}
        for span in ("ema50", "ema200"):
            add(ind.get(span), ind.get(span), span, tf)
    return items


def _cluster_items(items: list[dict[str, Any]], price: float,
                   tol: float) -> list[LevelCluster]:
    """Слияние пересекающихся/близких (<= tol) уровней в кластеры."""
    if not items:
        return []
    ordered = sorted(items, key=lambda it: ((it["lo"] + it["hi"]) / 2, it["lo"]))
    clusters: list[LevelCluster] = []
    cur = dict(ordered[0], sources=[ordered[0]["source"]],
               tfs=[ordered[0]["tf"]] if ordered[0]["tf"] else [])
    for it in ordered[1:]:
        if it["lo"] <= cur["hi"] + tol:
            cur["hi"] = max(cur["hi"], it["hi"])
            cur["sources"].append(it["source"])
            if it["tf"]:
                cur["tfs"].append(it["tf"])
        else:
            clusters.append(_finalize_cluster(cur, price))
            cur = dict(it, sources=[it["source"]],
                       tfs=[it["tf"]] if it["tf"] else [])
    clusters.append(_finalize_cluster(cur, price))
    return clusters


def _finalize_cluster(raw: dict[str, Any], price: float) -> LevelCluster:
    center = (raw["lo"] + raw["hi"]) / 2
    return LevelCluster(
        lo=round(raw["lo"], 2), hi=round(raw["hi"], 2),
        kind="support" if center <= price else "resistance",
        sources=sorted(set(raw["sources"])),
        tfs=sorted(set(raw["tfs"])),
        strength=_cluster_strength(raw["sources"], raw["tfs"]))


def _cluster_strength(sources: list[str], tfs: list[str]) -> float:
    """0..1: число НЕЗАВИСИМЫХ источников + подтверждение на разных ТФ.

    Детерминированная формула этапа 1 (канонизируется на этапе 5):
    0.25 за каждый уникальный источник + 0.10 за каждый дополнительный ТФ.
    """
    return round(min(1.0, 0.25 * len(set(sources))
                     + 0.10 * max(0, len(set(tfs)) - 1)), 2)


def _entry_zone_candidates(ob: dict[str, Any], fvg: dict[str, Any],
                           price: float) -> list[LevelCluster]:
    """Зоны (OB/FVG), внутри которых находится цена, — кандидаты entry-зон."""
    out: list[LevelCluster] = []
    pairs = [(ob.get("bullish_ob"), ob.get("price_in_bullish_ob"), "ob_bullish"),
             (ob.get("bearish_ob"), ob.get("price_in_bearish_ob"), "ob_bearish"),
             (fvg.get("bullish_fvg"), fvg.get("price_in_bullish_fvg"), "fvg_bullish"),
             (fvg.get("bearish_fvg"), fvg.get("price_in_bearish_fvg"), "fvg_bearish")]
    for zone, inside, source in pairs:
        if zone and inside:
            out.append(_finalize_cluster(
                {"lo": float(zone["low"]), "hi": float(zone["high"]),
                 "sources": [source], "tfs": [zone["tf"]] if zone.get("tf") else []},
                price))
    return out


def _stop_tp_candidates(ctx: dict[str, Any], price: float,
                        atr: float) -> tuple[list[PricePoint], list[PricePoint]]:
    """Кандидаты стопов и целей — те же источники, что у _structural_stop и
    _structure_targets в pipeline (без буферов и фильтра 1R: это политика
    analyze(), а не контекста)."""
    levels = ctx.get("htf_levels") or {}
    ob = ctx.get("order_blocks") or {}
    vp = ctx.get("volume_profile") or {}
    stops: list[PricePoint] = []
    tps: list[PricePoint] = []

    # Зеркально канону pipeline._structure_targets: lows ниже цены — стоп для
    # LONG и структурная цель для SHORT; highs выше цены — стоп для SHORT и
    # цель для LONG. Уровни «не на своей стороне» кандидатами не являются.
    for l in levels.get("lows") or []:
        if l < price:
            stops.append(price_point(l, "structural", price, atr))
            tps.append(price_point(l, "structural", price, atr))
    for h in levels.get("highs") or []:
        if h > price:
            stops.append(price_point(h, "structural", price, atr))
            tps.append(price_point(h, "structural", price, atr))

    z = ob.get("bullish_ob")
    if z and z.get("low") is not None and z["low"] < price:
        stops.append(price_point(z["low"], "ob_edge", price, atr, z.get("tf")))
    z = ob.get("bearish_ob")
    if z and z.get("high") is not None and z["high"] > price:
        stops.append(price_point(z["high"], "ob_edge", price, atr, z.get("tf")))

    for key in ("poc", "vah", "val"):
        v = vp.get(key)
        if v:
            tps.append(price_point(v, "volume_profile", price, atr))

    stops.sort(key=lambda p: (p.atr_distance or 0, p.price))
    tps.sort(key=lambda p: (p.atr_distance or 0, p.price))
    return stops, tps


# ---------------------------------------------------------------------------
# Reversal / Derivatives / SpotFlow / Intermarket / Macro / News / DataQuality
# ---------------------------------------------------------------------------

def _reversal_read(tf: str, r: dict[str, Any]) -> ReversalRead:
    """Partial-данные (score/факторы) сохраняются даже без подтверждения —
    слабые звоночки нужны раннему предупреждению и legacy display-parity."""
    bull = bool(r.get("bullish_reversal"))
    bear = bool(r.get("bearish_reversal"))
    strong = bool(r.get("bull_strong") if bull
                  else r.get("bear_strong") if bear else False)
    return ReversalRead(
        tf=tf, bullish=bull, bearish=bear,
        confirmed_side="bull" if bull else "bear" if bear else None,
        strong=strong,
        bull_score=r.get("bull_score"),
        bear_score=r.get("bear_score"),
        bull_factors=list(r.get("factors_bull") or []),
        bear_factors=list(r.get("factors_bear") or []))


def _reversal_from_legacy(ctx: dict[str, Any], htf_bias: str | None,
                          meta: MetaContext, missing: set[str]) -> ReversalContext:
    mtf = ctx.get("reversal_mtf") or {}
    if not mtf:
        missing.add("reversal.mtf")
    per_tf = {tf: _reversal_read(tf, r)
              for tf, r in (mtf.get("per_tf") or {}).items()}
    bull_n = mtf.get("bull_tf_count", 0)
    bear_n = mtf.get("bear_tf_count", 0)
    if mtf.get("combined_bullish"):
        side, tf_count, confirmed = "bull", bull_n, mtf.get("bull_candle_confirm")
    elif mtf.get("combined_bearish"):
        side, tf_count, confirmed = "bear", bear_n, mtf.get("bear_candle_confirm")
    else:
        # Нет подтверждённой стороны -> tf_count = 0; partial-счёт сторон
        # доступен раздельно через bull_tf_count / bear_tf_count.
        side, tf_count, confirmed = None, 0, None
    alignment = (reversal_trend_alignment(side, htf_bias or "neutral")
                 if side else None)
    missing.update({"reversal.structure_shift", "reversal.probability",
                    "reversal.invalidation"})  # этап 6
    return ReversalContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=not mtf),
        per_tf=per_tf, combined_side=side, tf_count=tf_count,
        candle_confirmed=confirmed, trend_alignment=alignment,
        effect=_reversal_effect(side, alignment),
        bull_tf_count=bull_n, bear_tf_count=bear_n)


def _reversal_effect(side: str | None, alignment: str | None) -> ReversalEffect:
    """Правило этапа 1 (explanation-only, скоринг не читает):
    подтверждённый разворот ПО тренду — усиливает сценарий отката,
    ПРОТИВ тренда — ослабляет; BLOCK требует MSS (этап 6)."""
    if side is None:
        return ReversalEffect.NEUTRAL
    if alignment == "aligned":
        return ReversalEffect.AMPLIFY
    if alignment == "counter":
        return ReversalEffect.DAMPEN
    return ReversalEffect.NEUTRAL


def _derivatives_from_legacy(ctx: dict[str, Any], meta: MetaContext,
                             missing: set[str]) -> DerivativesContext:
    funding = ctx.get("funding") or {}
    ls = ctx.get("long_short_ratio") or {}
    if not funding:
        missing.add("derivatives.funding")
    if "oi_rising" not in ctx:
        missing.add("derivatives.oi_rising")
    missing.update({"derivatives.basis", "derivatives.oi_change",
                    "derivatives.price_oi_relation"})  # этап 2
    z = funding.get("zscore")
    return DerivativesContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=not funding),
        funding=funding.get("current"),
        funding_zscore=z,
        open_interest=ctx.get("open_interest") or None,
        oi_rising=ctx.get("oi_rising"),
        long_short_ratio=ls.get("ratio"),
        futures_cvd=ctx.get("cvd"),
        liquidation_map=ctx.get("liquidation_map"),
        sweep_signal=ctx.get("sweep_signal"),
        overheated=(abs(z) >= FUNDING_Z_EXTREME) if z is not None else None)


def _spot_flow_from_legacy(ctx: dict[str, Any], meta: MetaContext,
                           missing: set[str]) -> SpotFlowContext:
    onchain = ctx.get("onchain") or {}
    netflow = onchain.get("exchange_netflow")
    if netflow is None:
        missing.add("spot_flow.exchange_netflow")
    missing.update({"spot_flow.spot_cvd", "spot_flow.taker_imbalance",
                    "spot_flow.spot_led"})  # этап 7
    return SpotFlowContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=netflow is None),
        exchange_netflow=netflow)


def _intermarket_from_legacy(ctx: dict[str, Any], meta: MetaContext,
                             missing: set[str]) -> IntermarketContext:
    corr = ctx.get("correlation") or {}
    if not corr:
        missing.add("intermarket")
    return IntermarketContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=not corr),
        correlations=corr or None,
        verdict=corr.get("verdict"))


def _macro_from_legacy(ctx: dict[str, Any], meta: MetaContext,
                       missing: set[str]) -> MacroContext:
    macro = ctx.get("macro") or {}
    if not macro:
        missing.add("macro")
    missing.update({"macro.events", "macro.dxy"})  # календарь/DXY — этап 7
    return MacroContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=not macro),
        us10y=macro.get("us10y"),
        us10y_trend=macro.get("us10y_trend"))


def _news_from_legacy(meta: MetaContext, missing: set[str]) -> NewsContext:
    missing.add("news")  # news-pipeline — этап 7
    return NewsContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       degraded=True))


def _data_quality_from_legacy(ctx: dict[str, Any], meta: MetaContext,
                              missing: set[str]) -> DataQualityContext:
    last_close = meta.signal_candle_close_time
    freshness = None
    if last_close is not None:
        try:
            freshness = max((meta.decision_time - last_close).total_seconds(), 0.0)
        except TypeError:
            freshness = None
    latency = None
    fired = ctx.get("job_fired_at")
    if fired is not None:
        try:
            latency = max((meta.decision_time - fired).total_seconds(), 0.0)
        except TypeError:
            latency = None
    return DataQualityContext(
        meta=BlockMeta(source=_LEGACY_SOURCE, as_of=meta.gathered_at,
                       freshness_seconds=freshness),
        data_freshness_seconds=freshness,
        decision_latency_seconds=latency,
        stale=stale_data(last_close, meta.entry_timeframe,
                         now=meta.decision_time),
        executable_price_degraded=meta.executable_price_degraded,
        missing_blocks=sorted(missing),
        freshness_by_tf={meta.entry_timeframe: freshness}
        if freshness is not None else None)


# ---------------------------------------------------------------------------
# Решения: результат run_cascade -> SwingAnalysisResult / FinalDecision
# ---------------------------------------------------------------------------

def analysis_from_legacy(result: dict[str, Any],
                         profile: dict[str, Any]) -> SwingAnalysisResult:
    direction = result.get("direction") or result.get("candidate_direction")
    long_s, short_s = result.get("long_score"), result.get("short_score")
    return SwingAnalysisResult(
        mode=result.get("analysis_type") or profile.get("analysis_type", "SWING"),
        direction=direction,
        direction_probability=_direction_probability(direction, long_s, short_s),
        setup_quality=_setup_quality(result),
        execution_quality=_execution_quality(result, profile),
        contradiction_score=_contradiction_score(result, long_s, short_s),
        evidence={"category_scores": result.get("category_scores"),
                  "long_total": long_s, "short_total": short_s,
                  "regime": result.get("market_regime")},
        reasons=list(result.get("reasons") or []),
        counter_reasons=list(result.get("contradicting_factors") or []),
        mtf_alignment={"trends": result.get("mtf") or {},
                       **(result.get("mtf_agreement") or {})},
        regime_hierarchy={"1w": None,  # этап 8
                          "1d": result.get("market_regime"),
                          "12h": None, "4h": None, "1h": None},
        scenario=_scenario_from_legacy(result))


def decision_from_legacy(result: dict[str, Any], ctx: dict[str, Any],
                         profile: dict[str, Any]) -> FinalDecision:
    analysis = analysis_from_legacy(result, profile)
    gates = _gates_from_result(result)
    cancel = result.get("cancel_conditions") or []
    warnings = []
    if result.get("executable_price_degraded"):
        warnings.append("executable price degraded до close сигнальной свечи")
    if result.get("stale_decision"):
        warnings.append("решение позже лимита свежести режима — ENTER понижен до WAIT")
    return FinalDecision(
        decision=_decision_status(result),
        mode=analysis.mode,
        direction=analysis.direction,
        meta=_meta_from_legacy(ctx, profile),
        scenario=analysis.scenario,
        direction_probability=analysis.direction_probability,
        setup_quality=analysis.setup_quality,
        execution_quality=analysis.execution_quality,
        contradiction_score=analysis.contradiction_score,
        gates=gates,
        vetoes_triggered=[g.name for g in gates if not g.passed],
        reasons=analysis.reasons[:3],
        invalidation_note=cancel[0] if cancel else None,
        warnings=warnings)


def _decision_status(result: dict[str, Any]) -> Decision:
    status = result.get("status")
    if status == "alert":
        return (Decision.ENTER_LONG if result.get("direction") == "long"
                else Decision.ENTER_SHORT)
    if status in ("journal", "cooldown"):
        return Decision.WAIT
    return Decision.NO_TRADE


def _direction_probability(direction: str | None, long_s: Any,
                           short_s: Any) -> float | None:
    """Провизорно (shadow-only): доля evidence выбранного направления."""
    if direction is None or long_s is None or short_s is None:
        return None
    total = long_s + short_s
    if total <= 0:
        return None
    chosen = long_s if direction == "long" else short_s
    return round(chosen / total, 3)


def _setup_quality(result: dict[str, Any]) -> float | None:
    """Провизорно (shadow-only): взвешенный confluence-балл, нормированный
    к диапазону alert-порогов (10 ~ верх шкалы score)."""
    total = result.get("score_weighted")
    if total is None:
        total = result.get("score")
    if total is None:
        return None
    return round(min(float(total) / 10.0, 1.0), 3)


def _execution_quality(result: dict[str, Any],
                       profile: dict[str, Any]) -> float | None:
    """Провизорно (shadow-only): свежесть решения x смещение executable-цены.

    1.0 — решение сразу после закрытия свечи и executable == close;
    0.0 — свежесть за двойным лимитом режима или цена ушла на >= 1 ATR.
    """
    parts: list[float] = []
    freshness = result.get("data_freshness_seconds")
    max_delay = profile.get("max_decision_delay_seconds")
    if freshness is not None and max_delay:
        parts.append(_clamp01(1.0 - freshness / (2.0 * max_delay)))
    atr = result.get("atr")
    close = result.get("signal_close_price")
    executable = result.get("executable_price_at_decision")
    if atr and close is not None and executable is not None:
        displacement = abs(executable - close) / atr
        parts.append(_clamp01(1.0 - displacement))
    if not parts:
        return None
    quality = 1.0
    for p in parts:
        quality *= p
    return round(quality, 3)


def _contradiction_score(result: dict[str, Any], long_s: Any,
                         short_s: Any) -> float | None:
    """Провизорно (shadow-only): доля противоречащих доказательств.
    0 — чистый сетап, 0.5 — равные доказательства в обе стороны."""
    counter = result.get("counter_score")
    total = result.get("score_weighted")
    if counter is None and long_s is not None and short_s is not None:
        counter = min(long_s, short_s)
        total = max(long_s, short_s)
    if counter is None or total is None or (total + counter) <= 0:
        return None
    return round(counter / (total + counter), 3)


def _scenario_from_legacy(result: dict[str, Any]) -> Scenario | None:
    direction = result.get("direction")
    stop = result.get("stop_loss")
    if direction is None or stop is None:
        return None
    entry = result.get("entry_price") or result.get("price")
    atr = result.get("atr")
    tps = result.get("take_profit_levels") or [
        t for t in (result.get("target_1"), result.get("target_2")) if t]
    tp1 = tps[0] if tps else None
    tp2 = tps[1] if len(tps) > 1 else None
    tp1_prov = ("structural" if result.get("targets_structure")
                else "r_multiple_fallback")
    ez = result.get("entry_zone")
    zone = ((ez.get("low"), ez.get("high"))
            if isinstance(ez, dict) and ez.get("low") is not None else None)
    return Scenario(
        direction=direction,
        entry_zone=zone,
        stop=price_point(stop,
                         "structural" if result.get("stop_basis") == "structure"
                         else "atr", entry, atr),
        tp1=price_point(tp1, tp1_prov, entry, atr) if tp1 else None,
        tp2=price_point(tp2, result.get("tp2_source") or tp1_prov,
                        entry, atr) if tp2 else None,
        risk_reward=result.get("risk_reward"),
        expected_move_points=result.get("expected_move_points"),
        expected_move_atr=result.get("expected_move_atr"),
        setup_type=_setup_type(result))


def _setup_type(result: dict[str, Any]) -> str | None:
    """Эвристика этапа 1 (полноценная классификация — этап 8)."""
    reasons = result.get("reasons") or []
    if any(("Разворот" in r) or ("Дно" in r) or ("Пик" in r) for r in reasons):
        return "reversal"
    regime = result.get("market_regime")
    if regime in ("trend_up", "trend_down"):
        return "continuation"
    if regime == "range":
        return "range"
    return None


# Порядок стадий legacy-каскада. Blocked-результат несёт только точку
# останова; всё ДО неё считается пройденным, всё ПОСЛЕ — не исполнялось
# (в gates не попадает). Полный 25-гейт трейс — этап 3 (final_gate()).
_STAGE_ORDER = [
    "stale_data", "abnormal_volatility", "direction_conflict", "htf_filter",
    "diversity", "below_threshold", "dead_zone", "crowded_funding",
    "wait_for_sweep", "no_trade", "cooldown", "daily_limit", "freshness",
]


def _gates_from_result(result: dict[str, Any]) -> list[GateCheck]:
    status = result.get("status")
    if status == "blocked":
        stage = result.get("blocked_at") or "unknown"
        idx = _STAGE_ORDER.index(stage) if stage in _STAGE_ORDER else 0
        gates = [GateCheck(name=s, passed=True) for s in _STAGE_ORDER[:idx]]
        detail = None
        if stage == "no_trade":
            detail = "; ".join(result.get("no_trade_reasons") or []) or None
        gates.append(GateCheck(name=stage, passed=False, detail=detail))
        return gates

    passed_all = [GateCheck(name=s, passed=True)
                  for s in _STAGE_ORDER[:_STAGE_ORDER.index("cooldown")]]
    if status == "cooldown":
        return passed_all + [GateCheck(
            name="cooldown", passed=False,
            detail="действующий сетап в пределах cooldown — не новый вход")]
    passed_all.append(GateCheck(name="cooldown", passed=True))
    if status == "journal":
        score = result.get("score_weighted") or result.get("score") or 0
        if result.get("stale_decision"):
            passed_all.append(GateCheck(
                name="freshness", passed=False,
                detail="решение позже лимита свежести режима"))
        elif score >= config.SCORE_ALERT_MIN:
            passed_all.append(GateCheck(
                name="daily_limit", passed=False,
                detail="дневной лимит сигналов исчерпан"))
        else:
            passed_all.append(GateCheck(
                name="alert_threshold", passed=False,
                detail=f"score {score} ниже порога алерта {config.SCORE_ALERT_MIN}"))
        return passed_all
    if status == "alert":
        return passed_all + [GateCheck(name="daily_limit", passed=True),
                             GateCheck(name="freshness", passed=True)]
    # ignored / неизвестный статус
    return passed_all + [GateCheck(name="alert_threshold", passed=False)]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    to_py = getattr(value, "to_pydatetime", None)
    dt = to_py() if callable(to_py) else value
    if isinstance(dt, datetime) and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt if isinstance(dt, datetime) else None
