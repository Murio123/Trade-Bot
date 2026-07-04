"""Контракты UnifiedMarketContext (этап 1).

Каждый блок несёт BlockMeta (источник/свежесть/degraded). Поля без
legacy-источника на этапе 1 всегда None и задокументированы как «этап N»
— контракт фиксируется сейчас, чтобы последующие этапы наполняли его без
изменения формы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from contracts.base import (BlockMeta, PricePoint, ReversalEffect,
                            VolatilityRegime)


@dataclass(frozen=True)
class MetaContext:
    symbol: str                                # <- ctx["symbol"]
    analysis_type: str                         # <- profile["analysis_type"]
    entry_timeframe: str                       # <- ctx["timeframe"]
    gathered_at: datetime                      # <- ctx["timestamp"]
    decision_time: datetime                    # <- ctx["decision_time"]
    signal_candle_close_time: datetime | None  # <- ctx["last_close_time"]; None => stale-гейт
    signal_close_price: float                  # <- ctx["price"] (воспроизводимый якорь прогноза)
    executable_price: float                    # <- ctx["executable_price"] (живой тикер)
    executable_price_degraded: bool            # <- ctx["executable_price_degraded"]
    engine: str = "unified_v2"                 # идентификация движка в ledger
    engine_version: str = "0.1.0"
    prompt_version: str | None = None          # <- config.PROMPT_VERSION


@dataclass(frozen=True)
class TFSnapshot:
    """Технический срез одного таймфрейма с его ролью в иерархии режима."""
    tf: str
    role: str                                  # "regime"|"bias"|"phase"|"setup"|"execution"
    indicators: dict[str, Any]                 # <- ctx["inds_by_tf"][tf] без "_series"
    structure: dict[str, Any] | None = None    # <- ctx["structure_<tf>"] (есть только в deep-ctx)
    trend: str | None = None                   # <- mtf_confidence.trend_label(indicators)


@dataclass(frozen=True)
class TechnicalContext:
    meta: BlockMeta
    snapshots: dict[str, TFSnapshot]           # ключ = tf; "1w" появится на этапе 8
    htf_bias: str | None                       # <- get_htf_bias(inds[profile["htf"]])
    market_regime: str | None                  # <- detect_regime(ind_1d, volatility_1d)
    divergence: dict[str, Any] | None = None   # <- ctx["divergence"]
    sessions: dict[str, Any] | None = None     # <- ctx["sessions"] (explanation-only)
    equilibrium: dict[str, Any] | None = None  # <- ctx["equilibrium"]


@dataclass(frozen=True)
class VolatilityContext:
    meta: BlockMeta
    atr: float | None                          # <- ctx["atr"]
    atr_percent: float | None                  # atr / signal_close_price * 100 (адаптер)
    atr_percentile: float | None               # <- ctx["volatility"]["atr_percentile"]
    hv_percentile: float | None                # <- ctx["volatility"]["hv_percentile"]
    realized_volatility: float | None          # этап 2 (сырой RV): пока None
    range_width_atr: float | None              # этап 2: пока None
    candle_expansion_ratio: float | None       # этап 2: пока None
    compression_score: float | None            # этап 2 (BBW-перцентиль): пока None
    abnormal_volatility: bool | None           # <- vetoes.abnormal_volatility(ctx["volatility"])
                                               #    (ATR-перцентиль окна, НЕ свечной спайк)
    abnormal_candle: bool | None               # этап 2 (candle_expansion_ratio): пока None
    liquidation_event: bool | None             # этап 2: пока None
    regime: VolatilityRegime | None            # маппинг legacy regime/percentile; None => гейт NO_TRADE
    expected_move: dict[str, Any] | None = None  # <- ctx["volatility"]["expected_move"]


@dataclass(frozen=True)
class LevelCluster:
    lo: float                                  # границы зоны (точечный уровень: lo == hi)
    hi: float
    kind: str                                  # "support"|"resistance"
    sources: list[str]                         # provenance компонентов ("ob_bullish:12h", "poc", ...)
    tfs: list[str]                             # таймфреймы источников (без None)
    strength: float                            # 0..1, детерминированная формула (см. legacy._cluster_strength)


@dataclass(frozen=True)
class LevelsContext:
    meta: BlockMeta
    clusters: list[LevelCluster]               # union всех источников уровней, слитый по 0.25*ATR
    nearest_support: LevelCluster | None
    nearest_resistance: LevelCluster | None
    liquidity_above: list[PricePoint] = field(default_factory=list)   # <- liquidity.equal_highs
    liquidity_below: list[PricePoint] = field(default_factory=list)   # <- liquidity.equal_lows
    entry_zone_candidates: list[LevelCluster] = field(default_factory=list)  # OB/FVG, где цена сейчас
    stop_candidates: list[PricePoint] = field(default_factory=list)   # структура за ценой (без буфера)
    invalidation_candidates: list[PricePoint] = field(default_factory=list)
    tp_candidates: list[PricePoint] = field(default_factory=list)     # структура/VP по обе стороны
    range_position: dict[str, Any] | None = None                      # <- ctx["equilibrium"]
    distance_to_next_target_atr: float | None = None
    distance_to_invalidation_atr: float | None = None


@dataclass(frozen=True)
class ReversalRead:
    """Чтение одного ТФ. Слабые (неподтверждённые) сигналы сохраняются:
    partial-счёт и факторы обеих сторон нужны для роли «предупреждать»
    и display-parity с legacy /reversal («🟢· слабо (1/2)»)."""
    tf: str
    bullish: bool                              # подтверждён <- per_tf[tf]["bullish_reversal"]
    bearish: bool                              # подтверждён <- per_tf[tf]["bearish_reversal"]
    confirmed_side: str | None = None          # "bull"|"bear"|None — производное от флагов
    strong: bool = False                       # <- bull_strong / bear_strong подтверждённой стороны
    bull_score: int | None = None              # <- bull_score (и без подтверждения)
    bear_score: int | None = None              # <- bear_score (и без подтверждения)
    bull_factors: list[str] = field(default_factory=list)  # <- factors_bull (типизация — этап 6)
    bear_factors: list[str] = field(default_factory=list)  # <- factors_bear


@dataclass(frozen=True)
class ReversalContext:
    meta: BlockMeta
    per_tf: dict[str, ReversalRead]            # <- ctx["reversal_mtf"]["per_tf"]
    combined_side: str | None                  # "bull"|"bear"|None <- combined_bullish/bearish
    tf_count: int                              # ТФ, подтвердившие ИМЕННО combined_side; 0 без него
    candle_confirmed: bool | None              # <- bull/bear_candle_confirm (1H)
    trend_alignment: str | None                # <- vetoes.reversal_trend_alignment(side, htf_bias)
    effect: ReversalEffect                     # правило этапа 1 (explanation-only, см. legacy)
    bull_tf_count: int = 0                     # <- reversal_mtf["bull_tf_count"] (стороны раздельно)
    bear_tf_count: int = 0                     # <- reversal_mtf["bear_tf_count"]
    structure_shift_confirmed: bool | None = None  # этап 6 (MSS): пока None
    reversal_probability: float | None = None      # этап 6: пока None (без подгонки)
    invalidation: PricePoint | None = None         # этап 6: пока None


@dataclass(frozen=True)
class DerivativesContext:
    meta: BlockMeta
    funding: float | None                      # <- ctx["funding"]["current"]
    funding_zscore: float | None               # <- ctx["funding"]["zscore"]
    open_interest: float | None                # <- ctx["open_interest"]
    oi_rising: bool | None                     # <- swing-ctx["oi_rising"]; в основном ctx None
    long_short_ratio: float | None             # <- ctx["long_short_ratio"]["ratio"]
    futures_cvd: dict[str, Any] | None         # <- ctx["cvd"]
    liquidation_map: dict[str, Any] | None     # <- ctx["liquidation_map"]
    sweep_signal: dict[str, Any] | None        # <- ctx["sweep_signal"]
    overheated: bool | None                    # |funding_z| >= FUNDING_Z_EXTREME
    oi_change_pct: float | None = None         # этап 2 (из oi_hist): пока None
    price_oi_relation: str | None = None       # этап 2: пока None (рост OI сам по себе не направление)
    basis: float | None = None                 # этап 2: пока None


@dataclass(frozen=True)
class SpotFlowContext:
    meta: BlockMeta
    exchange_netflow: float | None             # <- ctx["onchain"]["exchange_netflow"]
    spot_volume: float | None = None           # этап 7: пока None
    spot_cvd: dict[str, Any] | None = None     # этап 7: пока None
    taker_imbalance: float | None = None       # этап 7: пока None
    spot_led: bool | None = None               # этап 7: spot- vs derivatives-led движение


@dataclass(frozen=True)
class IntermarketContext:
    meta: BlockMeta
    correlations: dict[str, Any] | None        # <- swing-ctx["correlation"]; в основном ctx None
    verdict: str | None                        # <- correlation["verdict"]
    vix: float | None = None                   # этап 7: пока None
    nasdaq_trend: str | None = None            # этап 7: пока None


@dataclass(frozen=True)
class EconEvent:
    """Макро-событие с point-in-time корректностью: решения (и бэктест)
    видят только то, что было received_at <= decision_time; revision
    никогда не участвует в решениях задним числом."""
    name: str
    category: str                              # "CPI"|"NFP"|"FOMC"|...
    scheduled_at: datetime
    received_at: datetime | None               # когда данные стали известны НАМ
    published_at: datetime | None = None       # None = ещё не вышло
    actual: float | None = None
    consensus: float | None = None
    previous: float | None = None
    revision: float | None = None              # explanation-only
    surprise: float | None = None              # (actual - consensus) / sigma
    source: str = "unknown"
    expiry: datetime | None = None             # когда событие перестаёт влиять


@dataclass(frozen=True)
class MacroContext:
    meta: BlockMeta
    us10y: float | None                        # <- ctx["macro"]["us10y"]
    us10y_trend: str | None                    # <- ctx["macro"]["us10y_trend"]
    dxy: float | None = None                   # этап 7: пока None
    historical_regime: str | None = None       # этап 7: пока None
    upcoming_events: list[EconEvent] = field(default_factory=list)   # этап 7 (календарь)
    released_events: list[EconEvent] = field(default_factory=list)   # этап 7
    event_risk_window: bool | None = None      # этап 7: станет входом event-veto


@dataclass(frozen=True)
class NewsItem:
    """Выход news-pipeline (этап 7); контракт фиксируется сейчас."""
    headline: str
    source: str
    published_at: datetime
    received_at: datetime                      # point-in-time граница для бэктеста
    topic: str | None = None
    entities: list[str] = field(default_factory=list)
    relevance_to_btc: float | None = None      # 0..1
    impact_direction: str | None = None        # "bullish"|"bearish"|"unclear"
    impact_strength: float | None = None       # 0..1
    novelty: float | None = None
    source_quality: float | None = None
    confirmation_count: int = 0                # критичное требует >=2 надёжных источников
    duplicate_cluster: str | None = None
    expiry: datetime | None = None             # time decay


@dataclass(frozen=True)
class NewsContext:
    meta: BlockMeta
    items: list[NewsItem] = field(default_factory=list)   # этап 7: пока []
    fear_greed: dict[str, Any] | None = None               # не в legacy ctx: пока None
    aggregate_sentiment: float | None = None               # этап 7
    event_risk: bool | None = None                         # этап 7
    breaking_news_risk: bool | None = None                 # этап 7


@dataclass(frozen=True)
class DataQualityContext:
    meta: BlockMeta
    data_freshness_seconds: float | None       # decision_time - last_close (entry-TF)
    decision_latency_seconds: float | None     # <- ctx["job_fired_at"] (если был)
    stale: bool                                # <- vetoes.stale_data(..., now=decision_time)
    executable_price_degraded: bool
    missing_blocks: list[str] = field(default_factory=list)  # всё, что degraded/None
    freshness_by_tf: dict[str, float] | None = None           # этап 2: пока только entry-TF


@dataclass(frozen=True)
class UnifiedMarketContext:
    meta: MetaContext
    technical: TechnicalContext
    volatility: VolatilityContext
    levels: LevelsContext
    reversal: ReversalContext
    derivatives: DerivativesContext
    spot_flow: SpotFlowContext
    intermarket: IntermarketContext
    macro: MacroContext
    news: NewsContext
    data_quality: DataQualityContext

    @classmethod
    def from_legacy(cls, ctx: dict[str, Any],
                    profile: dict[str, Any]) -> "UnifiedMarketContext":
        """Адаптер основного каскадного контекста (gather_market_context)."""
        from contracts.legacy import context_from_legacy
        return context_from_legacy(ctx, profile)

    @classmethod
    def from_legacy_deep(cls, ctx: dict[str, Any]) -> "UnifiedMarketContext":
        """Адаптер deep-контекста (gather_swing_context): часть блоков degraded."""
        from contracts.legacy import context_from_legacy_deep
        return context_from_legacy_deep(ctx)

    def to_json(self) -> dict[str, Any]:
        from contracts.base import to_jsonable
        return to_jsonable(self)
