"""Типизированные контракты unified-движка (этап 1: контракты + адаптеры).

Production-код (pipeline/scheduler/handlers) на этом этапе НЕ импортирует
contracts — зависимость односторонняя: contracts.legacy читает legacy-словари
и модули signal_engine, но ничего не меняет.
"""
from contracts.base import (BlockMeta, ContractError, Decision, PricePoint,
                            ReversalEffect, VolatilityRegime, price_point,
                            to_jsonable)
from contracts.context import (DataQualityContext, DerivativesContext,
                               EconEvent, IntermarketContext, LevelCluster,
                               LevelsContext, MacroContext, MetaContext,
                               NewsContext, NewsItem, ReversalContext,
                               ReversalRead, SpotFlowContext, TechnicalContext,
                               TFSnapshot, UnifiedMarketContext,
                               VolatilityContext)
from contracts.decision import (FinalDecision, GateCheck, Scenario,
                                SwingAnalysisResult, SynthesisResult)
from contracts.legacy import (analysis_from_legacy, context_from_legacy,
                              context_from_legacy_deep, decision_from_legacy)

__all__ = [
    "BlockMeta", "ContractError", "Decision", "PricePoint", "ReversalEffect",
    "VolatilityRegime", "price_point", "to_jsonable",
    "DataQualityContext", "DerivativesContext", "EconEvent",
    "IntermarketContext", "LevelCluster", "LevelsContext", "MacroContext",
    "MetaContext", "NewsContext", "NewsItem", "ReversalContext",
    "ReversalRead", "SpotFlowContext", "TechnicalContext", "TFSnapshot",
    "UnifiedMarketContext", "VolatilityContext",
    "FinalDecision", "GateCheck", "Scenario", "SwingAnalysisResult",
    "SynthesisResult",
    "analysis_from_legacy", "context_from_legacy", "context_from_legacy_deep",
    "decision_from_legacy",
]
