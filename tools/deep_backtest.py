"""Stage C1.3c: offline deep-backtest core на строгом датасете kline_dataset.

Read-only измерительный инструмент. НЕ торгует, НЕ отправляет ордера, НЕ ходит в
сеть, НЕ пишет в БД, НЕ меняет схему, scoring, thresholds, pipeline, scheduler,
backtest.py или live runtime. НЕ импортируется runtime-кодом — только ручной CLI.

    .venv/bin/python -m tools.deep_backtest \
      --dataset data/klines \
      --exchange binance \
      --symbol BTCUSDT \
      --profile intraday \
      --bars 5000 \
      --json

Что этот модуль ЕСТЬ:

  Точная копия воронки backtest._walk (порядок гейтов и семантика скоринга
  сохранены бит-в-бит), обвешанная измерительной аппаратурой:

    * skip-ledger — именованный счётчик КАЖДОГО `continue`, с примерами. В
      backtest._walk знаменатель отчёта — это `n - start`, то есть число
      пройденных баров, а не число баров, дошедших до сделки; сколько сетапов и
      ГДЕ именно отваливалось, было невидимо. Инвариант
      bars_walked == bars_evaluated + sum(skip_counts) не даёт потерять бар.

    * timeout — backtest._resolve держит позицию до конца истории и молча
      выбрасывает (`return None`) всё, что не закрылось. Здесь позиция живёт не
      дольше forecast_horizon_hours профиля и, если не закрылась, оценивается
      по рынку (mark-to-market) как "timeout". Настоящая нехватка будущих
      баров — отдельный исход "unresolved", он исключён из агрегатов.

    * dual regime tags — см. ниже.

Что этот модуль НЕ ЕСТЬ (сознательно, стадия C1.3c):

  Здесь нет walk-forward и нет рекомендации порога. Отчёт НЕ содержит поля
  recommended_threshold: сквозной прогон по одной истории — это in-sample
  подгонка, и называть её рекомендацией было бы враньём. Пороговая таблица
  приводится как измерение, а не как совет.

Dual regime tags (D13)
----------------------
backtest._walk считает режим так::

    regime = detect_regime(ind_htf if htf == "1d" else None)

У intraday htf == "4h", значит в detect_regime уходит None, и режим ВСЕГДА
"range" (см. regime.detect_regime: без 1D-индикаторов возврат "range"). Это и
есть drift point D13: интрадей-бэктест взвешивает скоры range-весами всегда,
а live-конвейер (pipeline.run_cascade) считает режим по ind_1d + volatility_1d.

C1.3c ИЗМЕРЯЕТ этот дрейф, а не чинит его:

  * ``regime_current``     — ровно текущая семантика backtest. ТОЛЬКО он
                             попадает в weighted_total и влияет на скоринг.
  * ``regime_live_parity`` — detect_regime(ind_1d_asof, volatility_1d_asof),
                             как в live. Это ТЕНЕВАЯ метаданная: она не
                             участвует ни в одном гейте и ни в одном скоре.

Расхождение видно в regime_confusion_matrix и regime_disagreement_rate. Чинить
D13 — задача отдельной стадии: смена режима меняет веса, а значит и пороги,
а значит и SCORE_ALERT_MIN.

Остальные известные дрейфы (funding/on-chain отсутствуют, стоп внутри свечи
считается сработавшим первым) сохранены как есть и перечислены в блоке
``limitations`` отчёта.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import config
from analyzer.cvd import compute_cvd_from_klines, cvd_series
from analyzer.divergence import detect_divergence
from analyzer.equilibrium import compute_equilibrium
from analyzer.indicators import compute_indicators
from analyzer.liquidity import detect_liquidity
from analyzer.reversal import detect_reversal
from analyzer.volatility import analyze_volatility
from analyzer.volume_profile import compute_volume_profile
from pipeline import (REVERSAL_TFS, _build_htf_zones, _near_key_level,
                      _stop_atr, _structural_stop, _structure_targets)
from risk.position_sizing import calculate_position
from signal_engine.confluence import (calculate_confluence_score,
                                      has_diverse_confirmation)
from signal_engine.htf_filter import apply_htf_policy, get_htf_bias
from signal_engine.profiles import PROFILES, get_profile
from signal_engine.regime import detect_regime, weighted_total
from signal_engine.no_trade_gate import effective_expected_move
from signal_engine.vetoes import TF_HOURS, abnormal_volatility, dead_zone
from tools.kline_dataset import (ENTRY_WARMUP, HTF_WARMUP, REGIME_1D_WARMUP,
                                 REGIME_TF, DatasetError, LoadedFrame,
                                 aux_depth, aux_depth_deficits, load_frames,
                                 required_timeframes)

log = logging.getLogger(__name__)

# Биржа выбирается явно, без failover: смешанный источник — это D15.
EXCHANGES = ("binance", "bybit")

# Тот же список порогов, что и в backtest.THRESHOLDS. Импортировать его оттуда
# нельзя: backtest.py на импорте тянет сетевой клиент биржи, а этот модуль обязан
# оставаться сетевым нулём. Значения продублированы осознанно и покрыты тестом.
THRESHOLDS = [5, 6, 7, 8, 9, 10]

# Окно 1D-фрейма для regime_live_parity. Live-конвейер (pipeline.
# gather_market_context) забирает klines(tf, limit=300) и отбрасывает
# формирующуюся свечу -> 299 закрытых баров. Берём столько же.
LIVE_1D_WINDOW = 299

# Окна срезов — из backtest._walk, не свободные параметры.
HTF_WINDOW = 250
ZONE_WINDOW = 160
ZONE_MIN_BARS = 30

# Режим, который нельзя посчитать: 1D-истории как-of этого бара не хватает на
# EMA200. Отдельная метка вместо тихого detect_regime(None) -> "range": иначе
# "нет данных" и "рынок в диапазоне" слились бы в одну клетку матрицы.
REGIME_UNAVAILABLE = "unavailable"

SKIP_EXAMPLES_MAX = 5

# Каждый `continue` воронки имеет имя. Порядок = порядок гейтов.
SKIP_REASONS: tuple[str, ...] = (
    "no_atr",
    "htf_insufficient_history",
    "no_zone_frames",
    "direction_conflict",
    "htf_policy_blocked",
    "insufficient_diverse_categories",
    "below_min_threshold",
    "dead_zone",
    "abnormal_volatility",
    "below_min_expected_move",
    "unresolved_no_future_data",
)

# Исходы, участвующие в агрегатах. "unresolved" сюда не входит: у него нет R.
RESOLVED_OUTCOMES = ("win", "loss", "breakeven", "timeout")
OUTCOMES = RESOLVED_OUTCOMES + ("unresolved",)

LIMITATIONS = [
    "C1.3c has no walk-forward: every number below is in-sample over one "
    "contiguous history.",
    "No recommended threshold is produced. The threshold table is a "
    "measurement, not advice.",
    "regime_live_parity is shadow metadata only: scoring and every gate use "
    "regime_current (D13 is measured here, not fixed).",
    "Funding / on-chain / liquidation-map history is unavailable, so the macro "
    "points and the wait-for-sweep gate are not reproduced (drift remains).",
    "Intra-candle order is unknown; the stop is assumed to hit before the "
    "target within the same candle (stop-first assumption).",
]


class DeepBacktestError(Exception):
    """Прогон невозможен: датасет или аргументы не удовлетворяют требованиям."""


# ---------------------------------------------------------------------------
# Skip ledger
# ---------------------------------------------------------------------------

@dataclass
class SkipLedger:
    """Счётчик пропущенных баров по причинам, с первыми N примерами.

    Существует ради одного инварианта: каждый пройденный бар либо дошёл до
    исхода, либо назван по имени. Молчаливый `continue` — это потерянный
    знаменатель, а на потерянном знаменателе винрейт можно нарисовать любой.
    """
    counts: dict[str, int] = field(
        default_factory=lambda: {r: 0 for r in SKIP_REASONS})
    examples: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {r: [] for r in SKIP_REASONS})

    def add(self, reason: str, idx: int, open_time: Any,
            details: str | None = None) -> None:
        if reason not in self.counts:
            raise KeyError(f"unknown skip reason: {reason!r}")
        self.counts[reason] += 1
        if len(self.examples[reason]) < SKIP_EXAMPLES_MAX:
            example: dict[str, Any] = {
                "idx": int(idx),
                "open_time_iso": _iso(open_time),
            }
            if details:
                example["details"] = details
            self.examples[reason].append(example)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "counts": dict(self.counts),
            "examples": {r: list(v) for r, v in self.examples.items()},
        }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.isoformat()


# ---------------------------------------------------------------------------
# Resolution: backtest._resolve + timeout + unresolved
# ---------------------------------------------------------------------------

def max_hold_bars(profile: dict[str, Any]) -> int:
    """Сколько entry-баров укладывается в forecast_horizon профиля.

    backtest._resolve держал позицию до конца истории: сделка, открытая за 3
    бара до конца, имела шанс закрыться, а такая же в начале истории — почти
    гарантированно закрывалась. Горизонт делает выборку однородной и совпадает
    с тем, что профиль обещает пользователю (forecast_horizon_hours).
    """
    tf_hours = TF_HOURS.get(profile["entry"], 1.0)
    horizon = float(profile.get("forecast_horizon_hours", 24.0))
    return max(1, math.ceil(horizon / tf_hours))


def resolve(df: pd.DataFrame, entry_idx: int, direction: str,
            pos: dict[str, Any], hold_bars: int) -> dict[str, Any]:
    """Жизненный цикл сделки: stop / TP1->breakeven / TP2, плюс два новых исхода.

    Семантика stop/TP1/TP2 скопирована из backtest._resolve БЕЗ изменений
    (в т.ч. правило "брейкивен-стоп действует со СЛЕДУЮЩЕЙ свечи после TP1",
    как в bot/journal.evaluate_trade) и консервативное допущение "внутри свечи
    стоп срабатывает первым".

    Добавлено:
      * ``timeout``    — на баре entry_idx + hold_bars позиция ещё открыта:
                         закрываем по close этого бара (mark-to-market).
      * ``unresolved`` — будущих баров не хватило даже до горизонта; исход
                         неизвестен. R не назначается: такой сетап исключается
                         из агрегатов, а не считается нулём.
    """
    entry, stop = pos["entry_price"], pos["stop_loss"]
    tp1, tp2 = pos["target_1"], pos["target_2"]
    risk = abs(entry - stop)
    long = direction == "long"
    hit_tp1 = False

    horizon_idx = entry_idx + hold_bars
    last_idx = min(horizon_idx, len(df) - 1)

    for j in range(entry_idx + 1, last_idx + 1):
        high, low = float(df["high"].iloc[j]), float(df["low"].iloc[j])
        if not hit_tp1:
            stop_hit = (low <= stop) if long else (high >= stop)
            tp1_hit = (high >= tp1) if long else (low <= tp1)
            if stop_hit:
                return {"outcome": "loss", "r": -1.0, "hit_tp1": False,
                        "exit_idx": j}
            if tp1_hit:
                hit_tp1 = True
                continue
        if hit_tp1:
            be_hit = (low <= entry) if long else (high >= entry)
            tp2_hit = (high >= tp2) if long else (low <= tp2)
            if be_hit:
                return {"outcome": "breakeven", "r": 0.0, "hit_tp1": True,
                        "exit_idx": j}
            if tp2_hit:
                r = round(abs(tp2 - entry) / risk, 2) if risk else 0.0
                return {"outcome": "win", "r": r, "hit_tp1": True, "exit_idx": j}

    if last_idx < horizon_idx:
        # История кончилась раньше горизонта — исход НЕИЗВЕСТЕН.
        return {"outcome": "unresolved", "r": None, "hit_tp1": hit_tp1,
                "exit_idx": None}

    close = float(df["close"].iloc[horizon_idx])
    sign = 1.0 if long else -1.0
    r = round(sign * (close - entry) / risk, 2) if risk else 0.0
    return {"outcome": "timeout", "r": r, "hit_tp1": hit_tp1,
            "exit_idx": horizon_idx}


# ---------------------------------------------------------------------------
# Regime tags
# ---------------------------------------------------------------------------

def regime_current(profile: dict[str, Any],
                   ind_htf: dict[str, Any] | None) -> str:
    """РОВНО текущая семантика backtest._walk. Только этот тег кормит скоринг.

    Для intraday/bounce/... где htf != "1d", сюда уходит None и результат всегда
    "range" — это и есть D13. Не чинить здесь.
    """
    return detect_regime(ind_htf if profile["htf"] == REGIME_TF else None)


def regime_live_parity(ind_1d: dict[str, Any] | None,
                       vol_1d: dict[str, Any] | None) -> str:
    """Режим так, как его считает live (pipeline.run_cascade). ТЕНЕВОЙ тег.

    Не передаётся ни в weighted_total, ни в один гейт. Если 1D-истории не
    хватает на EMA200 — честное "unavailable", а не молчаливое "range".
    """
    if ind_1d is None:
        return REGIME_UNAVAILABLE
    return detect_regime(ind_1d, vol_1d)


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """regime_current x regime_live_parity -> счётчики."""
    matrix: dict[str, dict[str, int]] = {}
    for current, parity in pairs:
        matrix.setdefault(current, {})
        matrix[current][parity] = matrix[current].get(parity, 0) + 1
    return matrix


def disagreement_rate(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    disagree = sum(1 for a, b in pairs if a != b)
    return round(disagree / len(pairs), 6)


# ---------------------------------------------------------------------------
# As-of срезы
# ---------------------------------------------------------------------------

def _close_ms(frame: LoadedFrame) -> np.ndarray:
    """close_time в миллисекундах. NaT — отказ, а не тихий мусор в int64.

    pandas приводит NaT к -9223372036854775808; такая «дата» прошла бы
    searchsorted и молча сдвинула КАЖДЫЙ as-of срез. Проверка до арифметики.
    """
    df = frame.df
    if "close_time" not in df.columns:
        raise DatasetError(
            f"{frame.data_path}: close_time column is required for as-of slicing")
    col = df["close_time"]
    if col.isna().any():
        raise DatasetError(f"{frame.data_path}: close_time contains NaT")
    return col.astype("int64").to_numpy() // 1_000_000


def _asof_end(close_ms: np.ndarray, t_ms: int) -> int:
    """Число баров с close_time <= t. Эквивалент df[df.close_time <= t]."""
    return int(np.searchsorted(close_ms, t_ms, side="right"))


class _IndicatorCache:
    """compute_indicators на as-of срезе зависит только от конца среза.

    Срезы старших ТФ меняются раз в 4h/12h/1d, а обход идёт по 15m-барам: без
    кэша 96% вызовов compute_indicators повторяют предыдущий результат бит-в-бит.
    Ключ — (tf, end, window), поэтому кэш НЕ может изменить ни один результат.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, int, int], Any] = {}

    def get(self, tf: str, end: int, window: int, factory) -> Any:
        key = (tf, end, window)
        if key not in self._store:
            self._store[key] = factory()
        return self._store[key]


# ---------------------------------------------------------------------------
# Deep walk
# ---------------------------------------------------------------------------

def walk_start(n: int, bars: int) -> int:
    """Индекс первого обходимого entry-бара: warmup или хвост длиной bars.

    Ровно тот старт, что и в backtest._walk. Вынесен, чтобы обход (deep_walk) и
    проверка as-of выравнивания (aux_alignment) не разошлись: обе обязаны
    считать «первый пройденный бар» одинаково, иначе валидация проверяла бы не
    тот бар, с которого начнётся прогон.
    """
    return max(ENTRY_WARMUP, n - bars)


def deep_walk(frames: dict[str, LoadedFrame], profile: dict[str, Any],
              bars: int) -> dict[str, Any]:
    """Копия воронки backtest._walk с skip-ledger и двойным тегом режима.

    Порядок гейтов и семантика скоринга не менялись. Единственные добавления —
    измерительные: счётчики, теневой режим, timeout/unresolved.
    """
    entry_tf = profile["entry"]
    htf = profile["htf"]
    zone_tfs = profile["zone_tfs"]

    entry_df = frames[entry_tf].df
    n = len(entry_df)
    hold_bars = max_hold_bars(profile)

    cost_pct = (2 * config.TAKER_FEE_PCT + config.SLIPPAGE_PCT) / 100

    close_ms = {tf: _close_ms(f) for tf, f in frames.items()}
    entry_close_ms = close_ms[entry_tf]
    cache = _IndicatorCache()

    ledger = SkipLedger()
    qualified: list[dict[str, Any]] = []
    regime_pairs: list[tuple[str, str]] = []

    # Тот же старт, что и в backtest._walk: warmup или хвост длиной bars.
    # Последний бар не обходится — у него нет ни одного будущего бара.
    start = walk_start(n, bars)
    bars_walked = 0

    for i in range(start, n - 1):
        bars_walked += 1
        open_time = entry_df["open_time"].iloc[i]

        entry_sub = entry_df.iloc[max(0, i - (ENTRY_WARMUP - 1)):i + 1]
        ind = compute_indicators(entry_sub)
        atr = ind.get("atr")
        if not atr:
            ledger.add("no_atr", i, open_time)
            continue
        t_ms = int(entry_close_ms[i])
        price = float(entry_df["close"].iloc[i])

        htf_end = _asof_end(close_ms[htf], t_ms)
        htf_slice = frames[htf].df.iloc[max(0, htf_end - HTF_WINDOW):htf_end]
        if len(htf_slice) < HTF_WARMUP:
            ledger.add("htf_insufficient_history", i, open_time,
                       f"{htf} bars={len(htf_slice)} < {HTF_WARMUP}")
            continue
        ind_htf = cache.get(htf, htf_end, HTF_WINDOW,
                            lambda: compute_indicators(htf_slice))
        htf_bias = get_htf_bias(ind_htf)

        zdfs: dict[str, Any] = {}
        zinds: dict[str, Any] = {}
        for tf in zone_tfs:
            end = _asof_end(close_ms[tf], t_ms)
            s = frames[tf].df.iloc[max(0, end - ZONE_WINDOW):end]
            if len(s) >= ZONE_MIN_BARS:
                zdfs[tf] = s
                zinds[tf] = cache.get(tf, end, ZONE_WINDOW,
                                      lambda s=s: compute_indicators(s))
        if not zdfs:
            ledger.add("no_zone_frames", i, open_time)
            continue

        zones = _build_htf_zones(zdfs, zinds, list(zdfs.keys()), price, entry_sub)
        ob, fvg, levels = zones["order_blocks"], zones["fvg"], zones["levels"]

        stop_atr = _stop_atr(profile, {**zinds, entry_tf: ind, htf: ind_htf}, atr)

        eq = compute_equilibrium(htf_slice)
        liq = detect_liquidity(entry_sub, atr_value=atr)
        at_level = _near_key_level(price, levels, ob, fvg,
                                   max(atr * 0.3, price * 0.002))
        rev = detect_reversal(entry_sub, ind, cvd_series(entry_sub),
                              at_key_level=at_level)

        rev_frames: dict[str, Any] = {}
        if entry_tf in REVERSAL_TFS:
            rev_frames[entry_tf] = (entry_sub, ind)
        if htf in REVERSAL_TFS:
            rev_frames.setdefault(htf, (htf_slice, ind_htf))
        for tf in zdfs:
            if tf in REVERSAL_TFS:
                rev_frames.setdefault(tf, (zdfs[tf], zinds[tf]))
        bull_tfs = bear_tfs = 0
        for tf, (df_tf, ind_tf) in rev_frames.items():
            if tf == entry_tf:
                r_tf = rev
            else:
                r_tf = detect_reversal(df_tf, ind_tf, cvd_series(df_tf),
                                       at_key_level=at_level)
            bull_tfs += bool(r_tf.get("bullish_reversal"))
            bear_tfs += bool(r_tf.get("bearish_reversal"))

        ser = ind.get("_series", {})
        drsi = detect_divergence(entry_sub, ser.get("rsi"))
        dmacd = detect_divergence(entry_sub, ser.get("macd"))
        cvd = compute_cvd_from_klines(entry_sub)

        flat = dict(ind)
        flat.update({
            "bullish_divergence": drsi["bullish_divergence"] or dmacd["bullish_divergence"],
            "bearish_divergence": drsi["bearish_divergence"] or dmacd["bearish_divergence"],
            "price_in_bullish_ob": ob["price_in_bullish_ob"],
            "price_in_bearish_ob": ob["price_in_bearish_ob"],
            "rejection_wick": ob["rejection_wick"],
            "liquidity_swept_below": liq["liquidity_swept_below"],
            "liquidity_swept_above": liq["liquidity_swept_above"],
            "reversal_candle": liq["reversal_candle"],
            "price_in_bullish_fvg": fvg["price_in_bullish_fvg"],
            "price_in_bearish_fvg": fvg["price_in_bearish_fvg"],
            "in_discount": eq["zone"] == "discount",
            "in_premium": eq["zone"] == "premium",
            "bullish_reversal": rev["bullish_reversal"],
            "bearish_reversal": rev["bearish_reversal"],
            "reversal_strong_bull": rev["bull_strong"],
            "reversal_strong_bear": rev["bear_strong"],
            "cvd_bullish": cvd["cvd_bullish"],
            "cvd_bearish": cvd["cvd_bearish"],
            "funding": None, "exchange_netflow": None,
            "trend_aligned_bottom": htf_bias == "bullish" and bull_tfs >= 2,
            "trend_aligned_top": htf_bias == "bearish" and bear_tfs >= 2,
        })

        _, ls, _ = calculate_confluence_score(flat, "long")
        _, ss, _ = calculate_confluence_score(flat, "short")

        # --- dual regime tags -------------------------------------------
        # regime_cur кормит скоринг. parity — только метаданные.
        regime_cur = regime_current(profile, ind_htf)
        parity = _live_parity_tag(frames, close_ms, cache, t_ms)
        regime_pairs.append((regime_cur, parity))

        lt = weighted_total(ls, regime_cur)
        st = weighted_total(ss, regime_cur)
        if lt == st:
            ledger.add("direction_conflict", i, open_time, f"score={lt}")
            continue
        direction, total, scores = ("long", lt, ls) if lt > st else ("short", st, ss)

        policy_ctx = _htf_policy_ctx(rev, bull_tfs, bear_tfs, eq, liq, ob, fvg,
                                     cvd, flat["bullish_divergence"],
                                     flat["bearish_divergence"])
        if apply_htf_policy(direction, htf_bias, profile, policy_ctx) is None:
            ledger.add("htf_policy_blocked", i, open_time,
                       f"{direction} vs htf_bias={htf_bias}")
            continue
        if not has_diverse_confirmation(scores, config.MIN_DIVERSE_CATEGORIES):
            ledger.add("insufficient_diverse_categories", i, open_time)
            continue
        if total < min(THRESHOLDS):
            ledger.add("below_min_threshold", i, open_time, f"total={total}")
            continue
        if dead_zone(scores.get("structure", 0), eq):
            ledger.add("dead_zone", i, open_time)
            continue
        if abnormal_volatility(analyze_volatility(
                entry_sub, tf_per_day=24.0 / TF_HOURS.get(entry_tf, 1.0))):
            ledger.add("abnormal_volatility", i, open_time)
            continue

        pos = calculate_position(price, stop_atr, direction=direction,
                                 atr_multiplier=profile["atr_mult"],
                                 targets_r=profile["targets"])

        ctx_min = {"htf_levels": levels, "order_blocks": ob,
                   "volume_profile": compute_volume_profile(
                       zdfs.get(zone_tfs[0], entry_sub))}
        s_stop = _structural_stop(ctx_min, direction, price, stop_atr, profile)
        if s_stop is not None:
            sign = 1 if direction == "long" else -1
            dist = abs(price - s_stop)
            pos.update({
                "stop_loss": round(s_stop, 2),
                "target_1": round(price + sign * dist * profile["targets"][0], 2),
                "target_2": round(price + sign * dist * profile["targets"][1], 2),
            })
        risk = abs(price - pos["stop_loss"])
        tp1, tp2, _, _tp2_src = _structure_targets(ctx_min, direction, price, risk,
                                                   pos["target_1"], pos["target_2"])
        pos["target_1"], pos["target_2"] = tp1, tp2

        move = effective_expected_move(
            price, pos["target_2"], atr,
            profile.get("forecast_horizon_hours", 24.0),
            TF_HOURS.get(entry_tf, 1.0))
        if move < profile.get("minimum_expected_move_points", 0):
            ledger.add("below_min_expected_move", i, open_time,
                       f"move={move}")
            continue

        outcome = resolve(entry_df, i, direction, pos, hold_bars)
        setup = {
            "idx": i,
            "open_time_iso": _iso(open_time),
            "score": total,
            "direction": direction,
            "regime_current": regime_cur,
            "regime_live_parity": parity,
            **outcome,
        }
        if outcome["outcome"] == "unresolved":
            # Не сетап-без-исхода, а бар без будущего: он не имеет права
            # попасть в знаменатель winrate. Учитывается в ledger.
            ledger.add("unresolved_no_future_data", i, open_time,
                       f"{direction} score={total}")
            qualified.append(setup)
            continue

        risk_dist = abs(price - pos["stop_loss"])
        cost_r = (cost_pct * price / risk_dist) if risk_dist else 0.0
        setup["r"] = round(outcome["r"] - cost_r, 2)
        qualified.append(setup)

    resolved = [s for s in qualified if s["outcome"] != "unresolved"]
    bars_evaluated = len(resolved)
    if bars_walked != bars_evaluated + ledger.total:
        raise DeepBacktestError(
            f"skip ledger invariant broken: bars_walked={bars_walked} != "
            f"bars_evaluated={bars_evaluated} + skips={ledger.total}")

    return {
        "bars_walked": bars_walked,
        "bars_evaluated": bars_evaluated,
        "max_hold_bars": hold_bars,
        "setups": qualified,
        "ledger": ledger,
        "regime_pairs": regime_pairs,
        "cooldown_bars": max(
            1, int(round(profile["cooldown_hours"] / TF_HOURS.get(entry_tf, 1)))),
    }


def _live_parity_tag(frames: dict[str, LoadedFrame], close_ms: dict[str, np.ndarray],
                     cache: _IndicatorCache, t_ms: int) -> str:
    """Теневой режим по 1D as-of, как в live. Не влияет ни на что."""
    end = _asof_end(close_ms[REGIME_TF], t_ms)
    slice_1d = frames[REGIME_TF].df.iloc[max(0, end - LIVE_1D_WINDOW):end]
    if len(slice_1d) < REGIME_1D_WARMUP:
        return REGIME_UNAVAILABLE
    ind_1d = cache.get(REGIME_TF, end, LIVE_1D_WINDOW,
                       lambda: compute_indicators(slice_1d))
    vol_1d = cache.get("_vol_1d", end, LIVE_1D_WINDOW,
                       lambda: analyze_volatility(slice_1d, tf_per_day=1.0))
    return regime_live_parity(ind_1d, vol_1d)


def _htf_policy_ctx(rev: dict[str, Any], bull_tfs: int, bear_tfs: int,
                    eq: dict[str, Any], liq: dict[str, Any],
                    ob: dict[str, Any], fvg: dict[str, Any],
                    cvd: dict[str, Any], bullish_div: bool,
                    bearish_div: bool) -> dict[str, Any]:
    """Идентична backtest._htf_policy_ctx. ``funding`` намеренно ОТСУТСТВУЕТ:
    подделанный пустой dict прошёл бы условие "фандинг не против нас" вакуумно
    и раздул бы результаты counter-trend профиля."""
    return {
        "reversal": rev,
        "reversal_mtf": {"bull_tf_count": bull_tfs, "bear_tf_count": bear_tfs},
        "equilibrium": eq,
        "liquidity": liq,
        "order_blocks": ob,
        "fvg": fvg,
        "cvd": cvd,
        "divergence": {"bullish_divergence": bullish_div,
                       "bearish_divergence": bearish_div},
    }


# ---------------------------------------------------------------------------
# Агрегаты
# ---------------------------------------------------------------------------

def outcome_counts(setups: list[dict[str, Any]]) -> dict[str, int]:
    return {o: sum(1 for s in setups if s["outcome"] == o) for o in OUTCOMES}


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"trades": 0, "win_rate": None, "expectancy_r": None, "sum_r": 0.0}
    wins = sum(1 for r in rows if r["outcome"] == "win")
    sum_r = round(sum(r["r"] for r in rows), 2)
    return {
        "trades": len(rows),
        "win_rate": round(wins / len(rows), 4),
        "expectancy_r": round(sum_r / len(rows), 4),
        "sum_r": sum_r,
    }


def threshold_stats(setups: list[dict[str, Any]], threshold: float,
                    cooldown_bars: int) -> dict[str, Any]:
    """Статистика одного порога с тем же кулдауном, что и в backtest._aggregate.

    Кулдаун применяется к ПОЛНОМУ потоку сетапов, включая unresolved: в live
    такой сетап тоже занял бы слот и заблокировал следующий. Но в агрегаты
    производительности unresolved не попадает — исход неизвестен, а не нулевой.
    """
    taken, last_idx = [], -10 ** 9
    for s in setups:
        if s["score"] < threshold or s["idx"] - last_idx < cooldown_bars:
            continue
        taken.append(s)
        last_idx = s["idx"]

    unresolved = [s for s in taken if s["outcome"] == "unresolved"]
    resolved = [s for s in taken if s["outcome"] != "unresolved"]
    timeouts = [s for s in resolved if s["outcome"] == "timeout"]
    no_timeout = [s for s in resolved if s["outcome"] != "timeout"]

    return {
        "threshold": threshold,
        "trades": len(resolved),
        "qualified": len(taken),
        "unresolved": len(unresolved),
        "unresolved_share": round(len(unresolved) / len(taken), 4) if taken else 0.0,
        "timeout_share": round(len(timeouts) / len(resolved), 4) if resolved else 0.0,
        "including_timeouts": _stats(resolved),
        "excluding_timeouts": _stats(no_timeout),
    }


# ---------------------------------------------------------------------------
# Aux as-of alignment (D14, C1.3c)
# ---------------------------------------------------------------------------
#
# aux_depth (kline_dataset) доказывает КОЛИЧЕСТВО баров, но не то, что эти бары
# лежат ДО первого пройденного entry-бара по времени. Датасет с верным счётом,
# но сдвинутым вперёд aux-фреймом (частый след старого limit=500: старшие ТФ
# покрывают меньше истории, чем entry) проходил бы aux_depth и затем тихо
# осыпался бы в skip-ledger (htf_insufficient_history), а для 1d — молча
# сползал бы в regime "unavailable", искажая знаменатель матрицы режимов.
#
# aux_alignment проверяет ровно то же условие, что и deep_walk: сколько баров
# каждого aux-ТФ имеют close_time <= t0, где t0 — close_time первого пройденного
# entry-бара (walk_start). Требуемая глубина = warmup роли (тот же guard, что и
# в обходе). entry-ТФ исключён: его warmup позиционный (iloc-срез) и уже покрыт
# aux_depth. Функция НЕ падает — возвращает таблицу; решение принимает prepare.

# required as-of по ролям — те же guard'ы, что применяет deep_walk на каждом баре.
_ASOF_REQUIRED = {"htf": HTF_WARMUP, "zone": ZONE_MIN_BARS,
                  "regime_1d": REGIME_1D_WARMUP}


def _aux_roles(profile: dict[str, Any]) -> dict[str, set[str]]:
    """tf -> роли. Один ТФ может играть несколько ролей (у swing htf == 1d)."""
    roles: dict[str, set[str]] = {}
    roles.setdefault(profile["entry"], set()).add("entry")
    roles.setdefault(profile["htf"], set()).add("htf")
    for tf in profile.get("zone_tfs", []):
        roles.setdefault(tf, set()).add("zone")
    roles.setdefault(REGIME_TF, set()).add("regime_1d")
    return roles


def aux_alignment(frames: dict[str, LoadedFrame], profile: dict[str, Any],
                  bars: int) -> dict[str, dict[str, Any]]:
    """As-of покрытие каждого aux-ТФ на первом пройденном entry-баре.

    Использует те же примитивы, что и обход (_close_ms, _asof_end, walk_start) и
    то же условие close_time <= t0. entry-ТФ пропускается (роль warmup у него
    позиционная). Возвращает по строке на aux-ТФ и НЕ бросает исключений.

    Недоступность close_time (колонки нет или в ней NaT — _close_ms поднимает
    DatasetError) НЕ прерывает валидацию стеком: такой ТФ становится дефицитной
    строкой с полем ``error``, и prepare() отказывает штатным путём выравнивания.
    Отсутствие entry close_time делает as-of невычислимым для ВСЕХ aux-ТФ — тогда
    каждая строка несёт эту ошибку. Каждая строка имеет ключ ``error`` (None у
    выровненных), чтобы форма таблицы не зависела от исхода.
    """
    entry_tf = profile["entry"]
    entry_df = frames[entry_tf].df
    n = len(entry_df)

    table: dict[str, dict[str, Any]] = {}
    i0 = walk_start(n, bars)
    if i0 >= n - 1:
        # Обходить нечего (aux_depth по entry уже отсёк бы такой датасет).
        return table

    first_open = entry_df["open_time"].iloc[i0]
    first_open_ms = int(pd.Timestamp(first_open).value // 1_000_000)
    first_open_iso = _iso(first_open)

    # entry close_time — общий якорь t0. Его недоступность = as-of невычислим для
    # всех aux; представляем дефицитом, а не исключением из aux_alignment.
    t0: int | None = None
    entry_error: str | None = None
    try:
        t0 = int(_close_ms(frames[entry_tf])[i0])
    except DatasetError as exc:
        entry_error = f"entry {entry_tf} close_time: {exc}"

    for tf, roles in _aux_roles(profile).items():
        aux_roles = roles - {"entry"}
        if not aux_roles:  # чистый entry-ТФ — не проверяем
            continue
        required = max(_ASOF_REQUIRED[r] for r in aux_roles)
        available = 0
        error = entry_error
        if t0 is not None:
            if tf not in frames:
                error = f"{tf} frame is missing"
            else:
                try:
                    available = _asof_end(_close_ms(frames[tf]), t0)
                    error = None
                except DatasetError as exc:
                    available, error = 0, str(exc)
        deficit = max(0, required - available)
        table[tf] = {
            "roles": sorted(roles),
            "required_asof": required,
            "available_asof": available,
            "first_walked_open_time": first_open_ms,
            "first_walked_open_time_iso": first_open_iso,
            "deficit": deficit,
            "error": error,
            "ok": deficit == 0 and error is None,
        }
    return table


def aux_alignment_deficits(table: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Не-ok ТФ -> дефицит. Строка с error (close_time недоступен) тоже дефицит."""
    return {tf: row["deficit"] for tf, row in table.items() if not row["ok"]}


# ---------------------------------------------------------------------------
# Датасет
# ---------------------------------------------------------------------------

def provenance(frames: dict[str, LoadedFrame]) -> dict[str, dict[str, Any]]:
    return {
        tf: {
            "data_file": f.manifest.get("data_file"),
            "first_open_time": _iso(f.df["open_time"].iloc[0]),
            "last_open_time": _iso(f.df["open_time"].iloc[-1]),
            "actual_bars": int(f.bars),
            "gap_count": int(f.gaps["gap_count"]),
            "missing_bars": int(f.gaps["missing_bars"]),
            "has_taker_buy_base": bool(f.has_taker_buy_base),
        }
        for tf, f in frames.items()
    }


def prepare(dataset: str, exchange: str, symbol: str, profile_name: str,
            bars: int, max_gap_ratio: float,
            allow_estimated_cvd: bool) -> tuple[dict[str, LoadedFrame],
                                                dict[str, Any], dict[str, Any], str]:
    """Загрузить, провалидировать и отказать НА ВХОДЕ, а не посреди обхода.

    Возвращает (frames, profile, aux_depth_table, cvd_method). Любое из трёх
    условий — дефицит глубины (D14), смешанный источник (D15) или отсутствие
    taker_buy_base без явного флага — это отказ, а не предупреждение.
    """
    if bars <= 0:
        raise DeepBacktestError("--bars must be positive")

    # get_profile молча откатывается на swing для неизвестного имени — здесь
    # это недопустимо: прогон не того профиля выглядит как валидный отчёт.
    if profile_name not in PROFILES:
        raise DeepBacktestError(f"unknown profile: {profile_name!r}")
    profile = get_profile(profile_name)

    tfs = required_timeframes(profile)
    frames = load_frames(dataset, exchange, symbol, tfs,
                         max_gap_ratio=max_gap_ratio)

    table = aux_depth(profile, bars, frames)
    deficits = aux_depth_deficits(table)
    if deficits:
        detail = ", ".join(f"{tf}: {d} bars short" for tf, d in sorted(deficits.items()))
        raise DeepBacktestError(
            f"insufficient dataset depth for {bars} entry bars (D14) — {detail}; "
            f"re-run tools.kline_cache with a larger --bars for those timeframes")

    # As-of выравнивание: счёт может быть верным, а aux-фрейм — сдвинут вперёд.
    # Проверяем ПЕРЕД обходом, иначе дефицит превратился бы в тихий mass-skip.
    alignment = aux_alignment(frames, profile, bars)
    misaligned = aux_alignment_deficits(alignment)
    if misaligned:
        first_iso = next(iter(alignment.values()))["first_walked_open_time_iso"]

        def _detail(tf: str) -> str:
            row = alignment[tf]
            base = (f"{tf}: needs {row['required_asof']} as-of bars "
                    f"(close_time<=t0) but only {row['available_asof']} precede "
                    f"the first walked bar ({row['deficit']} short)")
            if row.get("error"):
                base += f" [{row['error']}]"
            return base

        detail = ", ".join(_detail(tf) for tf in sorted(misaligned))
        raise DeepBacktestError(
            f"aux frame time-misaligned for {bars} entry bars (D14/as-of): first "
            f"walked bar {first_iso} — {detail}; increase kline_cache --bars for "
            f"that timeframe, and if the deficit persists, re-fetch all timeframes "
            f"together so their endpoints align")

    entry = frames[profile["entry"]]
    cvd_method = entry.cvd_method
    if cvd_method != "exact" and not allow_estimated_cvd:
        raise DeepBacktestError(
            f"{entry.data_path}: entry frame has no taker_buy_base — CVD would be "
            "ESTIMATED from candle close position, not exact (D15). Pass "
            "--allow-estimated-cvd to accept a different CVD semantics, or use a "
            "binance dataset")

    return frames, profile, table, cvd_method


def build_report(frames: dict[str, LoadedFrame], profile: dict[str, Any],
                 profile_name: str, exchange: str, symbol: str,
                 bars_requested: int, table: dict[str, Any],
                 alignment: dict[str, Any], cvd_method: str,
                 walk: dict[str, Any]) -> dict[str, Any]:
    """Отчёт C1.3c. Поля recommended_threshold здесь нет и быть не должно."""
    setups = walk["setups"]
    pairs = walk["regime_pairs"]
    return {
        "stage": "C1.3c",
        "profile": profile_name,
        "symbol": symbol,
        "exchange": exchange,
        "entry_timeframe": profile["entry"],
        "bars_requested": bars_requested,
        "bars_walked": walk["bars_walked"],
        "bars_evaluated": walk["bars_evaluated"],
        "cvd_method": cvd_method,
        "max_hold_bars": walk["max_hold_bars"],
        "cooldown_bars": walk["cooldown_bars"],
        "dataset": provenance(frames),
        "aux_depth": table,
        "aux_alignment": alignment,
        "skip_ledger": walk["ledger"].as_dict(),
        "regime_confusion_matrix": confusion_matrix(pairs),
        "regime_disagreement_rate": disagreement_rate(pairs),
        "regime_samples": len(pairs),
        "outcome_counts": outcome_counts(setups),
        "thresholds": [threshold_stats(setups, t, walk["cooldown_bars"])
                       for t in THRESHOLDS],
        "limitations": list(LIMITATIONS),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(dataset: str, exchange: str, symbol: str, profile_name: str, bars: int,
        max_gap_ratio: float = 0.001,
        allow_estimated_cvd: bool = False) -> dict[str, Any]:
    frames, profile, table, cvd_method = prepare(
        dataset, exchange, symbol, profile_name, bars, max_gap_ratio,
        allow_estimated_cvd)
    alignment = aux_alignment(frames, profile, bars)
    walk = deep_walk(frames, profile, bars)
    return build_report(frames, profile, profile_name, exchange, symbol, bars,
                        table, alignment, cvd_method, walk)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"deep backtest ({report['stage']}) — {report['exchange']} "
        f"{report['symbol']} {report['profile']} (entry {report['entry_timeframe']})",
        f"  bars           : requested {report['bars_requested']}, "
        f"walked {report['bars_walked']}, evaluated {report['bars_evaluated']}",
        f"  cvd_method     : {report['cvd_method']}",
        f"  max_hold_bars  : {report['max_hold_bars']}",
        "",
        "dataset provenance:",
        f"  {'tf':<4} {'bars':>6} {'gaps':>5} {'missing':>7} {'taker':>6}  "
        f"first_open .. last_open",
    ]
    for tf, row in report["dataset"].items():
        lines.append(
            f"  {tf:<4} {row['actual_bars']:>6} {row['gap_count']:>5} "
            f"{row['missing_bars']:>7} {str(row['has_taker_buy_base']):>6}  "
            f"{row['first_open_time']} .. {row['last_open_time']}")

    lines += ["", "aux depth (count):",
              f"  {'tf':<4} {'roles':<18} {'required':>8} {'available':>9} "
              f"{'deficit':>7} ok"]
    for tf, row in report["aux_depth"].items():
        lines.append(
            f"  {tf:<4} {','.join(row['roles']):<18} {row['required']:>8} "
            f"{row['available']:>9} {row['deficit']:>7} {row['ok']}")

    lines += ["", "aux as-of alignment (at first walked bar):",
              f"  {'tf':<4} {'roles':<18} {'req_asof':>8} {'avail_asof':>10} "
              f"{'deficit':>7} ok"]
    for tf, row in report["aux_alignment"].items():
        lines.append(
            f"  {tf:<4} {','.join(row['roles']):<18} {row['required_asof']:>8} "
            f"{row['available_asof']:>10} {row['deficit']:>7} {row['ok']}")

    lines += ["", "skip ledger:"]
    for reason, count in report["skip_ledger"]["counts"].items():
        lines.append(f"  {reason:<32} {count:>7}")
    lines += [
        "",
        f"regime disagreement rate: {report['regime_disagreement_rate']:.4f} "
        f"over {report['regime_samples']} samples",
    ]
    for current, row in sorted(report["regime_confusion_matrix"].items()):
        for parity, count in sorted(row.items()):
            lines.append(f"  current={current:<16} parity={parity:<16} {count:>7}")

    counts = report["outcome_counts"]
    lines += ["", "outcomes: " + "  ".join(f"{k}={v}" for k, v in counts.items()), ""]
    lines.append("thr │ trades │ win% │  ΣR   │ E[R]  │ timeout% │ unresolved%")
    lines.append("────┼────────┼──────┼───────┼───────┼──────────┼────────────")
    for row in report["thresholds"]:
        inc = row["including_timeouts"]
        wr = "  —  " if inc["win_rate"] is None else f"{inc['win_rate'] * 100:>4.0f}%"
        er = "  —  " if inc["expectancy_r"] is None else f"{inc['expectancy_r']:>+.2f}"
        lines.append(
            f" {row['threshold']:>2} │  {row['trades']:>4}  │{wr}│ "
            f"{inc['sum_r']:>+5.1f} │ {er} │  {row['timeout_share'] * 100:>5.1f}% │"
            f"   {row['unresolved_share'] * 100:>6.1f}%")

    lines += ["", "limitations:"]
    lines += [f"  - {item}" for item in report["limitations"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline deep backtest core on a pinned kline dataset "
                    "(Stage C1.3c). Reads cached files only; never touches the "
                    "network, the database, or an exchange.")
    parser.add_argument("--dataset", required=True,
                        help="directory written by tools.kline_cache")
    parser.add_argument("--exchange", required=True, choices=EXCHANGES)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--bars", type=int, required=True,
                        help="entry-timeframe bars to walk")
    parser.add_argument("--max-gap-ratio", type=float, default=0.001)
    parser.add_argument("--allow-estimated-cvd", action="store_true",
                        help="accept a dataset without taker_buy_base (D15): CVD "
                             "will be estimated from candle close position")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.bars <= 0:
        parser.error("--bars must be positive")

    try:
        report = run(args.dataset, args.exchange, args.symbol, args.profile,
                     args.bars, args.max_gap_ratio, args.allow_estimated_cvd)
    except (DeepBacktestError, DatasetError) as exc:
        print(f"deep_backtest: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
