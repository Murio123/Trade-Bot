"""Stage 15A: pure setup-lifecycle classifier.

Сравнивает ДВА уже персистируемых forecasts-ряда (предыдущий сопоставимый и
текущий) и выводит, что произошло с сетапом между ними: NEW_SETUP /
CONTINUATION / UPGRADED / DOWNGRADED / INVALIDATED / EXPIRED / RESOLVED — плюс
набор объясняющих флагов (change reasons). Это ДЕРИВАЦИЯ ПОСТФАКТУМ, а не
торговое решение:

  * НЕ меняет и не читает scoring, thresholds, final_gate, run_cascade, risk,
    DRY_RUN, live trading, AI, Telegram, scheduler, producer или схему БД;
  * НЕ принимает решений ENTER/WAIT/NO_TRADE — статус берётся из уже сохранённого
    ``analysis_status`` обоих рядов;
  * сравнивает только внутри одной стратегии: при расхождении
    strategy_version / context_version / symbol / analysis_type ряды НЕ
    сопоставимы и текущий трактуется как NEW_SETUP.

Leaf-модуль: только stdlib/typing/datetime. НЕ импортирует database, scheduler,
pipeline, signal_engine, tools или другой analyzer — остаётся полностью
независимым от runtime/decision-path и переиспользуемым offline-раннером
(tools/setup_lifecycle_report.py, следующий шаг).

Пороги (Thresholds) — шум-фильтр отчёта, НЕ торговые пороги: их назначение —
отсечь микро-дельты score/confidence, чтобы не плодить ложные upgrade/downgrade.
Дефолты подобраны консервативно и калибруются на offline-истории (Stage 15A).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# --- статусы жизненного цикла (стабильные строковые константы) ---------------
NEW_SETUP = "NEW_SETUP"
CONTINUATION = "CONTINUATION"
UPGRADED = "UPGRADED"
DOWNGRADED = "DOWNGRADED"
INVALIDATED = "INVALIDATED"
EXPIRED = "EXPIRED"
RESOLVED = "RESOLVED"

STATUSES = (NEW_SETUP, CONTINUATION, UPGRADED, DOWNGRADED,
            INVALIDATED, EXPIRED, RESOLVED)

ENTER, WAIT, NO_TRADE = "ENTER", "WAIT", "NO_TRADE"

# Ось «силы» состояния: чем выше ранг, тем ближе к входу.
_STATE_RANK = {NO_TRADE: 0, WAIT: 1, ENTER: 2}
# Активный сетап — то, что ещё может отработать (WAIT ждёт, ENTER в игре).
_ACTIVE = {WAIT, ENTER}

# Поля, определяющие сопоставимость двух рядов (иначе — разные «миры»).
_COMPARABLE_KEYS = ("symbol", "analysis_type", "strategy_version",
                    "context_version")


@dataclass(frozen=True)
class Thresholds:
    """Минимальные дельты, ниже которых изменение считается шумом (CONTINUATION).

    confidence_delta — по raw_confidence (шкала 0..1);
    score_delta      — по long_score/short_score (regime-weighted, шкала ~0..10+);
    max_gap_seconds  — максимальный разрыв между сопоставимыми рядами, сверх
                       которого активный сетап считается EXPIRED. Зависит от
                       таймфрейма mode — раннер задаёт его per analysis_type;
                       дефолт (6ч) рассчитан на 4H swing/positional.
    """
    confidence_delta: float = 0.05
    score_delta: float = 1.0
    max_gap_seconds: float = 6 * 3600


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class LifecycleResult:
    status: str
    reasons: tuple[str, ...]
    previous_id: int | None = None
    comparable: bool = True


def classify_transition(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    previous_outcome: dict[str, Any] | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> LifecycleResult:
    """Классифицировать переход сетапа из ``previous`` в ``current``.

    Чистая функция: результат зависит только от переданных полей рядов и порогов.
    ``previous_outcome`` — строка forecast_outcomes предыдущего ряда (или None);
    нужна лишь чтобы отличить RESOLVED (предыдущий ENTER уже отработал) от прочих
    переходов. None previous_outcome НИКОГДА не трактуется как resolved.

    Приоритет (первое совпадение выигрывает):
        1. нет предыдущего / несопоставим        -> NEW_SETUP
        2. предыдущий ENTER с resolved-исходом    -> RESOLVED
        3. разрыв > max_gap_seconds, предыдущий активен -> EXPIRED
        4. разворот направления, либо активный сетап убит новым гейтом/причиной
                                                   -> INVALIDATED
        5. сдвиг по оси состояния вверх/вниз       -> UPGRADED / DOWNGRADED
        6. та же ось+направление: сила выросла/упала >= порога -> UPGRADED /
           DOWNGRADED, иначе                        -> CONTINUATION
    """
    comparable = previous is not None and _is_comparable(previous, current)
    if not comparable:
        return LifecycleResult(NEW_SETUP, (), previous_id=None,
                               comparable=comparable)

    prev_id = previous.get("id")
    reasons = _change_reasons(previous, current, thresholds)

    prev_status = previous.get("analysis_status")
    cur_status = current.get("analysis_status")
    pr_rank = _STATE_RANK.get(prev_status)
    cur_rank = _STATE_RANK.get(cur_status)

    # 2. Предыдущий вход уже отыгран (TP/stop/72h) — терминальная отметка.
    if prev_status == ENTER and previous_outcome is not None \
            and previous_outcome.get("resolved"):
        return LifecycleResult(RESOLVED, reasons, previous_id=prev_id)

    # 3. Цепочка прервана по времени, а предыдущий сетап ещё был активен.
    gap = _gap_seconds(previous, current)
    if gap is not None and gap > thresholds.max_gap_seconds \
            and prev_status in _ACTIVE:
        return LifecycleResult(EXPIRED, reasons, previous_id=prev_id)

    # 4. Инвалидация: жёсткий разворот направления, либо активный сетап,
    #    обнулённый новым блокирующим фактором (упал в NO_TRADE).
    if "bias_flip" in reasons:
        return LifecycleResult(INVALIDATED, reasons, previous_id=prev_id)
    if cur_rank == 0 and (pr_rank or 0) >= 1 \
            and ("gate_appeared" in reasons or "reasons_added" in reasons):
        return LifecycleResult(INVALIDATED, reasons, previous_id=prev_id)

    # 5. Сдвиг по оси состояния.
    if pr_rank is not None and cur_rank is not None and cur_rank != pr_rank:
        status = UPGRADED if cur_rank > pr_rank else DOWNGRADED
        return LifecycleResult(status, reasons, previous_id=prev_id)

    # 6. Та же ось (и направление) — судим по силе относительно порогов.
    rel_up, rel_down = _relevant_strength(previous, current, reasons)
    if rel_up and not rel_down:
        return LifecycleResult(UPGRADED, reasons, previous_id=prev_id)
    if rel_down and not rel_up:
        return LifecycleResult(DOWNGRADED, reasons, previous_id=prev_id)
    return LifecycleResult(CONTINUATION, reasons, previous_id=prev_id)


# ---------------------------------------------------------------------------
# Сопоставимость и объясняющие флаги (чистые помощники)
# ---------------------------------------------------------------------------

def _is_comparable(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Ряды сравнимы только внутри одного symbol/mode/версии стратегии."""
    return all(previous.get(k) == current.get(k) for k in _COMPARABLE_KEYS)


def _change_reasons(previous: dict[str, Any], current: dict[str, Any],
                    thresholds: Thresholds) -> tuple[str, ...]:
    """Набор булевых дельт между рядами (детерминированно отсортирован).

    Флаги независимы от выбранного статуса — они дают детализацию для отчёта."""
    reasons: list[str] = []

    pr_rank = _STATE_RANK.get(previous.get("analysis_status"))
    cur_rank = _STATE_RANK.get(current.get("analysis_status"))
    if pr_rank is not None and cur_rank is not None:
        if cur_rank > pr_rank:
            reasons.append("state_up")
        elif cur_rank < pr_rank:
            reasons.append("state_down")

    pd_, cd = previous.get("candidate_direction"), current.get("candidate_direction")
    pd_dir = pd_ if pd_ in ("long", "short") else None
    cd_dir = cd if cd in ("long", "short") else None
    if pd_dir and cd_dir and pd_dir != cd_dir:
        reasons.append("bias_flip")
    elif pd_dir and cd_dir is None:
        reasons.append("bias_to_neutral")
    elif pd_dir is None and cd_dir:
        reasons.append("bias_from_neutral")

    _delta_flags(reasons, _conf(previous), _conf(current),
                 thresholds.confidence_delta, "confidence_up", "confidence_down")
    _delta_flags(reasons, previous.get("long_score"), current.get("long_score"),
                 thresholds.score_delta, "long_score_up", "long_score_down")
    _delta_flags(reasons, previous.get("short_score"), current.get("short_score"),
                 thresholds.score_delta, "short_score_up", "short_score_down")

    pg, cg = previous.get("blocked_gate"), current.get("blocked_gate")
    if pg is None and cg is not None:
        reasons.append("gate_appeared")
    elif pg is not None and cg is None:
        reasons.append("gate_cleared")
    elif pg is not None and cg is not None and pg != cg:
        reasons.append("gate_changed")

    prev_reasons = _reason_set(previous.get("no_trade_reasons"))
    cur_reasons = _reason_set(current.get("no_trade_reasons"))
    if cur_reasons - prev_reasons:
        reasons.append("reasons_added")
    if prev_reasons - cur_reasons:
        reasons.append("reasons_cleared")

    return tuple(sorted(reasons))


def _relevant_strength(previous: dict[str, Any], current: dict[str, Any],
                       reasons: tuple[str, ...]) -> tuple[bool, bool]:
    """(up, down) по силе, относящейся к направлению сетапа: confidence + score
    той стороны, в которую смотрит сетап (long_score для long и т.д.)."""
    direction = (current.get("candidate_direction")
                 or previous.get("candidate_direction"))
    side = "long_score" if direction == "long" else \
           "short_score" if direction == "short" else None

    up = "confidence_up" in reasons
    down = "confidence_down" in reasons
    if side == "long_score":
        up = up or "long_score_up" in reasons
        down = down or "long_score_down" in reasons
    elif side == "short_score":
        up = up or "short_score_up" in reasons
        down = down or "short_score_down" in reasons
    return up, down


def _delta_flags(reasons: list[str], prev: Any, cur: Any, threshold: float,
                 up_flag: str, down_flag: str) -> None:
    d = _delta(prev, cur)
    if d is None:
        return
    if d >= threshold:
        reasons.append(up_flag)
    elif d <= -threshold:
        reasons.append(down_flag)


def _delta(prev: Any, cur: Any) -> float | None:
    if prev is None or cur is None:
        return None
    return float(cur) - float(prev)


def _conf(forecast: dict[str, Any]) -> Any:
    raw = forecast.get("raw_confidence")
    return raw if raw is not None else forecast.get("confidence")


def _reason_set(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


def _gap_seconds(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    """Разрыв (сек) между якорями рядов; None — если хоть один не распознан."""
    pt, ct = _anchor_ts(previous), _anchor_ts(current)
    if pt is None or ct is None:
        return None
    return (ct - pt).total_seconds()


def _anchor_ts(forecast: dict[str, Any]) -> datetime | None:
    """Якорь времени ряда: signal_candle_close_time -> decision_time ->
    created_at. Строки ISO приводятся к aware-datetime (naive -> UTC)."""
    for key in ("signal_candle_close_time", "decision_time", "created_at"):
        ts = _to_datetime(forecast.get(key))
        if ts is not None:
            return ts
    return None


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
