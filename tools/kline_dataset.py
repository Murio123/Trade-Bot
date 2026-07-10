"""Stage C1.3a: строгий offline-загрузчик датасетов, записанных tools.kline_cache.

Read-only. НЕ торгует, НЕ ходит в сеть, НЕ пишет в БД, НЕ меняет схему, scoring,
thresholds, pipeline, scheduler или backtest. НЕ импортируется runtime-кодом.

Загрузчик fail-closed: манифест читается ПЕРВЫМ, файл данных — только по ссылке
из манифеста, и каждое утверждение манифеста перепроверяется на самих данных.
Датасет, который не совпадает со своим манифестом, не загружается вовсе.

Зачем так строго (drift points C1.1):

  * D15 — CVD ветвится ровно по наличию колонки taker_buy_base
    (analyzer.cvd.compute_cvd_from_klines): exact против estimate. Биржа
    пинуется на этапе записи, но CSV-раунд-трип может ТИХО потерять колонку или
    превратить её в сплошные NaN — и метод CVD сменится молча, вернув ровно ту
    ошибку, ради которой C1.2 существовал. Поэтому CVD-инвариант проверяется на
    загруженном фрейме, а не берётся на веру из манифеста. Набор фреймов для
    одного прогона обязан иметь ОДНУ биржу и ОДИН символ.

  * D14 — вспомогательные htf/zone фреймы исторически брались одним запросом
    limit=500, из-за чего backtest молча пропускал бары, а знаменатель отчёта
    этого не показывал. aux_depth() считает нужную глубину ЗАРАНЕЕ и возвращает
    таблицу required/available/deficit. Решение «падать или нет» принимает
    вызывающий (deep_backtest), а не загрузчик.

Пропуски (gaps) не игнорируются: метаданные всегда возвращаются наружу, а
превышение max_gap_ratio поднимает GapRatioExceeded. Индикатор, посчитанный
через дыру в истории, — это не тот индикатор (EMA200 поперёк двухдневного
простоя биржи не является EMA200), поэтому порог здесь — риск-контроль.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from tools.kline_cache import INTERVAL_MS, dataset_stem, gap_report

# Часы на свечу — выводятся из общей таблицы интервалов, чтобы у загрузчика и
# писателя не было двух независимых представлений о длине таймфрейма.
TF_HOURS: dict[str, float] = {
    tf: ms / 3_600_000 for tf, ms in INTERVAL_MS.items()
}

# --- Warmup-константы -------------------------------------------------------
# Взяты из backtest._walk и НЕ являются свободными параметрами. Менять их можно
# только вслед за backtest.py (который в этой стадии не трогаем).
#
#   ENTRY_WARMUP — backtest._walk берёт entry_sub = entry_df.iloc[i-299 : i+1],
#                  то есть 300-баровое окно (live-паритет с klines limit=300).
#   HTF_WARMUP   — guard `if len(htf_slice) < 210: continue`.
#   ZONE_WARMUP  — guard `if len(s) >= 30` для каждого zone-ТФ.
#   REGIME_1D_WARMUP — связывающее ограничение здесь EMA200 из
#                  compute_indicators: без 200 баров 1D режим не считается.
#                  analyze_volatility требует лишь len(df) >= 30 и берёт
#                  percentile_rank по всей серии, то есть жёсткого окна у неё
#                  нет (проверено в analyzer/volatility.py). 210 = 200 (ema200)
#                  + 10 запаса, симметрично HTF_WARMUP.
ENTRY_WARMUP = 300
HTF_WARMUP = 210
ZONE_WARMUP = 30
REGIME_1D_WARMUP = 210

REGIME_TF = "1d"

# Колонки, которые обязаны стать tz-aware UTC datetime после загрузки.
TIME_COLUMNS = ("open_time", "close_time")

# Минимум баров, при котором проверки вообще что-то значат: на 0 и 1 баре
# сверка границ и отчёт о пропусках вырождаются в тавтологию (нечего сравнивать
# и не с чем), и датасет «проходит» валидацию, ничего не доказав.
MIN_BARS = 2


class DatasetError(Exception):
    """Датасет не соответствует своему манифесту или запросу."""


class GapRatioExceeded(DatasetError):
    """Доля пропущенных свечей выше допустимой."""


@dataclass(frozen=True)
class LoadedFrame:
    """Провалидированный фрейм плюс его происхождение.

    ``gaps`` — ПЕРЕСЧИТАННЫЕ по данным значения, а не копия полей манифеста.
    Манифест здесь свидетель, а не источник истины.
    """
    exchange: str
    symbol: str
    timeframe: str
    df: pd.DataFrame
    manifest: dict[str, Any]
    data_path: str
    manifest_path: str
    gaps: dict[str, Any]

    @property
    def bars(self) -> int:
        return len(self.df)

    @property
    def has_taker_buy_base(self) -> bool:
        return bool(self.manifest["has_taker_buy_base"])

    @property
    def cvd_method(self) -> str:
        """Как analyzer.cvd будет считать дельту на этом фрейме (D15)."""
        return "exact" if self.has_taker_buy_base else "estimate"


def gap_ratio(missing_bars: int, bars: int) -> float:
    return (missing_bars / bars) if bars > 0 else 0.0


def _recompute_gaps(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Отчёт о пропусках, посчитанный по САМИМ данным.

    Используется ровно та же функция, которой пользовался писатель
    (kline_cache.gap_report), поэтому расхождение с манифестом означает дрейф
    данных или подделанный манифест, а не две несогласованные реализации.
    """
    report = gap_report(df, timeframe)
    missing = int(report["missing_bars"])
    return {
        "has_gaps": bool(report["has_gaps"]),
        "gap_count": int(report["gap_count"]),
        "missing_bars": missing,
        "gap_ratio": gap_ratio(missing, len(df)),
        "gap_examples": report["gap_examples"],
        "expected_interval_ms": report["expected_interval_ms"],
    }


# ---------------------------------------------------------------------------
# Пути и чтение
# ---------------------------------------------------------------------------

def manifest_path_for(outdir: str, exchange: str, symbol: str,
                      timeframe: str) -> str:
    """Путь к манифесту. Путь к данным берётся ИЗ манифеста, а не угадывается."""
    stem = dataset_stem(exchange, symbol, timeframe)
    return os.path.join(outdir, f"{stem}.manifest.json")


def _read_manifest(manifest_path: str) -> dict[str, Any]:
    if not os.path.exists(manifest_path):
        raise DatasetError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"manifest is not valid JSON: {manifest_path}") from exc


def _read_data(data_path: str, file_format: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise DatasetError(f"data file not found: {data_path}")
    if file_format == "parquet":
        return pd.read_parquet(data_path)
    if file_format == "csv":
        # Времена парсим явно и приводим к UTC: backtest сравнивает
        # close_time <= t, а строковая колонка сравнивается неверно, не падая.
        return pd.read_csv(data_path)
    raise DatasetError(f"unsupported file_format: {file_format!r}")


def _coerce_times(df: pd.DataFrame, data_path: str) -> pd.DataFrame:
    """Привести временные колонки к tz-aware UTC.

    Невалидные значения НЕ коэрсятся в NaT: чужой или правленый вручную CSV
    (например, со смешанной точностью timestamp) обязан упасть с внятной
    ошибкой, а не превратиться в дыры, которые потом кто-то будет отлаживать.
    """
    for col in TIME_COLUMNS:
        if col not in df.columns:
            continue
        try:
            df[col] = pd.to_datetime(df[col], utc=True)
        except (ValueError, TypeError, OverflowError) as exc:
            raise DatasetError(
                f"{data_path}: column {col!r} is not parseable as UTC datetime: "
                f"{exc}") from exc
    return df


def _to_ms(value: Any) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Валидация одного фрейма
# ---------------------------------------------------------------------------

def _validate_manifest(m: Mapping[str, Any], *, exchange: str, symbol: str,
                       timeframe: str, manifest_path: str) -> None:
    if not m.get("source_pinned"):
        raise DatasetError(
            f"{manifest_path}: source_pinned is not true — датасет мог быть "
            "собран со сменой источника (D15); отказ загружать")

    for key, want in (("exchange", exchange), ("symbol", symbol),
                      ("timeframe", timeframe)):
        got = m.get(key)
        if got != want:
            raise DatasetError(
                f"{manifest_path}: manifest {key}={got!r} != requested {want!r}")

    expected = INTERVAL_MS.get(timeframe)
    if m.get("expected_interval_ms") != expected:
        raise DatasetError(
            f"{manifest_path}: expected_interval_ms="
            f"{m.get('expected_interval_ms')!r} != interval table {expected!r}")


def _validate_frame(df: pd.DataFrame, m: Mapping[str, Any],
                    *, data_path: str) -> None:
    if int(m.get("actual_bars", -1)) != len(df):
        raise DatasetError(
            f"{data_path}: manifest actual_bars={m.get('actual_bars')} != "
            f"len(df)={len(df)}")

    # Пустой (и однобаровый) датасет раньше проходил валидацию ВАКУУМНО:
    # сверка границ спрятана за `if len(df)`, а gap_report на len < 2 возвращает
    # нули. Bybit-образный пустой фрейм вдобавок сходился с CVD-инвариантом
    # (False == False) и грузился молча. Отказ здесь, без allow_empty: ниже по
    # стеку нулевой датасет всё равно означает прогон по нулю баров.
    if len(df) < MIN_BARS:
        raise DatasetError(
            f"{data_path}: dataset has too few bars ({len(df)} < {MIN_BARS}) — "
            "проверки границ и интервалов на таком фрейме ничего не доказывают")

    if "open_time" not in df.columns:
        raise DatasetError(f"{data_path}: missing open_time column")

    for col in TIME_COLUMNS:
        if col in df.columns and not isinstance(df[col].dtype,
                                                pd.DatetimeTZDtype):
            raise DatasetError(f"{data_path}: {col} is not tz-aware datetime")

    # Порядок важен для диагностики: дубликат или перемешанный порядок сначала
    # проявились бы как «endpoint mismatch» и увели бы разбор не туда. Сперва
    # говорим, что не так со СТРУКТУРОЙ open_time, и только потом сверяем
    # границы с манифестом.
    if not df["open_time"].is_unique:
        raise DatasetError(f"{data_path}: open_time contains duplicates")
    if not df["open_time"].is_monotonic_increasing:
        raise DatasetError(f"{data_path}: open_time is not strictly increasing")

    if len(df):
        first, last = _to_ms(df["open_time"].iloc[0]), _to_ms(df["open_time"].iloc[-1])
        if m.get("first_open_time") != first:
            raise DatasetError(
                f"{data_path}: manifest first_open_time={m.get('first_open_time')} "
                f"!= frame {first}")
        if m.get("last_open_time") != last:
            raise DatasetError(
                f"{data_path}: manifest last_open_time={m.get('last_open_time')} "
                f"!= frame {last}")

    # D15-инвариант: то, во что верит манифест, обязано быть правдой о данных.
    # Именно здесь ловится потеря taker_buy_base на CSV-раунд-трипе.
    data_has_taker = bool("taker_buy_base" in df.columns
                          and df["taker_buy_base"].notna().any())
    if data_has_taker != bool(m.get("has_taker_buy_base")):
        raise DatasetError(
            f"{data_path}: has_taker_buy_base={m.get('has_taker_buy_base')} in "
            f"manifest but data says {data_has_taker} — CVD метод сменился бы "
            "молча (D15)")


def _validate_grid_alignment(df: pd.DataFrame, expected_interval_ms: int,
                             *, data_path: str) -> None:
    """Каждый шаг open_time обязан быть ЦЕЛЫМ кратным шага таймфрейма.

    gap_report считает пропуски как ``delta // expected - 1``, то есть дрейф
    меньше одного интервала обнуляется делением. Свеча 15m, открытая в 7.5
    минут от сетки, давала missing_bars=0 и проезжала даже max_gap_ratio=0.0.
    Смещение по сетке — это не пропуск, а испорченные данные: индикаторы на
    неравномерных барах неверны, и никакой счётчик пропусков этого не покажет.

    Честная дыра (delta = 2x, 3x ... интервала) остаётся допустимой и
    контролируется max_gap_ratio, а не этой проверкой.
    """
    if expected_interval_ms <= 0:
        raise DatasetError(f"{data_path}: expected_interval_ms must be positive")

    times_ms = df["open_time"].astype("int64").to_numpy() // 1_000_000
    deltas = times_ms[1:] - times_ms[:-1]

    # delta <= 0 недостижимо после проверок уникальности и монотонности;
    # оставлено как defense-in-depth на случай их перестановки.
    nonpositive = (deltas <= 0).nonzero()[0]
    if len(nonpositive):
        i = int(nonpositive[0])
        raise DatasetError(
            f"{data_path}: non-positive open_time delta {int(deltas[i])}ms at "
            f"row {i + 1}")

    misaligned = (deltas % expected_interval_ms != 0).nonzero()[0]
    if len(misaligned):
        i = int(misaligned[0])
        raise DatasetError(
            f"{data_path}: off-grid open_time at row {i + 1} "
            f"({df['open_time'].iloc[i + 1]}): delta {int(deltas[i])}ms is not a "
            f"multiple of expected_interval_ms={expected_interval_ms} "
            f"({len(misaligned)} misaligned bar(s) total) — это смещение сетки, "
            "а не пропуск")


def _validate_gaps(gaps: Mapping[str, Any], m: Mapping[str, Any],
                   *, data_path: str) -> None:
    """Манифест обязан совпасть с пересчитанным по данным отчётом о пропусках.

    Раньше эти поля принимались на веру — единственное утверждение манифеста,
    которое не перепроверялось. Устаревший, правленый или собранный чужим
    инструментом манифест мог объявить датасет бездырочным, и max_gap_ratio
    пропускал бы его: EMA200 поперёк дыры перестаёт быть EMA200 молча.

    gap_examples намеренно НЕ сравниваются: это усечённая до 5 штук иллюстрация,
    а не инвариант. Расхождение по count/missing/has_gaps уже фатально.
    """
    for key in ("gap_count", "missing_bars", "has_gaps"):
        want = gaps[key]
        got = m.get(key)
        if got is None:
            raise DatasetError(f"{data_path}: manifest has no {key}")
        if type(want) is bool:
            got = bool(got)
        else:
            got = int(got)
        if got != want:
            raise DatasetError(
                f"{data_path}: manifest {key}={got!r} != recomputed {want!r} "
                "— манифест не описывает эти данные")


def load_frame(outdir: str, exchange: str, symbol: str, timeframe: str,
               max_gap_ratio: float | None = None) -> LoadedFrame:
    """Загрузить и провалидировать один таймфрейм. Манифест читается первым."""
    manifest_path = manifest_path_for(outdir, exchange, symbol, timeframe)
    manifest = _read_manifest(manifest_path)
    _validate_manifest(manifest, exchange=exchange, symbol=symbol,
                       timeframe=timeframe, manifest_path=manifest_path)

    data_file = manifest.get("data_file")
    if not data_file:
        raise DatasetError(f"{manifest_path}: manifest has no data_file")
    data_path = os.path.join(outdir, data_file)

    df = _coerce_times(_read_data(data_path, manifest.get("file_format", "")),
                       data_path)
    _validate_frame(df, manifest, data_path=data_path)

    # Сетка проверяется ДО пересчёта пропусков: off-grid бар не должен даже
    # рассматриваться как «пропуск», который потом простит max_gap_ratio.
    _validate_grid_alignment(df, INTERVAL_MS[timeframe], data_path=data_path)

    # Пропуски считаются ПО ДАННЫМ, затем манифест сверяется с ними. Порог
    # применяется к пересчитанному значению — иначе враньё в манифесте
    # отключало бы риск-контроль.
    gaps = _recompute_gaps(df, timeframe)
    _validate_gaps(gaps, manifest, data_path=data_path)

    if max_gap_ratio is not None and gaps["gap_ratio"] > max_gap_ratio:
        raise GapRatioExceeded(
            f"{data_path}: gap ratio {gaps['gap_ratio']:.6f} > max_gap_ratio "
            f"{max_gap_ratio:.6f} ({gaps['missing_bars']} missing bars over "
            f"{len(df)}); индикаторы поперёк дыр недостоверны")

    return LoadedFrame(exchange=exchange, symbol=symbol, timeframe=timeframe,
                       df=df, manifest=dict(manifest), data_path=data_path,
                       manifest_path=manifest_path, gaps=gaps)


# ---------------------------------------------------------------------------
# Набор фреймов на один прогон: одна биржа, один символ (D15)
# ---------------------------------------------------------------------------

def required_timeframes(profile: Mapping[str, Any]) -> list[str]:
    """entry + htf + zone_tfs + 1d (режим/волатильность), без повторов."""
    ordered: list[str] = [profile["entry"], profile["htf"]]
    ordered.extend(profile.get("zone_tfs", []))
    ordered.append(REGIME_TF)
    seen: list[str] = []
    for tf in ordered:
        if tf not in seen:
            seen.append(tf)
    return seen


def assert_single_source(frames: Mapping[str, LoadedFrame]) -> None:
    """Один прогон — одна биржа и один символ.

    Смешанный по бирже набор — это ровно D15: часть фреймов даёт exact-CVD,
    часть estimate, и ниже по стеку это уже не чинится.

    При загрузке через load_frames() с явной биржей смешение недостижимо по
    построению (каждый манифест сверяется с запрошенным значением). Проверка
    оставлена как defense-in-depth: она ловит набор, собранный в обход
    load_frames, и её дешевле держать, чем доказывать недостижимость заново
    после каждого рефакторинга.
    """
    exchanges = {f.manifest["exchange"] for f in frames.values()}
    symbols = {f.manifest["symbol"] for f in frames.values()}
    if len(exchanges) > 1:
        raise DatasetError(
            f"mixed exchanges in one dataset: {sorted(exchanges)} — CVD "
            "семантика различается между источниками (D15)")
    if len(symbols) > 1:
        raise DatasetError(f"mixed symbols in one dataset: {sorted(symbols)}")


def load_frames(outdir: str, exchange: str, symbol: str,
                timeframes: Iterable[str],
                max_gap_ratio: float | None = None) -> dict[str, LoadedFrame]:
    """Загрузить набор ТФ и запретить смешение источников (D15)."""
    frames: dict[str, LoadedFrame] = {}
    for tf in timeframes:
        frames[tf] = load_frame(outdir, exchange, symbol, tf,
                                max_gap_ratio=max_gap_ratio)
    assert_single_source(frames)
    return frames


# ---------------------------------------------------------------------------
# Aux-depth (D14)
# ---------------------------------------------------------------------------

def _warmup_for_roles(roles: set[str]) -> int:
    per_role = {"entry": ENTRY_WARMUP, "htf": HTF_WARMUP,
                "zone": ZONE_WARMUP, "regime_1d": REGIME_1D_WARMUP}
    return max(per_role[r] for r in roles)


def _roles(profile: Mapping[str, Any]) -> dict[str, set[str]]:
    """Один ТФ может играть несколько ролей (у swing htf == 1d == regime)."""
    roles: dict[str, set[str]] = {}
    roles.setdefault(profile["entry"], set()).add("entry")
    roles.setdefault(profile["htf"], set()).add("htf")
    for tf in profile.get("zone_tfs", []):
        roles.setdefault(tf, set()).add("zone")
    roles.setdefault(REGIME_TF, set()).add("regime_1d")
    return roles


def _available_bars(value: Any) -> int:
    if isinstance(value, LoadedFrame):
        return value.bars
    if isinstance(value, pd.DataFrame):
        return len(value)
    if isinstance(value, Mapping):  # манифест
        return int(value.get("actual_bars", 0) or 0)
    return int(value)


def aux_depth(profile: Mapping[str, Any], walk_bars: int,
              frames: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Сколько баров нужно каждому ТФ, чтобы пройти walk_bars entry-баров.

        required(h_aux) = ceil(walk_bars * h_entry / h_aux) + warmup(roles)

    Для entry-ТФ календарный множитель равен 1, поэтому формула вырождается в
    walk_bars + ENTRY_WARMUP — то самое 300-баровое окно live-паритета.

    Возвращает таблицу, а НЕ падает: решение о том, фатален ли дефицит,
    принимает вызывающий (deep_backtest). ``frames`` принимает LoadedFrame,
    DataFrame, манифест или просто число доступных баров.
    """
    if walk_bars <= 0:
        raise ValueError("walk_bars must be positive")

    entry_tf = profile["entry"]
    h_entry = TF_HOURS[entry_tf]
    span_hours = walk_bars * h_entry

    table: dict[str, dict[str, Any]] = {}
    for tf, roles in _roles(profile).items():
        h_aux = TF_HOURS[tf]
        warmup = _warmup_for_roles(roles)
        required = math.ceil(span_hours / h_aux) + warmup
        available = _available_bars(frames[tf]) if tf in frames else 0
        deficit = max(0, required - available)
        table[tf] = {
            "roles": sorted(roles),
            "warmup": warmup,
            "required": required,
            "available": available,
            "deficit": deficit,
            "ok": deficit == 0,
        }
    return table


def aux_depth_deficits(table: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Только дефицитные ТФ — удобно для сообщения об ошибке у вызывающего."""
    return {tf: row["deficit"] for tf, row in table.items() if row["deficit"] > 0}
