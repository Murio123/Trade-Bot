"""Part A quality pass: honest, deterministic confidence."""
from __future__ import annotations

from pipeline import conflict_factor


def test_clean_setup_has_no_penalty():
    assert conflict_factor(6, 0) == 1.0
    assert conflict_factor(0, 0) == 1.0  # degenerate: no evidence at all


def test_conflict_penalty_monotonic():
    # More opposing evidence -> lower factor, floor ~0.65.
    factors = [conflict_factor(6, loser) for loser in range(0, 6)]
    assert factors == sorted(factors, reverse=True)
    assert factors[0] == 1.0
    assert 0.6 < factors[-1] < 0.75


def test_near_tie_is_visibly_penalised():
    # 6 vs 5 must not look like 6 vs 0.
    assert conflict_factor(6, 5) < 0.75 < conflict_factor(6, 1)


# --- decorrelation (stage 2) -------------------------------------------------

from signal_engine.confluence import (CATEGORY_CAPS,  # noqa: E402
                                      calculate_confluence_score)


def test_reversal_counted_once_not_stacked():
    # The multi-TF trend-aligned bottom and the single-TF reversal are the
    # same phenomenon: together they must contribute once (+3), not +6.
    base = {"trend_aligned_bottom": True, "bullish_reversal": True,
            "reversal_strong_bull": True}
    total, scores, reasons = calculate_confluence_score(base, "long")
    assert scores["structure"] == 3
    assert sum("дна" in r or "Дно" in r for r in reasons) == 1


def test_category_caps_saturate_structure():
    everything = {
        "price_in_bullish_ob": True, "rejection_wick": True,
        "liquidity_swept_below": True, "reversal_candle": True,
        "price_in_bullish_fvg": True, "in_discount": True,
        "trend_aligned_bottom": True, "bullish_reversal": True,
    }
    _, scores, _ = calculate_confluence_score(everything, "long")
    assert scores["structure"] == CATEGORY_CAPS["structure"]


def test_bb_squeeze_alone_scores_neither_side():
    long_t, _, _ = calculate_confluence_score({"bb_squeeze": True}, "long")
    short_t, _, _ = calculate_confluence_score({"bb_squeeze": True}, "short")
    assert long_t == 0 and short_t == 0


# --- data-quality gates (stage 3) ---------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

from signal_engine.vetoes import abnormal_volatility, stale_data  # noqa: E402


def test_stale_data_gate():
    now = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
    fresh = now - timedelta(hours=1)
    old = now - timedelta(hours=5)
    assert not stale_data(fresh, "1h", now=now)   # 1 bar old — fine
    assert stale_data(old, "1h", now=now)         # 5 bars old — stale
    assert not stale_data(old, "4h", now=now)     # same age fine on 4H
    assert stale_data(None, "1h", now=now)        # no candle at all — stale


def test_abnormal_volatility_gate():
    assert abnormal_volatility({"atr_percentile": 97.0})
    assert not abnormal_volatility({"atr_percentile": 80.0})
    # Missing data is handled by graceful degradation, not this gate.
    assert not abnormal_volatility(None)
    assert not abnormal_volatility({"atr_percentile": None})


def test_funding_requires_stretched_zscore():
    # Ordinary base funding (positive, z≈0) must not hand points to shorts.
    ordinary = {"funding": 0.0001, "funding_z": 0.2}
    total, scores, _ = calculate_confluence_score(ordinary, "short")
    assert scores["macro"] == 0
    stretched = {"funding": 0.0008, "funding_z": 2.1}
    _, scores2, _ = calculate_confluence_score(stretched, "short")
    assert scores2["macro"] == 2
