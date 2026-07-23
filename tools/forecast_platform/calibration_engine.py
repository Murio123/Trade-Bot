"""C4.2 §5: Calibration Engine — Platt/isotonic/reliability/Brier/ECE.

Independent of any specific model family; exists for the future Direction
model (C4.0 §9 / C4.1a), the first family in this project that will
actually need probability calibration (the first, volatility, model is a
point-forecast regression and uses tools.forecast_platform.
evaluation_engine.bucket_calibration instead, per reports/c41/
volatility_model_frozen_spec.md §10).

Uses only numpy — no new dependency (this project has no sklearn/scipy
installed). Platt scaling is two-parameter logistic regression fit via
Newton-Raphson; isotonic regression is the classic Pool-Adjacent-Violators
Algorithm (PAVA) — both are small, well-understood, closed-form-adjacent
methods that don't need a heavier library at this problem size.
"""
from __future__ import annotations

import numpy as np


def brier_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


def reliability_curve(y_true, p, n_bins: int = 10) -> list[dict[str, object]]:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, object]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            out.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                       "mean_predicted": None, "mean_actual": None})
            continue
        out.append({"bin_lo": float(lo), "bin_hi": float(hi),
                   "n": int(mask.sum()),
                   "mean_predicted": float(p[mask].mean()),
                   "mean_actual": float(y_true[mask].mean())})
    return out


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    curve = reliability_curve(y_true, p, n_bins)
    total = sum(b["n"] for b in curve)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in curve:
        if b["n"] == 0:
            continue
        ece += (b["n"] / total) * abs(b["mean_actual"] - b["mean_predicted"])
    return float(ece)


def fit_platt(scores, y_true, *, max_iter: int = 100, tol: float = 1e-6
             ) -> tuple[float, float]:
    """Fit p = sigmoid(a*score + b) via Newton-Raphson on the (a, b)
    log-likelihood. Returns (a, b). Two parameters, closed-form Hessian."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        grad_a = np.sum((p - y) * x)
        grad_b = np.sum(p - y)
        w = p * (1 - p)
        h_aa = np.sum(w * x * x)
        h_ab = np.sum(w * x)
        h_bb = np.sum(w)
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * grad_a - h_ab * grad_b) / det
        db = (h_aa * grad_b - h_ab * grad_a) / det
        a -= da
        b -= db
        if abs(da) < tol and abs(db) < tol:
            break
    return float(a), float(b)


def apply_platt(scores, a: float, b: float) -> np.ndarray:
    z = a * np.asarray(scores, dtype=float) + b
    return 1.0 / (1.0 + np.exp(-z))


def fit_isotonic(scores, y_true) -> tuple[np.ndarray, np.ndarray]:
    """Pool-Adjacent-Violators isotonic regression, sorted by `scores`.
    Returns (sorted_scores, fitted_values); use apply_isotonic with the
    same pair at prediction time."""
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    x = np.asarray(scores, dtype=float)[order]
    y = list(np.asarray(y_true, dtype=float)[order])
    weights = [1] * len(y)
    i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1]:
            merged_w = weights[i] + weights[i + 1]
            merged_v = (y[i] * weights[i] + y[i + 1] * weights[i + 1]) / merged_w
            y[i:i + 2] = [merged_v]
            weights[i:i + 2] = [merged_w]
            i = max(i - 1, 0)
        else:
            i += 1
    expanded: list[float] = []
    for val, w in zip(y, weights):
        expanded.extend([val] * w)
    return x, np.asarray(expanded, dtype=float)


def apply_isotonic(scores, fitted_x: np.ndarray, fitted_y: np.ndarray
                   ) -> np.ndarray:
    return np.interp(np.asarray(scores, dtype=float), fitted_x, fitted_y)
