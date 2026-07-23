"""C4.2: Forecast Platform — generic offline infrastructure for ML forecasting
models (volatility first; direction/magnitude/regime later, C4.0/C4.1).

Read-only, offline-only. NOT imported by runtime, signal_engine, execution,
Telegram, or deep_backtest.py. No network, no DB, no live trading. Every
module here only reuses existing, already-audited pure functions from
analyzer/*, signal_engine/*, pipeline.py, and tools/deep_backtest.py/
tools/deep_discovery.py — it never modifies them and never duplicates their
logic.

Layers (see reports/c42/architecture.md for the full design):
  feature_store      — central feature registry (name/dtype/tf/source/version).
  dataset_builder     — point-in-time dataset construction + manifest.
  model_interface     — the one contract every forecasting model implements.
  training_engine     — reusable walk-forward training loop, no model-specific code.
  calibration_engine  — Platt/isotonic/Brier/ECE for probabilistic models.
  evaluation_engine   — MAE/RMSE/Spearman/calibration-bucket/baseline comparisons.
  model_registry      — immutable trained-model metadata store.
  contracts           — ForecastRecord (the one output every model emits) +
                        ModelMetadata + spec_hash.
"""
