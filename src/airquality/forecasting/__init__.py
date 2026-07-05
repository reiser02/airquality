"""Forecasting pipeline: anomaly cleaning + imputation + raw-vs-preprocessed backtest.

This subpackage consumes the anomaly and imputation subsystems (kept as pure
benchmarks) to build, per series, a *raw* and a *preprocessed* arm and compare
their multi-step forecasting error over a shared observed holdout.
"""

from airquality.forecasting.backtest import backtest_forecast, select_holdout_window
from airquality.forecasting.cleaning import (
    CleaningResult,
    detect_anomaly_mask,
    remove_anomalies,
)
from airquality.forecasting.fill import build_imputer, impute_series, nan_gap_windows
from airquality.forecasting.pipeline import run_comparison_from_config

__all__ = [
    "run_comparison_from_config",
    "detect_anomaly_mask",
    "remove_anomalies",
    "CleaningResult",
    "build_imputer",
    "impute_series",
    "nan_gap_windows",
    "backtest_forecast",
    "select_holdout_window",
]
