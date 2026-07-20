"""Forecasting benchmark: anomaly-cleaning arms + shared multi-step backtest.

This subpackage consumes the anomaly and imputation subsystems (kept as pure
benchmarks) to build, per series, one training arm per (detection strategy,
imputation) combination — plus the ``raw`` baseline — and compare their
multi-step forecasting error over a shared observed holdout.
"""

from airquality.forecasting.backtest import (
    backtest_forecast,
    get_forecast_model_requirements,
    select_holdout_window,
)
from airquality.forecasting.cleaning import detect_anomaly_mask, remove_anomalies
from airquality.forecasting.detection import (
    ConsensusDetection,
    DetectionResult,
    DetectionStrategy,
    InjectionTopKDetection,
    MaskTransform,
    SeriesDetectionContext,
    apply_mask_transforms,
    build_detection_strategy,
)
from airquality.forecasting.fill import build_imputer, impute_series, nan_gap_windows
from airquality.forecasting.pipeline import (
    ForecastArm,
    ForecastRegime,
    build_arms,
    run_benchmark_from_config,
)

__all__ = [
    "run_benchmark_from_config",
    "ForecastArm",
    "ForecastRegime",
    "build_arms",
    "detect_anomaly_mask",
    "remove_anomalies",
    "DetectionResult",
    "DetectionStrategy",
    "ConsensusDetection",
    "InjectionTopKDetection",
    "SeriesDetectionContext",
    "MaskTransform",
    "apply_mask_transforms",
    "build_detection_strategy",
    "build_imputer",
    "impute_series",
    "nan_gap_windows",
    "backtest_forecast",
    "get_forecast_model_requirements",
    "select_holdout_window",
]
