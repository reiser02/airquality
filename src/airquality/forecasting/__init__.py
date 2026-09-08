"""Forecasting benchmark: anomaly-cleaning arms + shared multi-step backtest.

This subpackage consumes the anomaly and imputation subsystems (kept as pure
benchmarks) to build, per series, one training arm per (detection strategy,
imputation) combination — plus the ``raw`` baseline — and compare their
multi-step forecasting error over a shared observed test. A separate paired
synthetic-context experiment measures preprocessing recovery for frozen
foundation models.
"""

from importlib import import_module
from typing import Any

from airquality.forecasting.backtest import (
    backtest_forecast,
    get_forecast_model_requirements,
    get_strict_forecast_requirements,
    select_holdout_window,
)
from airquality.forecasting.cleaning import remove_anomalies
from airquality.forecasting.detection import (
    ConsensusDetection,
    DetectionResult,
    DetectionStrategy,
    InjectionSoftDetection,
    InjectionTopKDetection,
    MaskTransform,
    SeriesDetectionContext,
    apply_mask_transforms,
    build_detection_strategy,
    common_detection_support,
)
from airquality.forecasting.fill import (
    GapImputationOutcome,
    GapImputationPolicy,
    GapImputationResult,
    GapImputationRule,
    build_imputer,
    impute_series,
    impute_series_by_gap,
    impute_series_by_gap_result,
    nan_gap_windows,
    parse_imputation_gap_rules,
)

_PIPELINE_EXPORTS = {
    "ForecastArm",
    "build_arms",
    "run_benchmark_from_config",
}


def __getattr__(name: str) -> Any:
    """Load pipeline exports lazily so ``python -m ...pipeline`` runs once."""
    if name in _PIPELINE_EXPORTS:
        pipeline = import_module("airquality.forecasting.pipeline")
        value = getattr(pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "run_benchmark_from_config",
    "ForecastArm",
    "build_arms",
    "remove_anomalies",
    "DetectionResult",
    "DetectionStrategy",
    "ConsensusDetection",
    "InjectionSoftDetection",
    "InjectionTopKDetection",
    "SeriesDetectionContext",
    "MaskTransform",
    "apply_mask_transforms",
    "build_detection_strategy",
    "common_detection_support",
    "build_imputer",
    "GapImputationOutcome",
    "GapImputationPolicy",
    "GapImputationResult",
    "GapImputationRule",
    "impute_series",
    "impute_series_by_gap",
    "impute_series_by_gap_result",
    "nan_gap_windows",
    "parse_imputation_gap_rules",
    "backtest_forecast",
    "get_forecast_model_requirements",
    "get_strict_forecast_requirements",
    "select_holdout_window",
]
