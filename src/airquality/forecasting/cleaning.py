"""Detect and remove anomalies from a *real* (unlabeled) air-quality series.

Production-facing entrypoint over the ``unlabeled`` detection strategy
(:class:`airquality.forecasting.detection.ConsensusDetection`): in real time
there is no ground truth, so no detector can be selected or calibrated against
labels. The strategy scores every requested detector per **contiguous observed
segment** (no ``dropna()`` gluing), discards detectors over the
detection-rate budget (sensor faults are rare) and re-thresholds the consensus
of the survivors with the MAD rule — see :mod:`airquality.forecasting.detection`
for the full method and for the injection-based alternatives used by the
forecasting benchmark.

Flagged timestamps are then set to NaN by :func:`remove_anomalies` so the
imputation step can fill them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airquality.anomaly.metrics import DEFAULT_MAX_DETECTION_RATE, DEFAULT_THRESHOLD_K
from airquality.forecasting.detection import (
    DEFAULT_SEED,
    ConsensusDetection,
    DetectionResult,
    SeriesDetectionContext,
)


def detect_anomaly_mask(
    series: pd.Series,
    *,
    detectors: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    freq: str = "h",
    threshold_k: float = DEFAULT_THRESHOLD_K,
    max_detection_rate: float = DEFAULT_MAX_DETECTION_RATE,
) -> DetectionResult:
    """Flag anomalies with the consensus of the detectors that pass the rate filter."""
    context = SeriesDetectionContext(
        series, detectors=detectors, seed=seed, device=device, freq=freq
    )
    strategy = ConsensusDetection(
        threshold_k=threshold_k, max_detection_rate=max_detection_rate
    )
    return strategy.detect(context)


def remove_anomalies(series: pd.Series, result: DetectionResult) -> pd.Series:
    """Return a copy of ``series`` with flagged timestamps set to NaN."""
    cleaned = series.copy()
    flagged_index = result.mask.index[result.mask.to_numpy()]
    cleaned.loc[cleaned.index.intersection(flagged_index)] = np.nan
    return cleaned


__all__ = [
    "DetectionResult",
    "detect_anomaly_mask",
    "remove_anomalies",
]
