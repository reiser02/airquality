"""Remove detected anomalies from a *real* (unlabeled) air-quality series.

The forecasting pipeline detects anomalies through
:mod:`airquality.forecasting.detection`, then uses :func:`remove_anomalies` to
set flagged timestamps to NaN before optional imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airquality.forecasting.detection import DetectionResult


def remove_anomalies(series: pd.Series, result: DetectionResult) -> pd.Series:
    """Return a copy of ``series`` with flagged timestamps set to NaN."""
    cleaned = series.copy()
    flagged_index = result.mask.index[result.mask.to_numpy()]
    cleaned.loc[cleaned.index.intersection(flagged_index)] = np.nan
    return cleaned


__all__ = [
    "DetectionResult",
    "remove_anomalies",
]
