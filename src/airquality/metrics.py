"""Shared benchmark metrics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mase as darts_mase
from darts.metrics import rmsse as darts_rmsse


def metric_higher_is_better(metric: str) -> bool:
    """Return whether larger values indicate better benchmark performance."""
    return str(metric).upper() == "R2"


def compute_mase(
    actual: pd.Series | TimeSeries,
    pred: pd.Series | TimeSeries,
    insample: pd.Series | TimeSeries,
    *,
    seasonality_m: int,
) -> float:
    """Compute MASE with Darts after finite-value forecast alignment."""
    actual_s = actual.to_series() if isinstance(actual, TimeSeries) else actual
    pred_s = pred.to_series() if isinstance(pred, TimeSeries) else pred
    insample_s = insample.to_series() if isinstance(insample, TimeSeries) else insample

    aligned_idx = actual_s.index.intersection(pred_s.index)
    actual_s = actual_s.reindex(aligned_idx).astype(float)
    pred_s = pred_s.reindex(aligned_idx).astype(float)
    valid = np.isfinite(actual_s.to_numpy()) & np.isfinite(pred_s.to_numpy())
    actual_values = actual_s.to_numpy()[valid]
    pred_values = pred_s.to_numpy()[valid]
    if len(actual_values) == 0:
        return float("nan")

    insample_s = insample_s.astype(float)
    insample_values = insample_s.to_numpy()
    if len(insample_values) <= seasonality_m or np.isinf(insample_values).any():
        return float("nan")

    # Darts requires insample to end one step before prediction. Synthetic
    # integer indices preserve the values while satisfying that metric contract.
    split = len(insample_values)
    insample_ts = TimeSeries.from_times_and_values(
        pd.RangeIndex(split), insample_values
    )
    eval_index = pd.RangeIndex(split, split + len(actual_values))
    actual_ts = TimeSeries.from_times_and_values(eval_index, actual_values)
    pred_ts = TimeSeries.from_times_and_values(eval_index, pred_values)
    try:
        value = float(
            darts_mase(
                actual_series=actual_ts,
                pred_series=pred_ts,
                insample=insample_ts,
                m=int(seasonality_m),
            )
        )
        return value if np.isfinite(value) else float("nan")
    except Exception as exc:
        logging.warning("MASE no computable: %s", exc)
        return float("nan")


def compute_rmsse(
    actual: pd.Series | TimeSeries,
    pred: pd.Series | TimeSeries,
    insample: pd.Series | TimeSeries,
    *,
    seasonality_m: int,
) -> float:
    """Compute Darts RMSSE with the same alignment policy as :func:`compute_mase`."""
    actual_s = actual.to_series() if isinstance(actual, TimeSeries) else actual
    pred_s = pred.to_series() if isinstance(pred, TimeSeries) else pred
    insample_s = insample.to_series() if isinstance(insample, TimeSeries) else insample

    aligned_idx = actual_s.index.intersection(pred_s.index)
    actual_s = actual_s.reindex(aligned_idx).astype(float)
    pred_s = pred_s.reindex(aligned_idx).astype(float)
    valid = np.isfinite(actual_s.to_numpy()) & np.isfinite(pred_s.to_numpy())
    actual_values = actual_s.to_numpy()[valid]
    pred_values = pred_s.to_numpy()[valid]
    if len(actual_values) == 0:
        return float("nan")

    insample_s = insample_s.astype(float)
    insample_values = insample_s.to_numpy()
    if len(insample_values) <= seasonality_m or np.isinf(insample_values).any():
        return float("nan")

    # Darts requires insample to end immediately before the prediction.
    split = len(insample_values)
    insample_ts = TimeSeries.from_times_and_values(
        pd.RangeIndex(split), insample_values
    )
    eval_index = pd.RangeIndex(split, split + len(actual_values))
    actual_ts = TimeSeries.from_times_and_values(eval_index, actual_values)
    pred_ts = TimeSeries.from_times_and_values(eval_index, pred_values)
    try:
        value = float(
            darts_rmsse(
                actual_series=actual_ts,
                pred_series=pred_ts,
                insample=insample_ts,
                m=int(seasonality_m),
            )
        )
        return value if np.isfinite(value) else float("nan")
    except Exception as exc:
        logging.warning("RMSSE no computable: %s", exc)
        return float("nan")


__all__ = ["compute_mase", "compute_rmsse", "metric_higher_is_better"]
