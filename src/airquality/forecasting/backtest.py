"""Multi-step forecasting backtest on a held-out tail window.

Used to compare forecasting error between a *raw* hourly series and its
*preprocessed* (anomaly-removed + imputed) version. To keep the comparison fair
both arms forecast over the **same** contiguous, observed holdout window and are
scored against the **same** observed values; only the training data (and thus the
learned weights) differ.

The forecast input (context + holdout) is a contiguous observed block of the raw
series, so neither arm needs its gaps imputed *for inference* -- the preprocessing
effect is carried entirely by the trained model.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mase as darts_mase, rmse
from darts.utils.missing_values import extract_subseries
from sklearn.preprocessing import StandardScaler

from airquality.data.series import ensure_datetime_series
from airquality.modeling.training import fit_darts_model
from airquality.modeling.training_config import build_model_configs


def _compute_backtest_mase(
    actual: TimeSeries,
    pred: TimeSeries,
    insample: pd.Series,
    *,
    seasonality_m: int,
    freq: str,
) -> float:
    """MASE of the holdout forecast via :func:`darts.metrics.mase`.

    ``insample`` is the training history: darts requires it to end exactly one
    step before ``pred`` starts, which holds by construction here. Its gaps are
    time-interpolated first so the seasonal-naive scale uses the full history
    (same convention as ``imputation.benchmark._compute_gap_mase``). Returns
    NaN when darts cannot compute the metric (short or constant history).
    """
    try:
        insample_clean = (
            insample.interpolate(method="time", limit_direction="both").ffill().bfill()
        )
        insample_ts = TimeSeries.from_series(insample_clean, freq=freq)
        value = darts_mase(
            actual_series=actual,
            pred_series=pred,
            insample=insample_ts,
            m=int(seasonality_m),
            intersect=True,
        )
        if isinstance(value, (list, np.ndarray)):
            value = np.nanmean(value)
        value = float(value)
        return value if np.isfinite(value) else float("nan")
    except Exception as exc:
        logging.warning("[mase] no computable sobre el holdout: %s", exc)
        return float("nan")


def _longest_observed_run(series: pd.Series) -> tuple[int, int]:
    """Return ``(start, end)`` positions of the longest contiguous non-NaN run.

    Real station series are gappy and their longest observed block is rarely at
    the tail, so the eval window is carved from the longest run anywhere.
    """
    observed = series.notna().to_numpy()
    best_start, best_end = 0, 0
    cur_start = None
    for i, is_obs in enumerate(observed):
        if is_obs and cur_start is None:
            cur_start = i
        elif not is_obs and cur_start is not None:
            if i - cur_start > best_end - best_start:
                best_start, best_end = cur_start, i
            cur_start = None
    if cur_start is not None and len(observed) - cur_start > best_end - best_start:
        best_start, best_end = cur_start, len(observed)
    return best_start, best_end


def select_holdout_window(
    series: pd.Series,
    *,
    holdout: int,
    context_len: int,
    freq: str = "h",
) -> dict | None:
    """Pick a contiguous observed window = ``context_len`` context + ``holdout`` test.

    The window is carved from the **end of the longest contiguous observed run**
    (anywhere in the series); training uses everything before it. Returns ``None``
    when the longest run cannot host both context and holdout.
    """
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    start, end = _longest_observed_run(s)
    run_len = end - start
    needed = context_len + holdout
    if run_len < needed:
        return None

    eval_start = end - needed
    holdout_start_pos = end - holdout
    index = pd.DatetimeIndex(s.index)
    return {
        "eval_index": index[eval_start:end],
        "context_index": index[eval_start:holdout_start_pos],
        "holdout_index": index[holdout_start_pos:end],
        "holdout_start": index[holdout_start_pos],
    }


def backtest_forecast(
    train_series: pd.Series,
    eval_series: pd.Series,
    model_name: str,
    *,
    size_k: int,
    holdout_start: pd.Timestamp,
    seasonality_m: int = 24,
    freq: str = "h",
) -> dict:
    """Train ``model_name`` on ``train_series`` and backtest over the holdout.

    ``train_series`` may contain gaps (raw arm): it is split into gap-free
    subseries for training. ``eval_series`` is the contiguous observed block
    (context + holdout) shared by both arms. Returns RMSE/MAE/MASE plus metadata.
    """
    result = {
        "model": model_name,
        "rmse": float("nan"),
        "mae": float("nan"),
        "mase": float("nan"),
        "n_eval": 0,
    }

    configs = build_model_configs()
    if model_name not in configs:
        raise ValueError(f"Modelo de forecasting desconocido: {model_name}")
    model_cls, model_kwargs = configs[model_name]
    input_chunk = int(model_kwargs.get("input_chunk_length", model_kwargs.get("lags", 72)) or 72)
    min_len = input_chunk + size_k

    train_s = ensure_datetime_series(train_series, freq=freq, name=str(train_series.name or "series"))
    train_ts = TimeSeries.from_series(train_s, freq=freq)
    subseries = sorted(
        (ss for ss in extract_subseries(train_ts, min_gap_size=1) if len(ss) >= min_len),
        key=len,
        reverse=True,
    )
    if not subseries:
        logging.warning("[%s] sin subseries entrenables (min_len=%d)", model_name, min_len)
        return result

    # Carve a disjoint validation block from the longest subseries so the Darts
    # EarlyStopping callback (monitors val_loss) has data; skip if too short.
    longest = subseries[0]
    val_block = max(size_k, min(48, len(longest) // 5))
    if len(longest) < min_len + val_block + size_k:
        logging.warning("[%s] subserie insuficiente para split train/val", model_name)
        return result
    train_subs = [longest[:-val_block], *subseries[1:]]
    val_subs = [longest[-(input_chunk + val_block):]]

    scaler = Scaler(global_fit=True, scaler=StandardScaler())
    train_scaled = scaler.fit_transform([ss.astype(np.float32) for ss in train_subs])
    val_scaled = scaler.transform([ss.astype(np.float32) for ss in val_subs])

    eval_s = ensure_datetime_series(eval_series, freq=freq, name=str(eval_series.name or "series"))
    eval_ts = TimeSeries.from_series(eval_s, freq=freq).astype(np.float32)
    eval_scaled = scaler.transform(eval_ts)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_darts_model(model_cls, train_scaled, val_scaled, size_k, model_kwargs)
        try:
            forecasts = model.historical_forecasts(
                series=eval_scaled,
                start=holdout_start,
                forecast_horizon=size_k,
                stride=size_k,
                retrain=False,
                last_points_only=False,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - model/series specific
            logging.warning("[%s] historical_forecasts fallo: %s", model_name, exc)
            return result

    if not forecasts:
        return result
    pred_scaled = concatenate(forecasts) if isinstance(forecasts, list) else forecasts
    pred_ts = scaler.inverse_transform(pred_scaled)

    # Align predictions and observed actuals on their common timestamps.
    actual = eval_ts.slice_intersect(pred_ts)
    pred = pred_ts.slice_intersect(actual)
    if len(actual) == 0:
        return result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["rmse"] = float(rmse(actual, pred))
        result["mae"] = float(mae(actual, pred))
        result["mase"] = _compute_backtest_mase(
            actual,
            pred,
            train_s,
            seasonality_m=seasonality_m,
            freq=freq,
        )
    result["n_eval"] = int(len(actual))
    return result


__all__ = ["select_holdout_window", "backtest_forecast"]
