"""Multi-step forecasting backtest on a fixed test window.

Used to compare forecasting error between a *raw* hourly series and its
*preprocessed* (anomaly-removed + imputed) version. Every arm forecasts over the
same fixed, contiguous holdout selected by the pipeline's common support.

The test input (context + targets) is a contiguous observed block of the raw
series, so neither arm needs its gaps imputed *for inference* -- the preprocessing
effect is carried entirely by the trained model.
"""

from __future__ import annotations

import logging
import time
from typing import Mapping
import warnings

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import extract_subseries
from sklearn.preprocessing import StandardScaler

from airquality.data.series import ensure_datetime_series
from airquality.forecasting.registry import (
    ForecastModelConfig,
    resolve_forecasting_model_configs,
)
from airquality.metrics import compute_mase
from airquality.modeling.training import (
    DartsModelSeriesRequirements,
    fit_darts_model,
    get_model_series_requirements,
)


def _observed_runs(series: pd.Series) -> list[tuple[int, int]]:
    """Return ``(start, end)`` positions of every contiguous non-NaN run, in order."""
    observed = series.notna().to_numpy()
    runs: list[tuple[int, int]] = []
    cur_start: int | None = None
    for i, is_obs in enumerate(observed):
        if is_obs and cur_start is None:
            cur_start = i
        elif not is_obs and cur_start is not None:
            runs.append((cur_start, i))
            cur_start = None
    if cur_start is not None:
        runs.append((cur_start, len(observed)))
    return runs


def select_holdout_window(
    series: pd.Series,
    *,
    holdout: int,
    context_len: int,
    train_min_len: int,
    validation_len: int = 48,
    freq: str = "h",
    host_min_len: int | None = None,
) -> dict | None:
    """Reserve the latest viable observed block for context plus fixed test.

    A candidate run must provide ``context_len`` observed context points followed
    by exactly ``holdout`` observed targets. All timestamps before the first
    target remain available to train/validation, including an earlier prefix of
    the same run and the inference context.

    The history strictly before the first target must contain a block long enough
    for ``host_min_len`` when supplied, otherwise for ``train_min_len`` training
    points plus ``validation_len`` held-out targets. Returns ``None`` when no run
    satisfies both requirements.
    """
    if min(holdout, context_len, train_min_len, validation_len) <= 0:
        raise ValueError("Las longitudes de train, validacion y test deben ser positivas")
    if host_min_len is not None and host_min_len <= 0:
        raise ValueError("host_min_len debe ser positivo")

    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    runs = _observed_runs(s)
    host_len = host_min_len or train_min_len + validation_len
    index = pd.DatetimeIndex(s.index)
    for start, end in reversed(runs):
        if end - start < context_len + holdout:
            continue

        test_target_pos = end - holdout
        test_start_pos = test_target_pos - context_len
        train_runs = _observed_runs(s.iloc[:test_target_pos])
        if not any(host_end - host_start >= host_len for host_start, host_end in train_runs):
            continue

        return {
            "train_index": index[:test_target_pos],
            "train_end": index[test_target_pos - 1],
            "test_context_start": index[test_start_pos],
            "source_run_start": index[start],
            "test_index": index[test_start_pos:end],
            "context_index": index[test_start_pos:test_target_pos],
            "test_target_index": index[test_target_pos:end],
            "test_target_start": index[test_target_pos],
            "test_target_end": index[end - 1],
            "test_target_hours": holdout,
        }
    return None


def split_train_val_subseries(
    train_ts: TimeSeries,
    *,
    input_chunk: int,
    size_k: int,
    validation_len: int = 48,
    validation_stride: int | None = None,
    requirements: DartsModelSeriesRequirements | None = None,
) -> tuple[list[TimeSeries], list[TimeSeries]] | None:
    """Split a (possibly gappy) training series into train/val subseries.

    The validation block driving the Darts EarlyStopping is the tail of the
    **most recent** gap-free block long enough to host it; training uses only
    strictly-earlier data (every earlier block plus that host's prefix), and any
    block *after* the validation host is dropped. Ordering the split this way
    keeps ``val_loss`` honest: no training point reaches the first native
    validation target. Only the short recent blocks that cannot host a
    validation tail are dropped, so a large posterior block is never discarded
    (it would be the host instead).

    ``val_subs`` contains one exact native Darts fit window per rolling origin.
    For shifted TCN/RNN datasets, the whole native target sequence is kept
    strictly after the training boundary. Keeping each window minimal makes
    Darts produce exactly one validation sample from it.

    Returns ``(train_subs, val_subs)`` or ``None`` when no block can host the
    model's native fit and validation windows.
    """
    validation_stride = size_k if validation_stride is None else validation_stride
    if min(input_chunk, size_k, validation_len, validation_stride) <= 0:
        raise ValueError("Las longitudes y el stride de validacion deben ser positivos")
    if validation_len < size_k:
        raise ValueError("validation_len debe ser >= size_k")

    if requirements is None:
        min_len = input_chunk + size_k
        validation_target_offset: int | None = input_chunk
        validation_target_length = size_k
    else:
        min_len = requirements.min_train_series_length
        validation_target_offset = requirements.validation_target_offset
        validation_target_length = requirements.validation_target_length

    # extract_subseries yields the gap-free blocks in chronological order.
    subseries = [ss for ss in extract_subseries(train_ts, min_gap_size=1) if len(ss) >= min_len]
    if not subseries:
        return None

    if validation_target_offset is None:
        return subseries, []

    # Scan from the newest block back to the first one long enough to host the
    # fixed validation tail plus a training prefix.
    for i in range(len(subseries) - 1, -1, -1):
        val_block = max(validation_target_length, validation_len)
        if len(subseries[i]) < min_len + val_block:
            continue
        val_host = subseries[i]
        train_subs = [
            ss for ss in (*subseries[:i], val_host[:-val_block]) if len(ss) >= min_len
        ]
        if not train_subs:
            return None
        first_target = len(val_host) - val_block
        val_subs = []
        for target in range(
            first_target,
            len(val_host) - validation_target_length + 1,
            validation_stride,
        ):
            window_start = target - validation_target_offset
            val_subs.append(val_host[window_start : window_start + min_len])
        return train_subs, val_subs
    return None


def get_forecast_model_requirements(
    config: ForecastModelConfig,
    *,
    size_k: int,
    seasonality_m: int,
    context_len: int,
) -> DartsModelSeriesRequirements:
    """Return native or protocol-specific train geometry for one model family."""
    if config.mode == "trained":
        return get_model_series_requirements(config.model_cls, config.kwargs, size_k)
    if config.mode == "local":
        return DartsModelSeriesRequirements(
            min_train_series_length=max(10, 2 * seasonality_m),
            prediction_context_length=10,
            validation_target_offset=None,
            validation_target_length=0,
        )
    return DartsModelSeriesRequirements(
        min_train_series_length=context_len + size_k,
        prediction_context_length=context_len,
        validation_target_offset=None,
        validation_target_length=0,
    )


def get_strict_forecast_requirements(
    model_configs: Mapping[str, ForecastModelConfig],
    *,
    size_k: int,
    validation_len: int,
    validation_stride: int,
    seasonality_m: int,
    context_len: int,
    training_arms_only: bool = False,
) -> dict[str, object]:
    """Return the conservative geometry envelope across configured models."""
    if min(size_k, validation_len, validation_stride, seasonality_m, context_len) <= 0:
        raise ValueError("Los requisitos de forecasting deben ser positivos")
    candidates = []
    for name, config in model_configs.items():
        if training_arms_only and not config.uses_training_arms:
            continue
        native = get_forecast_model_requirements(
            config,
            size_k=size_k,
            seasonality_m=seasonality_m,
            context_len=context_len,
        )
        reserve = (
            max(validation_len, native.validation_target_length)
            if native.validation_target_offset is not None
            else 0
        )
        candidates.append(
            {
                "model": name,
                "minimum": native.min_train_series_length,
                "prediction_context": native.prediction_context_length,
                "reserve": reserve,
                "host": native.min_train_series_length + reserve,
                "validation_forecasts": (
                    (reserve - native.validation_target_length) // validation_stride + 1
                    if reserve
                    else 0
                ),
            }
        )
    if not candidates:
        raise ValueError("No hay modelos compatibles para calcular requisitos")

    minimum = max(int(item["minimum"]) for item in candidates)
    native_context = max(int(item["prediction_context"]) for item in candidates)
    prediction_context = max(context_len, native_context)
    reserve = max(int(item["reserve"]) for item in candidates)
    host = max(int(item["host"]) for item in candidates)
    return {
        "minimum_hours": minimum,
        "minimum_models": "/".join(
            str(item["model"]) for item in candidates if item["minimum"] == minimum
        ),
        "prediction_context_hours": prediction_context,
        "context_models": "/".join(
            str(item["model"])
            for item in candidates
            if item["prediction_context"] == prediction_context
        ) or "configured_context",
        "validation_hours": reserve,
        "host_minimum_hours": host,
        "limiting_models": "/".join(
            str(item["model"]) for item in candidates if item["host"] == host
        ),
        "validation_forecasts": min(
            int(item["validation_forecasts"])
            for item in candidates
            if item["reserve"] == reserve
        ),
    }


def _fit_forecast_model(
    config: ForecastModelConfig,
    train_scaled: list[TimeSeries],
    val_scaled: list[TimeSeries],
    *,
    size_k: int,
):
    """Fit a trained/global model or register a local/zero-shot model."""
    if config.mode == "local":
        model = config.model_cls(**config.kwargs)
        model.fit(train_scaled[-1], verbose=False)
        return model
    return fit_darts_model(
        config.model_cls,
        train_scaled,
        val_scaled,
        size_k,
        config.kwargs,
    )


def prepare_foundation_model(
    train_series: pd.Series,
    model_name: str,
    *,
    size_k: int,
    seasonality_m: int = 24,
    freq: str = "h",
    context_len: int = 72,
    model_config: ForecastModelConfig | None = None,
) -> tuple[object, Scaler, pd.Series, float]:
    """Load one frozen foundation model and its shared raw-history scaler."""
    if model_config is None:
        model_config = resolve_forecasting_model_configs(
            [model_name],
            seasonality_m=seasonality_m,
            context_length=context_len,
        )[model_name]
    if model_config.mode != "foundation":
        raise ValueError("prepare_foundation_model requiere un modelo foundation")

    train_s = ensure_datetime_series(
        train_series, freq=freq, name=str(train_series.name or "series")
    )
    requirements = get_forecast_model_requirements(
        model_config,
        size_k=size_k,
        seasonality_m=seasonality_m,
        context_len=context_len,
    )
    split = split_train_val_subseries(
        TimeSeries.from_series(train_s, freq=freq),
        input_chunk=requirements.prediction_context_length,
        size_k=size_k,
        validation_len=size_k,
        requirements=requirements,
    )
    if split is None:
        raise ValueError(
            f"{model_name}: no hay bloque para registrar el foundation "
            f"(min_len={requirements.min_train_series_length})"
        )
    train_subs, _ = split
    latest = train_subs[-1].astype(np.float32)
    scaler = Scaler(global_fit=True, scaler=StandardScaler())
    train_scaled = [scaler.fit_transform(latest)]

    load_start = time.perf_counter()
    model = _fit_forecast_model(model_config, train_scaled, [], size_k=size_k)
    return model, scaler, train_s, time.perf_counter() - load_start


def forecast_foundation_context(
    model: object,
    scaler: Scaler,
    context: pd.Series,
    target: pd.Series,
    mase_insample: pd.Series,
    *,
    seasonality_m: int = 24,
    freq: str = "h",
) -> dict[str, float | int]:
    """Forecast one exact target horizon from one as-of-origin context."""
    context_s = ensure_datetime_series(
        context, freq=freq, name=str(context.name or "series")
    )
    target_s = ensure_datetime_series(
        target, freq=freq, name=str(target.name or context_s.name)
    )
    if context_s.index[-1] + pd.tseries.frequencies.to_offset(freq) != target_s.index[0]:
        raise ValueError("El target debe comenzar inmediatamente despues del contexto")

    context_ts = TimeSeries.from_series(context_s, freq=freq).astype(np.float32)
    context_scaled = scaler.transform(context_ts)
    inference_start = time.perf_counter()
    prediction = model.predict(
        n=len(target_s),
        series=context_scaled,
        verbose=False,
    )
    inference_seconds = time.perf_counter() - inference_start
    if isinstance(prediction, list):
        if len(prediction) != 1:
            raise RuntimeError("El foundation devolvio un numero inesperado de forecasts")
        prediction = prediction[0]
    prediction = scaler.inverse_transform(prediction)
    if not pd.DatetimeIndex(prediction.time_index).equals(target_s.index):
        raise RuntimeError("El foundation devolvio timestamps distintos del target")

    actual = TimeSeries.from_series(target_s, freq=freq)
    errors = target_s.to_numpy(dtype=float) - prediction.to_series().to_numpy(dtype=float)
    if not np.isfinite(errors).all():
        raise RuntimeError("El foundation devolvio predicciones no finitas")
    insample = ensure_datetime_series(
        mase_insample,
        freq=freq,
        name=str(mase_insample.name or context_s.name),
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mase": compute_mase(
            actual,
            prediction,
            insample,
            seasonality_m=seasonality_m,
        ),
        "inference_seconds": inference_seconds,
        "n_test_predictions": len(errors),
    }


def backtest_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    model_name: str,
    *,
    size_k: int,
    test_target_start: pd.Timestamp,
    seasonality_m: int = 24,
    freq: str = "h",
    mase_insample: pd.Series | None = None,
    validation_len: int = 48,
    validation_stride: int | None = None,
    forecast_stride: int | None = None,
    context_len: int = 72,
    model_config: ForecastModelConfig | None = None,
) -> dict:
    """Train ``model_name`` on ``train_series`` and backtest over the holdout.

    ``train_series`` may contain gaps (raw arm): it is split into gap-free
    subseries for training. Trained global models use the causal validation split
    from :func:`split_train_val_subseries`; local statistical and zero-shot
    foundation models use the latest eligible block without validation.
    ``test_series`` is the
    contiguous observed block
    (context + holdout) shared by both arms. Returns RMSE/MAE/MASE, the model
    ``train_seconds`` (wall time of the ``fit`` only) and ``inference_seconds``
    (wall time of ``historical_forecasts`` over the holdout), plus metadata.
    Validation origins are explicit minimal windows separated by
    ``validation_stride``; test origins use ``forecast_stride`` and may overlap.

    ``mase_insample`` overrides the in-sample history behind the MASE
    seasonal-naive denominator (defaults to ``train_series``). When comparing
    preprocessing arms, pass the shared RAW training series for every arm:
    cleaning/imputation smooth the history and shrink its naive error, so
    per-arm denominators would inflate the preprocessed arms' MASE.

    """
    if model_config is None:
        model_config = resolve_forecasting_model_configs(
            [model_name],
            seasonality_m=seasonality_m,
            context_length=context_len,
        )[model_name]

    result = {
        "model": model_name,
        "model_mode": model_config.mode,
        "rmse": float("nan"),
        "mae": float("nan"),
        "mase": float("nan"),
        "train_seconds": float("nan"),
        "inference_seconds": float("nan"),
        "n_test_predictions": 0,
        "n_forecasts": 0,
        "n_expected_forecasts": 0,
        "n_unique_targets": 0,
        "origin_mae_mean": float("nan"),
        "origin_mae_std": float("nan"),
        "origin_rmse_mean": float("nan"),
        "origin_rmse_std": float("nan"),
    }

    requirements = get_forecast_model_requirements(
        model_config,
        size_k=size_k,
        seasonality_m=seasonality_m,
        context_len=context_len,
    )
    input_chunk = requirements.prediction_context_length
    min_len = requirements.min_train_series_length
    validation_stride = size_k if validation_stride is None else validation_stride
    forecast_stride = size_k if forecast_stride is None else forecast_stride
    if min(validation_stride, forecast_stride) <= 0:
        raise ValueError("Los strides de validacion y forecast deben ser positivos")

    train_s = ensure_datetime_series(train_series, freq=freq, name=str(train_series.name or "series"))
    train_ts = TimeSeries.from_series(train_s, freq=freq)
    split = split_train_val_subseries(
        train_ts,
        input_chunk=input_chunk,
        size_k=size_k,
        validation_len=validation_len,
        validation_stride=validation_stride,
        requirements=requirements,
    )
    if split is None:
        logging.warning("[%s] sin bloque entrenable (min_len=%d)", model_name, min_len)
        return result
    train_subs, val_subs = split
    if model_config.mode != "trained":
        train_subs = [train_subs[-1]]

    scaler = Scaler(global_fit=True, scaler=StandardScaler())
    train_scaled = scaler.fit_transform([ss.astype(np.float32) for ss in train_subs])
    val_scaled = (
        scaler.transform([ss.astype(np.float32) for ss in val_subs])
        if val_subs
        else []
    )

    test_s = ensure_datetime_series(
        test_series, freq=freq, name=str(test_series.name or "series")
    )
    try:
        test_target_pos = int(test_s.index.get_loc(test_target_start))
    except KeyError as exc:
        raise ValueError("test_target_start debe pertenecer a test_series") from exc
    expected_positions = list(
        range(test_target_pos, len(test_s) - size_k + 1, forecast_stride)
    )
    expected_starts = pd.DatetimeIndex(test_s.index[expected_positions])
    result["n_expected_forecasts"] = len(expected_starts)
    test_ts = TimeSeries.from_series(test_s, freq=freq).astype(np.float32)
    test_scaled = scaler.transform(test_ts)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_start = time.perf_counter()
        model = _fit_forecast_model(
            model_config,
            train_scaled,
            val_scaled,
            size_k=size_k,
        )
        result["train_seconds"] = time.perf_counter() - fit_start
        try:
            inference_start = time.perf_counter()
            forecasts = model.historical_forecasts(
                series=test_scaled,
                start=test_target_start,
                forecast_horizon=size_k,
                stride=forecast_stride,
                retrain=False,
                last_points_only=False,
                verbose=False,
            )
            result["inference_seconds"] = time.perf_counter() - inference_start
        except Exception as exc:  # pragma: no cover - model/series specific
            logging.warning("[%s] historical_forecasts fallo: %s", model_name, exc)
            return result

    if not forecasts:
        return result
    forecast_list = forecasts if isinstance(forecasts, list) else [forecasts]
    actual_starts = pd.DatetimeIndex(forecast.start_time() for forecast in forecast_list)
    exact_windows = len(forecast_list) == len(expected_positions) and actual_starts.equals(
        expected_starts
    )
    if exact_windows:
        exact_windows = all(
            pd.DatetimeIndex(forecast.time_index).equals(
                pd.DatetimeIndex(test_s.index[position : position + size_k])
            )
            for forecast, position in zip(forecast_list, expected_positions, strict=True)
        )
    if not exact_windows:
        logging.warning(
            "[%s] Darts devolvio ventanas distintas al plan explicito de test",
            model_name,
        )
        return result
    predictions = [scaler.inverse_transform(forecast) for forecast in forecast_list]

    insample = (
        train_s
        if mase_insample is None
        else ensure_datetime_series(mase_insample, freq=freq, name=str(mase_insample.name or "series"))
    )
    actual_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    origin_mae: list[float] = []
    origin_rmse: list[float] = []
    origin_mase: list[float] = []
    origin_lengths: list[int] = []
    target_times: set[pd.Timestamp] = set()
    for prediction in predictions:
        actual = test_ts.slice_intersect(prediction)
        pred = prediction.slice_intersect(actual)
        if len(actual) == 0:
            continue
        actual_array = actual.to_series().to_numpy(dtype=float)
        predicted_array = pred.to_series().to_numpy(dtype=float)
        errors = actual_array - predicted_array
        actual_values.append(actual_array)
        predicted_values.append(predicted_array)
        origin_mae.append(float(np.mean(np.abs(errors))))
        origin_rmse.append(float(np.sqrt(np.mean(np.square(errors)))))
        origin_mase.append(
            compute_mase(
                actual,
                pred,
                insample,
                seasonality_m=seasonality_m,
            )
        )
        origin_lengths.append(len(actual_array))
        target_times.update(pd.DatetimeIndex(actual.time_index))

    if not actual_values:
        return result

    actual_array = np.concatenate(actual_values)
    predicted_array = np.concatenate(predicted_values)
    errors = actual_array - predicted_array
    result["mae"] = float(np.mean(np.abs(errors)))
    result["rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    finite_mase = np.isfinite(origin_mase)
    if np.any(finite_mase):
        result["mase"] = float(
            np.average(
                np.asarray(origin_mase)[finite_mase],
                weights=np.asarray(origin_lengths)[finite_mase],
            )
        )
    result["n_test_predictions"] = int(len(errors))
    result["n_forecasts"] = len(origin_mae)
    result["n_unique_targets"] = len(target_times)
    result["origin_mae_mean"] = float(np.mean(origin_mae))
    result["origin_mae_std"] = float(np.std(origin_mae))
    result["origin_rmse_mean"] = float(np.mean(origin_rmse))
    result["origin_rmse_std"] = float(np.std(origin_rmse))
    return result


__all__ = [
    "select_holdout_window",
    "get_forecast_model_requirements",
    "get_strict_forecast_requirements",
    "backtest_forecast",
    "forecast_foundation_context",
    "prepare_foundation_model",
    "split_train_val_subseries",
]
