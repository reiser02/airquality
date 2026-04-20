from __future__ import annotations

import os  # Read optional Hugging Face token from environment variables.
import warnings  # Suppress optional runtime warnings in prediction helpers.
from dataclasses import dataclass  # Structured diagnostics for skipped gaps.
from typing import Any, Mapping, Sequence  # Typing utilities for flexible public API.

import numpy as np  # Numeric operations for masks, metrics, and random sampling.
import pandas as pd  # Time-indexed series/dataframe processing.
from pandas.tseries.frequencies import (
    to_offset,
)  # Frequency-aware timestamp arithmetic.

from darts import TimeSeries  # Darts time series container used across the module.


try:
    from tsfm_public import (
        TimeSeriesPreprocessor,
    )  # TSPulse feature extractor / preprocessor.
    from tsfm_public.models.tspulse import (
        TSPulseForReconstruction,
    )  # TSPulse reconstruction model.
    from tsfm_public.toolkit.time_series_imputation_pipeline import (  # Official zero-shot imputation pipeline.
        TimeSeriesImputationPipeline,
    )

    TSFM_PUBLIC_AVAILABLE = True
    TSFM_PUBLIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency path.
    TimeSeriesPreprocessor = None
    TSPulseForReconstruction = None
    TimeSeriesImputationPipeline = None
    TSFM_PUBLIC_AVAILABLE = False
    TSFM_PUBLIC_IMPORT_ERROR = exc


DEFAULT_CONFIG_WORKERS = {
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
}


@dataclass(frozen=True)
class GapContextFailure:
    """Diagnostic payload describing why one Darts gap could not be imputed."""

    model_name: str
    series_name: str
    gap_start: pd.Timestamp
    gap_length: int
    required_context: int
    available_context: int
    reason: str


def _ensure_datetime_series(series: pd.Series, *, freq: str, name: str) -> pd.Series:
    """Validate and normalize one pandas series to a regular datetime grid.

    Parameters
    ----------
    series : pd.Series
        Series to validate.
    freq : str
        Frequency string used to regularize the index (`asfreq`).
    name : str
        Logical name used in error messages.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} debe ser pd.Series; recibido: {type(series)}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{name} debe tener DatetimeIndex")

    out = series.sort_index().copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.asfreq(freq)
    out.name = str(series.name) if series.name is not None else name
    return out.astype(float)


def _ts_to_series(ts: TimeSeries, *, freq: str, name: str) -> pd.Series:
    """Convert Darts `TimeSeries` into normalized `pd.Series`."""
    out = ts.to_series()
    out.name = name
    return _ensure_datetime_series(out, freq=freq, name=name)


def _normalize_series_collection(
    series_like: Mapping[str, Any]
    | Sequence[Any]
    | pd.DataFrame
    | pd.Series
    | TimeSeries,
    *,
    freq: str,
    default_prefix: str,
) -> dict[str, pd.Series]:
    """Normalize one-or-many input series into `{series_name: pd.Series}`.

    Supported inputs:
    - `pd.Series`
    - `TimeSeries`
    - `pd.DataFrame` with one or more columns
    - mapping `{name: (Series | TimeSeries | single-column DataFrame)}`
    - sequence of Series/TimeSeries/single-column DataFrames
    """
    out: dict[str, pd.Series] = {}

    if isinstance(series_like, pd.Series):
        name = (
            str(series_like.name)
            if series_like.name is not None
            else f"{default_prefix}_0"
        )
        out[name] = _ensure_datetime_series(series_like, freq=freq, name=name)
        return out

    if isinstance(series_like, TimeSeries):
        raw = series_like.to_series()
        name = str(raw.name) if raw.name is not None else f"{default_prefix}_0"
        out[name] = _ts_to_series(series_like, freq=freq, name=name)
        return out

    if isinstance(series_like, pd.DataFrame):
        if not isinstance(series_like.index, pd.DatetimeIndex):
            raise TypeError("DataFrame de series debe tener DatetimeIndex")
        for col in series_like.columns:
            out[str(col)] = _ensure_datetime_series(
                series_like[col], freq=freq, name=str(col)
            )
        return out

    if isinstance(series_like, Mapping):
        for key, value in series_like.items():
            name = str(key)
            if isinstance(value, pd.Series):
                s = value.copy()
                s.name = name
                out[name] = _ensure_datetime_series(s, freq=freq, name=name)
            elif isinstance(value, TimeSeries):
                out[name] = _ts_to_series(value, freq=freq, name=name)
            elif isinstance(value, pd.DataFrame):
                if len(value.columns) != 1:
                    raise ValueError(
                        "Cada DataFrame del mapeo debe tener exactamente una columna"
                    )
                s = value.iloc[:, 0].copy()
                s.name = name
                out[name] = _ensure_datetime_series(s, freq=freq, name=name)
            else:
                raise TypeError(f"Tipo no soportado para '{name}': {type(value)}")
        return out

    if isinstance(series_like, Sequence) and not isinstance(series_like, (str, bytes)):
        for i, value in enumerate(series_like):
            auto = f"{default_prefix}_{i}"
            if isinstance(value, pd.Series):
                s = value.copy()
                if s.name is None:
                    s.name = auto
                out[str(s.name)] = _ensure_datetime_series(
                    s, freq=freq, name=str(s.name)
                )
            elif isinstance(value, TimeSeries):
                raw = value.to_series()
                name = str(raw.name) if raw.name is not None else auto
                out[name] = _ts_to_series(value, freq=freq, name=name)
            elif isinstance(value, pd.DataFrame):
                if len(value.columns) != 1:
                    raise ValueError(
                        "Cada DataFrame de secuencia debe tener exactamente una columna"
                    )
                s = value.iloc[:, 0].copy()
                if s.name is None:
                    s.name = auto
                out[str(s.name)] = _ensure_datetime_series(
                    s, freq=freq, name=str(s.name)
                )
            else:
                raise TypeError(f"Elemento no soportado en posicion {i}: {type(value)}")
        return out

    raise TypeError(f"Formato de series no soportado: {type(series_like)}")


def _extract_test_series_from_dataset_bundle(
    dataset_bundle: Mapping[str, Any], *, freq: str
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    """Build unscaled test series from project `dataset_bundle` structure."""
    valid_cols = list(dataset_bundle["valid_cols"])
    series_test = list(dataset_bundle["series_test"])
    dict_scalers = dict(dataset_bundle.get("dict_scalers", {}))

    if len(valid_cols) != len(series_test):
        raise ValueError("`valid_cols` y `series_test` deben tener la misma longitud")

    out: dict[str, pd.Series] = {}
    for i, col in enumerate(valid_cols):
        ts = series_test[i]
        if not isinstance(ts, TimeSeries):
            raise TypeError("Cada elemento de `series_test` debe ser TimeSeries")
        scaler = dict_scalers.get(col)
        ts_unscaled = scaler.inverse_transform(ts) if scaler is not None else ts
        out[col] = _ts_to_series(ts_unscaled, freq=freq, name=col)

    return out, dict_scalers


def _build_all_series_map(
    all_series: Sequence[pd.DataFrame | pd.Series] | Mapping[str, Any] | None,
    *,
    freq: str,
) -> dict[str, pd.Series]:
    """Normalize optional full/reference series collection to `{name: series}`."""
    if all_series is None:
        return {}
    return _normalize_series_collection(
        all_series, freq=freq, default_prefix="all_series"
    )


def _derive_train_series_from_full(
    *,
    series_name: str,
    test_series: pd.Series,
    all_series_by_name: Mapping[str, pd.Series],
) -> pd.Series | None:
    """Derive train history as all points before the first test timestamp."""
    full = all_series_by_name.get(series_name)
    if full is None or len(full) == 0 or len(test_series) == 0:
        return None
    test_start = pd.Timestamp(test_series.index.min())
    out = full.loc[full.index < test_start].copy()
    return out if len(out) > 0 else None


def _build_gap_index(
    start: pd.Timestamp, length: int, *, freq: str
) -> pd.DatetimeIndex:
    """Create one contiguous datetime index representing a synthetic gap."""
    return pd.date_range(start=pd.Timestamp(start), periods=int(length), freq=freq)


def _sample_non_overlapping_starts(
    *,
    n_points: int,
    gap_size: int,
    num_gaps: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample non-overlapping block starts over `[0, n_points-gap_size]`."""
    if n_points < gap_size or num_gaps <= 0:
        return []

    candidates = np.arange(0, n_points - gap_size + 1, dtype=int)
    rng.shuffle(candidates)

    starts: list[int] = []
    for start in candidates:
        overlap = any(
            not (start + gap_size <= s or s + gap_size <= start) for s in starts
        )
        if overlap:
            continue
        starts.append(int(start))
        if len(starts) >= num_gaps:
            break

    return sorted(starts)


def _generate_block_gaps(
    *,
    series: pd.Series,
    gap_size: int,
    num_gaps: int,
    rng: np.random.Generator,
    freq: str,
) -> list[pd.DatetimeIndex]:
    """Generate fixed-size non-overlapping artificial block gaps."""
    starts = _sample_non_overlapping_starts(
        n_points=len(series),
        gap_size=int(gap_size),
        num_gaps=int(num_gaps),
        rng=rng,
    )
    return [_build_gap_index(series.index[s], gap_size, freq=freq) for s in starts]


def _generate_hybrid_tspulse_gaps(
    *,
    series: pd.Series,
    gap_size: int,
    num_gaps: int,
    rng: np.random.Generator,
    freq: str,
    random_fraction: float,
) -> list[pd.DatetimeIndex]:
    """Generate hybrid mask strategy (random points + blocks) like TSPulse notebook.

    The official notebook uses ~3/4 random missing points + ~1/4 block-missing points.
    """
    total_missing = max(1, int(gap_size * num_gaps))
    random_missing = int(total_missing * random_fraction)
    block_missing = max(0, total_missing - random_missing)

    block_count = max(1, block_missing // max(1, gap_size))
    block_windows = _generate_block_gaps(
        series=series,
        gap_size=gap_size,
        num_gaps=block_count,
        rng=rng,
        freq=freq,
    )

    used_points: set[pd.Timestamp] = set()
    for window in block_windows:
        used_points.update(window.tolist())

    available_points = [ts for ts in series.index if ts not in used_points]
    rng.shuffle(available_points)
    random_points = sorted(available_points[:random_missing])
    point_windows = [_build_gap_index(ts, 1, freq=freq) for ts in random_points]

    return sorted(block_windows + point_windows, key=lambda idx: pd.Timestamp(idx[0]))


def _gap_windows_to_mask_index(
    gap_windows: Sequence[pd.DatetimeIndex],
) -> pd.DatetimeIndex:
    """Flatten a list of gap windows to a sorted, unique mask index."""
    if not gap_windows:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    idx = pd.DatetimeIndex(np.concatenate([w.to_numpy() for w in gap_windows]))
    return idx.sort_values().drop_duplicates()


def _mask_test_series(
    test_series: pd.Series,
    gap_windows: Sequence[pd.DatetimeIndex],
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Apply NaNs at gap windows and return `(masked_series, mask_index)`."""
    mask_index = _gap_windows_to_mask_index(gap_windows)
    masked = test_series.copy()
    if len(mask_index) > 0:
        masked.loc[masked.index.intersection(mask_index)] = np.nan
    return masked, mask_index


def _max_context_from_lags(lags: Any) -> int:
    """Infer context length from Darts lag specs (`int`, sequence, or mapping)."""
    if lags is None:
        return 0
    if isinstance(lags, int):
        return max(0, int(lags))
    if isinstance(lags, (list, tuple, np.ndarray)):
        vals = [int(v) for v in lags if v is not None]
        if not vals:
            return 0
        negatives = [abs(v) for v in vals if v < 0]
        return max(negatives) if negatives else max(abs(v) for v in vals)
    if isinstance(lags, Mapping):
        return max((_max_context_from_lags(v) for v in lags.values()), default=0)
    return 0


def infer_darts_minimum_context(model: Any) -> int:
    """Infer a robust minimum clean-left-context size for Darts prediction.

    Priority is given to:
    - `input_chunk_length`
    - `training_length` (for autoregressive RNN-like models)
    - `lags`
    - `extreme_lags[0]` (minimum target lag)
    """
    required = 1

    for attr in ("input_chunk_length", "training_length"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            required = max(required, int(value))

    required = max(required, _max_context_from_lags(getattr(model, "lags", None)))

    extreme_lags = getattr(model, "extreme_lags", None)
    if isinstance(extreme_lags, tuple) and len(extreme_lags) >= 1:
        min_target_lag = extreme_lags[0]
        if isinstance(min_target_lag, int) and min_target_lag < 0:
            required = max(required, abs(min_target_lag))

    return int(required)


def _build_clean_left_context(
    *,
    test_series: pd.Series,
    train_series: pd.Series | None,
    gap_start: pd.Timestamp,
    required_context: int,
    freq: str,
) -> pd.Series:
    """Build contiguous, NaN-free left context ending exactly before one gap.

    If early test points are insufficient, this function prepends train history.
    """
    offset = to_offset(freq)
    cutoff = pd.Timestamp(gap_start) - offset

    parts: list[pd.Series] = []
    if train_series is not None and len(train_series) > 0:
        parts.append(train_series.copy())
    parts.append(test_series.loc[test_series.index <= cutoff].copy())

    history = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    history = history[~history.index.duplicated(keep="last")].asfreq(freq)
    if len(history) == 0 or cutoff not in history.index:
        return pd.Series(dtype=float)

    values = history.to_numpy(dtype=float)
    index = pd.DatetimeIndex(history.index)
    cutoff_pos = index.get_loc(cutoff)

    if not isinstance(cutoff_pos, (int, np.integer)):
        return pd.Series(dtype=float)

    start = int(cutoff_pos)
    need = int(required_context)
    while start >= 0 and need > 0 and np.isfinite(values[start]):
        start -= 1
        need -= 1

    slice_start = start + 1
    if slice_start > int(cutoff_pos):
        return pd.Series(dtype=float)

    return pd.Series(
        values[slice_start : int(cutoff_pos) + 1],
        index=index[slice_start : int(cutoff_pos) + 1],
        name=test_series.name,
    )


def darts_left_context_imputation_for_gaps(
    *,
    model_name: str,
    model: Any,
    series_name: str,
    test_series: pd.Series,
    train_series: pd.Series | None,
    gap_windows: Sequence[pd.DatetimeIndex],
    freq: str,
) -> tuple[pd.Series, list[GapContextFailure]]:
    """Impute each gap with Darts using only clean left context.

    Important behavior:
    - No interpolation/forward-fill is applied inside the gap for model input.
    - Context contains only valid points before the gap.
    - If test history is insufficient, train tail is prepended.
    - If still insufficient, the gap is skipped and left as NaN, with diagnostics.
    """
    required_context = infer_darts_minimum_context(model)
    failures: list[GapContextFailure] = []

    pred_out = pd.Series(
        index=_gap_windows_to_mask_index(gap_windows), dtype=float, name=series_name
    )
    if len(pred_out) == 0:
        return pred_out, failures

    for gap_idx in gap_windows:
        if len(gap_idx) == 0:
            continue

        gap_start = pd.Timestamp(gap_idx.min())
        context = _build_clean_left_context(
            test_series=test_series,
            train_series=train_series,
            gap_start=gap_start,
            required_context=required_context,
            freq=freq,
        )

        if len(context) < required_context:
            failures.append(
                GapContextFailure(
                    model_name=model_name,
                    series_name=series_name,
                    gap_start=gap_start,
                    gap_length=int(len(gap_idx)),
                    required_context=required_context,
                    available_context=int(len(context)),
                    reason=(
                        "No se alcanza el contexto minimo del modelo incluso "
                        "completando con train; se deja NaN en ese gap."
                    ),
                )
            )
            continue

        context_ts = TimeSeries.from_series(context, freq=freq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pred_ts = model.predict(
                    n=int(len(gap_idx)), series=context_ts, verbose=False
                )
            except TypeError:
                pred_ts = model.predict(n=int(len(gap_idx)), series=context_ts)

        pred_block = pred_ts.to_series().astype(float)
        if len(pred_block) == len(gap_idx):
            pred_block.index = pd.DatetimeIndex(gap_idx)
            pred_out.loc[pd.DatetimeIndex(gap_idx)] = pred_block.to_numpy(dtype=float)
        else:
            pred_out.loc[pd.DatetimeIndex(gap_idx)] = pred_block.reindex(
                gap_idx
            ).to_numpy(dtype=float)

    return pred_out, failures


def darts_historical_forecasts_for_mask(
    *,
    model: Any,
    ts_series_test: TimeSeries,
    scaler: Any,
    gap_starts: Sequence[int],
    gap_size: int,
    mask_index: pd.DatetimeIndex,
    config_workers: Mapping[str, Any] | None,
) -> pd.Series:
    """Legacy helper selecting Darts historical blocks over requested gaps.

    This helper exists for compatibility with existing tests and custom models
    exposing only `historical_forecasts()` (not `predict()`).
    """
    predict_kwargs: dict[str, Any] = {}
    if config_workers:
        predict_kwargs["dataloader_kwargs"] = dict(config_workers)

    raw_blocks = model.historical_forecasts(
        series=ts_series_test,
        forecast_horizon=int(gap_size),
        stride=1,
        retrain=False,
        last_points_only=False,
        verbose=False,
        predict_kwargs=predict_kwargs,
    )

    blocks = [raw_blocks] if isinstance(raw_blocks, TimeSeries) else list(raw_blocks)

    pred = pd.Series(index=pd.DatetimeIndex(mask_index), dtype=float)
    cursor = 0
    for start in gap_starts:
        if int(start) < 0 or int(start) >= len(blocks):
            cursor += int(gap_size)
            continue
        block_scaled = blocks[int(start)]
        block = (
            scaler.inverse_transform(block_scaled)
            if scaler is not None
            else block_scaled
        )
        block_pd = block.to_series().astype(float)

        next_cursor = min(cursor + int(gap_size), len(mask_index))
        target_idx = pd.DatetimeIndex(mask_index[cursor:next_cursor])
        pred.loc[target_idx] = block_pd.reindex(target_idx).to_numpy(dtype=float)
        cursor = next_cursor

    return pred


def build_tspulse_context_frame(
    *,
    test_series: pd.Series,
    train_series: pd.Series | None,
    context_length: int,
    freq: str,
    timestamp_column: str = "timestamp",
    target_column: str = "value",
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Build TSPulse-ready context frame from train tail + test segment.

    Behavior aligned with TSPulse context constraints (base context length 512):
    - If test history is shorter than required context, prepend train tail.
    - If still short, left-pad by extending timeline and filling from nearest values.
    - Return `(frame, original_test_index)`.
    """
    test_clean = _ensure_datetime_series(test_series, freq=freq, name="test_series")
    train_clean = (
        _ensure_datetime_series(train_series, freq=freq, name="train_series")
        if train_series is not None and len(train_series) > 0
        else pd.Series(dtype=float)
    )

    if len(test_clean) == 0:
        raise ValueError("`test_series` no puede estar vacia")

    combined = pd.concat([train_clean, test_clean]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    end_ts = pd.Timestamp(test_clean.index.max())
    context_index = pd.date_range(end=end_ts, periods=int(context_length), freq=freq)
    context_values = combined.reindex(context_index)
    context_values = (
        context_values.interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )

    if context_values.isna().any():
        raise ValueError(
            "No fue posible construir un contexto valido para TSPulse tras completar con train y padding"
        )

    frame = pd.DataFrame(
        {
            timestamp_column: context_index,
            target_column: context_values.to_numpy(dtype=float),
        }
    )
    return frame, pd.DatetimeIndex(test_clean.index)


def _torch_cuda_available() -> bool:
    """Return CUDA availability without hard dependency on torch import."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_mps_available() -> bool:
    """Return MPS availability without hard dependency on torch import."""
    try:
        import torch

        return bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        return False


class TSPulseHistoricalImputer:
    """Adapter exposing TSPulse imputation via a Darts-like API.

    This object supports `historical_imputation_forecasts(...)`
    for direct mask-based imputation.
    """

    def __init__(
        self,
        *,
        model_id: str = "ibm-granite/granite-timeseries-tspulse-r1",
        revision: str = "tspulse-hybrid-dualhead-512-p8-r1",
        context_length: int = 512,
        freq: str = "h",
        batch_size: int = 1000,
        device: str | None = None,
        scaling: bool = True,
        model: Any | None = None,
        hf_token: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Store adapter configuration and optional pre-loaded model instance."""
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.context_length = int(context_length)
        self.freq = str(freq)
        self.batch_size = int(batch_size)
        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if _torch_cuda_available()
                else "mps"
                if _torch_mps_available()
                else "cpu"
            )
        )
        self.scaling = bool(scaling)
        self.model = model
        self.hf_token = hf_token if hf_token is not None else os.getenv("HF_TOKEN")
        self.local_files_only = bool(local_files_only)

    def _ensure_model(self, num_input_channels: int) -> Any:
        """Load TSPulse model lazily on first use."""
        if self.model is not None:
            return self.model

        if not TSFM_PUBLIC_AVAILABLE:
            raise ImportError(
                "tsfm_public no esta disponible en este entorno"
            ) from TSFM_PUBLIC_IMPORT_ERROR

        self.model = TSPulseForReconstruction.from_pretrained(
            self.model_id,
            revision=self.revision,
            num_input_channels=int(num_input_channels),
            mask_type="user",
            token=self.hf_token,
            local_files_only=self.local_files_only,
        )
        return self.model

    def _impute_full_series(
        self,
        *,
        series: pd.Series,
        train_series: pd.Series | None,
        freq: str,
    ) -> pd.Series:
        """Run official TSPulse zero-shot imputation on one masked test series."""
        if not TSFM_PUBLIC_AVAILABLE:
            raise ImportError(
                "tsfm_public no esta instalado; no se puede ejecutar TSPulse"
            ) from TSFM_PUBLIC_IMPORT_ERROR

        prepared, test_index = build_tspulse_context_frame(
            test_series=series,
            train_series=train_series,
            context_length=self.context_length,
            freq=freq,
            timestamp_column="timestamp",
            target_column="value",
        )

        tsp = TimeSeriesPreprocessor(
            id_columns=[],
            timestamp_column="timestamp",
            target_columns=["value"],
            context_length=self.context_length,
            prediction_length=0,
            scaling=self.scaling,
            encode_categorical=False,
            scaler_type="standard",
        )
        tsp.train(prepared)

        model = self._ensure_model(num_input_channels=tsp.num_input_channels)
        pipe = TimeSeriesImputationPipeline(
            model,
            feature_extractor=tsp,
            batch_size=self.batch_size,
            device=self.device,
        )

        out = pipe(prepared)
        idx = pd.DatetimeIndex(out["timestamp"])
        value_col = "value_imputed" if "value_imputed" in out.columns else "value"
        imputed = pd.Series(
            out[value_col].to_numpy(dtype=float), index=idx, name=series.name
        )
        return imputed.reindex(test_index)

    def historical_imputation_forecasts(
        self,
        *,
        masked_test_series: pd.Series,
        train_series: pd.Series | None,
        mask_index: pd.DatetimeIndex,
        freq: str,
        **_: Any,
    ) -> pd.Series:
        """Impute all missing points and return predictions over mask timestamps."""
        imputed = self._impute_full_series(
            series=masked_test_series,
            train_series=train_series,
            freq=freq,
        )
        return imputed.reindex(mask_index)


def _compute_mase_denominator(
    insample: pd.Series, *, seasonality_m: int, freq: str
) -> float:
    """Compute MASE denominator from in-sample history (`mean(|y_t - y_{t-m}|)`)."""
    if len(insample) <= int(seasonality_m):
        return float("nan")

    s = _ensure_datetime_series(insample, freq=freq, name="insample")
    s = s.interpolate(method="time", limit_direction="both").ffill().bfill()
    values = s.to_numpy(dtype=float)
    m = int(seasonality_m)

    if len(values) <= m:
        return float("nan")

    denom = float(np.mean(np.abs(values[m:] - values[:-m])))
    return denom if np.isfinite(denom) and denom > 0 else float("nan")


def _compute_metrics_on_mask(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    mase_denominator: float,
    metrics: Sequence[str],
) -> dict[str, float]:
    """Compute selected metrics (MAE, RMSE, MASE) only on mask timestamps."""
    idx = y_true.index.intersection(y_pred.index)
    true_vals = y_true.reindex(idx).to_numpy(dtype=float)
    pred_vals = y_pred.reindex(idx).to_numpy(dtype=float)

    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if not np.any(valid):
        return {m.upper(): float("nan") for m in metrics}

    err = true_vals[valid] - pred_vals[valid]
    out: dict[str, float] = {}

    for metric_name in metrics:
        m = metric_name.lower()
        if m == "mae":
            out["MAE"] = float(np.mean(np.abs(err)))
        elif m == "rmse":
            out["RMSE"] = float(np.sqrt(np.mean(np.square(err))))
        elif m == "mase":
            mae_value = float(np.mean(np.abs(err)))
            out["MASE"] = (
                float(mae_value / mase_denominator)
                if np.isfinite(mase_denominator) and mase_denominator > 0
                else float("nan")
            )
        else:
            raise ValueError(f"Metrica no soportada: {metric_name}")

    return out


def execute_complete_pipeline(
    *,
    model_dict: Mapping[str, Any],
    test_series: Mapping[str, Any]
    | Sequence[Any]
    | pd.DataFrame
    | pd.Series
    | TimeSeries
    | None = None,
    train_series: Mapping[str, Any]
    | Sequence[Any]
    | pd.DataFrame
    | pd.Series
    | TimeSeries
    | None = None,
    dataset_bundle: Mapping[str, Any] | None = None,
    gap_sizes: Sequence[int] = (1, 2, 5, 10),
    num_gaps: int = 3,
    gap_strategy: str = "block",
    hybrid_random_fraction: float = 0.75,
    gap_spec_by_series: Mapping[str, Sequence[tuple[pd.Timestamp, int]]] | None = None,
    metrics: Sequence[str] = ("mae", "rmse", "mase"),
    seasonality_m: int = 24,
    freq: str = "h",
    random_seed: int = 42,
    config_workers: Mapping[str, Any] | None = None,
) -> tuple[
    pd.DataFrame, dict[int, dict[str, dict[str, pd.Series]]], dict[int, dict[str, Any]]
]:
    """Execute imputation benchmark across TSPulse and Darts models.

    Main responsibilities:
    - Receive one or many test series and already-loaded models.
    - Generate (or receive) artificial gaps compatible with TSPulse notebook ideas.
    - Impute with TSPulse and Darts.
    - Evaluate MAE, RMSE, and MASE strictly on missing points.
    - If `dataset_bundle` includes `all_series_unscaled`, use it as fallback
      history for train-context/MASE when `train_series` is not provided.
    - Return predictions, metrics, and plotting payload.

    Returns
    -------
    tuple[pd.DataFrame, dict, dict]
        `(results_df, predictions, plot_store)` where:
        - `results_df` has columns `Modelo, Serie, Gap_Size, [metricas...]`.
        - `predictions` is `{gap_size: {model_name: {series_name: pred_series}}}`.
        - `plot_store` matches existing plotting helpers in `complete_pipeline.py`.
    """
    if not model_dict:
        raise ValueError("`model_dict` no puede estar vacio")

    metric_list = [str(m).strip().lower() for m in metrics]
    for m in metric_list:
        if m not in {"mae", "rmse", "mase"}:
            raise ValueError(f"Metrica no soportada: {m}")

    if config_workers is None:
        config_workers = DEFAULT_CONFIG_WORKERS

    if test_series is not None:
        test_map = _normalize_series_collection(
            test_series, freq=freq, default_prefix="test"
        )
    elif dataset_bundle is not None:
        test_map, _ = _extract_test_series_from_dataset_bundle(
            dataset_bundle, freq=freq
        )
    else:
        raise ValueError("Debes pasar `test_series` o `dataset_bundle`")

    train_map = (
        _normalize_series_collection(train_series, freq=freq, default_prefix="train")
        if train_series is not None
        else {}
    )

    bundle_all_series_map: dict[str, pd.Series] = {}
    if dataset_bundle is not None:
        bundle_all_series = dataset_bundle.get("all_series_unscaled")
        if bundle_all_series is not None:
            bundle_all_series_map = _build_all_series_map(bundle_all_series, freq=freq)

    all_series_map = dict(bundle_all_series_map)

    # If train was not provided explicitly, derive from full/reference series.
    for series_name, ts_test in test_map.items():
        if series_name not in train_map:
            derived = _derive_train_series_from_full(
                series_name=series_name,
                test_series=ts_test,
                all_series_by_name=all_series_map,
            )
            if derived is not None:
                train_map[series_name] = _ensure_datetime_series(
                    derived, freq=freq, name=series_name
                )

    strategy = str(gap_strategy).strip().lower()
    if strategy not in {"block", "hybrid_tspulse"}:
        raise ValueError("`gap_strategy` debe ser 'block' o 'hybrid_tspulse'")

    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    predictions: dict[int, dict[str, dict[str, pd.Series]]] = {}
    plot_store: dict[int, dict[str, Any]] = {}
    failures_by_gap: dict[int, list[GapContextFailure]] = {}

    for gap_size in [int(g) for g in gap_sizes]:
        if gap_size <= 0:
            raise ValueError("Todos los `gap_sizes` deben ser > 0")

        gaps_per_series: dict[str, list[pd.DatetimeIndex]] = {}
        for series_name, ts_test in test_map.items():
            if gap_spec_by_series is not None and series_name in gap_spec_by_series:
                windows = [
                    _build_gap_index(start=s, length=l, freq=freq)
                    for s, l in gap_spec_by_series[series_name]
                    if int(l) > 0
                ]
            elif strategy == "hybrid_tspulse":
                windows = _generate_hybrid_tspulse_gaps(
                    series=ts_test,
                    gap_size=gap_size,
                    num_gaps=int(num_gaps),
                    rng=rng,
                    freq=freq,
                    random_fraction=float(hybrid_random_fraction),
                )
            else:
                windows = _generate_block_gaps(
                    series=ts_test,
                    gap_size=gap_size,
                    num_gaps=int(num_gaps),
                    rng=rng,
                    freq=freq,
                )

            # Keep only windows fully contained in current test index.
            test_index_set = set(ts_test.index.tolist())
            windows = [
                w
                for w in windows
                if set(pd.DatetimeIndex(w).tolist()).issubset(test_index_set)
            ]
            gaps_per_series[series_name] = windows

        predictions[gap_size] = {}
        plot_store[gap_size] = {"series": {}}
        failures_by_gap[gap_size] = []

        # Initialize plotting payload by series.
        for series_name, ts_test in test_map.items():
            _, mask_index = _mask_test_series(ts_test, gaps_per_series[series_name])
            reference_full = all_series_map.get(series_name)
            if reference_full is None:
                reference_full = pd.concat(
                    [train_map.get(series_name, pd.Series(dtype=float)), ts_test]
                ).sort_index()
                reference_full = reference_full[
                    ~reference_full.index.duplicated(keep="last")
                ]

            naive_mase = (
                reference_full.shift(int(seasonality_m)).reindex(mask_index)
                if len(reference_full) > 0
                else pd.Series(dtype=float)
            )

            plot_store[gap_size]["series"][series_name] = {
                "actual": ts_test,
                "preds": {},
                "naive_mase": naive_mase,
            }

        for model_name, model in model_dict.items():
            predictions[gap_size][model_name] = {}

            for series_name, ts_test in test_map.items():
                gap_windows = gaps_per_series[series_name]
                ts_train = train_map.get(series_name)

                ts_masked, mask_index = _mask_test_series(ts_test, gap_windows)
                pred_mask = pd.Series(index=mask_index, dtype=float, name=series_name)

                if len(mask_index) > 0:
                    if hasattr(model, "historical_imputation_forecasts"):
                        pred_mask = model.historical_imputation_forecasts(
                            series_name=series_name,
                            test_series=ts_test,
                            masked_test_series=ts_masked,
                            train_series=ts_train,
                            mask_index=mask_index,
                            gap_size=gap_size,
                            freq=freq,
                            config_workers=config_workers,
                        )
                        pred_mask = pred_mask.reindex(mask_index).astype(float)
                    elif hasattr(model, "predict"):
                        pred_mask, failures = darts_left_context_imputation_for_gaps(
                            model_name=model_name,
                            model=model,
                            series_name=series_name,
                            test_series=ts_test,
                            train_series=ts_train,
                            gap_windows=gap_windows,
                            freq=freq,
                        )
                        failures_by_gap[gap_size].extend(failures)
                    elif hasattr(model, "historical_forecasts"):
                        ts_for_legacy = TimeSeries.from_series(ts_test, freq=freq)
                        gap_starts = [
                            int(ts_test.index.get_loc(pd.Timestamp(window.min())))
                            for window in gap_windows
                            if len(window) > 0
                            and pd.Timestamp(window.min()) in ts_test.index
                        ]
                        pred_mask = darts_historical_forecasts_for_mask(
                            model=model,
                            ts_series_test=ts_for_legacy,
                            scaler=None,
                            gap_starts=gap_starts,
                            gap_size=gap_size,
                            mask_index=mask_index,
                            config_workers=config_workers,
                        )
                        pred_mask = pred_mask.reindex(mask_index).astype(float)
                    else:
                        raise TypeError(
                            f"Modelo '{model_name}' no expone una interfaz de imputacion soportada"
                        )

                predictions[gap_size][model_name][series_name] = pred_mask
                plot_store[gap_size]["series"][series_name]["preds"][model_name] = (
                    pred_mask
                )

                y_true = ts_test.reindex(mask_index)
                if len(mask_index) > 0:
                    first_gap_start = pd.Timestamp(mask_index.min())
                    insample_parts: list[pd.Series] = []
                    if ts_train is not None and len(ts_train) > 0:
                        insample_parts.append(ts_train)

                    pre_gap_test = ts_test.loc[ts_test.index < first_gap_start]
                    if len(pre_gap_test) > 0:
                        insample_parts.append(pre_gap_test)

                    if insample_parts:
                        insample = pd.concat(insample_parts).sort_index()
                        insample = insample[~insample.index.duplicated(keep="last")]
                    else:
                        insample = pd.Series(dtype=float)

                    if len(insample) == 0:
                        ref = all_series_map.get(series_name)
                        if ref is not None:
                            insample = ref.loc[ref.index < first_gap_start]
                else:
                    insample = (
                        ts_train.copy()
                        if ts_train is not None
                        else pd.Series(dtype=float)
                    )

                mase_denom = _compute_mase_denominator(
                    insample,
                    seasonality_m=seasonality_m,
                    freq=freq,
                )
                metric_values = _compute_metrics_on_mask(
                    y_true=y_true,
                    y_pred=pred_mask,
                    mase_denominator=mase_denom,
                    metrics=metric_list,
                )

                row: dict[str, Any] = {
                    "Modelo": str(model_name),
                    "Serie": str(series_name),
                    "Gap_Size": int(gap_size),
                }
                row.update(metric_values)
                rows.append(row)

        if failures_by_gap[gap_size]:
            unique_reasons = sorted({f.reason for f in failures_by_gap[gap_size]})
            print(
                f"[execute_complete_pipeline] Gap={gap_size}: "
                f"{len(failures_by_gap[gap_size])} gaps sin contexto minimo. "
                f"Motivos: {unique_reasons}"
            )

    results_df = pd.DataFrame(rows)
    metric_columns = [m.upper() for m in metric_list]
    ordered_cols = ["Modelo", "Serie", "Gap_Size", *metric_columns]
    for col in ordered_cols:
        if col not in results_df.columns:
            results_df[col] = float("nan")

    return results_df[ordered_cols], predictions, plot_store


def execute_complete_pipeline_imputation(
    *,
    model_dict: Mapping[str, Any],
    dataset_bundle: Mapping[str, Any],
    gap_sizes: Sequence[int] = (1, 2, 5, 10),
    num_gaps: int = 3,
    metrics: Sequence[str] = ("mae", "rmse", "mase"),
    random_seed: int = 42,
    freq: str = "h",
    seasonality_m: int = 24,
    gap_strategy: str = "block",
    config_workers: Mapping[str, Any] | None = None,
) -> tuple[
    pd.DataFrame, dict[int, dict[str, dict[str, pd.Series]]], dict[int, dict[str, Any]]
]:
    """Backward-compatible wrapper preserving previous public function name."""
    return execute_complete_pipeline(
        model_dict=model_dict,
        dataset_bundle=dataset_bundle,
        gap_sizes=gap_sizes,
        num_gaps=num_gaps,
        metrics=metrics,
        random_seed=random_seed,
        freq=freq,
        seasonality_m=seasonality_m,
        gap_strategy=gap_strategy,
        config_workers=config_workers,
    )


__all__ = [
    "GapContextFailure",
    "TSPulseHistoricalImputer",
    "build_tspulse_context_frame",
    "darts_historical_forecasts_for_mask",
    "darts_left_context_imputation_for_gaps",
    "execute_complete_pipeline",
    "execute_complete_pipeline_imputation",
    "infer_darts_minimum_context",
]
