"""Core gap generation, imputation, and scoring utilities for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field  # Structured diagnostics for skipped gaps.
from typing import Any, Mapping, Sequence  # Typing utilities for flexible public API.

import numpy as np  # Numeric operations for masks, metrics, and random sampling.
import pandas as pd  # Time-indexed series/dataframe processing.
from pandas.tseries.frequencies import (
    to_offset,
)  # Frequency-aware timestamp arithmetic.

from darts import TimeSeries  # Darts time series container used across the module.
from darts.metrics import mase as darts_mase
from airquality.data.io import to_pd_series
from airquality.data.series import ensure_datetime_series
from airquality.modeling.training_config import BenchmarkDatasetBundle


DEFAULT_CONFIG_WORKERS = {
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
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


@dataclass(slots=True)
class PlotSeriesPayload:
    """Internal benchmark plotting payload for one series."""

    actual: pd.Series
    preds: dict[str, pd.Series] = field(default_factory=dict)
    naive_mase: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass(slots=True)
class PlotGapPayload:
    """Internal benchmark plotting payload for one gap size."""

    series: dict[str, PlotSeriesPayload]
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_plot_series_payload(actual: pd.Series, naive_mase: pd.Series) -> PlotSeriesPayload:
    """Create the plot payload container used for one benchmark series."""
    return PlotSeriesPayload(
        actual=actual,
        preds={},
        naive_mase=naive_mase,
    )


def _serialize_plot_series_payload(payload: PlotSeriesPayload) -> dict[str, Any]:
    """Convert one plot-series payload dataclass into the plotting dict format."""
    return {
        "actual": payload.actual,
        "preds": dict(payload.preds),
        "naive_mase": payload.naive_mase,
    }


def _serialize_plot_gap_payload(payload: PlotGapPayload) -> dict[str, Any]:
    """Convert one gap payload dataclass into the public plot-store format."""
    serialized: dict[str, Any] = {
        "series": {
            series_name: _serialize_plot_series_payload(series_payload)
            for series_name, series_payload in payload.series.items()
        }
    }
    serialized.update(payload.metadata)
    return serialized


_ensure_datetime_series = ensure_datetime_series


def _ts_to_series(ts: TimeSeries, freq: str, name: str) -> pd.Series:
    """Convert Darts `TimeSeries` into normalized `pd.Series`."""
    out = to_pd_series(ts, freq=freq, name=name)
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
    dataset_bundle: BenchmarkDatasetBundle, *, freq: str
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    """Build unscaled test series from project `dataset_bundle`."""
    valid_cols = list(dataset_bundle.valid_cols)
    series_test = list(dataset_bundle.series_test)
    dict_scalers = dict(dataset_bundle.dict_scalers)

    if len(valid_cols) != len(series_test):
        raise ValueError("`valid_cols` y `series_test` deben tener la misma longitud")

    out_unscaled: dict[str, pd.Series] = {}
    for i, col in enumerate(valid_cols):
        ts = series_test[i]
        if not isinstance(ts, TimeSeries):
            raise TypeError("Cada elemento de `series_test` debe ser TimeSeries")

        scaler = dict_scalers.get(col)
        ts_unscaled = ts
        if scaler is not None and hasattr(scaler, "inverse_transform"):
            try:
                ts_unscaled = scaler.inverse_transform(ts)
            except Exception:
                ts_unscaled = ts
        out_unscaled[col] = _ts_to_series(ts_unscaled, freq=freq, name=col)

    return out_unscaled, dict_scalers


def _prepare_pipeline_series_maps(
    dataset_bundle: BenchmarkDatasetBundle,
    freq: str,
) -> tuple[
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, Any],
]:
    """Build benchmark series maps from one required dataset bundle."""
    test_map_unscaled, bundle_scalers = _extract_test_series_from_dataset_bundle(
        dataset_bundle, freq=freq
    )

    if not dataset_bundle.all_series_unscaled:
        raise ValueError(
            "`dataset_bundle.all_series_unscaled` es obligatorio para ejecutar el benchmark."
        )

    all_series_map = _normalize_series_collection(
        dataset_bundle.all_series_unscaled,
        freq=freq,
        default_prefix="all_series",
    )

    return (
        test_map_unscaled,
        all_series_map,
        bundle_scalers,
    )


def _compute_gap_mase(
    actual_gap: pd.Series,
    pred_gap: pd.Series,
    insample: pd.Series,
    seasonality_m: int,
    freq: str,
) -> float:
    """Compute MASE for a single gap using Darts mase function."""
    if len(actual_gap) == 0 or len(pred_gap) == 0 or len(insample) == 0:
        return float("nan")
    try:
        aligned_idx = actual_gap.index.intersection(pred_gap.index)
        if len(aligned_idx) == 0:
            return float("nan")

        actual_clean = actual_gap.reindex(aligned_idx).astype(float)
        pred_clean = pred_gap.reindex(aligned_idx).astype(float)
        valid = np.isfinite(actual_clean.to_numpy(dtype=float)) & np.isfinite(
            pred_clean.to_numpy(dtype=float)
        )
        if not np.any(valid):
            return float("nan")

        actual_clean = actual_clean.iloc[valid]
        pred_clean = pred_clean.iloc[valid]

        insample_clean = insample.copy()
        if insample_clean.isna().any():
            insample_clean = (
                insample_clean.interpolate(method="time", limit_direction="both")
                .ffill()
                .bfill()
            )
        if not np.isfinite(insample_clean.to_numpy(dtype=float)).any():
            return float("nan")

        actual_ts = TimeSeries.from_series(actual_clean, freq=freq)
        pred_ts = TimeSeries.from_series(pred_clean, freq=freq)
        insample_ts = TimeSeries.from_series(insample_clean, freq=freq)
        val = darts_mase(
            actual_series=actual_ts,
            pred_series=pred_ts,
            insample=insample_ts,
            m=int(seasonality_m),
            intersect=True,
        )
        if isinstance(val, (list, np.ndarray)):
            val = np.nanmean(val)
        val = float(val)
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


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
    min_gap_points: int = 0,
) -> list[int]:
    """Sample non-overlapping block starts over `[0, n_points-gap_size]`."""
    if n_points < gap_size or num_gaps <= 0:
        return []

    separation = max(0, int(min_gap_points))
    candidates = np.arange(0, n_points - gap_size + 1, dtype=int)
    rng.shuffle(candidates)

    starts: list[int] = []
    for start in candidates:
        overlap = any(
            not (
                start + gap_size + separation <= s
                or s + gap_size + separation <= start
            )
            for s in starts
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
    min_gap_points: int = 0,
) -> list[pd.DatetimeIndex]:
    """Generate fixed-size non-overlapping artificial block gaps."""
    starts = _sample_non_overlapping_starts(
        n_points=len(series),
        gap_size=int(gap_size),
        num_gaps=int(num_gaps),
        rng=rng,
        min_gap_points=min_gap_points,
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
    Random points are kept isolated, and block windows are separated by at least
    one clean timestamp so they cannot merge into larger contiguous gaps.
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
        min_gap_points=1,
    )

    used_points: set[pd.Timestamp] = set()
    for window in block_windows:
        used_points.update(window.tolist())

    offset = to_offset(freq)
    available_points = list(series.index)
    rng.shuffle(available_points)

    random_points: list[pd.Timestamp] = []
    for ts in available_points:
        point = pd.Timestamp(ts)
        if point in used_points:
            continue
        if (point - offset) in used_points or (point + offset) in used_points:
            continue

        random_points.append(point)
        used_points.add(point)
        if len(random_points) >= random_missing:
            break

    random_points = sorted(random_points)
    point_windows = [_build_gap_index(ts, 1, freq=freq) for ts in random_points]

    return sorted(block_windows + point_windows, key=lambda idx: pd.Timestamp(idx[0]))


def _build_gap_windows_for_series(
    *,
    series_name: str,
    ts_test: pd.Series,
    gap_size: int,
    num_gaps: int,
    strategy: str,
    rng: np.random.Generator,
    freq: str,
    hybrid_random_fraction: float,
    gap_spec_by_series: Mapping[str, Sequence[tuple[pd.Timestamp, int]]] | None,
) -> list[pd.DatetimeIndex]:
    """Choose the synthetic gap windows to evaluate for one test series."""
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

    test_index_set = set(ts_test.index.tolist())
    return [
        w
        for w in windows
        if set(pd.DatetimeIndex(w).tolist()).issubset(test_index_set)
    ]


def _plan_gaps_for_all_series(
    *,
    test_map_unscaled: Mapping[str, pd.Series],
    gap_size: int,
    num_gaps: int,
    strategy: str,
    rng: np.random.Generator,
    freq: str,
    hybrid_random_fraction: float,
    gap_spec_by_series: Mapping[str, Sequence[tuple[pd.Timestamp, int]]] | None,
) -> dict[str, list[pd.DatetimeIndex]]:
    """Plan synthetic gaps for every test series in the benchmark run."""
    return {
        series_name: _build_gap_windows_for_series(
            series_name=series_name,
            ts_test=ts_test,
            gap_size=gap_size,
            num_gaps=num_gaps,
            strategy=strategy,
            rng=rng,
            freq=freq,
            hybrid_random_fraction=hybrid_random_fraction,
            gap_spec_by_series=gap_spec_by_series,
        )
        for series_name, ts_test in test_map_unscaled.items()
    }


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
    metrics: Sequence[str],
) -> dict[str, float]:
    """Compute selected metrics (MAE, RMSE) only on mask timestamps."""
    idx = y_true.index.intersection(y_pred.index)
    true_vals = y_true.reindex(idx).to_numpy(dtype=float)
    pred_vals = y_pred.reindex(idx).to_numpy(dtype=float)

    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if not np.any(valid):
        return {m.upper(): float("nan") for m in metrics if m.lower() in ("mae", "rmse")}

    err = true_vals[valid] - pred_vals[valid]
    out: dict[str, float] = {}

    for metric_name in metrics:
        m = metric_name.lower()
        if m == "mae":
            out["MAE"] = float(np.mean(np.abs(err)))
        elif m == "rmse":
            out["RMSE"] = float(np.sqrt(np.mean(np.square(err))))
    return out


def _build_plot_store_series_payloads(
    test_map_unscaled: Mapping[str, pd.Series],
    gaps_per_series: Mapping[str, Sequence[pd.DatetimeIndex]],
    all_series_map: Mapping[str, pd.Series],
    seasonality_m: int,
) -> tuple[dict[str, PlotSeriesPayload], dict[str, pd.DatetimeIndex]]:
    """Prepare plotting payloads and mask indexes before model inference."""
    plot_series_payload: dict[str, PlotSeriesPayload] = {}
    mask_index_by_series: dict[str, pd.DatetimeIndex] = {}

    for series_name, ts_test in test_map_unscaled.items():
        _, mask_index = _mask_test_series(ts_test, gaps_per_series[series_name])
        mask_index_by_series[series_name] = mask_index
        reference_full = all_series_map[series_name]

        naive_mase = (
            reference_full.shift(int(seasonality_m)).reindex(mask_index)
            if len(reference_full) > 0
            else pd.Series(dtype=float)
        )

        plot_series_payload[series_name] = _make_plot_series_payload(
            actual=ts_test,
            naive_mase=naive_mase,
        )

    return plot_series_payload, mask_index_by_series


def _predict_mask_for_model_series(
    model: Any,
    series_name: str,
    test_index: pd.DatetimeIndex,
    all_series_map: Mapping[str, pd.Series],
    gap_windows: Sequence[pd.DatetimeIndex],
    scaler: Any | None,
    freq: str,
    config_workers: Mapping[str, Any],
) -> tuple[pd.Series, list[GapContextFailure]]:
    """Impute one series with one `GapImputer` over the pooled mask timestamps.

    Every model exposes the same `impute_gaps` contract and returns predictions in
    the original scale; scaling/inverse-scaling is internal to each imputer.
    """
    mask_index = _gap_windows_to_mask_index(gap_windows)
    if len(mask_index) == 0:
        return pd.Series(index=mask_index, dtype=float, name=series_name), []

    pred_mask, failures = model.impute_gaps(
        series_name=series_name,
        all_series_map=all_series_map,
        gap_windows=gap_windows,
        test_index=test_index,
        scaler=scaler,
        freq=freq,
        config_workers=config_workers,
    )
    return pred_mask.reindex(mask_index).astype(float), failures


def _build_metric_row(
    model_name: str,
    series_name: str,
    gap_size: int,
    pred_mask: pd.Series,
    ts_test_unscaled: pd.Series,
    all_series_map: Mapping[str, pd.Series],
    gap_windows: Sequence[pd.DatetimeIndex],
    metric_list: Sequence[str],
    seasonality_m: int,
    freq: str,
) -> dict[str, Any]:
    """Build one benchmark result row for a model, series, and gap size."""
    # Compute MAE/RMSE on the pooled mask timestamps
    mae_rmse_metrics = [m for m in metric_list if m in ("mae", "rmse")]
    row: dict[str, Any] = {
        "Modelo": str(model_name),
        "Serie": str(series_name),
        "Gap_Size": int(gap_size),
    }

    if mae_rmse_metrics:
        # Get pooled true values and predictions on mask index
        mask_index = _gap_windows_to_mask_index(gap_windows)
        y_true = ts_test_unscaled.reindex(mask_index)
        y_pred = pred_mask.reindex(mask_index)
        mae_rmse_values = _compute_metrics_on_mask(
            y_true=y_true,
            y_pred=y_pred,
            metrics=mae_rmse_metrics,
        )
        row.update(mae_rmse_values)

    if "mase" in metric_list:
        gap_mases: list[float] = []
        gap_lengths: list[int] = []

        full_series = all_series_map[series_name]

        for gap_idx in gap_windows:
            if len(gap_idx) == 0:
                continue

            gap_start = pd.Timestamp(gap_idx.min())

            actual_gap = ts_test_unscaled.reindex(gap_idx)
            pred_gap = pred_mask.reindex(gap_idx)

            insample = full_series.loc[full_series.index < gap_start].copy()

            gap_mase = _compute_gap_mase(
                actual_gap=actual_gap,
                pred_gap=pred_gap,
                insample=insample,
                seasonality_m=seasonality_m,
                freq=freq,
            )

            if np.isfinite(gap_mase):
                gap_mases.append(gap_mase)
                gap_lengths.append(len(gap_idx))

        if len(gap_mases) > 0:
            total_len = sum(gap_lengths)
            weighted_sum = sum(m * l for m, l in zip(gap_mases, gap_lengths))
            final_mase = weighted_sum / total_len
        else:
            final_mase = float("nan")

        row["MASE"] = final_mase

    return row


def _execute_gap_size_pipeline(
    gap_size: int,
    model_dict: Mapping[str, Any],
    test_map_unscaled: Mapping[str, pd.Series],
    all_series_map: Mapping[str, pd.Series],
    bundle_scalers: Mapping[str, Any],
    strategy: str,
    num_gaps: int,
    rng: np.random.Generator,
    freq: str,
    hybrid_random_fraction: float,
    gap_spec_by_series: Mapping[str, Sequence[tuple[pd.Timestamp, int]]] | None,
    seasonality_m: int,
    metric_list: Sequence[str],
    config_workers: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, PlotSeriesPayload], list[GapContextFailure]]:
    """Execute gap generation, inference, and scoring for one gap size."""
    gaps_per_series = _plan_gaps_for_all_series(
        test_map_unscaled=test_map_unscaled,
        gap_size=gap_size,
        num_gaps=int(num_gaps),
        strategy=strategy,
        rng=rng,
        freq=freq,
        hybrid_random_fraction=float(hybrid_random_fraction),
        gap_spec_by_series=gap_spec_by_series,
    )

    plot_series_payload, mask_index_by_series = _build_plot_store_series_payloads(
        test_map_unscaled=test_map_unscaled,
        gaps_per_series=gaps_per_series,
        all_series_map=all_series_map,
        seasonality_m=seasonality_m,
    )

    failures: list[GapContextFailure] = []
    rows: list[dict[str, Any]] = []
    for model_name, model in model_dict.items():
        # Keep failure diagnostics labelled with the registry name.
        if getattr(model, "model_name", "") != model_name:
            try:
                model.model_name = model_name
            except AttributeError:
                pass

        for series_name, ts_test_unscaled in test_map_unscaled.items():
            gap_windows = gaps_per_series[series_name]

            pred_mask, series_failures = _predict_mask_for_model_series(
                model=model,
                series_name=series_name,
                test_index=ts_test_unscaled.index,
                all_series_map=all_series_map,
                gap_windows=gap_windows,
                scaler=bundle_scalers.get(series_name),
                freq=freq,
                config_workers=config_workers,
            )
            failures.extend(series_failures)

            plot_series_payload[series_name].preds[model_name] = pred_mask
            rows.append(
                _build_metric_row(
                    model_name=model_name,
                    series_name=series_name,
                    gap_size=gap_size,
                    pred_mask=pred_mask,
                    ts_test_unscaled=ts_test_unscaled,
                    all_series_map=all_series_map,
                    gap_windows=gap_windows,
                    metric_list=metric_list,
                    seasonality_m=seasonality_m,
                    freq=freq,
                )
            )

    return rows, plot_series_payload, failures


def execute_complete_pipeline(
    model_dict: Mapping[str, Any],
    dataset_bundle: BenchmarkDatasetBundle,
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
    pd.DataFrame, dict[int, dict[str, Any]]
]:
    """Execute imputation benchmark across TSPulse and Darts models.

    Main responsibilities:
    - Receive one dataset bundle and already-loaded models.
    - Generate (or receive) artificial gaps compatible with TSPulse notebook ideas.
    - Impute with TSPulse and Darts.
    - Evaluate MAE, RMSE, and MASE strictly on missing points.
    - Scale-sensitive models (those without `requires_unscaled_input`) predict on
      scaled values and their output is inverse-transformed before scoring; models
      that require unscaled input consume the original scale directly.
    - `dataset_bundle.all_series_unscaled` is required and used as the source of
      pre-test history for context/MASE.
    - Return predictions, metrics, and plotting payload.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        `(results_df, plot_store)` where:
        - `results_df` has columns `Modelo, Serie, Gap_Size, [metricas...]`.
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

    (
        test_map_unscaled,
        all_series_map,
        bundle_scalers,
    ) = _prepare_pipeline_series_maps(dataset_bundle, freq)

    strategy = str(gap_strategy).strip().lower()
    if strategy not in {"block", "hybrid_tspulse"}:
        raise ValueError("`gap_strategy` debe ser 'block' o 'hybrid_tspulse'")

    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    plot_store: dict[int, dict[str, Any]] = {}
    failures_by_gap: dict[int, list[GapContextFailure]] = {}

    for gap_size in [int(g) for g in gap_sizes]:
        if gap_size <= 0:
            raise ValueError("Todos los `gap_sizes` deben ser > 0")

        gap_rows, plot_series_payload, gap_failures = _execute_gap_size_pipeline(
            gap_size=gap_size,
            model_dict=model_dict,
            test_map_unscaled=test_map_unscaled,
            all_series_map=all_series_map,
            bundle_scalers=bundle_scalers,
            strategy=strategy,
            num_gaps=int(num_gaps),
            rng=rng,
            freq=freq,
            hybrid_random_fraction=float(hybrid_random_fraction),
            gap_spec_by_series=gap_spec_by_series,
            seasonality_m=seasonality_m,
            metric_list=metric_list,
            config_workers=config_workers,
        )
        rows.extend(gap_rows)
        plot_store[gap_size] = _serialize_plot_gap_payload(
            PlotGapPayload(series=plot_series_payload)
        )
        failures_by_gap[gap_size] = gap_failures

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

    return results_df[ordered_cols], plot_store


__all__ = [
    "GapContextFailure",
    "execute_complete_pipeline",
]
