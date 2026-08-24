"""Core gap generation, imputation, and scoring utilities for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field  # Structured diagnostics for skipped gaps.
from typing import Any, Mapping, Sequence  # Typing utilities for flexible public API.

import numpy as np  # Numeric operations for masks, metrics, and random sampling.
import pandas as pd  # Time-indexed series/dataframe processing.

from darts import TimeSeries  # Darts time series container used across the module.
from airquality.data.series import ensure_datetime_series, to_pd_series
from airquality.modeling.training_config import BenchmarkDatasetBundle
from airquality.metrics import compute_mase


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
    return serialized


def _ts_to_series(ts: TimeSeries, freq: str, name: str) -> pd.Series:
    """Convert Darts `TimeSeries` into normalized `pd.Series`."""
    return to_pd_series(ts, freq=freq, name=name)


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
        out[name] = ensure_datetime_series(series_like, freq=freq, name=name)
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
            out[str(col)] = ensure_datetime_series(
                series_like[col], freq=freq, name=str(col)
            )
        return out

    if isinstance(series_like, Mapping):
        for key, value in series_like.items():
            name = str(key)
            if isinstance(value, pd.Series):
                s = value.copy()
                s.name = name
                out[name] = ensure_datetime_series(s, freq=freq, name=name)
            elif isinstance(value, TimeSeries):
                out[name] = _ts_to_series(value, freq=freq, name=name)
            elif isinstance(value, pd.DataFrame):
                if len(value.columns) != 1:
                    raise ValueError(
                        "Cada DataFrame del mapeo debe tener exactamente una columna"
                    )
                s = value.iloc[:, 0].copy()
                s.name = name
                out[name] = ensure_datetime_series(s, freq=freq, name=name)
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
                out[str(s.name)] = ensure_datetime_series(
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
                out[str(s.name)] = ensure_datetime_series(
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
            except Exception as exc:
                # Continuing with the scaled series as if it were unscaled would
                # silently corrupt every metric of this column.
                raise RuntimeError(
                    f"Fallo al des-escalar la serie de test de '{col}'"
                ) from exc
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
        min_gap_points=max(1, int(min_gap_points)),
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
    block_starts = _sample_non_overlapping_starts(
        n_points=len(series),
        gap_size=int(gap_size),
        num_gaps=int(block_count),
        rng=rng,
        min_gap_points=1,
    )
    block_windows = [_build_gap_index(series.index[s], gap_size, freq=freq) for s in block_starts]

    # Work on integer positions instead of Timestamp sets: on the regular test
    # grid `position +- 1` is exactly `timestamp +- freq`, and boolean-array
    # lookups avoid hashing millions of Timestamp objects. The shuffle draws
    # the same RNG stream as the previous list-of-timestamps shuffle (it only
    # depends on the sequence length), so the sampled gaps are unchanged.
    n_points = len(series)
    used = np.zeros(n_points, dtype=bool)
    for start in block_starts:
        used[start : start + int(gap_size)] = True

    candidate_positions = list(range(n_points))
    rng.shuffle(candidate_positions)

    random_positions: list[int] = []
    for pos in candidate_positions:
        if used[pos]:
            continue
        if (pos > 0 and used[pos - 1]) or (pos + 1 < n_points and used[pos + 1]):
            continue

        random_positions.append(pos)
        used[pos] = True
        if len(random_positions) >= random_missing:
            break

    random_positions.sort()
    point_windows = [
        _build_gap_index(series.index[pos], 1, freq=freq) for pos in random_positions
    ]

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

    # `get_indexer` reuses the index's cached hash engine across windows, so
    # membership is vectorized instead of building Timestamp sets per window.
    test_index = pd.DatetimeIndex(ts_test.index)
    return [
        w
        for w in windows
        if len(w) == 0 or (test_index.get_indexer(pd.DatetimeIndex(w)) >= 0).all()
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


def _compute_metrics_on_mask(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    metrics: Sequence[str],
    scale_std: float | None = None,
) -> dict[str, float]:
    """Compute selected MAE/RMSE on mask timestamps, optionally standardized."""
    idx = y_true.index.intersection(y_pred.index)
    true_vals = y_true.reindex(idx).to_numpy(dtype=float)
    pred_vals = y_pred.reindex(idx).to_numpy(dtype=float)

    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    if not np.any(valid):
        return {m.upper(): float("nan") for m in metrics if m.lower() in ("mae", "rmse")}

    err = true_vals[valid] - pred_vals[valid]
    if scale_std is not None:
        if not np.isfinite(scale_std) or scale_std <= 0.0:
            return {
                m.upper(): float("nan")
                for m in metrics
                if m.lower() in ("mae", "rmse")
            }
        err = err / float(scale_std)
    out: dict[str, float] = {}

    for metric_name in metrics:
        m = metric_name.lower()
        if m == "mae":
            out["MAE"] = float(np.mean(np.abs(err)))
        elif m == "rmse":
            out["RMSE"] = float(np.sqrt(np.mean(np.square(err))))
    return out


def _scaler_standard_deviation(scaler: Any | None) -> float:
    """Recover the train-only standard deviation represented by a Darts scaler."""
    if scaler is None or not hasattr(scaler, "transform"):
        return float("nan")

    probe = TimeSeries.from_values(np.asarray([[0.0], [1.0]], dtype=np.float32))
    try:
        transformed = scaler.transform(probe).values(copy=False).reshape(-1)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo obtener la escala de entrenamiento para MAE/RMSE"
        ) from exc

    if len(transformed) != 2:
        return float("nan")
    slope = float(transformed[1] - transformed[0])
    if not np.isfinite(slope) or slope == 0.0:
        return float("nan")
    return abs(1.0 / slope)


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
        mask_index = _gap_windows_to_mask_index(gaps_per_series[series_name])
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
) -> tuple[pd.Series, list[GapContextFailure], dict[str, float]]:
    """Impute one series with one `GapImputer` over the pooled mask timestamps.

    Every model exposes the same `impute_gaps` contract and returns predictions in
    the original scale; scaling/inverse-scaling is internal to each imputer.

    Each imputer runs its own two timers during the call — ``_last_train_seconds``
    (fitting, non-zero only for local fitters like Prophet) and
    ``_last_impute_seconds`` (prediction/fill) — so we just read them here (no
    external timing, no subtraction). Returns a ``timing`` dict with **per-hole
    means** for this series/gap size (never sums over the holes):

    - ``impute_seconds``: prediction time / number of holes.
    - ``train_seconds``: for local fitters, amortized fit time / number of holes; else
      the model's one-time train cost (TSPulse load, 0 for interpolation, NaN for
      pretrained Darts whose training time is recorded by `train_global_methods`).

    Empty masks report NaN timings.
    """
    mask_index = _gap_windows_to_mask_index(gap_windows)
    nan_timing = {"impute_seconds": float("nan"), "train_seconds": float("nan")}
    if len(mask_index) == 0:
        return pd.Series(index=mask_index, dtype=float, name=series_name), [], nan_timing

    pred_mask, failures = model.impute_gaps(
        series_name=series_name,
        all_series_map=all_series_map,
        gap_windows=gap_windows,
        test_index=test_index,
        scaler=scaler,
        freq=freq,
        config_workers=config_workers,
    )

    n_holes = sum(1 for gap_idx in gap_windows if len(gap_idx) > 0)
    train_total = float(getattr(model, "_last_train_seconds", 0.0) or 0.0)
    impute_total = float(getattr(model, "_last_impute_seconds", 0.0) or 0.0)

    impute_mean = impute_total / n_holes if n_holes else float("nan")
    if train_total > 0.0:
        train_row = train_total / n_holes  # amortized per-hole fit mean (Prophet)
    else:
        # No per-gap fit: TSPulse reports its one-time load, interpolation 0,
        # Darts leaves it absent -> NaN (its training time is in the train CSV).
        train_row = float(getattr(model, "train_seconds", float("nan")))

    timing = {"impute_seconds": impute_mean, "train_seconds": train_row}
    return pred_mask.reindex(mask_index).astype(float), failures, timing


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
    metric_scaler: Any | None = None,
    train_seconds: float = float("nan"),
    impute_seconds: float = float("nan"),
) -> dict[str, Any]:
    """Build one benchmark result row for a model, series, and gap size.

    ``train_seconds`` / ``impute_seconds`` are the per-hole mean timings for this
    (model, series, gap size); the graphs average them across series.
    """
    mask_index = _gap_windows_to_mask_index(gap_windows)
    y_true = ts_test_unscaled.reindex(mask_index)
    y_pred = pred_mask.reindex(mask_index)
    true_values = y_true.to_numpy(dtype=float)
    pred_values = y_pred.to_numpy(dtype=float)
    valid_pairs = np.isfinite(true_values) & np.isfinite(pred_values)
    complete_gaps = 0
    for gap_idx in gap_windows:
        if len(gap_idx) == 0:
            continue
        gap_true = ts_test_unscaled.reindex(gap_idx).to_numpy(dtype=float)
        gap_pred = pred_mask.reindex(gap_idx).to_numpy(dtype=float)
        if len(gap_true) == len(gap_idx) and np.all(
            np.isfinite(gap_true) & np.isfinite(gap_pred)
        ):
            complete_gaps += 1

    n_target_points = int(len(mask_index))
    n_scored_points = int(valid_pairs.sum())
    scale_std = _scaler_standard_deviation(metric_scaler)

    # MAE and RMSE use the station's train-only standard scale. MASE retains
    # its seasonal-naive scale and remains directly comparable to prior runs.
    mae_rmse_metrics = [m for m in metric_list if m in ("mae", "rmse")]
    row: dict[str, Any] = {
        "Modelo": str(model_name),
        "Serie": str(series_name),
        "Gap_Size": int(gap_size),
        "Train_Seconds": float(train_seconds),
        "Impute_Seconds": float(impute_seconds),
        "Scale_Std": scale_std,
        "N_Gaps_Target": int(sum(len(gap) > 0 for gap in gap_windows)),
        "N_Gaps_Scored": int(complete_gaps),
        "N_Target_Points": n_target_points,
        "N_Scored_Points": n_scored_points,
        "Support_Fraction": (
            float(n_scored_points / n_target_points)
            if n_target_points > 0
            else float("nan")
        ),
    }

    if mae_rmse_metrics:
        mae_rmse_values = _compute_metrics_on_mask(
            y_true=y_true,
            y_pred=y_pred,
            metrics=mae_rmse_metrics,
            scale_std=scale_std,
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

            gap_mase = compute_mase(
                actual=actual_gap,
                pred=pred_gap,
                insample=insample,
                seasonality_m=seasonality_m,
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

            pred_mask, series_failures, timing = _predict_mask_for_model_series(
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
                    metric_scaler=bundle_scalers.get(series_name),
                    train_seconds=timing["train_seconds"],
                    impute_seconds=timing["impute_seconds"],
                )
            )

    return rows, plot_series_payload, failures


def execute_complete_pipeline(
    model_dict: Mapping[str, Any],
    dataset_bundle: BenchmarkDatasetBundle,
    gap_sizes: Sequence[int] = (1, 2, 5, 10),
    num_gaps: int = 3,
    gap_counts: Sequence[int] | None = None,
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
    - Evaluate train-standardized MAE/RMSE and MASE strictly on missing points.
    - Scale-sensitive models (those without `requires_unscaled_input`) predict on
      scaled values and their output is inverse-transformed before applying the
      common train-only station scale; models that require unscaled input consume
      the original scale directly.
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

    normalized_gap_sizes = [int(g) for g in gap_sizes]
    if gap_counts is None:
        normalized_gap_counts = [int(num_gaps)] * len(normalized_gap_sizes)
    else:
        normalized_gap_counts = [int(count) for count in gap_counts]
        if len(normalized_gap_counts) != len(normalized_gap_sizes):
            raise ValueError("`gap_counts` debe tener un valor por cada `gap_sizes`.")
        if any(count <= 0 for count in normalized_gap_counts):
            raise ValueError("Todos los `gap_counts` deben ser > 0.")

    for gap_size, gap_count in zip(
        normalized_gap_sizes, normalized_gap_counts, strict=True
    ):
        if gap_size <= 0:
            raise ValueError("Todos los `gap_sizes` deben ser > 0")

        gap_rows, plot_series_payload, gap_failures = _execute_gap_size_pipeline(
            gap_size=gap_size,
            model_dict=model_dict,
            test_map_unscaled=test_map_unscaled,
            all_series_map=all_series_map,
            bundle_scalers=bundle_scalers,
            strategy=strategy,
            num_gaps=gap_count,
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

    holdout_metadata = dataset_bundle.holdout_metadata
    if isinstance(holdout_metadata, pd.DataFrame) and not holdout_metadata.empty:
        metadata = holdout_metadata.drop_duplicates("Serie").set_index("Serie")
        for column in (
            "Test_Start",
            "Test_End",
            "Test_Block_Points",
            "Train_Points_Before",
            "Train_Points_After",
        ):
            if column in metadata.columns:
                results_df[column] = results_df["Serie"].map(metadata[column])

    # Timing is per row (per-hole means built in `_predict_mask_for_model_series`):
    # `Impute_Seconds` for every model, `Train_Seconds` for the models trained at
    # benchmark time (Prophet's per-hole fit, TSPulse's one-time load, 0 for
    # interpolation). Pretrained Darts leave `Train_Seconds` NaN — their training
    # time comes from `train_global_methods`' CSV, merged in only at plot time.
    metric_columns = [m.upper() for m in metric_list]
    ordered_cols = [
        "Modelo",
        "Serie",
        "Gap_Size",
        "Train_Seconds",
        "Impute_Seconds",
        "Scale_Std",
        "N_Gaps_Target",
        "N_Gaps_Scored",
        "N_Target_Points",
        "N_Scored_Points",
        "Support_Fraction",
        "Test_Start",
        "Test_End",
        "Test_Block_Points",
        "Train_Points_Before",
        "Train_Points_After",
        *metric_columns,
    ]
    for col in ordered_cols:
        if col not in results_df.columns:
            results_df[col] = float("nan")

    return results_df[ordered_cols], plot_store


__all__ = [
    "GapContextFailure",
    "execute_complete_pipeline",
]
