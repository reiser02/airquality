"""Unified imputation interface (`GapImputer`) and per-origin adapters.

Every imputation model (Darts global forecasters, Darts-backed Prophet, TSPulse)
is exposed behind a single :class:`GapImputer` protocol so the benchmark pipeline
can call them uniformly. Each adapter owns its own scaling contract and always
returns predictions in the **original** (unscaled) scale, indexed by the pooled
mask index.

Scaling rule (no flags): whoever scales, inverse-scales.
- :class:`DartsGlobalGapImputer` predicts on scaled context **iff** a ``scaler`` is
  provided and inverse-transforms its output.
- :class:`ProphetGapImputer` and :class:`TSPulseGapImputer` consume the original
  scale and ignore the external ``scaler`` (TSPulse standardizes internally).
"""

from __future__ import annotations

import os  # Read optional Hugging Face token from environment variables.
import warnings  # Suppress optional runtime warnings during model prediction.
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from darts import TimeSeries
from airquality.data.io import resolve_device
from airquality.data.series import ensure_datetime_series
from airquality.imputation.benchmark import (
    GapContextFailure,
    _gap_windows_to_mask_index,
    _ts_to_series,
)


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


try:
    from darts.models import Prophet as DartsProphet  # Local forecasting model.

    PROPHET_AVAILABLE = True
    PROPHET_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency path.
    DartsProphet = None
    PROPHET_AVAILABLE = False
    PROPHET_IMPORT_ERROR = exc


_INSUFFICIENT_CONTEXT_REASON = (
    "No se alcanza el contexto minimo del modelo incluso "
    "completando con historial; se deja NaN en ese gap."
)


@runtime_checkable
class GapImputer(Protocol):
    """Single imputation contract shared by every model origin.

    Implementations return predictions in the **original** scale, indexed by the
    pooled mask index of ``gap_windows``, together with per-gap diagnostics.
    """

    model_name: str

    def impute_gaps(
        self,
        *,
        series_name: str,
        all_series_map: Mapping[str, pd.Series],
        gap_windows: Sequence[pd.DatetimeIndex],
        test_index: pd.DatetimeIndex,
        scaler: Any | None,
        freq: str,
        config_workers: Mapping[str, Any] | None = None,
    ) -> tuple[pd.Series, list[GapContextFailure]]:
        """Impute ``gap_windows`` of one series; return ``(predictions, failures)``.

        Predictions are indexed by the pooled mask index and expressed in the
        original scale; unfillable gaps are reported as
        :class:`GapContextFailure` entries instead of raising.
        """
        ...


# --------------------------------------------------------------------------- #
# Shared context / scaling helpers (moved verbatim from benchmark.py)
# --------------------------------------------------------------------------- #
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
    series: pd.Series,
    gap_start: pd.Timestamp,
    required_context: int,
    freq: str,
) -> pd.Series:
    """Build contiguous, NaN-free left context ending exactly before one gap.

    The series passed should contain the history before the gap.
    """
    offset = to_offset(freq)
    cutoff = pd.Timestamp(gap_start) - offset

    history = series.loc[series.index <= cutoff].copy()
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
        name=series.name,
    )


def _derive_context_before_gap(
    series_name: str,
    gap_start: pd.Timestamp,
    all_series_map: Mapping[str, pd.Series],
    scaler: Any | None,
    freq: str,
    required_context: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Derive unscaled and scaled context before the gap starting at gap_start.

    When ``required_context`` is given, the history is first trimmed to the
    time window that can possibly feed a clean left context of that many grid
    steps (``_build_clean_left_context`` never reads past the first NaN slot,
    so anything earlier can only be dead weight). This avoids copying and
    scaling the whole history once per gap; scaling is elementwise, so the
    resulting context values are identical.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (unscaled_context, scaled_context)

    Raises
    ------
    RuntimeError
        If the scaler fails on a non-empty context. Falling back to the
        unscaled values would silently mix scales and corrupt every metric
        downstream, which is worse than failing loudly.
    """
    full = all_series_map[series_name]
    unscaled_context = full.loc[full.index < gap_start]

    if required_context is not None and len(unscaled_context) > 0:
        # Clamp by the number of available points so the offset multiplication
        # cannot overflow the Timestamp range (Prophet requests ~1e9 points).
        span = min(int(required_context), len(unscaled_context))
        earliest_needed = pd.Timestamp(gap_start) - span * to_offset(freq)
        if unscaled_context.index[0] < earliest_needed:
            unscaled_context = unscaled_context.loc[unscaled_context.index >= earliest_needed]

    unscaled_context = unscaled_context.copy()
    if scaler is None or not hasattr(scaler, "transform") or len(unscaled_context) == 0:
        # An empty context is legitimate (gap at the series start): the caller
        # records it as a GapContextFailure after the min-context check.
        return unscaled_context, unscaled_context.copy()

    try:
        ts_scaled = scaler.transform(TimeSeries.from_series(unscaled_context, freq=freq))
        scaled_context = _ts_to_series(ts_scaled, freq=freq, name=series_name).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Fallo al escalar el contexto de '{series_name}' antes del hueco en {gap_start}"
        ) from exc
    return unscaled_context, scaled_context


def _inverse_scale_prediction_series(
    pred_series: pd.Series,
    *,
    scaler: Any | None,
    freq: str,
    name: str,
) -> pd.Series:
    """Inverse-transform one prediction series, preserving sparse mask index.

    Raises
    ------
    RuntimeError
        If the inverse transform fails. Returning the scaled prediction as if
        it were in the original scale would corrupt the metrics silently.
    """
    out = pred_series.copy().astype(float)
    out.name = name

    if len(out) == 0 or scaler is None or not hasattr(scaler, "inverse_transform"):
        return out

    try:
        ts_scaled = TimeSeries.from_series(out, freq=freq)
        ts_unscaled = scaler.inverse_transform(ts_scaled)
        inv = ts_unscaled.to_series().astype(float).reindex(out.index)
        inv.name = name
        return inv
    except Exception as exc:
        raise RuntimeError(
            f"Fallo al des-escalar las predicciones de '{name}'"
        ) from exc


def build_tspulse_context_frame(
    series_name: str,
    all_series_map: Mapping[str, pd.Series],
    mask_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    context_length: int,
    freq: str,
    timestamp_column: str = "timestamp",
    target_column: str = "value",
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Build TSPulse-ready context frame from all_series_map + mask.

    The synthetic-gap timestamps (``mask_index``) are kept as NaN: the official
    `TimeSeriesImputationPipeline` only substitutes the model reconstruction
    where the input frame has NaN, so pre-filling them would silence the model
    (the benchmark would score the pre-fill instead of TSPulse). Only NaN
    outside the mask (real historical holes / padding) are pre-filled with time
    interpolation.
    """
    full = all_series_map[series_name].copy()
    if len(mask_index) > 0:
        full.loc[full.index.intersection(mask_index)] = np.nan

    end_ts = pd.Timestamp(test_index.max())
    context_index = pd.date_range(end=end_ts, periods=int(context_length), freq=freq)
    context_values = full.reindex(context_index)

    # Fill only the real (non-mask) missing points; the mask must stay NaN so
    # the official pipeline imputes it with the model reconstruction.
    filled = (
        context_values.interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    in_mask = context_index.isin(mask_index)
    if filled.loc[~in_mask].isna().any():
        raise ValueError(
            "No fue posible construir un contexto valido para TSPulse tras completar con historial y padding"
        )
    context_values = filled
    context_values[in_mask] = np.nan

    frame = pd.DataFrame(
        {
            timestamp_column: context_index,
            target_column: context_values.to_numpy(dtype=float),
        }
    )
    return frame, pd.DatetimeIndex(test_index)


# --------------------------------------------------------------------------- #
# Darts adapters (clean left-context imputation)
# --------------------------------------------------------------------------- #
class _DartsContextImputer:
    """Shared per-gap clean-left-context imputation loop for Darts-style models.

    Subclasses provide the per-block prediction via :meth:`_predict_block` and
    declare their minimum context and whether they consume the external scaler.
    """

    #: When True the adapter scales the context with the provided `scaler` and
    #: inverse-transforms its predictions; when False it works in original scale.
    _uses_external_scaler: bool = False

    def __init__(self, model: Any, *, model_name: str = "") -> None:
        """Wrap one underlying model instance under a benchmark display name."""
        self._model = model
        self.model_name = str(model_name)

    @property
    def model(self) -> Any:
        """The wrapped underlying model instance (``None`` for per-gap fitters)."""
        return self._model

    def _context_window(self) -> int:
        """How many clean left-context points to gather before each gap."""
        raise NotImplementedError

    def _min_context(self) -> int:
        """Minimum context below which the gap is skipped and reported."""
        raise NotImplementedError

    def _predict_block(
        self,
        *,
        context: pd.Series,
        n: int,
        freq: str,
        config_workers: Mapping[str, Any] | None,
    ) -> pd.Series:
        """Return one prediction block (length `n`) in the model's input space."""
        raise NotImplementedError

    def impute_gaps(
        self,
        *,
        series_name: str,
        all_series_map: Mapping[str, pd.Series],
        gap_windows: Sequence[pd.DatetimeIndex],
        test_index: pd.DatetimeIndex,
        scaler: Any | None,
        freq: str,
        config_workers: Mapping[str, Any] | None = None,
    ) -> tuple[pd.Series, list[GapContextFailure]]:
        """Impute each gap using only clean left context, then inverse-scale."""
        del test_index  # Darts adapters derive context from `all_series_map`.
        use_scaled = self._uses_external_scaler and scaler is not None
        context_window = self._context_window()
        min_context = self._min_context()
        failures: list[GapContextFailure] = []

        mask_index = _gap_windows_to_mask_index(gap_windows)
        pred_out = pd.Series(index=mask_index, dtype=float, name=series_name)
        if len(pred_out) == 0:
            return pred_out, failures

        for gap_idx in gap_windows:
            if len(gap_idx) == 0:
                continue

            gap_start = pd.Timestamp(gap_idx.min())
            unscaled_context, scaled_context = _derive_context_before_gap(
                series_name=series_name,
                gap_start=gap_start,
                all_series_map=all_series_map,
                scaler=scaler if use_scaled else None,
                freq=freq,
                required_context=context_window,
            )
            context_series = scaled_context if use_scaled else unscaled_context

            context = _build_clean_left_context(
                series=context_series,
                gap_start=gap_start,
                required_context=context_window,
                freq=freq,
            )

            if len(context) < min_context:
                failures.append(
                    GapContextFailure(
                        model_name=self.model_name,
                        series_name=series_name,
                        gap_start=gap_start,
                        gap_length=int(len(gap_idx)),
                        required_context=min_context,
                        available_context=int(len(context)),
                        reason=_INSUFFICIENT_CONTEXT_REASON,
                    )
                )
                continue

            pred_block = self._predict_block(
                context=context,
                n=int(len(gap_idx)),
                freq=freq,
                config_workers=config_workers,
            )
            if len(pred_block) == len(gap_idx):
                pred_block = pred_block.copy()
                pred_block.index = pd.DatetimeIndex(gap_idx)
                pred_out.loc[pd.DatetimeIndex(gap_idx)] = pred_block.to_numpy(dtype=float)
            else:
                pred_out.loc[pd.DatetimeIndex(gap_idx)] = pred_block.reindex(
                    gap_idx
                ).to_numpy(dtype=float)

        pred = (
            _inverse_scale_prediction_series(
                pred_out,
                scaler=scaler if use_scaled else None,
                freq=freq,
                name=series_name,
            )
            .reindex(mask_index)
            .astype(float)
        )
        return pred, failures


class DartsGlobalGapImputer(_DartsContextImputer):
    """Adapter for pretrained Darts global forecasters (TiDE, NHiTS, RNN, ...).

    Predicts forward from clean left context with `model.predict(n=, series=)`,
    scaling the context (and inverse-scaling predictions) when a `scaler` is
    given. Inputs are cast to float32 before prediction.
    """

    _uses_external_scaler = True

    def __init__(self, model: Any, *, model_name: str = "") -> None:
        """Wrap one pretrained Darts global model under a benchmark name."""
        super().__init__(model, model_name=model_name)
        #: Extra predict kwargs that succeeded last time; retried first so the
        #: failing signature variants are not re-attempted on every gap.
        self._predict_extra_kwargs: dict[str, Any] | None = None

    def _context_window(self) -> int:
        """Gather exactly the model's inferred minimum context before each gap."""
        return infer_darts_minimum_context(self._model)

    def _min_context(self) -> int:
        """Skip gaps whose clean context is below the model's inferred minimum."""
        return infer_darts_minimum_context(self._model)

    def _predict_block(
        self,
        *,
        context: pd.Series,
        n: int,
        freq: str,
        config_workers: Mapping[str, Any] | None,
    ) -> pd.Series:
        """Forecast ``n`` steps from ``context``, retrying predict kwargs variants.

        Tries ``dataloader_kwargs``/``verbose`` combinations first so worker
        settings apply when the Darts model supports them, falling back to a
        bare ``predict`` call otherwise.
        """
        context_ts = TimeSeries.from_series(context, freq=freq).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predict_base_kwargs = {"n": int(n), "series": context_ts}
            predict_attempts: list[dict[str, Any]] = []
            if self._predict_extra_kwargs is not None:
                predict_attempts.append(self._predict_extra_kwargs)
            if config_workers:
                worker_kwargs = dict(config_workers)
                predict_attempts.append(
                    {"verbose": False, "dataloader_kwargs": worker_kwargs}
                )
                predict_attempts.append({"dataloader_kwargs": worker_kwargs})
            predict_attempts.append({"verbose": False})
            predict_attempts.append({})

            pred_ts = None
            last_type_error: TypeError | None = None
            for extra_kwargs in predict_attempts:
                try:
                    pred_ts = self._model.predict(**predict_base_kwargs, **extra_kwargs)
                    self._predict_extra_kwargs = extra_kwargs
                    break
                except TypeError as exc:
                    last_type_error = exc

            if pred_ts is None:
                if last_type_error is not None:
                    raise last_type_error
                raise RuntimeError(
                    f"No fue posible ejecutar predict para '{self.model_name}'"
                )

        return pred_ts.to_series().astype(float)


class ProphetGapImputer(_DartsContextImputer):
    """Adapter for Darts' local Prophet model.

    Prophet is fit fresh on the clean left context of each gap and then forecasts
    forward. It works in the original scale, so the external `scaler` is ignored.
    """

    _uses_external_scaler = False

    #: Grab every contiguous clean point before the gap (Prophet fits better with
    #: more history); ``_build_clean_left_context`` stops at NaN barriers / start.
    _CONTEXT_WINDOW = 10**9

    def __init__(
        self,
        *,
        model_name: str = "Prophet",
        min_context: int = 3,
        prophet_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate Prophet availability and store per-gap fitting options."""
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "darts.models.Prophet no esta disponible en este entorno"
            ) from PROPHET_IMPORT_ERROR
        super().__init__(model=None, model_name=model_name)
        # Prophet requires at least 3 observations to fit.
        self._min_required = max(3, int(min_context))
        self._prophet_kwargs = dict(prophet_kwargs or {})

    def _context_window(self) -> int:
        """Request all contiguous clean history (capped by ``_CONTEXT_WINDOW``)."""
        return self._CONTEXT_WINDOW

    def _min_context(self) -> int:
        """Skip gaps with fewer clean points than Prophet's fitting minimum."""
        return self._min_required

    def _predict_block(
        self,
        *,
        context: pd.Series,
        n: int,
        freq: str,
        config_workers: Mapping[str, Any] | None,
    ) -> pd.Series:
        """Fit a fresh Prophet on ``context`` and forecast the ``n`` gap steps."""
        del config_workers
        context_ts = TimeSeries.from_series(context, freq=freq)
        model = DartsProphet(**self._prophet_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(context_ts)
            pred_ts = model.predict(n=int(n))
        return pred_ts.to_series().astype(float)


# --------------------------------------------------------------------------- #
# Interpolation adapter (lightweight, no artifacts)
# --------------------------------------------------------------------------- #
class InterpolationGapImputer:
    """Lightweight `GapImputer`: seasonal-aware time interpolation, no artifacts.

    The whole masked series is filled at once by adding back a 24h hour-of-day
    climatology to the time-interpolated residual (so long gaps fall back to the
    seasonal mean instead of a flat line). Works in the original scale and ignores
    the external ``scaler``. Serves as the baseline imputer for the preprocessing
    pipeline, isolating the effect of removing anomalies.
    """

    _uses_external_scaler = False

    def __init__(self, *, model_name: str = "interp", seasonality: int = 24) -> None:
        """Store the display name and the climatology seasonality (24 = hourly)."""
        self.model_name = str(model_name)
        self._seasonality = int(seasonality)

    def _fill(self, series: pd.Series, *, freq: str) -> pd.Series:
        """Return a fully-filled copy of ``series`` (no NaN) on the regular grid."""
        s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
        linear = s.interpolate(method="time", limit_direction="both")
        if self._seasonality == 24:
            # Hour-of-day climatology + interpolated residual (seasonal fallback).
            climatology = s.groupby(s.index.hour).transform("mean")
            residual = (s - climatology).interpolate(method="time", limit_direction="both")
            seasonal = climatology + residual
            filled = seasonal.where(seasonal.notna(), linear)
        else:
            filled = linear
        return filled.ffill().bfill()

    def impute_gaps(
        self,
        *,
        series_name: str,
        all_series_map: Mapping[str, pd.Series],
        gap_windows: Sequence[pd.DatetimeIndex],
        test_index: pd.DatetimeIndex,
        scaler: Any | None = None,
        freq: str = "h",
        config_workers: Mapping[str, Any] | None = None,
    ) -> tuple[pd.Series, list[GapContextFailure]]:
        """Fill all missing points; return predictions over the mask timestamps."""
        del scaler, config_workers, test_index  # Interpolation ignores these.
        mask_index = _gap_windows_to_mask_index(gap_windows)
        if len(mask_index) == 0:
            return pd.Series(index=mask_index, dtype=float, name=series_name), []

        # `all_series_map` still holds the true values at the gap timestamps, so
        # they must be re-masked before filling; otherwise both the interpolation
        # and the hour-of-day climatology in `_fill` would read the ground truth
        # back (leakage), like `LinearGapImputer` re-masks too.
        series = all_series_map[series_name].copy()
        series.loc[series.index.intersection(mask_index)] = np.nan
        filled = self._fill(series, freq=freq)
        return filled.reindex(mask_index).astype(float), []


class LinearGapImputer:
    """Literal linear-interpolation baseline exposed as a `GapImputer`.

    Every missing point is filled by a straight line between its nearest observed
    neighbours (``pandas.Series.interpolate(method="linear")``); for a size-1 gap
    this is exactly the mean of the two adjacent values. It works in the original
    scale and ignores the external ``scaler``. Unlike :class:`InterpolationGapImputer`
    there is no seasonal climatology term — the fill is purely linear by design.
    """

    _uses_external_scaler = False

    def __init__(self, *, model_name: str = "LinearInterp") -> None:
        """Store the display name used in benchmark results."""
        self.model_name = str(model_name)

    def _fill(self, series: pd.Series, *, freq: str) -> pd.Series:
        """Return a fully-filled copy of ``series`` (no NaN) using linear interp."""
        s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
        return (
            s.interpolate(method="linear", limit_direction="both").ffill().bfill()
        )

    def impute_gaps(
        self,
        *,
        series_name: str,
        all_series_map: Mapping[str, pd.Series],
        gap_windows: Sequence[pd.DatetimeIndex],
        test_index: pd.DatetimeIndex,
        scaler: Any | None = None,
        freq: str = "h",
        config_workers: Mapping[str, Any] | None = None,
    ) -> tuple[pd.Series, list[GapContextFailure]]:
        """Fill all missing points linearly; return predictions over the mask."""
        del scaler, config_workers, test_index  # Linear interpolation ignores these.
        mask_index = _gap_windows_to_mask_index(gap_windows)
        if len(mask_index) == 0:
            return pd.Series(index=mask_index, dtype=float, name=series_name), []

        # `all_series_map` still holds the true values at the gap timestamps, so
        # they must be re-masked before interpolating; otherwise the fill would
        # just read the ground truth back (leakage), like TSPulse re-masks too.
        series = all_series_map[series_name].copy()
        series.loc[series.index.intersection(mask_index)] = np.nan
        filled = self._fill(series, freq=freq)
        return filled.reindex(mask_index).astype(float), []


# --------------------------------------------------------------------------- #
# TSPulse adapter (mask-based whole-series reconstruction)
# --------------------------------------------------------------------------- #
class TSPulseGapImputer:
    """Adapter exposing TSPulse zero-shot imputation behind `GapImputer`.

    TSPulse reconstructs the whole masked series at once and standardizes inputs
    internally, so it consumes the original scale and ignores the external
    `scaler`.
    """

    _uses_external_scaler = False

    def __init__(
        self,
        *,
        model_id: str = "ibm-granite/granite-timeseries-tspulse-r1",
        revision: str = "tspulse-hybrid-dualhead-512-p8-r1",
        model_path: str | os.PathLike[str] | None = None,
        context_length: int = 512,
        freq: str = "h",
        batch_size: int = 1000,
        device: str | None = None,
        scaling: bool = True,
        model: Any | None = None,
        hf_token: str | None = None,
        local_files_only: bool = False,
        model_name: str = "TSPulse",
    ) -> None:
        """Store adapter configuration and optional pre-loaded model instance."""
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.model_path = str(model_path) if model_path is not None else None
        self.context_length = int(context_length)
        self.freq = str(freq)
        self.batch_size = int(batch_size)
        preferred = "cpu" if device is None else str(device)
        self.device = resolve_device(preferred)
        self.scaling = bool(scaling)
        self.model = model
        self.hf_token = hf_token if hf_token is not None else os.getenv("HF_TOKEN")
        self.local_files_only = bool(local_files_only)
        self.model_name = str(model_name)

    def _ensure_model(self, num_input_channels: int) -> Any:
        """Load TSPulse model lazily on first use."""
        if self.model is not None:
            return self.model

        if not TSFM_PUBLIC_AVAILABLE:
            raise ImportError(
                "tsfm_public no esta disponible en este entorno"
            ) from TSFM_PUBLIC_IMPORT_ERROR

        source = self.model_path if self.model_path is not None else self.model_id
        load_kwargs: dict[str, Any] = {
            "num_input_channels": int(num_input_channels),
            "mask_type": "user",
            "token": self.hf_token,
            "local_files_only": self.local_files_only,
        }
        if self.model_path is None:
            load_kwargs["revision"] = self.revision

        self.model = TSPulseForReconstruction.from_pretrained(source, **load_kwargs)
        return self.model

    def _impute_full_series(
        self,
        *,
        series_name: str,
        all_series_map: Mapping[str, pd.Series],
        mask_index: pd.DatetimeIndex,
        test_index: pd.DatetimeIndex,
        freq: str,
    ) -> pd.Series:
        """Run official TSPulse zero-shot imputation on one masked test series."""
        if not TSFM_PUBLIC_AVAILABLE:
            raise ImportError(
                "tsfm_public no esta instalado; no se puede ejecutar TSPulse"
            ) from TSFM_PUBLIC_IMPORT_ERROR

        prepared, test_index_out = build_tspulse_context_frame(
            series_name=series_name,
            all_series_map=all_series_map,
            mask_index=mask_index,
            test_index=test_index,
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
            out[value_col].to_numpy(dtype=float), index=idx, name=series_name
        )
        return imputed.reindex(test_index_out)

    def impute_gaps(
        self,
        *,
        series_name: str,
        all_series_map: Mapping[str, pd.Series],
        gap_windows: Sequence[pd.DatetimeIndex],
        test_index: pd.DatetimeIndex,
        scaler: Any | None = None,
        freq: str = "h",
        config_workers: Mapping[str, Any] | None = None,
    ) -> tuple[pd.Series, list[GapContextFailure]]:
        """Impute all missing points; return predictions over mask timestamps."""
        del scaler, config_workers  # TSPulse standardizes internally.
        mask_index = _gap_windows_to_mask_index(gap_windows)
        if len(mask_index) == 0:
            return pd.Series(index=mask_index, dtype=float, name=series_name), []

        imputed = self._impute_full_series(
            series_name=series_name,
            all_series_map=all_series_map,
            mask_index=mask_index,
            test_index=test_index,
            freq=freq,
        )
        return imputed.reindex(mask_index).astype(float), []


__all__ = [
    "GapImputer",
    "DartsGlobalGapImputer",
    "ProphetGapImputer",
    "TSPulseGapImputer",
    "InterpolationGapImputer",
    "LinearGapImputer",
    "PROPHET_AVAILABLE",
    "PROPHET_IMPORT_ERROR",
    "TSFM_PUBLIC_AVAILABLE",
    "TSFM_PUBLIC_IMPORT_ERROR",
    "build_tspulse_context_frame",
    "infer_darts_minimum_context",
    "_build_clean_left_context",
]
