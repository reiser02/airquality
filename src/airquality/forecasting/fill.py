"""Fill the NaN gaps of one series with any :class:`GapImputer`.

Thin helper layered on top of the unified ``GapImputer.impute_gaps`` contract
(:mod:`airquality.imputation.imputers`). It turns a single gappy series into a
complete one, picking the concrete adapter from the imputer registry so the
preprocessing pipeline can swap imputation models by config name (``interp``,
``LinearInterp``, ``Prophet``, a Darts model such as ``TiDE``, or ``TSPulse``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

from airquality.data.series import ensure_datetime_series
from airquality.imputation.imputers import (
    GapImputer,
    InterpolationGapImputer,
    LinearGapImputer,
    ProphetGapImputer,
)
from airquality.imputation.registry import (
    DARTS_GLOBAL,
    INTERP,
    LINEAR,
    PROPHET,
    TSPULSE,
    resolve_imputer_family,
)


def _repo_root() -> Path:
    """Return the repository root (three levels above this module)."""
    return Path(__file__).resolve().parents[3]


def nan_gap_windows(series: pd.Series) -> list[pd.DatetimeIndex]:
    """Split the NaN positions of ``series`` into contiguous gap windows."""
    mask = series.isna().to_numpy()
    if not mask.any():
        return []
    index = pd.DatetimeIndex(series.index)
    windows: list[pd.DatetimeIndex] = []
    start: int | None = None
    for i, is_nan in enumerate(mask):
        if is_nan and start is None:
            start = i
        elif not is_nan and start is not None:
            windows.append(index[start:i])
            start = None
    if start is not None:
        windows.append(index[start:])
    return windows


def _fit_scaler(series: pd.Series, *, freq: str) -> Scaler:
    """Fit a Darts ``Scaler`` (StandardScaler) on the gap-filled observed series."""
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    s = s.interpolate(method="time", limit_direction="both").ffill().bfill()
    scaler = Scaler(scaler=StandardScaler(), global_fit=True)
    scaler.fit(TimeSeries.from_series(s, freq=freq).astype(np.float32))
    return scaler


def build_imputer(
    model_name: str,
    *,
    freq: str = "h",
    size_k: int = 5,
    force_cpu: bool = True,
) -> GapImputer:
    """Construct the configured :class:`GapImputer` for one model name."""
    family = resolve_imputer_family(model_name)
    if family == INTERP:
        return InterpolationGapImputer(model_name=model_name)
    if family == LINEAR:
        return LinearGapImputer(model_name=model_name)
    if family == PROPHET:
        return ProphetGapImputer(model_name=model_name)
    if family == DARTS_GLOBAL:
        # Imported lazily: loading artifacts pulls in heavy Darts/torch machinery.
        from airquality.imputation.run_benchmark import load_darts_models_from_artifacts

        loaded = load_darts_models_from_artifacts(
            repo_root=_repo_root(),
            size_k=size_k,
            model_names=[model_name],
            force_cpu=force_cpu,
            strict=True,
        )
        return loaded[model_name]
    if family == TSPULSE:
        from airquality.config import cfg_get_int, cfg_get_str
        from airquality.imputation.imputers import TSPulseGapImputer

        model_path = cfg_get_str("benchmark", "tspulse_model_path", "") or None
        return TSPulseGapImputer(
            model_id=cfg_get_str(
                "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
            ),
            revision=cfg_get_str(
                "tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1"
            ),
            model_path=model_path,
            context_length=cfg_get_int("tspulse", "context_length", 512),
            freq=freq,
            device=cfg_get_str("tspulse", "device", "cpu"),
            model_name=model_name,
        )
    raise ValueError(f"Familia de imputacion no soportada: {family}")


def impute_series(
    series: pd.Series,
    imputer: GapImputer,
    *,
    freq: str = "h",
    use_scaler: bool = False,
) -> pd.Series:
    """Return a fully-imputed copy of ``series`` (no NaN) on the regular grid.

    Predictions come from ``imputer.impute_gaps`` over the NaN windows. Any point
    the imputer leaves unfilled (e.g. a Darts gap without enough left context,
    reported as a ``GapContextFailure``) falls back to seasonal interpolation so
    the returned series is always complete.
    """
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    gap_windows = nan_gap_windows(s)
    if not gap_windows:
        return s

    scaler = _fit_scaler(s, freq=freq) if use_scaler else None
    test_index = pd.DatetimeIndex(s.index)
    pred, _failures = imputer.impute_gaps(
        series_name=str(s.name),
        all_series_map={str(s.name): s},
        gap_windows=gap_windows,
        test_index=test_index,
        scaler=scaler,
        freq=freq,
    )

    filled = s.copy()
    if len(pred) > 0:
        aligned = pred.reindex(filled.index)
        filled = filled.where(filled.notna(), aligned)

    if filled.isna().any():
        # Seasonal interpolation backstop for anything the imputer left as NaN.
        fallback = InterpolationGapImputer()._fill(s, freq=freq)
        filled = filled.where(filled.notna(), fallback.reindex(filled.index))

    return filled.astype(float)


__all__ = ["nan_gap_windows", "build_imputer", "impute_series"]
