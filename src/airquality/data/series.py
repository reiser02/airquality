"""Helpers for converting project series to normalized pandas time series."""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts import TimeSeries


def ensure_datetime_series(series: pd.Series, freq: str, name: str) -> pd.Series:
    """Validate and normalize one pandas series to a regular datetime grid."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"La serie '{name}' debe ser pd.Series; recibido: {type(series)}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"La serie '{name}' debe tener DatetimeIndex")

    out = series.sort_index().copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.asfreq(freq)
    out.name = str(series.name) if series.name is not None else name
    return out.astype(float)


def to_pd_series(
    obj: Any,
    *,
    freq: str | None = None,
    name: str | None = None,
    sort_index: bool = True,
) -> pd.Series:
    """Convert supported series-like inputs into a pandas Series."""
    if isinstance(obj, TimeSeries):
        series = obj.to_series()
    elif isinstance(obj, pd.Series):
        series = obj.copy()
    elif isinstance(obj, pd.DataFrame) and len(obj.columns) == 1:
        series = obj.iloc[:, 0].copy()
    else:
        raise TypeError(f"No se puede convertir a pd.Series: {type(obj)}")

    if name is not None:
        series.name = name
    if sort_index:
        series = series.sort_index()
    if freq is not None:
        resolved_name = str(series.name) if series.name is not None else (name or "series")
        series = ensure_datetime_series(series, freq=freq, name=resolved_name)
    return series
