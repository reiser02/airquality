from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from darts import TimeSeries

from airquality.data.utils import load_dataset_paths, load_to_df


def resolve_device(preferred: str) -> str:
    choice = str(preferred).strip().lower()
    if choice not in {"cpu", "cuda"}:
        raise ValueError("preferred debe ser 'cpu' o 'cuda'")
    if choice == "cpu":
        return "cpu"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def configure_warnings(quiet: bool = True) -> None:
    if not quiet:
        return

    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in (
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "transformers",
        "transformers.pipelines",
        "transformers.pipelines.base",
        "darts",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def to_pd_series(
    obj: Any,
    *,
    freq: str | None = None,
    name: str | None = None,
    sort_index: bool = True,
) -> pd.Series:
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
        series = series[~series.index.duplicated(keep="last")]
        series = series.asfreq(freq)
    return series


def load_and_normalize_series(
    *,
    base_path: str,
    key_word: str,
    file_extension: str,
    freq: str,
    name_from_path: bool = True,
    target_column_index: int | None = None,
) -> list[pd.DataFrame]:
    file_paths = sorted(
        load_dataset_paths(
            base_path=base_path,
            key_word=key_word,
            file_extension=file_extension,
        )
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No se encontraron archivos en '{base_path}' con keyword='{key_word}' y extension='{file_extension}'."
        )

    out: list[pd.DataFrame] = []
    for file_path in file_paths:
        df = load_to_df(file_path, name_from_path=name_from_path)
        if df is None or df.empty:
            continue

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                continue

        df = df[~df.index.isna()].sort_index()

        if target_column_index is not None:
            if target_column_index < 0 or target_column_index >= len(df.columns):
                raise ValueError(
                    f"target_column_index={target_column_index} fuera de rango en '{file_path}'. "
                    f"Columnas disponibles ({len(df.columns)}): {list(df.columns)}"
                )
            col_name = str(df.columns[int(target_column_index)])
            series = pd.to_numeric(df[col_name], errors="coerce")
            series = series[~series.index.duplicated(keep="last")].sort_index().asfreq(freq)
            out.append(series.to_frame(name=col_name).astype(float))
            continue

        if len(df.columns) != 1:
            continue

        col_name = str(df.columns[0])
        series = df.iloc[:, 0].astype(float).sort_index()
        series = series[~series.index.duplicated(keep="last")].asfreq(freq)
        out.append(series.to_frame(name=col_name))

    return out
