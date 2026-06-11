"""Runtime helpers for device selection, logging, and series loading."""

from __future__ import annotations

import logging

import pandas as pd

from airquality.data.loaders import load_dataset_paths, load_to_df
from airquality.data.series import ensure_datetime_series, to_pd_series


def resolve_device(preferred: str) -> str:
    """Resolve `cpu` or fall back from requested `cuda` when unavailable."""
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
    """Reduce noisy logs from training and transformer dependencies."""
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


def load_and_normalize_series(
    *,
    freq: str,
    name_from_path: bool = True,
    target_column_index: int | None = None,
) -> list[pd.DataFrame]:
    """Load matching files and normalize each target column to one datetime series."""
    file_paths = sorted(load_dataset_paths())
    if not file_paths:
        raise FileNotFoundError(
            "No se encontraron archivos para la configuracion actual."
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
            normalized = ensure_datetime_series(series, freq=freq, name=col_name)
            out.append(normalized.to_frame(name=col_name))
            continue

        if len(df.columns) != 1:
            continue

        col_name = str(df.columns[0])
        normalized = ensure_datetime_series(df.iloc[:, 0], freq=freq, name=col_name)
        out.append(normalized.to_frame(name=col_name))

    return out
