"""Runtime helpers for device selection, logging, and series loading."""

from __future__ import annotations

import logging

import pandas as pd

from airquality.config import cfg_get_str
from airquality.data.loaders import load_raw_5m
from airquality.data.preprocessing import preprocess
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
) -> list[pd.DataFrame]:
    """Load raw 5-minute stations and apply the shared hourly preprocessing."""
    if freq != "h":
        raise ValueError("El preprocesado desde datos de 5 min requiere freq='h'")

    pollutant = cfg_get_str("data", "key_word", "NO2")
    raw_base_dir = cfg_get_str(
        "data", "raw_base_dir", "data/raw/datos_estaciones_5m"
    )
    stations = load_raw_5m(pollutant, raw_base_dir)
    if not stations:
        raise FileNotFoundError(
            f"No se encontraron datos raw de {pollutant} bajo {raw_base_dir}."
        )

    out: list[pd.DataFrame] = []
    for station, raw in stations:
        (hourly,), _ = preprocess([raw], pollutant)
        if hourly.empty:
            continue
        normalized = ensure_datetime_series(
            hourly.iloc[:, 0], freq=freq, name=station
        )
        out.append(normalized.to_frame(name=station))
    return out
