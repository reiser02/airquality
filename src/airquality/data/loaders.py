"""File discovery and pandas loaders for the supported dataset formats."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import pandas as pd

from airquality.config import cfg_get_str


class UnsupportedFileFormatError(Exception):
    """Exception raised when the file format is not supported."""


def load_dataset_paths() -> list[str]:
    """Return files selected by the shared project configuration."""
    base_path = cfg_get_str("data", "base_path_glob", "../../data/*/")
    key_word = cfg_get_str("data", "key_word", "CO_media_horaria")
    file_extension = cfg_get_str("data", "file_extension", "json")
    return glob.glob(os.path.join(base_path, f"*{key_word}*.{file_extension}"))


def _load_json_df(file_path: str) -> pd.DataFrame:
    """Load the project JSON payload format into an hourly dataframe."""
    data_series = pd.read_json(file_path, typ="series")
    df = pd.DataFrame(data_series.rows, columns=data_series.cols)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.set_index(df.columns[0], inplace=True)
    return df.asfreq("h")


def _load_csv_df(file_path: str) -> pd.DataFrame:
    """Load one CSV file using its first column as the datetime index."""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)


def load_raw_5m(
    pollutant: str,
    base_dir: str = "data/raw/datos_estaciones_5m",
) -> list[tuple[str, pd.DataFrame]]:
    """Discover and load the raw 5-minute series for one pollutant.

    Globs ``<base_dir>/<station>/<station>_<pollutant>.csv`` (the layout produced
    by the scraper) and returns ``(station_name, dataframe)`` pairs sorted by
    station, each a single value column indexed by the ``fecha`` datetime column.
    The dataframes are ready to feed to
    :func:`airquality.data.preprocessing.preprocess`.
    """
    pattern = os.path.join(base_dir, "*", f"*_{pollutant}.csv")
    out: list[tuple[str, pd.DataFrame]] = []
    for file_path in sorted(glob.glob(pattern)):
        station = os.path.basename(os.path.dirname(file_path))
        df = _load_csv_df(file_path)
        if df is None or df.empty:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        out.append((station, df.sort_index()))
    return out


def load_to_df(file_path: str, name_from_path: bool = True) -> pd.DataFrame | None:
    """Load a supported file into a dataframe, optionally renaming its column."""
    path = Path(file_path)
    extension = path.suffix.lower()

    try:
        if extension == ".json":
            df = _load_json_df(file_path)
        elif extension == ".csv":
            df = _load_csv_df(file_path)
        else:
            raise UnsupportedFileFormatError(
                f"Unsupported file format: '{extension}'. Only .json and .csv are supported."
            )

        if name_from_path:
            column_name = os.path.basename(file_path).split("_")[0]
            df.columns = [column_name]

        return df
    except UnsupportedFileFormatError as exc:
        # A dropped file silently changes the dataset composition: keep the
        # skip visible in logs (not just stdout).
        logging.warning("Skipping file %s: %s", file_path, exc)
    except Exception as exc:
        logging.warning("Error processing %s (file dropped from dataset): %s", file_path, exc)
        return None

    return None
