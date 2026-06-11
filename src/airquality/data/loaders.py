"""File discovery and pandas loaders for the supported dataset formats."""

from __future__ import annotations

import glob
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
        print(f"Skipping file {file_path}: {exc}")
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return None

    return None
