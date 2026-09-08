"""File discovery and pandas loaders for the supported dataset formats."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd


class UnsupportedFileFormatError(Exception):
    """Exception raised when the file format is not supported."""


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


def _load_upct_pollutant_df(file_path: str, pollutant: str) -> pd.DataFrame:
    """Load one UPCT export and normalize it to a 5-minute pollutant series."""
    df = pd.read_csv(file_path, sep=";", decimal=",", skiprows=1)
    if "Datetime" not in df.columns:
        raise ValueError("el CSV UPCT no contiene la columna 'Datetime'")

    pollutant_name = pollutant.strip().upper()
    value_columns = [
        column
        for column in df.columns
        if str(column).strip().upper() == pollutant_name
        or str(column).strip().upper().startswith(f"{pollutant_name} ")
    ]
    value_columns = [
        column
        for column in value_columns
        if not str(column).strip().upper().startswith(("FLAG ", "DESCRIPTION "))
    ]
    if len(value_columns) != 1:
        raise ValueError(
            f"no se pudo identificar una única columna de {pollutant_name}: {value_columns}"
        )

    index = pd.to_datetime(df["Datetime"], errors="coerce")
    values = pd.to_numeric(df[value_columns[0]], errors="coerce")
    out = pd.DataFrame({pollutant_name: values.to_numpy()}, index=index)
    out = out[~out.index.isna()]

    # UPCT samples drift by a few seconds around the nominal five-minute grid.
    out.index = out.index.round("5min")
    return out.groupby(level=0).mean().sort_index()


def load_pollutant_file(
    file_path: str,
    pollutant: str,
    *,
    target_column_index: int = 0,
) -> tuple[str, pd.DataFrame] | None:
    """Load one raw pollutant file and return its canonical series name and values."""
    path = Path(file_path)
    is_upct = path.parent.name.strip().upper() == "UPCT"

    try:
        if is_upct:
            df = _load_upct_pollutant_df(file_path, pollutant)
            series_name = f"UPCT_{pollutant.strip().upper()}"
        else:
            raw = _load_csv_df(file_path)
            if target_column_index < 0 or target_column_index >= len(raw.columns):
                raise ValueError(
                    f"target_column_index={target_column_index} fuera de rango; "
                    f"columnas disponibles ({len(raw.columns)}): {list(raw.columns)}"
                )
            df = raw.iloc[:, [target_column_index]].copy()
            series_name = path.parent.name.strip()

        if df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        if df.empty:
            return None
        return series_name, df.sort_index()
    except Exception as exc:
        logging.warning("Error processing %s (file dropped from dataset): %s", file_path, exc)
        return None


def _filename_matches_pollutant(path: Path, pollutant: str) -> bool:
    """Return whether a filename contains the pollutant as a complete token."""
    token = re.escape(pollutant.strip())
    return re.search(rf"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])", path.stem, re.I) is not None


def load_raw_5m(
    pollutant: str,
    base_dir: str = "data/raw/datos_estaciones_5m",
) -> list[tuple[str, pd.DataFrame]]:
    """Discover and load the raw 5-minute series for one pollutant.

    Discovers scraper and UPCT CSV exports and returns ``(series_name,
    dataframe)`` pairs sorted by path, each a single value column on a regular
    datetime index. UPCT identifiers are normalized to ``UPCT_<pollutant>``.
    The dataframes are ready to feed to
    :func:`airquality.data.preprocessing.preprocess`.
    """
    out: list[tuple[str, pd.DataFrame]] = []
    paths = sorted(
        path
        for path in Path(base_dir).glob("*/*.csv")
        if _filename_matches_pollutant(path, pollutant)
    )
    for path in paths:
        loaded = load_pollutant_file(str(path), pollutant)
        if loaded is None:
            continue
        out.append(loaded)
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
