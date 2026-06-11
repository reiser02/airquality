"""Tests device, warning, and dataset normalization utility functions."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality.data.io import (
    configure_warnings,
    load_and_normalize_series,
    resolve_device,
)


def test_resolve_device_cpu_and_bad_value() -> None:
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_configure_warnings_quiet_false_does_not_crash() -> None:
    configure_warnings(quiet=False)


def test_load_and_normalize_series_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.to_datetime(["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    df = pd.DataFrame({"NO2": [2, 1, 3]}, index=idx)

    monkeypatch.setattr("airquality.data.io.load_dataset_paths", lambda: ["a.csv"])
    monkeypatch.setattr("airquality.data.io.load_to_df", lambda *_args, **_kwargs: df)

    out = load_and_normalize_series(freq="h")

    assert len(out) == 1
    assert list(out[0].columns) == ["NO2"]
    assert float(out[0].iloc[1, 0]) == 3.0


def test_load_and_normalize_series_raises_when_no_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("airquality.data.io.load_dataset_paths", lambda: [])

    with pytest.raises(FileNotFoundError):
        load_and_normalize_series(freq="h")


def test_load_and_normalize_series_target_column_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    df = pd.DataFrame({"A": [1, 2]}, index=idx)
    monkeypatch.setattr("airquality.data.io.load_dataset_paths", lambda: ["a.csv"])
    monkeypatch.setattr("airquality.data.io.load_to_df", lambda *_args, **_kwargs: df)

    with pytest.raises(ValueError):
        load_and_normalize_series(freq="h", target_column_index=1)
