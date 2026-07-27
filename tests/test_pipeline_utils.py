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
    raw = pd.DataFrame(
        {"NO2": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])
    )
    hourly = pd.DataFrame(
        {"NO2": [1.0, float("nan"), 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )
    calls = {}

    monkeypatch.setattr(
        "airquality.data.io.cfg_get_str",
        lambda section, option, default: {
            ("data", "key_word"): "NO2",
            ("data", "raw_base_dir"): "raw/5m",
        }.get((section, option), default),
    )
    monkeypatch.setattr(
        "airquality.data.io.load_raw_5m",
        lambda pollutant, base_dir: calls.update(
            {"load": (pollutant, base_dir)}
        )
        or [("Station A", raw)],
    )
    monkeypatch.setattr(
        "airquality.data.io.preprocess",
        lambda frames, pollutant: calls.update(
            {"preprocess": (frames, pollutant)}
        )
        or ([hourly], [0]),
    )

    out = load_and_normalize_series(freq="h")

    assert len(out) == 1
    assert list(out[0].columns) == ["Station A"]
    assert pd.isna(out[0].iloc[1, 0])
    assert calls["load"] == ("NO2", "raw/5m")
    assert calls["preprocess"] == ([raw], "NO2")


def test_load_and_normalize_series_raises_when_no_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("airquality.data.io.load_raw_5m", lambda *_args: [])

    with pytest.raises(FileNotFoundError, match="datos raw"):
        load_and_normalize_series(freq="h")


def test_load_and_normalize_series_requires_hourly_output() -> None:
    with pytest.raises(ValueError, match="freq='h'"):
        load_and_normalize_series(freq="30min")
