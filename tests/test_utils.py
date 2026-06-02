from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airquality.data.utils import (
    UnsupportedFileFormatError,
    ensure_datetime_series,
    get_longest_segment,
    load_dataset_paths,
    load_to_df,
)


def test_ensure_datetime_series_normalizes_sorts_and_sets_freq() -> None:
    idx = pd.to_datetime(["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    s = pd.Series([2, 1, 3], index=idx, name="x")

    out = ensure_datetime_series(s, freq="h", name="fallback")

    assert list(out.index) == list(pd.date_range("2024-01-01 00:00", periods=2, freq="h"))
    assert out.iloc[1] == 3.0
    assert out.name == "x"


def test_ensure_datetime_series_raises_with_bad_inputs() -> None:
    with pytest.raises(TypeError):
        ensure_datetime_series([1, 2], freq="h", name="a")

    with pytest.raises(TypeError):
        ensure_datetime_series(pd.Series([1, 2]), freq="h", name="a")


def test_load_dataset_paths_filters_by_keyword_and_extension(tmp_path: Path) -> None:
    (tmp_path / "estacion_NO2.csv").write_text("x", encoding="utf-8")
    (tmp_path / "estacion_CO.csv").write_text("x", encoding="utf-8")

    paths = load_dataset_paths(str(tmp_path), key_word="NO2", file_extension="csv")

    assert len(paths) == 1
    assert paths[0].endswith("estacion_NO2.csv")


def test_load_to_df_csv_renames_when_requested(tmp_path: Path) -> None:
    p = tmp_path / "Station_NO2.csv"
    p.write_text("time,value\n2024-01-01 00:00:00,1\n", encoding="utf-8")

    df = load_to_df(str(p), name_from_path=True)

    assert df is not None
    assert list(df.columns) == ["Station"]


def test_load_to_df_unsupported_extension_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("abc", encoding="utf-8")

    assert load_to_df(str(p)) is None


def test_get_longest_segment_finds_best_block_without_force_end() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h")
    a = pd.DataFrame({"A": [1, 1, None, 1, 1, 1]}, index=idx)
    b = pd.DataFrame({"B": [1, 1, None, 1, 1, 1]}, index=idx)

    out = get_longest_segment([a, b], force_end=False)

    assert list(out.columns) == ["A", "B"]
    assert len(out) == 3
    assert out.index.min() == idx[3]


def test_get_longest_segment_force_end_returns_last_connected_block() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    a = pd.DataFrame({"A": [1, None, 2, 3, 4]}, index=idx)
    b = pd.DataFrame({"B": [1, None, 2, 3, 4]}, index=idx)

    out = get_longest_segment([a, b], force_end=True)

    assert out.index.min() == idx[2]
    assert out.index.max() == idx[4]
