"""Tests extra loader scenarios for JSON payloads, CSV naming, and failures."""

from __future__ import annotations

import pandas as pd

from airquality.data.loaders import _load_json_df, load_to_df


def test_load_json_df_builds_hourly_index_from_rows_and_cols(tmp_path) -> None:
    payload = (
        '{"rows": [[1704067200000, 1.0], [1704070800000, 2.0]], '
        '"cols": ["time", "value"]}'
    )
    path = tmp_path / "station.json"
    path.write_text(payload, encoding="utf-8")

    df = _load_json_df(str(path))

    assert list(df.columns) == ["value"]
    assert list(df.index) == list(pd.date_range("2024-01-01 00:00:00", periods=2, freq="h"))
    assert df.iloc[-1, 0] == 2.0


def test_load_to_df_preserves_columns_when_name_from_path_is_false(tmp_path) -> None:
    path = tmp_path / "Station_NO2.csv"
    path.write_text("time,value\n2024-01-01 00:00:00,1\n", encoding="utf-8")

    df = load_to_df(str(path), name_from_path=False)

    assert df is not None
    assert list(df.columns) == ["value"]


def test_load_to_df_returns_none_on_loader_exception(tmp_path, monkeypatch) -> None:
    path = tmp_path / "Station_NO2.csv"
    path.write_text("time,value\n2024-01-01 00:00:00,1\n", encoding="utf-8")

    monkeypatch.setattr(
        "airquality.data.loaders._load_csv_df",
        lambda file_path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert load_to_df(str(path), name_from_path=False) is None
