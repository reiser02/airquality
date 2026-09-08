"""Tests extra loader scenarios for JSON payloads, CSV naming, and failures."""

from __future__ import annotations

import pandas as pd
import pytest

from airquality.data.loaders import _load_json_df, load_raw_5m, load_to_df


def _write_5m_csv(path, rows: list[tuple[str, float]]) -> None:
    """Write a tiny ``fecha,value`` 5-minute CSV at ``path`` (parent dirs created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "fecha,value\n" + "".join(f"{ts},{value}\n" for ts, value in rows)
    path.write_text(body, encoding="utf-8")


def _write_upct_csv(path, pollutant: str) -> None:
    """Write a minimal UPCT export with its metadata row and decimal commas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Data Validated + Temporary;\n"
        f'Device;Datetime;"{pollutant} GCc (ug/m3)";"Flag {pollutant} GCc";'
        f'"Description {pollutant} GCc";\n'
        f'"UPCT CAMPUS PASEO";2024-01-01 00:01:44;18,31;T;"";\n'
        f'"UPCT CAMPUS PASEO";2024-01-01 00:07:04;16,43;T;"";\n',
        encoding="utf-8",
    )


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


# --- load_raw_5m ---------------------------------------------------------


def test_load_raw_5m_discovers_and_sorts_stations(tmp_path) -> None:
    base = tmp_path / "raw"
    _write_5m_csv(base / "StationB" / "StationB_NO2.csv", [("2024-01-01 00:00:00", 1.0)])
    _write_5m_csv(base / "StationA" / "StationA_NO2.csv", [("2024-01-01 00:00:00", 2.0)])

    out = load_raw_5m("NO2", str(base))

    assert [station for station, _ in out] == ["StationA", "StationB"]
    for _, df in out:
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["value"]


def test_load_raw_5m_filters_by_pollutant(tmp_path) -> None:
    base = tmp_path / "raw"
    _write_5m_csv(base / "StationA" / "StationA_NO2.csv", [("2024-01-01 00:00:00", 1.0)])
    _write_5m_csv(base / "StationA" / "StationA_CO.csv", [("2024-01-01 00:00:00", 9.0)])

    out = load_raw_5m("NO2", str(base))

    assert len(out) == 1
    assert out[0][0] == "StationA"


@pytest.mark.parametrize("pollutant", ["NO2", "O3"])
def test_load_raw_5m_normalizes_upct_name_column_and_grid(
    tmp_path, pollutant: str
) -> None:
    base = tmp_path / "raw"
    path = base / "UPCT" / f"UPCT CAMPUS PASEO {pollutant} (ugm3).csv"
    _write_upct_csv(path, pollutant)

    out = load_raw_5m(pollutant, str(base))

    assert len(out) == 1
    series_name, df = out[0]
    assert series_name == f"UPCT_{pollutant}"
    assert list(df.columns) == [pollutant]
    assert list(df.index) == list(
        pd.DatetimeIndex(["2024-01-01 00:00:00", "2024-01-01 00:05:00"])
    )
    assert df.iloc[:, 0].tolist() == pytest.approx([18.31, 16.43])


def test_load_raw_5m_skips_empty_frames(tmp_path) -> None:
    base = tmp_path / "raw"
    (base / "StationA").mkdir(parents=True)
    (base / "StationA" / "StationA_NO2.csv").write_text("fecha,value\n", encoding="utf-8")
    _write_5m_csv(base / "StationB" / "StationB_NO2.csv", [("2024-01-01 00:00:00", 1.0)])

    out = load_raw_5m("NO2", str(base))

    assert [station for station, _ in out] == ["StationB"]


def test_load_raw_5m_sorts_rows_within_station(tmp_path) -> None:
    base = tmp_path / "raw"
    _write_5m_csv(
        base / "StationA" / "StationA_NO2.csv",
        [("2024-01-01 00:05:00", 2.0), ("2024-01-01 00:00:00", 1.0)],
    )

    (_, df), = load_raw_5m("NO2", str(base))

    assert df.index.is_monotonic_increasing
    assert df.iloc[0, 0] == 1.0


def test_load_raw_5m_coerces_non_datetime_index(tmp_path) -> None:
    # Mixed index: one valid timestamp, one junk row that must be dropped.
    base = tmp_path / "raw"
    _write_5m_csv(
        base / "StationA" / "StationA_NO2.csv",
        [("2024-01-01 00:00:00", 1.0), ("notadate", 2.0)],
    )

    (_, df), = load_raw_5m("NO2", str(base))

    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 1
    assert df.iloc[0, 0] == 1.0
