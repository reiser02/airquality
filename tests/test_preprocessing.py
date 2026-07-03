"""Tests del preprocesamiento de 5 minutos a media horaria y el filtrado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.data.preprocessing import (
    DETECTION_LIMITS,
    frozen_mask,
    hourly_mean,
    preprocess,
)


def _series_5m(values: list[float], start: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    """Construye un df de 5 minutos con una sola columna 'NO2'."""
    idx = pd.date_range(start, periods=len(values), freq="5min")
    return pd.DataFrame({"NO2": values}, index=idx)


# --- frozen_mask ---------------------------------------------------------


def test_frozen_mask_flags_runs_at_or_above_min_run() -> None:
    s = pd.Series([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0])  # seis 2.0 seguidos
    mask = frozen_mask(s, min_run=6)
    assert mask.tolist() == [False, True, True, True, True, True, True, False]


def test_frozen_mask_ignores_short_runs() -> None:
    s = pd.Series([2.0, 2.0, 2.0, 3.0])  # solo tres iguales, min_run=6
    assert not frozen_mask(s, min_run=6).any()


def test_frozen_mask_skips_nan_when_measuring_run() -> None:
    # Los NaN no rompen el tramo: siguen siendo seis 2.0 "consecutivos".
    s = pd.Series([2.0, 2.0, np.nan, 2.0, 2.0, np.nan, 2.0, 2.0])
    assert int(frozen_mask(s, min_run=6).sum()) == 6


# --- hourly_mean ---------------------------------------------------------


def test_hourly_mean_averages_only_useful_readings() -> None:
    # 6 utiles (>=3.762) y 6 basura: media solo de las utiles.
    vals = [8.52, 6.79, 7.40, 5.81, 4.09, 3.90, 1.0, 1.0, 1.0, 1.88, 1.1, 1.88]
    out = hourly_mean(_series_5m(vals), "NO2")["NO2"]
    expected = np.mean([8.52, 6.79, 7.40, 5.81, 4.09, 3.90])
    assert out.notna().sum() == 1
    assert out.iloc[0] == round(expected, 10) or abs(out.iloc[0] - expected) < 1e-9


def test_hourly_mean_nan_when_fewer_than_min_useful() -> None:
    # Solo 2 lecturas por encima del umbral -> NaN (min_useful=3).
    vals = [10.0, 10.0] + [1.0] * 10
    out = hourly_mean(_series_5m(vals), "NO2")["NO2"]
    assert out.isna().all()


def test_hourly_mean_frozen_counts_as_below_threshold() -> None:
    # 12 lecturas iguales y altas, pero congeladas -> se ponen a 0 -> NaN.
    out = hourly_mean(_series_5m([50.0] * 12), "NO2")["NO2"]
    assert out.isna().all()


def test_hourly_mean_excludes_below_threshold_from_average() -> None:
    # 3 utiles exactos; los valores bajo umbral no entran en la media.
    vals = [10.0, 20.0, 30.0] + [1.0] * 9
    out = hourly_mean(_series_5m(vals), "NO2")["NO2"]
    assert out.iloc[0] == 20.0  # media de 10,20,30 (no de los 1.0)


def test_hourly_mean_output_never_below_threshold() -> None:
    # Invariante que hace innecesario el filtro de umbral en preprocess:
    # todo valor no-NaN que sale de hourly_mean es >= umbral.
    vals = [4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = hourly_mean(_series_5m(vals), "NO2")["NO2"].dropna()
    assert (out >= DETECTION_LIMITS["NO2"]).all()


# --- preprocess (extremo a extremo) -------------------------------------


def test_preprocess_rescues_hour_a_simple_mean_would_drop() -> None:
    # Objetivo central: 3 lecturas altas + 9 ceros. La media simple (30/12=2.5)
    # cae bajo umbral y datos_estaciones la eliminaria; la media inteligente
    # (solo las 3 utiles = 10.0) la conserva.
    vals = [10.0, 10.0, 10.0] + [0.0] * 9
    (out,), (count,) = preprocess([_series_5m(vals)], "NO2")
    col = out[out.columns[0]].dropna()
    assert col.tolist() == [10.0]
    assert count == 0


def test_preprocess_drops_consecutive_frozen_hours() -> None:
    # Hora 1: 6 utiles -> media. Hora 2: misma media (congelado horario).
    block = [8.52, 6.79, 7.40, 5.81, 4.09, 3.90, 1.0, 1.0, 1.0, 1.88, 1.1, 1.88]
    df = _series_5m(block * 2)  # dos horas identicas
    (out,), (count,) = preprocess([df], "NO2")
    col = out[out.columns[0]].dropna()
    # La segunda hora (repeticion) cae; no quedan duplicados consecutivos.
    assert (col == col.shift()).sum() == 0
    assert count == 1


def test_detection_limits_present() -> None:
    assert set(DETECTION_LIMITS) == {"CO", "NO2"}


# --- frozen_mask (ramas extra) ------------------------------------------


def test_frozen_mask_all_nan_returns_empty_mask() -> None:
    # Serie sin valores validos -> rama valid.empty -> mascara todo False.
    mask = frozen_mask(pd.Series([np.nan, np.nan, np.nan]), min_run=6)
    assert not mask.any()
    assert len(mask) == 3


def test_frozen_mask_run_one_short_of_min_not_flagged() -> None:
    # Cinco 2.0 con min_run=6 (justo por debajo del umbral) -> no se marca.
    s = pd.Series([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0])
    assert not frozen_mask(s, min_run=6).any()


# --- hourly_mean (otros contaminantes / errores / rejilla) --------------


def test_hourly_mean_uses_co_threshold() -> None:
    # CO usa umbral 6.0: solo 6.0/8.0/10.0 son utiles; los 1.0 quedan fuera.
    vals = [6.0, 8.0, 10.0] + [1.0] * 9
    out = hourly_mean(_series_5m(vals), "CO")["NO2"]
    assert out.iloc[0] == 8.0  # media de 6,8,10


def test_hourly_mean_unknown_pollutant_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        hourly_mean(_series_5m([10.0] * 12), "O3")


def test_hourly_mean_preserves_regular_hourly_grid() -> None:
    # Hora 0 util + hora 1 totalmente ausente + hora 2 util.
    block = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
    idx0 = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
    idx2 = pd.date_range("2024-01-01 02:00", periods=12, freq="5min")
    df = pd.DataFrame({"NO2": block + block}, index=idx0.append(idx2))

    out = hourly_mean(df, "NO2")["NO2"]

    expected_index = pd.date_range("2024-01-01 00:00", periods=3, freq="h")
    assert list(out.index) == list(expected_index)
    assert pd.isna(out.loc["2024-01-01 01:00"])  # la hora ausente se conserva como NaN
    assert out.notna().sum() == 2


# --- preprocess (multi-estacion) ----------------------------------------


def test_preprocess_returns_per_station_frozen_counts() -> None:
    block = [8.52, 6.79, 7.40, 5.81, 4.09, 3.90, 1.0, 1.0, 1.0, 1.88, 1.1, 1.88]
    df_frozen = _series_5m(block * 2)  # segunda hora repetida -> 1 congelado
    df_clean = _series_5m([10.0, 20.0, 30.0] + [1.0] * 9)  # una hora, sin repeticion

    processed, counts = preprocess([df_frozen, df_clean], "NO2")

    assert len(processed) == 2
    assert counts == [1, 0]
