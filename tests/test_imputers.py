"""Tests the unified GapImputer adapters and the imputer registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.imputation.imputers import (
    GapImputer,
    PROPHET_AVAILABLE,
    ProphetGapImputer,
)
from airquality.imputation.registry import (
    available_imputer_names,
    resolve_imputer_family,
    resolve_imputer_names,
    DARTS_GLOBAL,
    PROPHET,
    TSPULSE,
)


def test_registry_classifies_known_names() -> None:
    assert resolve_imputer_family("TiDE") == DARTS_GLOBAL
    assert resolve_imputer_family("Prophet") == PROPHET
    assert resolve_imputer_family("TSPulse") == TSPULSE
    assert resolve_imputer_family("TSPulse_FineTuned") == TSPULSE


def test_registry_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        resolve_imputer_family("NotAModel")


def test_resolve_imputer_names_expands_all_and_validates() -> None:
    all_names = resolve_imputer_names(["all"])
    assert all_names == available_imputer_names()
    assert "TiDE" in all_names

    subset = resolve_imputer_names(["NLinear", "TiDE"])
    assert subset == ["NLinear", "TiDE"]

    with pytest.raises(ValueError, match="Unknown imputer name"):
        resolve_imputer_names(["Bogus"])


@pytest.mark.skipif(not PROPHET_AVAILABLE, reason="darts Prophet unavailable")
def test_prophet_gap_imputer_fills_gap_with_finite_values() -> None:
    imputer = ProphetGapImputer(model_name="Prophet")
    assert isinstance(imputer, GapImputer)

    index = pd.date_range("2024-01-01", periods=72, freq="h")
    values = 10.0 + 3.0 * np.sin(np.arange(72) * 2 * np.pi / 24)
    full = pd.Series(values, index=index, name="S")

    gap = pd.date_range("2024-01-03 00:00:00", periods=3, freq="h")  # leaves 48h of left context
    pred, failures = imputer.impute_gaps(
        series_name="S",
        all_series_map={"S": full},
        gap_windows=[gap],
        test_index=index,
        scaler=None,
        freq="h",
        config_workers=None,
    )

    assert failures == []
    assert list(pred.index) == list(gap)
    assert np.isfinite(pred.to_numpy(dtype=float)).all()


@pytest.mark.skipif(not PROPHET_AVAILABLE, reason="darts Prophet unavailable")
def test_prophet_gap_imputer_reports_failure_without_enough_context() -> None:
    imputer = ProphetGapImputer(model_name="Prophet", min_context=2)

    full = pd.Series(
        [1.0],
        index=pd.date_range("2024-01-01", periods=1, freq="h"),
        name="S",
    )
    gap = pd.date_range("2024-01-01 01:00:00", periods=1, freq="h")

    pred, failures = imputer.impute_gaps(
        series_name="S",
        all_series_map={"S": full},
        gap_windows=[gap],
        test_index=full.index,
        scaler=None,
        freq="h",
        config_workers=None,
    )

    assert len(failures) == 1
    assert failures[0].model_name == "Prophet"
    assert np.isnan(pred.iloc[0])
