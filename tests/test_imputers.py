"""Tests the unified GapImputer adapters and the imputer registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.imputation.imputers import (
    GapImputer,
    InterpolationGapImputer,
    LinearGapImputer,
    PROPHET_AVAILABLE,
    ProphetGapImputer,
)
from airquality.imputation.registry import (
    available_imputer_names,
    resolve_imputer_family,
    resolve_imputer_names,
    DARTS_GLOBAL,
    LINEAR,
    PROPHET,
    TSPULSE,
)


def test_registry_classifies_known_names() -> None:
    assert resolve_imputer_family("TiDE") == DARTS_GLOBAL
    assert resolve_imputer_family("Prophet") == PROPHET
    assert resolve_imputer_family("TSPulse") == TSPULSE
    assert resolve_imputer_family("TSPulse_FineTuned") == TSPULSE
    # The linear baseline has its own family, decoupled from the seasonal `interp`.
    assert resolve_imputer_family("LinearInterp") == LINEAR


def test_linear_gap_imputer_is_literal_linear_interpolation() -> None:
    imputer = LinearGapImputer()
    assert isinstance(imputer, GapImputer)

    idx = pd.date_range("2020-01-01", periods=6, freq="h")
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=idx, name="s")

    # Isolated size-1 gap -> exactly the mean of the two neighbours.
    masked_1 = series.copy()
    masked_1.iloc[2] = np.nan
    pred_1, failures_1 = imputer.impute_gaps(
        series_name="s",
        all_series_map={"s": masked_1},
        gap_windows=[pd.DatetimeIndex([idx[2]])],
        test_index=idx,
        freq="h",
    )
    assert failures_1 == []
    assert pred_1.iloc[0] == pytest.approx(30.0)

    # Size-2 block gap -> straight line between the endpoints (20 -> 50).
    masked_2 = series.copy()
    masked_2.iloc[2] = np.nan
    masked_2.iloc[3] = np.nan
    pred_2, _ = imputer.impute_gaps(
        series_name="s",
        all_series_map={"s": masked_2},
        gap_windows=[pd.DatetimeIndex([idx[2], idx[3]])],
        test_index=idx,
        freq="h",
    )
    assert pred_2.to_numpy() == pytest.approx([30.0, 40.0])


def test_linear_gap_imputer_remasks_gap_to_avoid_leakage() -> None:
    # `all_series_map` still holds the ground truth at the gap timestamps. The
    # imputer must re-mask them and interpolate, not read the truth back.
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    truth = pd.Series([10.0, 20.0, 999.0, 40.0, 50.0], index=idx, name="s")

    pred, _ = LinearGapImputer().impute_gaps(
        series_name="s",
        all_series_map={"s": truth},  # unmasked on purpose
        gap_windows=[pd.DatetimeIndex([idx[2]])],
        test_index=idx,
        freq="h",
    )
    assert pred.iloc[0] == pytest.approx(30.0)  # linear, not the 999.0 ground truth


def test_interpolation_gap_imputer_remasks_gap_to_avoid_leakage() -> None:
    # Same contract as LinearGapImputer: `all_series_map` holds the ground truth
    # at the gap timestamps and must be re-masked, otherwise the benchmark would
    # score the truth read back (near-perfect fake metrics).
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    truth = pd.Series([10.0, 20.0, 999.0, 40.0, 50.0], index=idx, name="s")

    pred, failures = InterpolationGapImputer().impute_gaps(
        series_name="s",
        all_series_map={"s": truth},  # unmasked on purpose
        gap_windows=[pd.DatetimeIndex([idx[2]])],
        test_index=idx,
        freq="h",
    )
    assert failures == []
    # One observation per hour-of-day: after re-masking, the climatology at the
    # gap hour is NaN and the fill falls back to linear interpolation (30.0).
    assert pred.iloc[0] == pytest.approx(30.0)  # not the 999.0 ground truth


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


def test_tspulse_gap_imputer_uses_model_reconstruction_at_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test for the critical benchmark bug: the mask must reach the
    # official pipeline as NaN. The stub mimics the official postprocess
    # (`out.where(~out.isna(), reconstruction)`): if the frame arrived pre-filled
    # the output would be the pre-fill, never the model reconstruction.
    from airquality.imputation import imputers as imputers_mod

    reconstruction_value = 123.0
    captured: dict[str, pd.DataFrame] = {}

    class StubPreprocessor:
        num_input_channels = 1

        def __init__(self, **kwargs: object) -> None:
            pass

        def train(self, df: pd.DataFrame) -> None:
            pass

    class StubPipeline:
        def __init__(self, model: object, **kwargs: object) -> None:
            pass

        def __call__(self, prepared: pd.DataFrame) -> pd.DataFrame:
            captured["prepared"] = prepared.copy()
            value = prepared["value"]
            return pd.DataFrame(
                {
                    "timestamp": prepared["timestamp"],
                    "value": value,
                    "value_imputed": value.where(value.notna(), reconstruction_value),
                }
            )

    monkeypatch.setattr(imputers_mod, "TSFM_PUBLIC_AVAILABLE", True)
    monkeypatch.setattr(imputers_mod, "TimeSeriesPreprocessor", StubPreprocessor)
    monkeypatch.setattr(imputers_mod, "TimeSeriesImputationPipeline", StubPipeline)

    index = pd.date_range("2024-01-01", periods=72, freq="h")
    truth = pd.Series(
        10.0 + 3.0 * np.sin(np.arange(72) * 2 * np.pi / 24), index=index, name="S"
    )
    gap = pd.DatetimeIndex(index[[60, 61, 62]])

    imputer = imputers_mod.TSPulseGapImputer(
        context_length=64, device="cpu", model=object()
    )
    pred, failures = imputer.impute_gaps(
        series_name="S",
        all_series_map={"S": truth},
        gap_windows=[gap],
        test_index=index[48:],
        freq="h",
    )

    assert failures == []
    # The frame handed to the pipeline keeps NaN exactly at the mask.
    prepared = captured["prepared"].set_index("timestamp")["value"]
    assert prepared.loc[gap].isna().all()
    assert prepared.drop(gap).notna().all()
    # And the prediction at the mask is the model reconstruction, not a pre-fill.
    assert pred.to_numpy() == pytest.approx([reconstruction_value] * 3)


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
