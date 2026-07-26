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
    captured: dict[str, object] = {"calls": 0}

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
            captured["calls"] = int(captured["calls"]) + 1
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

    index = pd.date_range("2024-01-01", periods=1300, freq="h")
    truth = pd.Series(
        10.0 + 3.0 * np.sin(np.arange(len(index)) * 2 * np.pi / 24),
        index=index,
        name="S",
    )
    old_gap = pd.DatetimeIndex(index[[600, 601, 602]])
    recent_gap = pd.DatetimeIndex(index[[1200, 1201, 1202]])
    mask = old_gap.append(recent_gap)

    imputer = imputers_mod.TSPulseGapImputer(
        context_length=512, device="cpu", model=object()
    )
    pred, failures = imputer.impute_gaps(
        series_name="S",
        all_series_map={"S": truth},
        gap_windows=[old_gap, recent_gap],
        test_index=index[512:],
        freq="h",
    )

    assert failures == []
    assert captured["calls"] == 1  # pooled masks use one sliding-pipeline call
    prepared = captured["prepared"]
    assert isinstance(prepared, pd.DataFrame)
    prepared_values = prepared.set_index("timestamp")["value"]
    assert len(prepared_values) > imputer.context_length
    assert prepared_values.index[0] == index[89]  # 511 points before the old gap
    assert prepared_values.index[-1] == index[-1]
    assert prepared_values.loc[mask].isna().all()
    pd.testing.assert_series_equal(
        prepared_values.drop(mask), truth.reindex(prepared_values.index).drop(mask),
        check_names=False,
    )
    # Both old and recent gaps come from reconstruction, not observed truth.
    assert list(pred.index) == list(mask)
    assert pred.to_numpy() == pytest.approx([reconstruction_value] * len(mask))


def test_tspulse_gap_imputer_rejects_uncovered_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from airquality.imputation import imputers as imputers_mod

    index = pd.date_range("2024-01-01", periods=20, freq="h")
    mask = pd.DatetimeIndex(index[[5, 15]])
    imputer = imputers_mod.TSPulseGapImputer(
        context_length=8, device="cpu", model=object()
    )
    monkeypatch.setattr(
        imputer,
        "_impute_full_series",
        lambda **_: pd.Series(1.0, index=mask[-1:]),
    )

    with pytest.raises(RuntimeError, match="no values.*mask"):
        imputer.impute_gaps(
            series_name="S",
            all_series_map={"S": pd.Series(range(20), index=index, dtype=float)},
            gap_windows=[mask],
            test_index=index,
            freq="h",
        )


def test_tspulse_gap_imputer_does_not_count_lazy_load_as_imputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from airquality.imputation import imputers as imputers_mod

    index = pd.date_range("2024-01-01", periods=3, freq="h")
    gap = pd.DatetimeIndex(index[1:2])
    imputer = imputers_mod.TSPulseGapImputer(context_length=2, device="cpu")

    def fake_impute(**_: object) -> pd.Series:
        if imputer.model is None:
            imputer.model = object()
            imputer.train_seconds = 3.0
        return pd.Series(1.0, index=gap)

    class Clock:
        values = iter((0.0, 10.0, 20.0, 24.0))

        @classmethod
        def perf_counter(cls) -> float:
            return next(cls.values)

    monkeypatch.setattr(imputer, "_impute_full_series", fake_impute)
    monkeypatch.setattr(imputers_mod, "time", Clock)

    kwargs = {
        "series_name": "S",
        "all_series_map": {"S": pd.Series(range(3), index=index, dtype=float)},
        "gap_windows": [gap],
        "test_index": index,
        "freq": "h",
    }
    imputer.impute_gaps(**kwargs)
    assert imputer.train_seconds == pytest.approx(3.0)
    assert imputer._last_impute_seconds == pytest.approx(7.0)

    imputer.impute_gaps(**kwargs)
    assert imputer._last_impute_seconds == pytest.approx(4.0)


def test_tspulse_model_setup_time_includes_device_transfer(monkeypatch) -> None:
    from airquality.imputation import imputers as imputers_mod

    class Model:
        def __init__(self) -> None:
            self.device = None
            self.evaluating = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluating = True
            return self

    model = Model()
    monkeypatch.setattr(imputers_mod, "TSFM_PUBLIC_AVAILABLE", True)
    monkeypatch.setattr(
        imputers_mod.TSPulseForReconstruction,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )
    imputer = imputers_mod.TSPulseGapImputer(context_length=2, device="cpu")

    assert imputer._ensure_model(1) is model
    assert model.device == "cpu"
    assert model.evaluating
    assert imputer.train_seconds >= 0.0


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
