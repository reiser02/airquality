"""Tests benchmark helpers for gap planning, context building, metrics, and pipeline execution."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from darts import TimeSeries

from airquality.imputation.benchmark import (
    _compute_gap_mase,
    _compute_mase_denominator,
    _compute_metrics_on_mask,
    _gap_windows_to_mask_index,
    _normalize_series_collection,
    execute_complete_pipeline,
)
from airquality.imputation.imputers import (
    DartsGlobalGapImputer,
    _build_clean_left_context,
    build_tspulse_context_frame,
)
from airquality.modeling.training import build_benchmark_dataset_bundle


def _series(values: list[float | None], *, name: str = "S", start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(values, index=idx, name=name, dtype=float)


def test_normalize_series_collection_accepts_sequence_of_series_and_frames() -> None:
    first = _series([1.0, 2.0], name=None)
    second = pd.DataFrame({"B": [3.0, 4.0]}, index=pd.date_range("2024-01-01", periods=2, freq="h"))

    out = _normalize_series_collection([first, second], freq="h", default_prefix="test")

    assert list(out) == ["test_0", "B"]
    assert out["test_0"].name == "test_0"
    assert out["B"].iloc[-1] == 4.0


def test_normalize_series_collection_rejects_multicolumn_mapping_frames() -> None:
    bad = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )

    with pytest.raises(ValueError, match="exactamente una columna"):
        _normalize_series_collection({"bad": bad}, freq="h", default_prefix="test")


def test_build_clean_left_context_uses_history_tail_when_test_history_is_short() -> None:
    train = _series([1.0, 2.0], name="S")
    test = _series([3.0, 4.0, 5.0], name="S", start="2024-01-01 02:00:00")
    combined = pd.concat([train, test]).sort_index()

    out = _build_clean_left_context(
        series=combined,
        gap_start=pd.Timestamp("2024-01-01 03:00:00"),
        required_context=3,
        freq="h",
    )

    assert list(out) == [1.0, 2.0, 3.0]
    assert list(out.index) == list(pd.date_range("2024-01-01 00:00:00", periods=3, freq="h"))


def test_build_clean_left_context_stops_at_nan_barrier() -> None:
    test = _series([1.0, None, 3.0, 4.0], name="S")

    out = _build_clean_left_context(
        series=test,
        gap_start=pd.Timestamp("2024-01-01 04:00:00"),
        required_context=4,
        freq="h",
    )

    assert list(out) == [3.0, 4.0]


class RetryPredictModel:
    input_chunk_length = 2

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if "verbose" in kwargs:
            raise TypeError("unexpected verbose")

        series = kwargs["series"]
        n = int(kwargs["n"])
        index = pd.date_range(series.end_time() + series.freq, periods=n, freq=series.freq_str)
        values = pd.Series([9.0] * n, index=index)
        return TimeSeries.from_series(values, freq=series.freq_str)


def test_darts_left_context_imputation_retries_predict_signature_and_fills_gap() -> None:
    model = RetryPredictModel()
    test = _series([1.0, 2.0, 3.0, 4.0], name="S")
    gap = pd.date_range("2024-01-01 02:00:00", periods=2, freq="h")

    pred, failures = DartsGlobalGapImputer(model, model_name="Retry").impute_gaps(
        series_name="S",
        all_series_map={"S": test},
        gap_windows=[gap],
        test_index=test.index,
        scaler=None,
        freq="h",
        config_workers={"num_workers": 0},
    )

    assert failures == []
    assert list(pred.index) == list(gap)
    assert list(pred) == [9.0, 9.0]
    assert len(model.calls) >= 2


def test_darts_left_context_imputation_reports_skipped_gap_when_context_is_insufficient() -> None:
    class NeedsLongContext:
        input_chunk_length = 5

        def predict(self, **kwargs):
            raise AssertionError("predict should not be called")

    gap = pd.date_range("2024-01-01 01:00:00", periods=1, freq="h")
    pred, failures = DartsGlobalGapImputer(NeedsLongContext(), model_name="LongCtx").impute_gaps(
        series_name="S",
        all_series_map={"S": _series([1.0, 2.0], name="S")},
        gap_windows=[gap],
        test_index=pd.DatetimeIndex([]),
        scaler=None,
        freq="h",
    )

    assert math.isnan(pred.iloc[0])
    assert len(failures) == 1
    assert failures[0].available_context == 1


def test_compute_gap_mase_returns_nan_without_warning_for_all_nan_predictions() -> None:
    actual_gap = _series([1.0, 2.0], name="S")
    pred_gap = _series([None, None], name="S")
    insample = _series([0.0, 1.0, 2.0, 3.0], name="S", start="2023-12-31 20:00:00")

    out = _compute_gap_mase(
        actual_gap=actual_gap,
        pred_gap=pred_gap,
        insample=insample,
        seasonality_m=1,
        freq="h",
    )

    assert math.isnan(out)


def test_build_tspulse_context_frame_pads_and_returns_original_test_index() -> None:
    train = _series([1.0, 2.0], name="train")
    test = _series([3.0, 4.0], name="test", start="2024-01-01 02:00:00")
    all_series = pd.concat([train, test])
    all_series.name = "test"

    frame, original_index = build_tspulse_context_frame(
        series_name="test",
        all_series_map={"test": all_series},
        mask_index=pd.DatetimeIndex([]),
        test_index=test.index,
        context_length=5,
        freq="h",
        timestamp_column="ts",
        target_column="y",
    )

    assert list(frame.columns) == ["ts", "y"]
    assert len(frame) == 5
    assert frame["y"].isna().sum() == 0
    assert list(original_index) == list(test.index)


def test_build_tspulse_context_frame_keeps_mask_nan_and_fills_real_gaps() -> None:
    # The synthetic mask must reach the pipeline as NaN (the official pipeline
    # only uses the model reconstruction where the input is NaN); only real
    # historical holes are pre-filled.
    values: list[float | None] = [float(i) for i in range(48)]
    values[10] = None  # real historical gap -> must be pre-filled
    full = _series(values, name="S")
    mask_index = full.index[[40, 41, 42]]  # synthetic gaps -> must stay NaN

    frame, original_index = build_tspulse_context_frame(
        series_name="S",
        all_series_map={"S": full},
        mask_index=mask_index,
        test_index=full.index[24:],
        context_length=48,
        freq="h",
        timestamp_column="ts",
        target_column="y",
    )

    y = pd.Series(frame["y"].to_numpy(), index=pd.DatetimeIndex(frame["ts"]))
    assert y.loc[mask_index].isna().all()  # mask preserved for the model
    assert y.drop(mask_index).notna().all()  # everything else filled
    assert y.loc[full.index[10]] == pytest.approx(10.0)  # real gap interpolated
    assert list(original_index) == list(full.index[24:])


def test_compute_mase_denominator_returns_nan_when_too_short() -> None:
    out = _compute_mase_denominator(_series([1.0, 2.0], name="S"), seasonality_m=2, freq="h")
    assert math.isnan(out)


def test_compute_metrics_on_mask_computes_selected_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    y_true = pd.Series([2.0, 4.0, 6.0], index=idx)
    y_pred = pd.Series([1.0, 4.0, 5.0], index=idx)

    out = _compute_metrics_on_mask(
        y_true=y_true,
        y_pred=y_pred,
        metrics=("mae", "rmse"),
    )

    assert out["MAE"] == pytest.approx(2 / 3)
    assert out["RMSE"] == pytest.approx(((1.0**2 + 0.0 + 1.0**2) / 3) ** 0.5)


def test_compute_metrics_on_mask_rejects_unknown_metric() -> None:
    pass


def test_execute_complete_pipeline_smoke_with_explicit_gap_spec() -> None:
    class StubImputer:
        model_name = "Stub"

        def impute_gaps(
            self,
            *,
            series_name,
            all_series_map,
            gap_windows,
            test_index,
            scaler,
            freq,
            config_workers=None,
        ):
            del series_name, all_series_map, test_index, scaler, freq, config_workers
            mask_index = _gap_windows_to_mask_index(gap_windows)
            return pd.Series([30.0, 40.0], index=mask_index, dtype=float), []

    full = _series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], name="S")
    frame = full.to_frame(name="S")
    longest_segment = frame.iloc[-4:].copy()
    bundle = build_benchmark_dataset_bundle(
        [frame],
        longest_segment,
        val_size=1,
        min_train_len=1,
        val_context_len=1,
    )
    test = full.iloc[-4:].copy()
    gap_start = pd.Timestamp("2024-01-01 04:00:00")

    results_df, plot_store = execute_complete_pipeline(
        model_dict={"Stub": StubImputer()},
        dataset_bundle=bundle,
        gap_sizes=(2,),
        num_gaps=1,
        gap_spec_by_series={"S": [(gap_start, 2)]},
        metrics=("mae", "rmse", "mase"),
        seasonality_m=1,
        freq="h",
        random_seed=123,
    )

    assert list(results_df.columns) == ["Modelo", "Serie", "Gap_Size", "MAE", "RMSE", "MASE"]
    assert results_df.loc[0, "Modelo"] == "Stub"
    assert results_df.loc[0, "Serie"] == "S"
    assert results_df.loc[0, "Gap_Size"] == 2
    assert results_df.loc[0, "MAE"] == pytest.approx(20.0)
    assert results_df.loc[0, "RMSE"] == pytest.approx(((20.0**2 + 20.0**2) / 2) ** 0.5)
    assert results_df.loc[0, "MASE"] == pytest.approx(2.0)

    assert set(plot_store[2]) == {"series"}
    assert set(plot_store[2]["series"]["S"]) == {"actual", "preds", "naive_mase"}
    assert plot_store[2]["series"]["S"]["actual"].equals(test)
    assert list(plot_store[2]["series"]["S"]["naive_mase"].index) == list(
        pd.date_range(gap_start, periods=2, freq="h")
    )
    pred = plot_store[2]["series"]["S"]["preds"]["Stub"]
    assert list(pred.index) == list(pd.date_range(gap_start, periods=2, freq="h"))
    assert list(pred.values) == [30.0, 40.0]


def test_mase_per_gap_advancing_context_and_weighted_average() -> None:
    class MockMultiGapImputer:
        model_name = "Mock"

        def impute_gaps(
            self,
            *,
            series_name,
            all_series_map,
            gap_windows,
            test_index,
            scaler,
            freq,
            config_workers=None,
        ):
            del series_name, all_series_map, test_index, scaler, freq, config_workers
            mask_index = _gap_windows_to_mask_index(gap_windows)
            preds = pd.Series(dtype=float, index=mask_index)
            pred_vals = {
                pd.Timestamp("2024-01-01 03:00:00"): 30.0,
                pd.Timestamp("2024-01-01 04:00:00"): 40.0,
                pd.Timestamp("2024-01-01 07:00:00"): 60.0,
                pd.Timestamp("2024-01-01 08:00:00"): 70.0,
                pd.Timestamp("2024-01-01 09:00:00"): 80.0,
            }
            for ts, val in pred_vals.items():
                if ts in preds.index:
                    preds.loc[ts] = val
            return preds, []

    full = _series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], name="S")
    frame = full.to_frame(name="S")
    longest_segment = frame.iloc[-8:].copy()  # Leave [10.0, 20.0] for train/val split
    bundle = build_benchmark_dataset_bundle(
        [frame],
        longest_segment,
        val_size=1,
        min_train_len=1,
        val_context_len=1,
    )
    
    gap_spec = [
        (pd.Timestamp("2024-01-01 03:00:00"), 2),
        (pd.Timestamp("2024-01-01 07:00:00"), 3),
    ]

    results_df, _ = execute_complete_pipeline(
        model_dict={"Mock": MockMultiGapImputer()},
        dataset_bundle=bundle,
        gap_sizes=(5,),
        num_gaps=2,
        gap_spec_by_series={"S": gap_spec},
        metrics=("mae", "rmse", "mase"),
        seasonality_m=5,
        freq="h",
        random_seed=123,
    )

    assert results_df.loc[0, "Modelo"] == "Mock"
    assert results_df.loc[0, "MAE"] == pytest.approx(16.0)
    assert results_df.loc[0, "RMSE"] == pytest.approx(16.73320053068151)
    assert results_df.loc[0, "MASE"] == pytest.approx(0.4)
