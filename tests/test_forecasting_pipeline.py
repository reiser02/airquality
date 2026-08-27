"""Tests for the multi-arm forecasting benchmark pipeline."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

import airquality.forecasting.backtest as backtest_module
import airquality.forecasting.fill as fill
import airquality.forecasting.pipeline as cp
from darts import TimeSeries

from _forecasting_helpers import BASELINE_DETECTORS, _seasonal_series
from airquality.forecasting.backtest import (
    backtest_forecast,
    select_holdout_window,
    split_train_val_subseries,
)
from airquality.forecasting.cleaning import remove_anomalies
from airquality.forecasting.detection import DetectionResult
from airquality.forecasting.fill import build_imputer, impute_series, nan_gap_windows
from airquality.imputation.registry import resolve_imputer_family

# --------------------------------------------------------------------------- #
# Anomaly removal
# --------------------------------------------------------------------------- #
def test_remove_anomalies_masks_flagged_timestamps():
    series = _seasonal_series(seed=1)
    mask = pd.Series(False, index=series.index)
    mask.iloc[[300, 500]] = True
    result = DetectionResult("test", [], [], {}, 3.5, mask)

    cleaned = remove_anomalies(series, result)
    assert np.isnan(cleaned.iloc[300]) and np.isnan(cleaned.iloc[500])
    assert int(cleaned.isna().sum()) >= int(series.isna().sum()) + 2


def test_contiguous_observed_segments_splits_on_gaps():
    from airquality.data.segments import contiguous_observed_segments

    series = _seasonal_series(n=60)
    series.iloc[10:15] = np.nan  # gap -> two runs of 10 and 45
    series.iloc[57] = np.nan  # short tail run of 2

    segments = contiguous_observed_segments(series, min_len=3)

    assert [len(seg) for seg in segments] == [10, 42]
    assert all(not seg.isna().any() for seg in segments)
    # min_len filters the 2-point tail run
    assert contiguous_observed_segments(series, min_len=1)[-1].index[0] == series.index[58]


def test_common_detection_support_masks_every_strategy_timestamp():
    series = _seasonal_series(n=20)
    first = pd.Series(False, index=series.index)
    second = pd.Series(False, index=series.index)
    first.iloc[5] = True
    second.iloc[12] = True
    detections = {
        name: DetectionResult(name, [], [], {}, 3.5, mask)
        for name, mask in (("first", first), ("second", second))
    }

    support, common_mask = cp._common_detection_support(series, detections)

    assert common_mask[common_mask].index.tolist() == [series.index[5], series.index[12]]
    assert support.isna().iloc[[5, 12]].all()


def test_common_detection_support_keeps_unscored_timestamps():
    series = _seasonal_series(n=20)
    mask = pd.Series(False, index=series.index)
    scored = pd.Series(True, index=series.index)
    scored.iloc[7] = False
    detection = DetectionResult(
        "inject-vote", [], [], {}, 3.5, mask, scored_mask=scored, n_unscored=1
    )

    support, common_mask = cp._common_detection_support(
        series, {"inject-vote": detection}
    )

    assert not common_mask.any()
    assert support.iloc[7] == series.iloc[7]


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #
def test_repo_root_resolves_to_repository_root():
    # Regression: `parents[2]` from src/airquality/forecasting/ is `src/`, which
    # sent comparison artifacts to src/reports/ instead of reports/.
    root = cp._repo_root()
    assert (root / "pyproject.toml").exists()
    assert root.name != "src"


def test_load_raw_hourly_series_applies_preprocess_and_preserves_station(monkeypatch):
    index = pd.date_range("2024-01-01", periods=36, freq="5min")
    raw = pd.DataFrame({"NO2": 10.0 + np.arange(36)}, index=index)
    raw = raw.drop(index[12:24])
    monkeypatch.setattr(cp, "load_raw_5m", lambda pollutant, base_dir: [("Station A", raw)])

    (frame,) = cp._load_raw_hourly_series(
        pollutant="NO2", raw_base_dir="raw", freq="h"
    )

    assert frame.columns.tolist() == ["Station A"]
    assert len(frame) == 3
    assert pd.isna(frame.iloc[1, 0])


def test_load_raw_hourly_series_can_preserve_frozen_values(monkeypatch):
    index = pd.date_range("2024-01-01", periods=24, freq="5min")
    raw = pd.DataFrame(
        {"NO2": np.r_[np.full(12, 10.0), np.arange(12.0, 24.0)]},
        index=index,
    )
    monkeypatch.setattr(cp, "load_raw_5m", lambda *_args: [("Station A", raw)])

    (filtered,) = cp._load_raw_hourly_series(
        pollutant="NO2", raw_base_dir="raw", freq="h"
    )
    (preserved,) = cp._load_raw_hourly_series(
        pollutant="NO2", raw_base_dir="raw", freq="h", preserve_frozen=True
    )

    assert pd.isna(filtered.iloc[0, 0])
    assert preserved.iloc[0, 0] == 10.0


def test_load_raw_hourly_series_can_preserve_repeated_hourly_values(monkeypatch):
    index = pd.date_range("2024-01-01", periods=36, freq="5min")
    raw = pd.DataFrame(
        {"NO2": np.r_[np.arange(12.0), np.arange(11.0, -1.0, -1.0), np.arange(24.0, 36.0)]},
        index=index,
    )
    monkeypatch.setattr(cp, "load_raw_5m", lambda *_args: [("Station A", raw)])

    (filtered,) = cp._load_raw_hourly_series(
        pollutant="NO2", raw_base_dir="raw", freq="h"
    )
    (preserved,) = cp._load_raw_hourly_series(
        pollutant="NO2", raw_base_dir="raw", freq="h", preserve_frozen=True
    )

    assert pd.isna(filtered.iloc[1, 0])
    assert preserved.iloc[1, 0] == preserved.iloc[0, 0]


def test_series_by_name_rejects_duplicate_stations():
    series = _seasonal_series(n=20, name="ST0")

    with pytest.raises(RuntimeError, match="duplicado: ST0"):
        cp._series_by_name([series.to_frame(), series.to_frame()])


# --------------------------------------------------------------------------- #
# Imputation (interp family via the GapImputer interface)
# --------------------------------------------------------------------------- #
def test_interp_family_registered():
    assert resolve_imputer_family("interp") == "interp"


def test_nan_gap_windows_splits_contiguous_runs():
    series = _seasonal_series(n=50)
    series.iloc[5:9] = np.nan
    series.iloc[20:21] = np.nan
    windows = nan_gap_windows(series)
    assert [len(w) for w in windows] == [4, 1]


def test_impute_series_only_fills_complete_gaps_up_to_five():
    series = _seasonal_series(n=300)
    original = series.copy()
    series.iloc[5:10] = np.nan
    series.iloc[60:66] = np.nan

    imputer = build_imputer("interp", freq="h")
    filled = impute_series(series, imputer, freq="h")

    assert filled.iloc[5:10].notna().all()
    assert filled.iloc[60:66].isna().all()
    pd.testing.assert_series_equal(
        filled[series.notna()], original[series.notna()], check_names=False
    )


def test_build_imputer_routes_base_and_finetuned_tspulse(
    tmp_path, monkeypatch
):
    fine_tuned_path = tmp_path / "fine-tuned"
    fine_tuned_path.mkdir()
    captured: list[dict[str, object]] = []

    class DummyTSPulse:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    str_values = {
        ("tspulse", "model_id"): "base-model",
        ("tspulse", "revision"): "base-revision",
        ("tspulse", "finetuned_model_path"): "fine-tuned",
        ("tspulse", "device"): "cpu",
    }
    monkeypatch.setattr(fill, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "airquality.config.cfg_get_str",
        lambda section, option, default: str_values.get((section, option), default),
    )
    monkeypatch.setattr("airquality.config.cfg_get_int", lambda *args, **kwargs: 512)
    monkeypatch.setattr(
        "airquality.imputation.imputers.TSPulseGapImputer",
        DummyTSPulse,
    )

    build_imputer("TSPulse")
    build_imputer("TSPulse_FineTuned")

    assert captured[0]["model_path"] is None
    assert captured[1]["model_path"] == str(fine_tuned_path)


def test_build_imputer_rejects_finetuned_tspulse_without_path(monkeypatch):
    monkeypatch.setattr(
        "airquality.config.cfg_get_str",
        lambda section, option, default: ""
        if (section, option) == ("tspulse", "finetuned_model_path")
        else default,
    )

    with pytest.raises(RuntimeError, match="finetuned_model_path"):
        build_imputer("TSPulse_FineTuned")


# --------------------------------------------------------------------------- #
# Forecasting backtest
# --------------------------------------------------------------------------- #
def test_select_holdout_window_requires_contiguous_run():
    series = _seasonal_series(n=410)
    series.iloc[150:170] = np.nan  # host [0:150], test block [170:410]
    kwargs = dict(
        context_len=72,
        train_min_len=77,
        validation_len=48,
    )
    assert select_holdout_window(series, holdout=500, **kwargs) is None
    window = select_holdout_window(series, holdout=40, **kwargs)
    assert window is not None
    assert len(window["test_target_index"]) == 40
    assert len(window["test_index"]) == 112
    assert window["train_index"][-1] == window["test_target_start"] - pd.Timedelta(hours=1)
    assert window["source_run_start"] == series.index[170]
    assert window["test_context_start"] == window["context_index"][0]


def test_select_holdout_window_prefers_most_recent_run_over_longest():
    # A long early block then a shorter (still viable) recent block: the holdout
    # must land in the RECENT block so all the early history becomes training,
    # instead of stranding everything after the longest block.
    series = _seasonal_series(n=1000, seed=3)
    series.iloc[600:750] = np.nan  # long run [0:600], shorter recent run [750:1000]
    window = select_holdout_window(
        series,
        holdout=40,
        context_len=72,
        train_min_len=77,
        validation_len=48,
    )
    assert window is not None
    # Holdout ends at the tail of the recent run, not inside the longer early one.
    assert window["test_target_index"][-1] == series.index[999]
    assert window["test_target_start"] > series.index[750]


def test_select_holdout_window_uses_same_run_prefix_as_validation_host():
    kwargs = dict(
        holdout=40,
        context_len=72,
        train_min_len=77,
        validation_len=48,
    )
    assert select_holdout_window(_seasonal_series(n=160, seed=4), **kwargs) is None
    single_run = select_holdout_window(_seasonal_series(n=300, seed=4), **kwargs)
    assert single_run is not None
    assert single_run["source_run_start"] == single_run["train_index"][0]
    assert single_run["train_end"] == single_run["test_target_start"] - pd.Timedelta(hours=1)

    series = _seasonal_series(n=500, seed=4)
    series.iloc[150:170] = np.nan
    window = select_holdout_window(series, **kwargs)
    assert window is not None
    assert window["source_run_start"] == series.index[170]
    assert window["train_end"] == series.index[459]


def test_backtest_forecast_returns_finite_metrics():
    series = _seasonal_series(n=600, seed=2)
    series.iloc[200:240] = np.nan
    window = select_holdout_window(
        series,
        holdout=40,
        context_len=72,
        train_min_len=77,
        validation_len=48,
    )
    train = series.loc[window["train_index"]]
    test_series = series.loc[window["test_index"]]

    res = backtest_forecast(
        train, test_series, "LinearRegression",
        size_k=5, test_target_start=window["test_target_start"], seasonality_m=24,
        forecast_stride=2,
    )
    assert res["n_test_predictions"] > 0
    assert np.isfinite(res["_rmse"]) and np.isfinite(res["_mae"])
    assert np.isfinite(res["mase"]) and np.isfinite(res["rmsse"])
    assert res["n_forecasts"] > 1
    assert res["n_test_predictions"] > res["n_unique_targets"]


def test_train_val_split_keeps_train_before_validation():
    # A long early block followed by a shorter (still trainable) recent block:
    # the split must host validation in the RECENT block, never the long early
    # one, and no training point may be at/after the validation start.
    series = _seasonal_series(n=600, seed=7)
    series.iloc[400:430] = np.nan  # gap splits into [0:400] and [430:600]
    train_ts = TimeSeries.from_series(series, freq="h")

    input_chunk, size_k = 72, 8
    split = split_train_val_subseries(
        train_ts,
        input_chunk=input_chunk,
        size_k=size_k,
        validation_len=48,
        validation_stride=4,
    )
    assert split is not None
    train_subs, val_subs = split
    assert len(val_subs) == 11
    assert all(len(val) == input_chunk + size_k for val in val_subs)
    assert all(
        later.time_index[input_chunk] - earlier.time_index[input_chunk]
        == pd.Timedelta(hours=4)
        for earlier, later in zip(val_subs, val_subs[1:])
    )

    val_start = val_subs[0].start_time()
    val_block_start = val_subs[0].time_index[input_chunk]  # first validation target
    # Validation lives in the recent block (after the gap), not the long early one.
    assert val_start >= series.index[430]
    # Every training target precedes the validation targets: the only overlap
    # allowed is the val context (< val_block_start).
    for ts in train_subs:
        assert ts.end_time() < val_block_start

    long_split = split_train_val_subseries(
        train_ts,
        input_chunk=72,
        size_k=48,
        validation_len=96,
        validation_stride=24,
    )
    assert long_split is not None
    long_train, long_val = long_split
    assert len(long_val) == 3
    assert all(len(val) == 120 for val in long_val)
    assert all(ts.end_time() < long_val[0].time_index[72] for ts in long_train)


def test_train_val_split_returns_none_when_no_block_fits():
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    short = TimeSeries.from_series(pd.Series(np.arange(50.0), index=idx))
    assert split_train_val_subseries(short, input_chunk=72, size_k=5) is None


def test_backtest_forecast_scaled_metrics_scale_with_reference_amplitude():
    # Changing the reference affects only the scaled metrics, not pooled errors.
    series = _seasonal_series(n=600, seed=5)
    series.iloc[200:240] = np.nan
    window = select_holdout_window(
        series,
        holdout=40,
        context_len=72,
        train_min_len=77,
        validation_len=48,
    )
    train = series.loc[window["train_index"]]
    test_series = series.loc[window["test_index"]]
    reference = series.loc[: window["test_target_end"]]
    kwargs = dict(
        size_k=5,
        test_target_start=window["test_target_start"],
        seasonality_m=24,
        reference_insample=reference,
    )

    res_own = backtest_forecast(train, test_series, "LinearRegression", **kwargs)
    res_shared = backtest_forecast(
        train,
        test_series,
        "LinearRegression",
        **{**kwargs, "reference_insample": reference * 2.0},
    )

    assert res_shared["_mae"] == pytest.approx(res_own["_mae"])
    assert res_shared["_rmse"] == pytest.approx(res_own["_rmse"])
    assert res_shared["rmsse"] == pytest.approx(res_own["rmsse"] / 2.0, rel=1e-6)
    assert res_shared["mase"] == pytest.approx(res_own["mase"] / 2.0, rel=1e-6)


def test_backtest_scaled_metrics_use_raw_history_before_each_origin(monkeypatch):
    series = _seasonal_series(n=600, seed=11)
    window = select_holdout_window(
        series,
        holdout=40,
        context_len=72,
        train_min_len=77,
        validation_len=48,
    )
    train = series.loc[window["train_index"]]
    test_series = series.loc[window["test_index"]]
    reference = series.loc[: window["test_target_end"]]
    seen_mase: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seen_rmsse: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def record_seen(seen):
        def metric(actual, _pred, insample, *, seasonality_m):
            del seasonality_m
            seen.append((actual.start_time(), insample.index[-1]))
            return float(len(insample))

        return metric

    monkeypatch.setattr(backtest_module, "compute_mase", record_seen(seen_mase))
    monkeypatch.setattr(backtest_module, "compute_rmsse", record_seen(seen_rmsse))

    result = backtest_forecast(
        train,
        test_series,
        "LinearRegression",
        size_k=5,
        test_target_start=window["test_target_start"],
        seasonality_m=24,
        forecast_stride=2,
        reference_insample=reference,
    )

    assert result["n_forecasts"] > 1
    assert seen_mase == seen_rmsse
    assert all(
        reference_end == origin - pd.Timedelta(hours=1)
        for origin, reference_end in seen_mase
    )
    assert all(
        reference_end < origin and reference_end in reference.index
        for origin, reference_end in seen_mase
    )
    assert [reference_end for _, reference_end in seen_mase] == sorted(
        reference_end for _, reference_end in seen_mase
    )


def test_darts_error_metrics_pool_overlapping_forecasts():
    from darts.metrics import mae, rmse
    from airquality.forecasting.backtest import _darts_error_metrics

    actual_values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    predicted_values = np.array([2.0, 1.0, 6.0, 5.0, 12.0])
    index = pd.RangeIndex(len(actual_values))
    actual = TimeSeries.from_times_and_values(index, actual_values)
    predicted = TimeSeries.from_times_and_values(index, predicted_values)

    got_mae, got_rmse = _darts_error_metrics(actual_values, predicted_values)

    assert got_mae == pytest.approx(float(mae(actual, predicted)))
    assert got_rmse == pytest.approx(float(rmse(actual, predicted)))


# --------------------------------------------------------------------------- #
# Arm construction
# --------------------------------------------------------------------------- #
def test_build_arms_expands_strategies_and_imputation_variants():
    arms = cp.build_arms(["unlabeled", "inject-vote"], "both")
    assert [arm.name for arm in arms] == [
        "raw",
        "raw+frozen",
        "unlabeled+impute",
        "unlabeled+noimpute",
        "inject-vote+impute",
        "inject-vote+noimpute",
    ]
    assert arms[0].strategy is None and not arms[0].impute
    assert arms[1].strategy is None and not arms[1].impute
    assert arms[2].strategy == "unlabeled" and arms[2].impute
    assert arms[3].strategy == "unlabeled" and not arms[3].impute

    only_impute = cp.build_arms(["unlabeled"], "impute")
    assert [arm.name for arm in only_impute] == [
        "raw",
        "raw+frozen",
        "unlabeled+impute",
    ]
    only_gaps = cp.build_arms(["unlabeled", "unlabeled"], "none")  # dedupes
    assert [arm.name for arm in only_gaps] == [
        "raw",
        "raw+frozen",
        "unlabeled+noimpute",
    ]

    with pytest.raises(ValueError):
        cp.build_arms(["unlabeled"], "sometimes")


def test_only_complete_backtests_are_cacheable():
    complete = {
        "_mae": 1.0,
        "_rmse": 1.0,
        "mase": 1.0,
        "rmsse": 1.0,
        "n_test_predictions": 16,
        "n_forecasts": 2,
        "n_expected_forecasts": 2,
    }

    assert cp._cacheable_backtest(complete)
    assert not cp._cacheable_backtest({**complete, "n_forecasts": 1})
    assert not cp._cacheable_backtest({**complete, "_rmse": float("nan")})


def test_relative_metrics_pair_each_arm_with_matching_raw_result():
    rows = [
        {
            "series": "ST0",
            "regime": "short",
            "model": "NLinear",
            "arm": "raw+frozen",
            "_mae": 1.5,
            "_rmse": 3.0,
        },
        {
            "series": "ST0",
            "regime": "short",
            "model": "NLinear",
            "arm": "raw",
            "_mae": 2.0,
            "_rmse": 4.0,
        },
        {
            "series": "ST0",
            "regime": "short",
            "model": "LinearRegression",
            "arm": "raw",
            "_mae": 5.0,
            "_rmse": 10.0,
        },
    ]

    cp._apply_raw_relative_metrics(rows)

    frozen = rows[0]
    assert frozen["relmae"] == pytest.approx(0.75)
    assert frozen["relrmse"] == pytest.approx(0.75)
    assert rows[1]["relmae"] == pytest.approx(1.0)
    assert rows[1]["relrmse"] == pytest.approx(1.0)
    assert rows[2]["relmae"] == pytest.approx(1.0)
    assert rows[2]["relrmse"] == pytest.approx(1.0)


def test_relative_metrics_are_nan_when_pooled_raw_error_is_zero():
    rows = [
        {
            "series": "ST0",
            "regime": "short",
            "model": "NLinear",
            "arm": "raw",
            "_mae": 0.0,
            "_rmse": 0.0,
        },
        {
            "series": "ST0",
            "regime": "short",
            "model": "NLinear",
            "arm": "unlabeled+impute",
            "_mae": 1.0,
            "_rmse": 2.0,
        },
    ]

    cp._apply_raw_relative_metrics(rows)

    assert np.isnan(rows[1]["relmae"])
    assert np.isnan(rows[1]["relrmse"])


def test_resolve_forecasting_devices_uses_visible_gpus_and_cpu_fallback(
    monkeypatch,
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)

    assert cp.resolve_forecasting_devices("multi-gpu") == ("cuda:0", "cuda:1")
    assert cp.resolve_forecasting_devices("cuda") == ("cuda:0",)
    assert cp.resolve_forecasting_devices("cpu") == ("cpu",)

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert cp.resolve_forecasting_devices("multi-gpu") == ("cpu",)
    with pytest.raises(ValueError, match="device"):
        cp.resolve_forecasting_devices("tpu")


def test_gpu_worker_preserves_train_and_inference_timings(monkeypatch):
    series = _seasonal_series(n=120)
    task = cp._BacktestTask(
        train_series=series.iloc[:80],
        test_series=series.iloc[80:],
        reference_insample=series.iloc[:80],
        model_name="TiDE",
        size_k=8,
        test_target_start=series.index[88],
        seasonality_m=24,
        freq="h",
        validation_len=16,
        validation_stride=4,
        forecast_stride=4,
        context_len=8,
    )
    seen = {}
    monkeypatch.setattr(cp, "_FORECAST_WORKER_GPU", 1)

    def fake_configs(names, **kwargs):
        seen.update(kwargs)
        return {names[0]: object()}

    expected = {"train_seconds": 1.25, "inference_seconds": 0.5}
    monkeypatch.setattr(cp, "resolve_forecasting_model_configs", fake_configs)
    def fake_backtest(*args, **kwargs):
        seen["reference_insample"] = kwargs["reference_insample"].copy()
        return expected

    monkeypatch.setattr(cp, "backtest_forecast", fake_backtest)
    released = []
    monkeypatch.setattr(cp, "_release_cuda_memory", lambda: released.append(True))

    assert cp._run_gpu_backtest(task) is expected
    pd.testing.assert_series_equal(seen["reference_insample"], task.reference_insample)
    assert seen["accelerator"] == "gpu"
    assert seen["devices"] == [1]
    assert released == [True]


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #
def test_run_benchmark_selects_holdout_from_full_series_common_support(
    tmp_path, monkeypatch
):
    series = _seasonal_series(n=600, name="ST0", seed=8)
    preserved = series + 100.0
    seen_detection_index: list[pd.DatetimeIndex] = []
    seen_detection_values: list[pd.Series] = []
    seen_detection_context: dict[str, object] = {}
    test_indices: list[pd.DatetimeIndex] = []

    def fake_loader(**kwargs):
        source = preserved if kwargs.get("preserve_frozen") else series
        return [source.to_frame()]

    monkeypatch.setattr(cp, "_load_raw_hourly_series", fake_loader)
    csv_map = {
        ("forecasting", "detectors"): tuple(BASELINE_DETECTORS),
        ("forecasting", "forecast_models"): ("LinearRegression",),
        ("forecasting", "strategies"): ("unlabeled", "inject-vote"),
    }
    monkeypatch.setattr(
        cp,
        "cfg_get_csv_list",
        lambda section, option, default, *, cfg=None: csv_map.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(
        cp,
        "cfg_get_int",
        lambda section, option, default, cfg=None: 96
        if (section, option) == ("forecasting", "holdout")
        else 7
        if (section, option) == ("forecasting", "carla_stride")
        else default,
    )
    monkeypatch.setattr(
        cp,
        "cfg_get_str",
        lambda section, option, default, cfg=None: (
            "none"
            if (section, option) == ("forecasting", "imputation")
            else "drift"
            if (section, option) == ("synthetic", "injection_variant")
            else default
        ),
    )
    monkeypatch.setattr(cp, "cfg_get_bool", lambda *args, **kwargs: False)
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path)

    def fake_detect(full_series, strategies, *_args, **_kwargs):
        seen_detection_index.append(pd.DatetimeIndex(full_series.index))
        seen_detection_values.append(full_series.copy())
        seen_detection_context.update(_kwargs["context_kwargs"])
        out = {}
        for strategy, position in zip(strategies, (500, 550), strict=True):
            mask = pd.Series(False, index=full_series.index)
            mask.iloc[position] = True
            out[strategy.name] = DetectionResult(
                strategy.name,
                [],
                [],
                {},
                3.5,
                mask,
                n_flagged=1,
                detection_rate=1 / len(full_series),
            )
        return out

    def fake_backtest(_train, test_series, _model, **kwargs):
        test_indices.append(pd.DatetimeIndex(test_series.index))
        horizon = kwargs["size_k"]
        stride = kwargs["forecast_stride"]
        n_forecasts = (96 - horizon) // stride + 1
        return {
            "_mae": 1.0,
            "_rmse": 1.0,
            "mase": 1.0,
            "rmsse": 1.0,
            "n_test_predictions": n_forecasts * horizon,
            "n_forecasts": n_forecasts,
            "n_unique_targets": 96,
        }

    monkeypatch.setattr(cp, "_detect_for_strategies", fake_detect)
    monkeypatch.setattr(cp, "backtest_forecast", fake_backtest)
    monkeypatch.setattr(cp, "resolve_forecasting_devices", lambda _request: ("cpu",))

    artifacts = cp.run_benchmark_from_config()

    assert len(seen_detection_index) == 1
    assert seen_detection_index[0].equals(series.index)
    pd.testing.assert_series_equal(seen_detection_values[0], series)
    assert seen_detection_context["carla_stride"] == 7
    assert seen_detection_context["injection_variant"] == "drift"
    selection = artifacts["selection_df"].iloc[0]
    assert bool(selection["selected"])
    assert selection["split_n_flagged"] == 2
    assert pd.Timestamp(selection["test_target_end"]) == series.index[499]
    assert all(index.equals(test_indices[0]) for index in test_indices)
    assert series.index[500] not in test_indices[0]


def test_raw_frozen_uses_own_train_and_test_on_common_timestamps(
    tmp_path, monkeypatch
):
    primary = _seasonal_series(n=700, name="ST0", seed=9)
    primary.iloc[200:230] = np.nan
    preserved = primary.copy()
    preserved.iloc[200:230] = np.linspace(25.0, 35.0, 30)
    preserved.iloc[500:] += 20.0

    def fake_loader(**kwargs):
        source = preserved if kwargs.get("preserve_frozen") else primary
        return [source.to_frame()]

    csv_map = {
        ("forecasting", "detectors"): tuple(BASELINE_DETECTORS),
        ("forecasting", "forecast_models"): ("LinearRegression",),
        ("forecasting", "strategies"): (),
    }
    int_map = {
        ("forecasting", "holdout"): 96,
        ("forecasting", "context_len"): 72,
    }
    seen: list[tuple[pd.Series, pd.Series, pd.Series]] = []

    def fake_backtest(train, test, _model, **kwargs):
        seen.append((train.copy(), test.copy(), kwargs["reference_insample"].copy()))
        horizon = kwargs["size_k"]
        stride = kwargs["forecast_stride"]
        n_forecasts = (96 - horizon) // stride + 1
        return {
            "_mae": float(test.loc[kwargs["test_target_start"] :].mean()),
            "_rmse": float(test.loc[kwargs["test_target_start"] :].mean()),
            "mase": 1.0,
            "rmsse": 1.0,
            "n_test_predictions": n_forecasts * horizon,
            "n_forecasts": n_forecasts,
            "n_expected_forecasts": n_forecasts,
            "n_unique_targets": 96,
        }

    monkeypatch.setattr(cp, "_load_raw_hourly_series", fake_loader)
    monkeypatch.setattr(
        cp,
        "cfg_get_csv_list",
        lambda section, option, default, *, cfg=None: csv_map.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(
        cp,
        "cfg_get_int",
        lambda section, option, default, cfg=None: int_map.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(cp, "cfg_get_bool", lambda *args, **kwargs: False)
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path)
    monkeypatch.setattr(cp, "resolve_forecasting_devices", lambda _request: ("cpu",))
    monkeypatch.setattr(cp, "backtest_forecast", fake_backtest)

    artifacts = cp.run_benchmark_from_config()

    assert set(artifacts["results_df"]["arm"]) == {"raw", "raw+frozen"}
    assert artifacts["results_df"].groupby("arm")["relrmse"].mean().nunique() == 2
    assert len(seen) == 4
    for raw_call, frozen_call in ((seen[0], seen[1]), (seen[2], seen[3])):
        raw_train, raw_test, raw_mase = raw_call
        frozen_train, frozen_test, frozen_mase = frozen_call
        assert frozen_train.notna().sum() > raw_train.notna().sum()
        assert raw_test.index.equals(frozen_test.index)
        assert not raw_test.equals(frozen_test)
        pd.testing.assert_series_equal(
            raw_mase,
            primary.loc[: raw_test.index[-1]],
        )
        pd.testing.assert_series_equal(raw_mase, frozen_mase)
        assert raw_train.index.max() < raw_test.index[72]
        assert frozen_train.index.max() < frozen_test.index[72]


def test_run_benchmark_from_config_end_to_end(tmp_path, monkeypatch):
    def fake_loader(**_kwargs):
        s = _seasonal_series(n=900, name="ST0", seed=0)
        s.iloc[200] = 130.0
        s.iloc[300:330] = np.nan
        return [s.to_frame()]

    csv_map = {
        ("forecasting", "detectors"): tuple(BASELINE_DETECTORS),
        ("forecasting", "forecast_models"): ("LinearRegression",),
        ("forecasting", "strategies"): ("unlabeled", "inject-vote"),
    }
    int_map = {
        ("forecasting", "holdout"): 96,
        ("forecasting", "context_len"): 72,
        ("forecasting", "min_series_points"): 300,
    }
    str_map = {("forecasting", "imputation_model"): "interp"}

    def fake_csv(section, option, default, *, cfg=None):
        return csv_map.get((section, option), default)

    def fake_int(section, option, default, cfg=None):
        return int_map.get((section, option), default)

    monkeypatch.setattr(cp, "_load_raw_hourly_series", fake_loader)
    monkeypatch.setattr(cp, "cfg_get_csv_list", fake_csv)
    monkeypatch.setattr(cp, "cfg_get_int", fake_int)
    monkeypatch.setattr(
        cp,
        "cfg_get_str",
        lambda section, option, default, cfg=None: str_map.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(cp, "cfg_get_bool", lambda s, o, d, cfg=None: False)  # no disk cache
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path)
    monkeypatch.setattr(cp, "resolve_forecasting_devices", lambda _request: ("cpu",))

    artifacts = cp.run_benchmark_from_config()

    results_df = artifacts["results_df"]
    summary_df = artifacts["summary_df"]
    detection_df = artifacts["detection_df"]
    expected_arms = {
        "raw",
        "raw+frozen",
        "unlabeled+impute",
        "unlabeled+noimpute",
        "inject-vote+impute",
        "inject-vote+noimpute",
    }
    assert set(results_df["arm"]) == expected_arms
    assert set(results_df["series"]) == {"ST0"}
    # Raw arm never runs detection/imputation; strategy arms record theirs.
    raw_rows = results_df[results_df["arm"] == "raw"]
    assert (raw_rows["strategy"] == "none").all() and (~raw_rows["imputed"]).all()
    imputed_rows = results_df[results_df["imputed"]]
    assert set(imputed_rows["imputation_model"]) == {"interp"}
    # The public forecasting metric contract contains only scaled and relative metrics.
    assert set(results_df.columns) >= {"rmsse", "mase", "relmae", "relrmse"}
    assert not {
        "mae",
        "rmse",
        "_mae",
        "_rmse",
        "scale_ref",
        "origin_mae_mean",
        "origin_mae_std",
        "origin_rmse_mean",
        "origin_rmse_std",
    } & set(results_df.columns)
    assert set(results_df["regime"]) == {"short", "long"}
    trainable = ~results_df["arm"].str.endswith("+noimpute")
    assert np.isfinite(results_df.loc[trainable, "rmsse"]).all()
    assert np.isfinite(results_df.loc[trainable, "mase"]).all()
    assert np.isfinite(results_df.loc[trainable, "relmae"]).all()
    assert np.isfinite(results_df.loc[trainable, "relrmse"]).all()
    raw_rows = results_df[results_df["arm"] == "raw"]
    assert np.allclose(raw_rows["relmae"], 1.0)
    assert np.allclose(raw_rows["relrmse"], 1.0)
    for regime, horizon, stride in (("short", 8, 4), ("long", 48, 24)):
        subset = results_df.loc[(results_df["regime"] == regime) & trainable]
        assert (subset["horizon"] == horizon).all()
        assert (subset["forecast_stride"] == stride).all()
        assert (
            subset["n_forecasts"]
            == (subset["test_target_hours"] - horizon) // stride + 1
        ).all()
        assert (subset["n_expected_forecasts"] == subset["n_forecasts"]).all()
        assert (
            subset["n_test_predictions"] == subset["n_forecasts"] * horizon
        ).all()
        assert (
            subset["n_unique_targets"] == subset["test_target_hours"]
        ).all()

    for artifact in ("results.csv", "summary.csv", "detection.csv", "selection.csv"):
        assert (tmp_path / artifact).exists()
    progress_log = artifacts["log_path"].read_text(encoding="utf-8")
    assert "[log-guide] elapsed=wall-clock time" in progress_log
    assert "branch=forecast arm" in progress_log
    assert "[station 1/1][ST0] start" in progress_log
    assert "[backtest task=" in progress_log
    assert "cache=off" in progress_log
    assert "[run] done" in progress_log
    for col in (
        "rmsse_raw",
        "mase_raw+frozen",
        "relmae_unlabeled+impute",
        "relmae_unlabeled+impute_delta",
        "relrmse_unlabeled+impute_improve_pct",
        "mase_inject-vote+noimpute_delta",
    ):
        assert col in summary_df.columns

    assert set(detection_df["strategy"]) == {"unlabeled", "inject-vote"}
    assert set(detection_df["scope"]) == {"full_series"}
    assert artifacts["selection_df"].iloc[0]["test_target_hours"] == 96
    vote_row = detection_df[detection_df["strategy"] == "inject-vote"].iloc[0]
    assert vote_row["ranking"]  # the injection ranking is persisted


def test_run_benchmark_applies_mask_transforms(tmp_path, monkeypatch):
    def fake_loader(**_kwargs):
        s = _seasonal_series(n=900, name="ST0", seed=1)
        s.iloc[200] = 130.0
        s.iloc[300:330] = np.nan
        return [s.to_frame()]

    csv_map = {
        ("forecasting", "detectors"): tuple(BASELINE_DETECTORS),
        ("forecasting", "forecast_models"): ("LinearRegression",),
        ("forecasting", "strategies"): ("unlabeled",),
    }
    int_map = {
        ("forecasting", "holdout"): 96,
        ("forecasting", "context_len"): 72,
        ("forecasting", "min_series_points"): 300,
    }
    str_map = {("forecasting", "imputation"): "none"}

    monkeypatch.setattr(cp, "_load_raw_hourly_series", fake_loader)
    monkeypatch.setattr(cp, "cfg_get_csv_list", lambda s, o, d, *, cfg=None: csv_map.get((s, o), d))
    monkeypatch.setattr(cp, "cfg_get_int", lambda s, o, d, cfg=None: int_map.get((s, o), d))
    monkeypatch.setattr(cp, "cfg_get_str", lambda s, o, d, cfg=None: str_map.get((s, o), d))
    monkeypatch.setattr(cp, "cfg_get_bool", lambda s, o, d, cfg=None: False)  # no disk cache
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path)
    monkeypatch.setattr(cp, "resolve_forecasting_devices", lambda _request: ("cpu",))

    def clear_all(_series, mask):
        return pd.Series(False, index=mask.index, name=mask.name)

    artifacts = cp.run_benchmark_from_config(mask_transforms=[clear_all])

    # The transform wiped every detection before removal.
    assert (artifacts["detection_df"]["n_flagged"] == 0).all()
    assert (artifacts["results_df"]["n_anomalies"] == 0).all()


def test_foundation_only_common_test_skips_training_arms(tmp_path, monkeypatch):
    submitted = []

    class FakeExecutor:
        def submit(self, fn, task):
            submitted.append(task)
            future = Future()
            future.set_result(fn(task))
            return future

        def shutdown(self, wait=True):
            assert wait

    class FakeQueue:
        def close(self):
            pass

        def join_thread(self):
            pass

    def fake_loader(**kwargs):
        series = _seasonal_series(n=900, name="ST0", seed=2)
        series.iloc[300:330] = np.nan
        if kwargs.get("preserve_frozen"):
            series = series + 10.0
        return [series.to_frame()]

    csv_map = {
        ("forecasting", "forecast_models"): ("Chronos2",),
        ("forecasting", "strategies"): ("unlabeled",),
    }
    int_map = {
        ("forecasting", "holdout"): 96,
        ("forecasting", "context_len"): 72,
    }
    monkeypatch.setattr(cp, "_load_raw_hourly_series", fake_loader)
    monkeypatch.setattr(
        cp,
        "cfg_get_csv_list",
        lambda section, option, default, *, cfg=None: csv_map.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(
        cp,
        "cfg_get_int",
        lambda section, option, default, cfg=None: int_map.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(cp, "cfg_get_bool", lambda *args, **kwargs: False)
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cp,
        "resolve_forecasting_devices",
        lambda _request: ("cuda:0", "cuda:1"),
    )
    monkeypatch.setattr(
        cp,
        "_create_gpu_executor",
        lambda _indices, _log_queue=None: (FakeExecutor(), FakeQueue()),
    )
    monkeypatch.setattr(
        cp,
        "build_detection_strategy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("common foundation reference must not build training arms")
        ),
    )
    monkeypatch.setattr(
        cp,
        "_run_gpu_backtest",
        lambda task: {
            "_mae": 1.0,
            "_rmse": 1.0,
            "mase": 1.0,
            "rmsse": 1.0,
            "train_seconds": 1.25,
            "inference_seconds": 0.5,
            "n_test_predictions": task.size_k,
            "n_forecasts": 1,
            "n_expected_forecasts": 1,
            "n_unique_targets": task.size_k,
        },
    )

    artifacts = cp.run_benchmark_from_config()

    assert [arm.name for arm in artifacts["arms"]] == ["raw", "raw+frozen"]
    assert set(artifacts["results_df"]["arm"]) == {"raw", "raw+frozen"}
    assert set(artifacts["results_df"]["model_mode"]) == {"foundation"}
    assert len(artifacts["results_df"]) == 4
    assert len(submitted) == 4
    for raw_task, frozen_task in ((submitted[0], submitted[1]), (submitted[2], submitted[3])):
        assert raw_task.train_series.index.equals(frozen_task.train_series.index)
        assert raw_task.test_series.index.equals(frozen_task.test_series.index)
        assert not raw_task.train_series.equals(frozen_task.train_series)
        assert not raw_task.test_series.equals(frozen_task.test_series)
    assert (artifacts["results_df"]["train_seconds"] == 1.25).all()
    assert (artifacts["results_df"]["inference_seconds"] == 0.5).all()
    assert artifacts["detection_df"].empty
    assert list(pd.read_csv(tmp_path / "detection.csv").columns) == [
        "series",
        "strategy",
        "detectors",
            "discarded",
            "n_flagged",
            "n_unscored",
            "coverage_rate",
            "detection_rate",
        "ranking",
        "scope",
        "n_observed",
    ]


def test_run_benchmark_closes_gpu_resources_on_failure(monkeypatch):
    calls = []

    class FakeExecutor:
        def shutdown(self, *, wait, cancel_futures):
            calls.append(("executor", wait, cancel_futures))

    class FakeQueue:
        def close(self):
            calls.append(("queue-close",))

        def join_thread(self):
            calls.append(("queue-join",))

    def fail_run(*, mask_transforms):
        cp._ACTIVE_GPU_EXECUTOR = FakeExecutor()
        cp._ACTIVE_DEVICE_QUEUE = FakeQueue()
        raise RuntimeError("worker failed")

    monkeypatch.setattr(cp, "_run_benchmark_from_config", fail_run)

    with pytest.raises(RuntimeError, match="worker failed"):
        cp.run_benchmark_from_config()

    assert calls == [
        ("executor", True, True),
        ("queue-close",),
        ("queue-join",),
    ]
    assert cp._ACTIVE_GPU_EXECUTOR is None
    assert cp._ACTIVE_DEVICE_QUEUE is None


def test_backtest_scaled_metrics_match_darts_denominators():
    from darts import TimeSeries
    from darts.metrics import mae, mase, rmsse
    from airquality.metrics import compute_mase, compute_rmsse

    rng = np.random.default_rng(13)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    insample = pd.Series(
        10 + 3 * np.sin(2 * np.pi * np.arange(n) / 24.0) + rng.normal(0, 0.5, n),
        index=idx,
        name="S",
    )
    insample.iloc[30:35] = np.nan

    hold_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=48, freq="h")
    actual_vals = 10 + 3 * np.sin(2 * np.pi * np.arange(48) / 24.0) + rng.normal(0, 0.5, 48)
    actual = TimeSeries.from_series(pd.Series(actual_vals, index=hold_idx), freq="h")
    pred = TimeSeries.from_series(
        pd.Series(actual_vals + rng.normal(0, 0.7, 48), index=hold_idx), freq="h"
    )

    # The wrappers use the interpolated training history and the native Darts
    # scaled metrics on a contiguous synthetic index.
    filled = insample.interpolate(method="time", limit_direction="both").ffill().bfill()
    values = filled.to_numpy(dtype=float)
    insample_ts = TimeSeries.from_times_and_values(pd.RangeIndex(n), values)
    eval_index = pd.RangeIndex(n, n + len(actual_vals))
    actual_ts = TimeSeries.from_times_and_values(eval_index, actual_vals)
    pred_ts = TimeSeries.from_times_and_values(
        eval_index, pred.to_series().to_numpy(dtype=float)
    )

    mase_out = compute_mase(actual, pred, insample, seasonality_m=24)
    rmsse_out = compute_rmsse(actual, pred, insample, seasonality_m=24)
    assert mase_out == pytest.approx(
        float(mase(actual_ts, pred_ts, insample_ts, m=24)), rel=1e-9
    )
    assert rmsse_out == pytest.approx(
        float(rmsse(actual_ts, pred_ts, insample_ts, m=24)), rel=1e-9
    )

    assert float(mae(actual_ts, pred_ts)) > 0.0


def test_backtest_mase_returns_nan_when_history_is_too_short():
    from darts import TimeSeries

    from airquality.metrics import compute_mase

    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    insample = pd.Series(np.arange(10, dtype=float), index=idx, name="S")
    hold_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=4, freq="h")
    ts = TimeSeries.from_series(pd.Series([1.0, 2.0, 3.0, 4.0], index=hold_idx), freq="h")

    out = compute_mase(ts, ts, insample, seasonality_m=24)
    assert np.isnan(out)


def test_backtest_rmsse_returns_nan_when_history_is_too_short():
    from darts import TimeSeries

    from airquality.metrics import compute_rmsse

    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    insample = pd.Series(np.arange(10, dtype=float), index=idx, name="S")
    hold_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=4, freq="h")
    ts = TimeSeries.from_series(pd.Series([1.0, 2.0, 3.0, 4.0], index=hold_idx), freq="h")

    out = compute_rmsse(ts, ts, insample, seasonality_m=24)
    assert np.isnan(out)
