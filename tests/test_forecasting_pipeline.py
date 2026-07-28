"""Tests for the multi-arm forecasting benchmark pipeline."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

import airquality.forecasting.fill as fill
import airquality.forecasting.pipeline as cp
from darts import TimeSeries

from airquality.forecasting.backtest import (
    backtest_forecast,
    select_holdout_window,
    split_train_val_subseries,
)
from airquality.forecasting.cleaning import detect_anomaly_mask, remove_anomalies
from airquality.forecasting.detection import DetectionResult
from airquality.forecasting.fill import build_imputer, impute_series, nan_gap_windows
from airquality.imputation.registry import resolve_imputer_family

BASELINE_DETECTORS = ["ModifiedZScore", "IQR", "Hampel_w24"]


def _seasonal_series(n: int = 900, name: str = "ST", seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    vals = (
        30.0
        + 8.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        + 4.0 * np.sin(np.arange(n) * 2 * np.pi / 168)
        + rng.normal(0, 1, n)
    )
    return pd.Series(vals, index=idx, name=name)


# --------------------------------------------------------------------------- #
# Anomaly detection / removal
# --------------------------------------------------------------------------- #
def test_detect_anomaly_mask_flags_spikes():
    series = _seasonal_series(seed=1)
    series.iloc[300] = 140.0
    series.iloc[500] = 130.0
    series.iloc[100:105] = np.nan  # pre-existing gap stays out of detection

    result = detect_anomaly_mask(series, detectors=BASELINE_DETECTORS, device="cpu")

    assert result.detectors and set(result.detectors) <= set(BASELINE_DETECTORS)
    assert bool(result.mask.iloc[300]) and bool(result.mask.iloc[500])
    # Gaps are never flagged (they are not observed points).
    assert not result.mask.iloc[100:105].any()
    # The consensus itself must respect the rarity budget on this clean series.
    assert result.detection_rate <= 0.07

    cleaned = remove_anomalies(series, result)
    assert np.isnan(cleaned.iloc[300]) and np.isnan(cleaned.iloc[500])
    assert int(cleaned.isna().sum()) >= int(series.isna().sum()) + 2


def test_detect_anomaly_mask_short_series_is_noop():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    result = detect_anomaly_mask(pd.Series([1.0, 2, 3, 4, 5], index=idx, name="s"))
    assert result.detectors == []
    assert result.n_flagged == 0


def test_detect_anomaly_mask_discards_detectors_over_budget():
    # With a budget of 0, any detector that flags a point is discarded and the
    # final mask is empty (no survivors flag anything).
    series = _seasonal_series(seed=1)
    series.iloc[300] = 140.0

    result = detect_anomaly_mask(
        series, detectors=BASELINE_DETECTORS, device="cpu", max_detection_rate=0.0
    )

    assert set(result.discarded) >= {"ModifiedZScore"}
    assert all(result.rates[name] == 0.0 for name in result.detectors)
    assert result.n_flagged == 0


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


def test_detect_anomaly_mask_reports_per_detector_rates():
    series = _seasonal_series(seed=3)
    series.iloc[400] = 150.0

    result = detect_anomaly_mask(series, detectors=BASELINE_DETECTORS, device="cpu")

    assert set(result.rates) == set(BASELINE_DETECTORS)
    assert all(0.0 <= rate <= 1.0 for rate in result.rates.values())
    assert bool(result.mask.iloc[400])


def test_detect_anomaly_mask_detects_per_contiguous_segment():
    # A spike in a second contiguous stretch (after a long gap) must still be
    # flagged: detection runs per segment instead of gluing stretches.
    series = _seasonal_series(n=700, seed=4)
    series.iloc[300:340] = np.nan  # long gap -> two contiguous segments
    series.iloc[500] = 160.0  # spike in the SECOND segment

    result = detect_anomaly_mask(series, detectors=BASELINE_DETECTORS, device="cpu")

    assert bool(result.mask.iloc[500])
    # Nothing inside the gap can be flagged.
    assert not result.mask.iloc[300:340].any()


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


def test_common_detection_support_excludes_unscored_timestamps():
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

    assert common_mask[common_mask].index.tolist() == [series.index[7]]
    assert pd.isna(support.iloc[7])


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
    assert np.isfinite(res["rmse"]) and np.isfinite(res["mae"])
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


def test_backtest_forecast_mase_uses_shared_insample():
    # MASE = MAE / naive-MAE(insample): doubling the insample amplitude doubles
    # the denominator, so the reported MASE halves while RMSE/MAE stay put.
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
    kwargs = dict(
        size_k=5,
        test_target_start=window["test_target_start"],
        seasonality_m=24,
    )

    res_own = backtest_forecast(train, test_series, "LinearRegression", **kwargs)
    res_shared = backtest_forecast(
        train, test_series, "LinearRegression", **kwargs, mase_insample=train * 2.0
    )

    assert res_shared["rmse"] == pytest.approx(res_own["rmse"])
    assert res_shared["mase"] == pytest.approx(res_own["mase"] / 2.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Arm construction
# --------------------------------------------------------------------------- #
def test_build_arms_expands_strategies_and_imputation_variants():
    arms = cp.build_arms(["unlabeled", "inject-vote"], "both")
    assert [arm.name for arm in arms] == [
        "raw",
        "unlabeled+impute",
        "unlabeled+noimpute",
        "inject-vote+impute",
        "inject-vote+noimpute",
    ]
    assert arms[0].strategy is None and not arms[0].impute
    assert arms[1].strategy == "unlabeled" and arms[1].impute
    assert arms[2].strategy == "unlabeled" and not arms[2].impute

    only_impute = cp.build_arms(["unlabeled"], "impute")
    assert [arm.name for arm in only_impute] == ["raw", "unlabeled+impute"]
    only_gaps = cp.build_arms(["unlabeled", "unlabeled"], "none")  # dedupes
    assert [arm.name for arm in only_gaps] == ["raw", "unlabeled+noimpute"]

    with pytest.raises(ValueError):
        cp.build_arms(["unlabeled"], "sometimes")


def test_only_complete_backtests_are_cacheable():
    complete = {
        "rmse": 1.0,
        "n_test_predictions": 16,
        "n_forecasts": 2,
        "n_expected_forecasts": 2,
    }

    assert cp._cacheable_backtest(complete)
    assert not cp._cacheable_backtest({**complete, "n_forecasts": 1})
    assert not cp._cacheable_backtest({**complete, "rmse": float("nan")})


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
        mase_insample=series.iloc[:80],
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
    monkeypatch.setattr(cp, "backtest_forecast", lambda *args, **kwargs: expected)

    assert cp._run_gpu_backtest(task) is expected
    assert seen["accelerator"] == "gpu"
    assert seen["devices"] == [1]


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #
def test_run_benchmark_selects_holdout_from_full_series_common_support(
    tmp_path, monkeypatch
):
    series = _seasonal_series(n=600, name="ST0", seed=8)
    seen_detection_index: list[pd.DatetimeIndex] = []
    test_indices: list[pd.DatetimeIndex] = []

    monkeypatch.setattr(
        cp,
        "_load_raw_hourly_series",
        lambda **_kwargs: [series.to_frame()],
    )
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
        else default,
    )
    monkeypatch.setattr(
        cp,
        "cfg_get_str",
        lambda section, option, default, cfg=None: "none"
        if (section, option) == ("forecasting", "imputation")
        else default,
    )
    monkeypatch.setattr(cp, "cfg_get_bool", lambda *args, **kwargs: False)
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path)

    def fake_detect(full_series, strategies, *_args, **_kwargs):
        seen_detection_index.append(pd.DatetimeIndex(full_series.index))
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
            "rmse": 1.0,
            "mase": 1.0,
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
    selection = artifacts["selection_df"].iloc[0]
    assert bool(selection["selected"])
    assert selection["split_n_flagged"] == 2
    assert pd.Timestamp(selection["test_target_end"]) == series.index[499]
    assert all(index.equals(test_indices[0]) for index in test_indices)
    assert series.index[500] not in test_indices[0]


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
    # Metrics are the scale-free pair: scaled RMSE (+ its scale_ref) and MASE.
    assert "mae" not in results_df.columns
    assert (results_df["scale_ref"] > 0).all()
    assert set(results_df["regime"]) == {"short", "long"}
    trainable = ~results_df["arm"].str.endswith("+noimpute")
    assert np.isfinite(results_df.loc[trainable, "rmse"]).all()
    assert np.isfinite(results_df.loc[trainable, "mase"]).all()
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
    for col in (
        "rmse_raw",
        "rmse_unlabeled+impute",
        "rmse_unlabeled+impute_delta",
        "rmse_unlabeled+impute_improve_pct",
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

    def fake_loader(**_kwargs):
        series = _seasonal_series(n=900, name="ST0", seed=2)
        series.iloc[300:330] = np.nan
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
        lambda _indices: (FakeExecutor(), FakeQueue()),
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
            "rmse": 1.0,
            "mase": 1.0,
            "train_seconds": 1.25,
            "inference_seconds": 0.5,
            "n_test_predictions": task.size_k,
            "n_forecasts": 1,
            "n_expected_forecasts": 1,
            "n_unique_targets": task.size_k,
        },
    )

    artifacts = cp.run_benchmark_from_config()

    assert [arm.name for arm in artifacts["arms"]] == ["raw"]
    assert set(artifacts["results_df"]["arm"]) == {"raw"}
    assert set(artifacts["results_df"]["model_mode"]) == {"foundation"}
    assert len(artifacts["results_df"]) == 2
    assert len(submitted) == 2
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


def test_backtest_mase_matches_mae_over_seasonal_naive_denominator():
    from darts import TimeSeries
    from darts.metrics import mae
    from airquality.metrics import compute_mase

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

    # MASE equals MAE over the in-sample seasonal-naive MAE computed on the
    # interpolated training history.
    filled = insample.interpolate(method="time", limit_direction="both").ffill().bfill()
    values = filled.to_numpy(dtype=float)
    denominator = float(np.mean(np.abs(values[24:] - values[:-24])))

    out = compute_mase(actual, pred, insample, seasonality_m=24)
    assert out == pytest.approx(float(mae(actual, pred)) / denominator, rel=1e-9)


def test_backtest_mase_returns_nan_when_history_is_too_short():
    from darts import TimeSeries

    from airquality.metrics import compute_mase

    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    insample = pd.Series(np.arange(10, dtype=float), index=idx, name="S")
    hold_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=4, freq="h")
    ts = TimeSeries.from_series(pd.Series([1.0, 2.0, 3.0, 4.0], index=hold_idx), freq="h")

    out = compute_mase(ts, ts, insample, seasonality_m=24)
    assert np.isnan(out)
