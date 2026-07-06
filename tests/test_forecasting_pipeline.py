"""Tests for the raw-vs-preprocessed forecasting comparison pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import airquality.forecasting.pipeline as cp
from airquality.forecasting.backtest import backtest_forecast, select_holdout_window
from airquality.forecasting.cleaning import detect_anomaly_mask, remove_anomalies
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


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #
def test_repo_root_resolves_to_repository_root():
    # Regression: `parents[2]` from src/airquality/forecasting/ is `src/`, which
    # sent comparison artifacts to src/reports/ instead of reports/.
    root = cp._repo_root()
    assert (root / "pyproject.toml").exists()
    assert root.name != "src"


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


def test_impute_series_interp_fills_and_preserves_observed():
    series = _seasonal_series(n=300)
    observed_before = series.iloc[0]
    series.iloc[5:9] = np.nan
    series.iloc[60:90] = np.nan
    series.iloc[-3:] = np.nan

    imputer = build_imputer("interp", freq="h")
    filled = impute_series(series, imputer, freq="h")

    assert not filled.isna().any()
    assert filled.iloc[0] == pytest.approx(observed_before)
    assert filled.iloc[7] == pytest.approx(filled.iloc[7])  # finite


# --------------------------------------------------------------------------- #
# Forecasting backtest
# --------------------------------------------------------------------------- #
def test_select_holdout_window_requires_contiguous_run():
    series = _seasonal_series(n=200)
    assert select_holdout_window(series, holdout=500, context_len=72) is None
    window = select_holdout_window(series, holdout=40, context_len=72)
    assert window is not None
    assert len(window["holdout_index"]) == 40
    assert len(window["eval_index"]) == 112


def test_backtest_forecast_returns_finite_metrics():
    series = _seasonal_series(n=600, seed=2)
    series.iloc[100:140] = np.nan
    window = select_holdout_window(series, holdout=40, context_len=72)
    train = series.loc[: window["holdout_start"]].iloc[:-1]
    eval_obs = series.loc[window["eval_index"]]

    res = backtest_forecast(
        train, eval_obs, "LinearRegression",
        size_k=5, holdout_start=window["holdout_start"], seasonality_m=24,
    )
    assert res["n_eval"] > 0
    assert np.isfinite(res["rmse"]) and np.isfinite(res["mae"])


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #
def test_run_comparison_from_config_end_to_end(tmp_path, monkeypatch):
    def fake_loader(*, freq, name_from_path=True, target_column_index=None):
        frames = []
        for k in range(2):
            s = _seasonal_series(n=900, name=f"ST{k}", seed=k)
            s.iloc[200] = 130.0
            s.iloc[300:330] = np.nan
            frames.append(s.to_frame())
        return frames

    csv_map = {
        ("forecasting", "detectors"): tuple(BASELINE_DETECTORS),
        ("forecasting", "forecast_models"): ("LinearRegression",),
    }
    int_map = {
        ("forecasting", "holdout"): 40,
        ("forecasting", "context_len"): 72,
        ("forecasting", "min_series_points"): 300,
    }

    def fake_csv(section, option, default, *, cfg=None):
        return csv_map.get((section, option), default)

    def fake_int(section, option, default, cfg=None):
        return int_map.get((section, option), default)

    monkeypatch.setattr(cp, "load_and_normalize_series", fake_loader)
    monkeypatch.setattr(cp, "cfg_get_csv_list", fake_csv)
    monkeypatch.setattr(cp, "cfg_get_int", fake_int)
    monkeypatch.setattr(cp, "_build_output_dir", lambda: tmp_path)

    artifacts = cp.run_comparison_from_config()

    results_df = artifacts["results_df"]
    summary_df = artifacts["summary_df"]
    assert set(results_df["arm"]) == {"raw", "preprocessed"}
    assert set(results_df["series"]) == {"ST0", "ST1"}
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    for col in ("rmse_raw", "rmse_pre", "rmse_delta", "rmse_improve_pct"):
        assert col in summary_df.columns


def test_backtest_mase_matches_mae_over_seasonal_naive_denominator():
    from darts import TimeSeries
    from darts.metrics import mae

    from airquality.forecasting.backtest import _compute_backtest_mase

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

    # darts.metrics.mase must equal MAE over the in-sample seasonal-naive MAE
    # computed on the interpolated training history (the documented convention).
    filled = insample.interpolate(method="time", limit_direction="both").ffill().bfill()
    values = filled.to_numpy(dtype=float)
    denominator = float(np.mean(np.abs(values[24:] - values[:-24])))

    out = _compute_backtest_mase(actual, pred, insample, seasonality_m=24, freq="h")
    assert out == pytest.approx(float(mae(actual, pred)) / denominator, rel=1e-9)


def test_backtest_mase_returns_nan_when_history_is_too_short():
    from darts import TimeSeries

    from airquality.forecasting.backtest import _compute_backtest_mase

    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    insample = pd.Series(np.arange(10, dtype=float), index=idx, name="S")
    hold_idx = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=4, freq="h")
    ts = TimeSeries.from_series(pd.Series([1.0, 2.0, 3.0, 4.0], index=hold_idx), freq="h")

    out = _compute_backtest_mase(ts, ts, insample, seasonality_m=24, freq="h")
    assert np.isnan(out)
