"""Tests for the paired synthetic foundation-context experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

import airquality.forecasting.pipeline as pipeline
from airquality.forecasting.backtest import forecast_foundation_context
from airquality.forecasting.detection import DetectionResult
from airquality.forecasting.foundation_preprocessing import (
    CLEAN_REFERENCE,
    CORRUPTED,
    build_preprocessing_contexts,
    build_synthetic_context_cases,
    summarize_foundation_preprocessing,
)


def _test_series() -> pd.Series:
    index = pd.date_range("2025-01-01", periods=72 + 96, freq="h")
    values = 30.0 + np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    return pd.Series(values, index=index, name="ST")


def test_synthetic_cases_corrupt_only_context_and_keep_targets_paired() -> None:
    series = _test_series()
    target_start = series.index[72]

    cases = build_synthetic_context_cases(
        series,
        test_target_start=target_start,
        context_len=72,
        horizon=8,
        stride=4,
        repeats=3,
        test_seed=1001,
    )

    assert len(cases) == 12
    assert {case.anomaly_type for case in cases} == {
        "spikes",
        "scale",
        "noise",
        "drift",
    }
    assert len({case.test_seed for case in cases}) == 12
    for case in cases:
        assert case.injected_mask.any()
        assert case.injected_mask.index.intersection(case.target.index).empty
        pd.testing.assert_series_equal(
            case.corrupted_context[~case.injected_mask],
            case.clean_context[~case.injected_mask],
        )
        pd.testing.assert_series_equal(
            case.target,
            series.reindex(case.target.index),
        )


def test_preprocessing_contexts_mask_only_each_strategy_flags() -> None:
    case = build_synthetic_context_cases(
        _test_series(),
        test_target_start=_test_series().index[72],
        context_len=72,
        horizon=8,
        stride=4,
        repeats=1,
        test_seed=1001,
    )[0]
    mask = pd.Series(False, index=case.corrupted_context.index)
    mask.loc[case.injected_mask[case.injected_mask].index[0]] = True
    detection = DetectionResult("unlabeled", [], [], {}, 3.5, mask)

    contexts = build_preprocessing_contexts(case, {"unlabeled": detection})

    assert list(contexts) == [CLEAN_REFERENCE, CORRUPTED, "unlabeled"]
    assert contexts["unlabeled"].isna().sum() == 1
    pd.testing.assert_series_equal(contexts[CLEAN_REFERENCE], case.clean_context)
    pd.testing.assert_series_equal(contexts[CORRUPTED], case.corrupted_context)


def test_foundation_context_forecast_uses_exact_future_target() -> None:
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    history = pd.Series(np.arange(200, dtype=float), index=index, name="ST")
    context = history.iloc[-72:]
    target_index = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=8, freq="h")
    target = pd.Series(np.arange(200, 208, dtype=float), index=target_index, name="ST")
    scaler = Scaler(global_fit=True, scaler=StandardScaler()).fit(
        TimeSeries.from_series(history)
    )

    class FakeFoundation:
        def predict(self, n, series, verbose=False):
            assert n == len(target)
            assert series.end_time() == context.index[-1]
            return TimeSeries.from_times_and_values(
                target.index,
                np.zeros(n, dtype=np.float32),
            )

    result = forecast_foundation_context(
        FakeFoundation(),
        scaler,
        context,
        target,
        history,
        seasonality_m=24,
    )

    assert result["n_test_predictions"] == 8
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["mase"])


def test_summary_reports_damage_recovery_and_residual() -> None:
    rows = []
    common = {
        "series": "ST",
        "regime": "short",
        "horizon": 8,
        "model": "Chronos2",
        "case_id": "case",
        "anomaly_type": "spikes",
        "test_seed": 1001,
        "test_target_start": pd.Timestamp("2025-01-04"),
        "n_injected": 1,
        "n_injected_detected": 0,
        "imputation_applied": False,
        "n_test_predictions": 8,
    }
    for condition, rmse, mase in (
        (CLEAN_REFERENCE, 1.0, 1.0),
        (CORRUPTED, 2.0, 1.8),
        ("unlabeled", 1.25, 1.2),
    ):
        rows.append({**common, "condition": condition, "rmse": rmse, "mase": mase})

    summary = summarize_foundation_preprocessing(pd.DataFrame(rows)).iloc[0]

    assert summary["rmse_damage"] == pytest.approx(1.0)
    assert summary["rmse_recovery"] == pytest.approx(0.75)
    assert summary["rmse_residual"] == pytest.approx(0.25)
    assert summary["rmse_recovery_pct"] == pytest.approx(75.0)

    failed = pd.DataFrame(rows)
    failed.loc[failed["condition"] == "unlabeled", "rmse"] = np.nan
    assert summarize_foundation_preprocessing(failed).empty


def test_pipeline_runs_paired_foundation_conditions_with_imputation_fallback(
    tmp_path, monkeypatch
) -> None:
    index = pd.date_range("2024-01-01", periods=900, freq="h")
    values = 30.0 + 5.0 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    series = pd.Series(values, index=index, name="ST")
    csv_values = {
        ("forecasting", "forecast_models"): ("Chronos2",),
        ("forecasting", "strategies"): (
            "unlabeled",
            "inject-best",
            "inject-vote",
        ),
    }
    int_values = {
        ("forecasting", "holdout"): 96,
        ("forecasting", "context_len"): 72,
        ("forecasting", "foundation_test_seed"): 1001,
        ("forecasting", "foundation_test_repeats"): 1,
    }
    target_values: dict[tuple[pd.Timestamp, int], list[np.ndarray]] = {}
    prepare_calls: list[int] = []

    monkeypatch.setattr(
        pipeline,
        "cfg_get_csv_list",
        lambda section, option, default, *, cfg=None: csv_values.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "cfg_get_int",
        lambda section, option, default, cfg=None: int_values.get(
            (section, option), default
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "cfg_get_str",
        lambda section, option, default, cfg=None: (
            "interp"
            if (section, option) == ("forecasting", "imputation_model")
            else default
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "cfg_get_bool",
        lambda section, option, default, cfg=None: (
            option == "foundation_preprocessing_test"
        ),
    )
    monkeypatch.setattr(
        pipeline, "_load_raw_hourly_series", lambda **_kwargs: [series.to_frame()]
    )
    monkeypatch.setattr(pipeline, "_build_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        pipeline, "resolve_forecasting_devices", lambda _request: ("cpu",)
    )

    def fake_detect(values, strategies, *_args, base_key, **_kwargs):
        synthetic = base_key.get("experiment") == "foundation-preprocessing-synthetic-v1"
        output = {}
        for strategy in strategies:
            mask = pd.Series(False, index=values.index)
            if synthetic:
                mask.iloc[-1] = True
            output[strategy.name] = DetectionResult(
                strategy.name,
                ["dummy"],
                [],
                {},
                3.5,
                mask,
                scored_mask=pd.Series(True, index=values.index),
                n_flagged=int(mask.sum()),
            )
        return output

    def fake_backtest(_train, test_series, _model, **kwargs):
        horizon = kwargs["size_k"]
        stride = kwargs["forecast_stride"]
        target_hours = len(test_series) - int(
            test_series.index.get_loc(kwargs["test_target_start"])
        )
        n_forecasts = (target_hours - horizon) // stride + 1
        return {
            "rmse": 1.0,
            "mase": 1.0,
            "n_test_predictions": n_forecasts * horizon,
            "n_forecasts": n_forecasts,
            "n_expected_forecasts": n_forecasts,
            "n_unique_targets": target_hours,
        }

    def fake_prepare(train, _model, *, size_k, **_kwargs):
        prepare_calls.append(size_k)
        return object(), object(), train, 0.01

    def fake_forecast(_model, _scaler, context, target, _insample, **_kwargs):
        key = (pd.Timestamp(target.index[0]), len(target))
        target_values.setdefault(key, []).append(target.to_numpy(copy=True))
        if context.isna().any():
            raise ValueError("missing context")
        return {
            "rmse": 1.0,
            "mase": 1.0,
            "inference_seconds": 0.01,
            "n_test_predictions": len(target),
        }

    monkeypatch.setattr(pipeline, "_detect_for_strategies", fake_detect)
    monkeypatch.setattr(pipeline, "backtest_forecast", fake_backtest)
    monkeypatch.setattr(pipeline, "prepare_foundation_model", fake_prepare)
    monkeypatch.setattr(pipeline, "forecast_foundation_context", fake_forecast)
    monkeypatch.setattr(
        pipeline,
        "impute_series",
        lambda values, *_args, **_kwargs: values.interpolate(
            method="time", limit_direction="both"
        ),
    )

    artifacts = pipeline.run_benchmark_from_config()
    results = artifacts["foundation_preprocessing_df"]

    assert set(results["condition"]) == {
        CLEAN_REFERENCE,
        CORRUPTED,
        "unlabeled",
        "inject-best",
        "inject-vote",
    }
    assert len(results) == 40
    assert set(results.loc[results["imputation_applied"], "condition"]) == {
        "unlabeled",
        "inject-best",
        "inject-vote",
    }
    assert prepare_calls == [8, 48]
    assert all(
        np.array_equal(values[0], candidate)
        for values in target_values.values()
        for candidate in values[1:]
    )
    assert len(artifacts["foundation_preprocessing_summary_df"]) == 24
    assert (tmp_path / "foundation_preprocessing_results.csv").exists()
    assert (tmp_path / "foundation_preprocessing_summary.csv").exists()
