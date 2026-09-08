"""Tests for the paired synthetic foundation-context experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

import airquality.forecasting.backtest as backtest_module
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
from airquality.forecasting.fill import (
    GapImputationOutcome,
    GapImputationResult,
    nan_gap_windows,
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
        horizon=12,
        stride=6,
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
        horizon=12,
        stride=6,
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
    assert np.isfinite(result["mase"])
    assert np.isfinite(result["rmsse"])


def test_foundation_metrics_use_raw_history_before_origin(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    history = pd.Series(np.arange(200, dtype=float), index=index, name="ST")
    context = history.iloc[-72:]
    target_index = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=8, freq="h")
    target = pd.Series(np.arange(200, 208, dtype=float), index=target_index, name="ST")
    future = pd.Series(
        999.0,
        index=pd.date_range(target_index[-1] + pd.Timedelta(hours=1), periods=4, freq="h"),
        name="ST",
    )
    reference = pd.concat([history, target, future])
    scaler = Scaler(global_fit=True, scaler=StandardScaler()).fit(
        TimeSeries.from_series(history)
    )
    seen: list[pd.Series] = []

    def fake_metric(_actual, _pred, insample, *, seasonality_m):
        del seasonality_m
        seen.append(insample.copy())
        return 1.0

    monkeypatch.setattr(backtest_module, "compute_mase", fake_metric)
    monkeypatch.setattr(backtest_module, "compute_rmsse", fake_metric)

    class FakeFoundation:
        def predict(self, n, series, verbose=False):
            del series, verbose
            return TimeSeries.from_times_and_values(
                target.index,
                np.zeros(n, dtype=np.float32),
            )

    result = forecast_foundation_context(
        FakeFoundation(),
        scaler,
        context,
        target,
        reference,
        seasonality_m=24,
    )

    assert result["mase"] == 1.0 and result["rmsse"] == 1.0
    assert len(seen) == 2
    for insample in seen:
        assert insample.index.max() == target.index[0] - pd.Timedelta(hours=1)
        assert insample.index.intersection(target.index).empty
        assert insample.index.intersection(future.index).empty
    pd.testing.assert_series_equal(seen[0], seen[1])


def test_summary_reports_damage_recovery_and_residual() -> None:
    rows = []
    common = {
        "series": "ST",
        "horizon": 12,
        "model": "Chronos2",
        "case_id": "case",
        "anomaly_type": "spikes",
        "test_seed": 1001,
        "test_target_start": pd.Timestamp("2025-01-04"),
        "n_injected": 1,
        "n_injected_detected": 0,
        "imputation_applied": False,
        "n_test_predictions": 12,
    }
    for condition, mase, rmsse in (
        (CLEAN_REFERENCE, 1.0, 1.0),
        (CORRUPTED, 1.8, 2.0),
        ("unlabeled", 1.2, 1.25),
    ):
        rows.append({**common, "condition": condition, "mase": mase, "rmsse": rmsse})

    summary = summarize_foundation_preprocessing(pd.DataFrame(rows)).iloc[0]

    assert summary["rmsse_damage"] == pytest.approx(1.0)
    assert summary["rmsse_recovery"] == pytest.approx(0.75)
    assert summary["rmsse_residual"] == pytest.approx(0.25)
    assert summary["rmsse_recovery_pct"] == pytest.approx(75.0)

    failed = pd.DataFrame(rows)
    failed.loc[failed["condition"] == "unlabeled", "rmsse"] = np.nan
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
        ("forecasting", "carla_stride"): 7,
        ("forecasting", "foundation_test_seed"): 1001,
        ("forecasting", "foundation_test_repeats"): 1,
    }
    target_values: dict[tuple[pd.Timestamp, int], list[np.ndarray]] = {}
    prepare_calls: list[int] = []
    synthetic_contexts: list[dict[str, object]] = []
    foundation_references: list[pd.Series] = []
    imputation_policies: list[str] = []

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
            "1=LinearInterp"
            if (section, option) == ("forecasting", "imputation_gap_rules")
            else "drift"
            if (section, option) == ("synthetic", "injection_variant")
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
        if synthetic:
            synthetic_contexts.append(_kwargs["context_kwargs"])
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
            "_mae": 1.0,
            "_rmse": 1.0,
            "mase": 1.0,
            "rmsse": 1.0,
            "n_test_predictions": n_forecasts * horizon,
            "n_forecasts": n_forecasts,
            "n_expected_forecasts": n_forecasts,
            "n_unique_targets": target_hours,
        }

    def fake_prepare(train, _model, *, size_k, **_kwargs):
        prepare_calls.append(size_k)
        return object(), object(), train, 0.01

    def fake_forecast(_model, _scaler, context, target, insample, **_kwargs):
        key = (pd.Timestamp(target.index[0]), len(target))
        target_values.setdefault(key, []).append(target.to_numpy(copy=True))
        foundation_references.append(insample.copy())
        if context.isna().any():
            raise ValueError("missing context")
        return {
            "mase": 1.0,
            "rmsse": 1.0,
            "inference_seconds": 0.01,
            "n_test_predictions": len(target),
        }

    monkeypatch.setattr(pipeline, "_detect_for_strategies", fake_detect)
    monkeypatch.setattr(pipeline, "backtest_forecast", fake_backtest)
    monkeypatch.setattr(pipeline, "prepare_foundation_model", fake_prepare)
    monkeypatch.setattr(pipeline, "forecast_foundation_context", fake_forecast)
    def fake_impute(values, policy, *_args, **_kwargs):
        imputation_policies.append(policy.spec)
        filled = values.interpolate(method="time", limit_direction="both")
        outcomes = tuple(
            GapImputationOutcome(
                start=window[0],
                end=window[-1],
                hours=len(window),
                configured_imputer=policy.model_for(len(window)) or "none",
                effective_imputer=policy.model_for(len(window)) or "none",
                fallback_used=False,
                filled_hours=int(filled.loc[window].notna().sum()),
            )
            for window in nan_gap_windows(values)
        )
        return GapImputationResult(filled, outcomes)

    monkeypatch.setattr(pipeline, "impute_series_by_gap_result", fake_impute)

    artifacts = pipeline.run_benchmark_from_config()
    results = artifacts["foundation_preprocessing_df"]

    assert {"mase", "rmsse"}.issubset(results.columns)
    assert not {
        "mae",
        "rmse",
        "_mae",
        "_rmse",
        "relmae",
        "relrmse",
    } & set(results.columns)
    assert set(results["condition"]) == {
        CLEAN_REFERENCE,
        CORRUPTED,
        "inject-vote",
    }
    assert "regime" not in results.columns
    assert len(results) == 12
    assert set(results.loc[results["imputation_applied"], "condition"]) == {
        "inject-vote",
    }
    imputed_results = results.loc[results["imputation_applied"]]
    assert set(imputed_results["imputation_policy"]) == {"1=LinearInterp"}
    assert set(imputed_results["effective_imputer"]) == {"LinearInterp"}
    assert not imputed_results["fallback_used"].any()
    assert imputation_policies == ["1=LinearInterp"] * 4
    assert prepare_calls == [12]
    assert synthetic_contexts
    assert all(context["carla_stride"] == 7 for context in synthetic_contexts)
    assert all(context["injection_variant"] == "drift" for context in synthetic_contexts)
    assert all(
        context["cache_key"]["carla_stride"] == 7
        and context["cache_key"]["injection_variant"] == "drift"
        for context in synthetic_contexts
    )
    assert all(
        np.array_equal(values[0], candidate)
        for values in target_values.values()
        for candidate in values[1:]
    )
    assert foundation_references
    assert all(reference.index.equals(series.index) for reference in foundation_references)
    assert all(
        reference.equals(series)
        for reference in foundation_references
    )
    assert len(artifacts["foundation_preprocessing_summary_df"]) == 4
    assert (tmp_path / "foundation_preprocessing_results.csv").exists()
    assert (tmp_path / "foundation_preprocessing_summary.csv").exists()
