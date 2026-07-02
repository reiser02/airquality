"""Tests dataset-bundle construction and benchmark reuse of held-out series data."""

from __future__ import annotations

import pandas as pd
from darts import TimeSeries

from airquality.imputation.benchmark import execute_complete_pipeline
from airquality.imputation.imputers import DartsGlobalGapImputer
from airquality.modeling.training import build_benchmark_dataset_bundle


class LastValueModel:
    input_chunk_length = 1

    def predict(self, n, series, **kwargs):
        del kwargs
        last_value = float(series.to_series().iloc[-1])
        index = pd.date_range(
            start=series.end_time() + series.freq,
            periods=int(n),
            freq=series.freq_str,
        )
        return TimeSeries.from_series(pd.Series(last_value, index=index), freq=series.freq_str)


def _frame(name: str, values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({name: values}, index=index)


def _series(name: str, values: list[float], start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(values, index=index, name=name)


def test_build_benchmark_dataset_bundle_includes_test_only_series_without_leakage() -> None:
    train_eval_df = _frame("train_eval", list(range(10)))
    test_only_series = _series("test_only", list(range(10, 20)))
    longest_segment = train_eval_df.iloc[-4:].copy()

    bundle = build_benchmark_dataset_bundle(
        [train_eval_df],
        longest_segment,
        val_size=2,
        min_train_len=3,
        val_context_len=1,
        test_only_series=[test_only_series],
        test_only_train_fraction=0.6,
    )

    assert bundle.valid_cols == ["train_eval", "test_only"]
    assert set(bundle.dict_scalers) == {"train_eval", "test_only"}
    pd.testing.assert_series_equal(
        bundle.all_series_unscaled["test_only"],
        test_only_series,
        check_dtype=False,
    )

    test_only_idx = bundle.valid_cols.index("test_only")
    transformed_suffix = bundle.series_test[test_only_idx].to_series().astype(float)
    raw_suffix = test_only_series.iloc[6:].astype(float)

    pd.testing.assert_index_equal(transformed_suffix.index, raw_suffix.index)
    restored_suffix = (
        bundle.dict_scalers["test_only"]
        .inverse_transform(bundle.series_test[test_only_idx])
        .to_series()
        .astype(float)
    )
    pd.testing.assert_series_equal(restored_suffix, raw_suffix, check_names=False)


def test_execute_complete_pipeline_uses_bundle_history_for_test_only_series() -> None:
    train_eval_df = _frame("train_eval", list(range(12)))
    test_only_series = _series("test_only", list(range(20, 32)))
    longest_segment = train_eval_df.iloc[-4:].copy()

    bundle = build_benchmark_dataset_bundle(
        [train_eval_df],
        longest_segment,
        val_size=2,
        min_train_len=3,
        val_context_len=1,
        test_only_series=[test_only_series],
        test_only_train_fraction=0.6,
    )

    results_df, plot_store = execute_complete_pipeline(
        model_dict={"LastValue": DartsGlobalGapImputer(LastValueModel(), model_name="LastValue")},
        dataset_bundle=bundle,
        gap_sizes=(1,),
        num_gaps=1,
        metrics=("mae",),
        random_seed=0,
        freq="h",
        seasonality_m=1,
        gap_strategy="block",
    )

    assert set(results_df["Serie"]) == {"train_eval", "test_only"}
    assert "LastValue" in plot_store[1]["series"]["test_only"]["preds"]


def test_execute_complete_pipeline_fails_without_all_series_history() -> None:
    train_eval_df = _frame("train_eval", list(range(12)))
    longest_segment = train_eval_df.iloc[-4:].copy()

    bundle = build_benchmark_dataset_bundle(
        [train_eval_df],
        longest_segment,
        val_size=2,
        min_train_len=3,
        val_context_len=1,
    )
    bundle.all_series_unscaled = {}

    try:
        execute_complete_pipeline(
            {"LastValue": DartsGlobalGapImputer(LastValueModel(), model_name="LastValue")},
            bundle,
            gap_sizes=(1,),
            num_gaps=1,
            metrics=("mae",),
            random_seed=0,
            freq="h",
            seasonality_m=1,
            gap_strategy="block",
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "all_series_unscaled" in str(exc)
