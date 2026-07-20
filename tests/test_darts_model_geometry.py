"""Focused checks for native Darts fit and validation geometry."""

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries
from darts.models import LinearRegressionModel, NLinearModel, RNNModel, TCNModel

from airquality.forecasting.backtest import split_train_val_subseries
from airquality.modeling.training import get_model_series_requirements


@pytest.mark.parametrize(
    ("model_cls", "model_kwargs", "size_k", "expected"),
    [
        (NLinearModel, {"input_chunk_length": 12}, 4, (16, 12, 12, 4)),
        (
            TCNModel,
            {"input_chunk_length": 12, "output_chunk_shift": 3},
            4,
            (19, 12, 7, 12),
        ),
        (
            RNNModel,
            {"input_chunk_length": 48, "training_length": 72},
            4,
            (73, 48, 1, 72),
        ),
        (LinearRegressionModel, {"lags": 5}, 4, (7, 5, None, 0)),
    ],
)
def test_model_requirements_use_native_darts_geometry(
    model_cls, model_kwargs, size_k, expected
) -> None:
    requirements = get_model_series_requirements(model_cls, model_kwargs, size_k)

    assert (
        requirements.min_train_series_length,
        requirements.prediction_context_length,
        requirements.validation_target_offset,
        requirements.validation_target_length,
    ) == expected


@pytest.mark.parametrize(
    ("model_cls", "model_kwargs"),
    [
        (NLinearModel, {"input_chunk_length": 12}),
        (TCNModel, {"input_chunk_length": 12, "output_chunk_shift": 3}),
        (RNNModel, {"input_chunk_length": 5, "training_length": 12}),
    ],
)
def test_split_uses_one_native_window_with_targets_after_train(
    model_cls, model_kwargs
) -> None:
    requirements = get_model_series_requirements(model_cls, model_kwargs, size_k=4)
    index = pd.date_range("2024-01-01", periods=100, freq="h")
    train_ts = TimeSeries.from_series(pd.Series(np.arange(100.0), index=index))

    split = split_train_val_subseries(
        train_ts,
        input_chunk=requirements.prediction_context_length,
        size_k=4,
        validation_len=12,
        requirements=requirements,
    )

    assert split is not None
    train_subs, val_subs = split
    assert val_subs
    assert all(len(val) == requirements.min_train_series_length for val in val_subs)
    first_validation_target = val_subs[0].time_index[
        requirements.validation_target_offset
    ]
    assert all(train.end_time() < first_validation_target for train in train_subs)


def test_linear_regression_split_has_no_validation_series() -> None:
    requirements = get_model_series_requirements(
        LinearRegressionModel, {"lags": 5}, size_k=4
    )
    index = pd.date_range("2024-01-01", periods=20, freq="h")
    train_ts = TimeSeries.from_series(pd.Series(np.arange(20.0), index=index))

    split = split_train_val_subseries(
        train_ts,
        input_chunk=requirements.prediction_context_length,
        size_k=4,
        requirements=requirements,
    )

    assert split is not None
    train_subs, val_subs = split
    assert len(train_subs) == 1
    assert val_subs == []
