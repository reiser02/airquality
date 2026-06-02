from __future__ import annotations

import numpy as np

from airquality.modeling.training_config import (
    Float32StandardScaler,
    build_lightning_trainer_kwargs,
    build_model_configs,
)


def test_float32_standard_scaler_transform_dtype() -> None:
    scaler = Float32StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
    out = scaler.transform(np.array([[1.5], [2.5]]))
    assert out.dtype == np.float32


def test_build_lightning_trainer_kwargs_defaults_cpu() -> None:
    kwargs = build_lightning_trainer_kwargs(accelerator="cpu")
    assert kwargs["precision"] == "32-true"
    assert kwargs["devices"] == 1
    assert "callbacks" in kwargs


def test_build_model_configs_contains_expected_models() -> None:
    configs = build_model_configs()
    for name in ("TiDE", "NHiTS", "RNN", "LinearRegression"):
        assert name in configs
