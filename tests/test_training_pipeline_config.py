"""Tests training configuration defaults, scaler behavior, and model catalog contents."""

from __future__ import annotations

import numpy as np
import pytest

from airquality.modeling.training_config import (
    Float32StandardScaler,
    build_lightning_trainer_kwargs,
    build_model_configs,
    resolve_training_accelerator,
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


def test_resolve_training_accelerator_auto_falls_back_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "airquality.modeling.training_config.cfg_get_str",
        lambda section, option, default, cfg=None: "auto",
    )
    monkeypatch.setattr(
        "airquality.modeling.training_config.resolve_device", lambda preferred: "cpu"
    )
    assert resolve_training_accelerator() == "cpu"

    monkeypatch.setattr(
        "airquality.modeling.training_config.resolve_device", lambda preferred: "cuda"
    )
    assert resolve_training_accelerator() == "gpu"


def test_resolve_training_accelerator_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "airquality.modeling.training_config.cfg_get_str",
        lambda section, option, default, cfg=None: "tpu",
    )
    with pytest.raises(ValueError, match="accelerator"):
        resolve_training_accelerator()


def test_build_model_configs_uses_resolved_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the torch models used to hardcode accelerator="gpu", which
    # made training crash on machines without CUDA.
    monkeypatch.setattr(
        "airquality.modeling.training_config.resolve_training_accelerator",
        lambda cfg=None: "cpu",
    )
    configs = build_model_configs()
    for name, (_, kwargs) in configs.items():
        trainer_kwargs = kwargs.get("pl_trainer_kwargs")
        if trainer_kwargs is None:  # LinearRegression has no torch trainer
            continue
        assert trainer_kwargs["accelerator"] == "cpu", name
        assert trainer_kwargs["precision"] == "32-true", name
