"""Forecasting-only model catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import darts
from darts.models import (
    AutoARIMA,
    AutoETS,
    Chronos2Model,
    PatchTSTFMModel,
    TimesFM2p5Model,
)

from airquality.modeling.training_config import (
    build_lightning_trainer_kwargs,
    build_model_configs,
    resolve_training_accelerator,
)

ForecastModelMode = Literal["trained", "local", "foundation"]


@dataclass(frozen=True)
class ForecastModelConfig:
    """Construction and benchmark semantics for one forecasting model."""

    model_cls: type
    kwargs: dict[str, Any]
    mode: ForecastModelMode

    @property
    def uses_training_arms(self) -> bool:
        """Whether raw/preprocessed histories can alter this model's fit."""
        return self.mode != "foundation"


def build_forecasting_model_configs(
    *,
    seasonality_m: int = 24,
    context_length: int = 72,
) -> dict[str, ForecastModelConfig]:
    """Return trained, local-statistical, and zero-shot forecasting models."""
    if min(seasonality_m, context_length) <= 0:
        raise ValueError("seasonality_m y context_length deben ser positivos")

    configs = {
        name: ForecastModelConfig(model_cls, kwargs, "trained")
        for name, (model_cls, kwargs) in build_model_configs().items()
    }
    configs.update(
        {
            "AutoARIMA": ForecastModelConfig(
                AutoARIMA,
                {"season_length": seasonality_m},
                "local",
            ),
            "AutoETS": ForecastModelConfig(
                AutoETS,
                {"season_length": seasonality_m},
                "local",
            ),
        }
    )

    trainer_kwargs = build_lightning_trainer_kwargs(
        resolve_training_accelerator(),
        use_early_stopping=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    common = {
        "input_chunk_length": context_length,
        "enable_finetuning": False,
        "save_checkpoints": False,
        "force_reset": False,
        "pl_trainer_kwargs": trainer_kwargs,
    }
    configs.update(
        {
            "Chronos2": ForecastModelConfig(
                Chronos2Model,
                {
                    **common,
                    "hub_model_name": "amazon/chronos-2",
                    "hub_model_revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
                },
                "foundation",
            ),
            "TimesFM2p5": ForecastModelConfig(
                TimesFM2p5Model,
                {
                    **common,
                    "hub_model_name": "google/timesfm-2.5-200m-pytorch",
                    "hub_model_revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
                },
                "foundation",
            ),
            "PatchTSTFM": ForecastModelConfig(
                PatchTSTFMModel,
                {
                    **common,
                    "hub_model_name": "ibm-granite/granite-timeseries-patchtst-fm-r1",
                    "hub_model_revision": "151f9c6d576281b95c2ff784d0863bd3f12c80f1",
                },
                "foundation",
            ),
        }
    )
    return configs


def resolve_forecasting_model_configs(
    names: list[str],
    *,
    seasonality_m: int = 24,
    context_length: int = 72,
) -> dict[str, ForecastModelConfig]:
    """Validate requested names and preserve their configured order."""
    configs = build_forecasting_model_configs(
        seasonality_m=seasonality_m,
        context_length=context_length,
    )
    requested = list(dict.fromkeys(names))
    unknown = [name for name in requested if name not in configs]
    if unknown:
        raise ValueError(f"Modelo(s) de forecasting desconocido(s): {', '.join(unknown)}")
    return {name: configs[name] for name in requested}


def forecast_model_cache_identity(config: ForecastModelConfig) -> dict[str, Any]:
    """Return stable model details omitted from the shared training CFG snapshot."""
    identity: dict[str, Any] = {
        "darts_version": darts.__version__,
        "class": f"{config.model_cls.__module__}.{config.model_cls.__name__}",
        "mode": config.mode,
    }
    if config.mode != "trained":
        identity["kwargs"] = config.kwargs
    return identity


__all__ = [
    "ForecastModelConfig",
    "build_forecasting_model_configs",
    "forecast_model_cache_identity",
    "resolve_forecasting_model_configs",
]
