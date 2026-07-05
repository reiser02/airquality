"""Shared dataset containers and default training configuration builders."""

from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    DLinearModel,
    LinearRegressionModel,
    NHiTSModel,
    NLinearModel,
    RNNModel,
    TCNModel,
    TSMixerModel,
    TiDEModel,
    TransformerModel,
)
from airquality.config import cfg_get_float, cfg_get_int, cfg_get_str
from airquality.data.io import resolve_device


@dataclass(slots=True)
class TrainingDatasetBundle:
    """Estructura estándar del dataset escalado usado por training."""

    series_train: list[TimeSeries]
    series_val: list[TimeSeries]


@dataclass(slots=True)
class BenchmarkDatasetBundle:
    """Estructura estándar del dataset escalado usado por eval/benchmark."""

    series_test: list[TimeSeries]
    dict_scalers: dict[str, Scaler]
    valid_cols: list[str]
    all_series_unscaled: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida la coherencia columnas/series y la presencia de sus scalers."""
        if len(self.valid_cols) != len(self.series_test):
            raise ValueError("`valid_cols` y `series_test` deben tener la misma longitud")

        if missing := [col for col in self.valid_cols if col not in self.dict_scalers]:
            raise ValueError(
                "Faltan scalers para columnas validas en `dict_scalers`: "
                + ", ".join(missing)
            )


@dataclass(frozen=True)
class EvalConfig:
    """Configuración de evaluación para modelos globales."""

    size_k: int
    method_names: Sequence[str]
    forecast_sizes: Sequence[int] = (1, 2, 5, 10)


def build_base_training_kwargs(cfg: ConfigParser | None = None) -> dict[str, Any]:
    """Build the common optimizer and trainer kwargs used across Darts models."""
    return {
        "batch_size": cfg_get_int("training", "batch_size", 256, cfg=cfg),
        "n_epochs": cfg_get_int("training", "n_epochs", 100, cfg=cfg),
        "optimizer_cls": AdamW,
        "optimizer_kwargs": {
            "lr": cfg_get_float("training", "learning_rate", 1e-3, cfg=cfg),
            "weight_decay": cfg_get_float("training", "weight_decay", 1e-2, cfg=cfg),
        },
        "lr_scheduler_cls": ReduceLROnPlateau,
        "lr_scheduler_kwargs": {
            "mode": "min",
            "factor": cfg_get_float("training", "lr_scheduler_factor", 0.5, cfg=cfg),
            "patience": cfg_get_int("training", "lr_scheduler_patience", 2, cfg=cfg),
        },
        "save_checkpoints": True,
        "force_reset": True,
        "random_state": cfg_get_int("training", "random_state", 42, cfg=cfg),
    }


BASE_TRAINING_KWARGS: dict[str, Any] = build_base_training_kwargs()


class Float32StandardScaler(StandardScaler):
    """StandardScaler que devuelve arrays float32 tras `transform`."""

    def transform(self, X: Any, copy: bool | None = None) -> Any:
        """Transforma como StandardScaler y castea la salida a float32."""
        transformed = super().transform(X, copy=copy)
        return transformed.astype(np.float32, copy=False)


def build_early_stopping_callback(cfg: ConfigParser | None = None) -> EarlyStopping:
    """Construye EarlyStopping para minimizar `val_loss` durante entrenamiento."""

    return EarlyStopping(
        monitor="val_loss",
        patience=cfg_get_int("training", "early_stopping_patience", 5, cfg=cfg),
        min_delta=cfg_get_float("training", "early_stopping_min_delta", 1e-4, cfg=cfg),
        mode="min",
        verbose=True,
    )


def build_lightning_trainer_kwargs(
    accelerator: str = "gpu",
    use_early_stopping: bool = True,
    precision: str | int | None = None,
    devices: int | str | list[int] | None = None,
    enable_progress_bar: bool = True,
    enable_checkpointing: bool = True,
    enable_model_summary: bool = True,
    logger: bool | Any = True,
    cfg: ConfigParser | None = None,
) -> dict[str, Any]:
    """Genera kwargs de PyTorch Lightning para modelos Darts basados en Torch."""

    if precision is None:
        precision = "16-mixed" if accelerator == "gpu" else "32-true"
    if devices is None:
        devices = 1

    kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "enable_progress_bar": enable_progress_bar,
        "enable_checkpointing": enable_checkpointing,
        "enable_model_summary": enable_model_summary,
        "logger": logger,
    }
    if use_early_stopping:
        kwargs["callbacks"] = [build_early_stopping_callback(cfg=cfg)]
    return kwargs


def resolve_training_accelerator(cfg: ConfigParser | None = None) -> str:
    """Resuelve el accelerator de entrenamiento desde `[training] accelerator`.

    Valores admitidos: ``auto`` (por defecto: GPU si hay CUDA disponible, CPU si
    no), ``gpu`` o ``cpu``. Antes estaba fijado a ``"gpu"`` y el entrenamiento
    reventaba en máquinas sin CUDA.
    """
    requested = cfg_get_str("training", "accelerator", "auto", cfg=cfg).strip().lower()
    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError(
            f"`[training] accelerator` debe ser 'auto', 'gpu' o 'cpu'; recibido: {requested!r}"
        )
    if requested == "auto":
        return "gpu" if resolve_device("cuda") == "cuda" else "cpu"
    return requested


def make_encoders_full() -> dict[str, Any]:
    """Encoders temporales completos (pasado/futuro) para modelos globales."""

    return {
        "cyclic": {"future": ["month"]},
        "datetime_attribute": {"future": ["hour", "dayofweek"]},
        "position": {"past": ["relative"], "future": ["relative"]},
        "transformer": Scaler(scaler=Float32StandardScaler()),
    }


def make_encoders_past_only() -> dict[str, Any]:
    """Encoders mínimos basados solo en posición relativa del pasado."""

    return {
        "position": {"past": ["relative"]},
        "transformer": Scaler(scaler=Float32StandardScaler()),
    }


def make_encoders_rnn() -> dict[str, Any]:
    """Encoders orientados a RNN con covariables futuras y posición relativa."""

    return {
        "cyclic": {"future": ["month"]},
        "datetime_attribute": {"future": ["hour", "dayofweek"]},
        "position": {"future": ["relative"]},
        "transformer": Scaler(scaler=Float32StandardScaler()),
    }


def build_model_configs(cfg: ConfigParser | None = None) -> dict[str, tuple[type, dict[str, Any]]]:
    """Devuelve el catálogo de modelos y sus hiperparámetros por defecto."""

    accelerator = resolve_training_accelerator(cfg)
    return {
        "TiDE": (
            TiDEModel,
            {
                "input_chunk_length": cfg_get_int("models", "tide_input_chunk_length", 72, cfg=cfg),
                "temporal_width_past": 1,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "decoder_output_dim": 64,
                "hidden_size": cfg_get_int("models", "tide_hidden_size", 512, cfg=cfg),
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "NHiTS": (
            NHiTSModel,
            {
                "input_chunk_length": cfg_get_int("models", "nhits_input_chunk_length", 72, cfg=cfg),
                "num_stacks": 4,
                "num_blocks": 3,
                "layer_widths": cfg_get_int("models", "nhits_layer_widths", 512, cfg=cfg),
                "add_encoders": make_encoders_past_only(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "NLinear": (
            NLinearModel,
            {
                "input_chunk_length": cfg_get_int("models", "nlinear_input_chunk_length", 72, cfg=cfg),
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "DLinear": (
            DLinearModel,
            {
                "input_chunk_length": cfg_get_int("models", "dlinear_input_chunk_length", 72, cfg=cfg),
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "TCN": (
            TCNModel,
            {
                "input_chunk_length": cfg_get_int("models", "tcn_input_chunk_length", 72, cfg=cfg),
                "num_filters": cfg_get_int("models", "tcn_num_filters", 16, cfg=cfg),
                "add_encoders": make_encoders_past_only(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "Transformer": (
            TransformerModel,
            {
                "input_chunk_length": cfg_get_int("models", "transformer_input_chunk_length", 72, cfg=cfg),
                "add_encoders": make_encoders_past_only(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "TSMixer": (
            TSMixerModel,
            {
                "input_chunk_length": cfg_get_int("models", "tsmixer_input_chunk_length", 72, cfg=cfg),
                "hidden_size": cfg_get_int("models", "tsmixer_hidden_size", 128, cfg=cfg),
                "ff_size": cfg_get_int("models", "tsmixer_ff_size", 128, cfg=cfg),
                "num_blocks": 3,
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "RNN": (
            RNNModel,
            {
                "input_chunk_length": cfg_get_int("models", "rnn_input_chunk_length", 48, cfg=cfg),
                "training_length": cfg_get_int("models", "rnn_training_length", 72, cfg=cfg),
                "model": "GRU",
                "hidden_dim": cfg_get_int("models", "rnn_hidden_dim", 64, cfg=cfg),
                "n_rnn_layers": 3,
                "dropout": 0.1,
                "add_encoders": make_encoders_rnn(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs(accelerator, True, cfg=cfg),
            },
        ),
        "LinearRegression": (
            LinearRegressionModel,
            {
                "lags": cfg_get_int("models", "linear_regression_lags", 72, cfg=cfg),
                "random_state": cfg_get_int("training", "random_state", 42, cfg=cfg),
            },
        ),
    }
