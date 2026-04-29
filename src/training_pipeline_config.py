from dataclasses import dataclass
from typing import Any, NotRequired, Sequence, TypedDict

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


class DatasetBundle(TypedDict):
    """Estructura estándar del dataset escalado usado por training/eval."""

    series_train: list[TimeSeries]
    series_val: list[TimeSeries]
    series_test: list[TimeSeries]
    dict_scalers: dict[str, Scaler]
    valid_cols: list[str]
    all_series_unscaled: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class EvalConfig:
    """Configuración de evaluación para modelos globales."""

    size_k: int
    method_names: Sequence[str]
    forecast_sizes: Sequence[int] = (1, 2, 5, 10)


BASE_TRAINING_KWARGS: dict[str, Any] = {
    "batch_size": 256,
    "n_epochs": 100,
    "optimizer_cls": AdamW,
    "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 1e-2},
    "lr_scheduler_cls": ReduceLROnPlateau,
    "lr_scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 2},
    "save_checkpoints": True,
    "force_reset": True,
    "random_state": 42,
}


class Float32StandardScaler(StandardScaler):
    """StandardScaler que devuelve arrays float32 tras `transform`."""

    def transform(self, X: Any, copy: bool | None = None) -> Any:
        transformed = super().transform(X, copy=copy)
        return transformed.astype(np.float32, copy=False)


def build_early_stopping_callback() -> EarlyStopping:
    """Construye EarlyStopping para minimizar `val_loss` durante entrenamiento."""

    return EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-4,
        mode="min",
        verbose=True,
    )


def build_lightning_trainer_kwargs(
    accelerator: str = "gpu",
    use_early_stopping: bool = True,
    precision: str | int | None = None,
    devices: int | str | list[int] | None = None,
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
        "enable_progress_bar": True,
    }
    if use_early_stopping:
        kwargs["callbacks"] = [build_early_stopping_callback()]
    return kwargs


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


def build_model_configs() -> dict[str, tuple[type, dict[str, Any]]]:
    """Devuelve el catálogo de modelos y sus hiperparámetros por defecto."""

    return {
        "TiDE": (
            TiDEModel,
            {
                "input_chunk_length": 72,
                "temporal_width_past": 1,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "decoder_output_dim": 64,
                "hidden_size": 512,
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "NHiTS": (
            NHiTSModel,
            {
                "input_chunk_length": 72,
                "num_stacks": 4,
                "num_blocks": 3,
                "layer_widths": 512,
                "add_encoders": make_encoders_past_only(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "NLinear": (
            NLinearModel,
            {
                "input_chunk_length": 72,
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "DLinear": (
            DLinearModel,
            {
                "input_chunk_length": 72,
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "TCN": (
            TCNModel,
            {
                "input_chunk_length": 72,
                "num_filters": 16,
                "add_encoders": make_encoders_past_only(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "Transformer": (
            TransformerModel,
            {
                "input_chunk_length": 72,
                "add_encoders": make_encoders_past_only(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "TSMixer": (
            TSMixerModel,
            {
                "input_chunk_length": 72,
                "hidden_size": 128,
                "ff_size": 128,
                "num_blocks": 3,
                "add_encoders": make_encoders_full(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "RNN": (
            RNNModel,
            {
                "input_chunk_length": 48,
                "training_length": 72,
                "model": "GRU",
                "hidden_dim": 64,
                "n_rnn_layers": 3,
                "dropout": 0.1,
                "add_encoders": make_encoders_rnn(),
                "loss_fn": MSELoss(),
                "pl_trainer_kwargs": build_lightning_trainer_kwargs("gpu", True),
            },
        ),
        "LinearRegression": (
            LinearRegressionModel,
            {
                "lags": 72,
                "random_state": 42,
            },
        ),
    }
