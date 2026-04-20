import logging
import os
import tempfile
import warnings
import gc
from copy import deepcopy
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.nn import HuberLoss

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    TiDEModel,
    RNNModel,
    RegressionEnsembleModel,
    LinearRegressionModel,
)
from darts.utils.missing_values import extract_subseries

from training_pipeline_config import (
    BASE_TRAINING_KWARGS,
    DatasetBundle,
    build_lightning_trainer_kwargs,
    build_model_configs,
    make_encoders_full,
)


def configure_runtime_warnings() -> None:
    """Reduce warnings/logs ruidosos de dependencias durante entrenamiento."""

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", ".*isinstance\\(treespec, LeafSpec\\).*")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def build_scaled_train_val_series(
    series_dfs: Sequence[pd.DataFrame],
    val_size: int = 10,
    min_train_len: int = 82,
) -> tuple[list[TimeSeries], list[TimeSeries], dict[str, Scaler]]:
    """
    Construye series de train/validación escaladas a partir de subseries continuas.

    Parameters
    ----------
    series_dfs : Sequence[pd.DataFrame]
        Lista de DataFrames de una sola columna (una serie por DataFrame), cada
        uno con su propio índice temporal. No requiere timestamps compartidos
        entre series.
    Returns
    -------
    tuple[list[TimeSeries], list[TimeSeries], dict[str, Scaler]]
        `series_train`, `series_val` y escaladores por nombre de serie.
    """
    dict_scalers = {}
    train_series_list, val_series_list = [], []

    for series_df in series_dfs:
        if len(series_df.columns) != 1:
            raise ValueError(
                "Cada elemento de `series_dfs` debe tener exactamente una columna."
            )

        col = str(series_df.columns[0])
        series = series_df.iloc[:, 0].copy()
        series.name = col

        ts = TimeSeries.from_series(series, freq="h")
        subseries_raw = extract_subseries(ts, min_gap_size=1)
        subseries_validas = [
            s for s in subseries_raw if len(s) >= (min_train_len + val_size)
        ]

        if not subseries_validas:
            continue

        sc = Scaler(global_fit=True, scaler=StandardScaler()).fit(subseries_validas)
        dict_scalers[col] = sc

        for s in subseries_validas:
            s_scaled = sc.transform(s)
            train_part, _ = s_scaled.split_after(len(s_scaled) - val_size - 1)
            train_series_list.append(train_part)
            val_series_list.append(s_scaled)

    return train_series_list, val_series_list, dict_scalers


def build_train_val_test_series(
    series_dfs: Sequence[pd.DataFrame],
    longest_segment: pd.DataFrame,
    val_size: int = 10,
    min_train_len: int = 82,
) -> DatasetBundle:
    """
    Construye train/val/test escalados en un solo paso.

    Para evitar fuga de datos, elimina (pone NaN) el bloque temporal de
    `longest_segment` dentro de cada serie que esté presente en ese segmento
    antes de construir train/val. Luego aplica los mismos scalers al segmento
    de test para evitar repetir el flujo en distintos sitios.

    Además devuelve `all_series_unscaled` (solo para columnas válidas), útil
    para pipelines de imputación que necesitan recuperar historial de train a
    partir del propio `dataset_bundle` sin tener que pasar `all_series` aparte.
    """
    series_train_input: list[pd.DataFrame] = []
    seen_names: set[str] = set()
    all_series_unscaled: dict[str, pd.Series] = {}

    for series_df in series_dfs:
        if len(series_df.columns) != 1:
            raise ValueError(
                "Cada elemento de `series_dfs` debe tener exactamente una columna."
            )

        col = str(series_df.columns[0])
        if col in seen_names:
            raise ValueError(f"Nombre de serie duplicado en `series_dfs`: {col}")
        seen_names.add(col)

        series_full = series_df.iloc[:, 0].astype(float).copy()
        series_full.name = col
        all_series_unscaled[col] = series_full.copy()

        series_copy = series_full.copy()
        if col in longest_segment.columns:
            test_rows = series_copy.index.intersection(longest_segment.index)
            series_copy.loc[test_rows] = np.nan

        series_train_input.append(series_copy.to_frame())

    series_train, series_val, dict_scalers = build_scaled_train_val_series(
        series_train_input,
        val_size=val_size,
        min_train_len=min_train_len,
    )

    valid_cols = [c for c in longest_segment.columns if c in dict_scalers]
    series_test = [
        dict_scalers[col].transform(
            TimeSeries.from_series(longest_segment[col], freq="h")
        )
        for col in valid_cols
    ]

    if not valid_cols:
        raise ValueError(
            "No hay columnas válidas para test tras construir train/val sin fuga de datos."
        )

    return {
        "series_train": series_train,
        "series_val": series_val,
        "series_test": series_test,
        "dict_scalers": dict_scalers,
        "valid_cols": valid_cols,
        "all_series_unscaled": {
            col: all_series_unscaled[col].copy()
            for col in valid_cols
            if col in all_series_unscaled
        },
    }


def fit_darts_model(
    model_cls: type,
    series_train: list[TimeSeries],
    series_val: list[TimeSeries] | None,
    size_k: int,
    model_kwargs: dict[str, Any],
) -> Any:
    """
    Instancia y entrena un modelo Darts con configuración base + overrides.

    Aplica `output_chunk_length=size_k` cuando el modelo lo requiere y no viene
    explícito en `model_kwargs`.
    """
    configure_runtime_warnings()
    kwargs = deepcopy(BASE_TRAINING_KWARGS)
    kwargs.update(model_kwargs)

    # output_chunk_length no aplica a algunos modelos; se pone condicionalmente.
    if (
        "output_chunk_length" not in kwargs
        and model_cls is not LinearRegressionModel
        and model_cls is not RNNModel
    ):
        kwargs["output_chunk_length"] = size_k

    model = model_cls(**kwargs)

    fit_kwargs = {"series": series_train, "verbose": True}
    if series_val is not None and model_cls is not LinearRegressionModel:
        fit_kwargs["val_series"] = series_val
        fit_kwargs["dataloader_kwargs"] = {"num_workers": 2}

    model.fit(**fit_kwargs)
    return model


def train_tide_lr_ensemble(
    series_train: list[TimeSeries],
    series_val: list[TimeSeries] | None,
    size_k: int,
) -> RegressionEnsembleModel:
    """Entrena un ensemble de TiDE + LinearRegression con meta-regresión."""

    configure_runtime_warnings()

    tide = fit_darts_model(
        TiDEModel,
        series_train,
        series_val,
        size_k,
        model_kwargs={
            "input_chunk_length": 48,
            "output_chunk_length": size_k,
            "n_epochs": 1,
            "temporal_width_past": 1,
            "add_encoders": make_encoders_full(),
            "loss_fn": HuberLoss(),
            "pl_trainer_kwargs": build_lightning_trainer_kwargs("cpu", False),
        },
    )

    lr_model = LinearRegressionModel(
        lags=48, output_chunk_length=tide.output_chunk_length
    ).fit(series=series_train)

    ensemble = RegressionEnsembleModel(
        forecasting_models=[tide, lr_model],
        regression_train_n_points=24,
        train_forecasting_models=False,
    )
    ensemble.fit(series=series_train)
    return ensemble


def finetune_trained_models(
    trained_models: dict[str, Any],
    series_train: list[TimeSeries],
    series_val: list[TimeSeries] | None = None,
    n_epochs: int = 5,
    enable_finetuning: Any = True,
    model_specific_finetuning: dict[str, Any] | None = None,
    load_best: bool = True,
    dataloader_num_workers: int = 2,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Hace fine-tuning de modelos ya entrenados siguiendo el flujo recomendado
    por Darts para TorchForecastingModel y Foundation Models:

    1) Crear una nueva instancia con `enable_finetuning`.
    2) Cargar los pesos pre-entrenados con `load_weights()`.
    3) Ajustar con `fit(..., epochs=...)`.

    - Modelos Torch (Darts): continúan entrenamiento por `n_epochs`.
    - Modelos no iterativos (p. ej. LinearRegression/Ensemble): se omiten.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str]]
        (modelos_finetuned, modelos_omitidos_con_motivo)
    """
    configure_runtime_warnings()

    skipped_models: dict[str, str] = {}

    for name, model in trained_models.items():
        if isinstance(model, (LinearRegressionModel, RegressionEnsembleModel)):
            skipped_models[name] = "Modelo no iterativo para fine-tuning incremental"
            continue

        if not hasattr(model, "save") or not hasattr(model, "load_weights"):
            skipped_models[name] = (
                "Modelo sin soporte de save/load_weights para fine-tuning"
            )
            continue

        finetuning_cfg = enable_finetuning
        if model_specific_finetuning and name in model_specific_finetuning:
            finetuning_cfg = model_specific_finetuning[name]

        model_params = dict(model.model_params)
        if finetuning_cfg is not None:
            model_params["enable_finetuning"] = finetuning_cfg

        model_cls = model.__class__

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            weights_path = tmp.name

        try:
            model.save(weights_path)

            # Liberar memoria del modelo original antes de crear el nuevo.
            trained_models[name] = None
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            finetune_model = model_cls(**model_params)
            finetune_model.load_weights(weights_path)
        finally:
            if os.path.exists(weights_path):
                os.remove(weights_path)

        fit_kwargs = {
            "series": series_train,
            "verbose": verbose,
            "epochs": n_epochs,
            "load_best": load_best,
        }

        if series_val is not None:
            fit_kwargs["val_series"] = series_val
            fit_kwargs["dataloader_kwargs"] = {"num_workers": dataloader_num_workers}

        finetune_model.fit(**fit_kwargs)
        trained_models[name] = finetune_model

    return trained_models, skipped_models


def finetune_models_from_data(
    series_dfs: Sequence[pd.DataFrame],
    longest_segment: pd.DataFrame,
    trained_models: dict[str, Any],
    n_epochs: int = 5,
    enable_finetuning: Any = True,
    model_specific_finetuning: dict[str, Any] | None = None,
    load_best: bool = True,
    dataloader_num_workers: int = 2,
    verbose: bool = True,
    val_size: int = 10,
    min_train_len: int = 82,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Wrapper de conveniencia para fine-tuning cuando no se dispone de
    `series_train`/`series_val` explícitos.

    Reconstruye las series con `build_train_val_test_series()` usando el mismo
    flujo que `train_global_methods()` y luego aplica
    `finetune_trained_models()`.
    """
    dataset_bundle = build_train_val_test_series(
        series_dfs,
        longest_segment,
        val_size=val_size,
        min_train_len=min_train_len,
    )

    return finetune_trained_models(
        trained_models=trained_models,
        series_train=dataset_bundle["series_train"],
        series_val=dataset_bundle["series_val"],
        n_epochs=n_epochs,
        enable_finetuning=enable_finetuning,
        model_specific_finetuning=model_specific_finetuning,
        load_best=load_best,
        dataloader_num_workers=dataloader_num_workers,
        verbose=verbose,
    )


def train_global_methods(
    dataset_bundle: DatasetBundle,
    size_k: int,
    method_names: Sequence[str],
) -> dict[str, Any]:
    """
    Entrena métodos globales y devuelve modelos entrenados por nombre.
    """
    trained_models: dict[str, Any] = {}
    model_configs = build_model_configs()

    series_train = dataset_bundle["series_train"]
    series_val = dataset_bundle["series_val"]

    for name in method_names:
        print(f"Entrenando {name}")
        if name == "Ensemble":
            model = train_tide_lr_ensemble(
                series_train,
                series_val,
                size_k,
            )
        else:
            model_cls, model_kwargs = model_configs[name]
            model = fit_darts_model(
                model_cls,
                series_train,
                series_val,
                size_k,
                model_kwargs,
            )

        trained_models[name] = model

    return trained_models


# Backward compatibility for existing notebooks/scripts.
setup_runtime_warnings = configure_runtime_warnings
prepare_series = build_scaled_train_val_series
train_ensemble_model = train_tide_lr_ensemble
evaluate_global_methods = train_global_methods
eval_global_methods = train_global_methods
