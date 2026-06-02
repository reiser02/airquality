import os
import tempfile
import gc
import time
import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from airquality.data.io import configure_warnings

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    RNNModel,
    LinearRegressionModel,
)
from darts.utils.missing_values import extract_subseries

from airquality.modeling.training_config import (
    BASE_TRAINING_KWARGS,
    DatasetBundle,
    build_model_configs,
)


def build_scaled_train_val_series(
    series_dfs: Sequence[pd.DataFrame],
    val_size: int,
    min_train_len: int,
    val_context_len: int = 72,
) -> tuple[list[TimeSeries], list[TimeSeries], dict[str, Scaler]]:
    """
    Construye series de train/validación escaladas a partir de subseries continuas.

    Parameters
    ----------
    series_dfs : Sequence[pd.DataFrame]
        Lista de DataFrames de una sola columna (una serie por DataFrame), cada
        uno con su propio índice temporal. No requiere timestamps compartidos
        entre series.
    val_size : int
        Cantidad de puntos de validacion por bloque continuo.
    val_context_len : int
        Cantidad de puntos de contexto inmediatamente anteriores al bloque de
        validacion que se conservan en `series_val`.
    Returns
    -------
    tuple[list[TimeSeries], list[TimeSeries], dict[str, Scaler]]
        `series_train`, `series_val` y escaladores por nombre de serie.
    """
    val_points = int(val_size)
    if val_points < 0:
        raise ValueError("`val_size` no puede ser negativo.")

    val_context_points = int(val_context_len)
    if val_context_points < 0:
        raise ValueError("`val_context_len` no puede ser negativo.")

    dict_scalers: dict[str, Scaler] = {}
    train_series_list: list[TimeSeries] = []
    val_series_list: list[TimeSeries] = []

    for series_df in series_dfs:
        col = str(series_df.columns[0])
        series = series_df.iloc[:, 0].astype(np.float32).copy()
        series.name = col

        ts = TimeSeries.from_series(series, freq="h")
        subseries_raw = extract_subseries(ts, min_gap_size=1)
        if val_points > 0:
            min_required = max(min_train_len, val_context_points) + val_points
        else:
            min_required = min_train_len
        subseries_validas = [s for s in subseries_raw if len(s) >= min_required]

        if len(subseries_validas) < 1:
            continue

        # Split intra-bloque: train usa todo salvo la cola de validacion.
        # Val conserva solo contexto inmediato + la cola de validacion para
        # evitar evaluar con toda la subserie completa.
        if val_points == 0:
            train_subseries = list(subseries_validas)
            val_subseries: list[TimeSeries] = []
        else:
            train_subseries = []
            val_subseries = []
            val_window_len = val_context_points + val_points

            for s in subseries_validas:
                train_s = s.split_after(len(s) - val_points - 1)[0]
                if val_window_len == len(s):
                    val_s = s
                else:
                    val_s = s.split_after(len(s) - val_window_len - 1)[1]

                train_subseries.append(train_s)
                val_subseries.append(val_s)

        if not train_subseries:
            continue

        sc = Scaler(global_fit=True, scaler=StandardScaler()).fit(train_subseries)
        dict_scalers[col] = sc

        if val_points == 0:
            for s in train_subseries:
                train_series_list.append(sc.transform(s).astype(np.float32))
        else:
            for train_s, val_s in zip(train_subseries, val_subseries, strict=True):
                train_series_list.append(sc.transform(train_s).astype(np.float32))
                val_series_list.append(sc.transform(val_s).astype(np.float32))

    return train_series_list, val_series_list, dict_scalers


def build_train_val_test_series(
    series_dfs: Sequence[pd.DataFrame],
    longest_segment: pd.DataFrame,
    val_size: int,
    min_train_len: int,
    val_context_len: int = 72,
) -> DatasetBundle:
    """
    Construye train/val/test escalados en un solo paso.

    Para evitar fuga de datos, elimina (pone NaN) el bloque temporal de
    `longest_segment` dentro de cada serie que esté presente en ese segmento
    antes de construir train/val. Luego aplica los mismos scalers al segmento
    de test para evitar repetir el flujo en distintos sitios.

    `val_size` define la cola (en puntos) reservada para validacion en cada
    bloque continuo.

    `val_context_len` define cuántos puntos previos a esa cola se conservan en
    cada serie de validacion como contexto de entrada.

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

        series_full = series_df.iloc[:, 0].astype(np.float32).copy()
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
        val_context_len=val_context_len,
    )

    valid_cols = [c for c in longest_segment.columns if c in dict_scalers]
    series_test = [
        dict_scalers[col]
        .transform(TimeSeries.from_series(longest_segment[col], freq="h"))
        .astype(np.float32)
        for col in valid_cols
    ]

    if not valid_cols:
        raise ValueError(
            "No hay columnas válidas para test tras construir train/val sin fuga de datos."
        )

    return DatasetBundle(
        series_train=series_train,
        series_val=series_val,
        series_test=series_test,
        dict_scalers=dict_scalers,
        valid_cols=valid_cols,
        all_series_unscaled={
            col: all_series_unscaled[col].copy()
            for col in valid_cols
            if col in all_series_unscaled
        },
    )


def _metric_to_float(metric_value: Any) -> float | None:
    if metric_value is None:
        return None

    value = metric_value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_metric_from_callback_metrics(
    callback_metrics: Any,
    candidate_names: Sequence[str],
) -> float | None:
    if callback_metrics is None:
        return None

    for name in candidate_names:
        value = None
        if hasattr(callback_metrics, "get"):
            value = callback_metrics.get(name)
        elif isinstance(callback_metrics, dict):
            value = callback_metrics.get(name)

        value_float = _metric_to_float(value)
        if value_float is not None:
            return value_float

    return None


class LossHistoryCallback(Callback):
    """Captura train/val loss por época desde PyTorch Lightning."""

    def __init__(self) -> None:
        super().__init__()
        self.train_loss_by_epoch: dict[int, float] = {}
        self.val_loss_by_epoch: dict[int, float] = {}

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        epoch = int(getattr(trainer, "current_epoch", 0))
        train_loss = _read_metric_from_callback_metrics(
            getattr(trainer, "callback_metrics", None),
            ("train_loss", "train_loss_epoch"),
        )
        if train_loss is not None:
            self.train_loss_by_epoch[epoch] = train_loss

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if bool(getattr(trainer, "sanity_checking", False)):
            return

        epoch = int(getattr(trainer, "current_epoch", 0))
        val_loss = _read_metric_from_callback_metrics(
            getattr(trainer, "callback_metrics", None),
            ("val_loss", "val_loss_epoch"),
        )
        if val_loss is not None:
            self.val_loss_by_epoch[epoch] = val_loss


def _attach_loss_history_callback(
    model_cls: type,
    model_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], LossHistoryCallback | None]:
    kwargs = deepcopy(model_kwargs)

    if model_cls is LinearRegressionModel:
        return kwargs, None

    pl_trainer_kwargs = deepcopy(kwargs.get("pl_trainer_kwargs"))
    if pl_trainer_kwargs is None:
        return kwargs, None

    callbacks = list(pl_trainer_kwargs.get("callbacks", []))
    loss_callback = LossHistoryCallback()
    callbacks.append(loss_callback)
    pl_trainer_kwargs["callbacks"] = callbacks
    kwargs["pl_trainer_kwargs"] = pl_trainer_kwargs

    return kwargs, loss_callback


def _build_curve_rows(
    model_name: str,
    training_time_seconds: float,
    loss_callback: LossHistoryCallback | None,
) -> list[dict[str, Any]]:
    base_row = {
        "model_name": model_name,
        "training_time_seconds": float(training_time_seconds),
    }

    if loss_callback is None:
        return [
            {
                **base_row,
                "epoch": 0,
                "train_loss": np.nan,
                "val_loss": np.nan,
            }
        ]

    epochs = sorted(
        set(loss_callback.train_loss_by_epoch) | set(loss_callback.val_loss_by_epoch)
    )
    if not epochs:
        return [
            {
                **base_row,
                "epoch": 0,
                "train_loss": np.nan,
                "val_loss": np.nan,
            }
        ]

    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        rows.append(
            {
                **base_row,
                "epoch": int(epoch),
                "train_loss": float(
                    loss_callback.train_loss_by_epoch.get(epoch, np.nan)
                ),
                "val_loss": float(loss_callback.val_loss_by_epoch.get(epoch, np.nan)),
            }
        )
    return rows


def _merge_curve_rows_with_existing_csv(
    curve_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Combina métricas actuales con CSV previo reemplazando solo modelos repetidos."""

    if not output_path.exists():
        return curve_df

    try:
        existing_df = pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return curve_df

    if "model_name" not in existing_df.columns:
        return curve_df

    curve_columns = list(curve_df.columns)
    existing_df = existing_df.reindex(columns=curve_columns)

    model_names_to_replace = set(curve_df["model_name"].astype(str).unique())
    existing_df = existing_df[
        ~existing_df["model_name"].astype(str).isin(model_names_to_replace)
    ]

    merged_df = pd.concat([existing_df, curve_df], ignore_index=True)
    return merged_df.sort_values(["model_name", "epoch"], ignore_index=True)


def _filter_model_init_kwargs(
    model_cls: type,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Filtra kwargs para que coincidan con la firma de __init__ del modelo."""

    params = inspect.signature(model_cls.__init__).parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kwargs:
        return kwargs

    accepted_names = {
        name
        for name, param in params.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in accepted_names}


def fit_darts_model(
    model_cls: type,
    series_train: list[TimeSeries],
    series_val: list[TimeSeries] | None,
    size_k: int,
    model_kwargs: dict[str, Any],
    resume_mode: str | None = None,
) -> Any:
    """
    Instancia y entrena un modelo Darts con configuración base + overrides.

    Aplica `output_chunk_length=size_k` cuando el modelo lo requiere y no viene
    explícito en `model_kwargs`.

    `resume_mode` permite reanudar manualmente desde checkpoints:
    - `None`: entrenamiento desde cero.
    - `"last"`: carga el último checkpoint disponible.
    - `"best"`: carga el mejor checkpoint (según `val_loss`).

    Nota: para `resume_mode` se recomienda definir `model_name` estable en
    `model_kwargs` para apuntar siempre al mismo directorio de checkpoints.
    """

    if resume_mode is not None and model_cls is LinearRegressionModel:
        raise ValueError("`resume_mode` no aplica a LinearRegressionModel.")

    configure_warnings(quiet=True)
    if model_cls is LinearRegressionModel:
        kwargs = deepcopy(model_kwargs)
    else:
        kwargs = deepcopy(BASE_TRAINING_KWARGS)
        kwargs.update(model_kwargs)
        kwargs = _filter_model_init_kwargs(model_cls, kwargs)

        if not series_val:
            kwargs.pop("lr_scheduler_cls", None)
            kwargs.pop("lr_scheduler_kwargs", None)

    if resume_mode is not None:
        if not kwargs.get("save_checkpoints", False):
            raise ValueError(
                "`resume_mode` requiere `save_checkpoints=True` en la configuración del modelo."
            )
        if not kwargs.get("model_name"):
            raise ValueError(
                "`resume_mode` requiere `model_name` fijo en `model_kwargs` para localizar checkpoints."
            )
        kwargs["force_reset"] = False

    # output_chunk_length no aplica a algunos modelos; se pone condicionalmente.
    if (
        "output_chunk_length" not in kwargs
        and model_cls is not LinearRegressionModel
        and model_cls is not RNNModel
    ):
        kwargs["output_chunk_length"] = size_k

    model = model_cls(**kwargs)

    if resume_mode is not None:
        if not hasattr(model, "load_weights_from_checkpoint"):
            raise ValueError(
                f"`resume_mode` no está soportado para el modelo '{model_cls.__name__}'."
            )

        load_kwargs: dict[str, Any] = {
            "best": resume_mode == "best",
            "model_name": kwargs["model_name"],
        }
        if kwargs.get("work_dir") is not None:
            load_kwargs["work_dir"] = kwargs["work_dir"]

        try:
            model.load_weights_from_checkpoint(**load_kwargs)
            print(
                f"Reanudando {model_cls.__name__} desde checkpoint "
                f"({resume_mode}) con model_name='{kwargs['model_name']}'"
            )
        except Exception as exc:
            raise FileNotFoundError(
                "No se pudo cargar checkpoint para reanudar. Verifica `model_name`, "
                "`work_dir` y que existan checkpoints previos."
            ) from exc

    fit_kwargs = {
        "series": series_train,
        "verbose": True,
        "stride": 2,
        "max_samples_per_ts": 256,
    }
    if series_val and model_cls is not LinearRegressionModel:
        fit_kwargs["val_series"] = series_val
        fit_kwargs["dataloader_kwargs"] = {"num_workers": 2}
        fit_kwargs["load_best"] = True

    model.fit(**fit_kwargs)
    return model


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
    - Modelos no iterativos (p. ej. LinearRegression): se omiten.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str]]
        (modelos_finetuned, modelos_omitidos_con_motivo)
    """
    configure_warnings(quiet=True)

    skipped_models: dict[str, str] = {}

    for name, model in trained_models.items():
        if isinstance(model, LinearRegressionModel):
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

        if series_val:
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
    val_context_len: int = 72,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Wrapper de conveniencia para fine-tuning cuando no se dispone de
    `series_train`/`series_val` explícitos.

    Reconstruye las series con `build_train_val_test_series()` usando el mismo
    flujo que `train_global_methods()` y luego aplica
    `finetune_trained_models()`.

    `val_size` usa la misma semantica de cola fija por bloque que
    `build_train_val_test_series()`.

    `val_context_len` usa la misma semantica de ventana de contexto para val
    que `build_train_val_test_series()`.
    """
    dataset_bundle = build_train_val_test_series(
        series_dfs,
        longest_segment,
        val_size=val_size,
        min_train_len=min_train_len,
        val_context_len=val_context_len,
    )

    return finetune_trained_models(
        trained_models=trained_models,
        series_train=dataset_bundle.series_train,
        series_val=dataset_bundle.series_val,
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
    csv_output_path: str | None = "reports/metrics/training_curves_and_times.csv",
    resume_mode: str | None = None,
    model_output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Entrena métodos globales y devuelve modelos entrenados por nombre.

    Si `csv_output_path` no es `None`, exporta además un CSV con:
    - train loss por época
    - val loss por época
    - tiempo total de entrenamiento por modelo

    `resume_mode` permite reanudar manualmente checkpoints sin activar
    auto-resume por defecto:
    - `None`: desde cero.
    - `"last"`: carga último checkpoint.
    - `"best"`: carga mejor checkpoint.

    Guarda cada modelo entrenado en disco usando `model.save(...)`. Por defecto,
    la salida se escribe en `models/`.
    """
    if resume_mode not in (None, "best", "last"):
        raise ValueError("`resume_mode` debe ser None, 'best' o 'last'.")

    trained_models: dict[str, Any] = {}
    curve_rows: list[dict[str, Any]] = []
    model_configs = build_model_configs()

    if model_output_dir is None:
        models_dir = Path(__file__).resolve().parents[3] / "models"
    else:
        models_dir = Path(model_output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    series_train = dataset_bundle.series_train
    series_val = dataset_bundle.series_val

    for name in method_names:
        if name not in model_configs:
            raise ValueError(
                f"Método no soportado: '{name}'. Disponibles: {sorted(model_configs)}"
            )

        print(f"Entrenando {name}")
        model_cls, model_kwargs = model_configs[name]
        model_kwargs_with_name = deepcopy(model_kwargs)
        if model_cls is not LinearRegressionModel and not model_kwargs_with_name.get(
            "model_name"
        ):
            model_kwargs_with_name["model_name"] = f"{name}_k{size_k}"
        tracked_kwargs, loss_callback = _attach_loss_history_callback(
            model_cls,
            model_kwargs_with_name,
        )
        start_time = time.perf_counter()
        model = fit_darts_model(
            model_cls,
            series_train,
            series_val,
            size_k,
            tracked_kwargs,
            resume_mode=resume_mode,
        )
        elapsed = time.perf_counter() - start_time
        curve_rows.extend(_build_curve_rows(name, elapsed, loss_callback))

        model_save_path = models_dir / f"{name}_k{size_k}.pt"
        model.save(str(model_save_path))
        print(f"Modelo guardado en: {model_save_path.resolve()}")

        trained_models[name] = model

    if csv_output_path is not None:
        curve_df = pd.DataFrame(
            curve_rows,
            columns=[
                "model_name",
                "epoch",
                "train_loss",
                "val_loss",
                "training_time_seconds",
            ],
        ).sort_values(["model_name", "epoch"], ignore_index=True)

        output_path = Path(csv_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        curve_df = _merge_curve_rows_with_existing_csv(curve_df, output_path)
        curve_df.to_csv(output_path, index=False)
        print(f"CSV exportado en: {output_path.resolve()}")

    return trained_models
