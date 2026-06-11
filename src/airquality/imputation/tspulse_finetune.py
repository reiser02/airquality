"""CLI helpers for fine-tuning TSPulse on the project's air-quality series."""

from __future__ import annotations

import argparse
import re
import tempfile
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from airquality.data.io import resolve_device, to_pd_series
from airquality.data.loaders import load_dataset_paths, load_to_df
from airquality.data.segments import get_longest_segment
from airquality.config import cfg_get_float, cfg_get_int, cfg_get_str

try:
    from transformers import Trainer, TrainingArguments, set_seed
    try:
        from transformers import EarlyStoppingCallback, TrainerCallback
    except Exception:  # pragma: no cover - optional callback
        try:
            from transformers import EarlyStoppingCallback
        except Exception:
            EarlyStoppingCallback = None  # type: ignore[assignment]
        try:
            from transformers import TrainerCallback
        except Exception:
            TrainerCallback = None  # type: ignore[assignment]

    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    EarlyStoppingCallback = None  # type: ignore[assignment]
    TrainerCallback = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]
    set_seed = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = exc

try:
    from tsfm_public import ForecastDFDataset, TimeSeriesPreprocessor
    from tsfm_public.models.tspulse import TSPulseForReconstruction
    from tsfm_public.toolkit.lr_finder import optimal_lr_finder

    TSFM_AVAILABLE = True
    TSFM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    ForecastDFDataset = None  # type: ignore[assignment]
    TimeSeriesPreprocessor = None  # type: ignore[assignment]
    TSPulseForReconstruction = None  # type: ignore[assignment]
    optimal_lr_finder = None
    TSFM_AVAILABLE = False
    TSFM_IMPORT_ERROR = exc

warnings.filterwarnings("ignore")


if TrainerCallback is not None:

    class EpochBoundaryCallback(TrainerCallback):
        """Simple epoch boundary logger to complement global-step tqdm progress."""

        def on_epoch_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            current_epoch = 1 if state.epoch is None else int(state.epoch) + 1
            total_epochs = int(getattr(args, "num_train_epochs", 0) or 0)
            print(f"[epoch] {current_epoch}/{total_epochs} started")
            return control

        def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if state.epoch is None:
                return control
            current_epoch = int(round(state.epoch))
            total_epochs = int(getattr(args, "num_train_epochs", 0) or 0)
            print(f"[epoch] {current_epoch}/{total_epochs} finished")
            return control

else:
    EpochBoundaryCallback = None  # type: ignore[assignment]


def sanitize_name(text: str) -> str:
    """Normalize free-form text into a filesystem-friendly identifier."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\w\-\. ]+", "_", text)
    return text.replace(" ", "_")


def build_series_name(csv_path: Path, value_col: str) -> str:
    """Build a stable series identifier from file location and value column."""
    parent = sanitize_name(csv_path.parent.name)
    stem = sanitize_name(csv_path.stem)
    val = sanitize_name(value_col)
    if val.lower() in stem.lower():
        return f"{parent}__{stem}"
    return f"{parent}__{stem}__{val}"


def discover_csv_files() -> list[Path]:
    """Discover the input files that match the configured dataset filters."""
    by_keyword = sorted(Path(p).resolve() for p in load_dataset_paths())
    files = [p for p in by_keyword if p.is_file()]

    if not files:
        raise FileNotFoundError(
            "No se encontraron CSV. Revisa la configuracion de load_dataset_paths."
        )
    return files


def load_series_list(
    csv_files: Sequence[Path],
    *,
    target_column_index: int,
    freq: str,
    min_non_nan_ratio: float,
    min_points: int,
) -> list[pd.DataFrame]:
    """Load, validate, and filter the raw series used for TSPulse fine-tuning."""
    out: list[pd.DataFrame] = []
    seen_names: set[str] = set()

    for csv_path in csv_files:
        df = load_to_df(str(csv_path), name_from_path=False)
        if df is None or df.empty:
            continue

        if target_column_index < 0 or target_column_index >= len(df.columns):
            raise ValueError(
                f"target_column_index={target_column_index} fuera de rango en '{csv_path}'. "
                f"Columnas disponibles ({len(df.columns)}): {list(df.columns)}"
            )
        value_col = str(df.columns[int(target_column_index)])
        values = to_pd_series(df[[value_col]], freq=freq, name=value_col)

        name = build_series_name(csv_path, value_col)
        if name in seen_names:
            suffix = 1
            while f"{name}_{suffix}" in seen_names:
                suffix += 1
            name = f"{name}_{suffix}"
        seen_names.add(name)

        values.name = name

        if len(values) < int(min_points):
            continue

        obs_ratio = float(values.notna().mean())
        if obs_ratio < float(min_non_nan_ratio):
            continue

        out.append(values.to_frame(name=name).astype(float))

    if not out:
        raise RuntimeError(
            "No valid series found after filtering by min_points and min_non_nan_ratio."
        )

    return out


def build_train_long_df_from_series(
    series_dfs: list[pd.DataFrame],
    *,
    longest_segment: pd.DataFrame,
    timestamp_column: str,
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, int]:
    """Create long-format train dataframe from list-of-series.

    Any timestamp belonging to `longest_segment` for its selected columns is masked as NaN
    in train to avoid leakage from the held-out test block.
    """
    rows: list[pd.DataFrame] = []
    heldout_points = 0

    heldout_index = longest_segment.index
    heldout_cols = set(longest_segment.columns)

    for series_df in series_dfs:
        col = str(series_df.columns[0])
        s = series_df.iloc[:, 0].astype(float).copy()

        if col in heldout_cols and len(heldout_index) > 0:
            overlap = s.index.intersection(heldout_index)
            heldout_points += int(len(overlap))
            s.loc[overlap] = np.nan

        tmp = pd.DataFrame(
            {
                id_column: col,
                timestamp_column: s.index,
                target_column: s.values,
            }
        )
        rows.append(tmp)

    train_long = pd.concat(rows, ignore_index=True)
    train_long = train_long.sort_values([id_column, timestamp_column]).reset_index(drop=True)
    return train_long, heldout_points


def split_long_train_valid(
    train_df: pd.DataFrame,
    *,
    id_column: str,
    timestamp_column: str,
    valid_fraction: float,
    context_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split long dataframe into train/valid per-series.

    Valid starts `context_length` points before the split boundary so each
    validation window has left context, mirroring TSFM split behavior.
    """
    if not (0.0 < valid_fraction < 1.0):
        raise ValueError("valid_fraction must be in (0, 1)")

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []

    for _, grp in train_df.groupby(id_column, sort=False):
        g = grp.sort_values(timestamp_column).reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue

        split_idx = int(n * (1.0 - valid_fraction))
        split_idx = max(1, min(split_idx, n - 1))

        g_train = g.iloc[:split_idx, :].copy()
        valid_start = max(0, split_idx - int(context_length))
        g_valid = g.iloc[valid_start:, :].copy()

        train_parts.append(g_train)
        valid_parts.append(g_valid)

    if not train_parts or not valid_parts:
        raise RuntimeError(
            "No se pudo construir split train/valid; revisa valid_fraction y longitud mínima de series."
        )

    out_train = pd.concat(train_parts, ignore_index=True)
    out_valid = pd.concat(valid_parts, ignore_index=True)
    return out_train, out_valid


def build_train_valid_datasets(
    *,
    tsp: TimeSeriesPreprocessor,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> tuple[ForecastDFDataset, ForecastDFDataset]:
    """Build TSFM train and valid datasets directly from long dataframes."""
    tsp.train(train_df)
    train_df_prep = tsp.preprocess(train_df)
    valid_df_prep = tsp.preprocess(valid_df)

    common_kwargs = {
        "id_columns": tsp.id_columns,
        "timestamp_column": tsp.timestamp_column,
        "target_columns": tsp.target_columns,
        "observable_columns": tsp.observable_columns,
        "control_columns": tsp.control_columns,
        "conditional_columns": tsp.conditional_columns,
        "categorical_columns": tsp.categorical_columns,
        "static_categorical_columns": tsp.static_categorical_columns,
        "context_length": tsp.context_length,
        "prediction_length": tsp.prediction_length,
        "stride": 1,
        "enable_padding": True,
    }

    train_dataset = ForecastDFDataset(train_df_prep, **common_kwargs)
    valid_dataset = ForecastDFDataset(valid_df_prep, **common_kwargs)

    if len(train_dataset) == 0:
        raise RuntimeError(
            "El dataset de entrenamiento quedó vacío. Revisa context_length y longitud útil de las series."
        )
    if len(valid_dataset) == 0:
        raise RuntimeError(
            "El dataset de validación quedó vacío. Reduce valid_fraction o context_length."
        )

    return train_dataset, valid_dataset


def build_training_args(
    *,
    output_dir: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    report_to: str,
) -> object:
    common = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "do_eval": True,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "dataloader_num_workers": num_workers,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": seed,
        "remove_unused_columns": True,
    }

    if report_to.lower() == "none":
        common["report_to"] = []
    else:
        common["report_to"] = [report_to]

    try:
        return TrainingArguments(**common)
    except TypeError:
        common["evaluation_strategy"] = common.pop("eval_strategy")
        return TrainingArguments(**common)


def _ensure_runtime_dependencies() -> None:
    """Fail fast when transformers or Granite TSFM dependencies are missing."""
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers is not available. Install dependencies first, e.g. "
            "`pip install transformers`.\n"
            f"Original import error: {TRANSFORMERS_IMPORT_ERROR}"
        )
    if not TSFM_AVAILABLE:
        raise RuntimeError(
            "tsfm_public is not available. Install Granite TSFM first, e.g. "
            '`pip install "granite-tsfm[notebooks]"`.\n'
            f"Original import error: {TSFM_IMPORT_ERROR}"
        )


def _validate_run_args(args: argparse.Namespace) -> None:
    """Validate the fine-tuning arguments that are not enforced by argparse."""
    if not (0.0 < float(args.mask_ratio) <= 1.0):
        raise ValueError("mask_ratio must be in (0, 1].")
    if not (0.0 < float(args.plateau_factor) < 1.0):
        raise ValueError("plateau_factor must be in (0, 1).")
    if int(args.plateau_patience) < 0:
        raise ValueError("plateau_patience must be >= 0.")
    if float(args.plateau_min_lr) < 0.0:
        raise ValueError("plateau_min_lr must be >= 0.")
    if int(args.early_stopping_patience) < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if float(args.early_stopping_threshold) < 0.0:
        raise ValueError("early_stopping_threshold must be >= 0.")


def _resolve_run_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve and create the input and output paths for one run."""
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return data_root, output_dir


def _load_training_series_and_split(
    args: argparse.Namespace,
    data_root: Path,
) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame, int, pd.DataFrame, pd.DataFrame]:
    """Load raw series, hold out the evaluation block, and build train/valid tables."""
    del data_root
    csv_files = discover_csv_files(
    )

    series_dfs = load_series_list(
        csv_files,
        target_column_index=args.target_column_index,
        freq=args.freq,
        min_non_nan_ratio=args.min_non_nan_ratio,
        min_points=max(args.context_length + 16, args.min_series_points),
    )

    longest_segment = get_longest_segment(
        series_dfs,
        verbose=args.verbose_segment,
    )
    if longest_segment.empty:
        raise RuntimeError("get_longest_segment devolvio un bloque vacio.")

    train_remainder_df, heldout_points = build_train_long_df_from_series(
        series_dfs,
        longest_segment=longest_segment,
        timestamp_column=args.timestamp_column,
        id_column=args.id_column,
        target_column=args.tspulse_target_column,
    )

    print(f"[info] Files loaded: {len(csv_files)}")
    print(f"[info] Series used: {len(series_dfs)}")
    print(f"[info] Held-out test block shape (longest_segment): {longest_segment.shape}")
    print(f"[info] Points masked in train due to held-out test block: {heldout_points}")
    print(f"[info] Remaining long rows before train/valid split: {len(train_remainder_df)}")

    train_df, valid_df = split_long_train_valid(
        train_remainder_df,
        id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        valid_fraction=args.valid_fraction,
        context_length=args.context_length,
    )
    print(f"[info] Train rows: {len(train_df)} | Valid rows: {len(valid_df)}")
    return series_dfs, longest_segment, train_remainder_df, heldout_points, train_df, valid_df


def _build_preprocessor(args: argparse.Namespace) -> TimeSeriesPreprocessor:
    """Build the TSFM preprocessor used for TSPulse training datasets."""
    return TimeSeriesPreprocessor(
        id_columns=[args.id_column],
        timestamp_column=args.timestamp_column,
        target_columns=[args.tspulse_target_column],
        control_columns=[],
        context_length=args.context_length,
        prediction_length=0,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )


def _load_and_configure_model(
    args: argparse.Namespace,
    *,
    tsp: TimeSeriesPreprocessor,
    device: torch.device,
) -> object:
    """Load the pretrained TSPulse checkpoint and apply runtime overrides."""
    model = TSPulseForReconstruction.from_pretrained(
        args.model_id,
        revision=args.revision,
    ).to(device)
    model = model.float()

    ckpt_num_channels = int(
        getattr(model.config, "num_input_channels", tsp.num_input_channels)
    )
    if ckpt_num_channels != int(tsp.num_input_channels):
        raise ValueError(
            "num_input_channels del checkpoint no coincide con el dataset preparado. "
            f"checkpoint={ckpt_num_channels}, dataset={tsp.num_input_channels}. "
            "Para mantener máxima inicialización, el script espera coincidencia exacta."
        )

    runtime_config_updates: dict[str, object] = {
        "prediction_length": 0,
        "mask_type": args.mask_type,
        "mask_ratio": args.mask_ratio,
    }
    if args.dropout is not None:
        runtime_config_updates["dropout"] = float(args.dropout)
    if args.head_dropout is not None:
        runtime_config_updates["head_dropout"] = float(args.head_dropout)

    for key, value in runtime_config_updates.items():
        if hasattr(model.config, key):
            setattr(model.config, key, value)

    print(
        "[info] Model config after load "
        f"(decoder_mode={getattr(model.config, 'decoder_mode', None)}, "
        f"enable_fft_prob_loss={getattr(model.config, 'enable_fft_prob_loss', None)}, "
        f"mask_type={getattr(model.config, 'mask_type', None)}, "
        f"mask_ratio={getattr(model.config, 'mask_ratio', None)})"
    )

    for p in model.parameters():
        p.requires_grad = True
    return model


def _resolve_learning_rate(
    args: argparse.Namespace,
    *,
    model: object,
    train_dataset: ForecastDFDataset,
    device: torch.device,
) -> tuple[float, object]:
    """Return the configured learning rate or a value suggested by the LR finder."""
    lr = args.learning_rate
    if not args.auto_lr:
        return lr, model

    if optimal_lr_finder is None:
        print("[warn] optimal_lr_finder not available, using fixed learning_rate")
        return lr, model

    try:
        lr, model = optimal_lr_finder(
            model,
            train_dataset,
            batch_size=args.batch_size,
            device=str(device),
        )
        print(f"[info] Suggested LR from finder: {lr:.8f}")
    except Exception as exc:
        print(f"[warn] LR finder failed ({exc}); using fixed learning_rate={lr}")
    return lr, model


def _build_optimizer_scheduler(
    args: argparse.Namespace,
    *,
    model: object,
    learning_rate: float,
) -> tuple[AdamW, ReduceLROnPlateau]:
    """Build the optimizer and plateau scheduler used during fine-tuning."""
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=args.plateau_mode,
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        threshold=args.plateau_threshold,
        threshold_mode=args.plateau_threshold_mode,
        cooldown=args.plateau_cooldown,
        min_lr=args.plateau_min_lr,
        eps=args.plateau_eps,
    )
    print(
        "[info] LR scheduler: "
        "ReduceLROnPlateau("
        f"mode={args.plateau_mode}, factor={args.plateau_factor}, "
        f"patience={args.plateau_patience}, threshold={args.plateau_threshold}, "
        f"threshold_mode={args.plateau_threshold_mode}, cooldown={args.plateau_cooldown}, "
        f"min_lr={args.plateau_min_lr})"
    )
    return optimizer, scheduler


def _build_trainer_callbacks(args: argparse.Namespace) -> list[object]:
    """Build the optional trainer callbacks enabled for this fine-tuning run."""
    callbacks: list[object] = []
    if args.early_stopping_patience > 0 and EarlyStoppingCallback is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    if EpochBoundaryCallback is not None:
        callbacks.append(EpochBoundaryCallback())
    return callbacks


def _report_trainer_runtime(args: argparse.Namespace) -> None:
    """Print the effective callback and runtime behavior before training starts."""
    if args.early_stopping_patience > 0:
        if EarlyStoppingCallback is None:
            print("[warn] EarlyStoppingCallback no disponible; entrenamiento sin early stopping.")
        else:
            print(
                "[info] Early stopping enabled "
                f"(patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold})"
            )
    if EpochBoundaryCallback is None:
        print("[warn] TrainerCallback no disponible; no se mostrará separador por época.")


def _report_best_validation(trainer: object) -> None:
    """Print the best validation checkpoint recorded by the trainer."""
    best_eval_loss = None
    best_eval_epoch = None
    for row in trainer.state.log_history:
        if "eval_loss" not in row:
            continue
        loss_val = float(row["eval_loss"])
        if best_eval_loss is None or loss_val < best_eval_loss:
            best_eval_loss = loss_val
            best_eval_epoch = row.get("epoch")
    print(
        "[info] Best model from validation: "
        f"checkpoint={trainer.state.best_model_checkpoint}, "
        f"eval_loss={trainer.state.best_metric}, "
        f"epoch={best_eval_epoch}"
    )


def _save_finetuned_artifacts(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    trainer: object,
    tsp: TimeSeriesPreprocessor,
) -> None:
    """Persist the fine-tuned model and preprocessor to the output directory."""
    run_name = f"airquality_tspulse_ft_{args.mask_type}_{args.mask_ratio}"
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tsp.save_pretrained(str(save_dir / "preprocessor"))
    print(f"[done] Fine-tuned model saved to: {save_dir}")


def run(args: argparse.Namespace) -> None:
    """Execute the full TSPulse fine-tuning workflow from parsed CLI args."""
    _ensure_runtime_dependencies()

    set_seed(args.seed)
    np.random.seed(args.seed)
    _validate_run_args(args)

    data_root, output_dir = _resolve_run_paths(args)
    _, _, _, _, train_df, valid_df = _load_training_series_and_split(
        args,
        data_root,
    )

    # Real NaNs are preserved in the dataset. TSPulse will use the observed mask
    # directly only when mask_type='user'; other mask types apply synthetic masking.
    tsp = _build_preprocessor(args)

    train_dataset, valid_dataset = build_train_valid_datasets(
        tsp=tsp,
        train_df=train_df,
        valid_df=valid_df,
    )

    device = torch.device(resolve_device(args.device))
    print(f"[info] Device: {device}")

    model = _load_and_configure_model(args, tsp=tsp, device=device)
    lr, model = _resolve_learning_rate(
        args,
        model=model,
        train_dataset=train_dataset,
        device=device,
    )

    temp_dir = tempfile.mkdtemp(prefix="tspulse_ft_")
    train_args = build_training_args(
        output_dir=temp_dir,
        learning_rate=lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        report_to=args.report_to,
    )

    optimizer, scheduler = _build_optimizer_scheduler(
        args,
        model=model,
        learning_rate=lr,
    )
    callbacks = _build_trainer_callbacks(args)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks if callbacks else None,
    )
    _report_trainer_runtime(args)

    trainer.train()
    _report_best_validation(trainer)
    _save_finetuned_artifacts(args=args, output_dir=output_dir, trainer=trainer, tsp=tsp)


def _build_parser_defaults() -> dict[str, object]:
    """Read CLI default values from the project configuration files."""
    return {
        "data_root": cfg_get_str("data", "data_root", "Datos-post-COUTA"),
        "key_word": cfg_get_str("data", "key_word", "NO2"),
        "file_extension": cfg_get_str("data", "file_extension", "csv"),
        "timestamp_column": cfg_get_str("data", "timestamp_column", "fecha"),
        "target_column_index": cfg_get_int("data", "target_column_index", 0),
        "freq": cfg_get_str("data", "freq", "h"),
        "min_non_nan_ratio": cfg_get_float("data", "min_non_nan_ratio", 0.15),
        "min_series_points": cfg_get_int("data", "min_series_points", 600),
        "id_column": cfg_get_str("tspulse", "id_column", "series_id"),
        "tspulse_target_column": cfg_get_str("tspulse", "tspulse_target_column", "value"),
        "model_id": cfg_get_str("tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"),
        "revision": cfg_get_str("tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1"),
        "context_length": cfg_get_int("tspulse", "context_length", 512),
        "mask_type": cfg_get_str("tspulse", "mask_type", "var_hybrid"),
        "mask_ratio": cfg_get_float("tspulse", "mask_ratio", 0.7),
        "epochs": cfg_get_int("tspulse", "epochs", 40),
        "batch_size": cfg_get_int("tspulse", "batch_size", 16),
        "eval_batch_size": cfg_get_int("tspulse", "eval_batch_size", 16),
        "valid_fraction": cfg_get_float("tspulse", "valid_fraction", 0.2),
        "learning_rate": cfg_get_float("tspulse", "learning_rate", 1e-4),
        "weight_decay": cfg_get_float("tspulse", "weight_decay", 1e-2),
        "plateau_mode": cfg_get_str("tspulse", "plateau_mode", "min"),
        "plateau_factor": cfg_get_float("tspulse", "plateau_factor", 0.5),
        "plateau_patience": cfg_get_int("tspulse", "plateau_patience", 3),
        "plateau_threshold": cfg_get_float("tspulse", "plateau_threshold", 1e-4),
        "plateau_threshold_mode": cfg_get_str("tspulse", "plateau_threshold_mode", "rel"),
        "plateau_cooldown": cfg_get_int("tspulse", "plateau_cooldown", 0),
        "plateau_min_lr": cfg_get_float("tspulse", "plateau_min_lr", 1e-6),
        "plateau_eps": cfg_get_float("tspulse", "plateau_eps", 1e-8),
        "early_stopping_patience": cfg_get_int("tspulse", "early_stopping_patience", 5),
        "early_stopping_threshold": cfg_get_float("tspulse", "early_stopping_threshold", 0.0),
        "num_workers": cfg_get_int("tspulse", "num_workers", 1),
        "seed": cfg_get_int("tspulse", "seed", 42),
        "device": cfg_get_str("tspulse", "device", "cpu"),
        "report_to": cfg_get_str("tspulse", "report_to", "none"),
        "output_dir": cfg_get_str("tspulse", "output_dir", "models/tspulse_finetune"),
    }


class ConfigOnlyArgumentParser(argparse.ArgumentParser):
    """Argument parser that re-applies cfg-backed values after CLI parsing."""

    def __init__(self, config_defaults: dict[str, object], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._config_defaults = dict(config_defaults)

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args=args, namespace=namespace)
        for key, value in self._config_defaults.items():
            setattr(parsed, key, value)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the TSPulse fine-tuning entry point."""
    defaults = _build_parser_defaults()
    parser = ConfigOnlyArgumentParser(
        config_defaults=defaults,
        description=(
            "Fine-tune TSPulse for imputation using list-of-series loading, "
            "with held-out test block from get_longest_segment."
        )
    )

    parser.add_argument("--verbose-segment", action="store_true")
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If set, overrides checkpoint dropout after loading.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=None,
        help="If set, overrides checkpoint head_dropout after loading.",
    )
    parser.add_argument("--auto-lr", action="store_true")

    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    run(cli_args)
