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
from airquality.config import cfg_get_float, cfg_get_int, cfg_get_str

from airquality.data.utils import get_longest_segment, load_dataset_paths, load_to_df

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
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\w\-\. ]+", "_", text)
    return text.replace(" ", "_")


def build_series_name(csv_path: Path, value_col: str) -> str:
    parent = sanitize_name(csv_path.parent.name)
    stem = sanitize_name(csv_path.stem)
    val = sanitize_name(value_col)
    if val.lower() in stem.lower():
        return f"{parent}__{stem}"
    return f"{parent}__{stem}__{val}"


def discover_csv_files(
    data_root: Path,
    *,
    key_word: str,
    file_extension: str,
) -> list[Path]:
    # Directly use project utility, aligned with:
    # load_dataset_paths("../Datos-post-COUTA/*/", key_word="NO2", file_extension="csv")
    base_path = str((data_root / "*/").resolve())
    by_keyword = sorted(
        Path(p).resolve()
        for p in load_dataset_paths(
            base_path=base_path,
            key_word=key_word,
            file_extension=file_extension,
        )
    )
    files = [p for p in by_keyword if p.is_file()]

    if not files:
        raise FileNotFoundError(
            "No se encontraron CSV. Revisa ruta y filtros de load_dataset_paths.\n"
            f"base_path='{base_path}', key_word='{key_word}', file_extension='{file_extension}'"
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


def run(args: argparse.Namespace) -> None:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers is not available. Install dependencies first, e.g. "
            "`pip install transformers`.\n"
            f"Original import error: {TRANSFORMERS_IMPORT_ERROR}"
        )
    if not TSFM_AVAILABLE:
        raise RuntimeError(
            "tsfm_public is not available. Install Granite TSFM first, e.g. "
            "`pip install \"granite-tsfm[notebooks]\"`.\n"
            f"Original import error: {TSFM_IMPORT_ERROR}"
        )

    set_seed(args.seed)
    np.random.seed(args.seed)

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

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_csv_files(
        data_root,
        key_word=args.key_word,
        file_extension=args.file_extension,
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
        force_end=args.force_end,
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

    # Real NaNs are preserved in the dataset. TSPulse will use the observed mask
    # directly only when mask_type='user'; other mask types apply synthetic masking.
    tsp = TimeSeriesPreprocessor(
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

    train_dataset, valid_dataset = build_train_valid_datasets(
        tsp=tsp,
        train_df=train_df,
        valid_df=valid_df,
    )

    device = torch.device(resolve_device(args.device))
    print(f"[info] Device: {device}")

    model = TSPulseForReconstruction.from_pretrained(
        args.model_id,
        revision=args.revision,
    ).to(device)
    model = model.float()

    ckpt_num_channels = int(getattr(model.config, "num_input_channels", tsp.num_input_channels))
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

    lr = args.learning_rate
    if args.auto_lr:
        if optimal_lr_finder is None:
            print("[warn] optimal_lr_finder not available, using fixed learning_rate")
        else:
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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
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

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks if callbacks else None,
    )
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

    trainer.train()
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

    run_name = f"airquality_tspulse_ft_{args.mask_type}_{args.mask_ratio}"
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tsp.save_pretrained(str(save_dir / "preprocessor"))

    print(f"[done] Fine-tuned model saved to: {save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune TSPulse for imputation using list-of-series loading, "
            "with held-out test block from get_longest_segment."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=cfg_get_str("data", "data_root", "Datos-post-COUTA"),
    )
    parser.add_argument(
        "--key-word",
        type=str,
        default=cfg_get_str("data", "key_word", "NO2"),
        help="Keyword used by load_dataset_paths.",
    )
    parser.add_argument(
        "--file-extension",
        type=str,
        default=cfg_get_str("data", "file_extension", "csv"),
        help="File extension used by load_dataset_paths fallback.",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default=cfg_get_str("data", "timestamp_column", "fecha"),
    )
    parser.add_argument(
        "--target-column-index",
        type=int,
        default=cfg_get_int("data", "target_column_index", 0),
        help="Índice de la columna de valores dentro de cada CSV (0 = primera columna de valores).",
    )
    parser.add_argument("--freq", type=str, default=cfg_get_str("data", "freq", "h"))
    parser.add_argument(
        "--min-non-nan-ratio",
        type=float,
        default=cfg_get_float("data", "min_non_nan_ratio", 0.15),
    )
    parser.add_argument(
        "--min-series-points",
        type=int,
        default=cfg_get_int("data", "min_series_points", 600),
    )

    parser.add_argument("--force-end", action="store_true", help="Use trailing complete block in get_longest_segment")
    parser.add_argument("--verbose-segment", action="store_true")

    parser.add_argument(
        "--id-column",
        type=str,
        default=cfg_get_str("tspulse", "id_column", "series_id"),
    )
    parser.add_argument(
        "--tspulse-target-column",
        type=str,
        default=cfg_get_str("tspulse", "tspulse_target_column", "value"),
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=cfg_get_str(
            "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=cfg_get_str(
            "tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1"
        ),
        help="Imputation-optimized TSPulse branch",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=cfg_get_int("tspulse", "context_length", 512),
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        choices=["var_hybrid", "hybrid", "block", "random", "user"],
        default=cfg_get_str("tspulse", "mask_type", "var_hybrid"),
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=cfg_get_float("tspulse", "mask_ratio", 0.7),
    )

    parser.add_argument(
        "--epochs", type=int, default=cfg_get_int("tspulse", "epochs", 40)
    )
    parser.add_argument(
        "--batch-size", type=int, default=cfg_get_int("tspulse", "batch_size", 16)
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=cfg_get_int("tspulse", "eval_batch_size", 16),
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=cfg_get_float("tspulse", "valid_fraction", 0.2),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=cfg_get_float("tspulse", "learning_rate", 1e-4),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=cfg_get_float("tspulse", "weight_decay", 1e-2),
    )
    parser.add_argument(
        "--plateau-mode",
        type=str,
        choices=["min", "max"],
        default=cfg_get_str("tspulse", "plateau_mode", "min"),
        help="Metric direction for ReduceLROnPlateau (use 'min' for eval_loss).",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=cfg_get_float("tspulse", "plateau_factor", 0.5),
        help="new_lr = lr * plateau_factor when metric plateaus.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=cfg_get_int("tspulse", "plateau_patience", 3),
        help="Number of eval epochs without improvement before lowering LR.",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=cfg_get_float("tspulse", "plateau_threshold", 1e-4),
        help="Minimum significant improvement for plateau detection.",
    )
    parser.add_argument(
        "--plateau-threshold-mode",
        type=str,
        choices=["rel", "abs"],
        default=cfg_get_str("tspulse", "plateau_threshold_mode", "rel"),
    )
    parser.add_argument(
        "--plateau-cooldown",
        type=int,
        default=cfg_get_int("tspulse", "plateau_cooldown", 0),
        help="Cooldown eval epochs after an LR drop.",
    )
    parser.add_argument(
        "--plateau-min-lr",
        type=float,
        default=cfg_get_float("tspulse", "plateau_min_lr", 1e-6),
        help="Lower bound for learning rate.",
    )
    parser.add_argument(
        "--plateau-eps",
        type=float,
        default=cfg_get_float("tspulse", "plateau_eps", 1e-8),
        help="Minimal LR change to apply.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=cfg_get_int("tspulse", "early_stopping_patience", 5),
        help="Stop after N eval epochs without significant eval_loss improvement. Set 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=cfg_get_float("tspulse", "early_stopping_threshold", 0.0),
        help="Minimum eval_loss improvement to reset early stopping patience.",
    )
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

    parser.add_argument(
        "--num-workers",
        type=int,
        default=cfg_get_int("tspulse", "num_workers", 1),
    )
    parser.add_argument("--seed", type=int, default=cfg_get_int("tspulse", "seed", 42))
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=cfg_get_str("tspulse", "device", "cpu"),
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default=cfg_get_str("tspulse", "report_to", "none"),
        help="none | tensorboard | wandb | ...",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=cfg_get_str("tspulse", "output_dir", "src/artifacts/tspulse_finetune"),
    )

    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    run(cli_args)
