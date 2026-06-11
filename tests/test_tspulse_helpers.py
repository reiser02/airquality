"""Tests TSPulse fine-tuning helpers, argument validation, and run orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from airquality.imputation.tspulse_finetune import (
    _validate_run_args,
    build_parser,
    build_series_name,
    run,
    sanitize_name,
    split_long_train_valid,
)


def test_sanitize_name_normalizes_spaces_and_symbols() -> None:
    assert sanitize_name("  Calle Real #1  ") == "Calle_Real__1"


def test_build_series_name_avoids_duplicate_value_col_when_in_stem() -> None:
    p = Path("/tmp/estacion/Aquatec_NO2.csv")
    assert build_series_name(p, "NO2") == "estacion__Aquatec_NO2"


def test_split_long_train_valid_generates_contextual_validation() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "id": ["S"] * n,
            "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
            "y": list(range(n)),
        }
    )

    train, valid = split_long_train_valid(
        df,
        id_column="id",
        timestamp_column="ts",
        valid_fraction=0.2,
        context_length=2,
    )

    assert len(train) == 8
    assert len(valid) == 4


def test_split_long_train_valid_validates_fraction() -> None:
    df = pd.DataFrame({"id": ["S", "S"], "ts": pd.date_range("2024-01-01", periods=2, freq="h")})
    with pytest.raises(ValueError):
        split_long_train_valid(
            df,
            id_column="id",
            timestamp_column="ts",
            valid_fraction=1.0,
            context_length=2,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mask_ratio", 0.0, "mask_ratio"),
        ("plateau_factor", 1.0, "plateau_factor"),
        ("plateau_patience", -1, "plateau_patience"),
        ("plateau_min_lr", -1.0, "plateau_min_lr"),
        ("early_stopping_patience", -1, "early_stopping_patience"),
        ("early_stopping_threshold", -1.0, "early_stopping_threshold"),
    ],
)
def test_validate_run_args_rejects_invalid_values(field: str, value: object, message: str) -> None:
    args = argparse.Namespace(
        mask_ratio=0.7,
        plateau_factor=0.5,
        plateau_patience=3,
        plateau_min_lr=1e-6,
        early_stopping_patience=5,
        early_stopping_threshold=0.0,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        _validate_run_args(args)


def test_build_parser_resolves_runtime_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_cfg_get_int(section: str, option: str, default: int) -> int:
        overrides = {
            ("data", "target_column_index"): 2,
            ("data", "min_series_points"): 123,
            ("tspulse", "context_length"): 64,
            ("tspulse", "epochs"): 7,
            ("tspulse", "seed"): 99,
        }
        return overrides.get((section, option), default)

    def fake_cfg_get_float(section: str, option: str, default: float) -> float:
        overrides = {
            ("data", "min_non_nan_ratio"): 0.25,
            ("tspulse", "learning_rate"): 2e-4,
            ("tspulse", "mask_ratio"): 0.4,
        }
        return overrides.get((section, option), default)

    def fake_cfg_get_str(section: str, option: str, default: str) -> str:
        overrides = {
            ("data", "data_root"): "runtime-data",
            ("data", "key_word"): "O3",
            ("data", "freq"): "30min",
            ("tspulse", "model_id"): "runtime-model",
            ("tspulse", "device"): "cuda",
            ("tspulse", "output_dir"): "runtime-out",
        }
        return overrides.get((section, option), default)

    monkeypatch.setattr("airquality.imputation.tspulse_finetune.cfg_get_int", fake_cfg_get_int)
    monkeypatch.setattr("airquality.imputation.tspulse_finetune.cfg_get_float", fake_cfg_get_float)
    monkeypatch.setattr("airquality.imputation.tspulse_finetune.cfg_get_str", fake_cfg_get_str)

    args = build_parser().parse_args([])

    assert args.data_root == "runtime-data"
    assert args.key_word == "O3"
    assert args.freq == "30min"
    assert args.target_column_index == 2
    assert args.min_non_nan_ratio == 0.25
    assert args.min_series_points == 123
    assert args.model_id == "runtime-model"
    assert args.context_length == 64
    assert args.mask_ratio == 0.4
    assert args.epochs == 7
    assert args.learning_rate == 2e-4
    assert args.seed == 99
    assert args.device == "cuda"
    assert args.output_dir == "runtime-out"


def test_run_smoke_executes_refactored_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        seed=42,
        data_root=str(tmp_path / "data"),
        output_dir=str(output_dir),
        key_word="NO2",
        file_extension="csv",
        target_column_index=0,
        freq="h",
        min_non_nan_ratio=0.1,
        min_series_points=32,
        context_length=16,
        verbose_segment=False,
        timestamp_column="ts",
        id_column="series_id",
        tspulse_target_column="value",
        valid_fraction=0.2,
        device="cpu",
        model_id="model-id",
        revision="rev-1",
        mask_type="var_hybrid",
        mask_ratio=0.7,
        dropout=None,
        head_dropout=None,
        learning_rate=1e-4,
        auto_lr=False,
        batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        epochs=2,
        report_to="none",
        weight_decay=1e-2,
        plateau_mode="min",
        plateau_factor=0.5,
        plateau_patience=3,
        plateau_threshold=1e-4,
        plateau_threshold_mode="rel",
        plateau_cooldown=0,
        plateau_min_lr=1e-6,
        plateau_eps=1e-8,
        early_stopping_patience=1,
        early_stopping_threshold=0.0,
    )

    calls: dict[str, object] = {}

    class DummyPreprocessor:
        num_input_channels = 1

        def save_pretrained(self, path: str) -> None:
            calls["preprocessor_path"] = path

    class DummyModel:
        def parameters(self):
            return []

    class DummyTrainer:
        def __init__(self, **kwargs: object) -> None:
            calls["trainer_kwargs"] = kwargs
            self.state = argparse.Namespace(
                log_history=[{"eval_loss": 0.8, "epoch": 1}],
                best_model_checkpoint="checkpoint-1",
                best_metric=0.8,
            )

        def train(self) -> None:
            calls["trained"] = True

        def save_model(self, path: str) -> None:
            calls["saved_model_path"] = path

    monkeypatch.setattr("airquality.imputation.tspulse_finetune.TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr("airquality.imputation.tspulse_finetune.TSFM_AVAILABLE", True)
    monkeypatch.setattr("airquality.imputation.tspulse_finetune.set_seed", lambda seed: calls.setdefault("seed", seed))
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._load_training_series_and_split",
        lambda args, data_root: ([], pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._build_preprocessor",
        lambda args: DummyPreprocessor(),
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune.build_train_valid_datasets",
        lambda tsp, train_df, valid_df: ("train-dataset", "valid-dataset"),
    )
    monkeypatch.setattr("airquality.imputation.tspulse_finetune.resolve_device", lambda preferred: "cpu")
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._load_and_configure_model",
        lambda args, tsp, device: DummyModel(),
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._resolve_learning_rate",
        lambda args, model, train_dataset, device: (args.learning_rate, model),
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune.build_training_args",
        lambda **kwargs: "training-args",
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._build_optimizer_scheduler",
        lambda args, model, learning_rate: ("optimizer", "scheduler"),
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._build_trainer_callbacks",
        lambda args: ["callback"],
    )
    monkeypatch.setattr("airquality.imputation.tspulse_finetune.Trainer", DummyTrainer)
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._report_best_validation",
        lambda trainer: calls.setdefault("reported_best", True),
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune._save_finetuned_artifacts",
        lambda args, output_dir, trainer, tsp: calls.setdefault("saved", str(output_dir)),
    )

    run(args)

    assert calls["seed"] == 42
    assert calls["trained"] is True
    assert calls["reported_best"] is True
    assert calls["saved"] == str(output_dir.resolve())
    assert calls["trainer_kwargs"]["train_dataset"] == "train-dataset"
    assert calls["trainer_kwargs"]["eval_dataset"] == "valid-dataset"
