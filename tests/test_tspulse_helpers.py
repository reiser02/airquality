"""Tests TSPulse fine-tuning helpers, argument validation, and run orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airquality.imputation.tspulse_finetune import (
    _validate_run_args,
    build_parser,
    build_series_name,
    build_train_valid_datasets,
    discover_csv_files,
    load_series_list,
    run,
    sanitize_name,
    split_long_train_valid,
)


def test_sanitize_name_normalizes_spaces_and_symbols() -> None:
    assert sanitize_name("  Calle Real #1  ") == "Calle_Real__1"


def test_build_series_name_avoids_duplicate_value_col_when_in_stem() -> None:
    p = Path("/tmp/estacion/Aquatec_NO2.csv")
    assert build_series_name(p, "NO2") == "estacion__Aquatec_NO2"


def test_discover_csv_files_uses_requested_data_root(tmp_path: Path) -> None:
    expected = tmp_path / "station" / "station_NO2.csv"
    expected.parent.mkdir()
    expected.write_text("fecha,NO2\n", encoding="utf-8")
    (tmp_path / "station" / "station_CO.csv").write_text(
        "fecha,CO\n", encoding="utf-8"
    )

    assert discover_csv_files(
        tmp_path, key_word="NO2", file_extension="csv"
    ) == [expected.resolve()]


def test_load_series_list_applies_shared_preprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "station_NO2.csv"
    raw = pd.DataFrame(
        {"NO2": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])
    )
    hourly = pd.DataFrame(
        {"NO2": [2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=2, freq="h")
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune.load_to_df", lambda *_args, **_kwargs: raw
    )
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune.preprocess",
        lambda frames, pollutant: calls.update(
            {"frames": frames, "pollutant": pollutant}
        )
        or ([hourly], [0]),
    )

    out = load_series_list(
        [csv_path],
        pollutant="NO2",
        target_column_index=0,
        freq="h",
        min_non_nan_ratio=1.0,
        min_points=2,
    )

    assert calls["pollutant"] == "NO2"
    assert calls["frames"][0].equals(raw[["NO2"]])
    assert out[0].iloc[:, 0].tolist() == [2.0, 3.0]


def test_train_valid_examples_have_disjoint_observed_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 20
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

    assert len(train) == 16
    assert len(valid) == 4
    assert set(train["ts"]).isdisjoint(valid["ts"])

    class DummyPreprocessor:
        id_columns = ["id"]
        timestamp_column = "ts"
        target_columns = ["y"]
        observable_columns: list[str] = []
        control_columns: list[str] = []
        conditional_columns: list[str] = []
        categorical_columns: list[str] = []
        static_categorical_columns: list[str] = []
        context_length = 2
        prediction_length = 0

        def train(self, data: pd.DataFrame) -> None:
            pass

        def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
            return data

    class DummyDataset:
        def __init__(self, data: pd.DataFrame, **kwargs: object) -> None:
            self.data = data.reset_index(drop=True)
            self.context_length = int(kwargs["context_length"])
            self.stride = int(kwargs["stride"])
            self.enable_padding = kwargs["enable_padding"]
            self.starts = list(
                range(0, len(data) - self.context_length + 1, self.stride)
            )

        def __len__(self) -> int:
            return len(self.starts)

        def __getitem__(self, index: int) -> dict[str, np.ndarray]:
            start = self.starts[index]
            values = self.data["y"].iloc[start : start + self.context_length]
            return {"past_observed_mask": values.notna().to_numpy()}

    train.loc[train.index[2], "y"] = np.nan
    valid.loc[valid.index[0], "y"] = np.nan
    monkeypatch.setattr(
        "airquality.imputation.tspulse_finetune.ForecastDFDataset", DummyDataset
    )

    train_dataset, valid_dataset = build_train_valid_datasets(
        tsp=DummyPreprocessor(), train_df=train, valid_df=valid
    )

    assert train_dataset.dataset.stride == 2
    assert train_dataset.dataset.enable_padding is False
    assert train_dataset.indices == [0, 2, 3, 4, 5, 6, 7]
    assert valid_dataset.indices == [1]


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
            ("tspulse", "target_column_index"): 2,
            ("tspulse", "min_series_points"): 123,
            ("tspulse", "context_length"): 64,
            ("tspulse", "epochs"): 7,
            ("tspulse", "seed"): 99,
        }
        return overrides.get((section, option), default)

    def fake_cfg_get_float(section: str, option: str, default: float) -> float:
        overrides = {
            ("tspulse", "min_non_nan_ratio"): 0.25,
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
        lambda args, data_root: (
            [],
            {},
            pd.DataFrame(),
            {},
            pd.DataFrame(),
            0,
            pd.DataFrame(),
            pd.DataFrame(),
        ),
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
        lambda **kwargs: calls.setdefault("saved", str(kwargs["output_dir"])),
    )

    run(args)

    assert calls["seed"] == 42
    assert calls["trained"] is True
    assert calls["reported_best"] is True
    assert calls["saved"] == str(output_dir.resolve())
    assert calls["trainer_kwargs"]["train_dataset"] == "train-dataset"
    assert calls["trainer_kwargs"]["eval_dataset"] == "valid-dataset"
