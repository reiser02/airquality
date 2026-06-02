from __future__ import annotations

from pathlib import Path

import pandas as pd

from airquality.modeling.training import (
    _build_curve_rows,
    _filter_model_init_kwargs,
    _merge_curve_rows_with_existing_csv,
    _metric_to_float,
    _read_metric_from_callback_metrics,
)


class DummyTensorLike:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> "DummyTensorLike":
        return self

    def cpu(self) -> "DummyTensorLike":
        return self

    def item(self) -> float:
        return self.value


def test_metric_to_float_converts_tensor_like() -> None:
    assert _metric_to_float(DummyTensorLike(1.25)) == 1.25
    assert _metric_to_float(None) is None


def test_read_metric_from_callback_metrics_finds_first_valid() -> None:
    metrics = {"x": None, "train_loss": 1.0, "train_loss_epoch": 2.0}
    assert _read_metric_from_callback_metrics(metrics, ("x", "train_loss")) == 1.0


def test_filter_model_init_kwargs_respects_signature() -> None:
    class Model:
        def __init__(self, a: int, *, b: int = 0) -> None:
            self.a = a
            self.b = b

    out = _filter_model_init_kwargs(Model, {"a": 1, "b": 2, "c": 3})
    assert out == {"a": 1, "b": 2}


def test_build_curve_rows_without_callback_returns_single_nan_row() -> None:
    rows = _build_curve_rows("M", 0.5, None)
    assert len(rows) == 1
    assert rows[0]["model_name"] == "M"
    assert rows[0]["epoch"] == 0


def test_merge_curve_rows_replaces_existing_model_rows(tmp_path: Path) -> None:
    output = tmp_path / "curves.csv"
    existing = pd.DataFrame(
        [
            {"model_name": "A", "epoch": 0, "train_loss": 1.0, "val_loss": 2.0, "training_time_seconds": 10.0},
            {"model_name": "B", "epoch": 0, "train_loss": 3.0, "val_loss": 4.0, "training_time_seconds": 20.0},
        ]
    )
    existing.to_csv(output, index=False)

    current = pd.DataFrame(
        [
            {"model_name": "A", "epoch": 1, "train_loss": 0.9, "val_loss": 1.8, "training_time_seconds": 9.0},
        ]
    )

    merged = _merge_curve_rows_with_existing_csv(current, output)
    assert set(merged["model_name"]) == {"A", "B"}
    assert len(merged[merged["model_name"] == "A"]) == 1
