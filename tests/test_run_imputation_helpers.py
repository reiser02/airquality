from __future__ import annotations

from pathlib import Path

import pandas as pd

from airquality.imputation.run_benchmark import (
    Float32InputModelAdapter,
    _resolve_repo_root,
    summarize_results_by_model,
)


class DummyModel:
    def __init__(self) -> None:
        self.last_series = None

    def predict(self, *args, **kwargs):
        self.last_series = kwargs.get("series")
        return "ok"


def test_float32_input_model_adapter_casts_series_argument() -> None:
    model = DummyModel()
    adapter = Float32InputModelAdapter(model)
    s = pd.Series([1.0, 2.0], dtype="float64")

    result = adapter.predict(series=s)

    assert result == "ok"
    assert str(model.last_series.dtype) == "float32"


def test_resolve_repo_root_uses_explicit_path(tmp_path: Path) -> None:
    out = _resolve_repo_root(tmp_path)
    assert out == tmp_path.resolve()


def test_summarize_results_by_model_groups_and_sorts() -> None:
    df = pd.DataFrame(
        {
            "Modelo": ["A", "A", "B"],
            "MAE": [1.0, 2.0, 0.5],
            "RMSE": [1.2, 2.2, 0.4],
            "MASE": [1.1, 2.1, 0.3],
        }
    )

    out = summarize_results_by_model(df)

    assert list(out["Modelo"]) == ["B", "A"]
