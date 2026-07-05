"""Tests training pipeline helpers, callbacks, model fitting, and export behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from darts import TimeSeries
from darts.models import LinearRegressionModel

from airquality.modeling.training import (
    LossHistoryCallback,
    _build_curve_rows,
    _filter_model_init_kwargs,
    _merge_curve_rows_with_existing_csv,
    _metric_to_float,
    _read_metric_from_callback_metrics,
    _attach_loss_history_callback,
    build_scaled_train_val_series,
    build_benchmark_dataset_bundle,
    fit_darts_model,
    train_global_methods,
)
from airquality.modeling.training_config import TrainingDatasetBundle


class DummyTensorLike:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> "DummyTensorLike":
        return self

    def cpu(self) -> "DummyTensorLike":
        return self

    def item(self) -> float:
        return self.value


class BadItemTensorLike:
    def detach(self) -> "BadItemTensorLike":
        return self

    def cpu(self) -> "BadItemTensorLike":
        return self

    def item(self) -> float:
        raise RuntimeError("bad item")


def _frame(name: str, values: list[float | None], start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({name: values}, index=index)


def _series(name: str | None, values: list[float], start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(values, index=index, name=name)


def test_metric_to_float_converts_tensor_like() -> None:
    assert _metric_to_float(DummyTensorLike(1.25)) == 1.25
    assert _metric_to_float(None) is None


def test_metric_to_float_returns_none_when_item_access_fails() -> None:
    assert _metric_to_float(BadItemTensorLike()) is None


def test_read_metric_from_callback_metrics_finds_first_valid() -> None:
    metrics = {"x": None, "train_loss": 1.0, "train_loss_epoch": 2.0}
    assert _read_metric_from_callback_metrics(metrics, ("x", "train_loss")) == 1.0


def test_loss_history_callback_stores_train_and_validation_losses() -> None:
    callback = LossHistoryCallback()

    train_trainer = type(
        "Trainer",
        (),
        {"current_epoch": 3, "callback_metrics": {"train_loss": DummyTensorLike(0.25)}},
    )()
    val_trainer = type(
        "Trainer",
        (),
        {
            "current_epoch": 3,
            "callback_metrics": {"val_loss_epoch": DummyTensorLike(0.5)},
            "sanity_checking": False,
        },
    )()

    callback.on_train_epoch_end(train_trainer, None)
    callback.on_validation_epoch_end(val_trainer, None)

    assert callback.train_loss_by_epoch == {3: 0.25}
    assert callback.val_loss_by_epoch == {3: 0.5}


def test_loss_history_callback_skips_sanity_check_validation() -> None:
    callback = LossHistoryCallback()
    trainer = type(
        "Trainer",
        (),
        {
            "current_epoch": 1,
            "callback_metrics": {"val_loss": 1.0},
            "sanity_checking": True,
        },
    )()

    callback.on_validation_epoch_end(trainer, None)

    assert callback.val_loss_by_epoch == {}


def test_loss_history_callback_ignores_missing_or_invalid_metrics() -> None:
    callback = LossHistoryCallback()
    train_trainer = type(
        "Trainer",
        (),
        {"current_epoch": 0, "callback_metrics": {"train_loss": BadItemTensorLike()}},
    )()
    val_trainer = type(
        "Trainer",
        (),
        {"current_epoch": 0, "callback_metrics": {}, "sanity_checking": False},
    )()

    callback.on_train_epoch_end(train_trainer, None)
    callback.on_validation_epoch_end(val_trainer, None)

    assert callback.train_loss_by_epoch == {}
    assert callback.val_loss_by_epoch == {}


def test_filter_model_init_kwargs_respects_signature() -> None:
    class Model:
        def __init__(self, a: int, *, b: int = 0) -> None:
            self.a = a
            self.b = b

    out = _filter_model_init_kwargs(Model, {"a": 1, "b": 2, "c": 3})
    assert out == {"a": 1, "b": 2}


def test_attach_loss_history_callback_appends_callback_without_mutating_input() -> None:
    original_callback = "existing-callback"
    model_kwargs = {
        "pl_trainer_kwargs": {
            "callbacks": [original_callback],
            "logger": False,
        }
    }

    updated_kwargs, callback = _attach_loss_history_callback(type("TorchModel", (), {}), model_kwargs)

    assert isinstance(callback, LossHistoryCallback)
    assert updated_kwargs["pl_trainer_kwargs"]["callbacks"][0] == original_callback
    assert updated_kwargs["pl_trainer_kwargs"]["callbacks"][1] is callback
    assert model_kwargs["pl_trainer_kwargs"]["callbacks"] == [original_callback]


def test_attach_loss_history_callback_skips_linear_regression_model() -> None:
    model_kwargs = {"pl_trainer_kwargs": {"callbacks": []}}

    updated_kwargs, callback = _attach_loss_history_callback(LinearRegressionModel, model_kwargs)

    assert callback is None
    assert updated_kwargs == model_kwargs


def test_attach_loss_history_callback_skips_when_trainer_kwargs_are_missing() -> None:
    updated_kwargs, callback = _attach_loss_history_callback(type("TorchModel", (), {}), {"x": 1})

    assert callback is None
    assert updated_kwargs == {"x": 1}


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"val_size": -1, "min_train_len": 2, "val_context_len": 1}, "val_size"),
        ({"val_size": 1, "min_train_len": 2, "val_context_len": -1}, "val_context_len"),
    ],
)
def test_build_scaled_train_val_series_validates_negative_sizes(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_scaled_train_val_series([_frame("A", [1.0, 2.0, 3.0])], **kwargs)


def test_build_scaled_train_val_series_with_zero_validation_returns_only_train() -> None:
    train_series, val_series, scalers = build_scaled_train_val_series(
        [_frame("A", [1.0, 2.0, None, 4.0, 5.0, 6.0])],
        val_size=0,
        min_train_len=2,
        val_context_len=1,
    )

    assert len(train_series) == 2
    assert val_series == []
    assert set(scalers) == {"A"}


def test_build_scaled_train_val_series_uses_full_subseries_when_validation_window_matches_length() -> None:
    train_series, val_series, _ = build_scaled_train_val_series(
        [_frame("A", [1.0, 2.0, 3.0, 4.0, 5.0])],
        val_size=2,
        min_train_len=2,
        val_context_len=3,
    )

    assert len(train_series) == 1
    assert len(val_series) == 1
    assert len(train_series[0]) == 3
    assert len(val_series[0]) == 5


def test_build_scaled_train_val_series_skips_subseries_shorter_than_required_length() -> None:
    train_series, val_series, scalers = build_scaled_train_val_series(
        [_frame("A", [1.0, 2.0, 3.0])],
        val_size=2,
        min_train_len=3,
        val_context_len=2,
    )

    assert train_series == []
    assert val_series == []
    assert scalers == {}


@pytest.mark.parametrize(
    ("fraction", "message"),
    [(0.0, r"debe estar en \(0, 1\)"), (1.0, r"debe estar en \(0, 1\)")],
)
def test_build_benchmark_dataset_bundle_validates_test_only_fraction(
    fraction: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_benchmark_dataset_bundle(
            [_frame("A", [1.0, 2.0, 3.0, 4.0])],
            _frame("A", [3.0, 4.0], start="2024-01-01 02:00:00"),
            val_size=1,
            min_train_len=2,
            test_only_train_fraction=fraction,
        )


def test_build_benchmark_dataset_bundle_rejects_multicolumn_inputs() -> None:
    df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    with pytest.raises(ValueError, match="exactamente una columna"):
        build_benchmark_dataset_bundle([df], longest_segment=df[["A"]], val_size=1, min_train_len=2)


def test_build_benchmark_dataset_bundle_rejects_duplicate_source_names() -> None:
    a1 = _frame("A", [1.0, 2.0, 3.0, 4.0])
    a2 = _frame("A", [5.0, 6.0, 7.0, 8.0], start="2024-01-02")

    with pytest.raises(ValueError, match="duplicado"):
        build_benchmark_dataset_bundle(
            [a1, a2],
            a1.iloc[-2:].copy(),
            val_size=1,
            min_train_len=2,
        )


def test_build_benchmark_dataset_bundle_rejects_unnamed_test_only_series() -> None:
    with pytest.raises(ValueError, match="debe tener nombre"):
        build_benchmark_dataset_bundle(
            [_frame("A", [1.0, 2.0, 3.0, 4.0, 5.0])],
            _frame("A", [4.0, 5.0], start="2024-01-01 03:00:00"),
            val_size=1,
            min_train_len=2,
            test_only_series=[_series(None, [10.0, 11.0, 12.0, 13.0])],
        )


def test_build_benchmark_dataset_bundle_rejects_duplicate_test_only_name() -> None:
    with pytest.raises(ValueError, match="ya existe"):
        build_benchmark_dataset_bundle(
            [_frame("A", [1.0, 2.0, 3.0, 4.0, 5.0])],
            _frame("A", [4.0, 5.0], start="2024-01-01 03:00:00"),
            val_size=1,
            min_train_len=2,
            test_only_series=[_series("A", [10.0, 11.0, 12.0, 13.0])],
        )


def test_build_benchmark_dataset_bundle_raises_when_no_valid_test_columns_remain() -> None:
    with pytest.raises(ValueError, match="No hay columnas válidas"):
        build_benchmark_dataset_bundle(
            [_frame("A", [1.0, 2.0, 3.0])],
            _frame("A", [2.0, 3.0], start="2024-01-01 01:00:00"),
            val_size=2,
            min_train_len=3,
            val_context_len=2,
        )


def test_fit_darts_model_rejects_resume_mode_for_linear_regression() -> None:
    with pytest.raises(ValueError, match="no aplica a LinearRegressionModel"):
        fit_darts_model(
            LinearRegressionModel,
            series_train=[],
            series_val=None,
            size_k=2,
            model_kwargs={},
            resume_mode="last",
        )


def test_fit_darts_model_injects_output_chunk_and_validation_fit_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("airquality.modeling.training.build_base_training_kwargs", lambda: {})
    monkeypatch.setattr("airquality.modeling.training.configure_warnings", lambda quiet: None)

    calls: dict[str, object] = {}

    class DummyModel:
        def __init__(self, output_chunk_length: int, save_checkpoints: bool = False) -> None:
            calls["init"] = {
                "output_chunk_length": output_chunk_length,
                "save_checkpoints": save_checkpoints,
            }

        def fit(self, **kwargs: object) -> None:
            calls["fit"] = kwargs

    ts = TimeSeries.from_series(_series("A", [1.0, 2.0, 3.0, 4.0]), freq="h")

    fit_darts_model(
        DummyModel,
        series_train=[ts],
        series_val=[ts],
        size_k=4,
        model_kwargs={"save_checkpoints": True},
    )

    assert calls["init"] == {"output_chunk_length": 4, "save_checkpoints": True}
    assert calls["fit"]["val_series"] == [ts]
    assert calls["fit"]["dataloader_kwargs"] == {"num_workers": 2}
    assert calls["fit"]["load_best"] is True


def test_fit_darts_model_passes_stride_only_when_fit_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("airquality.modeling.training.build_base_training_kwargs", lambda: {})
    monkeypatch.setattr("airquality.modeling.training.configure_warnings", lambda quiet: None)

    calls: dict[str, object] = {}

    class TorchLikeModel:
        def __init__(self, output_chunk_length: int) -> None:
            del output_chunk_length

        def fit(self, series, verbose=False, stride=1, max_samples_per_ts=None) -> None:
            calls["fit"] = {"stride": stride, "max_samples_per_ts": max_samples_per_ts}

    ts = TimeSeries.from_series(_series("A", [1.0, 2.0, 3.0, 4.0]), freq="h")
    fit_darts_model(TorchLikeModel, series_train=[ts], series_val=None, size_k=3, model_kwargs={})

    assert calls["fit"] == {"stride": 2, "max_samples_per_ts": 256}


def test_fit_darts_model_trains_linear_regression_without_stride_typeerror() -> None:
    # Regression: `stride` is not a RegressionModel.fit parameter, so it used to
    # be forwarded to sklearn's LinearRegression.fit and raise TypeError.
    ts = TimeSeries.from_series(
        _series("A", [float(i % 5) for i in range(30)]), freq="h"
    )

    model = fit_darts_model(
        LinearRegressionModel,
        series_train=[ts],
        series_val=None,
        size_k=2,
        model_kwargs={"lags": 3},
    )

    pred = model.predict(n=2, series=ts)
    assert len(pred) == 2


def test_fit_darts_model_drops_lr_scheduler_when_validation_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("airquality.modeling.training.build_base_training_kwargs", lambda: {})
    monkeypatch.setattr("airquality.modeling.training.configure_warnings", lambda quiet: None)

    calls: dict[str, object] = {}

    class DummyModel:
        def __init__(self, output_chunk_length: int) -> None:
            calls["init"] = {"output_chunk_length": output_chunk_length}

        def fit(self, **kwargs: object) -> None:
            calls["fit"] = kwargs

    ts = TimeSeries.from_series(_series("A", [1.0, 2.0, 3.0, 4.0]), freq="h")

    fit_darts_model(
        DummyModel,
        series_train=[ts],
        series_val=None,
        size_k=3,
        model_kwargs={
            "lr_scheduler_cls": object,
            "lr_scheduler_kwargs": {"gamma": 0.9},
        },
    )

    assert calls["init"] == {"output_chunk_length": 3}
    assert "val_series" not in calls["fit"]


def test_fit_darts_model_resume_mode_loads_checkpoint_and_disables_force_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("airquality.modeling.training.build_base_training_kwargs", lambda: {})
    monkeypatch.setattr("airquality.modeling.training.configure_warnings", lambda quiet: None)

    calls: dict[str, object] = {}

    class DummyModel:
        def __init__(
            self,
            output_chunk_length: int,
            save_checkpoints: bool,
            model_name: str,
            force_reset: bool,
            work_dir: str,
        ) -> None:
            calls["init"] = {
                "output_chunk_length": output_chunk_length,
                "save_checkpoints": save_checkpoints,
                "model_name": model_name,
                "force_reset": force_reset,
                "work_dir": work_dir,
            }

        def load_weights_from_checkpoint(self, **kwargs: object) -> None:
            calls["load"] = kwargs

        def fit(self, **kwargs: object) -> None:
            calls["fit"] = kwargs

    ts = TimeSeries.from_series(_series("A", [1.0, 2.0, 3.0, 4.0]), freq="h")

    fit_darts_model(
        DummyModel,
        series_train=[ts],
        series_val=None,
        size_k=2,
        model_kwargs={
            "save_checkpoints": True,
            "model_name": "demo",
            "work_dir": "/tmp/work",
        },
        resume_mode="best",
    )

    assert calls["init"]["force_reset"] is False
    assert calls["load"] == {"best": True, "model_name": "demo", "work_dir": "/tmp/work"}


def test_train_global_methods_smoke_saves_models_and_exports_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_bundle = TrainingDatasetBundle(
        series_train=[],
        series_val=[],
    )

    calls: dict[str, object] = {}

    class DummyModel:
        def __init__(self) -> None:
            self.saved_paths: list[str] = []

        def save(self, path: str) -> None:
            self.saved_paths.append(path)

    monkeypatch.setattr(
        "airquality.modeling.training.build_model_configs",
        lambda: {"Fake": (object, {"alpha": 1})},
    )
    monkeypatch.setattr(
        "airquality.modeling.training._attach_loss_history_callback",
        lambda model_cls, model_kwargs: (dict(model_kwargs), None),
    )

    def fake_fit(
        model_cls, series_train, series_val, size_k, model_kwargs, resume_mode=None
    ):
        calls["fit"] = {
            "model_cls": model_cls,
            "series_train": series_train,
            "series_val": series_val,
            "size_k": size_k,
            "model_kwargs": model_kwargs,
            "resume_mode": resume_mode,
        }
        return DummyModel()

    monkeypatch.setattr("airquality.modeling.training.fit_darts_model", fake_fit)

    trained = train_global_methods(
        dataset_bundle=dataset_bundle,
        size_k=3,
        method_names=["Fake"],
        csv_output_path=str(tmp_path / "curves.csv"),
        model_output_dir=str(tmp_path / "models"),
    )

    assert set(trained) == {"Fake"}
    assert calls["fit"]["size_k"] == 3
    assert calls["fit"]["model_kwargs"] == {"alpha": 1, "model_name": "Fake_k3"}
    assert (tmp_path / "curves.csv").exists()
    assert trained["Fake"].saved_paths == [str(tmp_path / "models" / "Fake_k3.pt")]
