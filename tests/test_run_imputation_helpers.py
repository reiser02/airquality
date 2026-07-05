"""Tests benchmark runner utilities, model resolution, and orchestration helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airquality.imputation.imputers import DartsGlobalGapImputer
from airquality.imputation.registry import resolve_imputer_names
from airquality.imputation.run_benchmark import (
    BenchmarkRunConfig,
    TSPulseModelConfig,
    TSPULSE_FINETUNED_MODEL_NAME,
    TSPULSE_ORIGINAL_MODEL_NAME,
    _build_parallel_task_common,
    _build_montecarlo_seed_list,
    _merge_plot_stores,
    _normalize_tspulse_model_path,
    _resolve_requested_models,
    _resolve_repo_root,
    load_darts_models_from_artifacts,
    run_imputation_benchmark,
    run_imputation_benchmark_parallel,
    summarize_results_by_model,
    summarize_montecarlo_rankings,
)


def test_darts_global_imputer_casts_context_to_float32() -> None:
    from darts import TimeSeries

    class RecordingModel:
        input_chunk_length = 1

        def __init__(self) -> None:
            self.series_dtype = None

        def predict(self, *, n, series, **kwargs):
            del kwargs
            self.series_dtype = series.dtype
            index = pd.date_range(series.end_time() + series.freq, periods=n, freq=series.freq_str)
            return TimeSeries.from_series(pd.Series([0.0] * n, index=index), freq=series.freq_str)

    model = RecordingModel()
    full = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
        name="S",
        dtype="float64",
    )
    gap = pd.date_range("2024-01-01 02:00:00", periods=1, freq="h")

    DartsGlobalGapImputer(model, model_name="R").impute_gaps(
        series_name="S",
        all_series_map={"S": full},
        gap_windows=[gap],
        test_index=full.index,
        scaler=None,
        freq="h",
        config_workers={"num_workers": 0},
    )

    assert str(model.series_dtype) == "float32"


def test_resolve_repo_root_uses_explicit_path(tmp_path: Path) -> None:
    out = _resolve_repo_root(tmp_path)
    assert out == tmp_path.resolve()


def test_normalize_tspulse_model_path_treats_blank_as_none() -> None:
    assert _normalize_tspulse_model_path(None) is None
    assert _normalize_tspulse_model_path("   ") is None
    assert _normalize_tspulse_model_path(" models/tspulse ") == "models/tspulse"


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


def test_resolve_requested_models_deduplicates_darts_and_detects_tspulse() -> None:
    darts, prophet, tspulse_model_names, interp, linear = _resolve_requested_models(
        ["TiDE", "TiDE", " TCN ", TSPULSE_ORIGINAL_MODEL_NAME]
    )

    assert darts == ["TiDE", "TCN"]
    assert prophet == []
    assert tspulse_model_names == [TSPULSE_ORIGINAL_MODEL_NAME]
    assert interp == []
    assert linear == []


def test_resolve_requested_models_accepts_explicit_tspulse_variants() -> None:
    darts, prophet, tspulse_model_names, interp, linear = _resolve_requested_models(
        ["TSPulse_FineTuned", "TSPulse", "TSPulse"]
    )

    assert darts == []
    assert prophet == []
    assert tspulse_model_names == [
        TSPULSE_FINETUNED_MODEL_NAME,
        TSPULSE_ORIGINAL_MODEL_NAME,
    ]
    assert interp == []
    assert linear == []


def test_resolve_requested_models_detects_prophet_family() -> None:
    darts, prophet, tspulse_model_names, interp, linear = _resolve_requested_models(
        ["NLinear", "Prophet", "Prophet"]
    )

    assert darts == ["NLinear"]
    assert prophet == ["Prophet"]
    assert tspulse_model_names == []
    assert interp == []
    assert linear == []


def test_resolve_requested_models_detects_linear_family() -> None:
    darts, prophet, tspulse_model_names, interp, linear = _resolve_requested_models(
        ["NLinear", "LinearInterp", "LinearInterp"]
    )

    assert darts == ["NLinear"]
    assert prophet == []
    assert tspulse_model_names == []
    assert interp == []
    assert linear == ["LinearInterp"]


def test_resolve_requested_models_detects_interp_family() -> None:
    darts, prophet, tspulse_model_names, interp, linear = _resolve_requested_models(
        ["interp", "interp", "LinearInterp"]
    )

    assert darts == []
    assert prophet == []
    assert tspulse_model_names == []
    assert interp == ["interp"]
    assert linear == ["LinearInterp"]


def test_resolve_requested_models_accepts_every_registry_name() -> None:
    # Regression: `resolve_imputer_names(["all"])` used to include `interp`,
    # which crashed `_resolve_requested_models` with a raw KeyError.
    buckets = _resolve_requested_models(resolve_imputer_names(["all"]))

    resolved = [name for bucket in buckets for name in bucket]
    assert sorted(resolved) == sorted(resolve_imputer_names(["all"]))


def test_merge_plot_stores_combines_predictions_and_preserves_metadata() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    stores = [
        {
            2: {
                "series": {
                    "A": {
                        "actual": pd.Series([1.0, 2.0], index=idx),
                        "preds": {"M1": pd.Series([1.1, 2.1], index=idx)},
                    }
                },
                "strategy": "block",
            }
        },
        {
            2: {
                "series": {
                    "A": {
                        "naive_mase": pd.Series([0.5, 0.4], index=idx),
                        "preds": {"M2": pd.Series([0.9, 1.9], index=idx)},
                    }
                }
            }
        },
    ]

    merged = _merge_plot_stores(stores)

    assert list(merged) == [2]
    assert set(merged[2]["series"]["A"]) == {"actual", "preds", "naive_mase"}
    assert set(merged[2]["series"]["A"]["preds"]) == {"M1", "M2"}
    assert merged[2]["strategy"] == "block"
    assert merged[2]["series"]["A"]["naive_mase"].iloc[0] == 0.5


def test_build_parallel_task_common_roundtrips_into_benchmark_config(tmp_path: Path) -> None:
    config = BenchmarkRunConfig(
        size_k=5,
        force_cpu=True,
        tspulse=TSPulseModelConfig(
            model_id="model-id",
            revision="rev-1",
            model_path=None,
            local_files_only=True,
            hf_token="secret",
        ),
        gap_sizes=(1, 3),
        num_gaps=4,
        gap_strategy="hybrid_tspulse",
        metrics=("mae", "rmse"),
        random_seed=123,
        seasonality_m=24,
        freq="h",
        val_size=48,
        val_context_len=72,
        min_train_len_base=96,
    )

    task_common = _build_parallel_task_common(repo_root=tmp_path, config=config)
    rebuilt = BenchmarkRunConfig.from_mapping(task_common)

    assert task_common["repo_root"] == str(tmp_path)
    assert rebuilt == config


def test_run_imputation_benchmark_resolves_config_defaults_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.configure_warnings",
        lambda quiet: None,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._resolve_repo_root",
        lambda repo_root: Path("/tmp/runtime-root"),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._resolve_requested_models",
        lambda model_names: (list(model_names), [], [], [], []),
    )

    def fake_cfg_get_int(section: str, option: str, default: int) -> int:
        overrides = {
            ("benchmark", "size_k"): 9,
            ("benchmark", "num_gaps"): 4,
            ("benchmark", "random_seed"): 77,
            ("benchmark", "seasonality_m"): 12,
            ("benchmark", "val_size"): 30,
            ("benchmark", "val_context_len"): 31,
            ("benchmark", "min_train_len_base"): 32,
        }
        return overrides.get((section, option), default)

    def fake_cfg_get_str(section: str, option: str, default: str) -> str:
        overrides = {
            ("tspulse", "model_id"): "runtime-model",
            ("tspulse", "revision"): "runtime-rev",
            ("benchmark", "tspulse_model_path"): "runtime/path",
            ("benchmark", "gap_strategy"): "runtime-strategy",
            ("data", "freq"): "30min",
        }
        return overrides.get((section, option), default)

    monkeypatch.setattr("airquality.imputation.run_benchmark.cfg_get_int", fake_cfg_get_int)
    monkeypatch.setattr("airquality.imputation.run_benchmark.cfg_get_str", fake_cfg_get_str)
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.cfg_get_csv_list",
        lambda section, option, default: {
            ("benchmark", "model_names"): ("TiDE", "TCN"),
            ("benchmark", "gap_sizes"): ("2", "6"),
            ("benchmark", "metrics"): ("mae", "rmse"),
        }.get((section, option), default),
    )

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_dataset_bundle_from_config",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {"TiDE": object()},
    )

    def fake_execute(*, model_dict: dict[str, object], dataset_bundle: object, config: BenchmarkRunConfig) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
        captured["config"] = config
        return pd.DataFrame([{"Modelo": "TiDE", "MAE": 1.0}]), {}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.summarize_results_by_model",
        lambda df: df,
    )

    run_imputation_benchmark(
    )

    assert captured["config"] == BenchmarkRunConfig(
        size_k=9,
        force_cpu=True,
        tspulse=TSPulseModelConfig(
            model_id="runtime-model",
            revision="runtime-rev",
            model_path="runtime/path",
            local_files_only=False,
            hf_token=None,
        ),
        gap_sizes=(2, 6),
        num_gaps=4,
        gap_strategy="runtime-strategy",
        metrics=("mae", "rmse"),
        random_seed=77,
        seasonality_m=12,
        freq="30min",
        val_size=30,
        val_context_len=31,
        min_train_len_base=32,
    )


def test_run_imputation_benchmark_allows_benchmark_overrides_without_dataset_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("airquality.imputation.run_benchmark.configure_warnings", lambda quiet: None)
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._resolve_repo_root",
        lambda repo_root: Path("/tmp/runtime-root"),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_dataset_bundle_from_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {"TiDE": object()},
    )

    def fake_execute(*, model_dict: dict[str, object], dataset_bundle: object, config: BenchmarkRunConfig) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
        captured["config"] = config
        return pd.DataFrame([{"Modelo": "TiDE", "MAE": 1.0}]), {}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.summarize_results_by_model",
        lambda df: df,
    )

    run_imputation_benchmark(
        size_k=11,
        model_names=("TiDE",),
        tspulse_model_id="override-model",
        tspulse_revision="override-rev",
        tspulse_model_path="override/path",
        gap_sizes=(7, 9),
        num_gaps=8,
        gap_strategy="override-strategy",
        metrics=("mae",),
        random_seed=1234,
        seasonality_m=6,
        freq="15min",
        val_size=12,
        val_context_len=13,
        min_train_len_base=14,
    )

    assert captured["config"] == BenchmarkRunConfig(
        size_k=11,
        force_cpu=True,
        tspulse=TSPulseModelConfig(
            model_id="override-model",
            revision="override-rev",
            model_path="override/path",
            local_files_only=False,
            hf_token=None,
        ),
        gap_sizes=(7, 9),
        num_gaps=8,
        gap_strategy="override-strategy",
        metrics=("mae",),
        random_seed=1234,
        seasonality_m=6,
        freq="15min",
        val_size=12,
        val_context_len=13,
        min_train_len_base=14,
    )


def test_build_montecarlo_seed_list_uses_explicit_seeds() -> None:
    assert _build_montecarlo_seed_list(seeds=[5, 9], n_runs=3, seed_start=0, seed_step=1) == [5, 9]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seeds": [], "n_runs": 1, "seed_start": 0, "seed_step": 1}, "no puede estar vacio"),
        ({"seeds": None, "n_runs": 0, "seed_start": 0, "seed_step": 1}, "debe ser > 0"),
        ({"seeds": None, "n_runs": 2, "seed_start": 0, "seed_step": 0}, "no puede ser 0"),
    ],
)
def test_build_montecarlo_seed_list_validates_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build_montecarlo_seed_list(**kwargs)


def test_summarize_montecarlo_rankings_aggregates_and_sorts() -> None:
    df = pd.DataFrame(
        {
            "Modelo": ["A", "A", "B", "B"],
            "Seed": [1, 2, 1, 2],
            "MAE": [2.0, 4.0, 1.0, 1.0],
            "RMSE": [2.0, 4.0, 1.0, 1.0],
            "MASE": [2.0, 4.0, 1.0, 1.0],
        }
    )

    out = summarize_montecarlo_rankings(df)

    assert list(out["Modelo"]) == ["B", "A"]
    assert list(out["Runs"]) == [2, 2]
    assert out.loc[out["Modelo"] == "A", "MASE_Mean"].item() == 3.0


def test_load_darts_models_from_artifacts_loads_cpu_model_and_skips_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "Fake_k3.pt").write_text("weights", encoding="utf-8")

    calls: dict[str, object] = {}

    class DummyInnerModel:
        def __init__(self) -> None:
            self.float_called = False

        def float(self) -> None:
            self.float_called = True

    class DummyLoadedModel:
        def __init__(self) -> None:
            self.trainer_params = {"existing": True}
            self.model_params = {"pl_trainer_kwargs": {"before": True}}
            self.model = DummyInnerModel()

    class DummyModelCls:
        @classmethod
        def load(cls, path: str, **kwargs: object) -> DummyLoadedModel:
            calls["path"] = path
            calls["kwargs"] = kwargs
            return DummyLoadedModel()

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.build_model_configs",
        lambda: {"Fake": (DummyModelCls, {}), "Missing": (DummyModelCls, {})},
    )
    monkeypatch.setattr("airquality.imputation.run_benchmark.resolve_device", lambda preferred: "cpu")
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.build_lightning_trainer_kwargs",
        lambda *args, **kwargs: {"accelerator": args[0], **kwargs},
    )

    loaded = load_darts_models_from_artifacts(
        repo_root=tmp_path,
        size_k=3,
        model_names=("Fake", "Missing"),
        force_cpu=True,
        strict=False,
    )

    assert set(loaded) == {"Fake"}
    assert calls["path"] == str(models_dir / "Fake_k3.pt")
    trainer_kwargs = calls["kwargs"]
    assert trainer_kwargs["map_location"] == "cpu"
    assert trainer_kwargs["pl_trainer_kwargs"]["accelerator"] == "cpu"
    wrapped = loaded["Fake"]
    assert isinstance(wrapped, DartsGlobalGapImputer)
    assert wrapped._model.model.float_called is True
    assert wrapped._model.trainer_params["enable_progress_bar"] is False
    assert wrapped._model.model_params["pl_trainer_kwargs"]["logger"] is False


def test_load_darts_models_from_artifacts_raises_for_missing_weights_in_strict_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyModelCls:
        @classmethod
        def load(cls, path: str, **kwargs: object) -> object:
            raise AssertionError("load should not be called")

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.build_model_configs",
        lambda: {"Fake": (DummyModelCls, {})},
    )
    monkeypatch.setattr("airquality.imputation.run_benchmark.resolve_device", lambda preferred: "cpu")

    with pytest.raises(FileNotFoundError, match="Fake_k3.pt"):
        load_darts_models_from_artifacts(
            repo_root=tmp_path,
            size_k=3,
            model_names=("Fake",),
            force_cpu=True,
            strict=True,
        )


def test_run_imputation_benchmark_smoke_uses_shared_runner_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_bundle = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.build_dataset_bundle_for_imputation",
        lambda **kwargs: dataset_bundle,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {"TiDE": object()},
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_model_names",
        lambda: ("TiDE",),
    )

    def fake_execute(*, model_dict, dataset_bundle, config):
        captured["model_names"] = tuple(model_dict.keys())
        captured["dataset_bundle"] = dataset_bundle
        captured["config"] = config
        return (
            pd.DataFrame(
                {
                    "Modelo": ["TiDE"],
                    "Serie": ["S1"],
                    "Gap_Size": [2],
                    "MAE": [1.0],
                    "RMSE": [1.5],
                    "MASE": [0.5],
                }
            ),
            {
                2: {
                    "series": {
                        "S1": {
                            "actual": pd.Series(dtype=float),
                            "preds": {"TiDE": pd.Series(dtype=float)},
                            "naive_mase": pd.Series(dtype=float),
                        }
                    }
                }
            },
        )

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )

    results_df, ranking_df, plot_store = run_imputation_benchmark(
        repo_root=tmp_path,
    )

    assert list(results_df["Modelo"]) == ["TiDE"]
    assert list(ranking_df["Modelo"]) == ["TiDE"]
    assert list(plot_store) == [2]
    assert set(plot_store[2]["series"]["S1"]) == {"actual", "preds", "naive_mase"}
    assert captured["model_names"] == ("TiDE",)
    assert captured["dataset_bundle"] is dataset_bundle
    assert isinstance(captured["config"], BenchmarkRunConfig)


def test_run_imputation_benchmark_parallel_max_workers_one_combines_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_bundle = object()

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.build_dataset_bundle_for_imputation",
        lambda **kwargs: dataset_bundle,
    )

    def fake_load_models(*, model_names, **kwargs):
        return {str(model_names[0]): object()}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        fake_load_models,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_model_names",
        lambda: ("TiDE", "TCN"),
    )

    def fake_execute(*, model_dict, dataset_bundle, config):
        model_name = next(iter(model_dict))
        return (
            pd.DataFrame(
                {
                    "Modelo": [model_name],
                    "Serie": ["S1"],
                    "Gap_Size": [1],
                    "MAE": [1.0 if model_name == "TiDE" else 2.0],
                    "RMSE": [1.0 if model_name == "TiDE" else 2.0],
                    "MASE": [1.0 if model_name == "TiDE" else 2.0],
                }
            ),
            {
                1: {
                    "series": {
                        "S1": {
                            "actual": pd.Series(dtype=float),
                            "preds": {model_name: pd.Series(dtype=float)},
                            "naive_mase": pd.Series(dtype=float),
                        }
                    }
                }
            },
        )

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )

    results_df, ranking_df, plot_store = run_imputation_benchmark_parallel(
        repo_root=tmp_path,
        max_workers=1,
    )

    assert set(results_df["Modelo"]) == {"TiDE", "TCN"}
    assert list(ranking_df["Modelo"]) == ["TiDE", "TCN"]
    assert set(plot_store[1]["series"]["S1"]) == {"actual", "preds", "naive_mase"}
    assert set(plot_store[1]["series"]["S1"]["preds"]) == {"TiDE", "TCN"}


def test_run_imputation_benchmark_adds_original_and_finetuned_tspulse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_bundle = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.build_dataset_bundle_for_imputation",
        lambda **kwargs: dataset_bundle,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {},
    )
    monkeypatch.setattr("airquality.imputation.run_benchmark.TSFM_PUBLIC_AVAILABLE", True)
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_model_names",
        lambda: (TSPULSE_ORIGINAL_MODEL_NAME, TSPULSE_FINETUNED_MODEL_NAME),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_tspulse_model_path",
        lambda: "models/fine-tuned",
    )

    def fake_build_tspulse_model(model_config, *, freq):
        return {"model_path": model_config.model_path, "freq": freq}

    def fake_execute(*, model_dict, dataset_bundle, config):
        captured["model_dict"] = model_dict
        return (
            pd.DataFrame(
                {
                    "Modelo": list(model_dict),
                    "Serie": ["S1", "S1"],
                    "Gap_Size": [1, 1],
                    "MAE": [1.0, 2.0],
                    "RMSE": [1.0, 2.0],
                    "MASE": [1.0, 2.0],
                }
            ),
            {},
        )

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_tspulse_model",
        fake_build_tspulse_model,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )

    results_df, ranking_df, _ = run_imputation_benchmark(
        repo_root=tmp_path,
    )

    assert list(results_df["Modelo"]) == ["TSPulse", "TSPulse_FineTuned"]
    assert list(ranking_df["Modelo"]) == ["TSPulse", "TSPulse_FineTuned"]
    assert captured["model_dict"] == {
        "TSPulse": {"model_path": None, "freq": "h"},
        "TSPulse_FineTuned": {"model_path": "models/fine-tuned", "freq": "h"},
    }


def test_run_imputation_benchmark_parallel_adds_original_and_finetuned_tspulse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_bundle = object()

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.build_dataset_bundle_for_imputation",
        lambda **kwargs: dataset_bundle,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {},
    )
    monkeypatch.setattr("airquality.imputation.run_benchmark.TSFM_PUBLIC_AVAILABLE", True)
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_model_names",
        lambda: (TSPULSE_ORIGINAL_MODEL_NAME, TSPULSE_FINETUNED_MODEL_NAME),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_tspulse_model_path",
        lambda: "models/fine-tuned",
    )

    def fake_build_tspulse_model(model_config, *, freq):
        return {"model_path": model_config.model_path, "freq": freq}

    def fake_execute(*, model_dict, dataset_bundle, config):
        model_name = next(iter(model_dict))
        return (
            pd.DataFrame(
                {
                    "Modelo": [model_name],
                    "Serie": ["S1"],
                    "Gap_Size": [1],
                    "MAE": [1.0 if model_name == "TSPulse" else 2.0],
                    "RMSE": [1.0 if model_name == "TSPulse" else 2.0],
                    "MASE": [1.0 if model_name == "TSPulse" else 2.0],
                }
            ),
            {},
        )

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_tspulse_model",
        fake_build_tspulse_model,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )

    results_df, ranking_df, _ = run_imputation_benchmark_parallel(
        repo_root=tmp_path,
        max_workers=1,
    )

    assert set(results_df["Modelo"]) == {"TSPulse", "TSPulse_FineTuned"}
    assert list(ranking_df["Modelo"]) == ["TSPulse", "TSPulse_FineTuned"]


def test_run_imputation_benchmark_can_request_only_base_tspulse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.configure_warnings",
        lambda quiet: None,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._resolve_repo_root",
        lambda repo_root: tmp_path,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_dataset_bundle_from_config",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {},
    )
    monkeypatch.setattr("airquality.imputation.run_benchmark.TSFM_PUBLIC_AVAILABLE", True)
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_tspulse_model_path",
        lambda: "models/fine-tuned",
    )

    def fake_build_tspulse_model(model_config, *, freq):
        return {"model_path": model_config.model_path, "freq": freq}

    def fake_execute(*, model_dict, dataset_bundle, config):
        captured["model_dict"] = model_dict
        return (
            pd.DataFrame(
                {
                    "Modelo": list(model_dict),
                    "Serie": ["S1"],
                    "Gap_Size": [1],
                    "MAE": [1.0],
                    "RMSE": [1.0],
                    "MASE": [1.0],
                }
            ),
            {},
        )

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_tspulse_model",
        fake_build_tspulse_model,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )

    results_df, ranking_df, _ = run_imputation_benchmark(
        repo_root=tmp_path,
        model_names=[TSPULSE_ORIGINAL_MODEL_NAME],
    )

    assert list(results_df["Modelo"]) == [TSPULSE_ORIGINAL_MODEL_NAME]
    assert list(ranking_df["Modelo"]) == [TSPULSE_ORIGINAL_MODEL_NAME]
    assert captured["model_dict"] == {
        TSPULSE_ORIGINAL_MODEL_NAME: {"model_path": None, "freq": "h"},
    }


def test_run_imputation_benchmark_can_request_only_finetuned_tspulse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.configure_warnings",
        lambda quiet: None,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._resolve_repo_root",
        lambda repo_root: tmp_path,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_dataset_bundle_from_config",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {},
    )
    monkeypatch.setattr("airquality.imputation.run_benchmark.TSFM_PUBLIC_AVAILABLE", True)
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_tspulse_model_path",
        lambda: "models/fine-tuned",
    )

    def fake_build_tspulse_model(model_config, *, freq):
        return {"model_path": model_config.model_path, "freq": freq}

    def fake_execute(*, model_dict, dataset_bundle, config):
        captured["model_dict"] = model_dict
        return (
            pd.DataFrame(
                {
                    "Modelo": list(model_dict),
                    "Serie": ["S1"],
                    "Gap_Size": [1],
                    "MAE": [2.0],
                    "RMSE": [2.0],
                    "MASE": [2.0],
                }
            ),
            {},
        )

    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_tspulse_model",
        fake_build_tspulse_model,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._execute_benchmark_with_dataset",
        fake_execute,
    )

    results_df, ranking_df, _ = run_imputation_benchmark(
        repo_root=tmp_path,
        model_names=[TSPULSE_FINETUNED_MODEL_NAME],
    )

    assert list(results_df["Modelo"]) == [TSPULSE_FINETUNED_MODEL_NAME]
    assert list(ranking_df["Modelo"]) == [TSPULSE_FINETUNED_MODEL_NAME]
    assert captured["model_dict"] == {
        TSPULSE_FINETUNED_MODEL_NAME: {"model_path": "models/fine-tuned", "freq": "h"},
    }


def test_run_imputation_benchmark_rejects_finetuned_tspulse_without_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.configure_warnings",
        lambda quiet: None,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._resolve_repo_root",
        lambda repo_root: tmp_path,
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._build_dataset_bundle_from_config",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark.load_darts_models_from_artifacts",
        lambda **kwargs: {},
    )
    monkeypatch.setattr("airquality.imputation.run_benchmark.TSFM_PUBLIC_AVAILABLE", True)
    monkeypatch.setattr(
        "airquality.imputation.run_benchmark._default_tspulse_model_path",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="TSPulse_FineTuned"):
        run_imputation_benchmark(
            repo_root=tmp_path,
            model_names=[TSPULSE_FINETUNED_MODEL_NAME],
        )
