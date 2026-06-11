"""High-level orchestration for loading models and running imputation benchmarks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from airquality.data.io import configure_warnings, load_and_normalize_series, resolve_device
from airquality.data.segments import get_longest_segment
from airquality.config import cfg_get_csv_list, cfg_get_int, cfg_get_str

from airquality.imputation.benchmark import (
    TSFM_PUBLIC_AVAILABLE,
    TSPulseHistoricalImputer,
    execute_complete_pipeline,
)
from airquality.modeling.training import build_benchmark_dataset_bundle
from airquality.modeling.training_config import (
    BenchmarkDatasetBundle,
    build_lightning_trainer_kwargs,
    build_model_configs,
)


def _default_model_names() -> tuple[str, ...]:
    """Return the default Darts model names configured for benchmarking."""
    return cfg_get_csv_list(
        "benchmark",
        "model_names",
        ("TiDE", "NHiTS", "TCN", "TSMixer", "RNN", "NLinear", "DLinear"),
    )


def _default_gap_sizes() -> tuple[int, ...]:
    """Return the default synthetic gap sizes used during evaluation."""
    return tuple(int(v) for v in cfg_get_csv_list("benchmark", "gap_sizes", ("1", "2", "5", "10")))


def _default_metrics() -> tuple[str, ...]:
    """Return the default metric names computed by the benchmark."""
    return cfg_get_csv_list("benchmark", "metrics", ("mae", "rmse", "mase"))


DEFAULT_MODEL_NAMES = _default_model_names()
DEFAULT_GAP_SIZES = _default_gap_sizes()
DEFAULT_METRICS = _default_metrics()
TSPULSE_ORIGINAL_MODEL_NAME = "TSPulse"
TSPULSE_FINETUNED_MODEL_NAME = "TSPulse_FineTuned"


def _default_tspulse_model_id() -> str:
    """Return the default Hugging Face model identifier for TSPulse."""
    return cfg_get_str("tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1")


def _default_tspulse_revision() -> str:
    """Return the default model revision used for TSPulse inference."""
    return cfg_get_str("tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1")


def _default_tspulse_model_path() -> str | None:
    """Return the optional fine-tuned TSPulse artifact path from config."""
    return _normalize_tspulse_model_path(cfg_get_str("benchmark", "tspulse_model_path", ""))


@dataclass(frozen=True)
class TSPulseModelConfig:
    """Serializable configuration for loading one TSPulse model."""

    model_id: str
    revision: str
    model_path: str | None
    local_files_only: bool
    hf_token: str | None


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Normalized runtime settings shared across benchmark entry points."""

    size_k: int
    force_cpu: bool
    tspulse: TSPulseModelConfig
    gap_sizes: tuple[int, ...]
    num_gaps: int
    gap_strategy: str
    metrics: tuple[str, ...]
    random_seed: int
    seasonality_m: int
    freq: str
    val_size: int
    val_context_len: int
    min_train_len_base: int

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> BenchmarkRunConfig:
        return cls(
            size_k=int(raw["size_k"]),
            force_cpu=bool(raw["force_cpu"]),
            tspulse=TSPulseModelConfig(
                model_id=str(raw["tspulse_model_id"]),
                revision=str(raw["tspulse_revision"]),
                model_path=_normalize_tspulse_model_path(raw.get("tspulse_model_path")),
                local_files_only=bool(raw["local_files_only"]),
                hf_token=raw.get("hf_token"),
            ),
            gap_sizes=tuple(int(x) for x in raw["gap_sizes"]),
            num_gaps=int(raw["num_gaps"]),
            gap_strategy=str(raw["gap_strategy"]),
            metrics=tuple(str(metric) for metric in raw["metrics"]),
            random_seed=int(raw["random_seed"]),
            seasonality_m=int(raw["seasonality_m"]),
            freq=str(raw["freq"]),
            val_size=int(raw["val_size"]),
            val_context_len=int(raw["val_context_len"]),
            min_train_len_base=int(raw["min_train_len_base"]),
        )


def _to_float32_timeseries(series: Any) -> Any:
    """Best-effort cast of a series-like object to float32."""
    if hasattr(series, "astype"):
        try:
            return series.astype("float32")
        except Exception:
            return series
    return series


class Float32InputModelAdapter:
    """Proxy that casts input series to float32 before model prediction."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        if "series" in kwargs:
            kwargs["series"] = _to_float32_timeseries(kwargs["series"])
        elif len(args) >= 2:
            args_list = list(args)
            args_list[1] = _to_float32_timeseries(args_list[1])
            args = tuple(args_list)
        return self._model.predict(*args, **kwargs)


def _resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the repository root from an explicit path or the current working tree."""
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "src").exists():
        return cwd
    if cwd.name == "src" and (cwd.parent / "src").exists():
        return cwd.parent
    return cwd


def _apply_inference_trainer_overrides(model: Any) -> None:
    """Disable training-side logging/checkpointing during inference calls."""
    overrides = {
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "logger": False,
    }

    trainer_params = getattr(model, "trainer_params", None)
    if isinstance(trainer_params, dict):
        trainer_params.update(overrides)

    model_params = getattr(model, "model_params", None)
    if isinstance(model_params, dict):
        pl_kwargs = model_params.get("pl_trainer_kwargs")
        if isinstance(pl_kwargs, dict):
            pl_kwargs.update(overrides)


def _normalize_tspulse_model_path(model_path: str | None) -> str | None:
    """Trim an optional local TSPulse model path and coerce empty strings to None."""
    if model_path is None:
        return None

    normalized = str(model_path).strip()
    return normalized or None


def _load_benchmark_run_config(
    force_cpu: bool,
    local_files_only: bool,
    hf_token: str | None,
    size_k: int | None = None,
    tspulse_model_id: str | None = None,
    tspulse_revision: str | None = None,
    tspulse_model_path: str | None = None,
    gap_sizes: Sequence[int] | None = None,
    num_gaps: int | None = None,
    gap_strategy: str | None = None,
    metrics: Sequence[str] | None = None,
    random_seed: int | None = None,
    seasonality_m: int | None = None,
    freq: str | None = None,
    val_size: int | None = None,
    val_context_len: int | None = None,
    min_train_len_base: int | None = None,
) -> BenchmarkRunConfig:
    """Load the effective benchmark runtime config from the project cfg files."""
    return BenchmarkRunConfig(
        size_k=cfg_get_int("benchmark", "size_k", 5) if size_k is None else int(size_k),
        force_cpu=bool(force_cpu),
        tspulse=TSPulseModelConfig(
            model_id=str(_default_tspulse_model_id() if tspulse_model_id is None else tspulse_model_id),
            revision=str(_default_tspulse_revision() if tspulse_revision is None else tspulse_revision),
            model_path=_normalize_tspulse_model_path(
                _default_tspulse_model_path() if tspulse_model_path is None else tspulse_model_path
            ),
            local_files_only=bool(local_files_only),
            hf_token=hf_token,
        ),
        gap_sizes=tuple(int(x) for x in (_default_gap_sizes() if gap_sizes is None else gap_sizes)),
        num_gaps=cfg_get_int("benchmark", "num_gaps", 3) if num_gaps is None else int(num_gaps),
        gap_strategy=str(
            cfg_get_str("benchmark", "gap_strategy", "hybrid_tspulse")
            if gap_strategy is None
            else gap_strategy
        ),
        metrics=tuple(str(metric) for metric in (_default_metrics() if metrics is None else metrics)),
        random_seed=cfg_get_int("benchmark", "random_seed", 42) if random_seed is None else int(random_seed),
        seasonality_m=cfg_get_int("benchmark", "seasonality_m", 24) if seasonality_m is None else int(seasonality_m),
        freq=str(cfg_get_str("data", "freq", "h") if freq is None else freq),
        val_size=cfg_get_int("benchmark", "val_size", 48) if val_size is None else int(val_size),
        val_context_len=cfg_get_int("benchmark", "val_context_len", 72) if val_context_len is None else int(val_context_len),
        min_train_len_base=cfg_get_int("benchmark", "min_train_len_base", 72) if min_train_len_base is None else int(min_train_len_base),
    )


def _build_dataset_bundle_from_config(repo_root: Path, config: BenchmarkRunConfig) -> BenchmarkDatasetBundle:
    """Construct the train/validation/test bundle required for one benchmark run."""
    return build_dataset_bundle_for_imputation(
        size_k=config.size_k,
        val_size=config.val_size,
        val_context_len=config.val_context_len,
        min_train_len_base=config.min_train_len_base,
        freq=config.freq,
    )


def _build_tspulse_model(model_config: TSPulseModelConfig, freq: str) -> TSPulseHistoricalImputer:
    """Instantiate the TSPulse imputer adapter from normalized config values."""
    return TSPulseHistoricalImputer(
        model_id=model_config.model_id,
        revision=model_config.revision,
        model_path=model_config.model_path,
        freq=freq,
        local_files_only=model_config.local_files_only,
        hf_token=model_config.hf_token,
    )


def _build_tspulse_model_dict(
    model_config: TSPulseModelConfig,
    *,
    freq: str,
    model_names: Sequence[str],
) -> dict[str, TSPulseHistoricalImputer]:
    """Build only the requested benchmark-ready TSPulse variants."""
    model_dict: dict[str, TSPulseHistoricalImputer] = {}

    for model_name in model_names:
        if model_name == TSPULSE_ORIGINAL_MODEL_NAME:
            model_dict[TSPULSE_ORIGINAL_MODEL_NAME] = _build_tspulse_model(
                replace(model_config, model_path=None),
                freq=freq,
            )
        elif model_name == TSPULSE_FINETUNED_MODEL_NAME:
            if model_config.model_path is None:
                raise RuntimeError(
                    "No se puede evaluar TSPulse_FineTuned sin `tspulse_model_path`."
                )
            model_dict[TSPULSE_FINETUNED_MODEL_NAME] = _build_tspulse_model(
                model_config,
                freq=freq,
            )
        else:
            raise ValueError(f"Modelo TSPulse no soportado: {model_name}")

    return model_dict


def _execute_benchmark_with_dataset(
    model_dict: dict[str, Any],
    dataset_bundle: BenchmarkDatasetBundle,
    config: BenchmarkRunConfig,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Run the full imputation benchmark using a prepared dataset bundle."""
    return execute_complete_pipeline(
        model_dict=model_dict,
        dataset_bundle=dataset_bundle,
        gap_sizes=config.gap_sizes,
        num_gaps=config.num_gaps,
        metrics=config.metrics,
        random_seed=config.random_seed,
        freq=config.freq,
        seasonality_m=config.seasonality_m,
        gap_strategy=config.gap_strategy,
    )


def _build_parallel_task_common(repo_root: Path, config: BenchmarkRunConfig) -> dict[str, Any]:
    """Serialize the common task payload shared by parallel worker processes."""
    return {
        "repo_root": str(repo_root),
        "size_k": config.size_k,
        "force_cpu": config.force_cpu,
        "local_files_only": config.tspulse.local_files_only,
        "hf_token": config.tspulse.hf_token,
        "tspulse_model_id": config.tspulse.model_id,
        "tspulse_revision": config.tspulse.revision,
        "tspulse_model_path": config.tspulse.model_path,
        "gap_sizes": config.gap_sizes,
        "num_gaps": config.num_gaps,
        "gap_strategy": config.gap_strategy,
        "metrics": config.metrics,
        "random_seed": config.random_seed,
        "seasonality_m": config.seasonality_m,
        "freq": config.freq,
        "val_size": config.val_size,
        "val_context_len": config.val_context_len,
        "min_train_len_base": config.min_train_len_base,
    }


def load_series(freq: str) -> list[pd.DataFrame]:
    """Load and normalize the raw project series selected for benchmarking."""
    series_dfs = load_and_normalize_series(
        freq=freq,
        name_from_path=True,
    )
    if not series_dfs:
        raise RuntimeError(
            "No se pudieron construir series validas desde los archivos cargados."
        )
    return series_dfs


def build_dataset_bundle_for_imputation(
    size_k: int,
    val_size: int,
    val_context_len: int,
    min_train_len_base: int,
    freq: str,
) -> BenchmarkDatasetBundle:
    """Build the benchmark dataset bundle from raw files and held-out segments."""
    series_dfs = load_series(
        freq=freq,
    )
    longest_segment = get_longest_segment(series_dfs, verbose=False)
    if longest_segment.empty:
        raise RuntimeError("get_longest_segment devolvio un DataFrame vacio.")

    return build_benchmark_dataset_bundle(
        series_dfs=series_dfs,
        longest_segment=longest_segment,
        val_size=val_size,
        min_train_len=int(min_train_len_base + size_k),
        val_context_len=val_context_len,
    )


def load_darts_models_from_artifacts(
    *,
    repo_root: Path,
    size_k: int,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    force_cpu: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Load trained Darts model artifacts from the repository `models/` directory."""
    models_dir = repo_root / "models"
    model_configs = build_model_configs()
    preferred = "cpu" if force_cpu else "cuda"
    resolved = resolve_device(preferred)
    use_cuda = resolved == "cuda"

    loaded: dict[str, Any] = {}
    missing: list[str] = []
    for model_name in model_names:
        if model_name not in model_configs:
            raise ValueError(
                f"Modelo '{model_name}' no soportado. Disponibles: {sorted(model_configs)}"
            )

        model_cls, _ = model_configs[model_name]
        weights_path = models_dir / f"{model_name}_k{size_k}.pt"
        if not weights_path.exists():
            if strict:
                raise FileNotFoundError(f"No existe: {weights_path}")
            missing.append(model_name)
            continue

        if use_cuda:
            trainer_kwargs = build_lightning_trainer_kwargs(
                "gpu",
                use_early_stopping=False,
                precision="32-true",
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                logger=False,
            )
            map_location = "cuda"
        else:
            trainer_kwargs = build_lightning_trainer_kwargs(
                "cpu",
                use_early_stopping=False,
                precision="32-true",
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                logger=False,
            )
            map_location = "cpu"

        try:
            model = model_cls.load(
                str(weights_path),
                pl_trainer_kwargs=trainer_kwargs,
                map_location=map_location,
            )
        except TypeError:
            model = model_cls.load(str(weights_path))

        if not use_cuda and hasattr(model, "model") and hasattr(model.model, "float"):
            try:
                model.model.float()
            except Exception:
                pass

        _apply_inference_trainer_overrides(model)

        loaded[model_name] = Float32InputModelAdapter(model)

    if not loaded:
        raise RuntimeError(
            f"No se pudo cargar ningun modelo desde {models_dir}. Revisa size_k={size_k}."
        )
    if missing:
        print(f"[info] Modelos omitidos por no encontrar pesos: {sorted(missing)}")
    return loaded


def summarize_results_by_model(results_df: pd.DataFrame) -> pd.DataFrame:
    """Average benchmark metrics by model and sort by best overall score."""
    metric_cols = [m for m in ("MAE", "RMSE", "MASE") if m in results_df.columns]
    if not metric_cols:
        return pd.DataFrame(columns=["Modelo"])

    ranking_df = (
        results_df.groupby("Modelo", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values([c for c in ("MASE", "RMSE", "MAE") if c in metric_cols])
        .reset_index(drop=True)
    )
    return ranking_df


def _resolve_requested_models(
    model_names: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split requested model specs into explicit Darts and TSPulse model names."""
    if not model_names:
        raise ValueError("`model_names` no puede estar vacio")

    darts_model_names: list[str] = []
    seen_darts: set[str] = set()
    tspulse_model_names: list[str] = []
    seen_tspulse: set[str] = set()

    for spec in model_names:
        name = spec.strip()
        if not name:
            continue

        if name == TSPULSE_ORIGINAL_MODEL_NAME:
            if name not in seen_tspulse:
                tspulse_model_names.append(name)
                seen_tspulse.add(name)
            continue
        if name == TSPULSE_FINETUNED_MODEL_NAME:
            if name not in seen_tspulse:
                tspulse_model_names.append(name)
                seen_tspulse.add(name)
            continue

        if name not in seen_darts:
            darts_model_names.append(name)
            seen_darts.add(name)

    if not darts_model_names and not tspulse_model_names:
        raise ValueError("No se detectaron modelos validos en `model_names`")

    return darts_model_names, tspulse_model_names


def _merge_plot_stores(
    stores: Sequence[dict[int, dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Merge per-model plot payloads into one gap-indexed plot store."""
    def _normalize_series_payload(series_payload: Any) -> dict[str, Any]:
        payload = series_payload if isinstance(series_payload, dict) else {}
        preds = payload.get("preds", {})
        return {
            "actual": payload.get("actual"),
            "preds": dict(preds) if isinstance(preds, dict) else {},
            "naive_mase": payload.get("naive_mase"),
        }

    merged: dict[int, dict[str, Any]] = {}

    for store in stores:
        for gap_size, gap_payload in store.items():
            gap_entry = merged.setdefault(int(gap_size), {"series": {}})
            src_series = dict(gap_payload.get("series", {}))
            dst_series = gap_entry.setdefault("series", {})

            for series_name, series_payload in src_series.items():
                normalized_payload = _normalize_series_payload(series_payload)
                dst = dst_series.setdefault(series_name, normalized_payload)

                if dst.get("actual") is None and normalized_payload["actual"] is not None:
                    dst["actual"] = normalized_payload["actual"]
                if (
                    dst.get("naive_mase") is None
                    and normalized_payload["naive_mase"] is not None
                ):
                    dst["naive_mase"] = normalized_payload["naive_mase"]

                dst["preds"].update(normalized_payload["preds"])

            for key, value in gap_payload.items():
                if key == "series":
                    continue
                gap_entry.setdefault(key, value)

    return dict(sorted(merged.items(), key=lambda item: item[0]))


def _run_parallel_model_task(task: dict[str, Any]) -> tuple[str, pd.DataFrame, dict[int, dict[str, Any]]]:
    """Worker entry point that benchmarks one model in a separate process."""
    model_name = str(task["model_name"])
    resolved_root = _resolve_repo_root(task.get("repo_root"))
    config = BenchmarkRunConfig.from_mapping(task)

    dataset_bundle = _build_dataset_bundle_from_config(
        repo_root=resolved_root,
        config=config,
    )

    if model_name == TSPULSE_ORIGINAL_MODEL_NAME:
        model_dict: dict[str, Any] = {
            TSPULSE_ORIGINAL_MODEL_NAME: _build_tspulse_model(
                replace(config.tspulse, model_path=None),
                freq=config.freq,
            )
        }
    elif model_name == TSPULSE_FINETUNED_MODEL_NAME:
        if config.tspulse.model_path is None:
            raise RuntimeError(
                "No se puede evaluar TSPulse_FineTuned sin `tspulse_model_path`."
            )
        model_dict = {
            TSPULSE_FINETUNED_MODEL_NAME: _build_tspulse_model(
                config.tspulse,
                freq=config.freq,
            )
        }
    else:
        model_dict = load_darts_models_from_artifacts(
            repo_root=resolved_root,
            size_k=config.size_k,
            model_names=(model_name,),
            force_cpu=config.force_cpu,
            strict=False,
        )

    results_df, plot_store = _execute_benchmark_with_dataset(
        model_dict=model_dict,
        dataset_bundle=dataset_bundle,
        config=config,
    )

    return model_name, results_df, plot_store


def run_imputation_benchmark(
    repo_root: str | Path | None = None,
    size_k: int | None = None,
    model_names: Sequence[str] | None = None,
    force_cpu: bool = True,
    quiet_logs: bool = True,
    local_files_only: bool = False,
    hf_token: str | None = None,
    tspulse_model_id: str | None = None,
    tspulse_revision: str | None = None,
    tspulse_model_path: str | None = None,
    gap_sizes: Sequence[int] | None = None,
    num_gaps: int | None = None,
    gap_strategy: str | None = None,
    metrics: Sequence[str] | None = None,
    random_seed: int | None = None,
    seasonality_m: int | None = None,
    freq: str | None = None,
    val_size: int | None = None,
    val_context_len: int | None = None,
    min_train_len_base: int | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[str, Any]],
]:
    """Run the benchmark sequentially and return raw results, ranking, and plots."""
    configure_warnings(quiet=quiet_logs)
    resolved_root = _resolve_repo_root(repo_root)
    config = _load_benchmark_run_config(
        force_cpu=force_cpu,
        local_files_only=local_files_only,
        hf_token=hf_token,
        size_k=size_k,
        tspulse_model_id=tspulse_model_id,
        tspulse_revision=tspulse_revision,
        tspulse_model_path=tspulse_model_path,
        gap_sizes=gap_sizes,
        num_gaps=num_gaps,
        gap_strategy=gap_strategy,
        metrics=metrics,
        random_seed=random_seed,
        seasonality_m=seasonality_m,
        freq=freq,
        val_size=val_size,
        val_context_len=val_context_len,
        min_train_len_base=min_train_len_base,
    )
    model_names = _default_model_names() if model_names is None else model_names
    darts_model_names, tspulse_model_names = _resolve_requested_models(model_names)

    dataset_bundle = _build_dataset_bundle_from_config(
        repo_root=resolved_root,
        config=config,
    )
    model_dict: dict[str, Any] = {}
    if darts_model_names:
        model_dict.update(
            load_darts_models_from_artifacts(
                repo_root=resolved_root,
                size_k=config.size_k,
                model_names=darts_model_names,
                force_cpu=config.force_cpu,
                strict=False,
            )
        )

    if tspulse_model_names:
        if TSFM_PUBLIC_AVAILABLE:
            model_dict.update(
                _build_tspulse_model_dict(
                    config.tspulse,
                    freq=config.freq,
                    model_names=tspulse_model_names,
                )
            )
        else:
            print("[warn] tsfm_public no esta disponible; se omite TSPulse.")

    if not model_dict:
        raise RuntimeError(
            "No se pudo cargar ningun modelo para benchmark (Darts/TSPulse)."
        )

    results_df, plot_store = _execute_benchmark_with_dataset(
        model_dict=model_dict,
        dataset_bundle=dataset_bundle,
        config=config,
    )
    ranking_df = summarize_results_by_model(results_df)
    return results_df, ranking_df, plot_store


def run_imputation_benchmark_parallel(
    repo_root: str | Path | None = None,
    size_k: int | None = None,
    model_names: Sequence[str] | None = None,
    force_cpu: bool = True,
    quiet_logs: bool = True,
    local_files_only: bool = False,
    hf_token: str | None = None,
    tspulse_model_id: str | None = None,
    tspulse_revision: str | None = None,
    tspulse_model_path: str | None = None,
    gap_sizes: Sequence[int] | None = None,
    num_gaps: int | None = None,
    gap_strategy: str | None = None,
    metrics: Sequence[str] | None = None,
    random_seed: int | None = None,
    seasonality_m: int | None = None,
    freq: str | None = None,
    val_size: int | None = None,
    val_context_len: int | None = None,
    min_train_len_base: int | None = None,
    max_workers: int | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[str, Any]],
]:
    """Run the benchmark with one worker per model when possible."""
    configure_warnings(quiet=quiet_logs)
    resolved_root = _resolve_repo_root(repo_root)
    config = _load_benchmark_run_config(
        force_cpu=force_cpu,
        local_files_only=local_files_only,
        hf_token=hf_token,
        size_k=size_k,
        tspulse_model_id=tspulse_model_id,
        tspulse_revision=tspulse_revision,
        tspulse_model_path=tspulse_model_path,
        gap_sizes=gap_sizes,
        num_gaps=num_gaps,
        gap_strategy=gap_strategy,
        metrics=metrics,
        random_seed=random_seed,
        seasonality_m=seasonality_m,
        freq=freq,
        val_size=val_size,
        val_context_len=val_context_len,
        min_train_len_base=min_train_len_base,
    )
    model_names = _default_model_names() if model_names is None else model_names
    darts_model_names, tspulse_model_names = _resolve_requested_models(model_names)

    eval_model_names = list(darts_model_names)
    if tspulse_model_names:
        if TSFM_PUBLIC_AVAILABLE:
            if (
                TSPULSE_FINETUNED_MODEL_NAME in tspulse_model_names
                and config.tspulse.model_path is None
            ):
                raise RuntimeError(
                    "No se puede evaluar TSPulse_FineTuned sin `tspulse_model_path`."
                )
            eval_model_names.extend(tspulse_model_names)
        else:
            print("[warn] tsfm_public no esta disponible; se omite TSPulse.")

    if not eval_model_names:
        raise RuntimeError("No hay modelos para evaluar en paralelo.")

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(eval_model_names), max(1, cpu_count // 2))
    max_workers = max(1, min(int(max_workers), len(eval_model_names)))

    task_common = _build_parallel_task_common(repo_root=resolved_root, config=config)
    tasks = [
        {
            **task_common,
            "model_name": model_name,
        }
        for model_name in eval_model_names
    ]

    if max_workers == 1:
        outputs = [_run_parallel_model_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            outputs = list(executor.map(_run_parallel_model_task, tasks))

    results_df = pd.concat([item[1] for item in outputs], ignore_index=True)
    plot_store = _merge_plot_stores([item[2] for item in outputs])
    ranking_df = summarize_results_by_model(results_df)

    return results_df, ranking_df, plot_store


def _build_montecarlo_seed_list(
    seeds: Sequence[int] | None,
    n_runs: int,
    seed_start: int,
    seed_step: int,
) -> list[int]:
    """Build the list of random seeds used for Monte Carlo benchmarking."""
    if seeds is not None:
        seed_list = [int(seed) for seed in seeds]
        if not seed_list:
            raise ValueError("`seeds` no puede estar vacio si se proporciona.")
        return seed_list

    if int(n_runs) <= 0:
        raise ValueError("`n_runs` debe ser > 0 cuando `seeds` es None.")
    if int(seed_step) == 0:
        raise ValueError("`seed_step` no puede ser 0.")

    start = int(seed_start)
    step = int(seed_step)
    return [start + i * step for i in range(int(n_runs))]


def summarize_montecarlo_rankings(ranking_by_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-seed rankings with mean, spread, and quantiles by model."""
    metric_cols = [m for m in ("MAE", "RMSE", "MASE") if m in ranking_by_seed_df.columns]
    if ranking_by_seed_df.empty or not metric_cols:
        return pd.DataFrame(columns=["Modelo", "Runs"])

    rows: list[dict[str, Any]] = []
    for model_name, group_df in ranking_by_seed_df.groupby("Modelo", sort=False):
        row: dict[str, Any] = {
            "Modelo": str(model_name),
            "Runs": int(group_df["Seed"].nunique()) if "Seed" in group_df.columns else int(len(group_df)),
        }

        for metric in metric_cols:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            row[f"{metric}_Mean"] = float(values.mean()) if len(values) > 0 else float("nan")
            row[f"{metric}_Std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{metric}_P05"] = float(values.quantile(0.05)) if len(values) > 0 else float("nan")
            row[f"{metric}_P95"] = float(values.quantile(0.95)) if len(values) > 0 else float("nan")

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    sort_cols = [c for c in ("MASE_Mean", "RMSE_Mean", "MAE_Mean") if c in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)
    return summary_df


def run_imputation_benchmark_parallel_montecarlo(
    repo_root: str | Path | None = None,
    size_k: int | None = None,
    model_names: Sequence[str] | None = None,
    force_cpu: bool = True,
    quiet_logs: bool = True,
    local_files_only: bool = False,
    hf_token: str | None = None,
    tspulse_model_id: str | None = None,
    tspulse_revision: str | None = None,
    tspulse_model_path: str | None = None,
    gap_sizes: Sequence[int] | None = None,
    num_gaps: int | None = None,
    gap_strategy: str | None = None,
    metrics: Sequence[str] | None = None,
    seasonality_m: int | None = None,
    freq: str | None = None,
    val_size: int | None = None,
    val_context_len: int | None = None,
    min_train_len_base: int | None = None,
    max_workers: int | None = None,
    seeds: Sequence[int] | None = None,
    n_runs: int = 20,
    seed_start: int = 42,
    seed_step: int = 1,
    progress: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Ejecuta benchmark de imputacion en varias semillas (Monte Carlo).

    Devuelve:
    - `results_mc_df`: resultados fila-a-fila con columnas extra `Seed` y `MonteCarlo_Run`.
    - `summary_mc_df`: resumen por modelo sobre rankings por semilla.
    - `ranking_by_seed_df`: ranking por modelo en cada corrida/semilla (derivado de `results_mc_df`).
    """
    seed_list = _build_montecarlo_seed_list(
        seeds=seeds,
        n_runs=n_runs,
        seed_start=seed_start,
        seed_step=seed_step,
    )

    results_runs: list[pd.DataFrame] = []
    total_runs = len(seed_list)
    for run_idx, seed in enumerate(seed_list, start=1):
        if progress:
            print(f"[MonteCarlo] {run_idx}/{total_runs} con seed={seed}")

        run_results, _, _ = (
            run_imputation_benchmark_parallel(
                repo_root=repo_root,
                size_k=size_k,
                model_names=model_names,
                force_cpu=force_cpu,
                quiet_logs=quiet_logs,
                local_files_only=local_files_only,
                hf_token=hf_token,
                tspulse_model_id=tspulse_model_id,
                tspulse_revision=tspulse_revision,
                tspulse_model_path=tspulse_model_path,
                gap_sizes=gap_sizes,
                num_gaps=num_gaps,
                gap_strategy=gap_strategy,
                metrics=metrics,
                random_seed=int(seed),
                seasonality_m=seasonality_m,
                freq=freq,
                val_size=val_size,
                val_context_len=val_context_len,
                min_train_len_base=min_train_len_base,
                max_workers=max_workers,
            )
        )

        run_results = run_results.copy()
        run_results["Seed"] = int(seed)
        run_results["MonteCarlo_Run"] = int(run_idx)
        results_runs.append(run_results)

    results_mc_df = pd.concat(results_runs, ignore_index=True)
    metric_cols = [m for m in ("MAE", "RMSE", "MASE") if m in results_mc_df.columns]
    if metric_cols:
        ranking_by_seed_df = (
            results_mc_df.groupby(["Seed", "MonteCarlo_Run", "Modelo"], as_index=False)[metric_cols]
            .mean(numeric_only=True)
            .sort_values(["Seed", "MonteCarlo_Run", "Modelo"])
            .reset_index(drop=True)
        )
    else:
        ranking_by_seed_df = pd.DataFrame(columns=["Seed", "MonteCarlo_Run", "Modelo"])

    summary_mc_df = summarize_montecarlo_rankings(ranking_by_seed_df)

    return (
        results_mc_df,
        summary_mc_df,
        ranking_by_seed_df,
    )
