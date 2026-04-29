from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from imputation_benchmark_pipeline import (
    TSFM_PUBLIC_AVAILABLE,
    TSPulseHistoricalImputer,
    execute_complete_pipeline,
)
from training_pipeline import build_train_val_test_series
from training_pipeline_config import build_model_configs
from utils import get_longest_segment, load_dataset_paths, load_to_df


DEFAULT_MODEL_NAMES = (
    "TiDE",
    "NHiTS",
    "TCN",
    "TSMixer",
    "RNN",
    "NLinear",
    "DLinear",
)


def configure_quiet_runtime(quiet: bool = True) -> None:
    if not quiet:
        return

    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in (
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "transformers",
        "transformers.pipelines",
        "transformers.pipelines.base",
        "darts",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def _to_float32_timeseries(series: Any) -> Any:
    if hasattr(series, "astype"):
        try:
            return series.astype("float32")
        except Exception:
            return series
    return series


class Float32InputModelAdapter:
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
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "src").exists():
        return cwd
    if cwd.name == "src" and (cwd.parent / "src").exists():
        return cwd.parent
    return cwd


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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


def load_project_series(
    *,
    repo_root: Path,
    key_word: str = "NO2",
    file_extension: str = "csv",
    freq: str = "h",
) -> list[pd.DataFrame]:
    base_path = os.path.join(str(repo_root), "Datos-post-COUTA", "*/")
    file_paths = sorted(
        load_dataset_paths(
            base_path=base_path,
            key_word=key_word,
            file_extension=file_extension,
        )
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No se encontraron archivos en '{base_path}' con keyword='{key_word}' y extension='{file_extension}'."
        )

    series_dfs: list[pd.DataFrame] = []
    for path in file_paths:
        df = load_to_df(path)
        if df is None or df.empty or len(df.columns) != 1:
            continue

        col = str(df.columns[0])
        series = df.iloc[:, 0].astype(float).sort_index()
        series = series[~series.index.duplicated(keep="last")].asfreq(freq)
        series_dfs.append(series.to_frame(name=col))

    if not series_dfs:
        raise RuntimeError(
            "No se pudieron construir series validas desde los archivos cargados."
        )
    return series_dfs


def build_dataset_bundle_for_imputation(
    *,
    repo_root: Path,
    size_k: int,
    val_size: int = 48,
    val_context_len: int = 72,
    min_train_len_base: int = 72,
    key_word: str = "NO2",
    file_extension: str = "csv",
    freq: str = "h",
    force_end: bool = False,
) -> dict[str, Any]:
    series_dfs = load_project_series(
        repo_root=repo_root,
        key_word=key_word,
        file_extension=file_extension,
        freq=freq,
    )
    longest_segment = get_longest_segment(series_dfs, force_end=force_end, verbose=False)
    if longest_segment.empty:
        raise RuntimeError("get_longest_segment devolvio un DataFrame vacio.")

    return build_train_val_test_series(
        series_dfs=series_dfs,
        longest_segment=longest_segment,
        val_size=int(val_size),
        min_train_len=int(min_train_len_base + size_k),
        val_context_len=int(val_context_len),
    )


def load_darts_models_from_artifacts(
    *,
    repo_root: Path,
    size_k: int,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    force_cpu: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    models_dir = repo_root / "src" / "artifacts" / "models"
    model_configs = build_model_configs()
    use_cuda = _cuda_available() and not bool(force_cpu)

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
            trainer_kwargs = {
                "accelerator": "gpu",
                "devices": 1,
                "precision": "32-true",
                "enable_progress_bar": False,
                "enable_checkpointing": False,
                "enable_model_summary": False,
                "logger": False,
            }
            map_location = "cuda"
        else:
            trainer_kwargs = {
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "enable_progress_bar": False,
                "enable_checkpointing": False,
                "enable_model_summary": False,
                "logger": False,
            }
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


def _is_tspulse_model_spec(spec: Any) -> bool:
    if isinstance(spec, str):
        normalized = spec.strip().lower()
        return normalized in {"tspulse", "tspulsehistoricalimputer"}

    if isinstance(spec, type):
        try:
            return issubclass(spec, TSPulseHistoricalImputer)
        except TypeError:
            return False

    return False


def _resolve_requested_models(
    model_names: Sequence[str | type[Any]],
) -> tuple[list[str], bool]:
    if not model_names:
        raise ValueError("`model_names` no puede estar vacio")

    darts_model_names: list[str] = []
    seen_darts: set[str] = set()
    request_tspulse = False

    for spec in model_names:
        if _is_tspulse_model_spec(spec):
            request_tspulse = True
            continue

        if isinstance(spec, str):
            name = spec.strip()
            if not name:
                continue
            if name not in seen_darts:
                darts_model_names.append(name)
                seen_darts.add(name)
            continue

        if isinstance(spec, type):
            raise TypeError(
                "Las clases en `model_names` solo pueden ser TSPulseHistoricalImputer; "
                "usa strings para modelos Darts."
            )

        raise TypeError(
            "Cada elemento de `model_names` debe ser string o clase; "
            f"recibido: {type(spec)}"
        )

    if not darts_model_names and not request_tspulse:
        raise ValueError("No se detectaron modelos validos en `model_names`")

    return darts_model_names, request_tspulse


def _merge_plot_stores(
    stores: Sequence[dict[int, dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}

    for store in stores:
        for gap_size, gap_payload in store.items():
            gap_entry = merged.setdefault(int(gap_size), {"series": {}})
            src_series = dict(gap_payload.get("series", {}))
            dst_series = gap_entry.setdefault("series", {})

            for series_name, series_payload in src_series.items():
                dst = dst_series.setdefault(
                    series_name,
                    {
                        "actual": series_payload.get("actual"),
                        "preds": {},
                        "naive_mase": series_payload.get("naive_mase"),
                    },
                )

                if dst.get("actual") is None and "actual" in series_payload:
                    dst["actual"] = series_payload.get("actual")
                if dst.get("naive_mase") is None and "naive_mase" in series_payload:
                    dst["naive_mase"] = series_payload.get("naive_mase")

                preds = series_payload.get("preds", {})
                if isinstance(preds, dict):
                    dst["preds"].update(preds)

            for key, value in gap_payload.items():
                if key == "series":
                    continue
                gap_entry.setdefault(key, value)

    return dict(sorted(merged.items(), key=lambda item: item[0]))


def _run_parallel_model_task(task: dict[str, Any]) -> tuple[str, pd.DataFrame, dict[int, dict[str, Any]]]:
    model_name = str(task["model_name"])
    resolved_root = _resolve_repo_root(task.get("repo_root"))

    dataset_bundle = build_dataset_bundle_for_imputation(
        repo_root=resolved_root,
        size_k=int(task["size_k"]),
        val_size=int(task["val_size"]),
        val_context_len=int(task["val_context_len"]),
        min_train_len_base=int(task["min_train_len_base"]),
        key_word=str(task["key_word"]),
        file_extension=str(task["file_extension"]),
        freq=str(task["freq"]),
        force_end=bool(task["force_end"]),
    )

    if model_name == "TSPulse":
        model_dict: dict[str, Any] = {
            "TSPulse": TSPulseHistoricalImputer(
                freq=str(task["freq"]),
                local_files_only=bool(task["local_files_only"]),
                hf_token=task.get("hf_token"),
            )
        }
    else:
        model_dict = load_darts_models_from_artifacts(
            repo_root=resolved_root,
            size_k=int(task["size_k"]),
            model_names=(model_name,),
            force_cpu=bool(task["force_cpu"]),
            strict=False,
        )

    results_df, _, plot_store = execute_complete_pipeline(
        model_dict=model_dict,
        dataset_bundle=dataset_bundle,
        gap_sizes=tuple(int(x) for x in task["gap_sizes"]),
        num_gaps=int(task["num_gaps"]),
        metrics=tuple(task["metrics"]),
        random_seed=int(task["random_seed"]),
        freq=str(task["freq"]),
        seasonality_m=int(task["seasonality_m"]),
        gap_strategy=str(task["gap_strategy"]),
    )

    return model_name, results_df, plot_store


def run_imputation_benchmark(
    *,
    repo_root: str | Path | None = None,
    size_k: int = 5,
    model_names: Sequence[str | type[Any]] = DEFAULT_MODEL_NAMES,
    force_cpu: bool = True,
    quiet_logs: bool = True,
    local_files_only: bool = False,
    hf_token: str | None = None,
    gap_sizes: Sequence[int] = (1, 2, 5, 10),
    num_gaps: int = 3,
    gap_strategy: str = "hybrid_tspulse",
    metrics: Sequence[str] = ("mae", "rmse", "mase"),
    random_seed: int = 42,
    seasonality_m: int = 24,
    freq: str = "h",
    key_word: str = "NO2",
    file_extension: str = "csv",
    force_end: bool = False,
    val_size: int = 48,
    val_context_len: int = 72,
    min_train_len_base: int = 72,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[str, Any]],
]:
    configure_quiet_runtime(quiet=quiet_logs)
    resolved_root = _resolve_repo_root(repo_root)
    darts_model_names, request_tspulse = _resolve_requested_models(model_names)

    dataset_bundle = build_dataset_bundle_for_imputation(
        repo_root=resolved_root,
        size_k=size_k,
        val_size=val_size,
        val_context_len=val_context_len,
        min_train_len_base=min_train_len_base,
        key_word=key_word,
        file_extension=file_extension,
        freq=freq,
        force_end=force_end,
    )
    model_dict: dict[str, Any] = {}
    if darts_model_names:
        model_dict.update(
            load_darts_models_from_artifacts(
                repo_root=resolved_root,
                size_k=size_k,
                model_names=darts_model_names,
                force_cpu=force_cpu,
                strict=False,
            )
        )

    if request_tspulse:
        if TSFM_PUBLIC_AVAILABLE:
            model_dict["TSPulse"] = TSPulseHistoricalImputer(
                freq=freq,
                local_files_only=local_files_only,
                hf_token=hf_token,
            )
        else:
            print("[warn] tsfm_public no esta disponible; se omite TSPulse.")

    if not model_dict:
        raise RuntimeError(
            "No se pudo cargar ningun modelo para benchmark (Darts/TSPulse)."
        )

    results_df, _, plot_store = execute_complete_pipeline(
        model_dict=model_dict,
        dataset_bundle=dataset_bundle,
        gap_sizes=tuple(int(x) for x in gap_sizes),
        num_gaps=int(num_gaps),
        metrics=tuple(metrics),
        random_seed=int(random_seed),
        freq=freq,
        seasonality_m=int(seasonality_m),
        gap_strategy=gap_strategy,
    )
    ranking_df = summarize_results_by_model(results_df)
    return results_df, ranking_df, plot_store


def run_imputation_benchmark_parallel(
    *,
    repo_root: str | Path | None = None,
    size_k: int = 5,
    model_names: Sequence[str | type[Any]] = DEFAULT_MODEL_NAMES,
    force_cpu: bool = True,
    quiet_logs: bool = True,
    local_files_only: bool = False,
    hf_token: str | None = None,
    gap_sizes: Sequence[int] = (1, 2, 5, 10),
    num_gaps: int = 3,
    gap_strategy: str = "hybrid_tspulse",
    metrics: Sequence[str] = ("mae", "rmse", "mase"),
    random_seed: int = 42,
    seasonality_m: int = 24,
    freq: str = "h",
    key_word: str = "NO2",
    file_extension: str = "csv",
    force_end: bool = False,
    val_size: int = 48,
    val_context_len: int = 72,
    min_train_len_base: int = 72,
    max_workers: int | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[str, Any]],
]:
    configure_quiet_runtime(quiet=quiet_logs)
    resolved_root = _resolve_repo_root(repo_root)
    darts_model_names, request_tspulse = _resolve_requested_models(model_names)

    eval_model_names = list(darts_model_names)
    if request_tspulse:
        if TSFM_PUBLIC_AVAILABLE:
            eval_model_names.append("TSPulse")
        else:
            print("[warn] tsfm_public no esta disponible; se omite TSPulse.")

    if not eval_model_names:
        raise RuntimeError("No hay modelos para evaluar en paralelo.")

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(eval_model_names), max(1, cpu_count // 2))
    max_workers = max(1, min(int(max_workers), len(eval_model_names)))

    task_common = {
        "repo_root": str(resolved_root),
        "size_k": int(size_k),
        "force_cpu": bool(force_cpu),
        "local_files_only": bool(local_files_only),
        "hf_token": hf_token,
        "gap_sizes": tuple(int(x) for x in gap_sizes),
        "num_gaps": int(num_gaps),
        "gap_strategy": str(gap_strategy),
        "metrics": tuple(metrics),
        "random_seed": int(random_seed),
        "seasonality_m": int(seasonality_m),
        "freq": str(freq),
        "key_word": str(key_word),
        "file_extension": str(file_extension),
        "force_end": bool(force_end),
        "val_size": int(val_size),
        "val_context_len": int(val_context_len),
        "min_train_len_base": int(min_train_len_base),
    }
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
    *,
    seeds: Sequence[int] | None,
    n_runs: int,
    seed_start: int,
    seed_step: int,
) -> list[int]:
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
    *,
    repo_root: str | Path | None = None,
    size_k: int = 5,
    model_names: Sequence[str | type[Any]] = DEFAULT_MODEL_NAMES,
    force_cpu: bool = True,
    quiet_logs: bool = True,
    local_files_only: bool = False,
    hf_token: str | None = None,
    gap_sizes: Sequence[int] = (1, 2, 5, 10),
    num_gaps: int = 3,
    gap_strategy: str = "hybrid_tspulse",
    metrics: Sequence[str] = ("mae", "rmse", "mase"),
    seasonality_m: int = 24,
    freq: str = "h",
    key_word: str = "NO2",
    file_extension: str = "csv",
    force_end: bool = False,
    val_size: int = 48,
    val_context_len: int = 72,
    min_train_len_base: int = 72,
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
                gap_sizes=gap_sizes,
                num_gaps=num_gaps,
                gap_strategy=gap_strategy,
                metrics=metrics,
                random_seed=int(seed),
                seasonality_m=seasonality_m,
                freq=freq,
                key_word=key_word,
                file_extension=file_extension,
                force_end=force_end,
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