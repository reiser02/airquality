"""Anomaly-detection benchmark over air-quality series (genias-style, our data).

Stages:

1. Load raw 5-minute station data and preprocess to hourly means
   (:func:`airquality.data.loaders.load_raw_5m` +
   :func:`airquality.data.preprocessing.preprocess`).
2. Inject the single :data:`INJECTION_VARIANT` (``STL-combined``: STL-based
   synthetic base + a per-segment mix of the genias STL anomaly shapes) TWICE per
   station with two independent seeds — a *selection* injection (``seed``) and a
   held-out *evaluation* injection (``eval_seed``) — via
   :func:`.anomalies.inject_synthetic_anomalies`.
3. Run every registered detector on both injections; build the ensemble by
   ranking/weighting detectors on the *selection* injection and fusing + scoring
   their *evaluation* scores (top-k + weighted consensus). Ranking on one
   injection and scoring on another keeps the ensemble VUS-PR unbiased.
4. Compute the genias metric set (auroc/aupr/vus_pr/vus_roc/affiliation_f1) for
   every detector and the ensemble.
5. Persist ``results.json`` (+ ``scores.npz``); plots are rendered by the
   separate ``plot_benchmark_results`` script.

A "case" is one station evaluated with the ``STL-combined`` injection.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import inspect
import json
import logging
from multiprocessing import get_context
from pathlib import Path
import pickle
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

from airquality.data.loaders import load_raw_5m
from airquality.data.preprocessing import preprocess
from airquality.data.segments import contiguous_observed_segments

from .anomalies import inject_synthetic_anomalies
from .ensemble import DEFAULT_ENSEMBLE_METHOD, DEFAULT_TOP_K, consensus, rank_top_k
from .metrics import compute_metrics
from .registry import MODEL_REGISTRY, resolve_model_class, resolve_model_names

ENSEMBLE_NAME = "Ensemble"
METRIC_KEYS = ["auroc", "aupr", "vus_pr", "vus_roc", "affiliation_f1"]
DEFAULT_ENSEMBLE_WINDOW = 100

# The benchmark injects a single variant: an STL-based synthetic look-alike with a
# per-segment mix of the genias STL anomaly shapes (see :mod:`.anomalies`).
INJECTION_VARIANT = "STL-combined"

# Detectors that benefit from a GPU (windowed deep models). Everything else is
# CPU-only. Mirrors genias's ``GPU_MODEL_NAMES``.
GPU_MODEL_NAMES = {"COUTABase", "COUTAGenIAS", "CARLABase", "CARLAGenIAS", "LSTMAD", "TSPulse"}

# Set once per GPU worker process (genias-style device binding via a shared queue).
_worker_device: str | None = None


def normalize_device_request(device: str | None) -> str:
    """Validate/normalize a device request to ``cpu`` / ``cuda`` / ``multi-gpu``."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    normalized = str(device).strip().lower()
    if normalized in ("cpu", "cuda", "multi-gpu"):
        return normalized
    raise ValueError(f"Unsupported device '{device}'. Use cpu, cuda, or multi-gpu")


def available_cuda_devices() -> list[str]:
    """Return ``['cuda:0', ...]`` for every visible GPU (empty if no CUDA)."""
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{index}" for index in range(torch.cuda.device_count())]


def resolve_benchmark_devices(model_names: list[str], device_request: str | None) -> dict[str, str]:
    """Assign one device to each model (genias ``cpu``/``cuda``/``multi-gpu`` policy).

    CPU-only models always get ``cpu``; deep models get ``cuda:0`` under ``cuda`` or
    are round-robined across GPUs under ``multi-gpu``. Falls back to all-``cpu`` when
    no CUDA device is present.
    """
    normalized = normalize_device_request(device_request)
    if normalized == "cpu":
        return {name: "cpu" for name in model_names}

    cuda_devices = available_cuda_devices()
    if not cuda_devices:
        return {name: "cpu" for name in model_names}

    assignments = {name: "cpu" for name in model_names if name not in GPU_MODEL_NAMES}
    gpu_models = [name for name in model_names if name in GPU_MODEL_NAMES]
    if normalized == "cuda":
        for name in gpu_models:
            assignments[name] = cuda_devices[0]
    else:  # multi-gpu: spread the deep models across the available GPUs.
        for index, name in enumerate(gpu_models):
            assignments[name] = cuda_devices[index % len(cuda_devices)]
    return assignments


def bind_worker_device(device_queue) -> None:
    """Pool initializer: claim one device from the shared queue for this worker."""
    global _worker_device
    _worker_device = device_queue.get()


def current_worker_device() -> str | None:
    """Return the device bound to this worker process (``None`` outside a pool)."""
    return _worker_device


def synchronize_device(device: str) -> None:
    """Block until pending CUDA work on ``device`` finishes (accurate timing)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass
class AnomalyCase:
    """One station with two INDEPENDENT ``STL-combined`` injections.

    The ``*_select`` injection is used only to rank + weight detectors for the
    ensemble; ``values``/``labels`` are the held-out *evaluation* injection that
    every reported metric is scored on. Keeping them separate stops the ensemble
    from selecting and evaluating on the same labels (which inflates its VUS-PR).
    """

    name: str
    variant: str
    values: np.ndarray          # evaluation injection
    labels: np.ndarray          # evaluation labels
    values_select: np.ndarray   # selection injection (ranking/weighting only)
    labels_select: np.ndarray   # selection labels


@dataclass
class AnomalyBenchmarkConfig:
    """Runtime settings for one benchmark run (data, models, ensemble, seeds)."""

    pollutant: str = "NO2"
    raw_base_dir: str = "data/raw/datos_estaciones_5m"
    models: list[str] | None = None
    ensemble_method: str = DEFAULT_ENSEMBLE_METHOD
    ensemble_top_k: int = DEFAULT_TOP_K
    device: str = "cpu"
    seed: int = 13          # selection-injection seed
    eval_seed: int = 101    # held-out evaluation-injection seed (must differ from seed)
    min_series_points: int = 600
    series_limit: int | None = None
    output_dir: str | None = None


def _filter_model_kwargs(model_cls: type, kwargs: dict[str, object]) -> dict[str, object]:
    """Keep only kwargs the detector's ``__init__`` accepts (e.g. drop ``device``)."""
    parameters = inspect.signature(model_cls.__init__).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    valid = set(parameters) - {"self"}
    return {key: value for key, value in kwargs.items() if key in valid}


def build_cases(config: AnomalyBenchmarkConfig) -> list[AnomalyCase]:
    """Load + preprocess every station, then inject the ``STL-combined`` variant."""
    stations = load_raw_5m(config.pollutant, config.raw_base_dir)
    if config.series_limit is not None:
        stations = stations[: config.series_limit]

    logging.info("Building cases for %d station(s) with %s…", len(stations), INJECTION_VARIANT)
    cases: list[AnomalyCase] = []
    for station, frame in stations:
        processed, _ = preprocess([frame], config.pollutant)
        hourly = processed[0]
        # Longest contiguous run instead of `dropna()`: gluing non-contiguous
        # stretches would break the daily phase that STL(period=24) assumes.
        segments = contiguous_observed_segments(hourly.iloc[:, 0])
        values = (
            max(segments, key=len).to_numpy(dtype=np.float32)
            if segments
            else np.empty(0, dtype=np.float32)
        )
        if values.shape[0] < config.min_series_points:
            logging.info(
                "  skip %s  (longest contiguous run %d points < %d)",
                station, values.shape[0], config.min_series_points,
            )
            continue
        sel_values, sel_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, config.seed)
        eval_values, eval_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, config.eval_seed)
        cases.append(
            AnomalyCase(
                name=station,
                variant=INJECTION_VARIANT,
                values=eval_values,
                labels=eval_labels,
                values_select=sel_values,
                labels_select=sel_labels,
            )
        )
    logging.info("Built %d evaluation cases.", len(cases))
    return cases


def _run_detector(
    model_name: str,
    config: AnomalyBenchmarkConfig,
    cases: list[AnomalyCase],
    device: str = "cpu",
) -> dict[str, object]:
    """Fit/score one detector over every case on ``device``; log per-case progress."""
    model_cls = resolve_model_class(model_name)
    model_kwargs = _filter_model_kwargs(model_cls, {"device": device})

    per_case = []
    total = len(cases)
    model_started = time.perf_counter()
    logging.info("    [%s] START  %d cases on %s", model_name, total, device)
    for case_index, case in enumerate(cases):
        # Selection injection: its VUS-PR is only a ranking/weighting signal for
        # the ensemble; it is never the number we report or evaluate on.
        select_model = model_cls(seed=config.seed, **model_kwargs)
        select_model.fit(case.values_select)
        select_scores = select_model.score(case.values_select)
        select_window = int(getattr(select_model, "window_size", DEFAULT_ENSEMBLE_WINDOW) or DEFAULT_ENSEMBLE_WINDOW)
        vus_pr_select = compute_metrics(case.labels_select, np.asarray(select_scores), select_window)["vus_pr"]

        # Held-out evaluation injection: every reported metric/score comes from here.
        model = model_cls(seed=config.seed, **model_kwargs)
        synchronize_device(device)
        fit_started = time.perf_counter()
        model.fit(case.values)
        synchronize_device(device)
        fit_seconds = time.perf_counter() - fit_started

        inference_started = time.perf_counter()
        scores = model.score(case.values)
        synchronize_device(device)
        inference_seconds = time.perf_counter() - inference_started

        window_size = int(getattr(model, "window_size", DEFAULT_ENSEMBLE_WINDOW) or DEFAULT_ENSEMBLE_WINDOW)
        metrics = compute_metrics(case.labels, np.asarray(scores), window_size)
        per_case.append(
            {
                "series_name": case.name,
                "variant": case.variant,
                "series_length": int(case.values.shape[0]),
                "metrics": metrics,
                "vus_pr_select": float(vus_pr_select),
                "timing": {"fit_seconds": float(fit_seconds), "inference_seconds": float(inference_seconds)},
                "training_summary": getattr(model, "training_summary_", {}),
                "scores": np.asarray(scores, dtype=np.float32),
            }
        )
        logging.info(
            "    [%s] case %d/%d  %s  vus_sel=%.3f vus_eval=%.3f  (%.1fs)",
            model_name,
            case_index + 1,
            total,
            case.name,
            vus_pr_select,
            metrics["vus_pr"],
            fit_seconds + inference_seconds,
        )
    macro_vus_pr = float(np.mean([entry["metrics"]["vus_pr"] for entry in per_case])) if per_case else float("nan")
    logging.info(
        "    [%s] DONE   %d cases in %.1fs  macro_vus_pr=%.3f",
        model_name,
        total,
        time.perf_counter() - model_started,
        macro_vus_pr,
    )
    return {"per_case": per_case}


def _save_cases(cases: list[AnomalyCase], path: Path) -> None:
    """Persist built cases so parallel workers load them once instead of rebuilding."""
    with path.open("wb") as handle:
        pickle.dump(cases, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cases(path: str | Path) -> list[AnomalyCase]:
    """Load the pickled evaluation cases written by :func:`_save_cases`."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _run_detector_worker(model_name: str, cases_path: str, config: AnomalyBenchmarkConfig, device: str) -> tuple[str, dict[str, object]]:
    """CPU-pool worker: load cases from disk and run one detector."""
    return model_name, _run_detector(model_name, config, _load_cases(cases_path), device)


def _run_detector_worker_bound(model_name: str, cases_path: str, config: AnomalyBenchmarkConfig) -> tuple[str, dict[str, object]]:
    """GPU-pool worker: run one detector on the device bound to this worker."""
    device = current_worker_device()
    if device is None:
        raise RuntimeError("GPU worker device was not initialized")
    return model_name, _run_detector(model_name, config, _load_cases(cases_path), device)


def _run_detectors(
    model_names: list[str],
    config: AnomalyBenchmarkConfig,
    cases: list[AnomalyCase],
    cases_path: Path,
    device_assignments: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Run every detector, fanning out genias-style when a GPU is in play.

    With no GPU work (``cpu`` request or no CUDA) everything runs inline/sequentially.
    Otherwise deep models run in a device-bound GPU pool (one worker per GPU) while
    the CPU-only models run concurrently in a single CPU-pool worker.
    """
    gpu_models = [name for name in model_names if device_assignments[name].startswith("cuda")]
    cpu_models = [name for name in model_names if name not in gpu_models]

    if not gpu_models:
        results: dict[str, dict[str, object]] = {}
        for index, name in enumerate(model_names):
            logging.info("  ▶ [%d/%d] %s (%s)", index + 1, len(model_names), name, device_assignments[name])
            results[name] = _run_detector(name, config, cases, device_assignments[name])
        return results

    spawn = get_context("spawn")
    results = {}
    executors: list[ProcessPoolExecutor] = []
    futures = {}
    manager = None
    cuda_devices = sorted({device_assignments[name] for name in gpu_models})
    logging.info(
        "  Parallel fan-out: %d GPU model(s) over %s + %d CPU model(s) in a CPU worker",
        len(gpu_models),
        cuda_devices,
        len(cpu_models),
    )
    try:
        manager = spawn.Manager()
        device_queue = manager.Queue()
        for device in cuda_devices:
            device_queue.put(device)
        gpu_executor = ProcessPoolExecutor(
            max_workers=len(cuda_devices),
            mp_context=spawn,
            initializer=bind_worker_device,
            initargs=(device_queue,),
        )
        executors.append(gpu_executor)
        for name in gpu_models:
            futures[gpu_executor.submit(_run_detector_worker_bound, name, str(cases_path), config)] = name

        if cpu_models:
            cpu_executor = ProcessPoolExecutor(max_workers=1, mp_context=spawn)
            executors.append(cpu_executor)
            for name in cpu_models:
                futures[cpu_executor.submit(_run_detector_worker, name, str(cases_path), config, "cpu")] = name

        completed = 0
        for future in as_completed(futures):
            name, summary = future.result()
            results[name] = summary
            completed += 1
            logging.info("  ✓ [%d/%d] %s finished", completed, len(futures), name)
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
        if manager is not None:
            manager.shutdown()
    return results


def _build_ensemble(
    config: AnomalyBenchmarkConfig,
    cases: list[AnomalyCase],
    detector_results: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Per case: rank/weight by the SELECTION injection, fuse the EVAL scores, score on EVAL labels."""
    ensemble_results = []
    for index, case in enumerate(cases):
        # Ranking + weights come from the selection injection only (held-out eval labels
        # are never used to pick or weight detectors) -> unbiased ensemble metric.
        select_by_model = {name: result["per_case"][index]["vus_pr_select"] for name, result in detector_results.items()}
        top_models = rank_top_k(select_by_model, config.ensemble_top_k)
        score_arrays = [detector_results[name]["per_case"][index]["scores"] for name in top_models]
        weights = [select_by_model[name] for name in top_models]
        fused = consensus(score_arrays, config.ensemble_method, config.seed, weights=weights)
        metrics = compute_metrics(case.labels, fused, DEFAULT_ENSEMBLE_WINDOW)

        timings = [detector_results[name]["per_case"][index]["timing"] for name in top_models]
        fit_seconds = float(sum(timing["fit_seconds"] for timing in timings))
        inference_seconds = float(sum(timing["inference_seconds"] for timing in timings))
        ensemble_results.append(
            {
                "series_name": case.name,
                "variant": case.variant,
                "series_length": int(case.values.shape[0]),
                "metrics": metrics,
                "timing": {"fit_seconds": fit_seconds, "inference_seconds": inference_seconds},
                "training_summary": {"selected_models": top_models, "method": config.ensemble_method},
            }
        )
    return ensemble_results


def _summarize(series_results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-case entries into macro metrics + timing (drops ``scores``)."""
    clean = [{key: entry[key] for key in entry if key != "scores"} for entry in series_results]
    macro_metrics = {
        metric: float(np.mean([entry["metrics"][metric] for entry in clean])) for metric in METRIC_KEYS
    }
    timing = {
        "mean_fit_seconds": float(np.mean([entry["timing"]["fit_seconds"] for entry in clean])),
        "mean_inference_seconds": float(np.mean([entry["timing"]["inference_seconds"] for entry in clean])),
    }
    return {"series_results": clean, "macro_metrics": macro_metrics, "timing": timing}


def _resolve_output_dir(config: AnomalyBenchmarkConfig) -> Path:
    """Create and return the run's output directory (timestamped by default)."""
    if config.output_dir is not None:
        output_dir = Path(config.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / "anomaly" / f"{config.pollutant}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_benchmark(config: AnomalyBenchmarkConfig | None = None) -> dict[str, object]:
    """Run the full benchmark and persist ``results.json`` + ``scores.npz``.

    Plots are intentionally *not* rendered here; use the separate
    ``airquality.anomaly.plot_benchmark_results`` script on the produced
    ``results.json`` to generate them.
    """
    config = config or AnomalyBenchmarkConfig()
    model_names = resolve_model_names(config.models)
    cases = build_cases(config)
    if not cases:
        raise RuntimeError(
            f"No evaluation cases built for pollutant '{config.pollutant}'. Check raw data under "
            f"'{config.raw_base_dir}' and min_series_points={config.min_series_points}."
        )

    output_dir = _resolve_output_dir(config)
    device_assignments = resolve_benchmark_devices(model_names, config.device)
    logging.info(
        "Running %d detector(s) on %d case(s)  [device request=%s]",
        len(model_names),
        len(cases),
        normalize_device_request(config.device),
    )
    logging.info("  Device plan: %s", {name: device_assignments[name] for name in model_names})

    # Persist cases once so parallel workers load them instead of rebuilding (STL is costly).
    cases_path = output_dir / "_cases.pkl"
    _save_cases(cases, cases_path)
    try:
        detector_results = _run_detectors(model_names, config, cases, cases_path, device_assignments)
    finally:
        cases_path.unlink(missing_ok=True)

    logging.info("Building ensemble…")
    ensemble_results = _build_ensemble(config, cases, detector_results)

    model_summaries: dict[str, dict[str, object]] = {
        name: _summarize(result["per_case"]) for name, result in detector_results.items()
    }
    model_summaries[ENSEMBLE_NAME] = _summarize(ensemble_results)
    macro_vus_pr = model_summaries[ENSEMBLE_NAME]["macro_metrics"]["vus_pr"]
    logging.info("  ✓ Ensemble  VUS-PR=%.3f", macro_vus_pr)
    ordered_names = model_names + [ENSEMBLE_NAME]

    # Save raw per-case scores + labels so ensemble can be recomputed without retraining.
    scores_dict = {}
    for name, result in detector_results.items():
        for i, case_result in enumerate(result["per_case"]):
            scores_dict[f"{name}__case{i}"] = case_result["scores"]
    for i, case in enumerate(cases):
        scores_dict[f"__labels__case{i}"] = case.labels
    np.savez_compressed(output_dir / "scores.npz", **scores_dict)

    summary = {
        "config": asdict(config),
        "models": model_summaries,
        "model_names": ordered_names,
        "series_names": sorted({case.name for case in cases}),
        "variants": [INJECTION_VARIANT],
        "metrics_plot": "vus_pr_distribution.png",
        "scatter_plot": "vus_pr_vs_inference.png",
        "training_plot": "training_time.png",
        "timestamp": time.time(),
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    summary["output_dir"] = str(output_dir)
    return summary


def recompute_ensemble(
    run_dir: str | Path,
    method: str = DEFAULT_ENSEMBLE_METHOD,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, float]:
    """Recompute ensemble metrics from a saved run without retraining.

    Loads ``scores.npz`` and ``results.json`` from ``run_dir``, rebuilds the
    ensemble with the given ``method`` and ``top_k``, and returns a dict of
    macro VUS-PR for each detector and the new ensemble configuration.
    """
    run_dir = Path(run_dir)
    with (run_dir / "results.json").open() as fh:
        saved = json.load(fh)

    scores_npz = np.load(run_dir / "scores.npz")
    model_names = [n for n in saved["model_names"] if n != ENSEMBLE_NAME]
    n_cases = len(saved["models"][model_names[0]]["series_results"])

    ensemble_vus_pr_list = []
    for i in range(n_cases):
        # Rank/weight by the selection injection (fall back to eval VUS-PR for
        # runs saved before the selection/evaluation split existed).
        select_by_model = {
            name: saved["models"][name]["series_results"][i].get(
                "vus_pr_select", saved["models"][name]["series_results"][i]["metrics"]["vus_pr"]
            )
            for name in model_names
        }
        top_models = rank_top_k(select_by_model, top_k)
        score_arrays = [scores_npz[f"{name}__case{i}"] for name in top_models]
        weights = [select_by_model[name] for name in top_models]
        fused = consensus(score_arrays, method=method, weights=weights)
        labels = scores_npz[f"__labels__case{i}"]
        metrics = compute_metrics(labels, fused, DEFAULT_ENSEMBLE_WINDOW)
        ensemble_vus_pr_list.append(metrics["vus_pr"])

    out: dict[str, float] = {name: saved["models"][name]["macro_metrics"]["vus_pr"] for name in model_names}
    out[f"Ensemble(method={method},top_k={top_k})"] = float(np.mean(ensemble_vus_pr_list))
    return out
