"""Anomaly-detection benchmark over air-quality series, with two evaluation modes.

**``unlabeled`` (default)** — label-free screening, the mode that matches
production (real time, no ground truth):

1. Fit/score every registered detector on each real series (no injection).
2. Binarize each detector's scores with a robust threshold on their own
   distribution (:func:`.metrics.mad_threshold`, median + k * scaled MAD) and
   compute its **detection rate** (fraction of flagged points).
3. **Discard** detectors whose macro detection rate exceeds
   ``max_detection_rate`` (default 7%): target anomalies are sensor faults
   (spikes, calibration drift, cutouts), which are rare — a higher rate means
   the detector flags normal variation.
4. Fuse the surviving detectors' normalized scores into a consensus ensemble
   (:func:`.ensemble.consensus`) and report its detection rate too.

**``synthetic``** — supervised evaluation against injected anomalies. The
:data:`INJECTION_VARIANT` (``combined``: a per-segment mix of anomaly shapes)
is injected **directly into the real series** — the old STL synthetic base was
removed after ``docs/estudio_inyeccion_stl_2026-07-03.md`` showed it distorts
per-model metrics. Each station is injected TWICE with independent seeds — a
*selection* injection (``seed``) and a held-out *evaluation* injection
(``eval_seed``). The ensemble ranks/weights the top-k detectors on the
selection injection's VUS-PR and is scored on the evaluation one, keeping its
metric unbiased. Reported metrics: auroc/aupr/vus_pr/vus_roc/affiliation_f1.

Both modes share the loading (raw 5-minute data → hourly means → longest
contiguous observed run per station, no ``dropna()`` gluing), the detector
fan-out, and the persistence: ``results.json`` (+ ``scores.npz``); plots come
from the separate ``plot_benchmark_results`` script.

See ``docs/seleccion_detectores_sin_etiquetas.md`` for the label-free
rationale and for ranking criteria (consensus centrality) that could be added.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
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
from .metrics import (
    DEFAULT_MAX_DETECTION_RATE,
    DEFAULT_THRESHOLD_K,
    compute_metrics,
    detect_mask,
    detection_rate,
    mad_threshold,
)
from .registry import (
    MODEL_REGISTRY,
    filter_model_kwargs as _filter_model_kwargs,
    resolve_model_class,
    resolve_model_names,
)

ENSEMBLE_NAME = "Ensemble"
MODES = ("unlabeled", "synthetic")
UNLABELED_METRIC_KEYS = ["detection_rate"]
SYNTHETIC_METRIC_KEYS = ["auroc", "aupr", "vus_pr", "vus_roc", "affiliation_f1"]
DEFAULT_ENSEMBLE_WINDOW = 100

# Synthetic mode injects a single variant: a per-segment mix of the anomaly
# shapes, applied directly to the real series (see :mod:`.anomalies`).
INJECTION_VARIANT = "combined"

# Detectors that benefit from a GPU (windowed deep models). Everything else is
# CPU-only. Mirrors genias's ``GPU_MODEL_NAMES``.
GPU_MODEL_NAMES = {"COUTABase", "COUTAGenIAS", "CARLABase", "CARLAGenIAS", "LSTMAD", "TSPulse"}

# Set once per GPU worker process (genias-style device binding via a shared queue).
_worker_device: str | None = None


def normalize_mode(mode: str) -> str:
    """Validate/normalize the benchmark mode to ``unlabeled`` / ``synthetic``."""
    normalized = str(mode).strip().lower()
    if normalized not in MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Use one of {MODES}")
    return normalized


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
    """One station's evaluation series.

    In ``unlabeled`` mode only ``values`` is set (the real series). In
    ``synthetic`` mode ``values``/``labels`` are the held-out *evaluation*
    injection and ``values_select``/``labels_select`` the independent
    *selection* injection used only to rank + weight detectors for the
    ensemble — keeping them separate stops the ensemble from selecting and
    evaluating on the same labels (which inflates its VUS-PR).
    """

    name: str
    values: np.ndarray
    labels: np.ndarray | None = None
    values_select: np.ndarray | None = None
    labels_select: np.ndarray | None = None


@dataclass
class AnomalyBenchmarkConfig:
    """Runtime settings for one benchmark run (mode, data, models, seeds)."""

    mode: str = "unlabeled"
    pollutant: str = "NO2"
    raw_base_dir: str = "data/raw/datos_estaciones_5m"
    models: list[str] | None = None
    ensemble_method: str = DEFAULT_ENSEMBLE_METHOD
    device: str = "cpu"
    seed: int = 13
    # unlabeled mode:
    threshold_k: float = DEFAULT_THRESHOLD_K
    max_detection_rate: float = DEFAULT_MAX_DETECTION_RATE
    # synthetic mode:
    eval_seed: int = 101    # held-out evaluation-injection seed (must differ from seed)
    ensemble_top_k: int = DEFAULT_TOP_K
    min_series_points: int = 600
    series_limit: int | None = None
    output_dir: str | None = None


def build_cases(config: AnomalyBenchmarkConfig) -> list[AnomalyCase]:
    """Load + preprocess every station; in ``synthetic`` mode inject twice.

    Every mode keeps the station's longest contiguous real run (no ``dropna()``
    gluing: it would break the daily phase windowed detectors assume).
    """
    mode = normalize_mode(config.mode)
    stations = load_raw_5m(config.pollutant, config.raw_base_dir)
    if config.series_limit is not None:
        stations = stations[: config.series_limit]

    logging.info("Building cases for %d station(s)  [mode=%s]…", len(stations), mode)
    cases: list[AnomalyCase] = []
    for station, frame in stations:
        processed, _ = preprocess([frame], config.pollutant)
        hourly = processed[0]
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
        if mode == "synthetic":
            sel_values, sel_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, config.seed)
            eval_values, eval_labels = inject_synthetic_anomalies(values, INJECTION_VARIANT, config.eval_seed)
            cases.append(
                AnomalyCase(
                    name=station,
                    values=eval_values,
                    labels=eval_labels,
                    values_select=sel_values,
                    labels_select=sel_labels,
                )
            )
        else:
            cases.append(AnomalyCase(name=station, values=values))
    logging.info("Built %d evaluation cases.", len(cases))
    return cases


def _fit_score_timed(
    model_cls: type,
    model_kwargs: dict[str, object],
    values: np.ndarray,
    seed: int,
    device: str,
) -> tuple[object, np.ndarray, float, float]:
    """Fit + score one fresh detector instance; return (model, scores, fit_s, inf_s)."""
    model = model_cls(seed=seed, **model_kwargs)
    synchronize_device(device)
    fit_started = time.perf_counter()
    model.fit(values)
    synchronize_device(device)
    fit_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    scores = np.asarray(model.score(values), dtype=np.float64)
    synchronize_device(device)
    inference_seconds = time.perf_counter() - inference_started
    return model, scores, fit_seconds, inference_seconds


def _score_case_unlabeled(
    model_cls: type,
    model_kwargs: dict[str, object],
    case: AnomalyCase,
    config: AnomalyBenchmarkConfig,
    device: str,
) -> dict[str, object]:
    """Unlabeled mode: one fit/score on the real series + MAD-threshold detection rate."""
    model, scores, fit_seconds, inference_seconds = _fit_score_timed(
        model_cls, model_kwargs, case.values, config.seed, device
    )
    mask = detect_mask(scores, config.threshold_k)
    return {
        "series_name": case.name,
        "series_length": int(case.values.shape[0]),
        "metrics": {"detection_rate": detection_rate(mask)},
        "n_flagged": int(mask.sum()),
        "threshold": float(mad_threshold(scores, config.threshold_k)),
        "timing": {"fit_seconds": float(fit_seconds), "inference_seconds": float(inference_seconds)},
        "training_summary": getattr(model, "training_summary_", {}),
        "scores": np.asarray(scores, dtype=np.float32),
    }


def _score_case_synthetic(
    model_cls: type,
    model_kwargs: dict[str, object],
    case: AnomalyCase,
    config: AnomalyBenchmarkConfig,
    device: str,
) -> dict[str, object]:
    """Synthetic mode: selection injection for ranking, held-out injection for metrics."""
    # Selection injection: its VUS-PR is only a ranking/weighting signal for
    # the ensemble; it is never the number we report or evaluate on.
    select_model = model_cls(seed=config.seed, **model_kwargs)
    select_model.fit(case.values_select)
    select_scores = np.asarray(select_model.score(case.values_select), dtype=np.float64)
    select_window = int(getattr(select_model, "window_size", DEFAULT_ENSEMBLE_WINDOW) or DEFAULT_ENSEMBLE_WINDOW)
    vus_pr_select = compute_metrics(case.labels_select, select_scores, select_window)["vus_pr"]

    # Held-out evaluation injection: every reported metric/score comes from here.
    model, scores, fit_seconds, inference_seconds = _fit_score_timed(
        model_cls, model_kwargs, case.values, config.seed, device
    )
    window_size = int(getattr(model, "window_size", DEFAULT_ENSEMBLE_WINDOW) or DEFAULT_ENSEMBLE_WINDOW)
    metrics = compute_metrics(case.labels, scores, window_size)
    return {
        "series_name": case.name,
        "series_length": int(case.values.shape[0]),
        "metrics": metrics,
        "vus_pr_select": float(vus_pr_select),
        "timing": {"fit_seconds": float(fit_seconds), "inference_seconds": float(inference_seconds)},
        "training_summary": getattr(model, "training_summary_", {}),
        "scores": np.asarray(scores, dtype=np.float32),
    }


def _case_log_snippet(entry: dict[str, object]) -> str:
    """One-line progress summary for a per-case entry, adapted to the mode."""
    metrics = entry["metrics"]
    if "detection_rate" in metrics:
        return f"rate={100.0 * metrics['detection_rate']:.2f}% ({entry['n_flagged']}/{entry['series_length']})"
    return f"vus_sel={entry['vus_pr_select']:.3f} vus_eval={metrics['vus_pr']:.3f}"


def _run_detector(
    model_name: str,
    config: AnomalyBenchmarkConfig,
    cases: list[AnomalyCase],
    device: str = "cpu",
) -> dict[str, object]:
    """Fit/score one detector over every case on ``device``; log per-case progress."""
    mode = normalize_mode(config.mode)
    score_case = _score_case_synthetic if mode == "synthetic" else _score_case_unlabeled
    model_cls = resolve_model_class(model_name)
    model_kwargs = _filter_model_kwargs(model_cls, {"device": device})

    per_case = []
    total = len(cases)
    model_started = time.perf_counter()
    logging.info("    [%s] START  %d cases on %s", model_name, total, device)
    for case_index, case in enumerate(cases):
        entry = score_case(model_cls, model_kwargs, case, config, device)
        per_case.append(entry)
        timing = entry["timing"]
        logging.info(
            "    [%s] case %d/%d  %s  %s  (%.1fs)",
            model_name,
            case_index + 1,
            total,
            case.name,
            _case_log_snippet(entry),
            timing["fit_seconds"] + timing["inference_seconds"],
        )
    headline_key = "vus_pr" if mode == "synthetic" else "detection_rate"
    macro = float(np.mean([entry["metrics"][headline_key] for entry in per_case])) if per_case else float("nan")
    logging.info(
        "    [%s] DONE   %d cases in %.1fs  macro_%s=%.3f",
        model_name,
        total,
        time.perf_counter() - model_started,
        headline_key,
        macro,
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


def macro_detection_rate(result: dict[str, object]) -> float:
    """Mean per-case detection rate of one detector's ``per_case`` results."""
    per_case = result["per_case"]
    if not per_case:
        return float("nan")
    return float(np.mean([entry["metrics"]["detection_rate"] for entry in per_case]))


def split_by_detection_rate(
    detector_results: dict[str, dict[str, object]],
    max_detection_rate: float,
) -> tuple[list[str], list[str]]:
    """Split detectors into ``(kept, discarded)`` by the detection-rate budget.

    A detector is discarded when its macro detection rate exceeds
    ``max_detection_rate``: sensor faults are rare, so flagging more than the
    budget means the detector is marking normal variation as anomalous.
    """
    kept: list[str] = []
    discarded: list[str] = []
    for name in detector_results:
        rate = macro_detection_rate(detector_results[name])
        (discarded if rate > max_detection_rate else kept).append(name)
    return sorted(kept), sorted(discarded)


def _build_unlabeled_ensemble(
    config: AnomalyBenchmarkConfig,
    cases: list[AnomalyCase],
    detector_results: dict[str, dict[str, object]],
    kept_models: list[str],
) -> list[dict[str, object]]:
    """Per case: fuse the SURVIVING detectors' scores and re-threshold the consensus."""
    ensemble_results = []
    for index, case in enumerate(cases):
        score_arrays = [detector_results[name]["per_case"][index]["scores"] for name in kept_models]
        fused = consensus(score_arrays, config.ensemble_method, config.seed)
        mask = detect_mask(fused, config.threshold_k)

        timings = [detector_results[name]["per_case"][index]["timing"] for name in kept_models]
        ensemble_results.append(
            {
                "series_name": case.name,
                "series_length": int(case.values.shape[0]),
                "metrics": {"detection_rate": detection_rate(mask)},
                "n_flagged": int(mask.sum()),
                "threshold": float(mad_threshold(np.asarray(fused, dtype=np.float64), config.threshold_k)),
                "timing": {
                    "fit_seconds": float(sum(timing["fit_seconds"] for timing in timings)),
                    "inference_seconds": float(sum(timing["inference_seconds"] for timing in timings)),
                },
                "training_summary": {"selected_models": kept_models, "method": config.ensemble_method},
            }
        )
    return ensemble_results


def _build_synthetic_ensemble(
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
        ensemble_results.append(
            {
                "series_name": case.name,
                "series_length": int(case.values.shape[0]),
                "metrics": metrics,
                "timing": {
                    "fit_seconds": float(sum(timing["fit_seconds"] for timing in timings)),
                    "inference_seconds": float(sum(timing["inference_seconds"] for timing in timings)),
                },
                "training_summary": {"selected_models": top_models, "method": config.ensemble_method},
            }
        )
    return ensemble_results


def _summarize(series_results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-case entries into macro metrics + timing (drops ``scores``).

    The macro metric set is derived from the entries themselves, so the same
    helper serves both modes.
    """
    clean = [{key: entry[key] for key in entry if key != "scores"} for entry in series_results]
    metric_keys = list(clean[0]["metrics"]) if clean else []
    macro_metrics = {
        metric: float(np.mean([entry["metrics"][metric] for entry in clean])) for metric in metric_keys
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
    mode = normalize_mode(config.mode)
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
        "Running %d detector(s) on %d case(s)  [mode=%s, device request=%s]",
        len(model_names),
        len(cases),
        mode,
        normalize_device_request(config.device),
    )
    logging.info("  Device plan: %s", {name: device_assignments[name] for name in model_names})

    # Persist cases once so parallel workers load them instead of rebuilding.
    cases_path = output_dir / "_cases.pkl"
    _save_cases(cases, cases_path)
    try:
        detector_results = _run_detectors(model_names, config, cases, cases_path, device_assignments)
    finally:
        cases_path.unlink(missing_ok=True)

    model_summaries: dict[str, dict[str, object]] = {}
    ordered_names = list(model_names)

    if mode == "synthetic":
        for name, result in detector_results.items():
            model_summaries[name] = _summarize(result["per_case"])
        logging.info("Building ensemble (top-%d by selection VUS-PR)…", config.ensemble_top_k)
        ensemble_results = _build_synthetic_ensemble(config, cases, detector_results)
        model_summaries[ENSEMBLE_NAME] = _summarize(ensemble_results)
        logging.info("  ✓ Ensemble  VUS-PR=%.3f", model_summaries[ENSEMBLE_NAME]["macro_metrics"]["vus_pr"])
        ordered_names.append(ENSEMBLE_NAME)
        kept_models, discarded_models = list(model_names), []
    else:
        kept_models, discarded_models = split_by_detection_rate(detector_results, config.max_detection_rate)
        for name in discarded_models:
            logging.info(
                "  ✗ %s DISCARDED  macro_detection_rate=%.2f%% > %.2f%%",
                name,
                100.0 * macro_detection_rate(detector_results[name]),
                100.0 * config.max_detection_rate,
            )
        for name, result in detector_results.items():
            model_summaries[name] = {**_summarize(result["per_case"]), "discarded": name in discarded_models}
        if kept_models:
            logging.info("Building ensemble from %d surviving detector(s)…", len(kept_models))
            ensemble_results = _build_unlabeled_ensemble(config, cases, detector_results, kept_models)
            model_summaries[ENSEMBLE_NAME] = {**_summarize(ensemble_results), "discarded": False}
            macro_rate = model_summaries[ENSEMBLE_NAME]["macro_metrics"]["detection_rate"]
            logging.info("  ✓ Ensemble  detection_rate=%.2f%%", 100.0 * macro_rate)
            ordered_names.append(ENSEMBLE_NAME)
        else:
            logging.warning(
                "Every detector exceeded max_detection_rate=%.2f%%; no ensemble built.",
                100.0 * config.max_detection_rate,
            )

    # Save raw per-case scores (+ labels in synthetic mode) so the ensemble can
    # be recomputed without retraining.
    scores_dict = {}
    for name, result in detector_results.items():
        for i, case_result in enumerate(result["per_case"]):
            scores_dict[f"{name}__case{i}"] = case_result["scores"]
    if mode == "synthetic":
        for i, case in enumerate(cases):
            scores_dict[f"__labels__case{i}"] = case.labels
    np.savez_compressed(output_dir / "scores.npz", **scores_dict)

    if mode == "synthetic":
        plot_names = {
            "metrics_plot": "vus_pr_distribution.png",
            "scatter_plot": "vus_pr_vs_inference.png",
            "training_plot": "training_time.png",
        }
    else:
        plot_names = {
            "metrics_plot": "detection_rate_distribution.png",
            "scatter_plot": "detection_rate_vs_inference.png",
            "training_plot": "training_time.png",
        }

    summary = {
        "config": asdict(config),
        "mode": mode,
        "models": model_summaries,
        "model_names": ordered_names,
        "kept_models": kept_models,
        "discarded_models": discarded_models,
        "series_names": sorted({case.name for case in cases}),
        **plot_names,
        "timestamp": time.time(),
    }
    if mode == "synthetic":
        summary["variants"] = [INJECTION_VARIANT]
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    summary["output_dir"] = str(output_dir)
    return summary


def recompute_ensemble(
    run_dir: str | Path,
    method: str = DEFAULT_ENSEMBLE_METHOD,
    top_k: int = DEFAULT_TOP_K,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    max_detection_rate: float | None = None,
) -> dict[str, float]:
    """Recompute ensemble headline numbers from a saved run without retraining.

    Loads ``scores.npz`` + ``results.json`` from ``run_dir`` and rebuilds the
    ensemble according to the run's mode:

    - ``synthetic``: rank/weight the top-``top_k`` detectors by their saved
      selection VUS-PR, fuse with ``method``, score against the saved labels;
      returns macro VUS-PR per detector + new ensemble.
    - ``unlabeled``: re-apply the detection-rate filter (``max_detection_rate``
      defaults to the saved value), fuse the survivors with ``method``,
      re-threshold with ``threshold_k``; returns macro detection rates.
    """
    run_dir = Path(run_dir)
    with (run_dir / "results.json").open() as fh:
        saved = json.load(fh)

    scores_npz = np.load(run_dir / "scores.npz")
    model_names = [n for n in saved["model_names"] if n != ENSEMBLE_NAME]
    n_cases = len(saved["models"][model_names[0]]["series_results"])
    mode = normalize_mode(saved.get("mode", saved.get("config", {}).get("mode", "unlabeled")))

    if mode == "synthetic":
        ensemble_vus_pr_list = []
        for i in range(n_cases):
            select_by_model = {
                name: saved["models"][name]["series_results"][i]["vus_pr_select"] for name in model_names
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

    if max_detection_rate is None:
        max_detection_rate = float(saved["config"]["max_detection_rate"])

    out = {}
    kept: list[str] = []
    for name in model_names:
        rates = [
            detection_rate(detect_mask(scores_npz[f"{name}__case{i}"], threshold_k))
            for i in range(n_cases)
        ]
        out[name] = float(np.mean(rates))
        if out[name] <= max_detection_rate:
            kept.append(name)

    if kept:
        ensemble_rates = []
        for i in range(n_cases):
            fused = consensus([scores_npz[f"{name}__case{i}"] for name in kept], method=method)
            ensemble_rates.append(detection_rate(detect_mask(fused, threshold_k)))
        out[f"Ensemble(method={method},k={threshold_k})"] = float(np.mean(ensemble_rates))
    return out
