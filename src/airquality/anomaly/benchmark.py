"""Anomaly-detection benchmark over air-quality series, with two evaluation modes.

**``unlabeled`` (default)** — label-free screening, the mode that matches
production (real time, no ground truth):

1. Fit/score every registered detector on each real series (no injection).
2. Binarize each detector's scores with a robust threshold on their own
   distribution (:func:`.metrics.mad_threshold`, median + k * scaled MAD) and
   compute its **detection rate** (fraction of flagged points).
3. For each station, **discard** detectors whose detection rate over its scored
   segments exceeds ``max_detection_rate`` (default 7%).
4. Combine the surviving detectors by strict-majority vote
   (:func:`.ensemble.consensus`) and report its detection rate too.

**``synthetic``** — supervised evaluation against injected anomalies. The
:data:`INJECTION_VARIANT` (``combined``: a per-segment mix of anomaly shapes)
is injected **directly into the real series** — the old STL synthetic base was
removed after ``docs/estudio_inyeccion_stl_2026-07-03.md`` showed it distorts
per-model metrics. Each station is injected TWICE with independent seeds — a
*selection* injection (``seed``) and a held-out *evaluation* injection
(``eval_seed``). The ensemble ranks the top-k detectors on the selection
injection's VUS-PR and combines their masks by strict-majority vote on the
held-out evaluation injection. Reported metrics:
auroc/aupr/vus_pr/vus_roc/affiliation_f1.

Both modes share the loading (raw 5-minute data → hourly means → all eligible
contiguous observed runs per station, no ``dropna()`` gluing), the detector
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
from airquality.paths import create_run_dir

from .anomalies import inject_synthetic_anomalies
from .ensemble import DEFAULT_TOP_K, consensus, rank_top_k
from .metrics import (
    DEFAULT_MAX_DETECTION_RATE,
    DEFAULT_THRESHOLD_K,
    compute_metrics,
    detect_mask,
    detection_rate,
    mad_threshold,
    vus_sliding_window,
)
from .registry import (
    MODEL_REGISTRY,
    filter_model_kwargs as _filter_model_kwargs,
    fit_model_segments,
    resolve_model_class,
    resolve_model_names,
    score_model_segments,
)

ENSEMBLE_NAME = "Ensemble"
MODES = ("unlabeled", "synthetic")
UNLABELED_METRIC_KEYS = ["detection_rate"]
SYNTHETIC_METRIC_KEYS = ["auroc", "aupr", "vus_pr", "vus_roc", "affiliation_f1"]

# Synthetic mode injects a single variant: a per-segment mix of the anomaly
# shapes, applied directly to the real series (see :mod:`.anomalies`).
INJECTION_VARIANT = "combined"
MIN_SEGMENT_POINTS = 8
# Combined injection includes a 16..48-point drift. Shorter series make the
# synthetic labels dominate the signal and do not provide a meaningful ranking.
MIN_SYNTHETIC_SEGMENT_POINTS = 300

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
    segment_lengths: tuple[int, ...] | None = None


@dataclass
class AnomalyBenchmarkConfig:
    """Runtime settings for one benchmark run (mode, data, models, seeds)."""

    mode: str = "unlabeled"
    pollutant: str = "NO2"
    raw_base_dir: str = "data/raw/datos_estaciones_5m"
    models: list[str] | None = None
    device: str = "cpu"
    seed: int = 13
    carla_stride: int = 1
    # unlabeled mode:
    threshold_k: float = DEFAULT_THRESHOLD_K
    max_detection_rate: float = DEFAULT_MAX_DETECTION_RATE
    # synthetic mode:
    eval_seed: int = 101    # held-out evaluation-injection seed (must differ from seed)
    ensemble_top_k: int = DEFAULT_TOP_K
    min_series_points: int = MIN_SEGMENT_POINTS
    series_limit: int | None = None
    output_dir: str | None = None


def build_cases(config: AnomalyBenchmarkConfig) -> list[AnomalyCase]:
    """Load + preprocess every station; in ``synthetic`` mode inject twice.

    Every mode keeps all sufficiently long contiguous real runs. Arrays are
    concatenated only for storage; detector fitting and scoring split them back
    at ``segment_lengths`` so gaps are never crossed.
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
        minimum = max(
            MIN_SEGMENT_POINTS,
            int(config.min_series_points),
            MIN_SYNTHETIC_SEGMENT_POINTS if mode == "synthetic" else 0,
        )
        segments = contiguous_observed_segments(hourly.iloc[:, 0], min_len=minimum)
        if not segments:
            logging.info(
                "  skip %s  (no contiguous run with at least %d points)",
                station,
                minimum,
            )
            continue
        segment_values = [segment.to_numpy(dtype=np.float32) for segment in segments]
        segment_lengths = tuple(len(values) for values in segment_values)
        values = np.concatenate(segment_values)
        if mode == "synthetic":
            selected = [
                inject_synthetic_anomalies(
                    segment, INJECTION_VARIANT, config.seed + position
                )
                for position, segment in enumerate(segment_values)
            ]
            evaluated = [
                inject_synthetic_anomalies(
                    segment, INJECTION_VARIANT, config.eval_seed + position
                )
                for position, segment in enumerate(segment_values)
            ]
            cases.append(
                AnomalyCase(
                    name=station,
                    values=np.concatenate([pair[0] for pair in evaluated]),
                    labels=np.concatenate([pair[1] for pair in evaluated]),
                    values_select=np.concatenate([pair[0] for pair in selected]),
                    labels_select=np.concatenate([pair[1] for pair in selected]),
                    segment_lengths=segment_lengths,
                )
            )
        else:
            cases.append(
                AnomalyCase(
                    name=station,
                    values=values,
                    segment_lengths=segment_lengths,
                )
            )
    logging.info("Built %d evaluation cases.", len(cases))
    return cases


def _case_segment_lengths(case: AnomalyCase) -> tuple[int, ...]:
    lengths = case.segment_lengths or (len(case.values),)
    if any(length <= 0 for length in lengths) or sum(lengths) != len(case.values):
        raise ValueError(f"Invalid segment lengths for {case.name}: {lengths}")
    return tuple(int(length) for length in lengths)


def _split_segments(values: np.ndarray, lengths: tuple[int, ...]) -> list[np.ndarray]:
    if sum(lengths) != len(values):
        raise ValueError("Segment lengths do not match array length")
    boundaries = np.cumsum((0, *lengths))
    return [values[boundaries[i] : boundaries[i + 1]] for i in range(len(lengths))]


def _weighted_metrics(
    metrics: list[dict[str, float]], lengths: tuple[int, ...]
) -> dict[str, float]:
    total = sum(lengths)
    if not metrics or total <= 0:
        return {}
    return {
        key: float(
            sum(metric[key] * length for metric, length in zip(metrics, lengths, strict=True))
            / total
        )
        for key in metrics[0]
    }


def _finite_mean(values: list[float]) -> float:
    """Mean over finite values, or NaN when no value is available."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _fit_score_timed(
    model_cls: type,
    model_kwargs: dict[str, object],
    segments: list[np.ndarray],
    seed: int,
    device: str,
) -> tuple[object, list[np.ndarray | None], float, float]:
    """Fit one detector on all station segments, then score each segment."""
    synchronize_device(device)
    fit_started = time.perf_counter()
    model = fit_model_segments(
        model_cls, segments, seed=seed, model_kwargs=model_kwargs
    )
    synchronize_device(device)
    fit_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    scores = score_model_segments(model, segments)
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
    """Fit once on a station, then score and MAD-threshold each segment."""
    lengths = _case_segment_lengths(case)
    segments = _split_segments(case.values, lengths)
    score_parts: list[np.ndarray] = []
    thresholds: list[float | None] = []
    scored_segments: list[bool] = []
    failures: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    n_flagged = 0
    scored_points = 0
    fit_seconds = 0.0
    inference_seconds = 0.0

    try:
        model, segment_scores, fit_seconds, inference_seconds = _fit_score_timed(
            model_cls, model_kwargs, segments, config.seed, device
        )
        summaries.append(getattr(model, "training_summary_", {}))
    except Exception as exc:
        segment_scores = [None] * len(segments)
        fit_seconds = 0.0
        inference_seconds = 0.0
        failures.append({"type": type(exc).__name__, "message": str(exc)})

    for segment_index, (segment, scores) in enumerate(
        zip(segments, segment_scores, strict=True)
    ):
        try:
            if scores is None:
                raise RuntimeError("detector could not score this segment")
            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != segment.shape:
                raise ValueError(f"expected scores {segment.shape}, received {scores.shape}")
            finite = np.isfinite(scores)
            if not finite.any():
                raise ValueError("detector produced no finite scores")
            mask = detect_mask(scores, config.threshold_k)
            score_parts.append(np.asarray(scores, dtype=np.float32))
            thresholds.append(float(mad_threshold(scores, config.threshold_k)))
            scored_segments.append(True)
            n_flagged += int(mask.sum())
            scored_points += int(finite.sum())
        except Exception as exc:
            score_parts.append(np.full(len(segment), np.nan, dtype=np.float32))
            thresholds.append(None)
            scored_segments.append(False)
            failures.append(
                {
                    "segment_index": segment_index,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    return {
        "series_name": case.name,
        "series_length": int(case.values.shape[0]),
        "segment_lengths": list(lengths),
        "metrics": {
            "detection_rate": n_flagged / scored_points if scored_points else float("nan")
        },
        "n_flagged": n_flagged,
        "scored_points": scored_points,
        "scored_segments": scored_segments,
        "thresholds": thresholds,
        "failures": failures,
        "timing": {"fit_seconds": float(fit_seconds), "inference_seconds": float(inference_seconds)},
        "training_summary": {"segments": summaries},
        "scores": np.concatenate(score_parts),
    }


def _score_case_synthetic(
    model_cls: type,
    model_kwargs: dict[str, object],
    case: AnomalyCase,
    config: AnomalyBenchmarkConfig,
    device: str,
) -> dict[str, object]:
    """Fit selection and held-out models once each across station segments."""
    if case.values_select is None or case.labels_select is None or case.labels is None:
        raise ValueError("Synthetic case is missing injected values or labels")
    lengths = _case_segment_lengths(case)
    selected = _split_segments(case.values_select, lengths)
    selected_labels = _split_segments(case.labels_select, lengths)
    evaluated = _split_segments(case.values, lengths)
    evaluated_labels = _split_segments(case.labels, lengths)
    score_parts: list[np.ndarray] = []
    segment_metrics: list[dict[str, float]] = []
    selection_vus: list[float] = []
    failures: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    fit_seconds = 0.0
    inference_seconds = 0.0

    try:
        select_model = fit_model_segments(
            model_cls, selected, seed=config.seed, model_kwargs=model_kwargs
        )
        selected_scores = score_model_segments(select_model, selected)
    except Exception:
        selected_scores = [None] * len(selected)

    try:
        model, evaluated_scores, fit_seconds, inference_seconds = _fit_score_timed(
            model_cls, model_kwargs, evaluated, config.seed, device
        )
        summaries.append(getattr(model, "training_summary_", {}))
    except Exception as exc:
        evaluated_scores = [None] * len(evaluated)
        fit_seconds = 0.0
        inference_seconds = 0.0
        failures.append({"type": type(exc).__name__, "message": str(exc)})

    for segment_index, (
        select_values,
        select_labels,
        select_scores,
        values,
        labels,
        scores,
    ) in enumerate(
        zip(
            selected,
            selected_labels,
            selected_scores,
            evaluated,
            evaluated_labels,
            evaluated_scores,
            strict=True,
        )
    ):
        if select_scores is None or np.asarray(select_scores).shape != select_values.shape:
            select_scores = np.zeros(select_values.shape, dtype=np.float64)
        else:
            select_scores = np.asarray(select_scores, dtype=np.float64)
        selection_vus.append(
            compute_metrics(
                select_labels, select_scores, vus_sliding_window(select_labels)
            )["vus_pr"]
        )

        try:
            if scores is None:
                raise RuntimeError("detector could not score this segment")
            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != values.shape:
                raise ValueError("evaluation score length mismatch")
        except Exception as exc:
            scores = np.zeros(values.shape, dtype=np.float64)
            failures.append(
                {
                    "segment_index": segment_index,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        score_parts.append(np.asarray(scores, dtype=np.float32))
        segment_metrics.append(
            compute_metrics(labels, scores, vus_sliding_window(labels))
        )

    metrics = _weighted_metrics(segment_metrics, lengths)
    return {
        "series_name": case.name,
        "series_length": int(case.values.shape[0]),
        "segment_lengths": list(lengths),
        "metrics": metrics,
        "vus_pr_select": float(np.mean(selection_vus)),
        "timing": {"fit_seconds": float(fit_seconds), "inference_seconds": float(inference_seconds)},
        "training_summary": {"segments": summaries},
        "failures": failures,
        "scores": np.concatenate(score_parts),
    }


def _case_log_snippet(entry: dict[str, object]) -> str:
    """One-line progress summary for a per-case entry, adapted to the mode."""
    metrics = entry["metrics"]
    if "detection_rate" in metrics:
        return (
            f"rate={100.0 * metrics['detection_rate']:.2f}% "
            f"({entry['n_flagged']}/{entry['scored_points']})"
        )
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
    requested_kwargs: dict[str, object] = {"device": device}
    if model_name in {"CARLABase", "CARLAGenIAS"}:
        requested_kwargs["stride"] = config.carla_stride
    model_kwargs = _filter_model_kwargs(model_cls, requested_kwargs)

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
    macro = _finite_mean([entry["metrics"][headline_key] for entry in per_case])
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
    return _finite_mean([entry["metrics"]["detection_rate"] for entry in per_case])


def split_by_detection_rate(
    detector_results: dict[str, dict[str, object]],
    max_detection_rate: float,
) -> tuple[list[str], list[str]]:
    """Legacy global split retained for historical artifact/tests compatibility.

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


def _selection_by_series(
    cases: list[AnomalyCase],
    detector_results: dict[str, dict[str, object]],
    max_detection_rate: float,
) -> dict[str, dict[str, list[str]]]:
    """Classify detectors independently for each station."""
    selection: dict[str, dict[str, list[str]]] = {}
    for case_index, case in enumerate(cases):
        kept: list[str] = []
        discarded: list[str] = []
        unavailable: list[str] = []
        for name, result in detector_results.items():
            entry = result["per_case"][case_index]
            if int(entry.get("scored_points", 0)) <= 0:
                unavailable.append(name)
            elif float(entry["metrics"]["detection_rate"]) > max_detection_rate:
                discarded.append(name)
            else:
                kept.append(name)
        selection[case.name] = {
            "kept_models": sorted(kept),
            "discarded_models": sorted(discarded),
            "unavailable_models": sorted(unavailable),
        }
    return selection


def _build_unlabeled_ensemble(
    config: AnomalyBenchmarkConfig,
    cases: list[AnomalyCase],
    detector_results: dict[str, dict[str, object]],
    selection_by_series: dict[str, dict[str, list[str]]],
) -> list[dict[str, object]]:
    """Fuse each station's survivors independently, segment by segment."""
    ensemble_results = []
    for index, case in enumerate(cases):
        kept_models = selection_by_series[case.name]["kept_models"]
        lengths = _case_segment_lengths(case)
        scores_by_model = {
            name: _split_segments(
                detector_results[name]["per_case"][index]["scores"], lengths
            )
            for name in kept_models
        }
        mask_parts: list[np.ndarray] = []
        voted_points = 0
        for segment_index, length in enumerate(lengths):
            score_arrays = [
                scores_by_model[name][segment_index]
                for name in kept_models
                if detector_results[name]["per_case"][index]["scored_segments"][
                    segment_index
                ]
            ]
            if score_arrays:
                finite = np.stack([np.isfinite(scores) for scores in score_arrays])
                votes = np.stack(
                    [detect_mask(scores, config.threshold_k) for scores in score_arrays]
                )
                available = finite.sum(axis=0)
                supported = available > 0
                flagged = supported & (votes.sum(axis=0) > available / 2.0)
                voted_points += int(supported.sum())
            else:
                flagged = np.zeros(length, dtype=bool)
            mask_parts.append(flagged)
        mask = np.concatenate(mask_parts)

        timings = [
            detector_results[name]["per_case"][index]["timing"]
            for name in kept_models
        ]
        ensemble_results.append(
            {
                "series_name": case.name,
                "series_length": int(case.values.shape[0]),
                "segment_lengths": list(lengths),
                "metrics": {
                    "detection_rate": (
                        int(mask.sum()) / voted_points
                        if voted_points
                        else float("nan")
                    )
                },
                "n_flagged": int(mask.sum()),
                "voted_points": voted_points,
                "threshold": 0.5,
                "timing": {
                    "fit_seconds": float(sum(timing["fit_seconds"] for timing in timings)),
                    "inference_seconds": float(sum(timing["inference_seconds"] for timing in timings)),
                },
                "training_summary": {"selected_models": kept_models, "method": "VOTE"},
                "scores": mask.astype(np.float32),
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
        lengths = _case_segment_lengths(case)
        labels_by_segment = _split_segments(case.labels, lengths)
        scores_by_model = {
            name: _split_segments(
                detector_results[name]["per_case"][index]["scores"], lengths
            )
            for name in top_models
        }
        segment_metrics = []
        for segment_index, labels in enumerate(labels_by_segment):
            fused = consensus(
                [scores_by_model[name][segment_index] for name in top_models],
                threshold_k=config.threshold_k,
            )
            segment_metrics.append(
                compute_metrics(labels, fused, vus_sliding_window(labels))
            )
        metrics = _weighted_metrics(segment_metrics, lengths)

        timings = [detector_results[name]["per_case"][index]["timing"] for name in top_models]
        ensemble_results.append(
            {
                "series_name": case.name,
                "series_length": int(case.values.shape[0]),
                "segment_lengths": list(lengths),
                "metrics": metrics,
                "timing": {
                    "fit_seconds": float(sum(timing["fit_seconds"] for timing in timings)),
                    "inference_seconds": float(sum(timing["inference_seconds"] for timing in timings)),
                },
                "training_summary": {"selected_models": top_models, "method": "VOTE"},
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
        metric: _finite_mean([entry["metrics"][metric] for entry in clean])
        for metric in metric_keys
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
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return create_run_dir(
        Path("reports") / "anomaly", f"{config.pollutant}_{stamp}"
    )


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
        selection_by_series = None
    else:
        selection_by_series = _selection_by_series(
            cases, detector_results, config.max_detection_rate
        )
        kept_models = sorted(
            {
                name
                for selection in selection_by_series.values()
                for name in selection["kept_models"]
            }
        )
        discarded_models = sorted(set(model_names) - set(kept_models))
        for name, result in detector_results.items():
            counts = {
                state: sum(
                    name in selection[f"{state}_models"]
                    for selection in selection_by_series.values()
                )
                for state in ("kept", "discarded", "unavailable")
            }
            model_summaries[name] = {
                **_summarize(result["per_case"]),
                "discarded": counts["kept"] == 0,
                "selection_counts": counts,
            }
        if kept_models:
            logging.info("Building per-series ensembles from locally surviving detectors…")
            ensemble_results = _build_unlabeled_ensemble(
                config, cases, detector_results, selection_by_series
            )
            model_summaries[ENSEMBLE_NAME] = {**_summarize(ensemble_results), "discarded": False}
            macro_rate = model_summaries[ENSEMBLE_NAME]["macro_metrics"]["detection_rate"]
            logging.info("  ✓ Ensemble  detection_rate=%.2f%%", 100.0 * macro_rate)
            ordered_names.append(ENSEMBLE_NAME)
        else:
            logging.warning(
                "No series has a detector under max_detection_rate=%.2f%%; no ensemble built.",
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
        "selection_scope": "series" if mode == "unlabeled" else "synthetic_top_k",
        "selection_by_series": selection_by_series,
        "schema_version": 2,
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
    top_k: int = DEFAULT_TOP_K,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    max_detection_rate: float | None = None,
) -> dict[str, float]:
    """Recompute ensemble headline numbers from a saved run without retraining.

    Loads ``scores.npz`` + ``results.json`` from ``run_dir`` and rebuilds the
    ensemble according to the run's mode:

    - ``synthetic``: rank/weight the top-``top_k`` detectors by their saved
      selection VUS-PR, combine by majority vote, score against saved labels;
      returns macro VUS-PR per detector + new ensemble.
    - ``unlabeled``: re-apply the detection-rate filter (``max_detection_rate``
      defaults to the saved value), combine survivors by majority vote; returns
      macro detection rates.
    """
    run_dir = Path(run_dir)
    with (run_dir / "results.json").open() as fh:
        saved = json.load(fh)

    scores_npz = np.load(run_dir / "scores.npz")
    model_names = [n for n in saved["model_names"] if n != ENSEMBLE_NAME]
    n_cases = len(saved["models"][model_names[0]]["series_results"])
    mode = normalize_mode(saved.get("mode", saved.get("config", {}).get("mode", "unlabeled")))
    schema_version = int(saved.get("schema_version", 1))

    if schema_version >= 2:
        if mode == "synthetic":
            ensemble_vus_pr_list = []
            for i in range(n_cases):
                select_by_model = {
                    name: saved["models"][name]["series_results"][i][
                        "vus_pr_select"
                    ]
                    for name in model_names
                }
                top_models = rank_top_k(select_by_model, top_k)
                lengths = tuple(
                    saved["models"][model_names[0]]["series_results"][i][
                        "segment_lengths"
                    ]
                )
                labels_by_segment = _split_segments(
                    scores_npz[f"__labels__case{i}"], lengths
                )
                scores_by_model = {
                    name: _split_segments(scores_npz[f"{name}__case{i}"], lengths)
                    for name in top_models
                }
                segment_metrics = []
                for segment_index, labels in enumerate(labels_by_segment):
                    fused = consensus(
                        [
                            scores_by_model[name][segment_index]
                            for name in top_models
                        ],
                        threshold_k=threshold_k,
                    )
                    segment_metrics.append(
                        compute_metrics(labels, fused, vus_sliding_window(labels))
                    )
                ensemble_vus_pr_list.append(
                    _weighted_metrics(segment_metrics, lengths)["vus_pr"]
                )
            out = {
                name: saved["models"][name]["macro_metrics"]["vus_pr"]
                for name in model_names
            }
            out[f"Ensemble(method=VOTE,top_k={top_k})"] = float(
                np.mean(ensemble_vus_pr_list)
            )
            return out

        if max_detection_rate is None:
            max_detection_rate = float(saved["config"]["max_detection_rate"])
        model_rates: dict[str, list[float]] = {name: [] for name in model_names}
        ensemble_rates = []
        has_survivor = False
        for i in range(n_cases):
            first_entry = saved["models"][model_names[0]]["series_results"][i]
            lengths = tuple(first_entry["segment_lengths"])
            kept_for_series = []
            scores_by_model = {}
            availability_by_model = {}
            for name in model_names:
                entry = saved["models"][name]["series_results"][i]
                segments = _split_segments(scores_npz[f"{name}__case{i}"], lengths)
                availability = list(entry["scored_segments"])
                scored_points = sum(
                    length
                    for length, available in zip(lengths, availability, strict=True)
                    if available
                )
                flagged = sum(
                    int(detect_mask(scores, threshold_k).sum())
                    for scores, available in zip(segments, availability, strict=True)
                    if available
                )
                rate = flagged / scored_points if scored_points else float("nan")
                if scored_points:
                    model_rates[name].append(rate)
                if scored_points and rate <= max_detection_rate:
                    kept_for_series.append(name)
                scores_by_model[name] = segments
                availability_by_model[name] = availability

            has_survivor |= bool(kept_for_series)

            flagged_total = 0
            voted_total = 0
            for segment_index, length in enumerate(lengths):
                arrays = [
                    scores_by_model[name][segment_index]
                    for name in kept_for_series
                    if availability_by_model[name][segment_index]
                ]
                if arrays:
                    finite = np.stack([np.isfinite(scores) for scores in arrays])
                    votes = np.stack(
                        [detect_mask(scores, threshold_k) for scores in arrays]
                    )
                    available = finite.sum(axis=0)
                    supported = available > 0
                    flagged_total += int(
                        (supported & (votes.sum(axis=0) > available / 2.0)).sum()
                    )
                    voted_total += int(supported.sum())
            if voted_total:
                ensemble_rates.append(flagged_total / voted_total)

        out = {
            name: _finite_mean(rates) for name, rates in model_rates.items()
        }
        if has_survivor:
            out[f"Ensemble(method=VOTE,k={threshold_k})"] = float(
                _finite_mean(ensemble_rates)
            )
        return out

    if mode == "synthetic":
        ensemble_vus_pr_list = []
        for i in range(n_cases):
            select_by_model = {
                name: saved["models"][name]["series_results"][i]["vus_pr_select"] for name in model_names
            }
            top_models = rank_top_k(select_by_model, top_k)
            score_arrays = [scores_npz[f"{name}__case{i}"] for name in top_models]
            fused = consensus(score_arrays, threshold_k=threshold_k)
            labels = scores_npz[f"__labels__case{i}"]
            metrics = compute_metrics(labels, fused, vus_sliding_window(labels))
            ensemble_vus_pr_list.append(metrics["vus_pr"])

        out: dict[str, float] = {name: saved["models"][name]["macro_metrics"]["vus_pr"] for name in model_names}
        out[f"Ensemble(method=VOTE,top_k={top_k})"] = float(np.mean(ensemble_vus_pr_list))
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
            fused = consensus(
                [scores_npz[f"{name}__case{i}"] for name in kept],
                threshold_k=threshold_k,
            )
            ensemble_rates.append(detection_rate(fused.astype(bool)))
        out[f"Ensemble(method=VOTE,k={threshold_k})"] = float(np.mean(ensemble_rates))
    return out
