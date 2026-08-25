"""Config-driven forecasting benchmark over anomaly-cleaning arms.

Measures whether anomaly detection (and the subsequent imputation) improves
multi-step forecasting. For every configured series the pipeline builds one
training *arm* per (detection strategy, imputation) combination — plus the
``raw`` and ``raw+frozen`` source baselines — and backtests the same forecasting
models on each arm over the **same timestamps** in short (8 h, stride 4 h) and
long (48 h, stride 24 h) regimes:

- ``raw``: the hourly-mean series as loaded (gaps + anomalies kept).
- ``raw+frozen``: the parallel hourly series without either frozen-value filter.
- ``<strategy>+impute``: anomalies flagged by the strategy are removed and the
  resulting gaps imputed (:mod:`airquality.forecasting.fill`).
- ``<strategy>+noimpute``: anomalies removed, gaps left as NaN (the backtest
  trains on the contiguous subseries) — isolates the effect of imputation.

Detection strategies (:mod:`airquality.forecasting.detection`): ``unlabeled``
(rate-filtered consensus, the production method), ``inject-best`` (single best
detector by VUS-PR on a synthetic-injection copy) and ``inject-vote`` (top-3
by injection VUS-PR, 2-of-3 mask vote). Detector fits are shared across
strategies through a per-series :class:`~airquality.forecasting.detection.SeriesDetectionContext`,
and every strategy's mask can be post-processed through ``mask_transforms``
hooks before removal.

Detection runs over the complete primary series before splitting. The test is
chosen after masking the union of anomaly flags; strategy abstentions are
reported but do not remove support. Every arm uses the selected timestamps on
its own source series. Removal and imputation still touch only training.

Foundation models run on the two raw-source views in that common time window. A
separate paired experiment injects one synthetic anomaly type into copies of
clean test contexts and compares ``clean_reference``, ``corrupted`` and
``inject-vote``; imputation is only a compatibility fallback after a NaN causes
foundation prediction to fail.

Detections and backtests are cached on disk (:mod:`airquality.forecasting.cache`,
``[forecasting] use_cache`` / ``cache_dir``): an interrupted run resumes where
it stopped and re-runs only recompute what changed. Figures are rendered
separately from the persisted CSVs, so they never trigger recomputation.

Run with::

    uv run python -m airquality.forecasting.pipeline
    uv run python -m airquality.visualizations.forecasting [run_dir]
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import multiprocessing as mp
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import time
from typing import Any, Sequence

import pandas as pd

from airquality.anomaly.registry import resolve_model_names
from airquality.config import (
    cfg_get_bool,
    cfg_get_csv_list,
    cfg_get_float,
    cfg_get_int,
    cfg_get_str,
    get_config,
)
from airquality.data.loaders import load_raw_5m
from airquality.data.preprocessing import preprocess
from airquality.data.series import ensure_datetime_series
from airquality.forecasting.backtest import (
    backtest_forecast,
    forecast_foundation_context,
    get_strict_forecast_requirements,
    prepare_foundation_model,
    select_holdout_window,
)
from airquality.forecasting.cache import (
    CACHE_VERSION,
    BenchmarkCache,
    artifact_fingerprint,
    effective_config,
    series_fingerprint,
    transform_fingerprints,
)
from airquality.forecasting.cleaning import remove_anomalies
from airquality.forecasting.detection import (
    DEFAULT_INJECTION_VARIANT,
    DEFAULT_INJECTION_SEED,
    INJECTION_POLICY_VERSION,
    DEFAULT_MIN_SELECTION_POINTS,
    DEFAULT_VOTE_MIN_VOTES,
    DEFAULT_VOTE_TOP_K,
    MIN_SEGMENT_POINTS,
    DetectionResult,
    DetectionStrategy,
    MaskTransform,
    SeriesDetectionContext,
    STRATEGY_INJECT_VOTE,
    apply_mask_transforms,
    build_detection_strategy,
    common_detection_support,
    normalize_injection_variant,
)
from airquality.forecasting.fill import (
    DEFAULT_MAX_GAP_SIZE,
    _repo_root,
    _resolve_tspulse_model_path,
    build_imputer,
    impute_series,
)
from airquality.forecasting.foundation_preprocessing import (
    build_preprocessing_contexts,
    build_synthetic_context_cases,
    summarize_foundation_preprocessing,
)
from airquality.forecasting.registry import (
    forecast_model_cache_identity,
    resolve_forecasting_model_configs,
)
from airquality.forecasting.progress import (
    BenchmarkProgress,
    configure_worker_progress_logging,
    get_progress_logger,
)
from airquality.imputation.registry import (
    DARTS_GLOBAL,
    TSPULSE,
    resolve_imputer_family,
)
from airquality.paths import create_run_dir

RAW_ARM = "raw"
RAW_FROZEN_ARM = "raw+frozen"
RAW_SOURCE_ARMS = (RAW_ARM, RAW_FROZEN_ARM)
DEFAULT_STRATEGIES = ("unlabeled", "inject-best", "inject-vote")
IMPUTATION_CHOICES = ("both", "impute", "none")
#: The benchmark's reported metrics, both scale-free so they compare across
#: series of different levels and across arms:
#:
#: - ``rmse``: RMSE in raw units divided by ``scale_ref`` (std of the RAW
#:   observed training series) — i.e. the RMSE on standardized data.
#: - ``mase``: the scaled MAE — the existing darts MASE (MAE over the
#:   seasonal-naive MAE of the training history), computed for EVERY arm
#:   against the RAW training history.
#:
#: Both scale references are per series and shared by every arm; per-arm
#: scaling would bias the comparison (cleaning removes spikes, shrinking that
#: arm's std / naive error and inflating its scaled metric). Multiply ``rmse``
#: by the persisted ``scale_ref`` column to recover raw units.
METRIC_COLS = ("rmse", "mase")
PROGRESS_LOGGER = get_progress_logger()
_ACTIVE_PROGRESS: BenchmarkProgress | None = None
_ACTIVE_GPU_EXECUTOR: ProcessPoolExecutor | None = None
_ACTIVE_DEVICE_QUEUE: Any | None = None

RESULT_COLUMNS = (
    "regime", "horizon", "forecast_stride", "validation_len",
    "validation_stride", "series", "arm", "strategy", "imputed",
    "imputation_model", "detectors", "n_anomalies", "n_anomalies_full",
    "detection_scope", "split_basis", "split_n_flagged", "split_n_unscored",
    "test_context_start", "test_target_start", "test_target_end",
    "test_target_hours", "model",
    "model_mode", "rmse", "mase", "train_seconds", "inference_seconds",
    "scale_ref", "n_test_predictions", "n_forecasts", "n_expected_forecasts",
    "n_unique_targets", "origin_mae_mean", "origin_mae_std",
    "origin_rmse_mean", "origin_rmse_std",
)

SELECTION_COLUMNS = (
    "series", "series_start", "series_end", "observed_hours",
    "common_support_hours", "split_n_flagged", "split_n_unscored",
    "split_strategies", "selected", "exclusion_reason", "source_run_start",
    "test_context_start", "train_end", "test_target_start", "test_target_end",
    "test_target_hours", "test_age_hours",
)

FOUNDATION_PREPROCESSING_COLUMNS = (
    "series", "regime", "horizon", "model", "case_id", "anomaly_type",
    "test_seed", "test_target_start", "condition", "detectors", "n_injected",
    "n_context_flagged", "n_injected_detected", "n_context_nan",
    "imputation_applied", "imputation_model", "n_imputed", "rmse", "mase",
    "model_load_seconds", "inference_seconds", "scale_ref",
    "n_test_predictions", "failure_reason",
)


def _load_raw_hourly_series(
    *,
    pollutant: str,
    raw_base_dir: str,
    freq: str,
    preserve_frozen: bool = False,
) -> list[pd.DataFrame]:
    """Load raw 5-minute stations and apply the shared hourly preprocess."""
    if freq != "h":
        raise ValueError("El forecasting preprocesado desde 5 min requiere freq='h'")

    out: list[pd.DataFrame] = []
    for station, raw in load_raw_5m(pollutant, raw_base_dir):
        (hourly,), _ = preprocess(
            [raw],
            pollutant,
            exclude_frozen=not preserve_frozen,
            remove_repeated=not preserve_frozen,
        )
        series = ensure_datetime_series(
            hourly.iloc[:, 0].rename(station), freq=freq, name=station
        )
        out.append(series.to_frame())
    return out


def _series_by_name(frames: Sequence[pd.DataFrame]) -> dict[str, pd.Series]:
    """Index one source view by station, rejecting ambiguous duplicates."""
    series_by_name: dict[str, pd.Series] = {}
    for frame in frames:
        name = str(frame.columns[0])
        if name in series_by_name:
            raise RuntimeError(f"Nombre de estacion duplicado: {name}")
        series_by_name[name] = frame.iloc[:, 0]
    return series_by_name


@dataclass(frozen=True)
class ForecastArm:
    """One benchmark arm: how the training series is preprocessed before fitting."""

    name: str
    strategy: str | None  # detection strategy spec; None = raw-source arm
    impute: bool


@dataclass(frozen=True)
class ForecastRegime:
    """Forecast horizon, rolling cadence, and validation target span."""

    name: str
    horizon: int
    stride: int
    validation_len: int


@dataclass(frozen=True)
class _BacktestTask:
    """CPU inputs for one forecast backtest executed by a GPU worker."""

    train_series: pd.Series
    test_series: pd.Series
    mase_insample: pd.Series
    model_name: str
    size_k: int
    test_target_start: pd.Timestamp
    seasonality_m: int
    freq: str
    validation_len: int
    validation_stride: int
    forecast_stride: int
    context_len: int
    series_name: str = ""
    arm_name: str = ""
    regime_name: str = ""
    ordinal: int = 0
    total: int = 0


_FORECAST_WORKER_GPU: int | None = None


def resolve_forecasting_devices(requested: str) -> tuple[str, ...]:
    """Resolve a forecasting request to visible CUDA devices or CPU fallback."""
    choice = str(requested).strip().lower()
    if choice not in {"cpu", "cuda", "multi-gpu"}:
        raise ValueError("`[forecasting] device` debe ser cpu, cuda o multi-gpu")
    if choice == "cpu":
        return ("cpu",)
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    if count == 0:
        return ("cpu",)
    if choice == "cuda":
        return ("cuda:0",)
    return tuple(f"cuda:{index}" for index in range(count))


def _bind_forecast_worker(device_queue: Any, log_queue: Any | None = None) -> None:
    """Claim exactly one GPU for the lifetime of a spawned forecast worker."""
    import torch

    global _FORECAST_WORKER_GPU
    _FORECAST_WORKER_GPU = int(device_queue.get())
    torch.cuda.set_device(_FORECAST_WORKER_GPU)
    if log_queue is not None:
        configure_worker_progress_logging(log_queue)


def _release_cuda_memory() -> None:
    """Release unreachable objects and cached CUDA allocator blocks."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _run_gpu_backtest(task: _BacktestTask) -> dict[str, Any]:
    """Build and run one Darts model on the GPU owned by this worker."""
    if _FORECAST_WORKER_GPU is None:
        raise RuntimeError("El worker de forecasting no tiene una GPU asignada")
    model_config = resolve_forecasting_model_configs(
        [task.model_name],
        seasonality_m=task.seasonality_m,
        context_length=task.context_len,
        accelerator="gpu",
        devices=[_FORECAST_WORKER_GPU],
    )[task.model_name]
    started = time.perf_counter()
    PROGRESS_LOGGER.info(
        "[backtest task=%d/%d][%s][%s][%s][%s] start device=cuda:%d",
        task.ordinal,
        task.total,
        task.series_name,
        task.regime_name,
        task.arm_name,
        task.model_name,
        _FORECAST_WORKER_GPU,
    )
    try:
        result = backtest_forecast(
            task.train_series,
            task.test_series,
            task.model_name,
            size_k=task.size_k,
            test_target_start=task.test_target_start,
            seasonality_m=task.seasonality_m,
            freq=task.freq,
            mase_insample=task.mase_insample,
            validation_len=task.validation_len,
            validation_stride=task.validation_stride,
            forecast_stride=task.forecast_stride,
            context_len=task.context_len,
            cleanup_checkpoints=True,
            model_config=model_config,
        )
    except Exception:
        PROGRESS_LOGGER.exception(
            "[backtest task=%d/%d][%s][%s][%s][%s] failed device=cuda:%d",
            task.ordinal,
            task.total,
            task.series_name,
            task.regime_name,
            task.arm_name,
            task.model_name,
            _FORECAST_WORKER_GPU,
        )
        raise
    finally:
        _release_cuda_memory()
    PROGRESS_LOGGER.info(
        "[backtest task=%d/%d][%s][%s][%s][%s] worker-finish device=cuda:%d elapsed=%.1fs",
        task.ordinal,
        task.total,
        task.series_name,
        task.regime_name,
        task.arm_name,
        task.model_name,
        _FORECAST_WORKER_GPU,
        time.perf_counter() - started,
    )
    return result


def _uses_gpu_worker(model_config: Any) -> bool:
    """Return whether a model owns a Lightning trainer and benefits from a GPU."""
    return "pl_trainer_kwargs" in model_config.kwargs


def _create_gpu_executor(
    gpu_indices: tuple[int, ...],
    log_queue: Any | None = None,
) -> tuple[ProcessPoolExecutor, Any]:
    """Create one spawned, device-bound worker per visible GPU."""
    spawn_context = mp.get_context("spawn")
    device_queue = spawn_context.Queue()
    for gpu_index in gpu_indices:
        device_queue.put(gpu_index)
    executor = ProcessPoolExecutor(
        max_workers=len(gpu_indices),
        mp_context=spawn_context,
        initializer=_bind_forecast_worker,
        initargs=(device_queue, log_queue),
    )
    return executor, device_queue


def _shutdown_gpu_resources(*, cancel_futures: bool) -> None:
    """Close GPU workers and their device queue after success or failure."""
    global _ACTIVE_DEVICE_QUEUE, _ACTIVE_GPU_EXECUTOR
    executor = _ACTIVE_GPU_EXECUTOR
    device_queue = _ACTIVE_DEVICE_QUEUE
    _ACTIVE_GPU_EXECUTOR = None
    _ACTIVE_DEVICE_QUEUE = None
    if executor is not None:
        try:
            executor.shutdown(wait=True, cancel_futures=cancel_futures)
        except TypeError:
            # Keep lightweight test doubles and older executors compatible.
            executor.shutdown(wait=True)
    if device_queue is not None:
        device_queue.close()
        device_queue.join_thread()


def build_arms(strategies: Sequence[str], imputation: str) -> list[ForecastArm]:
    """Expand strategy specs; the two raw-source arms are always first.

    ``imputation`` picks the variants built per strategy: ``impute`` (detect →
    remove → impute), ``none`` (detect → remove, gaps stay NaN) or ``both``.
    """
    if imputation not in IMPUTATION_CHOICES:
        raise ValueError(
            f"Valor de imputacion desconocido: '{imputation}'. Usa uno de {IMPUTATION_CHOICES}"
        )
    arms = [
        ForecastArm(RAW_ARM, None, False),
        ForecastArm(RAW_FROZEN_ARM, None, False),
    ]
    for spec in dict.fromkeys(strategies):  # dedupe, keep order
        if imputation in ("both", "impute"):
            arms.append(ForecastArm(f"{spec}+impute", spec, True))
        if imputation in ("both", "none"):
            arms.append(ForecastArm(f"{spec}+noimpute", spec, False))
    return arms


def _build_output_dir() -> Path:
    """Create the timestamped output directory for one benchmark run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return create_run_dir(_repo_root() / "reports" / "forecasting", stamp)


def _detect_for_strategies(
    series: pd.Series,
    strategies: Sequence[DetectionStrategy],
    mask_transforms: Sequence[MaskTransform] | None,
    *,
    cache: BenchmarkCache,
    base_key: dict[str, Any],
    context_kwargs: dict[str, Any],
) -> dict[str, DetectionResult]:
    """Resolve every strategy's detection from the cache or by running it.

    The (expensive) scoring context is only built when at least one strategy
    misses the cache; cached and fresh results share the same post-processed
    (``mask_transforms``) form, so downstream code cannot tell them apart.
    """
    context: SeriesDetectionContext | None = None
    detections: dict[str, DetectionResult] = {}
    for strategy in strategies:
        key = {**base_key, "stage": "detection", "strategy": asdict(strategy)}
        started = time.perf_counter()
        detection = cache.get("detection", key)
        cache_state = (
            "hit" if detection is not None else ("miss" if cache.enabled else "off")
        )
        if detection is None:
            PROGRESS_LOGGER.info(
                "[detection][%s][%s] start cache=%s",
                series.name,
                strategy.name,
                cache_state,
            )
            if context is None:
                context = SeriesDetectionContext(series, **context_kwargs)
            detection = apply_mask_transforms(series, strategy.detect(context), mask_transforms)
            cache.put("detection", key, detection)
        PROGRESS_LOGGER.info(
            "[detection][%s][%s] done cache=%s detectors=%d flagged=%d elapsed=%.1fs",
            series.name,
            strategy.name,
            cache_state,
            len(detection.detectors),
            detection.n_flagged,
            time.perf_counter() - started,
        )
        detections[strategy.name] = detection
    return detections


def _common_detection_support(
    series: pd.Series,
    detections: dict[str, DetectionResult],
) -> tuple[pd.Series, pd.Series]:
    """Compatibility alias for :func:`common_detection_support`."""
    return common_detection_support(series, detections)


def _format_ranking(ranking: dict[str, float]) -> str:
    """Selection ranking as a compact ``name=vus`` string, best first."""
    ordered = sorted(ranking.items(), key=lambda item: (-item[1], item[0]))
    return ";".join(f"{name}={value:.3f}" for name, value in ordered)


def _summarize(results_df: pd.DataFrame, baseline_arm: str = RAW_ARM) -> pd.DataFrame:
    """Pivot per-(series, model) metrics into arm columns with deltas vs the baseline.

    For every metric (scaled RMSE/MAE, see :data:`METRIC_COLS`) and non-baseline
    arm the summary adds ``_delta`` (arm − baseline; negative = arm better) and
    ``_improve_pct`` (positive = arm better) columns, so each preprocessing path
    is read directly against ``raw``.
    """
    if results_df.empty:
        return pd.DataFrame(columns=["regime", "series", "model"])

    arm_order = list(dict.fromkeys(results_df["arm"]))
    wide = results_df.pivot_table(
        index=["regime", "series", "model"],
        columns="arm",
        values=list(METRIC_COLS),
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for (regime, series, model), row in wide.iterrows():
        entry: dict[str, Any] = {"regime": regime, "series": series, "model": model}
        for metric in METRIC_COLS:
            baseline_val = row.get((metric, baseline_arm))
            for arm in arm_order:
                value = row.get((metric, arm))
                entry[f"{metric}_{arm}"] = value
                if arm == baseline_arm:
                    continue
                if pd.notna(value) and pd.notna(baseline_val):
                    entry[f"{metric}_{arm}_delta"] = float(value) - float(baseline_val)
                    entry[f"{metric}_{arm}_improve_pct"] = (
                        100.0 * (float(baseline_val) - float(value)) / float(baseline_val)
                        if baseline_val
                        else float("nan")
                    )
        rows.append(entry)
    return pd.DataFrame(rows)


def _cacheable_backtest(result: dict[str, Any]) -> bool:
    """Only persist complete runs; transient failures must be retried."""
    expected = int(result.get("n_expected_forecasts", 0))
    return (
        expected > 0
        and int(result.get("n_forecasts", 0)) == expected
        and int(result.get("n_test_predictions", 0)) > 0
        and math.isfinite(float(result.get("rmse", float("nan"))))
    )


def _cacheable_foundation_test(result: dict[str, Any], horizon: int) -> bool:
    """Persist only complete finite single-origin foundation forecasts."""
    return (
        int(result.get("n_test_predictions", 0)) == horizon
        and math.isfinite(float(result.get("rmse", float("nan"))))
        and math.isfinite(float(result.get("mase", float("nan"))))
    )


def _run_benchmark_from_config(
    mask_transforms: Sequence[MaskTransform] | None = None,
) -> dict[str, Any]:
    """Run the multi-arm forecasting benchmark defined by the config.

    ``mask_transforms`` post-process every strategy's anomaly mask (in order)
    before removal — the hook for future preprocessing of the detection output.
    """
    freq = cfg_get_str("data", "freq", "h")
    pollutant = cfg_get_str("forecasting", "pollutant", "NO2")
    raw_base_dir = cfg_get_str(
        "data", "raw_base_dir", "data/raw/datos_estaciones_5m"
    )
    imputation_size_k = cfg_get_int("benchmark", "size_k", 5)
    seasonality_m = cfg_get_int("benchmark", "seasonality_m", 24)

    holdout = cfg_get_int("forecasting", "holdout", 192)
    context_len = cfg_get_int("forecasting", "context_len", 72)
    regimes = (
        ForecastRegime(
            "short",
            cfg_get_int("forecasting", "short_horizon", 8),
            cfg_get_int("forecasting", "short_stride", 4),
            cfg_get_int("forecasting", "short_validation_len", 48),
        ),
        ForecastRegime(
            "long",
            cfg_get_int("forecasting", "long_horizon", 48),
            cfg_get_int("forecasting", "long_stride", 24),
            cfg_get_int("forecasting", "long_validation_len", 96),
        ),
    )
    if holdout <= 0 or any(
        min(regime.horizon, regime.stride, regime.validation_len) <= 0
        or regime.stride > regime.horizon
        or regime.validation_len < regime.horizon
        or (regime.validation_len - regime.horizon) % regime.stride != 0
        or holdout < regime.horizon
        or (holdout - regime.horizon) % regime.stride != 0
        for regime in regimes
    ):
        raise ValueError(
            "Holdout, horizonte, stride y validacion de cada regimen deben ser validos"
        )
    seed = cfg_get_int("forecasting", "seed", 13)
    device_request = cfg_get_str("forecasting", "device", "multi-gpu")
    forecast_devices = resolve_forecasting_devices(device_request)
    gpu_indices = tuple(
        int(device.split(":", 1)[1])
        for device in forecast_devices
        if device.startswith("cuda:")
    )
    detection_device = "cuda" if gpu_indices else "cpu"
    threshold_k = cfg_get_float("forecasting", "threshold_k", 3.5)
    max_detection_rate = cfg_get_float("forecasting", "max_detection_rate", 0.07)
    carla_stride = cfg_get_int("forecasting", "carla_stride", 1)
    detectors = list(cfg_get_csv_list("forecasting", "detectors", ("all",)))
    imputation_model = cfg_get_str("forecasting", "imputation_model", "TSPulse")
    max_imputation_gap = cfg_get_int(
        "forecasting", "max_imputation_gap", DEFAULT_MAX_GAP_SIZE
    )
    if max_imputation_gap < 1:
        raise ValueError("max_imputation_gap debe ser positivo")
    forecast_models = list(
        cfg_get_csv_list("forecasting", "forecast_models", ("NLinear", "TiDE"))
    )
    forecast_model_configs = resolve_forecasting_model_configs(
        forecast_models,
        seasonality_m=seasonality_m,
        context_length=context_len,
        accelerator="gpu" if gpu_indices else "cpu",
        devices=[gpu_indices[0]] if gpu_indices else 1,
    )
    forecast_models = list(forecast_model_configs)
    foundation_model_configs = {
        name: config
        for name, config in forecast_model_configs.items()
        if config.mode == "foundation"
    }
    strict_requirements = {
        regime.name: get_strict_forecast_requirements(
            forecast_model_configs,
            size_k=regime.horizon,
            validation_len=regime.validation_len,
            validation_stride=regime.stride,
            seasonality_m=seasonality_m,
            context_len=context_len,
        )
        for regime in regimes
    }
    context_requirement = max(
        context_len,
        *(
            int(requirement["prediction_context_hours"])
            for requirement in strict_requirements.values()
        ),
    )
    train_requirement = max(
        int(requirement["minimum_hours"]) for requirement in strict_requirements.values()
    )
    host_requirement = max(
        int(requirement["host_minimum_hours"])
        for requirement in strict_requirements.values()
    )
    strategy_specs = [
        spec.strip().lower()
        for spec in cfg_get_csv_list("forecasting", "strategies", DEFAULT_STRATEGIES)
    ]
    imputation = cfg_get_str("forecasting", "imputation", "both").strip().lower()
    injection_seed = cfg_get_int("forecasting", "injection_seed", DEFAULT_INJECTION_SEED)
    injection_variant = normalize_injection_variant(
        cfg_get_str("synthetic", "injection_variant", DEFAULT_INJECTION_VARIANT)
    )
    foundation_test_requested = cfg_get_bool(
        "forecasting", "foundation_preprocessing_test", True
    )
    foundation_test_seed = cfg_get_int(
        "forecasting", "foundation_test_seed", 1001
    )
    foundation_test_repeats = cfg_get_int(
        "forecasting", "foundation_test_repeats", 1
    )
    min_selection_points = cfg_get_int(
        "forecasting", "min_selection_points", DEFAULT_MIN_SELECTION_POINTS
    )
    if threshold_k < 0:
        raise ValueError("threshold_k no puede ser negativo")
    if carla_stride < 1:
        raise ValueError("carla_stride debe ser positivo")
    if not 0.0 <= max_detection_rate <= 1.0:
        raise ValueError("max_detection_rate debe estar entre 0 y 1")
    if min_selection_points < MIN_SEGMENT_POINTS:
        raise ValueError(
            f"min_selection_points debe ser al menos {MIN_SEGMENT_POINTS}"
        )
    if foundation_test_repeats < 1:
        raise ValueError("foundation_test_repeats debe ser positivo")
    foundation_test_seeds = range(
        foundation_test_seed,
        foundation_test_seed + foundation_test_repeats * 4,
    )
    if injection_seed in foundation_test_seeds:
        raise ValueError(
            "Las semillas foundation deben diferir de injection_seed para separar "
            "seleccion y test sintetico"
        )
    vote_top_k = cfg_get_int("forecasting", "vote_top_k", DEFAULT_VOTE_TOP_K)
    vote_min_votes = cfg_get_int("forecasting", "vote_min_votes", DEFAULT_VOTE_MIN_VOTES)
    use_cache = cfg_get_bool("forecasting", "use_cache", True)
    cache_dir = cfg_get_str("forecasting", "cache_dir", "reports/forecasting/cache")
    heartbeat_seconds = cfg_get_int("forecasting", "heartbeat_seconds", 60)
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds debe ser positivo")
    cache = BenchmarkCache((_repo_root() / cache_dir) if use_cache else None)
    foundation_test_enabled = foundation_test_requested and bool(
        foundation_model_configs
    )

    arms = build_arms(strategy_specs, imputation)
    if all(not config.uses_training_arms for config in forecast_model_configs.values()):
        arms = arms[: len(RAW_SOURCE_ARMS)]
    active_strategy_specs = [arm.strategy for arm in arms if arm.strategy]
    strategies = [
        build_detection_strategy(
            spec,
            threshold_k=threshold_k,
            max_detection_rate=max_detection_rate,
            vote_top_k=vote_top_k,
            vote_min_votes=vote_min_votes,
        )
        for spec in dict.fromkeys(active_strategy_specs)
    ]
    foundation_strategies = (
        [
            build_detection_strategy(
                STRATEGY_INJECT_VOTE,
                threshold_k=threshold_k,
                max_detection_rate=max_detection_rate,
                vote_top_k=vote_top_k,
                vote_min_votes=vote_min_votes,
            )
        ]
        if foundation_test_enabled
        else []
    )

    series_dfs = _load_raw_hourly_series(
        pollutant=pollutant, raw_base_dir=raw_base_dir, freq=freq
    )
    if not series_dfs:
        raise RuntimeError("No se cargaron series para el benchmark.")
    frozen_series_dfs = _load_raw_hourly_series(
        pollutant=pollutant,
        raw_base_dir=raw_base_dir,
        freq=freq,
        preserve_frozen=True,
    )
    primary_series_by_name = _series_by_name(series_dfs)
    frozen_series_by_name = _series_by_name(frozen_series_dfs)
    if set(frozen_series_by_name) != set(primary_series_by_name):
        raise RuntimeError("Las vistas raw y raw+frozen no contienen las mismas estaciones")

    # Resolve "all" against the registry NOW so cache keys list concrete names
    # (registry availability, e.g. optional TSPulse, then invalidates entries).
    resolved_detectors = resolve_model_names(detectors)
    use_scaler = imputation_model not in ("interp", "LinearInterp")
    imputer_ref: list[Any] = []  # built lazily: only when an imputed arm misses the cache

    def get_imputer() -> Any:
        if not imputer_ref:
            imputer_ref.append(
                build_imputer(imputation_model, freq=freq, size_k=imputation_size_k)
            )
        return imputer_ref[0]

    strategy_by_spec = {strategy.name: strategy for strategy in strategies}
    transform_names = transform_fingerprints(mask_transforms)
    runtime_config = get_config()
    training_model_config = effective_config(
        {
            "training": {
                key: value
                for key, value in runtime_config.items("training")
                if key != "model_names"
            }
            if runtime_config.has_section("training")
            else {},
            "models": dict(runtime_config.items("models"))
            if runtime_config.has_section("models")
            else {},
        }
    )
    cache_config = effective_config(
        {
            "freq": freq,
            "pollutant": pollutant,
            "raw_base_dir": raw_base_dir,
            "holdout": holdout,
            "context_len": context_len,
            "split_policy": "fixed-anomaly-mask-support-v2",
            "regimes": [asdict(regime) for regime in regimes],
            "seed": seed,
            "device": device_request,
            "threshold_k": threshold_k,
            "max_detection_rate": max_detection_rate,
            "detectors": sorted(resolved_detectors),
            "carla_stride": carla_stride,
            "injection_seed": injection_seed,
            "injection_variant": injection_variant,
            "injection_policy": INJECTION_POLICY_VERSION,
            "min_selection_points": min_selection_points,
            "vote_top_k": vote_top_k,
            "vote_min_votes": vote_min_votes,
        }
    )
    foundation_test_config = effective_config(
        {
            "enabled": foundation_test_enabled,
            "test_seed": foundation_test_seed,
            "repeats": foundation_test_repeats,
            "anomaly_policy": "single-type-per-case-v1",
            "imputation_policy": "fallback-after-nan-prediction-failure-v1",
        }
    )

    imputer_identity: dict[str, Any] | None = None
    if any(arm.impute for arm in arms) or foundation_test_enabled:
        family = resolve_imputer_family(imputation_model)
        artifacts: dict[str, str | None] = {}
        imputer_config: dict[str, Any] = {
            "model": imputation_model,
            "family": family,
            "size_k": imputation_size_k,
            "max_gap_size": max_imputation_gap,
        }
        if family == DARTS_GLOBAL:
            weights = _repo_root() / "models" / f"{imputation_model}_k{imputation_size_k}.pt"
            artifacts = {
                "model": artifact_fingerprint(weights),
                "checkpoint": artifact_fingerprint(Path(f"{weights}.ckpt")),
            }
        elif family == TSPULSE:
            model_path = _resolve_tspulse_model_path(imputation_model)
            model_id = cfg_get_str(
                "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
            )
            imputer_config["tspulse"] = effective_config(
                {
                    "model_path": model_path,
                    "model_id": model_id,
                    "revision": cfg_get_str(
                        "tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1"
                    ),
                    "context_length": cfg_get_int("tspulse", "context_length", 512),
                    "device": cfg_get_str("tspulse", "device", "cpu"),
                }
            )
            local_source = Path(model_path or model_id).expanduser()
            if local_source.exists():
                artifacts["model"] = artifact_fingerprint(local_source)
        imputer_identity = {"config": imputer_config, "artifacts": artifacts}

    global _ACTIVE_DEVICE_QUEUE, _ACTIVE_GPU_EXECUTOR, _ACTIVE_PROGRESS
    output_dir = _build_output_dir()
    progress = BenchmarkProgress(
        output_dir,
        heartbeat_seconds=heartbeat_seconds,
    ).start()
    _ACTIVE_PROGRESS = progress
    run_started = time.perf_counter()
    backtests_per_station = len(regimes) * sum(
        len(arms)
        if config.uses_training_arms
        else sum(arm.name in RAW_SOURCE_ARMS for arm in arms)
        for config in forecast_model_configs.values()
    )
    PROGRESS_LOGGER.info(
        "[run] start stations=%d devices=%s models=%s arms=%s regimes=%s "
        "backtests_per_station=%d cache=%s output=%s",
        len(series_dfs),
        ",".join(forecast_devices),
        ",".join(forecast_models),
        ",".join(arm.name for arm in arms),
        ",".join(regime.name for regime in regimes),
        backtests_per_station,
        cache.root if cache.enabled else "off",
        output_dir,
    )
    PROGRESS_LOGGER.info(
        "[log-guide] elapsed=wall-clock time since the current operation started; "
        "task=position in the queued task list; completed=finished tasks; "
        "total=tasks for the current station; pending_gpu=queued GPU tasks"
    )
    PROGRESS_LOGGER.info(
        "[log-guide] backtest identity is [series][regime][branch][model]; "
        "series=station/series name, regime=short or long, "
        "branch=forecast arm (raw, unlabeled+impute, etc.), model=model name"
    )
    PROGRESS_LOGGER.info(
        "[log-guide] example: [backtest task=18/172 completed=7/172]"
        "[AQN1 - Puerto][long][unlabeled+impute][TiDE]"
    )

    rows: list[dict[str, Any]] = []
    foundation_preprocessing_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    gpu_executor: ProcessPoolExecutor | None = None
    device_queue: Any | None = None
    if gpu_indices and any(
        _uses_gpu_worker(config) for config in forecast_model_configs.values()
    ):
        gpu_executor, device_queue = _create_gpu_executor(
            gpu_indices,
            progress.log_queue,
        )
        _ACTIVE_GPU_EXECUTOR = gpu_executor
        _ACTIVE_DEVICE_QUEUE = device_queue
    for station_index, df in enumerate(series_dfs, start=1):
        station_started = time.perf_counter()
        series = df.iloc[:, 0]
        name = str(series.name)
        frozen_series = frozen_series_by_name[name]
        if not frozen_series.index.equals(series.index):
            raise RuntimeError(f"{name}: raw y raw+frozen no comparten la misma rejilla")
        progress.update(
            stage="detection",
            detail=f"station={station_index}/{len(series_dfs)} name={name}",
            completed=0,
            total=len(strategies),
            pending=0,
        )
        PROGRESS_LOGGER.info(
            "[station %d/%d][%s] start observed=%d",
            station_index,
            len(series_dfs),
            name,
            int(series.notna().sum()),
        )
        series_fp = series_fingerprint(series)
        series_foundation_test_seed = foundation_test_seed + int(series_fp[:8], 16)
        base_key = {
            "version": CACHE_VERSION,
            "series": name,
            "series_fp": series_fp,
            "config": cache_config,
            "freq": freq,
            "detectors": sorted(resolved_detectors),
            "seed": seed,
            "carla_stride": carla_stride,
            "injection_seed": injection_seed,
            "injection_variant": injection_variant,
            "injection_policy": INJECTION_POLICY_VERSION,
            "min_selection_points": min_selection_points,
            "transforms": transform_names,
            "regimes": [asdict(regime) for regime in regimes],
        }
        detections = _detect_for_strategies(
            series,
            strategies,
            mask_transforms,
            cache=cache,
            base_key=base_key,
            context_kwargs={
                "detectors": resolved_detectors,
                "seed": seed,
                "device": detection_device,
                "freq": freq,
                "carla_stride": carla_stride,
                "injection_seed": injection_seed,
                "injection_variant": injection_variant,
                "min_selection_points": min_selection_points,
                "cache": cache,
                "cache_key": {
                    "version": CACHE_VERSION,
                    "series": name,
                    "series_fp": series_fp,
                    "freq": freq,
                    "seed": seed,
                    "carla_stride": carla_stride,
                },
            },
        )
        for detection_index, (spec, detection) in enumerate(
            detections.items(), start=1
        ):
            n_observed = int(series.notna().sum())
            n_scored = (
                int(
                    (
                        detection.scored_mask.reindex(series.index, fill_value=False).astype(bool)
                        & series.notna()
                    ).sum()
                )
                if detection.scored_mask is not None
                else n_observed
            )
            PROGRESS_LOGGER.info(
                "[detection][%s][%s] summary detectors=%s discarded=%s "
                "rate=%.2f%% flagged=%d coverage=%.2f%%",
                name,
                spec,
                ",".join(detection.detectors) or "none",
                ",".join(detection.discarded) or "none",
                100.0 * detection.detection_rate,
                detection.n_flagged,
                100.0 * (n_scored / n_observed if n_observed else 0.0),
            )
            progress.update(completed=detection_index)
            detection_rows.append(
                {
                    "series": name,
                    "strategy": spec,
                    "detectors": ",".join(detection.detectors),
                    "discarded": ",".join(detection.discarded),
                    "n_flagged": detection.n_flagged,
                    "n_unscored": n_observed - n_scored,
                    "coverage_rate": n_scored / n_observed if n_observed else 0.0,
                    "detection_rate": detection.detection_rate,
                    "ranking": _format_ranking(detection.ranking),
                    "scope": "full_series",
                    "n_observed": n_observed,
                }
            )

        support, common_mask = _common_detection_support(series, detections)
        flagged_mask = pd.Series(False, index=series.index)
        unscored_mask = pd.Series(False, index=series.index)
        for detection in detections.values():
            flagged_mask |= detection.mask.reindex(series.index, fill_value=False).astype(bool)
            if detection.scored_mask is not None:
                scored = detection.scored_mask.reindex(series.index, fill_value=False).astype(bool)
                unscored_mask |= series.notna() & ~scored
        split_n_flagged = int((flagged_mask & series.notna()).sum())
        split_n_unscored = int(unscored_mask.sum())
        window = select_holdout_window(
            support,
            holdout=holdout,
            context_len=context_requirement,
            train_min_len=train_requirement,
            validation_len=max(regime.validation_len for regime in regimes),
            freq=freq,
            host_min_len=host_requirement,
        )
        selection_common = {
            "series": name,
            "series_start": series.index.min(),
            "series_end": series.index.max(),
            "observed_hours": int(series.notna().sum()),
            "common_support_hours": int(support.notna().sum()),
            "split_n_flagged": split_n_flagged,
            "split_n_unscored": split_n_unscored,
            "split_strategies": ",".join(detections),
        }
        if window is None:
            PROGRESS_LOGGER.info(
                "[station %d/%d][%s] skip reason=no_common_fixed_holdout_and_training_host "
                "context=%dh holdout=%dh elapsed=%.1fs",
                station_index,
                len(series_dfs),
                name,
                context_requirement,
                holdout,
                time.perf_counter() - station_started,
            )
            selection_rows.append(
                {
                    **selection_common,
                    "selected": False,
                    "exclusion_reason": "no_common_fixed_holdout_and_training_host",
                }
            )
            continue

        test_target_start = window["test_target_start"]
        test_series = series.loc[window["test_index"]]
        frozen_test_series = frozen_series.loc[window["test_index"]]
        train_raw = series.loc[series.index < test_target_start]
        train_frozen = frozen_series.loc[frozen_series.index < test_target_start]
        if test_series.isna().any() or common_mask.loc[window["test_index"]].any():
            raise RuntimeError(f"{name}: el test comun no es valido")
        if frozen_test_series.isna().any():
            raise RuntimeError(f"{name}: raw+frozen no cubre el test comun")
        selection_rows.append(
            {
                **selection_common,
                "selected": True,
                "exclusion_reason": "",
                "source_run_start": window["source_run_start"],
                "test_context_start": window["context_index"][0],
                "train_end": window["train_end"],
                "test_target_start": test_target_start,
                "test_target_end": window["test_target_end"],
                "test_target_hours": window["test_target_hours"],
                "test_age_hours": int(
                    (series.index.max() - window["test_target_end"])
                    / pd.Timedelta(hours=1)
                ),
            }
        )
        PROGRESS_LOGGER.info(
            "[station %d/%d][%s] selected train_end=%s test=%s..%s",
            station_index,
            len(series_dfs),
            name,
            window["train_end"],
            test_target_start,
            window["test_target_end"],
        )

        # One raw scale and MASE history are shared by every arm.
        scale_ref = float(train_raw.std())
        if not math.isfinite(scale_ref) or scale_ref <= 0.0:
            scale_ref = float("nan")
        train_fp = series_fingerprint(train_raw)
        frozen_train_fp = series_fingerprint(train_frozen)
        support_fp = series_fingerprint(support)

        # Arm training series are built lazily: fully-cached arms skip anomaly
        # removal AND imputation entirely (that is what makes resume cheap).
        train_by_arm: dict[str, pd.Series] = {}

        def train_for(arm: ForecastArm) -> pd.Series:
            if arm.name not in train_by_arm:
                if arm.name == RAW_FROZEN_ARM:
                    base = train_frozen
                elif arm.strategy is None:
                    base = train_raw
                else:
                    base = remove_anomalies(train_raw, detections[arm.strategy])
                train_by_arm[arm.name] = (
                    impute_series(
                        base,
                        get_imputer(),
                        freq=freq,
                        use_scaler=use_scaler,
                        max_gap_size=max_imputation_gap,
                    )
                    if arm.impute
                    else base
                )
            return train_by_arm[arm.name]

        def test_for(arm: ForecastArm) -> pd.Series:
            return frozen_test_series if arm.name == RAW_FROZEN_ARM else test_series

        series_backtest_rows: dict[int, dict[str, Any]] = {}
        pending_gpu: dict[
            Any, tuple[int, dict[str, Any], str, str, dict[str, Any], str, float]
        ] = {}
        backtest_ordinal = 0
        backtest_completed = 0
        progress.update(
            stage="backtest",
            detail=f"station={station_index}/{len(series_dfs)} name={name}",
            completed=0,
            total=backtests_per_station,
            pending=0,
        )

        def store_backtest_row(
            ordinal: int,
            common: dict[str, Any],
            model_name: str,
            model_mode: str,
            result: dict[str, Any],
            *,
            cache_state: str,
            wall_seconds: float,
        ) -> None:
            nonlocal backtest_completed
            series_backtest_rows[ordinal] = {
                **common,
                "model": model_name,
                "model_mode": model_mode,
                # Timings are measured inside the worker around fit/predict only.
                "rmse": result["rmse"] / scale_ref,
                "mase": result["mase"],
                "train_seconds": result.get("train_seconds", float("nan")),
                "inference_seconds": result.get("inference_seconds", float("nan")),
                "scale_ref": scale_ref,
                "n_test_predictions": result["n_test_predictions"],
                "n_forecasts": result.get("n_forecasts", 0),
                "n_expected_forecasts": result.get("n_expected_forecasts", 0),
                "n_unique_targets": result.get("n_unique_targets", 0),
                "origin_mae_mean": result.get("origin_mae_mean", float("nan")),
                "origin_mae_std": result.get("origin_mae_std", float("nan")),
                "origin_rmse_mean": result.get("origin_rmse_mean", float("nan")),
                "origin_rmse_std": result.get("origin_rmse_std", float("nan")),
            }
            backtest_completed += 1
            status = "ok" if _cacheable_backtest(result) else "incomplete"
            PROGRESS_LOGGER.info(
                "[backtest task=%d/%d completed=%d/%d][%s][%s][%s][%s] "
                "done cache=%s status=%s train=%.1fs inference=%.1fs wall=%.1fs",
                ordinal + 1,
                backtests_per_station,
                backtest_completed,
                backtests_per_station,
                common["series"],
                common["regime"],
                common["arm"],
                model_name,
                cache_state,
                status,
                float(result.get("train_seconds", float("nan"))),
                float(result.get("inference_seconds", float("nan"))),
                wall_seconds,
            )
            progress.update(
                completed=backtest_completed,
                pending=len(pending_gpu),
                detail=(
                    f"station={station_index}/{len(series_dfs)} name={name} "
                    f"last={common['regime']}/{common['arm']}/{model_name}"
                ),
            )

        for regime in regimes:
            for arm in arms:
                detection = detections.get(arm.strategy) if arm.strategy else None
                n_train_anomalies = (
                    int(
                        (
                            detection.mask.reindex(train_raw.index, fill_value=False).astype(bool)
                            & train_raw.notna()
                        ).sum()
                    )
                    if detection
                    else 0
                )
                common = {
                    "regime": regime.name,
                    "horizon": regime.horizon,
                    "forecast_stride": regime.stride,
                    "validation_len": regime.validation_len,
                    "validation_stride": regime.stride,
                    "series": name,
                    "arm": arm.name,
                    "strategy": arm.strategy or "none",
                    "imputed": arm.impute,
                    "imputation_model": imputation_model if arm.impute else "none",
                    "detectors": ",".join(detection.detectors) if detection else "",
                    "n_anomalies": n_train_anomalies,
                    "n_anomalies_full": detection.n_flagged if detection else 0,
                    "detection_scope": "full_series" if detection else "none",
                    "split_basis": "common_anomaly_mask_support" if detections else "raw",
                    "split_n_flagged": split_n_flagged,
                    "split_n_unscored": split_n_unscored,
                    "test_context_start": str(window["test_context_start"]),
                    "test_target_start": str(test_target_start),
                    "test_target_end": str(window["test_target_end"]),
                    "test_target_hours": window["test_target_hours"],
                }
                arm_strategy_key = (
                    asdict(strategy_by_spec[arm.strategy]) if arm.strategy else None
                )
                arm_test_series = test_for(arm)
                arm_train_fp = (
                    frozen_train_fp if arm.name == RAW_FROZEN_ARM else train_fp
                )
                arm_test_fp = series_fingerprint(arm_test_series)
                for model_name in forecast_models:
                    forecast_model_config = forecast_model_configs[model_name]
                    if (
                        not forecast_model_config.uses_training_arms
                        and arm.name not in RAW_SOURCE_ARMS
                    ):
                        continue
                    ordinal = backtest_ordinal
                    backtest_ordinal += 1
                    backtest_key = {
                        **base_key,
                        "train_fp": arm_train_fp,
                        "support_fp": support_fp,
                        "stage": "backtest",
                        "arm": arm.name,
                        "strategy": arm_strategy_key,
                        "impute": arm.impute,
                        "imputation_model": imputation_model if arm.impute else None,
                        "model": model_name,
                        "model_config": training_model_config,
                        "forecast_model": effective_config(
                            forecast_model_cache_identity(forecast_model_config)
                        ),
                        "imputer": imputer_identity if arm.impute else None,
                        "regime": asdict(regime),
                        "seasonality_m": seasonality_m,
                        "test_target_start": str(test_target_start),
                        "test_fp": arm_test_fp,
                    }
                    task_number = ordinal + 1
                    task_started = time.perf_counter()
                    res = cache.get("backtest", backtest_key)
                    cache_state = (
                        "hit" if res is not None else ("miss" if cache.enabled else "off")
                    )
                    if res is None:
                        PROGRESS_LOGGER.info(
                            "[backtest task=%d/%d][%s][%s][%s][%s] prepare cache=%s",
                            task_number,
                            backtests_per_station,
                            name,
                            regime.name,
                            arm.name,
                            model_name,
                            cache_state,
                        )
                        progress.update(
                            detail=(
                                f"station={station_index}/{len(series_dfs)} name={name} "
                                f"prepare={regime.name}/{arm.name}/{model_name}"
                            )
                        )
                        arm_train = train_for(arm)
                        if gpu_executor is not None and _uses_gpu_worker(
                            forecast_model_config
                        ):
                            PROGRESS_LOGGER.info(
                                "[backtest task=%d/%d][%s][%s][%s][%s] queued cache=%s",
                                task_number,
                                backtests_per_station,
                                name,
                                regime.name,
                                arm.name,
                                model_name,
                                cache_state,
                            )
                            future = gpu_executor.submit(
                                _run_gpu_backtest,
                                _BacktestTask(
                                    train_series=arm_train,
                                    test_series=arm_test_series,
                                    mase_insample=train_raw,
                                    model_name=model_name,
                                    size_k=regime.horizon,
                                    test_target_start=test_target_start,
                                    seasonality_m=seasonality_m,
                                    freq=freq,
                                    validation_len=regime.validation_len,
                                    validation_stride=regime.stride,
                                    forecast_stride=regime.stride,
                                    context_len=context_len,
                                    series_name=name,
                                    arm_name=arm.name,
                                    regime_name=regime.name,
                                    ordinal=task_number,
                                    total=backtests_per_station,
                                ),
                            )
                            pending_gpu[future] = (
                                ordinal,
                                common,
                                model_name,
                                forecast_model_config.mode,
                                backtest_key,
                                cache_state,
                                task_started,
                            )
                            progress.update(pending=len(pending_gpu))
                            continue
                        PROGRESS_LOGGER.info(
                            "[backtest task=%d/%d][%s][%s][%s][%s] start cache=%s device=cpu",
                            task_number,
                            backtests_per_station,
                            name,
                            regime.name,
                            arm.name,
                            model_name,
                            cache_state,
                        )
                        res = backtest_forecast(
                            arm_train,
                            arm_test_series,
                            model_name,
                            size_k=regime.horizon,
                            test_target_start=test_target_start,
                            seasonality_m=seasonality_m,
                            freq=freq,
                            validation_len=regime.validation_len,
                            validation_stride=regime.stride,
                            forecast_stride=regime.stride,
                            context_len=context_len,
                            cleanup_checkpoints=True,
                            model_config=forecast_model_config,
                            # Shared raw history: every arm's MASE uses the SAME
                            # seasonal-naive denominator (see METRIC_COLS).
                            mase_insample=train_raw,
                        )
                        if _cacheable_backtest(res):
                            cache.put("backtest", backtest_key, res)
                    store_backtest_row(
                        ordinal,
                        common,
                        model_name,
                        forecast_model_config.mode,
                        res,
                        cache_state=cache_state,
                        wall_seconds=time.perf_counter() - task_started,
                    )

        for gpu_completed, future in enumerate(as_completed(pending_gpu), start=1):
            (
                ordinal,
                common,
                model_name,
                model_mode,
                backtest_key,
                cache_state,
                task_started,
            ) = pending_gpu[future]
            res = future.result()
            if _cacheable_backtest(res):
                cache.put("backtest", backtest_key, res)
            store_backtest_row(
                ordinal,
                common,
                model_name,
                model_mode,
                res,
                cache_state=cache_state,
                wall_seconds=time.perf_counter() - task_started,
            )
            progress.update(pending=len(pending_gpu) - gpu_completed)
        rows.extend(series_backtest_rows[index] for index in sorted(series_backtest_rows))

        if foundation_test_enabled:
            for regime in regimes:
                cases = build_synthetic_context_cases(
                    test_series,
                    test_target_start=test_target_start,
                    context_len=context_len,
                    horizon=regime.horizon,
                    stride=regime.stride,
                    repeats=foundation_test_repeats,
                    test_seed=series_foundation_test_seed,
                    freq=freq,
                )
                progress.update(
                    stage="foundation-prepare",
                    detail=f"station={name} regime={regime.name}",
                    completed=0,
                    total=len(cases),
                    pending=0,
                )
                PROGRESS_LOGGER.info(
                    "[foundation][%s][%s] prepare-cases start cases=%d",
                    name,
                    regime.name,
                    len(cases),
                )
                prepared_cases = []
                for case_index, case in enumerate(cases, start=1):
                    case_started = time.perf_counter()
                    PROGRESS_LOGGER.info(
                        "[foundation-case %d/%d][%s][%s][%s] start origin=%s",
                        case_index,
                        len(cases),
                        name,
                        regime.name,
                        case.anomaly_type,
                        case.origin,
                    )
                    history = series.loc[: case.clean_context.index[-1]].copy()
                    history.loc[case.corrupted_context.index] = case.corrupted_context
                    synthetic_fp = series_fingerprint(history)
                    synthetic_base_key = {
                        **base_key,
                        "series_fp": synthetic_fp,
                        "experiment": "foundation-preprocessing-synthetic-v1",
                        "foundation_test_config": foundation_test_config,
                        "anomaly_type": case.anomaly_type,
                        "test_seed": case.test_seed,
                        "test_target_start": str(case.origin),
                    }
                    synthetic_detections = _detect_for_strategies(
                        history,
                        foundation_strategies,
                        mask_transforms,
                        cache=cache,
                        base_key=synthetic_base_key,
                        context_kwargs={
                            "detectors": resolved_detectors,
                            "seed": seed,
                            "device": detection_device,
                            "freq": freq,
                            "carla_stride": carla_stride,
                            "injection_seed": injection_seed,
                            "injection_variant": injection_variant,
                            "min_selection_points": min_selection_points,
                            "cache": cache,
                            "cache_key": {
                                "version": CACHE_VERSION,
                                "series": name,
                                "series_fp": synthetic_fp,
                                "experiment": "foundation-preprocessing-synthetic-v1",
                                "foundation_test_config": foundation_test_config,
                                "freq": freq,
                                "seed": seed,
                                "carla_stride": carla_stride,
                                "injection_variant": injection_variant,
                            },
                        },
                    )
                    prepared_cases.append(
                        (
                            case,
                            synthetic_detections,
                            build_preprocessing_contexts(case, synthetic_detections),
                        )
                    )
                    PROGRESS_LOGGER.info(
                        "[foundation-case %d/%d][%s][%s][%s] done elapsed=%.1fs",
                        case_index,
                        len(cases),
                        name,
                        regime.name,
                        case.anomaly_type,
                        time.perf_counter() - case_started,
                    )
                    progress.update(completed=case_index)

                foundation_total = len(foundation_model_configs) * sum(
                    len(contexts) for _, _, contexts in prepared_cases
                )
                foundation_completed = 0
                progress.update(
                    stage="foundation-forecast",
                    detail=f"station={name} regime={regime.name}",
                    completed=0,
                    total=foundation_total,
                    pending=0,
                )

                for model_name, model_config in foundation_model_configs.items():
                    prepared_model: tuple[object, Any, pd.Series, float] | None = None
                    prepare_error: str | None = None
                    for case, synthetic_detections, contexts in prepared_cases:
                        for condition, context in contexts.items():
                            task_number = foundation_completed + 1
                            task_started = time.perf_counter()
                            detection = synthetic_detections.get(condition)
                            context_mask = (
                                detection.mask.reindex(context.index, fill_value=False).astype(bool)
                                if detection is not None
                                else pd.Series(False, index=context.index)
                            )
                            n_injected_detected = int(
                                (context_mask & case.injected_mask).sum()
                            )
                            forecast_key = {
                                **base_key,
                                "stage": "foundation_preprocessing_forecast",
                                "experiment": "foundation-preprocessing-synthetic-v1",
                                "model": model_name,
                                "forecast_model": effective_config(
                                    forecast_model_cache_identity(model_config)
                                ),
                                "model_config": training_model_config,
                                "regime": asdict(regime),
                                "condition": condition,
                                "anomaly_type": case.anomaly_type,
                                "test_seed": case.test_seed,
                                "test_target_start": str(case.origin),
                                "train_fp": train_fp,
                                "context_fp": series_fingerprint(context),
                                "target_fp": series_fingerprint(case.target),
                                "compatibility_imputer": imputer_identity,
                            }
                            result = cache.get(
                                "foundation_preprocessing", forecast_key
                            )
                            cache_state = (
                                "hit"
                                if result is not None
                                else ("miss" if cache.enabled else "off")
                            )
                            if result is None:
                                PROGRESS_LOGGER.info(
                                    "[foundation task=%d/%d][%s][%s][%s][%s][%s] "
                                    "start cache=%s",
                                    task_number,
                                    foundation_total,
                                    name,
                                    regime.name,
                                    model_name,
                                    case.anomaly_type,
                                    condition,
                                    cache_state,
                                )
                                if prepared_model is None and prepare_error is None:
                                    PROGRESS_LOGGER.info(
                                        "[foundation-model][%s][%s][%s] load start",
                                        name,
                                        regime.name,
                                        model_name,
                                    )
                                    try:
                                        prepared_model = prepare_foundation_model(
                                            train_raw,
                                            model_name,
                                            size_k=regime.horizon,
                                            seasonality_m=seasonality_m,
                                            freq=freq,
                                            context_len=context_len,
                                            model_config=model_config,
                                        )
                                    except Exception as exc:
                                        prepare_error = f"{type(exc).__name__}: {exc}"
                                    PROGRESS_LOGGER.info(
                                        "[foundation-model][%s][%s][%s] load done "
                                        "status=%s elapsed=%.1fs",
                                        name,
                                        regime.name,
                                        model_name,
                                        "failed" if prepare_error else "ok",
                                        time.perf_counter() - task_started,
                                    )

                                result = {
                                    "rmse": float("nan"),
                                    "mase": float("nan"),
                                    "model_load_seconds": float("nan"),
                                    "inference_seconds": float("nan"),
                                    "n_test_predictions": 0,
                                    "imputation_applied": False,
                                    "n_imputed": 0,
                                    "failure_reason": prepare_error or "",
                                }
                                if prepared_model is not None:
                                    model, scaler, foundation_insample, load_seconds = (
                                        prepared_model
                                    )
                                    result["model_load_seconds"] = load_seconds
                                    try:
                                        result.update(
                                            forecast_foundation_context(
                                                model,
                                                scaler,
                                                context,
                                                case.target,
                                                foundation_insample,
                                                seasonality_m=seasonality_m,
                                                freq=freq,
                                            )
                                        )
                                    except Exception as first_error:
                                        if context.isna().any():
                                            result["imputation_applied"] = True
                                            try:
                                                filled = impute_series(
                                                    context,
                                                    get_imputer(),
                                                    freq=freq,
                                                    use_scaler=use_scaler,
                                                    max_gap_size=len(context),
                                                )
                                                result["n_imputed"] = int(
                                                    (
                                                        context.isna()
                                                        & filled.notna()
                                                    ).sum()
                                                )
                                                result.update(
                                                    forecast_foundation_context(
                                                        model,
                                                        scaler,
                                                        filled,
                                                        case.target,
                                                        foundation_insample,
                                                        seasonality_m=seasonality_m,
                                                        freq=freq,
                                                    )
                                                )
                                            except Exception as retry_error:
                                                result["failure_reason"] = (
                                                    f"{type(retry_error).__name__}: "
                                                    f"{retry_error}"
                                                )
                                        else:
                                            result["failure_reason"] = (
                                                f"{type(first_error).__name__}: "
                                                f"{first_error}"
                                            )
                                    if _cacheable_foundation_test(
                                        result, regime.horizon
                                    ):
                                        result["failure_reason"] = ""
                                        cache.put(
                                            "foundation_preprocessing",
                                            forecast_key,
                                            result,
                                        )

                            foundation_preprocessing_rows.append(
                                {
                                    "series": name,
                                    "regime": regime.name,
                                    "horizon": regime.horizon,
                                    "model": model_name,
                                    "case_id": case.case_id,
                                    "anomaly_type": case.anomaly_type,
                                    "test_seed": case.test_seed,
                                    "test_target_start": case.origin,
                                    "condition": condition,
                                    "detectors": (
                                        ",".join(detection.detectors)
                                        if detection is not None
                                        else ""
                                    ),
                                    "n_injected": int(case.injected_mask.sum()),
                                    "n_context_flagged": int(context_mask.sum()),
                                    "n_injected_detected": n_injected_detected,
                                    "n_context_nan": int(context.isna().sum()),
                                    "imputation_applied": bool(
                                        result.get("imputation_applied", False)
                                    ),
                                    "imputation_model": (
                                        imputation_model
                                        if result.get("imputation_applied", False)
                                        else "none"
                                    ),
                                    "n_imputed": int(result.get("n_imputed", 0)),
                                    "rmse": result["rmse"] / scale_ref,
                                    "mase": result["mase"],
                                    "model_load_seconds": result.get(
                                        "model_load_seconds", float("nan")
                                    ),
                                    "inference_seconds": result.get(
                                        "inference_seconds", float("nan")
                                    ),
                                    "scale_ref": scale_ref,
                                    "n_test_predictions": result.get(
                                        "n_test_predictions", 0
                                    ),
                                    "failure_reason": result.get(
                                        "failure_reason", ""
                                    ),
                                }
                            )
                            foundation_completed += 1
                            status = (
                                "ok"
                                if _cacheable_foundation_test(result, regime.horizon)
                                else "failed"
                            )
                            PROGRESS_LOGGER.info(
                                "[foundation task=%d/%d completed=%d/%d]"
                                "[%s][%s][%s][%s][%s] done cache=%s status=%s "
                                "load=%.1fs inference=%.1fs wall=%.1fs imputed=%s",
                                task_number,
                                foundation_total,
                                foundation_completed,
                                foundation_total,
                                name,
                                regime.name,
                                model_name,
                                case.anomaly_type,
                                condition,
                                cache_state,
                                status,
                                float(result.get("model_load_seconds", float("nan"))),
                                float(result.get("inference_seconds", float("nan"))),
                                time.perf_counter() - task_started,
                                bool(result.get("imputation_applied", False)),
                            )
                            progress.update(
                                completed=foundation_completed,
                                detail=(
                                    f"station={name} regime={regime.name} "
                                    f"last={model_name}/{case.anomaly_type}/{condition}"
                                ),
                            )
                    if prepared_model is not None:
                        prepared_model = None
                        del model, scaler, foundation_insample
                        _release_cuda_memory()

        PROGRESS_LOGGER.info(
            "[station %d/%d][%s] done backtests=%d foundation_rows=%d elapsed=%.1fs",
            station_index,
            len(series_dfs),
            name,
            len(series_backtest_rows),
            sum(1 for row in foundation_preprocessing_rows if row["series"] == name),
            time.perf_counter() - station_started,
        )

    _shutdown_gpu_resources(cancel_futures=False)
    imputer_ref.clear()
    _release_cuda_memory()
    progress.update(stage="finalizing", detail="building CSV artifacts", pending=0)
    PROGRESS_LOGGER.info("[cache] %s", cache.stats())
    for namespace, counts in cache.stats_by_namespace().items():
        PROGRESS_LOGGER.info(
            "[cache][%s] hits=%d misses=%d",
            namespace,
            counts["hits"],
            counts["misses"],
        )
    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    summary_df = _summarize(results_df)
    detection_df = pd.DataFrame(
        detection_rows,
        columns=[
            "series",
            "strategy",
            "detectors",
            "discarded",
            "n_flagged",
            "n_unscored",
            "coverage_rate",
            "detection_rate",
            "ranking",
            "scope",
            "n_observed",
        ],
    )
    selection_df = pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS)
    foundation_preprocessing_df = pd.DataFrame(
        foundation_preprocessing_rows,
        columns=FOUNDATION_PREPROCESSING_COLUMNS,
    )
    foundation_preprocessing_summary_df = summarize_foundation_preprocessing(
        foundation_preprocessing_df
    )

    results_df.to_csv(output_dir / "results.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    detection_df.to_csv(output_dir / "detection.csv", index=False)
    selection_df.to_csv(output_dir / "selection.csv", index=False)
    foundation_preprocessing_df.to_csv(
        output_dir / "foundation_preprocessing_results.csv", index=False
    )
    foundation_preprocessing_summary_df.to_csv(
        output_dir / "foundation_preprocessing_summary.csv", index=False
    )

    PROGRESS_LOGGER.info(
        "[run] done rows=%d selected_stations=%d foundation_rows=%d elapsed=%.1fs artifacts=%s",
        len(results_df),
        int(selection_df["selected"].sum()) if not selection_df.empty else 0,
        len(foundation_preprocessing_df),
        time.perf_counter() - run_started,
        output_dir,
    )
    progress.close()

    return {
        "output_dir": output_dir,
        "log_path": output_dir / "benchmark.log",
        "arms": arms,
        "results_df": results_df,
        "summary_df": summary_df,
        "detection_df": detection_df,
        "selection_df": selection_df,
        "foundation_preprocessing_df": foundation_preprocessing_df,
        "foundation_preprocessing_summary_df": foundation_preprocessing_summary_df,
    }


def run_benchmark_from_config(
    mask_transforms: Sequence[MaskTransform] | None = None,
) -> dict[str, Any]:
    """Run the configured benchmark and always stop progress resources."""
    global _ACTIVE_PROGRESS
    try:
        return _run_benchmark_from_config(mask_transforms=mask_transforms)
    finally:
        _shutdown_gpu_resources(cancel_futures=True)
        if _ACTIVE_PROGRESS is not None:
            _ACTIVE_PROGRESS.close()
            _ACTIVE_PROGRESS = None


def main() -> None:
    """Execute the benchmark and print the macro summary plus artifact locations."""
    artifacts = run_benchmark_from_config()
    results_df = artifacts["results_df"]
    output_dir = artifacts["output_dir"]

    if results_df.empty:
        print("[info] Benchmark sin filas (revisa los bloques de train/validacion/test).")
    else:
        macro = (
            results_df.groupby(["regime", "arm", "model"], sort=False)[list(METRIC_COLS)]
            .mean()
            .reset_index()
        )
        print("[info] Error medio por brazo (compara cada estrategia contra 'raw')")
        print(macro.to_string(index=False))
    print(f"[info] Artefactos en {output_dir}")


if __name__ == "__main__":
    main()
