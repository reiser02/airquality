"""Config-driven forecasting benchmark over anomaly-cleaning arms.

Measures whether anomaly detection (and the subsequent imputation) improves
    multi-step forecasting. For every configured series the pipeline builds one
training *arm* per (detection strategy, imputation) combination — plus the
    ``raw`` baseline — and backtests the same forecasting models on each arm over
    the **same** observed holdout window in short (8 h, stride 4 h) and long
    (48 h, stride 24 h) regimes:

- ``raw``: the hourly-mean series as loaded (gaps + anomalies kept).
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

Detection and imputation touch only the training portion; the evaluation
window (context + holdout) stays the raw observed values for every arm, so
forecast error differences reflect only the preprocessing of the training data.

Detections and backtests are cached on disk (:mod:`airquality.forecasting.cache`,
``[forecasting] use_cache`` / ``cache_dir``): an interrupted run resumes where
it stopped and re-runs only recompute what changed. Figures are rendered
separately from the persisted CSVs, so they never trigger recomputation.

Run with::

    uv run python -m airquality.forecasting.pipeline
    uv run python -m airquality.forecasting.plot_benchmark_results [run_dir]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
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
from airquality.data.io import load_and_normalize_series
from airquality.forecasting.backtest import backtest_forecast, select_holdout_window
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
    DEFAULT_INJECTION_SEED,
    DEFAULT_MIN_SELECTION_POINTS,
    DEFAULT_VOTE_MIN_VOTES,
    DEFAULT_VOTE_TOP_K,
    DetectionResult,
    DetectionStrategy,
    MaskTransform,
    SeriesDetectionContext,
    apply_mask_transforms,
    build_detection_strategy,
)
from airquality.forecasting.fill import _repo_root, build_imputer, impute_series
from airquality.forecasting.registry import (
    forecast_model_cache_identity,
    resolve_forecasting_model_configs,
)
from airquality.imputation.registry import DARTS_GLOBAL, TSPULSE, resolve_imputer_family
from airquality.paths import create_run_dir

RAW_ARM = "raw"
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


@dataclass(frozen=True)
class ForecastArm:
    """One benchmark arm: how the training series is preprocessed before fitting."""

    name: str
    strategy: str | None  # detection strategy spec; None = raw baseline
    impute: bool


@dataclass(frozen=True)
class ForecastRegime:
    """Forecast horizon, rolling cadence, and validation target span."""

    name: str
    horizon: int
    stride: int
    validation_len: int


def build_arms(strategies: Sequence[str], imputation: str) -> list[ForecastArm]:
    """Expand strategy specs into benchmark arms; ``raw`` is always first.

    ``imputation`` picks the variants built per strategy: ``impute`` (detect →
    remove → impute), ``none`` (detect → remove, gaps stay NaN) or ``both``.
    """
    if imputation not in IMPUTATION_CHOICES:
        raise ValueError(
            f"Valor de imputacion desconocido: '{imputation}'. Usa uno de {IMPUTATION_CHOICES}"
        )
    arms = [ForecastArm(RAW_ARM, None, False)]
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
    train_raw: pd.Series,
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
        detection = cache.get("detection", key)
        if detection is None:
            if context is None:
                context = SeriesDetectionContext(train_raw, **context_kwargs)
            detection = apply_mask_transforms(train_raw, strategy.detect(context), mask_transforms)
            cache.put("detection", key, detection)
        detections[strategy.name] = detection
    return detections


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
        return pd.DataFrame()

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


def run_benchmark_from_config(
    mask_transforms: Sequence[MaskTransform] | None = None,
) -> dict[str, Any]:
    """Run the multi-arm forecasting benchmark defined by the config.

    ``mask_transforms`` post-process every strategy's anomaly mask (in order)
    before removal — the hook for future preprocessing of the detection output.
    """
    freq = cfg_get_str("data", "freq", "h")
    imputation_size_k = cfg_get_int("benchmark", "size_k", 5)
    seasonality_m = cfg_get_int("benchmark", "seasonality_m", 24)

    holdout = cfg_get_int("forecasting", "holdout", 192)
    context_len = cfg_get_int("forecasting", "context_len", 72)
    test_alignment = cfg_get_int("forecasting", "test_alignment", 48)
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
    if any(
        min(regime.horizon, regime.stride, regime.validation_len) <= 0
        or regime.validation_len < regime.horizon
        for regime in regimes
    ):
        raise ValueError("Horizonte, stride y validacion de cada regimen deben ser validos")
    host_requirement = max(
        context_len + regime.horizon + regime.validation_len for regime in regimes
    )
    seed = cfg_get_int("forecasting", "seed", 13)
    device = cfg_get_str("forecasting", "device", "cpu")
    threshold_k = cfg_get_float("forecasting", "threshold_k", 3.5)
    max_detection_rate = cfg_get_float("forecasting", "max_detection_rate", 0.07)
    detectors = list(cfg_get_csv_list("forecasting", "detectors", ("all",)))
    imputation_model = cfg_get_str("forecasting", "imputation_model", "interp")
    forecast_models = list(
        cfg_get_csv_list("forecasting", "forecast_models", ("NLinear", "TiDE"))
    )
    forecast_model_configs = resolve_forecasting_model_configs(
        forecast_models,
        seasonality_m=seasonality_m,
        context_length=context_len,
    )
    forecast_models = list(forecast_model_configs)
    strategy_specs = [
        spec.strip().lower()
        for spec in cfg_get_csv_list("forecasting", "strategies", DEFAULT_STRATEGIES)
    ]
    imputation = cfg_get_str("forecasting", "imputation", "both").strip().lower()
    injection_seed = cfg_get_int("forecasting", "injection_seed", DEFAULT_INJECTION_SEED)
    min_selection_points = cfg_get_int(
        "forecasting", "min_selection_points", DEFAULT_MIN_SELECTION_POINTS
    )
    vote_top_k = cfg_get_int("forecasting", "vote_top_k", DEFAULT_VOTE_TOP_K)
    vote_min_votes = cfg_get_int("forecasting", "vote_min_votes", DEFAULT_VOTE_MIN_VOTES)
    use_cache = cfg_get_bool("forecasting", "use_cache", True)
    cache_dir = cfg_get_str("forecasting", "cache_dir", "reports/forecasting/cache")
    cache = BenchmarkCache((_repo_root() / cache_dir) if use_cache else None)

    arms = build_arms(strategy_specs, imputation)
    if all(config.raw_only for config in forecast_model_configs.values()):
        arms = arms[:1]
    strategies = [
        build_detection_strategy(
            spec,
            threshold_k=threshold_k,
            max_detection_rate=max_detection_rate,
            vote_top_k=vote_top_k,
            vote_min_votes=vote_min_votes,
        )
        for spec in dict.fromkeys(arm.strategy for arm in arms if arm.strategy)
    ]

    series_dfs = load_and_normalize_series(freq=freq, name_from_path=True)
    if not series_dfs:
        raise RuntimeError("No se cargaron series para el benchmark.")

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
            section: dict(runtime_config.items(section)) if runtime_config.has_section(section) else {}
            for section in ("training", "models")
        }
    )
    cache_config = effective_config(
        {
            "freq": freq,
            "holdout": holdout,
            "context_len": context_len,
            "test_alignment": test_alignment,
            "regimes": [asdict(regime) for regime in regimes],
            "seed": seed,
            "device": device,
            "threshold_k": threshold_k,
            "max_detection_rate": max_detection_rate,
            "detectors": sorted(resolved_detectors),
            "injection_seed": injection_seed,
            "min_selection_points": min_selection_points,
            "vote_top_k": vote_top_k,
            "vote_min_votes": vote_min_votes,
        }
    )

    imputer_identity: dict[str, Any] | None = None
    if any(arm.impute for arm in arms):
        family = resolve_imputer_family(imputation_model)
        artifacts: dict[str, str | None] = {}
        imputer_config: dict[str, Any] = {
            "model": imputation_model,
            "family": family,
            "size_k": imputation_size_k,
        }
        if family == DARTS_GLOBAL:
            weights = _repo_root() / "models" / f"{imputation_model}_k{imputation_size_k}.pt"
            artifacts = {
                "model": artifact_fingerprint(weights),
                "checkpoint": artifact_fingerprint(Path(f"{weights}.ckpt")),
            }
        elif family == TSPULSE:
            model_path = cfg_get_str("benchmark", "tspulse_model_path", "").strip()
            model_id = cfg_get_str(
                "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
            )
            imputer_config["tspulse"] = effective_config(
                {
                    "model_path": model_path or None,
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

    rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    for df in series_dfs:
        series = df.iloc[:, 0]
        name = str(series.name)
        window = select_holdout_window(
            series,
            holdout=holdout,
            context_len=context_len,
            train_min_len=context_len + max(regime.horizon for regime in regimes),
            validation_len=max(regime.validation_len for regime in regimes),
            test_alignment=test_alignment,
            freq=freq,
            host_min_len=host_requirement,
        )
        if window is None:
            print(
                f"[skip] {name}: falta un bloque de test >= {context_len + holdout} h "
                f"o un bloque anterior de train+validacion >= "
                f"{host_requirement} h"
            )
            continue

        holdout_start = window["holdout_start"]
        eval_obs = series.loc[window["eval_index"]]
        train_raw = series.loc[window["train_index"]]

        # One scale factor per series, from the RAW observed training values and
        # shared by every arm: rmse_scaled = rmse / scale_ref is the RMSE on
        # standardized data, comparable across series of different levels.
        scale_ref = float(train_raw.std())
        if not math.isfinite(scale_ref) or scale_ref <= 0.0:
            scale_ref = float("nan")

        # Cache keys embed the exact training data + every knob that shapes the
        # stage, so interrupted runs resume and config/data edits recompute.
        base_key = {
            "version": CACHE_VERSION,
            "series": name,
            "train_fp": series_fingerprint(train_raw),
            "config": cache_config,
            "freq": freq,
            "detectors": sorted(resolved_detectors),
            "seed": seed,
            "injection_seed": injection_seed,
            "min_selection_points": min_selection_points,
            "transforms": transform_names,
            "test_alignment": test_alignment,
            "regimes": [asdict(regime) for regime in regimes],
        }
        detections = _detect_for_strategies(
            train_raw,
            strategies,
            mask_transforms,
            cache=cache,
            base_key=base_key,
            context_kwargs={
                "detectors": resolved_detectors,
                "seed": seed,
                "device": device,
                "freq": freq,
                "injection_seed": injection_seed,
                "min_selection_points": min_selection_points,
                "cache": cache,
                "cache_key": {
                    "version": CACHE_VERSION,
                    "series": name,
                    "train_fp": base_key["train_fp"],
                    "freq": freq,
                    "seed": seed,
                },
            },
        )
        for spec, detection in detections.items():
            print(
                f"[info] {name}/{spec}: detectores={','.join(detection.detectors) or 'none'} "
                f"descartados={','.join(detection.discarded) or 'none'} "
                f"tasa={detection.detection_rate:.2%} anomalias={detection.n_flagged}"
            )
            detection_rows.append(
                {
                    "series": name,
                    "strategy": spec,
                    "detectors": ",".join(detection.detectors),
                    "discarded": ",".join(detection.discarded),
                    "n_flagged": detection.n_flagged,
                    "detection_rate": detection.detection_rate,
                    "ranking": _format_ranking(detection.ranking),
                }
            )

        # Arm training series are built lazily: fully-cached arms skip anomaly
        # removal AND imputation entirely (that is what makes resume cheap).
        train_by_arm: dict[str, pd.Series] = {}

        def train_for(arm: ForecastArm) -> pd.Series:
            if arm.name not in train_by_arm:
                base = (
                    train_raw
                    if arm.strategy is None
                    else remove_anomalies(train_raw, detections[arm.strategy])
                )
                train_by_arm[arm.name] = (
                    impute_series(base, get_imputer(), freq=freq, use_scaler=use_scaler)
                    if arm.impute
                    else base
                )
            return train_by_arm[arm.name]

        eval_fp = series_fingerprint(eval_obs)
        for regime in regimes:
            for arm in arms:
                detection = detections.get(arm.strategy) if arm.strategy else None
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
                    "n_anomalies": detection.n_flagged if detection else 0,
                    "test_block_start": str(window["test_block_start"]),
                    "holdout_start": str(holdout_start),
                    "holdout_end": str(window["holdout_end"]),
                    "test_hours": window["test_hours"],
                }
                arm_strategy_key = (
                    asdict(strategy_by_spec[arm.strategy]) if arm.strategy else None
                )
                for model_name in forecast_models:
                    forecast_model_config = forecast_model_configs[model_name]
                    if forecast_model_config.raw_only and arm.name != RAW_ARM:
                        continue
                    backtest_key = {
                        **base_key,
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
                        "holdout_start": str(holdout_start),
                        "eval_fp": eval_fp,
                    }
                    res = cache.get("backtest", backtest_key)
                    if res is None:
                        res = backtest_forecast(
                            train_for(arm),
                            eval_obs,
                            model_name,
                            size_k=regime.horizon,
                            holdout_start=holdout_start,
                            seasonality_m=seasonality_m,
                            freq=freq,
                            validation_len=regime.validation_len,
                            validation_stride=regime.stride,
                            forecast_stride=regime.stride,
                            context_len=context_len,
                            model_config=forecast_model_config,
                            # Shared raw history: every arm's MASE uses the SAME
                            # seasonal-naive denominator (see METRIC_COLS).
                            mase_insample=train_raw,
                        )
                        cache.put("backtest", backtest_key, res)
                    rows.append(
                        {
                            **common,
                            "model": model_name,
                            "model_mode": forecast_model_config.mode,
                            # RMSE scaled at row-assembly (the cache keeps raw-unit
                            # errors, so entries stay valid if scaling evolves).
                            "rmse": res["rmse"] / scale_ref,
                            "mase": res["mase"],
                            "train_seconds": res.get("train_seconds", float("nan")),
                            "inference_seconds": res.get("inference_seconds", float("nan")),
                            "scale_ref": scale_ref,
                            "n_eval": res["n_eval"],
                            "n_forecasts": res.get("n_forecasts", 0),
                            "n_unique_targets": res.get("n_unique_targets", 0),
                            "origin_mae_mean": res.get("origin_mae_mean", float("nan")),
                            "origin_mae_std": res.get("origin_mae_std", float("nan")),
                            "origin_rmse_mean": res.get("origin_rmse_mean", float("nan")),
                            "origin_rmse_std": res.get("origin_rmse_std", float("nan")),
                        }
                    )

    print(f"[cache] {cache.stats()}")
    results_df = pd.DataFrame(rows)
    summary_df = _summarize(results_df)
    detection_df = pd.DataFrame(
        detection_rows,
        columns=[
            "series",
            "strategy",
            "detectors",
            "discarded",
            "n_flagged",
            "detection_rate",
            "ranking",
        ],
    )

    output_dir = _build_output_dir()
    results_df.to_csv(output_dir / "results.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    detection_df.to_csv(output_dir / "detection.csv", index=False)

    return {
        "output_dir": output_dir,
        "arms": arms,
        "results_df": results_df,
        "summary_df": summary_df,
        "detection_df": detection_df,
    }


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
