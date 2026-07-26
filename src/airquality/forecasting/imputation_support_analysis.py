"""Separate audit of train-eligible support gained through imputation.

This experiment does not alter the forecasting benchmark or ``results.csv``.
It uses the benchmark's configured detection strategies, actual imputer and
single common holdout per series, then applies one conservative block threshold:
the largest ``min_train_series_length`` among configured models/regimes that can
run on imputed arms.

Run with::

    uv run python -m airquality.forecasting.imputation_support_analysis
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from airquality.anomaly.registry import resolve_model_names
from airquality.config import (
    cfg_get_bool,
    cfg_get_csv_list,
    cfg_get_float,
    cfg_get_int,
    cfg_get_str,
)
from airquality.data.segments import contiguous_observed_segments
from airquality.data.series import ensure_datetime_series
from airquality.forecasting.backtest import (
    get_forecast_model_requirements,
    select_holdout_window,
)
from airquality.forecasting.cache import (
    CACHE_VERSION,
    BenchmarkCache,
    artifact_fingerprint,
    effective_config,
    series_fingerprint,
)
from airquality.forecasting.cleaning import remove_anomalies
from airquality.forecasting.detection import (
    DEFAULT_INJECTION_SEED,
    DEFAULT_MIN_SELECTION_POINTS,
    DEFAULT_VOTE_MIN_VOTES,
    DEFAULT_VOTE_TOP_K,
    MIN_SEGMENT_POINTS,
    DetectionResult,
    SeriesDetectionContext,
    build_detection_strategy,
)
from airquality.forecasting.fill import (
    DEFAULT_MAX_GAP_SIZE,
    _repo_root,
    _resolve_tspulse_model_path,
    build_imputer,
    impute_series,
)
from airquality.forecasting.pipeline import (
    DEFAULT_STRATEGIES,
    ForecastRegime,
    _load_raw_hourly_series,
)
from airquality.forecasting.registry import resolve_forecasting_model_configs
from airquality.imputation.registry import DARTS_GLOBAL, TSPULSE, resolve_imputer_family
from airquality.paths import create_run_dir

ANALYSIS_VERSION = 1
SUPPORT_COLUMNS = (
    "series",
    "arm",
    "strategy",
    "imputation_model",
    "selected",
    "exclusion_reason",
    "worst_case_min_train_points",
    "worst_case_sources",
    "series_start",
    "series_end",
    "train_start",
    "train_end",
    "holdout_start",
    "holdout_end",
    "holdout_age_hours",
    "detectors",
    "n_detected_anomalies_before_holdout",
    "n_observed_before_imputation",
    "n_imputed_before_holdout",
    "n_imputed_anomalies_before_holdout",
    "n_imputed_preexisting_gaps_before_holdout",
    "n_eligible_blocks_before_imputation",
    "n_eligible_blocks_after_imputation",
    "n_eligible_points_before_imputation",
    "n_eligible_points_after_imputation",
    "n_eligible_points_added",
    "n_observed_points_recovered",
    "n_recovered_blocks",
    "n_imputed_in_eligible_blocks",
    "n_imputed_anomalies_in_eligible_blocks",
    "n_imputed_preexisting_gaps_in_eligible_blocks",
    "imputed_eligible_age_hours_median",
    "imputed_eligible_age_hours_max",
    "recovered_observed_age_hours_median",
    "recovered_observed_age_hours_max",
)
def _eligible_support(
    series: pd.Series, minimum: int
) -> tuple[list[pd.Series], pd.DatetimeIndex]:
    segments = contiguous_observed_segments(series, min_len=minimum)
    index = pd.DatetimeIndex(
        [timestamp for segment in segments for timestamp in segment.index]
    )
    return segments, index


def _age_stats(
    index: pd.DatetimeIndex, holdout_start: pd.Timestamp
) -> tuple[float, float]:
    if index.empty:
        return float("nan"), float("nan")
    ages = (holdout_start - index) / pd.Timedelta(hours=1)
    return float(np.median(ages)), float(np.max(ages))


def training_support_diagnostics(
    raw_train: pd.Series,
    cleaned_train: pd.Series,
    imputed_train: pd.Series,
    anomaly_mask: pd.Series,
    *,
    holdout_start: pd.Timestamp,
    min_train_points: int,
) -> dict[str, int | float]:
    """Compare worst-case block eligibility before and after actual imputation."""
    if min_train_points <= 0:
        raise ValueError("min_train_points debe ser positivo")

    name = str(raw_train.name or "series")
    raw = ensure_datetime_series(raw_train, freq="h", name=name)
    cleaned = ensure_datetime_series(cleaned_train, freq="h", name=name).reindex(raw.index)
    imputed = ensure_datetime_series(imputed_train, freq="h", name=name).reindex(raw.index)
    mask = anomaly_mask.reindex(raw.index, fill_value=False).astype(bool)
    if len(raw) and raw.index[-1] >= holdout_start:
        raise ValueError("El historial de train debe terminar antes del holdout")

    before_segments, before_index = _eligible_support(cleaned, min_train_points)
    after_segments, after_index = _eligible_support(imputed, min_train_points)
    imputed_index = raw.index[cleaned.isna() & imputed.notna()]
    imputed_eligible = after_index.intersection(imputed_index)
    added_index = after_index.difference(before_index)
    recovered_observed = added_index.intersection(cleaned.index[cleaned.notna()])
    imputed_anomalies = imputed_index.intersection(raw.index[raw.notna() & mask])
    imputed_preexisting = imputed_index.intersection(raw.index[raw.isna()])

    if len(added_index) != len(imputed_eligible) + len(recovered_observed):
        raise RuntimeError("La ganancia de soporte no cuadra con imputados + observados")

    imputed_age_median, imputed_age_max = _age_stats(imputed_eligible, holdout_start)
    recovered_age_median, recovered_age_max = _age_stats(
        recovered_observed, holdout_start
    )
    return {
        "n_detected_anomalies_before_holdout": int((mask & raw.notna()).sum()),
        "n_observed_before_imputation": int(cleaned.notna().sum()),
        "n_imputed_before_holdout": len(imputed_index),
        "n_imputed_anomalies_before_holdout": len(imputed_anomalies),
        "n_imputed_preexisting_gaps_before_holdout": len(imputed_preexisting),
        "n_eligible_blocks_before_imputation": len(before_segments),
        "n_eligible_blocks_after_imputation": len(after_segments),
        "n_eligible_points_before_imputation": len(before_index),
        "n_eligible_points_after_imputation": len(after_index),
        "n_eligible_points_added": len(added_index),
        "n_observed_points_recovered": len(recovered_observed),
        "n_recovered_blocks": sum(
            not pd.DatetimeIndex(segment.index)
            .intersection(recovered_observed)
            .empty
            for segment in after_segments
        ),
        "n_imputed_in_eligible_blocks": len(imputed_eligible),
        "n_imputed_anomalies_in_eligible_blocks": len(
            imputed_eligible.intersection(imputed_anomalies)
        ),
        "n_imputed_preexisting_gaps_in_eligible_blocks": len(
            imputed_eligible.intersection(imputed_preexisting)
        ),
        "imputed_eligible_age_hours_median": imputed_age_median,
        "imputed_eligible_age_hours_max": imputed_age_max,
        "recovered_observed_age_hours_median": recovered_age_median,
        "recovered_observed_age_hours_max": recovered_age_max,
    }


def _common_support(
    series: pd.Series, detections: dict[str, DetectionResult]
) -> pd.Series:
    excluded = pd.Series(False, index=series.index)
    for detection in detections.values():
        excluded |= detection.mask.reindex(series.index, fill_value=False).astype(bool)
        if detection.scored_mask is not None:
            scored = detection.scored_mask.reindex(series.index, fill_value=False).astype(bool)
            excluded |= series.notna() & ~scored
    return series.mask(excluded)


def _imputer_identity(
    model_name: str, *, size_k: int, max_gap_size: int
) -> dict[str, Any]:
    family = resolve_imputer_family(model_name)
    config: dict[str, Any] = {
        "model": model_name,
        "family": family,
        "size_k": size_k,
        "max_gap_size": max_gap_size,
    }
    artifacts: dict[str, str | None] = {}
    if family == DARTS_GLOBAL:
        weights = _repo_root() / "models" / f"{model_name}_k{size_k}.pt"
        artifacts = {
            "model": artifact_fingerprint(weights),
            "checkpoint": artifact_fingerprint(Path(f"{weights}.ckpt")),
        }
    elif family == TSPULSE:
        model_path = _resolve_tspulse_model_path(model_name)
        model_id = cfg_get_str(
            "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
        )
        config["tspulse"] = effective_config(
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
    return {"config": config, "artifacts": artifacts}


def run_analysis() -> dict[str, Any]:
    """Run the separate config-driven support audit and write one CSV."""
    freq = cfg_get_str("data", "freq", "h")
    pollutant = cfg_get_str("forecasting", "pollutant", "NO2")
    raw_base_dir = cfg_get_str(
        "forecasting", "raw_base_dir", "data/raw/datos_estaciones_5m"
    )
    holdout = cfg_get_int("forecasting", "holdout", 192)
    context_len = cfg_get_int("forecasting", "context_len", 72)
    seasonality_m = cfg_get_int("benchmark", "seasonality_m", 24)
    imputation_size_k = cfg_get_int("benchmark", "size_k", 5)
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

    model_names = list(
        cfg_get_csv_list("forecasting", "forecast_models", ("NLinear", "TiDE"))
    )
    model_configs = resolve_forecasting_model_configs(
        model_names,
        seasonality_m=seasonality_m,
        context_length=context_len,
    )
    requirements = [
        (
            model_name,
            regime,
            config,
            get_forecast_model_requirements(
                config,
                size_k=regime.horizon,
                seasonality_m=seasonality_m,
                context_len=context_len,
            ),
        )
        for regime in regimes
        for model_name, config in model_configs.items()
    ]
    imputed_requirements = [item for item in requirements if not item[2].raw_only]
    if not imputed_requirements:
        raise ValueError("No hay modelos configurados que admitan brazos imputados")
    worst_minimum = max(item[3].min_train_series_length for item in imputed_requirements)
    worst_sources = ",".join(
        f"{model_name}:{regime.name}"
        for model_name, regime, _config, native in imputed_requirements
        if native.min_train_series_length == worst_minimum
    )

    context_requirement = max(
        context_len, *(native.prediction_context_length for *_, native in requirements)
    )
    train_requirement = max(native.min_train_series_length for *_, native in requirements)
    host_requirement = max(
        native.min_train_series_length
        + (
            max(regime.validation_len, native.validation_target_length)
            if native.validation_target_offset is not None
            else 0
        )
        for _model_name, regime, _config, native in requirements
    )

    strategy_specs = [
        spec.strip().lower()
        for spec in cfg_get_csv_list("forecasting", "strategies", DEFAULT_STRATEGIES)
    ]
    threshold_k = cfg_get_float("forecasting", "threshold_k", 3.5)
    max_detection_rate = cfg_get_float("forecasting", "max_detection_rate", 0.07)
    vote_top_k = cfg_get_int("forecasting", "vote_top_k", DEFAULT_VOTE_TOP_K)
    vote_min_votes = cfg_get_int(
        "forecasting", "vote_min_votes", DEFAULT_VOTE_MIN_VOTES
    )
    strategies = [
        build_detection_strategy(
            spec,
            threshold_k=threshold_k,
            max_detection_rate=max_detection_rate,
            vote_top_k=vote_top_k,
            vote_min_votes=vote_min_votes,
        )
        for spec in dict.fromkeys(strategy_specs)
    ]
    if not strategies:
        raise ValueError("No hay estrategias de deteccion configuradas")

    seed = cfg_get_int("forecasting", "seed", 13)
    device = cfg_get_str("forecasting", "device", "cpu")
    injection_seed = cfg_get_int(
        "forecasting", "injection_seed", DEFAULT_INJECTION_SEED
    )
    min_selection_points = cfg_get_int(
        "forecasting", "min_selection_points", DEFAULT_MIN_SELECTION_POINTS
    )
    if threshold_k < 0:
        raise ValueError("threshold_k no puede ser negativo")
    if not 0.0 <= max_detection_rate <= 1.0:
        raise ValueError("max_detection_rate debe estar entre 0 y 1")
    if min_selection_points < MIN_SEGMENT_POINTS:
        raise ValueError(
            f"min_selection_points debe ser al menos {MIN_SEGMENT_POINTS}"
        )
    detectors = resolve_model_names(
        list(cfg_get_csv_list("forecasting", "detectors", ("all",)))
    )
    imputation_model = cfg_get_str("forecasting", "imputation_model", "TSPulse")
    max_imputation_gap = cfg_get_int(
        "forecasting", "max_imputation_gap", DEFAULT_MAX_GAP_SIZE
    )
    if max_imputation_gap < 1:
        raise ValueError("max_imputation_gap debe ser positivo")
    imputer_key = _imputer_identity(
        imputation_model,
        size_k=imputation_size_k,
        max_gap_size=max_imputation_gap,
    )
    use_scaler = imputation_model not in ("interp", "LinearInterp")
    use_cache = cfg_get_bool("forecasting", "use_cache", True)
    cache_dir = cfg_get_str("forecasting", "cache_dir", "reports/forecasting/cache")
    cache = BenchmarkCache((_repo_root() / cache_dir) if use_cache else None)
    imputer_ref: list[Any] = []

    def get_imputer() -> Any:
        if not imputer_ref:
            imputer_ref.append(
                build_imputer(imputation_model, freq=freq, size_k=imputation_size_k)
            )
        return imputer_ref[0]

    rows: list[dict[str, Any]] = []
    series_dfs = _load_raw_hourly_series(
        pollutant=pollutant,
        raw_base_dir=raw_base_dir,
        freq=freq,
    )
    if not series_dfs:
        raise RuntimeError("No se cargaron series para el analisis")

    for frame in series_dfs:
        series = frame.iloc[:, 0]
        name = str(series.name)
        series_fp = series_fingerprint(series)
        context = SeriesDetectionContext(
            series,
            detectors=detectors,
            seed=seed,
            device=device,
            freq=freq,
            injection_seed=injection_seed,
            min_selection_points=min_selection_points,
            cache=cache,
            cache_key={
                "version": CACHE_VERSION,
                "series": name,
                "series_fp": series_fp,
                "freq": freq,
                "seed": seed,
            },
        )
        detections = {strategy.name: strategy.detect(context) for strategy in strategies}
        support = _common_support(series, detections)
        window = select_holdout_window(
            support,
            holdout=holdout,
            context_len=context_requirement,
            train_min_len=train_requirement,
            validation_len=max(regime.validation_len for regime in regimes),
            freq=freq,
            host_min_len=host_requirement,
        )
        common = {
            "series": name,
            "imputation_model": imputation_model,
            "worst_case_min_train_points": worst_minimum,
            "worst_case_sources": worst_sources,
            "series_start": series.index.min(),
            "series_end": series.index.max(),
        }
        if window is None:
            rows.extend(
                {
                    **common,
                    "arm": f"{strategy.name}+impute",
                    "strategy": strategy.name,
                    "selected": False,
                    "exclusion_reason": "no_common_fixed_holdout_and_training_host",
                }
                for strategy in strategies
            )
            continue

        train_raw = series.loc[window["train_index"]]
        holdout_start = window["holdout_start"]
        for strategy in strategies:
            detection = detections[strategy.name]
            cleaned = remove_anomalies(train_raw, detection)
            support_key = {
                "analysis": "imputation-support",
                "analysis_version": ANALYSIS_VERSION,
                "train_fp": series_fingerprint(train_raw),
                "cleaned_fp": series_fingerprint(cleaned),
                "strategy": asdict(strategy),
                "imputer": imputer_key,
                "min_train_points": worst_minimum,
                "holdout_start": str(holdout_start),
            }
            metrics = cache.get("imputation_support", support_key)
            if metrics is None:
                imputed = impute_series(
                    cleaned,
                    get_imputer(),
                    freq=freq,
                    use_scaler=use_scaler,
                    max_gap_size=max_imputation_gap,
                )
                metrics = training_support_diagnostics(
                    train_raw,
                    cleaned,
                    imputed,
                    detection.mask,
                    holdout_start=holdout_start,
                    min_train_points=worst_minimum,
                )
                cache.put("imputation_support", support_key, metrics)
            rows.append(
                {
                    **common,
                    "arm": f"{strategy.name}+impute",
                    "strategy": strategy.name,
                    "selected": True,
                    "exclusion_reason": "",
                    "train_start": train_raw.index.min(),
                    "train_end": train_raw.index.max(),
                    "holdout_start": holdout_start,
                    "holdout_end": window["holdout_end"],
                    "holdout_age_hours": int(
                        (series.index.max() - window["holdout_end"])
                        / pd.Timedelta(hours=1)
                    ),
                    "detectors": ",".join(detection.detectors),
                    **metrics,
                }
            )

    output_dir = create_run_dir(
        _repo_root() / "reports" / "forecasting_support",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    results = pd.DataFrame(rows).reindex(columns=SUPPORT_COLUMNS)
    results.to_csv(output_dir / "training_support.csv", index=False)
    print(f"[cache] {cache.stats()}")
    print(f"[info] Diagnostico de soporte en {output_dir}")
    return {"output_dir": output_dir, "results_df": results}


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
