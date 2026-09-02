"""Paired statistical comparison of forecasting models and preprocessing arms."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from airquality.forecasting.pipeline import RAW_ARM, RAW_FROZEN_ARM


METRICS = ("mase", "rmsse")
MODEL_MODES = ("trained", "local", "foundation")
KEY_COLUMNS = ("series", "arm", "model")
PROTOCOL_COLUMNS = (
    "horizon",
    "forecast_stride",
    "validation_len",
    "validation_stride",
)
WINDOW_COLUMNS = (
    "test_context_start",
    "test_target_start",
    "test_target_end",
    "test_target_hours",
)
OUTPUT_FILES = {
    "model_station_scores": "model_station_scores.csv",
    "model_omnibus": "model_omnibus.csv",
    "model_pairwise": "model_pairwise.csv",
    "branch_station_scores": "branch_station_scores.csv",
    "branch_omnibus": "branch_omnibus.csv",
    "branch_pairwise": "branch_pairwise.csv",
    "statistical_exclusions": "statistical_exclusions.csv",
}
SCORE_COLUMNS = (
    "analysis", "scope", "metric", "series", "treatment", "score",
    "mean_error", "n_averaged", "rank", "seed",
)
OMNIBUS_COLUMNS = (
    "analysis", "scope", "metric", "test", "statistic", "p_value", "alpha",
    "significant", "confirmatory", "n_stations", "n_treatments",
    "n_permutations", "seed",
)
PAIRWISE_COLUMNS = (
    "analysis", "scope", "metric", "family", "confirmatory", "left", "right",
    "n_stations", "mean_difference", "median_difference", "error_ratio",
    "improvement_pct", "ci_ratio_low", "ci_ratio_high", "ci_difference_low",
    "ci_difference_high", "rank_biserial", "wins", "ties", "losses",
    "wilcoxon_statistic", "p_value", "sign_p_value", "p_holm", "sign_p_holm",
    "p_joint", "alpha", "n_bootstrap", "omnibus_significant",
    "typical_station_outcome", "seed",
)
EXCLUSION_COLUMNS = ("analysis", "scope", "metric", "series", "reason", "seed")


def _validate_results(results: pd.DataFrame) -> None:
    required = {
        *KEY_COLUMNS,
        "model_mode",
        *METRICS,
        *PROTOCOL_COLUMNS,
        *WINDOW_COLUMNS,
        "n_test_predictions",
        "n_forecasts",
        "n_expected_forecasts",
        "n_unique_targets",
        "status",
        "failure_reason",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Faltan columnas en results.csv: {', '.join(missing)}")
    if results.empty:
        return
    statuses = results["status"].astype(str)
    if not statuses.isin(("ok", "failed")).all():
        raise ValueError("status contiene valores desconocidos")
    if statuses.eq("failed").any():
        failed = results.loc[statuses.eq("failed"), list(KEY_COLUMNS)].head(3)
        examples = ", ".join("/".join(row) for row in failed.astype(str).to_numpy())
        raise ValueError(
            "El run contiene backtests fallidos y debe reintentarse antes del "
            f"analisis estadistico: {examples}"
        )
    if results[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("Las claves series, arm y model no pueden contener nulos")
    if results.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Hay filas duplicadas para series, arm y model")
    modes = results["model_mode"]
    if modes.isna().any() or not modes.isin(MODEL_MODES).all():
        raise ValueError("model_mode contiene valores desconocidos")
    if (results.groupby("model")["model_mode"].nunique() != 1).any():
        raise ValueError("Cada modelo debe tener un unico model_mode")
    source_arms = {RAW_ARM, RAW_FROZEN_ARM}
    if not source_arms.issubset(set(results["arm"])):
        raise ValueError("results.csv debe contener los brazos raw y raw+frozen")
    model_arms = results.groupby("model")["arm"].agg(set)
    mode_by_model = results.groupby("model")["model_mode"].first()
    nonfoundation = mode_by_model[mode_by_model != "foundation"].index
    if len(nonfoundation):
        expected_arms = model_arms.loc[nonfoundation[0]]
        if not source_arms.issubset(expected_arms):
            raise ValueError("Los modelos no-foundation deben incluir raw y raw+frozen")
        if any(model_arms.loc[model] != expected_arms for model in nonfoundation):
            raise ValueError("Los modelos no-foundation no comparten los mismos brazos")
    foundation = mode_by_model[mode_by_model == "foundation"].index
    if any(model_arms.loc[model] != source_arms for model in foundation):
        raise ValueError("Los modelos foundation deben usar solo raw y raw+frozen")
    if any(results[column].nunique(dropna=False) != 1 for column in PROTOCOL_COLUMNS):
        raise ValueError("Los resultados no comparten el mismo protocolo temporal")
    protocol: dict[str, int] = {}
    for column in PROTOCOL_COLUMNS:
        value = float(pd.to_numeric(results[column], errors="raise").iloc[0])
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("La geometria del protocolo temporal no es valida")
        protocol[column] = int(value)
    horizon = protocol["horizon"]
    stride = protocol["forecast_stride"]
    validation_len = protocol["validation_len"]
    validation_stride = protocol["validation_stride"]
    if (
        min(protocol.values()) <= 0
        or stride > horizon
        or validation_stride != stride
        or validation_len < horizon
        or (validation_len - horizon) % stride != 0
    ):
        raise ValueError("La geometria del protocolo temporal no es valida")
    inconsistent = results.groupby("series", dropna=False)[list(WINDOW_COLUMNS)].nunique(
        dropna=False
    )
    if (inconsistent > 1).any().any():
        raise ValueError("Las ramas no comparten la misma ventana temporal por estacion")
    windows = results.drop_duplicates("series")
    starts = pd.to_datetime(
        windows["test_target_start"], errors="coerce", format="mixed"
    )
    ends = pd.to_datetime(
        windows["test_target_end"], errors="coerce", format="mixed"
    )
    contexts = pd.to_datetime(
        windows["test_context_start"], errors="coerce", format="mixed"
    )
    hours = pd.to_numeric(windows["test_target_hours"], errors="coerce")
    if contexts.isna().any():
        raise ValueError("La ventana de contexto temporal no es valida")
    context_hours = (starts - contexts) / pd.Timedelta(hours=1)
    if (
        starts.isna().any()
        or ends.isna().any()
        or hours.isna().any()
        or hours.nunique() != 1
        or not np.isfinite(hours).all()
        or not (hours > 0).all()
        or not (hours == np.floor(hours)).all()
        or not (((ends - starts) / pd.Timedelta(hours=1) + 1) == hours).all()
    ):
        raise ValueError("Los resultados no comparten el mismo protocolo temporal")
    if (
        not np.isfinite(context_hours).all()
        or not (context_hours > 0).all()
        or not (context_hours == np.floor(context_hours)).all()
        or context_hours.nunique() != 1
    ):
        raise ValueError("La ventana de contexto temporal no es valida")


def _validate_manifest(results: pd.DataFrame, manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.json tiene una version desconocida")
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise ValueError("El run esta incompleto y debe reintentarse")

    try:
        selected = [str(series) for series in manifest["selected_stations"]]
        arms = [str(arm["name"]) for arm in manifest["arms"]]
        models = manifest["models"]
        expected_rows = int(manifest["expected_result_rows"])
        protocol = manifest["protocol"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest.json no contiene el contrato del run") from exc

    expected = {
        (series, arm, str(model["name"]))
        for series in selected
        for model in models
        for arm in (
            arms
            if bool(model["uses_training_arms"])
            else [source for source in arms if source in (RAW_ARM, RAW_FROZEN_ARM)]
        )
    }
    actual = set(
        results.loc[:, KEY_COLUMNS].astype(str).itertuples(index=False, name=None)
    )
    if len(expected) != expected_rows or actual != expected:
        raise ValueError("results.csv no contiene todas las filas esperadas del manifiesto")
    if int(manifest.get("result_rows", -1)) != len(results):
        raise ValueError("El numero de resultados no coincide con manifest.json")

    mode_by_model = {str(model["name"]): str(model["mode"]) for model in models}
    observed_modes = results.groupby("model")["model_mode"].first().astype(str).to_dict()
    if not results.empty and observed_modes != mode_by_model:
        raise ValueError("Los modelos de results.csv no coinciden con manifest.json")
    for column in PROTOCOL_COLUMNS:
        if results.empty:
            break
        if float(results[column].iloc[0]) != float(protocol[column]):
            raise ValueError("El protocolo temporal no coincide con manifest.json")


def _complete_rows(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    horizon = pd.to_numeric(frame["horizon"], errors="coerce").to_numpy()
    stride = pd.to_numeric(frame["forecast_stride"], errors="coerce").to_numpy()
    target_hours = pd.to_numeric(frame["test_target_hours"], errors="coerce").to_numpy()
    forecasts = pd.to_numeric(frame["n_forecasts"], errors="coerce").to_numpy()
    expected = pd.to_numeric(
        frame["n_expected_forecasts"], errors="coerce"
    ).to_numpy()
    predictions = pd.to_numeric(
        frame["n_test_predictions"], errors="coerce"
    ).to_numpy()
    unique_targets = pd.to_numeric(
        frame["n_unique_targets"], errors="coerce"
    ).to_numpy()
    complete = np.isfinite(values) & (values >= 0.0)
    for counts in (
        horizon,
        stride,
        target_hours,
        forecasts,
        expected,
        predictions,
        unique_targets,
    ):
        complete &= np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))
    complete &= expected > 0
    complete &= target_hours >= horizon
    complete &= (
        expected == (target_hours - horizon) / stride + 1
    )
    complete &= forecasts == expected
    complete &= predictions == forecasts * horizon
    complete &= unique_targets == target_hours
    return pd.Series(complete, index=frame.index)


def _station_scores(
    frame: pd.DataFrame,
    *,
    metric: str,
    analysis: str,
    scope: str,
    treatment_column: str,
    average_column: str,
    treatments: list[str],
    averages: list[str],
    stations: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = pd.MultiIndex.from_product(
        [treatments, averages], names=[treatment_column, average_column]
    )
    score_rows: list[dict] = []
    exclusions: list[dict] = []
    for series in stations:
        station = frame.loc[frame["series"] == series]
        panel = station.set_index([treatment_column, average_column]).reindex(expected)
        complete = len(panel) == len(expected) and _complete_rows(panel, metric).all()
        if not complete:
            exclusions.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "metric": metric,
                    "series": series,
                    "reason": "incomplete_or_nonfinite_panel",
                }
            )
            continue
        by_treatment = panel[metric].astype(float).groupby(
            level=treatment_column, sort=False
        ).mean()
        for treatment in treatments:
            score = float(by_treatment.loc[treatment])
            score_rows.append(
                {
                    "analysis": analysis,
                    "scope": scope,
                    "metric": metric,
                    "series": series,
                    "treatment": treatment,
                    "score": score,
                    "mean_error": score,
                    "n_averaged": len(averages),
                }
            )
    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        scores["rank"] = scores.groupby("series", sort=False)["score"].rank(
            method="average", ascending=True
        )
    return scores, pd.DataFrame(exclusions)


def build_model_station_scores(
    results: pd.DataFrame,
    *,
    metric: str = "mase",
    scope: str = "all_branches",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate branch errors into one paired score per station and model."""
    _validate_results(results)
    if metric not in METRICS:
        raise ValueError(f"Metrica no soportada: {metric}")
    if scope == "all_branches":
        frame = results.loc[results["model_mode"] != "foundation"].copy()
        arms = list(dict.fromkeys(frame["arm"].astype(str)))
    elif scope == "common_source_arms":
        frame = results.loc[results["arm"].isin((RAW_ARM, RAW_FROZEN_ARM))].copy()
        arms = [arm for arm in (RAW_ARM, RAW_FROZEN_ARM) if arm in set(frame["arm"])]
    else:
        raise ValueError(f"Scope de modelos no soportado: {scope}")
    models = list(dict.fromkeys(frame["model"].astype(str)))
    return _station_scores(
        frame,
        metric=metric,
        analysis="model",
        scope=scope,
        treatment_column="model",
        average_column="arm",
        treatments=models,
        averages=arms,
        stations=list(dict.fromkeys(results["series"])) if not frame.empty else [],
    )


def build_branch_station_scores(
    results: pd.DataFrame,
    *,
    metric: str = "mase",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate model errors for arms sharing the canonical raw targets."""
    _validate_results(results)
    if metric not in METRICS:
        raise ValueError(f"Metrica no soportada: {metric}")
    frame = results.loc[
        (results["model_mode"] != "foundation")
        & (results["arm"] != RAW_FROZEN_ARM)
    ].copy()
    arms = list(dict.fromkeys(frame["arm"].astype(str)))
    models = list(dict.fromkeys(frame["model"].astype(str)))
    return _station_scores(
        frame,
        metric=metric,
        analysis="branch",
        scope="common_raw_targets",
        treatment_column="arm",
        average_column="model",
        treatments=arms,
        averages=models,
        stations=list(dict.fromkeys(results["series"])) if not frame.empty else [],
    )


def _friedman_statistic(ranks: np.ndarray) -> float:
    n_blocks, n_treatments = ranks.shape
    tie_sum = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float(np.sum(counts**3 - counts))
    correction = 1.0 - tie_sum / (n_blocks * (n_treatments**3 - n_treatments))
    if correction <= 0.0:
        return 0.0
    rank_sums = ranks.sum(axis=0)
    statistic = (
        12.0 * float(np.sum(rank_sums**2))
        / (n_blocks * n_treatments * (n_treatments + 1))
        - 3.0 * n_blocks * (n_treatments + 1)
    )
    return statistic / correction


def _wilcoxon(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).copy()
    values[np.isclose(values, 0.0, rtol=0.0, atol=1e-12)] = 0.0
    if np.allclose(values, 0.0, rtol=0.0, atol=1e-12):
        return 0.0, 1.0
    result = wilcoxon(
        values,
        alternative="two-sided",
        zero_method="pratt",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _omnibus(
    wide: pd.DataFrame,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[str, float, float]:
    values = wide.to_numpy(dtype=float)
    n_treatments = values.shape[1]
    if n_treatments == 2:
        statistic, p_value = _wilcoxon(values[:, 0] - values[:, 1])
        return "wilcoxon_signed_rank", statistic, p_value
    ranks = np.apply_along_axis(rankdata, 1, np.round(values, 12), method="average")
    observed = _friedman_statistic(ranks)
    if observed == 0.0:
        return "friedman_permutation", 0.0, 1.0
    n_blocks, n_treatments = ranks.shape
    rank_sums = ranks.sum(axis=0)
    uncorrected = (
        12.0 * float(np.sum(rank_sums**2))
        / (n_blocks * n_treatments * (n_treatments + 1))
        - 3.0 * n_blocks * (n_treatments + 1)
    )
    tie_correction = uncorrected / observed
    extreme = 0
    remaining = n_permutations
    while remaining:
        batch = min(2_048, remaining)
        order = np.argsort(rng.random((batch, n_blocks, n_treatments)), axis=2)
        permuted = np.take_along_axis(
            np.broadcast_to(ranks, (batch, n_blocks, n_treatments)), order, axis=2
        )
        permuted_sums = permuted.sum(axis=1)
        statistics = (
            12.0 * np.sum(permuted_sums**2, axis=1)
            / (n_blocks * n_treatments * (n_treatments + 1))
            - 3.0 * n_blocks * (n_treatments + 1)
        ) / tie_correction
        extreme += int(np.sum(statistics >= observed - 1e-12))
        remaining -= batch
    return (
        "friedman_permutation",
        observed,
        (extreme + 1.0) / (n_permutations + 1.0),
    )


def _bootstrap_intervals(
    left: np.ndarray,
    right: np.ndarray,
    *,
    alpha: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    indices = rng.integers(0, len(left), size=(n_bootstrap, len(left)))
    sampled_left = left[indices]
    sampled_right = right[indices]
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    left_means = sampled_left.mean(axis=1)
    right_means = sampled_right.mean(axis=1)
    ratios = np.full(n_bootstrap, np.nan)
    positive_denominator = right_means > 0.0
    ratios[positive_denominator] = (
        left_means[positive_denominator] / right_means[positive_denominator]
    )
    ratios[(right_means == 0.0) & (left_means > 0.0)] = float("inf")
    ratios = ratios[~np.isnan(ratios)]
    if np.any((left == 0.0) & (right == 0.0)) or not len(ratios):
        ratio_low = ratio_high = float("nan")
    else:
        ratio_low, ratio_high = (
            float(value)
            for value in np.quantile(ratios, quantiles, method="inverted_cdf")
        )
    median_differences = np.median(sampled_left - sampled_right, axis=1)
    difference_low, difference_high = (
        float(value) for value in np.quantile(median_differences, quantiles)
    )
    return ratio_low, ratio_high, difference_low, difference_high


def _pairwise(
    wide: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    *,
    analysis: str,
    scope: str,
    metric: str,
    family: str,
    confirmatory: bool,
    alpha: float,
    omnibus_significant: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    for left, right in pairs:
        left_values = wide[left].to_numpy(dtype=float)
        right_values = wide[right].to_numpy(dtype=float)
        differences = left_values - right_values
        differences[np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)] = 0.0
        statistic, p_value = _wilcoxon(differences)
        zeros = differences == 0.0
        wins = int(np.sum(differences < -1e-12))
        losses = int(np.sum(differences > 1e-12))
        sign_p = (
            float(binomtest(wins, wins + losses, 0.5).pvalue)
            if wins + losses
            else 1.0
        )
        mean_difference = float(np.mean(differences))
        right_mean = float(np.mean(right_values))
        left_mean = float(np.mean(left_values))
        ratio = (
            left_mean / right_mean
            if right_mean > 0.0
            else (float("inf") if left_mean > 0.0 else float("nan"))
        )
        ci_low, ci_high, difference_low, difference_high = _bootstrap_intervals(
            left_values,
            right_values,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        nonzero = differences != 0.0
        if nonzero.any():
            pratt_ranks = rankdata(np.abs(differences), method="average")[nonzero]
            signed_ranks = pratt_ranks * np.sign(differences[nonzero])
            rank_biserial = float(np.sum(signed_ranks) / np.sum(np.abs(signed_ranks)))
        else:
            rank_biserial = 0.0
        rows.append(
            {
                "analysis": analysis,
                "scope": scope,
                "metric": metric,
                "family": family,
                "confirmatory": confirmatory,
                "left": left,
                "right": right,
                "n_stations": len(differences),
                "mean_difference": mean_difference,
                "median_difference": float(np.median(differences)),
                "error_ratio": ratio,
                "improvement_pct": 100.0 * (1.0 - ratio),
                "ci_ratio_low": ci_low,
                "ci_ratio_high": ci_high,
                "ci_difference_low": difference_low,
                "ci_difference_high": difference_high,
                "rank_biserial": rank_biserial,
                "wins": wins,
                "ties": int(np.sum(zeros)),
                "losses": losses,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "sign_p_value": sign_p,
                "alpha": alpha,
                "n_bootstrap": n_bootstrap,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = multipletests(frame["p_value"], alpha=alpha, method="holm")[1]
    frame["sign_p_holm"] = multipletests(
        frame["sign_p_value"], alpha=alpha, method="holm"
    )[1]
    joint_p_values = frame[["p_value", "sign_p_value"]].max(axis=1)
    frame["p_joint"] = multipletests(
        joint_p_values, alpha=alpha, method="holm"
    )[1]
    frame["omnibus_significant"] = omnibus_significant
    frame["typical_station_outcome"] = "inconclusive"
    # This conclusion concerns the paired station-level rank location. Mean
    # differences and error ratios remain descriptive and may point elsewhere.
    significant = (
        omnibus_significant
        & (frame["p_joint"] < alpha)
    )
    left_better = (
        significant
        & (frame["rank_biserial"] < 0.0)
        & (frame["median_difference"] < 0.0)
        & (frame["ci_difference_high"] < 0.0)
    )
    right_better = (
        significant
        & (frame["rank_biserial"] > 0.0)
        & (frame["median_difference"] > 0.0)
        & (frame["ci_difference_low"] > 0.0)
    )
    frame.loc[left_better, "typical_station_outcome"] = "left_lower_typical_error"
    frame.loc[right_better, "typical_station_outcome"] = "right_lower_typical_error"
    return frame


def _analyze_scores(
    scores: pd.DataFrame,
    *,
    analysis: str,
    scope: str,
    metric: str,
    alpha: float,
    n_permutations: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    branch: bool = False,
    confirmatory: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "analysis": analysis,
                        "scope": scope,
                        "metric": metric,
                        "test": "insufficient_stations",
                        "statistic": float("nan"),
                        "p_value": float("nan"),
                        "alpha": alpha,
                        "significant": False,
                        "confirmatory": confirmatory,
                        "n_stations": 0,
                        "n_treatments": 0,
                        "n_permutations": 0,
                    }
                ]
            ),
            pd.DataFrame(),
        )
    treatment_order = list(dict.fromkeys(scores["treatment"]))
    wide = scores.pivot(index="series", columns="treatment", values="score").loc[
        :, treatment_order
    ]
    insufficient = len(wide) < 2 or len(treatment_order) < 2
    if insufficient:
        return (
            pd.DataFrame(
                [
                    {
                        "analysis": analysis,
                        "scope": scope,
                        "metric": metric,
                        "test": (
                            "insufficient_stations"
                            if len(wide) < 2
                            else "insufficient_treatments"
                        ),
                        "statistic": float("nan"),
                        "p_value": float("nan"),
                        "alpha": alpha,
                        "significant": False,
                        "confirmatory": confirmatory,
                        "n_stations": len(wide),
                        "n_treatments": len(treatment_order),
                        "n_permutations": 0,
                    }
                ]
            ),
            pd.DataFrame(),
        )
    test, statistic, p_value = _omnibus(
        wide, n_permutations=n_permutations, rng=rng
    )
    omnibus = pd.DataFrame(
        [
            {
                "analysis": analysis,
                "scope": scope,
                "metric": metric,
                "test": test,
                "statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "significant": p_value < alpha,
                "confirmatory": confirmatory,
                "n_stations": len(wide),
                "n_treatments": len(treatment_order),
                "n_permutations": n_permutations if len(treatment_order) > 2 else 0,
            }
        ]
    )
    if p_value >= alpha:
        return omnibus, pd.DataFrame()
    common = {
        "analysis": analysis,
        "scope": scope,
        "metric": metric,
        "alpha": alpha,
        "omnibus_significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
        "rng": rng,
    }
    all_pairs = _pairwise(
        wide,
        combinations(treatment_order, 2),
        family="all_pairs",
        confirmatory=confirmatory and not branch,
        **common,
    )
    if not branch or RAW_ARM not in treatment_order:
        return omnibus, all_pairs
    vs_raw = _pairwise(
        wide,
        ((treatment, RAW_ARM) for treatment in treatment_order if treatment != RAW_ARM),
        family="vs_raw",
        confirmatory=confirmatory,
        **common,
    )
    return omnibus, pd.concat([all_pairs, vs_raw], ignore_index=True)


def analyze_forecasting_results(
    results: pd.DataFrame,
    *,
    alpha: float = 0.05,
    n_permutations: int = 99_999,
    n_bootstrap: int = 10_000,
    seed: int = 13,
) -> dict[str, pd.DataFrame]:
    """Build balanced station panels and run paired model/branch comparisons."""
    _validate_results(results)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre cero y uno")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in (n_permutations, n_bootstrap)
    ):
        raise ValueError(
            "Las repeticiones de permutacion y bootstrap deben ser enteros positivos"
        )
    rng = np.random.default_rng(seed)
    model_scores: list[pd.DataFrame] = []
    model_omnibus: list[pd.DataFrame] = []
    model_pairwise: list[pd.DataFrame] = []
    branch_scores: list[pd.DataFrame] = []
    branch_omnibus: list[pd.DataFrame] = []
    branch_pairwise: list[pd.DataFrame] = []
    exclusions: list[pd.DataFrame] = []

    for metric in METRICS:
        for scope in ("all_branches", "common_source_arms"):
            scores, excluded = build_model_station_scores(
                results, metric=metric, scope=scope
            )
            omnibus, pairwise = _analyze_scores(
                scores,
                analysis="model",
                scope=scope,
                metric=metric,
                alpha=alpha,
                n_permutations=n_permutations,
                n_bootstrap=n_bootstrap,
                rng=rng,
                confirmatory=metric == "mase" and scope == "all_branches",
            )
            model_scores.append(scores)
            model_omnibus.append(omnibus)
            model_pairwise.append(pairwise)
            exclusions.append(excluded)
        scores, excluded = build_branch_station_scores(results, metric=metric)
        omnibus, pairwise = _analyze_scores(
            scores,
            analysis="branch",
            scope="common_raw_targets",
            metric=metric,
            alpha=alpha,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            rng=rng,
            branch=True,
            confirmatory=metric == "mase",
        )
        branch_scores.append(scores)
        branch_omnibus.append(omnibus)
        branch_pairwise.append(pairwise)
        exclusions.append(excluded)

    def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [frame for frame in frames if not frame.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    artifacts = {
        "model_station_scores": combine(model_scores),
        "model_omnibus": combine(model_omnibus),
        "model_pairwise": combine(model_pairwise),
        "branch_station_scores": combine(branch_scores),
        "branch_omnibus": combine(branch_omnibus),
        "branch_pairwise": combine(branch_pairwise),
        "statistical_exclusions": combine(exclusions),
    }
    schemas = {
        "model_station_scores": SCORE_COLUMNS,
        "model_omnibus": OMNIBUS_COLUMNS,
        "model_pairwise": PAIRWISE_COLUMNS,
        "branch_station_scores": SCORE_COLUMNS,
        "branch_omnibus": OMNIBUS_COLUMNS,
        "branch_pairwise": PAIRWISE_COLUMNS,
        "statistical_exclusions": EXCLUSION_COLUMNS,
    }
    for key, frame in artifacts.items():
        frame["seed"] = seed
        artifacts[key] = frame.reindex(columns=schemas[key])
    return artifacts


def write_statistical_analysis(
    run_dir: Path,
    **analysis_kwargs,
) -> dict[str, Path]:
    """Analyze one forecasting run and persist its seven statistical tables."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results.csv"
    manifest_path = run_dir / "manifest.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No existe {results_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe {manifest_path}")
    results = pd.read_csv(results_path, keep_default_na=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(results, manifest)
    artifacts = analyze_forecasting_results(results, **analysis_kwargs)
    paths = {key: run_dir / filename for key, filename in OUTPUT_FILES.items()}
    for key, path in paths.items():
        artifacts[key].to_csv(path, index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrastes pareados de modelos y ramas del benchmark de forecasting"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=99_999)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    paths = write_statistical_analysis(
        args.run_dir,
        alpha=args.alpha,
        n_permutations=args.permutations,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    for path in paths.values():
        print(f"[info] {path}")


if __name__ == "__main__":
    main()
