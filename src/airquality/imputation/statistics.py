"""Paired statistical comparison of imputation models by gap size."""

from __future__ import annotations

import argparse
from itertools import combinations
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


METRICS = ("MASE", "RMSSE")
GAP_PLAN_COLUMN = "Gap_Plan_ID"
KEY_COLUMNS = ("Modelo", "Serie", "Gap_Size", "Seed", "MonteCarlo_Run")
COVERAGE_COLUMNS = (
    "N_Gaps_Target",
    "N_Gaps_Scored",
    "N_Target_Points",
    "N_Scored_Points",
    "Support_Fraction",
)
HOLDOUT_COLUMNS = (
    "Test_Start",
    "Test_End",
    "Test_Block_Points",
    "Train_Points_Before",
    "Train_Points_After",
)
OUTPUT_FILES = {
    "station_scores": "statistical_station_scores.csv",
    "omnibus": "statistical_omnibus.csv",
    "pairwise": "statistical_pairwise.csv",
    "exclusions": "statistical_exclusions.csv",
}
SCORE_COLUMNS = (
    "metric",
    "gap_size",
    "series",
    "model",
    "score",
    "rank",
    "n_runs",
    "gap_plan_verified",
    "analysis_seed",
)
OMNIBUS_COLUMNS = (
    "metric",
    "gap_size",
    "test",
    "statistic",
    "p_value",
    "alpha",
    "significant",
    "confirmatory",
    "n_stations",
    "n_models",
    "n_permutations",
    "gap_plan_verified",
    "analysis_seed",
)
PAIRWISE_COLUMNS = (
    "metric",
    "gap_size",
    "family",
    "confirmatory",
    "left",
    "right",
    "n_stations",
    "left_mean",
    "right_mean",
    "mean_difference",
    "median_difference",
    "error_ratio",
    "improvement_pct",
    "rank_biserial",
    "wins",
    "ties",
    "losses",
    "wilcoxon_statistic",
    "p_value",
    "p_holm",
    "alpha",
    "statistically_significant",
    "gap_plan_verified",
    "analysis_seed",
)
EXCLUSION_COLUMNS = (
    "metric",
    "gap_size",
    "series",
    "reason",
    "analysis_seed",
)


def _numeric_integers(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            return False
    return True


def _validate_results(results: pd.DataFrame) -> None:
    required = {*KEY_COLUMNS, *COVERAGE_COLUMNS, *HOLDOUT_COLUMNS, *METRICS}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Faltan columnas en results_mc.csv: {', '.join(missing)}")
    if results.empty:
        raise ValueError("results_mc.csv no contiene resultados")
    if results[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("Las claves del panel Monte Carlo no pueden contener nulos")
    if (results["Modelo"].astype(str).str.len() == 0).any() or (
        results["Serie"].astype(str).str.len() == 0
    ).any():
        raise ValueError("Modelo y Serie no pueden estar vacios")
    if not _numeric_integers(results, ("Gap_Size", "Seed", "MonteCarlo_Run")):
        raise ValueError("Gap_Size, Seed y MonteCarlo_Run deben ser enteros finitos")
    if (pd.to_numeric(results["Gap_Size"]) <= 0).any() or (
        pd.to_numeric(results["MonteCarlo_Run"]) <= 0
    ).any():
        raise ValueError("Gap_Size y MonteCarlo_Run deben ser positivos")
    if results.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Hay filas duplicadas en el panel Monte Carlo")

    seed_runs = results[["Seed", "MonteCarlo_Run"]].drop_duplicates()
    if (seed_runs.groupby("Seed")["MonteCarlo_Run"].nunique() != 1).any() or (
        seed_runs.groupby("MonteCarlo_Run")["Seed"].nunique() != 1
    ).any():
        raise ValueError("Seed y MonteCarlo_Run no tienen una correspondencia uno a uno")
    observed_runs = sorted(pd.to_numeric(seed_runs["MonteCarlo_Run"]).astype(int))
    if observed_runs != list(range(1, max(observed_runs) + 1)):
        raise ValueError("MonteCarlo_Run debe ser una secuencia consecutiva desde uno")
    if GAP_PLAN_COLUMN in results.columns and (
        results[GAP_PLAN_COLUMN].isna().any()
        or (results[GAP_PLAN_COLUMN].astype(str).str.len() == 0).any()
    ):
        raise ValueError("Gap_Plan_ID no puede contener valores vacios")

    holdout_variation = results.groupby("Serie", dropna=False)[
        list(HOLDOUT_COLUMNS)
    ].nunique(dropna=False)
    if (holdout_variation > 1).any().any():
        raise ValueError("El holdout no es constante dentro de cada estacion")
    holdouts = results.drop_duplicates("Serie")
    starts = pd.to_datetime(holdouts["Test_Start"], errors="coerce", format="mixed")
    ends = pd.to_datetime(holdouts["Test_End"], errors="coerce", format="mixed")
    if starts.isna().any() or ends.isna().any() or (ends < starts).any():
        raise ValueError("Las fechas del holdout no son validas")
    if not _numeric_integers(
        holdouts,
        ("Test_Block_Points", "Train_Points_Before", "Train_Points_After"),
    ):
        raise ValueError("Los recuentos del holdout deben ser enteros finitos")
    if (
        holdouts[
            ["Test_Block_Points", "Train_Points_Before", "Train_Points_After"]
        ].apply(pd.to_numeric)
        <= 0
    ).any().any():
        raise ValueError("Los recuentos del holdout deben ser positivos")


def _complete_rows(frame: pd.DataFrame, metric: str) -> bool:
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        return False
    if not _numeric_integers(
        frame,
        (
            "Gap_Size",
            "N_Gaps_Target",
            "N_Gaps_Scored",
            "N_Target_Points",
            "N_Scored_Points",
        ),
    ):
        return False
    gaps = pd.to_numeric(frame["Gap_Size"]).to_numpy(dtype=int)
    target_gaps = pd.to_numeric(frame["N_Gaps_Target"]).to_numpy(dtype=int)
    scored_gaps = pd.to_numeric(frame["N_Gaps_Scored"]).to_numpy(dtype=int)
    target_points = pd.to_numeric(frame["N_Target_Points"]).to_numpy(dtype=int)
    scored_points = pd.to_numeric(frame["N_Scored_Points"]).to_numpy(dtype=int)
    support = pd.to_numeric(frame["Support_Fraction"], errors="coerce").to_numpy(
        dtype=float
    )
    return bool(
        (target_gaps > 0).all()
        and np.equal(target_gaps, scored_gaps).all()
        and (target_points > 0).all()
        and np.equal(target_points, scored_points).all()
        and np.equal(target_points, gaps * target_gaps).all()
        and np.isfinite(support).all()
        and np.isclose(support, 1.0, rtol=0.0, atol=1e-9).all()
    )


def _panel_reason(
    panel: pd.DataFrame,
    *,
    expected_keys: pd.MultiIndex,
    metric: str,
) -> str | None:
    actual_keys = pd.MultiIndex.from_frame(
        panel[["Modelo", "Seed", "MonteCarlo_Run"]]
    )
    if len(panel) != len(expected_keys) or set(actual_keys) != set(expected_keys):
        return "incomplete_model_run_panel"
    if not _complete_rows(panel, metric):
        values = pd.to_numeric(panel[metric], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            return "nonfinite_or_invalid_metric"
        return "incomplete_support"
    protocol_columns = [*COVERAGE_COLUMNS]
    if GAP_PLAN_COLUMN in panel.columns:
        protocol_columns.append(GAP_PLAN_COLUMN)
    variation = panel.groupby(["Seed", "MonteCarlo_Run"], dropna=False)[
        protocol_columns
    ].nunique(dropna=False)
    if (variation > 1).any().any():
        return "inconsistent_paired_protocol"
    return None


def build_station_scores(
    results: pd.DataFrame,
    *,
    metric: str,
    gap_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average Monte Carlo runs into one paired score per station and model."""
    _validate_results(results)
    if metric not in METRICS:
        raise ValueError(f"Metrica no soportada: {metric}")
    if isinstance(gap_size, bool) or not isinstance(gap_size, Integral) or gap_size <= 0:
        raise ValueError("gap_size debe ser un entero positivo")

    models = list(dict.fromkeys(results["Modelo"].astype(str)))
    stations = list(dict.fromkeys(results["Serie"].astype(str)))
    seed_runs = (
        results[["Seed", "MonteCarlo_Run"]]
        .drop_duplicates()
        .sort_values("MonteCarlo_Run")
    )
    run_keys = list(seed_runs.itertuples(index=False, name=None))
    expected_keys = pd.MultiIndex.from_tuples(
        [
            (model, seed, run)
            for model in models
            for seed, run in run_keys
        ],
        names=["Modelo", "Seed", "MonteCarlo_Run"],
    )
    selected = results.loc[pd.to_numeric(results["Gap_Size"]) == int(gap_size)]
    score_rows: list[dict] = []
    exclusions: list[dict] = []
    for series in stations:
        panel = selected.loc[selected["Serie"].astype(str) == series].copy()
        reason = _panel_reason(panel, expected_keys=expected_keys, metric=metric)
        if reason is not None:
            exclusions.append(
                {
                    "metric": metric,
                    "gap_size": int(gap_size),
                    "series": series,
                    "reason": reason,
                }
            )
            continue
        panel[metric] = pd.to_numeric(panel[metric], errors="raise")
        means = panel.groupby("Modelo", sort=False)[metric].mean()
        for model in models:
            score_rows.append(
                {
                    "metric": metric,
                    "gap_size": int(gap_size),
                    "series": series,
                    "model": model,
                    "score": float(means.loc[model]),
                    "n_runs": len(run_keys),
                    "gap_plan_verified": GAP_PLAN_COLUMN in results.columns,
                }
            )
    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        scores["rank"] = scores.groupby("series", sort=False)["score"].rank(
            method="average", ascending=True
        )
    return scores, pd.DataFrame(exclusions)


def _wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float).copy()
    differences[np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)] = 0.0
    if np.all(differences == 0.0):
        return 0.0, 1.0
    result = wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="pratt",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


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


def _omnibus(
    wide: pd.DataFrame,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[str, float, float]:
    values = wide.to_numpy(dtype=float)
    if values.shape[1] == 2:
        statistic, p_value = _wilcoxon(values[:, 0] - values[:, 1])
        return "wilcoxon_signed_rank", statistic, p_value
    rounded = np.round(values, 12)
    ranks = np.apply_along_axis(rankdata, 1, rounded, method="average")
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


def _pairwise(
    wide: pd.DataFrame,
    *,
    metric: str,
    gap_size: int,
    alpha: float,
    n_permutations: int,
    gap_plan_verified: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for left, right in combinations(wide.columns, 2):
        left_values = wide[left].to_numpy(dtype=float)
        right_values = wide[right].to_numpy(dtype=float)
        differences = left_values - right_values
        differences[np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)] = 0.0
        statistic, p_value = _wilcoxon(differences)
        left_mean = float(np.mean(left_values))
        right_mean = float(np.mean(right_values))
        ratio = (
            left_mean / right_mean
            if right_mean > 0.0
            else (float("inf") if left_mean > 0.0 else float("nan"))
        )
        nonzero = differences != 0.0
        if nonzero.any():
            pratt_ranks = rankdata(np.abs(differences), method="average")[nonzero]
            signed_ranks = pratt_ranks * np.sign(differences[nonzero])
            rank_biserial = float(
                np.sum(signed_ranks) / np.sum(np.abs(signed_ranks))
            )
        else:
            rank_biserial = 0.0
        rows.append(
            {
                "metric": metric,
                "gap_size": int(gap_size),
                "family": "all_pairs_within_gap",
                "confirmatory": metric == "MASE",
                "left": str(left),
                "right": str(right),
                "n_stations": len(differences),
                "left_mean": left_mean,
                "right_mean": right_mean,
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "error_ratio": ratio,
                "improvement_pct": 100.0 * (1.0 - ratio),
                "rank_biserial": rank_biserial,
                "wins": int(np.sum(differences < 0.0)),
                "ties": int(np.sum(differences == 0.0)),
                "losses": int(np.sum(differences > 0.0)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "gap_plan_verified": gap_plan_verified,
            }
        )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise
    pairwise["p_holm"] = multipletests(
        pairwise["p_value"], alpha=alpha, method="holm"
    )[1]
    pairwise["statistically_significant"] = pairwise["p_holm"] < alpha
    return pairwise


def _analyze_scores(
    scores: pd.DataFrame,
    *,
    metric: str,
    gap_size: int,
    alpha: float,
    n_permutations: int,
    gap_plan_verified: bool,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = list(dict.fromkeys(scores["model"])) if not scores.empty else []
    wide = (
        scores.pivot(index="series", columns="model", values="score").loc[:, models]
        if models
        else pd.DataFrame()
    )
    if len(wide) < 2 or len(models) < 2:
        test = "insufficient_stations" if len(wide) < 2 else "insufficient_models"
        return (
            pd.DataFrame(
                [
                    {
                        "metric": metric,
                        "gap_size": int(gap_size),
                        "test": test,
                        "statistic": float("nan"),
                        "p_value": float("nan"),
                        "alpha": alpha,
                        "significant": False,
                        "confirmatory": metric == "MASE",
                        "n_stations": len(wide),
                        "n_models": len(models),
                        "gap_plan_verified": gap_plan_verified,
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
                "metric": metric,
                "gap_size": int(gap_size),
                "test": test,
                "statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "significant": p_value < alpha,
                "confirmatory": metric == "MASE",
                "n_stations": len(wide),
                "n_models": len(models),
                "n_permutations": n_permutations if len(models) > 2 else 0,
                "gap_plan_verified": gap_plan_verified,
            }
        ]
    )
    if p_value >= alpha:
        return omnibus, pd.DataFrame()
    return omnibus, _pairwise(
        wide,
        metric=metric,
        gap_size=gap_size,
        alpha=alpha,
        n_permutations=n_permutations,
        gap_plan_verified=gap_plan_verified,
    )


def analyze_imputation_results(
    results: pd.DataFrame,
    *,
    alpha: float = 0.05,
    n_permutations: int = 99_999,
    seed: int = 13,
) -> dict[str, pd.DataFrame]:
    """Run paired model comparisons independently for every configured gap size."""
    _validate_results(results)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre cero y uno")
    if (
        isinstance(n_permutations, bool)
        or not isinstance(n_permutations, Integral)
        or n_permutations < 1
    ):
        raise ValueError("n_permutations debe ser un entero positivo")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed debe ser un entero")

    rng = np.random.default_rng(seed)
    gap_plan_verified = GAP_PLAN_COLUMN in results.columns
    gaps = sorted(pd.to_numeric(results["Gap_Size"]).astype(int).unique())
    score_frames: list[pd.DataFrame] = []
    omnibus_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    for metric in METRICS:
        for gap_size in gaps:
            scores, exclusions = build_station_scores(
                results, metric=metric, gap_size=int(gap_size)
            )
            omnibus, pairwise = _analyze_scores(
                scores,
                metric=metric,
                gap_size=int(gap_size),
                alpha=alpha,
                n_permutations=n_permutations,
                gap_plan_verified=gap_plan_verified,
                rng=rng,
            )
            score_frames.append(scores)
            omnibus_frames.append(omnibus)
            pairwise_frames.append(pairwise)
            exclusion_frames.append(exclusions)

    def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [frame for frame in frames if not frame.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    artifacts = {
        "station_scores": combine(score_frames),
        "omnibus": combine(omnibus_frames),
        "pairwise": combine(pairwise_frames),
        "exclusions": combine(exclusion_frames),
    }
    schemas = {
        "station_scores": SCORE_COLUMNS,
        "omnibus": OMNIBUS_COLUMNS,
        "pairwise": PAIRWISE_COLUMNS,
        "exclusions": EXCLUSION_COLUMNS,
    }
    for key, frame in artifacts.items():
        frame["analysis_seed"] = int(seed)
        artifacts[key] = frame.reindex(columns=schemas[key])
    return artifacts


def write_statistical_analysis(
    run_dir: Path,
    **analysis_kwargs,
) -> dict[str, Path]:
    """Analyze ``results_mc.csv`` and persist the four statistical tables."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results_mc.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"No existe {results_path}")
    results = pd.read_csv(results_path, keep_default_na=False)
    artifacts = analyze_imputation_results(results, **analysis_kwargs)
    paths = {key: run_dir / filename for key, filename in OUTPUT_FILES.items()}
    for key, path in paths.items():
        artifacts[key].to_csv(path, index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrastes pareados por gap del benchmark de imputacion"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--permutations", type=int, default=99_999)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    paths = write_statistical_analysis(
        args.run_dir,
        alpha=args.alpha,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    for path in paths.values():
        print(f"[info] {path}")


if __name__ == "__main__":
    main()
