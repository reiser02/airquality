"""Persist plot data and render a completed Monte Carlo imputation run.

Run again without executing any model::

    uv run python -m airquality.visualizations.montecarlo \
        reports/benchmark/montecarlo_<stamp>

Compare two completed runs without executing any model::

    uv run python -m airquality.visualizations.montecarlo \
        reports/benchmark/montecarlo_<left> \
        reports/benchmark/montecarlo_<right>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from airquality.metrics import metric_higher_is_better
from airquality.visualizations.style import (
    AXIS_FACE,
    FIGURE_FACE,
    GRID_COLOR,
    MUTED_TEXT,
    SPINE_COLOR,
    TEXT_COLOR,
    add_plot_header,
    save_figure,
    style_axis,
)
PLOT_STORE_COLUMNS = (
    "gap_size",
    "series_name",
    "kind",
    "model_name",
    "timestamp",
    "value",
)
PLOT_METRICS = ("MASE", "RMSSE", "R2")
OBSOLETE_SUMMARY_PLOT_PATTERNS = (
    "metric_forest_*.png",
    "station_distribution_*.png",
    "paired_mean_rank_*.png",
    "paired_delta_vs_*.png",
    "station_vs_sampling_variability_*.png",
    "metrics_mean_station_sd_heatmaps.png",
    "metrics_across_stations.png",
    "global_rank_frequency_*.png",
    "global_performance_profile_*.png",
    "metrics_by_gap_mae.png",
    "metrics_by_gap_rmse.png",
    "model_performance_by_gap_mae.png",
    "model_performance_by_gap_rmse.png",
    "overall_model_performance_mae.png",
    "overall_model_performance_rmse.png",
    "global_station_error_mae.png",
    "global_station_error_rmse.png",
    "pairwise_win_rate_mae.png",
    "pairwise_win_rate_rmse.png",
    "gap_degradation_mae.png",
    "gap_degradation_rmse.png",
    "tail_risk_mae.png",
    "tail_risk_rmse.png",
    "error_correlation_mae.png",
    "error_correlation_rmse.png",
)

# Distinct line colours drawn from the same presentation.py hex family.
_LINE_PALETTE = (
    "#4b79a8",  # blue (SERIES_COLOR)
    "#f28c38",  # orange
    "#5b8c5a",  # green
    "#9b59b6",  # purple
    "#d6453c",  # red
    "#8c6d4b",  # brown
    "#cf6ba9",  # pink
    "#6d6258",  # taupe
    "#27313a",  # dark slate (TEXT_COLOR)
    "#2a9d8f",  # teal
    "#b59a00",  # mustard
    "#7b6ca8",  # muted violet
)
# Curated colours so key models stay visually stable across runs; the linear
# baseline gets the standout red so its gap=1 behaviour is easy to spot.
_MODEL_LINE_COLORS = {
    "DLinear": "#8c6d4b",
    "LinearInterp": "#d6453c",
    "LinearRegression": "#cf6ba9",
    "NLinear": "#b59a00",
    "TSPulse": "#9b59b6",
    "TSPulse_FineTuned": "#7b6ca8",
    "Prophet": "#5b8c5a",
    "TiDE": "#4b79a8",
    "NHiTS": "#f28c38",
    "TSMixer": "#6d6258",
    "RNN": "#27313a",
    "TCN": "#3a86ff",
    "interp": "#2a9d8f",
}


def _model_color_map(models: list[str]) -> dict[str, str]:
    """Assign a stable, distinct colour to each model from the base palette."""
    color_map: dict[str, str] = {}
    used = {color for name, color in _MODEL_LINE_COLORS.items() if name in models}
    cycle = [color for color in _LINE_PALETTE if color not in used]
    cycle_pos = 0
    for model in sorted(models):
        if model in _MODEL_LINE_COLORS:
            color_map[model] = _MODEL_LINE_COLORS[model]
        else:
            color_map[model] = cycle[cycle_pos % len(cycle)] if cycle else "#4b79a8"
            cycle_pos += 1
    return color_map


def _style_metric_axis(ax: "plt.Axes") -> None:
    """Apply the shared cream/base styling to one metric subplot."""
    style_axis(ax)


def _aggregate_metrics_by_gap(
    results_mc_df: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    """Aggregate runs within station, then stations with equal weight."""
    work = results_mc_df.copy()
    if "Serie" not in work.columns:
        work["Serie"] = "Serie unica"
    for metric in metrics:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    if "Support_Fraction" in work.columns:
        work["Support_Fraction"] = pd.to_numeric(
            work["Support_Fraction"], errors="coerce"
        )

    rows: list[dict[str, Any]] = []
    for (model, gap), group in work.groupby(["Modelo", "Gap_Size"], sort=True):
        station_means = group.groupby("Serie", sort=False)[metrics].mean(
            numeric_only=True
        )
        row: dict[str, Any] = {
            "Modelo": str(model),
            "Gap_Size": int(gap),
            "N_Runs": int(
                group["MonteCarlo_Run"].nunique()
                if "MonteCarlo_Run" in group.columns
                else group["Seed"].nunique()
                if "Seed" in group.columns
                else 1
            ),
        }
        for metric in metrics:
            values = station_means[metric].dropna()
            n_stations = len(values)
            mean = float(values.mean()) if n_stations else float("nan")
            standard_deviation = (
                float(values.std(ddof=1)) if n_stations > 1 else float("nan")
            )
            if n_stations > 1 and np.isfinite(standard_deviation):
                critical_value = float(student_t.ppf(0.975, n_stations - 1))
                half_width = critical_value * standard_deviation / np.sqrt(n_stations)
                ci_low = mean - half_width
                ci_high = mean + half_width
            else:
                half_width = ci_low = ci_high = float("nan")
            row[f"{metric}_Mean"] = mean
            row[f"{metric}_SD"] = standard_deviation
            row[f"{metric}_CI95_Low"] = ci_low
            row[f"{metric}_CI95_High"] = ci_high
            row[f"{metric}_CI95_Half_Width"] = half_width
            row[f"{metric}_P05"] = (
                float(values.quantile(0.05)) if n_stations else float("nan")
            )
            row[f"{metric}_P95"] = (
                float(values.quantile(0.95)) if n_stations else float("nan")
            )
            row[f"{metric}_Min"] = (
                float(values.min()) if n_stations else float("nan")
            )
            row[f"{metric}_Max"] = (
                float(values.max()) if n_stations else float("nan")
            )
            row[f"{metric}_N_Stations"] = n_stations
        if "Support_Fraction" in group.columns:
            station_support = group.groupby("Serie", sort=False)[
                "Support_Fraction"
            ].mean()
            row["Support_Mean"] = float(station_support.mean())
        else:
            row["Support_Mean"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def _station_seed_metric_tables(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return replicate and station means without pooling the hierarchy."""
    base_columns = ["Modelo", "Serie", "Gap_Size"]
    value_columns = [
        *metrics,
        *[
            column
            for column in ("Test_Block_Points", "Test_Hours")
            if column in results_mc_df.columns
        ],
    ]
    if results_mc_df.empty or not {"Modelo", "Gap_Size"}.issubset(
        results_mc_df.columns
    ):
        return (
            pd.DataFrame(columns=[*base_columns, *value_columns]),
            pd.DataFrame(columns=[*base_columns, *value_columns]),
        )

    work = results_mc_df.copy()
    if "Serie" not in work.columns:
        work["Serie"] = "Serie unica"
    work["Modelo"] = work["Modelo"].astype(str)
    work["Serie"] = work["Serie"].astype(str)
    work["Gap_Size"] = pd.to_numeric(work["Gap_Size"], errors="coerce")
    work = work.dropna(subset=["Gap_Size"])
    for column in value_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    sample_columns = [
        column
        for column in ("MonteCarlo_Run", "Seed")
        if column in work.columns
    ]
    if sample_columns:
        replicate_means = (
            work.groupby(
                [*base_columns, *sample_columns],
                as_index=False,
                dropna=False,
                sort=False,
            )[value_columns]
            .mean(numeric_only=True)
        )
    else:
        replicate_means = work[[*base_columns, *value_columns]].copy()
        replicate_means["_Sampling_Run"] = replicate_means.groupby(
            base_columns, sort=False
        ).cumcount()

    station_means = (
        replicate_means.groupby(base_columns, as_index=False, sort=False)[
            value_columns
        ]
        .mean(numeric_only=True)
    )
    station_means["Gap_Size"] = station_means["Gap_Size"].astype(int)
    replicate_means["Gap_Size"] = replicate_means["Gap_Size"].astype(int)

    support_columns = [
        column
        for column in ("Test_Block_Points", "Test_Hours")
        if column in station_means.columns
    ]
    station_means["Reduced_Test_Support"] = False
    for column in support_columns:
        panel_max = station_means.groupby("Gap_Size")[column].transform("max")
        finite = station_means[column].notna() & panel_max.notna()
        station_means.loc[finite, "Reduced_Test_Support"] |= ~np.isclose(
            station_means.loc[finite, column],
            panel_max.loc[finite],
            rtol=1e-9,
            atol=1e-12,
        )
    return replicate_means, station_means


def _complete_metric_panel(
    results_mc_df: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """Return seed-averaged values for complete station-gap model panels."""
    value_columns = [metric]
    if "Support_Fraction" in results_mc_df.columns:
        value_columns.append("Support_Fraction")
    _, station_means = _station_seed_metric_tables(results_mc_df, value_columns)
    if station_means.empty:
        return pd.DataFrame()
    panel = station_means.pivot(
        index=["Serie", "Gap_Size"], columns="Modelo", values=metric
    )
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if "Support_Fraction" in station_means.columns:
        support = station_means.pivot(
            index=["Serie", "Gap_Size"],
            columns="Modelo",
            values="Support_Fraction",
        ).reindex(panel.index)
        complete_support = support.ge(1.0 - 1e-9).all(axis=1)
        panel = panel.loc[complete_support]
    return panel


def _global_rank_frequency_tables(
    results_mc_df: pd.DataFrame, metrics: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired ranks with equal gap weight inside each station."""
    summary_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        ranks = panel.rank(
            axis=1,
            method="average",
            ascending=not metric_higher_is_better(metric),
        )
        details = (
            ranks.rename_axis(columns="Modelo")
            .stack()
            .rename("Rank")
            .reset_index()
        )
        details["Modelo"] = details["Modelo"].astype(str)
        n_models = int(len(panel.columns))
        rank_grid = sorted(
            {float(rank) for rank in range(1, n_models + 1)}
            | set(details["Rank"].astype(float))
        )
        for model, group in details.groupby("Modelo", sort=True):
            station_mean_ranks = group.groupby("Serie", sort=False)["Rank"].mean()
            top_three_by_station = (
                group.assign(Is_Top_3=group["Rank"] <= 3.0)
                .groupby("Serie", sort=False)["Is_Top_3"]
                .mean()
            )
            wins_by_station = (
                group.assign(Is_Winner=np.isclose(group["Rank"], 1.0))
                .groupby("Serie", sort=False)["Is_Winner"]
                .mean()
            )
            common = {
                "Metric": metric,
                "Modelo": str(model),
                "Mean_Rank": float(station_mean_ranks.mean()),
                "Winner_Percent": float(100.0 * wins_by_station.mean()),
                "Top_3_Percent": float(100.0 * top_three_by_station.mean()),
                "N_Contexts": int(len(group)),
                "N_Stations": int(group["Serie"].nunique()),
                "N_Gaps": int(group["Gap_Size"].nunique()),
                "N_Models": n_models,
            }
            summary_rows.append(common)
            for rank in rank_grid:
                rate_by_station = (
                    group.assign(At_Rank=np.isclose(group["Rank"], rank))
                    .groupby("Serie", sort=False)["At_Rank"]
                    .mean()
                )
                frequency_rows.append(
                    {
                        **common,
                        "Rank": rank,
                        "Context_Percent": float(100.0 * rate_by_station.mean()),
                    }
                )
    summary_columns = [
        "Metric",
        "Modelo",
        "Mean_Rank",
        "Winner_Percent",
        "Top_3_Percent",
        "N_Contexts",
        "N_Stations",
        "N_Gaps",
        "N_Models",
    ]
    frequency_columns = [*summary_columns, "Rank", "Context_Percent"]
    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(frequency_rows, columns=frequency_columns),
    )


def _global_performance_profile_table(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
    *,
    thresholds: tuple[float, ...] = (1.0, 1.05, 1.10, 1.25, 1.50, 2.0),
) -> pd.DataFrame:
    """Measure how often each model is within fixed ratios of the block winner."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        best = panel.min(axis=1)
        ratios = panel.div(best.mask(np.isclose(best, 0.0)), axis=0)
        zero_best = np.isclose(best, 0.0)
        if zero_best.any():
            zero_values = panel.loc[zero_best]
            ratios.loc[zero_best] = np.where(
                np.isclose(zero_values, 0.0), 1.0, np.inf
            )
        details = (
            ratios.rename_axis(columns="Modelo")
            .stack()
            .rename("Ratio_To_Best")
            .reset_index()
        )
        details["Modelo"] = details["Modelo"].astype(str)
        for model, group in details.groupby("Modelo", sort=True):
            for threshold in thresholds:
                within_by_station = (
                    group.assign(
                        Within=group["Ratio_To_Best"] <= threshold + 1e-12
                    )
                    .groupby("Serie", sort=False)["Within"]
                    .mean()
                )
                rows.append(
                    {
                        "Metric": metric,
                        "Modelo": str(model),
                        "Threshold_Ratio": float(threshold),
                        "Tolerance_Percent": round(100.0 * (threshold - 1.0), 10),
                        "Context_Percent": float(100.0 * within_by_station.mean()),
                        "N_Contexts": int(len(group)),
                        "N_Stations": int(group["Serie"].nunique()),
                        "N_Gaps": int(group["Gap_Size"].nunique()),
                    }
                )
    columns = [
        "Metric",
        "Modelo",
        "Threshold_Ratio",
        "Tolerance_Percent",
        "Context_Percent",
        "N_Contexts",
        "N_Stations",
        "N_Gaps",
    ]
    return pd.DataFrame(rows, columns=columns)


def _global_station_error_table(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Return one equally gap-weighted error value per station and model."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        station_values = panel.groupby(level="Serie", sort=False).mean()
        gap_counts = panel.groupby(level="Serie", sort=False).size()
        for station, values in station_values.iterrows():
            for model, error in values.items():
                rows.append(
                    {
                        "Metric": metric,
                        "Modelo": str(model),
                        "Serie": str(station),
                        "Error": float(error),
                        "N_Gaps": int(gap_counts.loc[station]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["Metric", "Modelo", "Serie", "Error", "N_Gaps"],
    )


def _pairwise_win_rate_table(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Compare every model pair on matched station-gap contexts."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        for model in panel.columns:
            for opponent in panel.columns:
                differences = panel[model] - panel[opponent]
                ties = np.isclose(differences, 0.0, rtol=1e-9, atol=1e-12)
                wins = (
                    differences > 0.0
                    if metric_higher_is_better(metric)
                    else differences < 0.0
                )
                scores = pd.Series(
                    np.where(ties, 0.5, wins.astype(float)), index=panel.index
                )
                station_scores = scores.groupby(level="Serie", sort=False).mean()
                station_deltas = differences.groupby(level="Serie", sort=False).mean()
                station_ties = pd.Series(
                    ties.astype(float), index=panel.index
                ).groupby(level="Serie", sort=False).mean()
                rows.append(
                    {
                        "Metric": metric,
                        "Modelo": str(model),
                        "Opponent": str(opponent),
                        "Win_Rate_Percent": float(100.0 * station_scores.mean()),
                        "Mean_Delta": float(station_deltas.mean()),
                        "Tie_Percent": float(100.0 * station_ties.mean()),
                        "N_Contexts": int(len(differences)),
                        "N_Stations": int(
                            panel.index.get_level_values("Serie").nunique()
                        ),
                        "N_Gaps": int(
                            panel.index.get_level_values("Gap_Size").nunique()
                        ),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Modelo",
            "Opponent",
            "Win_Rate_Percent",
            "Mean_Delta",
            "Tie_Percent",
            "N_Contexts",
            "N_Stations",
            "N_Gaps",
        ],
    )


def _gap_degradation_table(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Measure each model's station-balanced degradation from its shortest gap."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        gap_sizes = sorted(
            int(value) for value in panel.index.get_level_values("Gap_Size").unique()
        )
        baseline_gap = gap_sizes[0]
        baseline = panel.xs(baseline_gap, level="Gap_Size")
        for gap_size in gap_sizes:
            current = panel.xs(gap_size, level="Gap_Size")
            stations = baseline.index.intersection(current.index)
            for model in panel.columns:
                base_values = baseline.loc[stations, model].to_numpy(dtype=float)
                current_values = current.loc[stations, model].to_numpy(dtype=float)
                ratios = np.divide(
                    current_values,
                    base_values,
                    out=np.full_like(current_values, np.nan),
                    where=~np.isclose(base_values, 0.0),
                )
                both_zero = np.isclose(base_values, 0.0) & np.isclose(
                    current_values, 0.0
                )
                ratios[both_zero] = 1.0
                finite_ratios = ratios[np.isfinite(ratios)]
                rows.append(
                    {
                        "Metric": metric,
                        "Modelo": str(model),
                        "Gap_Size": gap_size,
                        "Baseline_Gap": baseline_gap,
                        "Mean_Error": float(np.mean(current_values)),
                        "Mean_Ratio": (
                            float(np.mean(finite_ratios))
                            if len(finite_ratios)
                            else float("nan")
                        ),
                        "Median_Ratio": (
                            float(np.median(finite_ratios))
                            if len(finite_ratios)
                            else float("nan")
                        ),
                        "Degradation_Percent": (
                            float(100.0 * (np.mean(finite_ratios) - 1.0))
                            if len(finite_ratios)
                            else float("nan")
                        ),
                        "N_Stations": int(len(stations)),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Modelo",
            "Gap_Size",
            "Baseline_Gap",
            "Mean_Error",
            "Mean_Ratio",
            "Median_Ratio",
            "Degradation_Percent",
            "N_Stations",
        ],
    )


def _tail_risk_table(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
    *,
    quantile: float = 0.90,
) -> pd.DataFrame:
    """Summarize typical and hard-station errors with equal gap weight."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        gap_sizes = sorted(
            int(value) for value in panel.index.get_level_values("Gap_Size").unique()
        )
        for model in panel.columns:
            gap_means: list[float] = []
            gap_quantiles: list[float] = []
            gap_tail_means: list[float] = []
            for gap_size in gap_sizes:
                values = panel.xs(gap_size, level="Gap_Size")[model].to_numpy(
                    dtype=float
                )
                threshold = float(np.quantile(values, quantile))
                gap_means.append(float(np.mean(values)))
                gap_quantiles.append(threshold)
                gap_tail_means.append(float(np.mean(values[values >= threshold])))
            mean_error = float(np.mean(gap_means))
            tail_mean = float(np.mean(gap_tail_means))
            rows.append(
                {
                    "Metric": metric,
                    "Modelo": str(model),
                    "Mean_Error": mean_error,
                    "P90_Error": float(np.mean(gap_quantiles)),
                    "Tail_Mean_Error": tail_mean,
                    "Tail_Penalty_Percent": (
                        float(100.0 * (tail_mean / mean_error - 1.0))
                        if not np.isclose(mean_error, 0.0)
                        else float("nan")
                    ),
                    "Quantile": quantile,
                    "N_Contexts": int(len(panel)),
                    "N_Stations": int(
                        panel.index.get_level_values("Serie").nunique()
                    ),
                    "N_Gaps": int(len(gap_sizes)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Modelo",
            "Mean_Error",
            "P90_Error",
            "Tail_Mean_Error",
            "Tail_Penalty_Percent",
            "Quantile",
            "N_Contexts",
            "N_Stations",
            "N_Gaps",
        ],
    )


def _error_correlation_table(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Average per-gap Spearman error correlations across stations."""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        gap_sizes = sorted(
            int(value) for value in panel.index.get_level_values("Gap_Size").unique()
        )
        correlations = [
            panel.xs(gap_size, level="Gap_Size").corr(method="spearman")
            for gap_size in gap_sizes
        ]
        for model in panel.columns:
            for other_model in panel.columns:
                values = np.asarray(
                    [correlation.loc[model, other_model] for correlation in correlations],
                    dtype=float,
                )
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "Metric": metric,
                        "Modelo": str(model),
                        "Other_Model": str(other_model),
                        "Mean_Spearman": (
                            float(np.mean(finite)) if len(finite) else float("nan")
                        ),
                        "Min_Spearman": (
                            float(np.min(finite)) if len(finite) else float("nan")
                        ),
                        "Max_Spearman": (
                            float(np.max(finite)) if len(finite) else float("nan")
                        ),
                        "N_Contexts": int(len(panel)),
                        "N_Stations": int(
                            panel.index.get_level_values("Serie").nunique()
                        ),
                        "N_Gaps": int(len(finite)),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Modelo",
            "Other_Model",
            "Mean_Spearman",
            "Min_Spearman",
            "Max_Spearman",
            "N_Contexts",
            "N_Stations",
            "N_Gaps",
        ],
    )


def _model_performance_tables(
    results_mc_df: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize absolute and relative performance by gap and overall."""
    gap_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for metric in metrics:
        panel = _complete_metric_panel(results_mc_df, metric)
        if panel.empty:
            continue
        ranks = panel.rank(
            axis=1,
            method="average",
            ascending=not metric_higher_is_better(metric),
        )
        gap_sizes = sorted(
            int(value) for value in panel.index.get_level_values("Gap_Size").unique()
        )
        n_models = int(len(panel.columns))

        for gap in gap_sizes:
            gap_values = panel.xs(gap, level="Gap_Size")
            gap_ranks = ranks.xs(gap, level="Gap_Size")
            for model in panel.columns:
                values = gap_values[model].dropna()
                model_ranks = gap_ranks[model].dropna()
                gap_rows.append(
                    {
                        "Metric": metric,
                        "Modelo": str(model),
                        "Gap_Size": gap,
                        "Mean": float(values.mean()),
                        "Median": float(values.median()),
                        "Station_SD": (
                            float(values.std(ddof=1))
                            if len(values) > 1
                            else float("nan")
                        ),
                        "Mean_Rank": float(model_ranks.mean()),
                        "Winner_Percent": float(
                            100.0 * np.isclose(model_ranks, 1.0).mean()
                        ),
                        "Top_3_Percent": float(100.0 * (model_ranks <= 3.0).mean()),
                        "N_Contexts": int(len(values)),
                        "N_Stations": int(len(values)),
                        "N_Gaps": 1,
                        "N_Models": n_models,
                    }
                )

        station_values = panel.groupby(level="Serie", sort=False).mean()
        station_ranks = ranks.groupby(level="Serie", sort=False).mean()
        station_wins = ranks.eq(1.0).groupby(level="Serie", sort=False).mean()
        station_top_three = (ranks <= 3.0).groupby(
            level="Serie", sort=False
        ).mean()
        for model in panel.columns:
            values = station_values[model].dropna()
            model_ranks = station_ranks[model].dropna()
            overall_rows.append(
                {
                    "Metric": metric,
                    "Modelo": str(model),
                    "Mean": float(values.mean()),
                    "Median": float(values.median()),
                    "Station_SD": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else float("nan")
                    ),
                    "Mean_Rank": float(model_ranks.mean()),
                    "Winner_Percent": float(100.0 * station_wins[model].mean()),
                    "Top_3_Percent": float(
                        100.0 * station_top_three[model].mean()
                    ),
                    "N_Contexts": int(len(panel)),
                    "N_Stations": int(len(values)),
                    "N_Gaps": int(len(gap_sizes)),
                    "N_Models": n_models,
                }
            )

    gap_columns = [
        "Metric",
        "Modelo",
        "Gap_Size",
        "Mean",
        "Median",
        "Station_SD",
        "Mean_Rank",
        "Winner_Percent",
        "Top_3_Percent",
        "N_Contexts",
        "N_Stations",
        "N_Gaps",
        "N_Models",
    ]
    overall_columns = [column for column in gap_columns if column != "Gap_Size"]
    return (
        pd.DataFrame(gap_rows, columns=gap_columns),
        pd.DataFrame(overall_rows, columns=overall_columns),
    )


def _metric_display_name(metric: str, *, scaled_errors: bool) -> str:
    if scaled_errors and metric in {"MAE", "RMSE"}:
        return f"{metric} escalado"
    return metric


def _format_metric_value(value: float) -> str:
    """Keep regular errors precise and extreme failures compact."""
    if abs(value) >= 10_000:
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _summary_support_text(frame: pd.DataFrame) -> str:
    """Describe the dynamic station-gap support represented in a summary."""
    row = frame.iloc[0]
    return (
        "Semillas promediadas antes de comparar; "
        f"{int(row['N_Stations'])} estaciones y {int(row['N_Gaps'])} huecos "
        f"pesan por igual ({int(row['N_Contexts'])} contextos pareados)."
    )


def _save_global_station_error_plots(
    station_errors: pd.DataFrame,
    overall_summary: pd.DataFrame,
    metrics: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> dict[str, Path]:
    """Render station-level global error distributions for every model."""
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_frame = station_errors[station_errors["Metric"] == metric]
        overall = overall_summary[overall_summary["Metric"] == metric]
        if metric_frame.empty or overall.empty:
            continue
        model_order = (
            overall.sort_values(
                ["Mean", "Median", "Modelo"],
                ascending=[
                    not metric_higher_is_better(metric),
                    not metric_higher_is_better(metric),
                    True,
                ],
                kind="stable",
            )["Modelo"]
            .astype(str)
            .tolist()
        )
        colors = _model_color_map(model_order)
        positions = np.arange(len(model_order))
        fig, ax = plt.subplots(
            figsize=(10.8, max(6.4, 0.48 * len(model_order) + 2.5)),
            facecolor=FIGURE_FACE,
        )
        for position, model in zip(positions, model_order, strict=True):
            model_frame = metric_frame[metric_frame["Modelo"] == model].sort_values(
                "Serie", kind="stable"
            )
            values = model_frame["Error"].to_numpy(dtype=float)
            box = ax.boxplot(
                values,
                positions=[position],
                widths=0.58,
                orientation="horizontal",
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                whis=(5, 95),
                boxprops={"facecolor": colors[model], "alpha": 0.62},
                medianprops={"color": "#f28c38", "linewidth": 2.0},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "#f3b43f",
                    "markeredgecolor": TEXT_COLOR,
                    "markersize": 5,
                },
                whiskerprops={"color": colors[model], "linewidth": 1.2},
                capprops={"color": colors[model], "linewidth": 1.2},
            )
            for artist in box["boxes"]:
                artist.set_edgecolor(colors[model])
            offsets = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else [0.0]
            ax.scatter(
                values,
                position + np.asarray(offsets),
                s=13,
                color=colors[model],
                edgecolor=AXIS_FACE,
                linewidth=0.35,
                alpha=0.52,
                zorder=3,
            )

        _style_metric_axis(ax)
        ax.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
        ax.grid(False, axis="y")
        ax.set_yticks(positions)
        ax.set_yticklabels(model_order, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(
            f"{_metric_display_name(metric, scaled_errors=scaled_errors)} medio por estación"
        )
        ax.set_ylabel("Modelo")
        if metric in {"MASE", "RMSSE"}:
            ax.axvline(1.0, color=MUTED_TEXT, linewidth=1.1, linestyle="--")

        display_metric = _metric_display_name(
            metric, scaled_errors=scaled_errors
        )
        add_plot_header(
            fig,
            f"Variación global entre estaciones: {display_metric}",
            _summary_support_text(overall)
            + " Cada punto es una estación; caja P25-P75, línea naranja mediana y diamante media.\n"
            + (
                "Cómo leer: más a la derecha es mejor; una caja corta indica comportamiento homogéneo entre estaciones."
                if metric_higher_is_better(metric)
                else "Cómo leer: más a la izquierda es mejor; una caja corta indica comportamiento homogéneo entre estaciones."
            ),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        image_path = output_dir / f"global_station_error_{metric.lower()}.png"
        save_figure(fig, image_path)
        paths[metric] = image_path
    return paths


def _save_pairwise_win_rate_plots(
    pairwise_rates: pd.DataFrame,
    overall_summary: pd.DataFrame,
    metrics: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> dict[str, Path]:
    """Render matched head-to-head win rates without reducing them to ranks."""
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_frame = pairwise_rates[pairwise_rates["Metric"] == metric]
        overall = overall_summary[overall_summary["Metric"] == metric]
        if metric_frame.empty or overall.empty:
            continue
        model_order = (
            overall.sort_values(
                ["Mean", "Median", "Modelo"],
                ascending=[
                    not metric_higher_is_better(metric),
                    not metric_higher_is_better(metric),
                    True,
                ],
                kind="stable",
            )["Modelo"]
            .astype(str)
            .tolist()
        )
        matrix = (
            metric_frame.pivot(
                index="Modelo", columns="Opponent", values="Win_Rate_Percent"
            )
            .reindex(index=model_order, columns=model_order)
        )
        values = matrix.to_numpy(dtype=float)
        fig, ax = plt.subplots(
            figsize=(10.6, max(7.3, 0.55 * len(model_order) + 2.6)),
            facecolor=FIGURE_FACE,
        )
        image = ax.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            cmap="RdYlBu",
            vmin=0.0,
            vmax=100.0,
        )
        for y, model in enumerate(model_order):
            for x, opponent in enumerate(model_order):
                value = float(matrix.loc[model, opponent])
                if not np.isfinite(value):
                    continue
                ax.text(
                    x,
                    y,
                    "--" if model == opponent else f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value <= 18.0 or value >= 82.0 else TEXT_COLOR,
                )
        _style_metric_axis(ax)
        ax.grid(False)
        ax.set_xlabel("Modelo comparado")
        ax.set_ylabel("Modelo de la fila")
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=8.5)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
        colorbar.set_label("Victorias de la fila (%)", color=TEXT_COLOR)
        colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        colorbar.ax.axhline(50.0, color=TEXT_COLOR, linewidth=1.0)

        display_metric = _metric_display_name(
            metric, scaled_errors=scaled_errors
        )
        add_plot_header(
            fig,
            f"Comparación global por pares: {display_metric}",
            _summary_support_text(overall)
            + " La celda indica cuánto gana la fila a la columna; los empates cuentan 0.5.\n"
            "Cómo leer: más de 50% favorece a la fila, menos de 50% a la columna; compare siempre una pareja concreta.",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        image_path = output_dir / f"pairwise_win_rate_{metric.lower()}.png"
        save_figure(fig, image_path)
        paths[metric] = image_path
    return paths


def _save_gap_degradation_plots(
    degradation: pd.DataFrame,
    overall_summary: pd.DataFrame,
    metrics: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> dict[str, Path]:
    """Render relative sensitivity to longer gaps for every model."""
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_frame = degradation[degradation["Metric"] == metric]
        overall = overall_summary[overall_summary["Metric"] == metric]
        if metric_frame.empty or overall.empty:
            continue
        model_order = (
            overall.sort_values(
                ["Mean", "Median", "Modelo"],
                ascending=[
                    not metric_higher_is_better(metric),
                    not metric_higher_is_better(metric),
                    True,
                ],
                kind="stable",
            )["Modelo"]
            .astype(str)
            .tolist()
        )
        colors = _model_color_map(model_order)
        gap_sizes = sorted(int(value) for value in metric_frame["Gap_Size"].unique())
        baseline_gap = int(metric_frame["Baseline_Gap"].iloc[0])
        fig, ax = plt.subplots(figsize=(11.8, 7.4), facecolor=FIGURE_FACE)
        for model in model_order:
            model_frame = metric_frame[metric_frame["Modelo"] == model].sort_values(
                "Gap_Size", kind="stable"
            )
            ax.plot(
                model_frame["Gap_Size"],
                model_frame["Degradation_Percent"],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                color=colors[model],
                label=model,
            )
        _style_metric_axis(ax)
        ax.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.58)
        ax.axhline(0.0, color=MUTED_TEXT, linewidth=1.1)
        ax.set_xticks(gap_sizes)
        ax.set_xlabel("Tamaño del hueco (h)")
        ax.set_ylabel(f"Cambio respecto a {baseline_gap} h (%)")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
            ncol=1,
        )
        display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
        add_plot_header(
            fig,
            f"Sensibilidad al tamaño del hueco: {display_metric}",
            _summary_support_text(overall)
            + f" Cada modelo se compara consigo mismo en {baseline_gap} h.\n"
            "Cómo leer: una curva baja y plana indica poca degradación; 50% significa un error 1.5 veces mayor.",
        )
        fig.tight_layout(rect=(0, 0, 0.86, 0.90))
        image_path = output_dir / f"gap_degradation_{metric.lower()}.png"
        save_figure(fig, image_path)
        paths[metric] = image_path
    return paths


def _save_tail_risk_plots(
    tail_risk: pd.DataFrame,
    metrics: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> dict[str, Path]:
    """Render typical, P90, and worst-decile mean error as dumbbells."""
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_frame = tail_risk[tail_risk["Metric"] == metric].sort_values(
            ["Mean_Error", "Modelo"], kind="stable"
        )
        if metric_frame.empty:
            continue
        model_order = metric_frame["Modelo"].astype(str).tolist()
        positions = np.arange(len(model_order))
        means = metric_frame["Mean_Error"].to_numpy(dtype=float)
        p90 = metric_frame["P90_Error"].to_numpy(dtype=float)
        tails = metric_frame["Tail_Mean_Error"].to_numpy(dtype=float)
        penalties = metric_frame["Tail_Penalty_Percent"].to_numpy(dtype=float)
        fig, ax = plt.subplots(
            figsize=(11.2, max(6.6, 0.48 * len(model_order) + 2.7)),
            facecolor=FIGURE_FACE,
        )
        _style_metric_axis(ax)
        ax.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.62)
        ax.grid(False, axis="y")
        ax.hlines(positions, means, tails, color="#9eb4c9", linewidth=3.0, zorder=1)
        ax.scatter(means, positions, color="#2a9d8f", s=45, label="Media", zorder=3)
        ax.scatter(p90, positions, color="#f28c38", s=36, label="P90", zorder=3)
        ax.scatter(
            tails,
            positions,
            color="#d6453c",
            marker="D",
            s=40,
            label="Media del peor 10%",
            zorder=3,
        )
        for position, (tail, penalty) in enumerate(zip(tails, penalties, strict=True)):
            ax.annotate(
                f"+{penalty:.0f}%",
                (tail, position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=TEXT_COLOR,
            )
        ax.set_yticks(positions)
        ax.set_yticklabels(model_order, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(_metric_display_name(metric, scaled_errors=scaled_errors))
        ax.set_ylabel("Modelo")
        if metric in {"MASE", "RMSSE"}:
            ax.axvline(1.0, color=MUTED_TEXT, linewidth=1.1, linestyle="--")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
        )
        display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
        add_plot_header(
            fig,
            f"Riesgo de errores extremos: {display_metric}",
            _summary_support_text(metric_frame)
            + " El peor 10% se calcula entre estaciones dentro de cada hueco.\n"
            "Cómo leer: más a la izquierda es mejor; un segmento corto indica poca penalización en estaciones difíciles.",
        )
        fig.tight_layout(rect=(0, 0, 0.86, 0.90))
        image_path = output_dir / f"tail_risk_{metric.lower()}.png"
        save_figure(fig, image_path)
        paths[metric] = image_path
    return paths


def _save_error_correlation_plots(
    correlations: pd.DataFrame,
    overall_summary: pd.DataFrame,
    metrics: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> dict[str, Path]:
    """Render lower-triangle error-profile correlations between models."""
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_frame = correlations[correlations["Metric"] == metric]
        overall = overall_summary[overall_summary["Metric"] == metric]
        if metric_frame.empty or overall.empty:
            continue
        model_order = (
            overall.sort_values(
                ["Mean", "Median", "Modelo"],
                ascending=[
                    not metric_higher_is_better(metric),
                    not metric_higher_is_better(metric),
                    True,
                ],
                kind="stable",
            )["Modelo"]
            .astype(str)
            .tolist()
        )
        matrix = (
            metric_frame.pivot(
                index="Modelo", columns="Other_Model", values="Mean_Spearman"
            )
            .reindex(index=model_order, columns=model_order)
        )
        values = matrix.to_numpy(dtype=float)
        upper_triangle = np.triu(np.ones_like(values, dtype=bool), k=1)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(AXIS_FACE)
        fig, ax = plt.subplots(
            figsize=(10.6, max(7.3, 0.55 * len(model_order) + 2.6)),
            facecolor=FIGURE_FACE,
        )
        image = ax.imshow(
            np.ma.masked_where(upper_triangle | ~np.isfinite(values), values),
            aspect="auto",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
        )
        for y, model in enumerate(model_order):
            for x, other_model in enumerate(model_order):
                value = float(matrix.loc[model, other_model])
                if x > y or not np.isfinite(value):
                    continue
                ax.text(
                    x,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="white" if abs(value) >= 0.72 else TEXT_COLOR,
                )
        _style_metric_axis(ax)
        ax.grid(False)
        ax.set_xlabel("Modelo comparado")
        ax.set_ylabel("Modelo de la fila")
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=8.5)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
        colorbar.set_label("Correlación de Spearman", color=TEXT_COLOR)
        colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
        add_plot_header(
            fig,
            f"Similitud de los patrones entre estaciones: {display_metric}",
            _summary_support_text(overall)
            + " Se correlacionan estaciones dentro de cada hueco y después se promedian los huecos.\n"
            "Cómo leer: +1 indica perfiles similares entre estaciones; 0 o valores negativos sugieren complementariedad, no mayor precisión.",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        image_path = output_dir / f"error_correlation_{metric.lower()}.png"
        save_figure(fig, image_path)
        paths[metric] = image_path
    return paths


def _annotate_heatmap(
    ax: "plt.Axes",
    image: Any,
    values: np.ndarray,
    *,
    formatter: Any,
    dark_at_low: bool = False,
) -> None:
    """Write compact values over a heatmap with readable contrast."""
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if not np.isfinite(value):
                label = "--"
                color = TEXT_COLOR
            else:
                label = formatter(float(value))
                normalized = float(image.norm(value))
                use_white = normalized < 0.34 if dark_at_low else normalized > 0.62
                color = "white" if use_white else TEXT_COLOR
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )


def _save_model_performance_by_gap_plots(
    gap_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    metrics: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> dict[str, Path]:
    """Render absolute error, mean rank, and top-three frequency by gap."""
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_frame = gap_summary[gap_summary["Metric"] == metric]
        overall = overall_summary[overall_summary["Metric"] == metric]
        if metric_frame.empty or overall.empty:
            continue
        model_order = (
            overall.sort_values(
                ["Mean_Rank", "Mean", "Modelo"],
                ascending=[True, not metric_higher_is_better(metric), True],
                kind="stable",
            )["Modelo"]
            .astype(str)
            .tolist()
        )
        gap_sizes = sorted(int(value) for value in metric_frame["Gap_Size"].unique())
        n_models = int(overall["N_Models"].max())
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15.6, max(6.8, 0.46 * len(model_order) + 2.7)),
            facecolor=FIGURE_FACE,
            sharey=True,
        )
        higher_is_better = metric_higher_is_better(metric)
        specifications = (
            (
                "Mean",
                f"{_metric_display_name(metric, scaled_errors=scaled_errors)} medio",
                "YlGnBu" if higher_is_better else "YlOrRd",
                None,
                None,
                _format_metric_value,
                False,
                "Valor medio",
            ),
            (
                "Mean_Rank",
                "Rango medio",
                "YlGnBu_r",
                1.0,
                float(n_models),
                lambda value: f"{value:.1f}",
                True,
                "Rango (1 = mejor)",
            ),
            (
                "Top_3_Percent",
                "Frecuencia en el top 3",
                "YlGnBu",
                0.0,
                100.0,
                lambda value: f"{value:.0f}%",
                False,
                "Estaciones en top 3 (%)",
            ),
        )
        for panel_index, (ax, specification) in enumerate(
            zip(axes, specifications, strict=True)
        ):
            (
                value_column,
                title,
                cmap_name,
                vmin,
                vmax,
                formatter,
                dark_at_low,
                colorbar_label,
            ) = specification
            matrix = (
                metric_frame.pivot(
                    index="Modelo", columns="Gap_Size", values=value_column
                )
                .reindex(index=model_order, columns=gap_sizes)
            )
            values = matrix.to_numpy(dtype=float)
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#ded8cf")
            image_kwargs: dict[str, Any] = {
                "aspect": "auto",
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
            }
            image = ax.imshow(np.ma.masked_invalid(values), **image_kwargs)
            _annotate_heatmap(
                ax,
                image,
                values,
                formatter=formatter,
                dark_at_low=dark_at_low,
            )
            _style_metric_axis(ax)
            ax.grid(False)
            ax.set_title(title, fontsize=11.5, fontweight="bold")
            ax.set_xlabel("Tamaño del hueco (h)")
            ax.set_xticks(range(len(gap_sizes)))
            ax.set_xticklabels([str(value) for value in gap_sizes])
            ax.set_yticks(range(len(model_order)))
            if panel_index == 0:
                ax.set_yticklabels(model_order, fontsize=8.5)
                ax.set_ylabel("Modelo")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
            colorbar.set_label(
                colorbar_label,
                color=TEXT_COLOR,
                fontsize=8.5,
            )
            colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)

        display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
        add_plot_header(
            fig,
            f"Rendimiento por tamaño de hueco: {display_metric}",
            _summary_support_text(overall)
            + " La métrica conserva su escala; rango y top 3 comparan dentro de cada estación-hueco.\n"
            + (
                "Cómo leer: R² alto, rango bajo y top 3 alto indican mejor rendimiento y consistencia."
                if higher_is_better
                else "Cómo leer: error y rango bajos son mejores; un porcentaje top 3 alto indica consistencia."
            ),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        image_path = output_dir / f"model_performance_by_gap_{metric.lower()}.png"
        save_figure(fig, image_path)
        paths[metric] = image_path
    return paths


def _save_overall_model_performance_plots(
    overall_summary: pd.DataFrame,
    metrics: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> dict[str, Path]:
    """Render a compact global scorecard without per-station clutter."""
    paths: dict[str, Path] = {}
    for metric in metrics:
        metric_frame = overall_summary[overall_summary["Metric"] == metric]
        if metric_frame.empty:
            continue
        metric_frame = metric_frame.sort_values(
            ["Mean_Rank", "Mean", "Modelo"],
            ascending=[True, not metric_higher_is_better(metric), True],
            kind="stable",
        ).reset_index(drop=True)
        model_order = metric_frame["Modelo"].astype(str).tolist()
        positions = np.arange(len(model_order))
        means = metric_frame["Mean"].to_numpy(dtype=float)
        mean_ranks = metric_frame["Mean_Rank"].to_numpy(dtype=float)
        winners = metric_frame["Winner_Percent"].to_numpy(dtype=float)
        top_three = metric_frame["Top_3_Percent"].to_numpy(dtype=float)
        n_models = int(metric_frame["N_Models"].max())

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15.2, max(6.8, 0.46 * len(model_order) + 2.7)),
            facecolor=FIGURE_FACE,
            sharey=True,
            gridspec_kw={"width_ratios": (1.15, 1.0, 1.2)},
        )
        for ax in axes:
            _style_metric_axis(ax)
            ax.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.6)
            ax.grid(False, axis="y")
            ax.set_yticks(positions)

        axes[0].barh(
            positions, means, height=0.62, color="#4b79a8", alpha=0.88
        )
        axes[0].set_yticklabels(model_order, fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].set_title(
            f"{_metric_display_name(metric, scaled_errors=scaled_errors)} medio",
            fontsize=11.5,
            fontweight="bold",
        )
        axes[0].set_xlabel(
            "Valor medio (mayor es mejor)"
            if metric_higher_is_better(metric)
            else "Error medio (menor es mejor)"
        )
        axes[0].set_ylabel("Modelo")
        if metric in {"MASE", "RMSSE"}:
            axes[0].axvline(1.0, color=MUTED_TEXT, linewidth=1.1, linestyle="--")
        for position, value in zip(positions, means, strict=True):
            axes[0].annotate(
                _format_metric_value(value),
                (value, position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
            )
        axes[0].margins(x=0.18)

        axes[1].barh(
            positions, mean_ranks, height=0.62, color="#d6453c", alpha=0.84
        )
        axes[1].set_title("Rango medio", fontsize=11.5, fontweight="bold")
        axes[1].set_xlabel("Rango (1 = mejor)")
        axes[1].set_xlim(0.0, n_models + 1.8)
        axes[1].tick_params(axis="y", labelleft=False)
        for position, value in zip(positions, mean_ranks, strict=True):
            axes[1].annotate(
                f"{value:.2f}",
                (value, position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
            )

        axes[2].barh(
            positions,
            top_three,
            height=0.62,
            color="#7fc8b2",
            alpha=0.9,
            label="Top 3",
        )
        axes[2].barh(
            positions,
            winners,
            height=0.30,
            color="#245b78",
            alpha=0.95,
            label="Ganador",
        )
        axes[2].set_title("Consistencia", fontsize=11.5, fontweight="bold")
        axes[2].set_xlabel("Contextos (%)")
        axes[2].set_xlim(0.0, 108.0)
        axes[2].tick_params(axis="y", labelleft=False)
        axes[2].legend(loc="lower right", fontsize=8, frameon=False)
        for position, (winner, top_rate) in enumerate(
            zip(winners, top_three, strict=True)
        ):
            axes[2].annotate(
                f"{winner:.0f}% / {top_rate:.0f}%",
                (top_rate, position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=TEXT_COLOR,
            )

        display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
        add_plot_header(
            fig,
            f"Rendimiento global de los modelos: {display_metric}",
            _summary_support_text(metric_frame)
            + " Las barras resumen la métrica media, el rango y la frecuencia ganador / top 3.\n"
            + (
                "Cómo leer: R² alto, rango bajo y consistencia alta son mejores."
                if metric_higher_is_better(metric)
                else "Cómo leer: error y rango bajos, y consistencia alta, son mejores."
            ),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        image_path = output_dir / f"overall_model_performance_{metric.lower()}.png"
        save_figure(fig, image_path)
        paths[metric] = image_path
    return paths


def _metric_gap_plot_inputs(
    results_mc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[int], list[str], bool] | None:
    """Prepare the shared data and ordering for metric-by-gap heatmaps."""
    if results_mc_df.empty:
        return None
    if not {"Modelo", "Gap_Size"}.issubset(results_mc_df.columns):
        return None

    metrics = [m for m in PLOT_METRICS if m in results_mc_df.columns]
    if not metrics:
        return None

    agg_df = _aggregate_metrics_by_gap(results_mc_df, metrics)
    if agg_df.empty:
        return None
    gap_sizes = sorted(int(g) for g in agg_df["Gap_Size"].unique())
    order_metric = "MASE" if "MASE" in metrics else metrics[0]
    model_order = (
        agg_df.groupby("Modelo")[f"{order_metric}_Mean"]
        .mean()
        .sort_values(ascending=not metric_higher_is_better(order_metric))
        .index.astype(str)
        .tolist()
    )
    scaled_errors = "Scale_Std" in results_mc_df.columns
    return agg_df, metrics, gap_sizes, model_order, scaled_errors


def _draw_metric_value_heatmap(
    ax: "plt.Axes",
    agg_df: pd.DataFrame,
    metric: str,
    *,
    gap_sizes: list[int],
    model_order: list[str],
    value_suffix: str,
    display_title: str,
    colorbar_label: str,
    cmap_name: str,
) -> bool:
    """Draw one annotated model-by-gap statistic heatmap."""
    matrix = (
        agg_df.pivot(
            index="Modelo",
            columns="Gap_Size",
            values=f"{metric}_{value_suffix}",
        )
        .reindex(index=model_order, columns=gap_sizes)
    )
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return False

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#ded8cf")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        cmap=cmap,
    )
    for row_index, model in enumerate(model_order):
        for col_index, gap in enumerate(gap_sizes):
            value = values[row_index, col_index]
            if not np.isfinite(value):
                label = "--"
            else:
                support_row = agg_df[
                    (agg_df["Modelo"] == model) & (agg_df["Gap_Size"] == gap)
                ]
                low_support = (
                    not support_row.empty
                    and float(support_row.iloc[0]["Support_Mean"]) < 0.999
                )
                label = (
                    f"{_format_metric_value(float(value))}"
                    f"{'*' if low_support else ''}"
                )
            normalized = image.norm(value) if np.isfinite(value) else 0.0
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if normalized > 0.62 else TEXT_COLOR,
            )

    _style_metric_axis(ax)
    ax.grid(False)
    ax.set_title(display_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Tamaño del hueco (h)")
    ax.set_ylabel("Modelo")
    ax.set_xticks(range(len(gap_sizes)))
    ax.set_xticklabels([str(gap) for gap in gap_sizes])
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8.5)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.035)
    colorbar.set_label(colorbar_label, color=TEXT_COLOR, fontsize=9)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    return True


def _draw_metric_gap_heatmap(
    ax: "plt.Axes",
    agg_df: pd.DataFrame,
    metric: str,
    *,
    gap_sizes: list[int],
    model_order: list[str],
    scaled_errors: bool,
) -> bool:
    """Draw one annotated model-by-gap mean heatmap."""
    display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
    return _draw_metric_value_heatmap(
        ax,
        agg_df,
        metric,
        gap_sizes=gap_sizes,
        model_order=model_order,
        value_suffix="Mean",
        display_title=display_metric,
        colorbar_label=display_metric,
        cmap_name="YlGnBu" if metric_higher_is_better(metric) else "YlOrRd",
    )


def _draw_metric_mean_uncertainty_heatmap(
    ax: "plt.Axes",
    agg_df: pd.DataFrame,
    metric: str,
    *,
    gap_sizes: list[int],
    model_order: list[str],
    scaled_errors: bool,
    uncertainty_suffix: str,
    uncertainty_label: str,
) -> bool:
    """Draw a mean heatmap annotated with an uncertainty half-width."""
    mean_matrix = (
        agg_df.pivot(index="Modelo", columns="Gap_Size", values=f"{metric}_Mean")
        .reindex(index=model_order, columns=gap_sizes)
    )
    half_width_matrix = (
        agg_df.pivot(
            index="Modelo",
            columns="Gap_Size",
            values=f"{metric}_{uncertainty_suffix}",
        )
        .reindex(index=model_order, columns=gap_sizes)
    )
    mean_values = mean_matrix.to_numpy(dtype=float)
    half_width_values = half_width_matrix.to_numpy(dtype=float)
    finite = mean_values[np.isfinite(mean_values)]
    if len(finite) == 0:
        return False

    cmap = plt.get_cmap(
        "YlGnBu" if metric_higher_is_better(metric) else "YlOrRd"
    ).copy()
    cmap.set_bad("#ded8cf")
    image = ax.imshow(
        np.ma.masked_invalid(mean_values),
        aspect="auto",
        cmap=cmap,
    )
    for row_index, model in enumerate(model_order):
        for col_index, gap in enumerate(gap_sizes):
            mean_value = mean_values[row_index, col_index]
            half_width = half_width_values[row_index, col_index]
            if not np.isfinite(mean_value):
                label = "--"
            elif np.isfinite(half_width):
                label = f"{mean_value:.2f}\n±{half_width:.2f}"
            else:
                label = f"{mean_value:.2f}\n±--"
            if np.isfinite(mean_value):
                support_row = agg_df[
                    (agg_df["Modelo"] == model) & (agg_df["Gap_Size"] == gap)
                ]
                low_support = (
                    not support_row.empty
                    and float(support_row.iloc[0]["Support_Mean"]) < 0.999
                )
                if low_support:
                    label += "*"
            normalized = image.norm(mean_value) if np.isfinite(mean_value) else 0.0
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if normalized > 0.62 else TEXT_COLOR,
            )

    _style_metric_axis(ax)
    ax.grid(False)
    display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
    ax.set_title(
        f"{display_metric}: media ± {uncertainty_label}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Tamaño del hueco (h)")
    ax.set_ylabel("Modelo")
    ax.set_xticks(range(len(gap_sizes)))
    ax.set_xticklabels([str(gap) for gap in gap_sizes])
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8.5)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.035)
    colorbar.set_label(display_metric, color=TEXT_COLOR, fontsize=9)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    return True


def _draw_metric_mean_ci_heatmap(
    ax: "plt.Axes",
    agg_df: pd.DataFrame,
    metric: str,
    *,
    gap_sizes: list[int],
    model_order: list[str],
    scaled_errors: bool,
) -> bool:
    """Draw a mean heatmap annotated with the 95% CI half-width."""
    return _draw_metric_mean_uncertainty_heatmap(
        ax,
        agg_df,
        metric,
        gap_sizes=gap_sizes,
        model_order=model_order,
        scaled_errors=scaled_errors,
        uncertainty_suffix="CI95_Half_Width",
        uncertainty_label="IC 95%",
    )


def _draw_metric_mean_sd_heatmap(
    ax: "plt.Axes",
    agg_df: pd.DataFrame,
    metric: str,
    *,
    gap_sizes: list[int],
    model_order: list[str],
    scaled_errors: bool,
) -> bool:
    """Draw a mean heatmap annotated with station-level standard deviation."""
    return _draw_metric_mean_uncertainty_heatmap(
        ax,
        agg_df,
        metric,
        gap_sizes=gap_sizes,
        model_order=model_order,
        scaled_errors=scaled_errors,
        uncertainty_suffix="SD",
        uncertainty_label="SD",
    )


def _save_metric_mean_uncertainty_heatmap(
    agg_df: pd.DataFrame,
    metrics: list[str],
    gap_sizes: list[int],
    model_order: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
    uncertainty_label: str,
    filename: str,
    title: str,
    subtitle: str,
    draw_function: Any,
) -> Path | None:
    """Save a combined mean-plus-uncertainty heatmap."""
    image_path = output_dir / filename
    image_path.unlink(missing_ok=True)
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.8 * len(metrics), max(6.0, 0.43 * len(model_order) + 2.2)),
        facecolor=FIGURE_FACE,
        squeeze=False,
    )
    axes_row = axes[0]
    has_data = False
    for ax, metric in zip(axes_row, metrics):
        drawn = draw_function(
            ax,
            agg_df,
            metric,
            gap_sizes=gap_sizes,
            model_order=model_order,
            scaled_errors=scaled_errors,
        )
        has_data = has_data or drawn
        if not drawn:
            ax.set_visible(False)
    if not has_data:
        plt.close(fig)
        return None
    add_plot_header(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(fig, image_path)
    return image_path


def _save_metric_gap_uncertainty_plots(
    agg_df: pd.DataFrame,
    metrics: list[str],
    gap_sizes: list[int],
    model_order: list[str],
    *,
    output_dir: Path,
    scaled_errors: bool,
) -> None:
    """Save station variability and 95% mean-confidence summaries."""
    uncertainty_rows: list[dict[str, Any]] = []
    for _, row in agg_df.iterrows():
        for metric in metrics:
            uncertainty_rows.append(
                {
                    "Metric": metric,
                    "Modelo": str(row["Modelo"]),
                    "Gap_Size": int(row["Gap_Size"]),
                    "Mean": row[f"{metric}_Mean"],
                    "SD": row[f"{metric}_SD"],
                    "P05": row[f"{metric}_P05"],
                    "P95": row[f"{metric}_P95"],
                    "Min": row[f"{metric}_Min"],
                    "Max": row[f"{metric}_Max"],
                    "CI95_Low": row[f"{metric}_CI95_Low"],
                    "CI95_High": row[f"{metric}_CI95_High"],
                    "CI95_Half_Width": row[f"{metric}_CI95_Half_Width"],
                    "N_Stations": row[f"{metric}_N_Stations"],
                    "Support_Mean": row["Support_Mean"],
                }
            )
    uncertainty_table = pd.DataFrame(uncertainty_rows)
    uncertainty_table_path = output_dir / "metrics_by_gap_uncertainty.csv"
    uncertainty_table.to_csv(uncertainty_table_path, index=False)

    _save_metric_mean_uncertainty_heatmap(
        agg_df,
        metrics,
        gap_sizes,
        model_order,
        output_dir=output_dir,
        scaled_errors=scaled_errors,
        uncertainty_label="SD",
        filename="metrics_by_gap_mean_sd.png",
        title="Rendimiento medio y variabilidad por tamaño de hueco",
        subtitle="Cada celda muestra la media ± la SD entre estaciones. "
        "El color representa únicamente la media.",
        draw_function=_draw_metric_mean_sd_heatmap,
    )
    _save_metric_mean_uncertainty_heatmap(
        agg_df,
        metrics,
        gap_sizes,
        model_order,
        output_dir=output_dir,
        scaled_errors=scaled_errors,
        uncertainty_label="IC 95%",
        filename="metrics_by_gap_mean_ci95.png",
        title="Rendimiento medio con intervalo de confianza por tamaño de hueco",
        subtitle="Cada celda muestra la media ± la semiamplitud del IC t del 95%. "
        "El color representa únicamente la media; los límites completos están en metrics_by_gap_uncertainty.csv.",
        draw_function=_draw_metric_mean_ci_heatmap,
    )

    risk_rows: list[dict[str, Any]] = []
    for _, row in agg_df.iterrows():
        for metric in metrics:
            higher_is_better = metric_higher_is_better(metric)
            tail_statistic = "P05" if higher_is_better else "P95"
            extreme_statistic = "Min" if higher_is_better else "Max"
            risk_rows.append(
                {
                    "Metric": metric,
                    "Modelo": str(row["Modelo"]),
                    "Gap_Size": int(row["Gap_Size"]),
                    "Mean": row[f"{metric}_Mean"],
                    "SD": row[f"{metric}_SD"],
                    "Tail_Statistic": tail_statistic,
                    "Tail_Value": row[f"{metric}_{tail_statistic}"],
                    "Extreme_Statistic": extreme_statistic,
                    "Extreme_Value": row[f"{metric}_{extreme_statistic}"],
                    "N_Stations": row[f"{metric}_N_Stations"],
                    "Support_Mean": row["Support_Mean"],
                }
            )
    pd.DataFrame(risk_rows).to_csv(
        output_dir / "metrics_by_gap_risk.csv",
        index=False,
    )

    risk_path = output_dir / "metrics_by_gap_risk.png"
    risk_path.unlink(missing_ok=True)
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.8 * len(metrics), max(6.0, 0.43 * len(model_order) + 2.2)),
        facecolor=FIGURE_FACE,
        squeeze=False,
    )
    axes_row = axes[0]
    has_risk_data = False
    for ax, metric in zip(axes_row, metrics):
        higher_is_better = metric_higher_is_better(metric)
        tail_statistic = "P05" if higher_is_better else "P95"
        display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
        drawn = _draw_metric_value_heatmap(
            ax,
            agg_df,
            metric,
            gap_sizes=gap_sizes,
            model_order=model_order,
            value_suffix=tail_statistic,
            display_title=f"{display_metric} | {tail_statistic}",
            colorbar_label=f"{tail_statistic} entre estaciones",
            cmap_name="YlGnBu" if higher_is_better else "YlOrRd",
        )
        has_risk_data = has_risk_data or drawn
        if not drawn:
            ax.set_visible(False)
    if has_risk_data:
        add_plot_header(
            fig,
            "Riesgo entre estaciones por tamaño de hueco",
            "MASE/RMSSE muestran P95 y R² muestra P05 porque los valores bajos son peores. "
            "El máximo o mínimo está disponible en metrics_by_gap_risk.csv.",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        save_figure(fig, risk_path)
    else:
        plt.close(fig)

    specifications = (
        (
            "SD",
            "Variabilidad entre estaciones por tamaño de hueco",
            "Cada celda muestra la desviación estándar de los valores medios por estación. "
            "Valores menores indican un comportamiento más homogéneo.",
            "Desviación estándar",
        ),
        (
            "CI95_Half_Width",
            "Incertidumbre de la media por tamaño de hueco",
            "Cada celda muestra la semiamplitud del intervalo de confianza t del 95% para la media. "
            "Los límites completos se encuentran en metrics_by_gap_uncertainty.csv.",
            "Semiamplitud del IC 95%",
        ),
    )
    for value_suffix, title, subtitle, colorbar_label in specifications:
        image_path = output_dir / (
            "metrics_by_gap_sd.png"
            if value_suffix == "SD"
            else "metrics_by_gap_ci95.png"
        )
        image_path.unlink(missing_ok=True)
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(4.8 * len(metrics), max(6.0, 0.43 * len(model_order) + 2.2)),
            facecolor=FIGURE_FACE,
            squeeze=False,
        )
        axes_row = axes[0]
        has_data = False
        for ax, metric in zip(axes_row, metrics):
            display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
            drawn = _draw_metric_value_heatmap(
                ax,
                agg_df,
                metric,
                gap_sizes=gap_sizes,
                model_order=model_order,
                value_suffix=value_suffix,
                display_title=f"{display_metric} | {value_suffix}",
                colorbar_label=colorbar_label,
                cmap_name="YlOrRd",
            )
            has_data = has_data or drawn
            if not drawn:
                ax.set_visible(False)
        if not has_data:
            plt.close(fig)
            continue
        add_plot_header(fig, title, subtitle)
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        save_figure(fig, image_path)


def _save_metric_gap_plots(
    results_mc_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> tuple[Path | None, dict[str, Path]]:
    """Save the combined heatmap and one readable heatmap per metric."""
    for filename in (
        "metrics_by_gap_mean_sd.png",
        "metrics_by_gap_sd.png",
        "metrics_by_gap_ci95.png",
        "metrics_by_gap_mean_ci95.png",
        "metrics_by_gap_risk.png",
        "metrics_by_gap_risk.csv",
        "metrics_by_gap_uncertainty.csv",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    prepared = _metric_gap_plot_inputs(results_mc_df)
    if prepared is None:
        return None, {}
    agg_df, metrics, gap_sizes, model_order, scaled_errors = prepared
    agg_df = agg_df.sort_values(["Modelo", "Gap_Size"]).reset_index(drop=True)
    agg_df.to_csv(output_dir / "metrics_by_gap.csv", index=False)
    _save_metric_gap_uncertainty_plots(
        agg_df,
        metrics,
        gap_sizes,
        model_order,
        output_dir=output_dir,
        scaled_errors=scaled_errors,
    )

    for metric in ("MAE", "RMSE", *PLOT_METRICS):
        (output_dir / f"metrics_by_gap_{metric.lower()}.png").unlink(
            missing_ok=True
        )

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.8 * len(metrics), max(6.0, 0.43 * len(model_order) + 2.2)),
        facecolor=FIGURE_FACE,
        squeeze=False,
    )
    axes_row = axes[0]

    for ax, metric in zip(axes_row, metrics):
        if not _draw_metric_gap_heatmap(
            ax,
            agg_df,
            metric,
            gap_sizes=gap_sizes,
            model_order=model_order,
            scaled_errors=scaled_errors,
        ):
            ax.set_visible(False)

    add_plot_header(
        fig,
        "Rendimiento medio por tamaño de hueco",
        "Cada celda promedia primero las semillas dentro de cada estación y después las estaciones por igual. "
        "* indica soporte incompleto.\n"
        "Cómo leer: MASE/RMSSE menores son mejores; R² mayor es mejor. Compare dentro de una misma métrica y hueco.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    combined_path = output_dir / "metrics_by_gap.png"
    save_figure(fig, combined_path)

    metric_paths: dict[str, Path] = {}
    for metric in metrics:
        fig, ax = plt.subplots(
            figsize=(8.6, max(6.2, 0.43 * len(model_order) + 2.5)),
            facecolor=FIGURE_FACE,
        )
        if not _draw_metric_gap_heatmap(
            ax,
            agg_df,
            metric,
            gap_sizes=gap_sizes,
            model_order=model_order,
            scaled_errors=scaled_errors,
        ):
            plt.close(fig)
            continue
        display_metric = _metric_display_name(metric, scaled_errors=scaled_errors)
        add_plot_header(
            fig,
            f"Rendimiento medio por tamaño de hueco: {display_metric}",
            "Cada celda promedia primero las semillas dentro de cada estación y después las estaciones por igual. "
            "* indica soporte incompleto.\n"
            + (
                "Cómo leer: valores menores son mejores. Compare los modelos dentro de cada tamaño de hueco."
                if not metric_higher_is_better(metric)
                else "Cómo leer: valores mayores son mejores. Compare los modelos dentro de cada tamaño de hueco."
            ),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        image_path = output_dir / f"metrics_by_gap_{metric.lower()}.png"
        save_figure(fig, image_path)
        metric_paths[metric] = image_path

    return combined_path, metric_paths


def _save_metric_gap_plot(
    results_mc_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path | None:
    """Compatibility wrapper returning the combined metric heatmap path."""
    combined_path, _ = _save_metric_gap_plots(
        results_mc_df,
        output_dir=output_dir,
    )
    return combined_path


def _save_montecarlo_diagnostic_plots(
    results_mc_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Persist compact global and per-gap model performance summaries."""
    for pattern in OBSOLETE_SUMMARY_PLOT_PATTERNS:
        for path in output_dir.glob(pattern):
            path.unlink()
    empty_artifacts: dict[str, Any] = {
        "model_performance_by_gap_plot_paths": {},
        "overall_model_performance_plot_paths": {},
        "global_station_error_plot_paths": {},
        "pairwise_win_rate_plot_paths": {},
        "gap_degradation_plot_paths": {},
        "tail_risk_plot_paths": {},
        "error_correlation_plot_paths": {},
        "diagnostic_table_paths": {},
    }
    metrics = [
        metric
        for metric in PLOT_METRICS
        if metric in results_mc_df.columns
    ]
    if not metrics or results_mc_df.empty:
        return empty_artifacts
    if not {"Modelo", "Gap_Size"}.issubset(results_mc_df.columns):
        return empty_artifacts

    gap_performance, overall_performance = _model_performance_tables(
        results_mc_df, metrics
    )
    if gap_performance.empty or overall_performance.empty:
        return empty_artifacts
    gap_performance.to_csv(
        output_dir / "model_performance_by_gap.csv", index=False
    )
    overall_performance.to_csv(
        output_dir / "overall_model_performance.csv", index=False
    )

    global_rank_summary, global_rank_frequencies = _global_rank_frequency_tables(
        results_mc_df, metrics
    )
    error_metrics = [
        metric for metric in metrics if not metric_higher_is_better(metric)
    ]
    performance_profiles = _global_performance_profile_table(
        results_mc_df, error_metrics
    )
    station_errors = _global_station_error_table(results_mc_df, metrics)
    pairwise_rates = _pairwise_win_rate_table(results_mc_df, metrics)
    gap_degradation = _gap_degradation_table(results_mc_df, error_metrics)
    tail_risk = _tail_risk_table(results_mc_df, error_metrics)
    error_correlations = _error_correlation_table(results_mc_df, metrics)
    global_rank_summary.to_csv(
        output_dir / "global_rank_summary.csv", index=False
    )
    global_rank_frequencies.to_csv(
        output_dir / "global_rank_frequencies.csv", index=False
    )
    performance_profiles.to_csv(
        output_dir / "global_performance_profiles.csv", index=False
    )
    station_errors.to_csv(
        output_dir / "global_station_errors.csv", index=False
    )
    pairwise_rates.to_csv(
        output_dir / "pairwise_win_rates.csv", index=False
    )
    gap_degradation.to_csv(output_dir / "gap_degradation.csv", index=False)
    tail_risk.to_csv(output_dir / "tail_risk.csv", index=False)
    error_correlations.to_csv(output_dir / "error_correlations.csv", index=False)

    scaled_errors = "Scale_Std" in results_mc_df.columns
    return {
        "model_performance_by_gap_plot_paths": (
            _save_model_performance_by_gap_plots(
                gap_performance,
                overall_performance,
                metrics,
                output_dir=output_dir,
                scaled_errors=scaled_errors,
            )
        ),
        "overall_model_performance_plot_paths": (
            _save_overall_model_performance_plots(
                overall_performance,
                metrics,
                output_dir=output_dir,
                scaled_errors=scaled_errors,
            )
        ),
        "global_station_error_plot_paths": _save_global_station_error_plots(
            station_errors,
            overall_performance,
            metrics,
            output_dir=output_dir,
            scaled_errors=scaled_errors,
        ),
        "pairwise_win_rate_plot_paths": _save_pairwise_win_rate_plots(
            pairwise_rates,
            overall_performance,
            metrics,
            output_dir=output_dir,
            scaled_errors=scaled_errors,
        ),
        "gap_degradation_plot_paths": _save_gap_degradation_plots(
            gap_degradation,
            overall_performance,
            error_metrics,
            output_dir=output_dir,
            scaled_errors=scaled_errors,
        ),
        "tail_risk_plot_paths": _save_tail_risk_plots(
            tail_risk,
            error_metrics,
            output_dir=output_dir,
            scaled_errors=scaled_errors,
        ),
        "error_correlation_plot_paths": _save_error_correlation_plots(
            error_correlations,
            overall_performance,
            metrics,
            output_dir=output_dir,
            scaled_errors=scaled_errors,
        ),
        "diagnostic_table_paths": {
            "global_rank_summary": output_dir / "global_rank_summary.csv",
            "global_rank_frequencies": output_dir / "global_rank_frequencies.csv",
            "global_performance_profiles": (
                output_dir / "global_performance_profiles.csv"
            ),
            "global_station_errors": output_dir / "global_station_errors.csv",
            "pairwise_win_rates": output_dir / "pairwise_win_rates.csv",
            "gap_degradation": output_dir / "gap_degradation.csv",
            "tail_risk": output_dir / "tail_risk.csv",
            "error_correlations": output_dir / "error_correlations.csv",
            "model_performance_by_gap": (
                output_dir / "model_performance_by_gap.csv"
            ),
            "overall_model_performance": (
                output_dir / "overall_model_performance.csv"
            ),
        },
    }


def _sanitize_filename(text: str) -> str:
    """Reduce free-form series names to a safe filesystem identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return sanitized.strip("._") or "series"


def _comparison_plot_files(run_dir: Path) -> dict[Path, Path]:
    """Return top-level run plots plus the station-mean grid.

    The per-station files under ``plots/gap_*`` are deliberately not included:
    the station-mean grid is the useful cross-run summary for those gaps and
    avoids producing a second, very large set of nearly duplicate figures.
    """
    files = {
        path.relative_to(run_dir): path
        for path in run_dir.glob("*.png")
        if path.is_file()
    }
    station_grid = run_dir / "plots" / "media_estaciones_grid.png"
    if station_grid.is_file():
        files[Path("plots") / station_grid.name] = station_grid
    return dict(sorted(files.items(), key=lambda item: str(item[0])))


def _default_comparison_output_dir(left_run_dir: Path, right_run_dir: Path) -> Path:
    """Build a deterministic sibling directory for a two-run comparison."""
    name = (
        f"comparison_{_sanitize_filename(left_run_dir.name)}"
        f"_vs_{_sanitize_filename(right_run_dir.name)}"
    )
    return left_run_dir.parent / name


def _validate_comparison_output_dir(
    output_dir: Path,
    left_run_dir: Path,
    right_run_dir: Path,
) -> None:
    """Keep generated comparison files outside both source runs."""
    for run_dir in (left_run_dir, right_run_dir):
        try:
            output_dir.relative_to(run_dir)
        except ValueError:
            continue
        raise ValueError(
            f"La salida de comparacion debe estar fuera de la ejecucion: {run_dir}"
        )


def _load_comparison_image(path: Path | None) -> np.ndarray | None:
    """Read a PNG for display and return ``None`` for a missing artifact."""
    if path is None:
        return None
    try:
        image = plt.imread(path)
    except (OSError, ValueError):
        return None
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        return None
    return image


def _comparison_content_start(artifact: Path, image: np.ndarray) -> int:
    """Locate the plot content after the source title and description."""
    if artifact.parts == ("plots", "media_estaciones_grid.png"):
        fraction = 0.115
    elif artifact.name == "metrics_by_gap.png" or (
        artifact.name.startswith("metrics_by_gap_")
        and artifact.stem.rsplit("_", 1)[-1].lower()
        in {"ci95", "mean_ci95", "mean_sd", "risk", "sd"}
    ):
        fraction = 0.14
    elif artifact.name.startswith("metrics_by_gap_"):
        fraction = 0.19
    else:
        fraction = 0.14
    return max(1, min(image.shape[0] - 1, int(image.shape[0] * fraction)))


def _comparison_header_end(image: np.ndarray) -> int:
    """Keep only the source title and description for the shared header."""
    return max(1, min(image.shape[0] - 1, int(image.shape[0] * 0.11)))


def _comparison_rgb(image: np.ndarray) -> np.ndarray:
    """Return the visible RGB channels as floating-point values."""
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    rgb = image[:, :, :3]
    return rgb / 255.0 if np.issubdtype(rgb.dtype, np.integer) else rgb


def _first_ink_fraction(image: np.ndarray, *, background: str) -> float | None:
    """Find the first column containing visible content over a flat background."""
    rgb = _comparison_rgb(image)
    background_rgb = np.asarray(to_rgb(background), dtype=float)
    ink = np.max(np.abs(rgb - background_rgb), axis=2) > 0.035
    columns = np.flatnonzero(ink.any(axis=0))
    return float(columns[0] / image.shape[1]) if len(columns) else None


def _plot_left_fraction(artifact: Path, image: np.ndarray) -> float:
    """Estimate the left edge of the actual plot, ignoring y-axis labels."""
    content_start = _comparison_content_start(artifact, image)
    content = image[content_start:]
    rgb = _comparison_rgb(content)
    figure_rgb = np.asarray(to_rgb(FIGURE_FACE), dtype=float)
    differs_from_figure = np.max(np.abs(rgb - figure_rgb), axis=2) > 0.035
    counts = differs_from_figure.sum(axis=0)
    minimum_count = max(5, int(content.shape[0] * 0.12))
    columns = np.flatnonzero(counts >= minimum_count)
    return float(columns[0] / image.shape[1]) if len(columns) else 0.05


def _comparison_header_shift(
    artifact: Path,
    header_image: np.ndarray,
    content_image: np.ndarray | None,
) -> float:
    """Align shared header text with the left edge of the left plot panel."""
    header_ink = _first_ink_fraction(header_image, background=FIGURE_FACE)
    if header_ink is None or content_image is None:
        return 0.0
    plot_left = _plot_left_fraction(artifact, content_image)
    target = 0.01 + 0.5 * plot_left
    # The shared header spans both panels, while its source margin belongs to
    # one panel.  This small correction keeps its text over the left plot.
    return header_ink - target + 0.01


def _shift_header_image(image: np.ndarray, fraction: float) -> np.ndarray:
    """Shift header pixels horizontally without changing their scale."""
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    shift = int(round(fraction * image.shape[1]))
    if shift == 0:
        return image
    shifted = np.empty_like(image)
    background = np.asarray(to_rgb(FIGURE_FACE), dtype=shifted.dtype)
    if np.issubdtype(shifted.dtype, np.integer):
        background *= 255
    shifted[:, :, :3] = background
    if shifted.shape[2] > 3:
        shifted[:, :, 3] = 1.0
    if shift > 0:
        shifted[:, :-shift] = image[:, shift:]
    else:
        shifted[:, -shift:] = image[:, :shift]
    return shifted


def _duplicate_legend_regions(artifact: Path) -> tuple[tuple[float, float, float, float], ...]:
    """Return source regions occupied by identical legends or colorbars."""
    if artifact.name.startswith("gap_degradation_"):
        return ((0.77, 0.20, 1.0, 0.82),)
    if artifact.name.startswith("tail_risk_"):
        return ((0.67, 0.22, 1.0, 0.78),)
    if artifact.name.startswith("overall_model_performance_"):
        return ((0.92, 0.82, 1.0, 0.99),)
    if artifact.name.startswith("pairwise_win_rate_"):
        return ((0.76, 0.20, 0.86, 0.90),)
    if artifact.name.startswith("error_correlation_"):
        return ((0.70, 0.20, 0.82, 0.90),)
    return ()


def _remove_duplicate_legends(image: np.ndarray, artifact: Path) -> np.ndarray:
    """Mask only the second copy of known, identical legends or colorbars."""
    regions = _duplicate_legend_regions(artifact)
    if not regions or image.ndim == 2:
        return image
    clean = image.copy()
    height, width = image.shape[:2]
    background = np.asarray(to_rgb(FIGURE_FACE), dtype=clean.dtype)
    if np.issubdtype(clean.dtype, np.integer):
        background *= 255
    for left, top, right, bottom in regions:
        x0, x1 = int(left * width), int(right * width)
        y0, y1 = int(top * height), int(bottom * height)
        clean[y0:y1, x0:x1, :3] = background
        if clean.shape[2] > 3:
            clean[y0:y1, x0:x1, 3] = 1.0
    return clean


def _comparison_run_label(run_dir: Path) -> str:
    """Use the pollutant code in a run name when it is available."""
    match = re.search(r"(?<![A-Za-z0-9])(no2|o3)(?![A-Za-z0-9])", run_dir.name, re.I)
    return match.group(1).upper() if match else run_dir.name


def _comparison_panel_title(artifact: Path, run_dir: Path) -> str:
    """Label a panel with its metric and pollutant instead of the run id."""
    metric_match = re.search(r"_(mase|rmsse|r2)(?:_|\.png$)", artifact.name, re.I)
    run_label = _comparison_run_label(run_dir)
    return f"{metric_match.group(1).upper()} {run_label}" if metric_match else run_label


def _save_comparison_image(
    left_path: Path | None,
    right_path: Path | None,
    *,
    left_run_dir: Path,
    right_run_dir: Path,
    artifact: Path,
    output_path: Path,
) -> Path:
    """Save one figure with a shared header and two run panels."""
    images = (
        _load_comparison_image(left_path),
        _load_comparison_image(right_path),
    )
    header_image = next((image for image in images if image is not None), None)
    content_image = images[0] if images[0] is not None else images[1]
    header_end = (
        _comparison_header_end(header_image)
        if header_image is not None
        else None
    )
    figure_width = 18.0
    header_units = (
        figure_width * header_end / header_image.shape[1]
        if header_image is not None and header_end is not None
        else 1.2
    )
    figure = plt.figure(
        figsize=(figure_width, max(9.0, header_units + 8.0)),
        facecolor=FIGURE_FACE,
    )
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(header_units, 8.0),
        hspace=0.08,
        wspace=0.03,
    )
    header_axis = figure.add_subplot(grid[0, :])
    header_axis.set_facecolor(FIGURE_FACE)
    if header_image is not None and header_end is not None:
        shared_header = header_image[:header_end]
        header_shift = _comparison_header_shift(
            artifact,
            shared_header,
            content_image,
        )
        header_axis.imshow(
            _shift_header_image(shared_header, header_shift),
            aspect="auto",
        )
    else:
        header_axis.text(
            0.5,
            0.5,
            "No hay cabecera disponible",
            ha="center",
            va="center",
            color=MUTED_TEXT,
            transform=header_axis.transAxes,
        )
    header_axis.set_axis_off()

    for panel_index, (axis, image, run_dir, source_path) in enumerate(zip(
        (figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])),
        images,
        (left_run_dir, right_run_dir),
        (left_path, right_path),
        strict=True,
    )):
        axis.set_facecolor(AXIS_FACE)
        axis.set_title(
            _comparison_panel_title(artifact, run_dir),
            color=TEXT_COLOR,
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        if image is None:
            axis.text(
                0.5,
                0.5,
                (
                    "Figura no disponible"
                    if source_path is None
                    else "No se pudo leer la figura"
                ),
                ha="center",
                va="center",
                color=MUTED_TEXT,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue
        if panel_index == 1 and images[0] is not None:
            image = _remove_duplicate_legends(image, artifact)
        content_start = _comparison_content_start(artifact, image)
        axis.imshow(image[content_start:], aspect="auto")
        axis.set_axis_off()

    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return save_figure(figure, output_path)


def render_comparison_figures(
    left_run_dir: Path,
    right_run_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Render all available PNG reports from two Monte Carlo run directories.

    Top-level report images are paired by filename and share one title and
    description taken from the first available source.  The special station-
    mean grid at ``plots/media_estaciones_grid.png`` is paired as well, while
    the individual station images under ``plots/gap_*`` are intentionally
    omitted.  The source runs are never modified.
    """
    left_run_dir = Path(left_run_dir).expanduser().resolve()
    right_run_dir = Path(right_run_dir).expanduser().resolve()
    if not left_run_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio izquierdo: {left_run_dir}")
    if not right_run_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio derecho: {right_run_dir}")
    if left_run_dir == right_run_dir:
        raise ValueError("Las dos ejecuciones deben ser directorios distintos")

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_comparison_output_dir(left_run_dir, right_run_dir).resolve()
    )
    _validate_comparison_output_dir(output_dir, left_run_dir, right_run_dir)

    left_files = _comparison_plot_files(left_run_dir)
    right_files = _comparison_plot_files(right_run_dir)
    artifacts = sorted(set(left_files) | set(right_files), key=str)
    if not artifacts:
        raise FileNotFoundError(
            "No se encontraron figuras PNG en ninguna de las dos ejecuciones"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    plot_paths: dict[str, Path] = {}
    for artifact in artifacts:
        left_path = left_files.get(artifact)
        right_path = right_files.get(artifact)
        output_path = output_dir / artifact
        _save_comparison_image(
            left_path,
            right_path,
            left_run_dir=left_run_dir,
            right_run_dir=right_run_dir,
            artifact=artifact,
            output_path=output_path,
        )
        relative_output = str(output_path.relative_to(output_dir))
        plot_paths[str(artifact)] = output_path
        rows.append(
            {
                "artifact": str(artifact),
                "left_source": str(left_path) if left_path is not None else "",
                "right_source": str(right_path) if right_path is not None else "",
                "left_exists": left_path is not None,
                "right_exists": right_path is not None,
                "output_path": relative_output,
            }
        )

    manifest_df = pd.DataFrame(
        rows,
        columns=[
            "artifact",
            "left_source",
            "right_source",
            "left_exists",
            "right_exists",
            "output_path",
        ],
    )
    manifest_path = output_dir / "comparison_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return {
        "output_dir": output_dir,
        "comparison_plot_paths": plot_paths,
        "plot_paths": plot_paths,
        "manifest_df": manifest_df,
        "manifest_path": manifest_path,
        "left_run_dir": left_run_dir,
        "right_run_dir": right_run_dir,
    }


def _to_pd_series(obj: Any) -> pd.Series:
    """Coerce a Series/TimeSeries-like plot payload into a sorted pandas Series."""
    if isinstance(obj, pd.Series):
        return obj.sort_index()
    if hasattr(obj, "to_series"):
        try:
            series = obj.to_series()
        except Exception:
            return pd.Series(dtype=float)
        if isinstance(series, pd.Series):
            return series.sort_index()
    return pd.Series(dtype=float)


def save_plot_store(
    plot_store: dict[int, dict[str, Any]], output_path: Path
) -> Path:
    """Persist actual values and predictions as a compressed long-form CSV."""
    frames: list[pd.DataFrame] = []
    for gap_size, gap_payload in sorted(plot_store.items()):
        by_series = gap_payload.get("series", {}) if isinstance(gap_payload, dict) else {}
        if not isinstance(by_series, dict):
            continue
        for series_name, payload in by_series.items():
            if not isinstance(payload, dict):
                continue
            series_items = [
                ("actual", "", payload.get("actual")),
                ("naive_mase", "", payload.get("naive_mase")),
            ]
            preds = payload.get("preds", {})
            if isinstance(preds, dict):
                series_items.extend(
                    ("prediction", str(model_name), prediction)
                    for model_name, prediction in preds.items()
                )
            for kind, model_name, value in series_items:
                series = _to_pd_series(value)
                if series.empty:
                    continue
                frames.append(
                    pd.DataFrame(
                        {
                            "gap_size": int(gap_size),
                            "series_name": str(series_name),
                            "kind": kind,
                            "model_name": model_name,
                            "timestamp": series.index,
                            "value": series.to_numpy(),
                        }
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=PLOT_STORE_COLUMNS)
    )
    frame.to_csv(output_path, index=False, compression="gzip")
    return output_path


def load_plot_store(path: Path) -> dict[int, dict[str, Any]]:
    """Load a plot store previously written by :func:`save_plot_store`."""
    frame = pd.read_csv(path)
    missing = set(PLOT_STORE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"CSV de plots invalido; faltan columnas: {sorted(missing)}")
    if frame.empty:
        return {}

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["gap_size"] = pd.to_numeric(frame["gap_size"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame[["timestamp", "gap_size"]].isna().any().any():
        raise ValueError("CSV de plots invalido; contiene timestamps o gaps no validos")

    def _restore(rows: pd.DataFrame) -> pd.Series:
        return pd.Series(
            rows["value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(rows["timestamp"]),
            dtype=float,
        ).sort_index()

    plot_store: dict[int, dict[str, Any]] = {}
    for (gap_size, series_name), rows in frame.groupby(
        ["gap_size", "series_name"], sort=False
    ):
        payload: dict[str, Any] = {
            "actual": _restore(rows[rows["kind"] == "actual"]),
            "naive_mase": _restore(rows[rows["kind"] == "naive_mase"]),
            "preds": {},
        }
        prediction_rows = rows[rows["kind"] == "prediction"]
        for model_name, model_rows in prediction_rows.groupby("model_name", sort=False):
            payload["preds"][str(model_name)] = _restore(model_rows)
        plot_store.setdefault(int(gap_size), {"series": {}})["series"][str(series_name)] = payload
    return plot_store


def _draw_station_mean_boxplot(
    ax: "plt.Axes",
    model_values: dict[str, np.ndarray],
    *,
    model_order: list[str] | None = None,
) -> list[str]:
    """Draw the station-level MASE distributions used by gap summaries."""
    model_scores = {
        model_name: float(values.mean())
        for model_name, values in model_values.items()
    }
    if model_order is None:
        ranked_models = sorted(
            model_scores,
            key=lambda name: (model_scores[name], name),
        )
    else:
        ranked_models = [
            model_name for model_name in model_order if model_name in model_values
        ]
        ranked_models.extend(
            sorted(
                set(model_values).difference(ranked_models),
                key=lambda name: (model_scores[name], name),
            )
        )

    colors = _model_color_map(ranked_models)
    positions = np.arange(len(ranked_models))
    for position, model_name in zip(positions, ranked_models, strict=True):
        box = ax.boxplot(
            model_values[model_name],
            positions=[position],
            widths=0.62,
            orientation="horizontal",
            patch_artist=True,
            showmeans=True,
            meanline=False,
            boxprops={"facecolor": colors[model_name], "alpha": 0.72},
            medianprops={"color": TEXT_COLOR, "linewidth": 1.5},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#f3b43f",
                "markeredgecolor": TEXT_COLOR,
                "markersize": 5,
            },
            whiskerprops={"color": colors[model_name], "linewidth": 1.2},
            capprops={"color": colors[model_name], "linewidth": 1.2},
            flierprops={
                "marker": ".",
                "markerfacecolor": colors[model_name],
                "markeredgecolor": colors[model_name],
                "alpha": 0.45,
                "markersize": 4,
            },
        )
        for artist in box["boxes"]:
            artist.set_edgecolor(colors[model_name])
        ax.annotate(
            f"media={model_scores[model_name]:.2f}",
            (model_scores[model_name], position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=TEXT_COLOR,
        )

    ax.axvline(1.0, color="#d6453c", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.set_xlabel("MASE medio por estacion")
    ax.set_ylabel("Modelo")
    ax.set_yticks(positions)
    ax.set_yticklabels(ranked_models, fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.55)
    ax.grid(False, axis="y")
    ax.margins(x=0.14)
    return ranked_models


def _save_station_mean_grid(
    station_values_by_gap: dict[int, dict[str, np.ndarray]],
    *,
    output_dir: Path,
) -> Path | None:
    """Save a 2x2 comparison of station-level MASE distributions."""
    grid_path = output_dir / "plots" / "media_estaciones_grid.png"
    requested_gaps = (1, 2, 5, 10)
    available_values = {
        gap: station_values_by_gap[gap]
        for gap in requested_gaps
        if gap in station_values_by_gap
    }
    if not available_values:
        grid_path.unlink(missing_ok=True)
        return None

    combined_values: dict[str, list[float]] = {}
    for model_values in available_values.values():
        for model_name, values in model_values.items():
            combined_values.setdefault(model_name, []).extend(values.tolist())
    global_order = sorted(
        combined_values,
        key=lambda name: (float(np.mean(combined_values[name])), name),
    )
    finite_values = np.concatenate(
        [
            values
            for model_values in available_values.values()
            for values in model_values.values()
        ]
    )
    x_max = max(1.15, float(np.nanmax(finite_values)) * 1.12)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.0, 13.5),
        facecolor=FIGURE_FACE,
        sharex=True,
    )
    axes_flat = axes.ravel()
    for ax, gap in zip(axes_flat, requested_gaps, strict=True):
        _style_metric_axis(ax)
        model_values = station_values_by_gap.get(gap)
        if not model_values:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"Sin datos para huecos de {gap} h",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=MUTED_TEXT,
            )
            continue
        _draw_station_mean_boxplot(
            ax,
            model_values,
            model_order=global_order,
        )
        ax.set_title(f"Huecos de {gap} h", fontsize=11.5, fontweight="bold")
        ax.set_xlim(0.0, x_max)

    add_plot_header(
        fig,
        "MASE medio entre estaciones | Comparación por tamaño de hueco",
        "Cada panel resume el MASE medio de cada estación tras promediar sus semillas. "
        "El diamante indica la media y la línea roja marca MASE = 1.\n"
        "Cómo leer: cajas más a la izquierda indican menor error; una caja más corta "
        "indica mayor homogeneidad entre estaciones.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, grid_path)
    return grid_path


def _save_plot_images(
    plot_store: dict[int, dict[str, Any]],
    *,
    output_dir: Path,
    results_mc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Render one PNG per (gap size, series) with real values vs. model predictions.

    Writes the images under ``<output_dir>/plots/gap_<n>/`` plus a
    ``plot_images.csv`` manifest, which is also returned as a dataframe.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    example_seed: int | None = None
    if results_mc_df is not None and not results_mc_df.empty:
        if "Seed" in results_mc_df.columns:
            seed_rows = results_mc_df
            if "MonteCarlo_Run" in seed_rows.columns:
                first_run = pd.to_numeric(
                    seed_rows["MonteCarlo_Run"], errors="coerce"
                ).min()
                seed_rows = seed_rows[
                    pd.to_numeric(seed_rows["MonteCarlo_Run"], errors="coerce")
                    == first_run
                ]
            seeds = pd.to_numeric(seed_rows["Seed"], errors="coerce").dropna()
            if len(seeds):
                example_seed = int(seeds.iloc[0])

    rows: list[dict[str, Any]] = []
    for gap_size, gap_payload in sorted(plot_store.items()):
        by_series = gap_payload.get("series", {}) if isinstance(gap_payload, dict) else {}
        if not isinstance(by_series, dict):
            continue

        gap_dir = plots_dir / f"gap_{int(gap_size)}"
        gap_dir.mkdir(parents=True, exist_ok=True)

        for series_name, series_payload in by_series.items():
            if not isinstance(series_payload, dict):
                continue

            actual = _to_pd_series(series_payload.get("actual", pd.Series(dtype=float)))
            preds_payload = series_payload.get("preds", {})
            preds_by_model = preds_payload if isinstance(preds_payload, dict) else {}

            mask_index = pd.DatetimeIndex([])
            naive_mase = _to_pd_series(series_payload.get("naive_mase", pd.Series(dtype=float)))
            if len(naive_mase) > 0:
                mask_index = pd.DatetimeIndex(naive_mase.index)

            clean_preds: dict[str, pd.Series] = {}
            for model_name, pred in preds_by_model.items():
                pred_series = _to_pd_series(pred).dropna()
                if len(pred_series) > 0:
                    clean_preds[str(model_name)] = pred_series
                    mask_index = mask_index.union(pd.DatetimeIndex(pred_series.index))

            gap_real = actual.reindex(mask_index).dropna().sort_index()
            model_values: dict[str, np.ndarray] = {}
            if (
                results_mc_df is not None
                and {"Modelo", "Serie", "Gap_Size", "MASE"}.issubset(
                    results_mc_df.columns
                )
            ):
                metric_rows = results_mc_df[
                    (results_mc_df["Serie"].astype(str) == str(series_name))
                    & (
                        pd.to_numeric(results_mc_df["Gap_Size"], errors="coerce")
                        == int(gap_size)
                    )
                ].copy()
                metric_rows["MASE"] = pd.to_numeric(
                    metric_rows["MASE"], errors="coerce"
                )
                sample_columns = [
                    column
                    for column in ("MonteCarlo_Run", "Seed")
                    if column in metric_rows.columns
                ]
                if sample_columns:
                    metric_rows = metric_rows.groupby(
                        ["Modelo", *sample_columns], as_index=False, sort=False
                    )["MASE"].mean()
                for model_name, model_rows in metric_rows.groupby("Modelo", sort=False):
                    values = model_rows["MASE"].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if len(values):
                        model_values[str(model_name)] = values

            # Old plot stores can be rendered without results_mc.csv metrics.
            # In that case, retain an explicitly unscaled absolute-error panel.
            metric_name = "MASE" if model_values else "Error absoluto"
            if not model_values:
                for model_name, pred_series in clean_preds.items():
                    common = gap_real.index.intersection(pred_series.index)
                    if len(common) > 0:
                        errors = np.abs(
                            gap_real.reindex(common).to_numpy(dtype=float)
                            - pred_series.reindex(common).to_numpy(dtype=float)
                        )
                        errors = errors[np.isfinite(errors)]
                        if len(errors):
                            model_values[model_name] = errors

            model_scores = {
                model_name: float(values.mean())
                for model_name, values in model_values.items()
            }

            ranked_models = sorted(model_scores, key=lambda name: (model_scores[name], name))
            colors = _model_color_map(list(clean_preds))

            fig = plt.figure(figsize=(13.2, 8.4), facecolor=FIGURE_FACE)
            grid = fig.add_gridspec(
                2, 1, height_ratios=(0.9, 1.55), hspace=0.42,
            )
            context_ax = fig.add_subplot(grid[0, :])
            error_ax = fig.add_subplot(grid[1, 0])

            for axis in (context_ax, error_ax):
                _style_metric_axis(axis)

            if len(actual) > 0:
                context_ax.plot(
                    actual.index, actual.values, color="#8f8b84", alpha=0.72,
                    lw=0.9, label="Serie observada",
                )
            if len(gap_real) > 0:
                context_ax.scatter(
                    gap_real.index, gap_real.values, color="#d6453c", s=27,
                    edgecolor=AXIS_FACE, linewidth=0.7, zorder=3,
                    label="Valores ocultados",
                )
            context_ax.set_title("Contexto temporal", loc="left", fontsize=11, fontweight="bold")
            context_ax.set_ylabel("NO2")
            context_ax.legend(loc="upper left", ncol=2, fontsize=8)
            context_ax.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.45)
            context_ax.tick_params(axis="x", labelrotation=0)

            gap_index = pd.DatetimeIndex(gap_real.index)
            boundaries = [0]
            if len(gap_index) > 1:
                actual_steps = pd.DatetimeIndex(actual.index).to_series().diff().dropna()
                cadence = actual_steps.median() if len(actual_steps) else None
                if cadence is not None and cadence > pd.Timedelta(0):
                    boundaries.extend(
                        index for index, delta in enumerate(gap_index[1:] - gap_index[:-1], start=1)
                        if delta > cadence * 1.5
                    )
            boundaries.append(len(gap_real))
            gap_count = max(0, len(boundaries) - 1)

            error_models = [model for model in ranked_models if model in model_values]
            positions = np.arange(len(error_models))
            for position, model_name in zip(positions, error_models, strict=True):
                box = error_ax.boxplot(
                    model_values[model_name],
                    positions=[position],
                    widths=0.62,
                    orientation="horizontal",
                    patch_artist=True,
                    showmeans=True,
                    meanline=False,
                    boxprops={"facecolor": colors[model_name], "alpha": 0.72},
                    medianprops={"color": TEXT_COLOR, "linewidth": 1.5},
                    meanprops={
                        "marker": "D",
                        "markerfacecolor": "#f3b43f",
                        "markeredgecolor": TEXT_COLOR,
                        "markersize": 5,
                    },
                    whiskerprops={"color": colors[model_name], "linewidth": 1.2},
                    capprops={"color": colors[model_name], "linewidth": 1.2},
                    flierprops={
                        "marker": ".",
                        "markerfacecolor": colors[model_name],
                        "markeredgecolor": colors[model_name],
                        "alpha": 0.45,
                        "markersize": 4,
                    },
                )
                for artist in box["boxes"]:
                    artist.set_edgecolor(colors[model_name])
                error_ax.annotate(
                    f"media={model_scores[model_name]:.2f}",
                    (model_scores[model_name], position),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7.5,
                    color=TEXT_COLOR,
                )
            error_ax.set_title(
                (
                    "Distribucion del MASE entre semillas por modelo"
                    if metric_name == "MASE"
                    else "Distribucion del error absoluto por modelo"
                ),
                loc="left", fontsize=11, fontweight="bold",
            )
            error_ax.set_xlabel(
                "MASE" if metric_name == "MASE" else "Error absoluto en los valores ocultados"
            )
            error_ax.set_ylabel("Modelo")
            error_ax.set_yticks(positions)
            error_ax.set_yticklabels(error_models, fontsize=8)
            error_ax.invert_yaxis()
            error_ax.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.55)
            error_ax.grid(False, axis="y")
            error_ax.text(
                0.99,
                1.02,
                "caja: P25-P75 | linea: mediana | diamante: media",
                transform=error_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=MUTED_TEXT,
            )
            error_ax.margins(x=0.14)

            add_plot_header(
                fig,
                f"Imputacion de {series_name} | Huecos de {int(gap_size)} h",
                f"{gap_count} huecos, {len(gap_real)} puntos ocultados. "
                "Arriba se muestra una semilla representativa; abajo se resumen todas las semillas.\n"
                "Como leer: cajas mas a la izquierda y estrechas indican menor MASE y mayor estabilidad entre semillas.",
            )
            fig.subplots_adjust(left=0.07, right=0.96, bottom=0.09, top=0.82)

            image_path = gap_dir / f"{_sanitize_filename(str(series_name))}.png"
            save_figure(fig, image_path)

            rows.append(
                {
                    "gap_size": int(gap_size),
                    "series_name": str(series_name),
                    "seed": example_seed,
                    "gap_count": gap_count,
                    "target_points": len(gap_real),
                    "model_count": len(ranked_models),
                    "image_path": str(image_path.relative_to(output_dir)),
                }
            )

    if (
        results_mc_df is not None
        and {"Modelo", "Serie", "Gap_Size", "MASE"}.issubset(results_mc_df.columns)
    ):
        _, station_means = _station_seed_metric_tables(results_mc_df, ["MASE"])
        station_values_by_gap: dict[int, dict[str, np.ndarray]] = {}
        for gap_size, gap_rows in station_means.groupby("Gap_Size", sort=True):
            model_values: dict[str, np.ndarray] = {}
            for model_name, model_rows in gap_rows.groupby("Modelo", sort=False):
                values = model_rows["MASE"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    model_values[str(model_name)] = values
            if not model_values:
                continue

            gap_size = int(gap_size)
            station_values_by_gap[gap_size] = model_values

            fig, ax = plt.subplots(figsize=(13.2, 7.8), facecolor=FIGURE_FACE)
            _style_metric_axis(ax)
            ranked_models = _draw_station_mean_boxplot(ax, model_values)
            add_plot_header(
                fig,
                f"MASE medio entre estaciones | Huecos de {gap_size} h",
                "Cada observacion es el MASE medio de una estacion tras promediar sus semillas.\n"
                "El diamante indica la media de todas las estaciones; la linea roja marca MASE = 1.",
            )
            fig.subplots_adjust(left=0.18, right=0.96, bottom=0.1, top=0.82)

            gap_dir = plots_dir / f"gap_{gap_size}"
            gap_dir.mkdir(parents=True, exist_ok=True)
            image_path = gap_dir / "media_estaciones.png"
            save_figure(fig, image_path)
            rows.append(
                {
                    "gap_size": gap_size,
                    "series_name": "Media de estaciones",
                    "seed": None,
                    "gap_count": None,
                    "target_points": None,
                    "model_count": len(ranked_models),
                    "image_path": str(image_path.relative_to(output_dir)),
                }
            )
        _save_station_mean_grid(station_values_by_gap, output_dir=output_dir)
    else:
        (output_dir / "plots" / "media_estaciones_grid.png").unlink(missing_ok=True)

    manifest_df = pd.DataFrame(
        rows,
        columns=[
            "gap_size",
            "series_name",
            "seed",
            "gap_count",
            "target_points",
            "model_count",
            "image_path",
        ],
    )
    manifest_df.to_csv(output_dir / "plot_images.csv", index=False)
    return manifest_df


def render_run_figures(
    output_dir: Path,
    *,
    results_mc_df: pd.DataFrame | None = None,
    plot_store: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render every figure from in-memory data or persisted run artifacts."""
    output_dir = Path(output_dir)
    if results_mc_df is None:
        results_mc_df = pd.read_csv(output_dir / "results_mc.csv")
    if plot_store is None:
        plot_store = load_plot_store(output_dir / "plot_store.csv.gz")
    metric_gap_plot_path, metric_gap_plot_paths = _save_metric_gap_plots(
        results_mc_df,
        output_dir=output_dir,
    )
    plot_manifest_df = _save_plot_images(
        plot_store,
        output_dir=output_dir,
        results_mc_df=results_mc_df,
    )
    station_mean_grid_path = output_dir / "plots" / "media_estaciones_grid.png"
    if not station_mean_grid_path.exists():
        station_mean_grid_path = None
    metric_gap_uncertainty_plot_paths = {
        name: output_dir / filename
        for name, filename in {
            "SD": "metrics_by_gap_sd.png",
            "CI95": "metrics_by_gap_ci95.png",
        }.items()
        if (output_dir / filename).exists()
    }
    metric_gap_uncertainty_table_path = output_dir / "metrics_by_gap_uncertainty.csv"
    if not metric_gap_uncertainty_table_path.exists():
        metric_gap_uncertainty_table_path = None
    metric_gap_mean_ci95_path = output_dir / "metrics_by_gap_mean_ci95.png"
    if not metric_gap_mean_ci95_path.exists():
        metric_gap_mean_ci95_path = None
    metric_gap_mean_sd_path = output_dir / "metrics_by_gap_mean_sd.png"
    if not metric_gap_mean_sd_path.exists():
        metric_gap_mean_sd_path = None
    metric_gap_risk_path = output_dir / "metrics_by_gap_risk.png"
    if not metric_gap_risk_path.exists():
        metric_gap_risk_path = None
    metric_gap_risk_table_path = output_dir / "metrics_by_gap_risk.csv"
    if not metric_gap_risk_table_path.exists():
        metric_gap_risk_table_path = None
    existing_artifacts = {
        "metric_gap_plot_path": metric_gap_plot_path,
        "metric_gap_plot_paths": metric_gap_plot_paths,
        "metric_gap_uncertainty_plot_paths": metric_gap_uncertainty_plot_paths,
        "metric_gap_uncertainty_table_path": metric_gap_uncertainty_table_path,
        "metric_gap_mean_ci95_path": metric_gap_mean_ci95_path,
        "metric_gap_mean_sd_path": metric_gap_mean_sd_path,
        "metric_gap_risk_path": metric_gap_risk_path,
        "metric_gap_risk_table_path": metric_gap_risk_table_path,
        "plot_manifest_df": plot_manifest_df,
        "station_mean_grid_path": station_mean_grid_path,
    }
    diagnostic_artifacts = _save_montecarlo_diagnostic_plots(
        results_mc_df,
        output_dir=output_dir,
    )
    return {**existing_artifacts, **diagnostic_artifacts}


def main(argv: list[str] | None = None) -> None:
    """Regenerate one run or compare the figures from two existing runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        help="Una ejecucion para regenerar o dos ejecuciones para comparar",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directorio de salida para la comparacion de dos ejecuciones",
    )
    args = parser.parse_args(argv)
    if len(args.run_dirs) > 2:
        parser.error("indica una o dos ejecuciones, no mas")
    if len(args.run_dirs) == 2:
        artifacts = render_comparison_figures(
            args.run_dirs[0],
            args.run_dirs[1],
            output_dir=args.output_dir,
        )
        print(f"[info] Comparacion guardada en {artifacts['output_dir']}")
        print(f"[info] Manifiesto: {artifacts['manifest_path']}")
        print(
            f"[info] Generadas {len(artifacts['manifest_df'])} figuras comparativas"
        )
        return

    if args.output_dir is not None:
        parser.error("--output-dir solo se puede usar al comparar dos ejecuciones")
    run_dir = args.run_dirs[0]
    if not run_dir.is_dir():
        raise SystemExit(f"No existe el directorio: {run_dir}")

    artifacts = render_run_figures(run_dir)
    metric_path = artifacts["metric_gap_plot_path"]
    if metric_path is not None:
        print(f"[info] Figura guardada en {metric_path}")
    for metric, path in artifacts.get("metric_gap_plot_paths", {}).items():
        print(f"[info] Figura separada ({metric}) guardada en {path}")
    if artifacts.get("station_mean_grid_path") is not None:
        print(
            f"[info] Grid de medias por estacion guardado en "
            f"{artifacts['station_mean_grid_path']}"
        )
    for name, path in artifacts.get("metric_gap_uncertainty_plot_paths", {}).items():
        print(f"[info] Figura de incertidumbre ({name}) guardada en {path}")
    if artifacts.get("metric_gap_uncertainty_table_path") is not None:
        print(
            f"[info] Tabla de incertidumbre guardada en "
            f"{artifacts['metric_gap_uncertainty_table_path']}"
        )
    if artifacts.get("metric_gap_mean_ci95_path") is not None:
        print(
            f"[info] Figura de media e IC95 guardada en "
            f"{artifacts['metric_gap_mean_ci95_path']}"
        )
    if artifacts.get("metric_gap_mean_sd_path") is not None:
        print(f"[info] Figura de media y SD guardada en {artifacts['metric_gap_mean_sd_path']}")
    if artifacts.get("metric_gap_risk_path") is not None:
        print(f"[info] Figura de riesgo guardada en {artifacts['metric_gap_risk_path']}")
    if artifacts.get("metric_gap_risk_table_path") is not None:
        print(f"[info] Tabla de riesgo guardada en {artifacts['metric_gap_risk_table_path']}")
    for key in (
        "model_performance_by_gap_plot_paths",
        "overall_model_performance_plot_paths",
        "global_station_error_plot_paths",
        "pairwise_win_rate_plot_paths",
        "gap_degradation_plot_paths",
        "tail_risk_plot_paths",
        "error_correlation_plot_paths",
    ):
        for path in artifacts[key].values():
            print(f"[info] Resumen global guardado en {path}")
    for path in artifacts["diagnostic_table_paths"].values():
        print(f"[info] Tabla diagnostica guardada en {path}")
    manifest = artifacts["plot_manifest_df"]
    print(f"[info] Regeneradas {len(manifest)} figuras en {run_dir / 'plots'}")


if __name__ == "__main__":
    main()
