"""Render and explain the statistical analysis of one forecasting run.

The forecasting benchmark and its statistical contrasts are deliberately
separate steps.  This module consumes the seven CSV tables written by
``airquality.forecasting.statistics`` and creates a small, reproducible report
next to them without rerunning any forecast.

Run with::

    uv run python -m airquality.visualizations.forecasting_statistics RUN_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from airquality.visualizations.style import (
    AXIS_FACE,
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    MUTED_TEXT,
    TEXT_COLOR,
    add_plot_header,
    save_figure,
    style_axis,
)


STATISTICAL_FILES = (
    "model_station_scores.csv",
    "model_omnibus.csv",
    "model_pairwise.csv",
    "branch_station_scores.csv",
    "branch_omnibus.csv",
    "branch_pairwise.csv",
    "statistical_exclusions.csv",
)

OUTCOME_COLORS = {
    "left_lower_typical_error": "#3d7ab5",
    "right_lower_typical_error": "#cf6f1e",
    "inconclusive": "#a89b8c",
}
SCORE_COLORS = (
    "#3d7ab5",
    "#cf6f1e",
    "#5b8c5a",
    "#9b59b6",
    "#2a7f76",
    "#c65353",
    "#8c6d4b",
    "#6677a8",
    "#b07d32",
    "#4c8b89",
)
METRIC_LABELS = {
    "mase": "MASE",
    "rmsse": "RMSSE",
    "relmae": "RelMAE",
    "relrmse": "RelRMSE",
}


def _load_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the persisted statistical tables and fail clearly if one is absent."""
    missing = [name for name in STATISTICAL_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan tablas estadísticas en "
            f"{run_dir}: {', '.join(missing)}. Ejecuta primero "
            "airquality.forecasting.statistics."
        )
    return {
        Path(name).stem: pd.read_csv(run_dir / name, keep_default_na=False)
        for name in STATISTICAL_FILES
    }


def _finite(values: pd.Series) -> np.ndarray:
    """Return finite numeric values from a possibly mixed CSV column."""
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _fmt_p(value: object) -> str:
    """Format a p-value compactly for a report table."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def _fmt_number(value: object, digits: int = 3) -> str:
    """Format a numeric report value while preserving infinities as symbols."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(number):
        return "NA"
    if np.isposinf(number):
        return "infinito"
    if np.isneginf(number):
        return "-infinito"
    return f"{number:.{digits}f}"


def _metric_label(metric: str) -> str:
    """Return the human-readable name of one metric."""
    return METRIC_LABELS.get(metric, metric.upper())


def _outcome_label(value: object) -> str:
    """Translate the machine-readable directional conclusion."""
    return {
        "left_lower_typical_error": "izquierda mejor",
        "right_lower_typical_error": "derecha mejor",
        "inconclusive": "inconcluso",
    }.get(str(value), str(value))


def _score_summary(scores: pd.DataFrame, *, scope: str, metric: str) -> pd.DataFrame:
    """Summarize station-level treatment scores for one panel."""
    subset = scores[(scores["scope"] == scope) & (scores["metric"] == metric)]
    if subset.empty:
        return pd.DataFrame(columns=["treatment", "mean", "median", "std", "n_stations"])
    return (
        subset.groupby("treatment", sort=False)["score"]
        .agg(mean="mean", median="median", std="std", n_stations="count")
        .sort_values(["mean", "median"])
        .reset_index()
    )


def _project_foundation_raw_over_all_branches(
    model_scores: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Add foundation scores to the all-branch model panel.

    Foundation models are evaluated on the raw view rather than refit for each
    training branch.  For the all-branch visualization their raw error is the
    branch value, so the same station-level score is carried over every arm.
    This creates no new forecast; it only makes the requested all-branch panel
    comparable across model families.
    """
    if results.empty or "model_mode" not in results or "arm" not in results:
        return model_scores
    arms = list(dict.fromkeys(results["arm"].astype(str)))
    foundation_raw = results[
        (results["model_mode"].astype(str) == "foundation")
        & (results["arm"].astype(str) == "raw")
    ]
    if foundation_raw.empty or not arms:
        return model_scores

    rows: list[dict[str, object]] = []
    for _, source in foundation_raw.iterrows():
        for metric in ("mase", "rmsse"):
            value = pd.to_numeric(pd.Series([source.get(metric)]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "analysis": "model",
                    "scope": "all_branches",
                    "metric": metric,
                    "series": source["series"],
                    "treatment": source["model"],
                    "score": float(value),
                    "mean_error": float(value),
                    "n_averaged": len(arms),
                    "rank": np.nan,
                    "seed": np.nan,
                }
            )
    if not rows:
        return model_scores
    projected = pd.DataFrame(rows).reindex(columns=model_scores.columns)
    return pd.concat([model_scores, projected], ignore_index=True)


def _build_relative_branch_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Build station-level RelMAE/RelRMSE scores for preprocessing branches."""
    required = {"series", "arm", "model_mode", "relmae", "relrmse"}
    if results.empty or not required <= set(results.columns):
        return pd.DataFrame(
            columns=[
                "analysis",
                "scope",
                "metric",
                "series",
                "treatment",
                "score",
                "mean_error",
                "n_averaged",
                "rank",
                "seed",
            ]
        )
    subset = results[
        results["model_mode"].astype(str) != "foundation"
    ].copy()
    rows: list[dict[str, object]] = []
    for metric in ("relmae", "relrmse"):
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        grouped = (
            subset.dropna(subset=[metric])
            .groupby(["series", "arm"], sort=False)[metric]
            .agg(score="mean", n_averaged="count")
            .reset_index()
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "analysis": "branch",
                    "scope": "common_raw_targets",
                    "metric": metric,
                    "series": row["series"],
                    "treatment": row["arm"],
                    "score": float(row["score"]),
                    "mean_error": float(row["score"]),
                    "n_averaged": int(row["n_averaged"]),
                    "rank": np.nan,
                    "seed": np.nan,
                }
            )
    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores["rank"] = scores.groupby(["metric", "series"], sort=False)["score"].rank(
            method="average", ascending=True
        )
    return scores


def _relative_branch_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize relative branch metrics for the explanatory report."""
    summaries: list[pd.DataFrame] = []
    for metric in ("relmae", "relrmse"):
        summary = _score_summary(scores, scope="common_raw_targets", metric=metric)
        if summary.empty:
            continue
        summaries.append(
            summary.rename(
                columns={
                    "mean": f"{metric}_mean",
                    "median": f"{metric}_median",
                    "std": f"{metric}_std",
                    "n_stations": f"{metric}_n_stations",
                }
            )
        )
    if not summaries:
        return pd.DataFrame()
    combined = summaries[0]
    for summary in summaries[1:]:
        combined = combined.merge(summary, on="treatment", how="outer")
    return combined.sort_values("relmae_mean" if "relmae_mean" in combined else "treatment")


def save_omnibus_plot(output_path: Path, omnibus: pd.DataFrame) -> bool:
    """Plot all omnibus p-values on a common ``-log10(p)`` scale."""
    if omnibus.empty:
        return False
    data = omnibus.copy()
    data["p_value"] = pd.to_numeric(data["p_value"], errors="coerce")
    data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
    data = data[np.isfinite(data["p_value"]) & (data["p_value"] > 0.0)].copy()
    if data.empty:
        return False
    scope_labels = {
        "all_branches": "todas las ramas",
        "common_raw_targets": "ramas",
        "common_source_arms": "raw",
    }
    data["label"] = (
        data["analysis"].str.title()
        + " / "
        + data["scope"].map(scope_labels).fillna(data["scope"].str.replace("_", " "))
        + " / "
        + data["metric"].str.upper()
    )
    data["evidence"] = -np.log10(data["p_value"])
    data = data.iloc[::-1].reset_index(drop=True)

    figure, axis = plt.subplots(
        figsize=(10.0, max(4.2, 0.62 * len(data) + 1.9)),
        facecolor=FIGURE_FACE,
    )
    add_plot_header(
        figure,
        "Evidencia global: contrastes omnibus",
        "Cada punto es un contraste por estaciones. Más a la derecha = menor p-valor. "
        "La línea marca alpha = 0,05.",
    )
    style_axis(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    y = np.arange(len(data))
    colors = np.where(data["p_value"].to_numpy() < data["alpha"].to_numpy(), "#3d7ab5", "#cf6f1e")
    axis.scatter(data["evidence"], y, s=75, c=colors, edgecolors="white", linewidths=0.8, zorder=3)
    alpha = float(data["alpha"].dropna().iloc[0])
    threshold = -np.log10(alpha)
    axis.axvline(threshold, color=MUTED_TEXT, linestyle="--", linewidth=1.2, zorder=2)
    for position, (_, row) in enumerate(data.iterrows()):
        axis.annotate(
            _fmt_p(row["p_value"]),
            (row["evidence"], position),
            textcoords="offset points",
            xytext=(7, 0),
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )
    axis.set_yticks(y, data["label"])
    axis.set_xlabel(r"evidencia $-\log_{10}(p)$")
    axis.set_xlim(left=0.0)
    axis.invert_yaxis()
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(figure, output_path)
    return True


def save_pairwise_forest_plot(
    output_path: Path,
    pairwise: pd.DataFrame,
    *,
    scope: str = "all_branches",
    metric: str = "mase",
    family: str = "all_pairs",
    treatment_label: str = "modelos",
) -> bool:
    """Plot paired median differences for one treatment panel."""
    data = pairwise[(pairwise["scope"] == scope) & (pairwise["metric"] == metric)].copy()
    if "family" in data:
        data = data[data["family"] == family].copy()
    if data.empty:
        return False
    for column in ("median_difference", "p_holm"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[np.isfinite(data["median_difference"])].copy()
    if data.empty:
        return False
    data["label"] = data["left"].astype(str) + "  vs  " + data["right"].astype(str)
    data = data.sort_values("improvement_pct", ascending=False).reset_index(drop=True)
    figure, axis = plt.subplots(
        figsize=(11.5, max(7.5, 0.31 * len(data) + 2.3)),
        facecolor=FIGURE_FACE,
    )
    add_plot_header(
        figure,
        f"Comparaciones pareadas de {treatment_label} — {_metric_label(metric)}",
        "Diferencia de mediana = izquierda − derecha; valores negativos favorecen al tratamiento de la izquierda.",
    )
    style_axis(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    y = np.arange(len(data))
    for position, (_, row) in enumerate(data.iterrows()):
        center = float(row["median_difference"])
        color = OUTCOME_COLORS.get(str(row["typical_station_outcome"]), OUTCOME_COLORS["inconclusive"])
        axis.scatter(
            center,
            position,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            s=48,
            zorder=3,
        )
        axis.text(
            1.01,
            position,
            f"p Holm={_fmt_p(row['p_holm'])}",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
            clip_on=False,
        )
    axis.axvline(0.0, color=MUTED_TEXT, linestyle="--", linewidth=1.1, zorder=2)
    axis.set_yticks(y, data["label"])
    axis.set_xlabel("diferencia de mediana (izquierda − derecha; menor = mejor)")
    axis.invert_yaxis()
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color=color, label=label, markersize=6)
            for outcome, label, color in (
                ("left_lower_typical_error", "izquierda mejor", OUTCOME_COLORS["left_lower_typical_error"]),
                ("right_lower_typical_error", "derecha mejor", OUTCOME_COLORS["right_lower_typical_error"]),
                ("inconclusive", "inconcluso", OUTCOME_COLORS["inconclusive"]),
            )
        ],
        loc="lower right",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 0.96, 0.89))
    save_figure(figure, output_path)
    return True


def save_empty_pairwise_plot(
    output_path: Path,
    *,
    metric: str,
    omnibus: pd.DataFrame,
) -> bool:
    """Persist an explicit no-post-hoc panel when the omnibus is not significant."""
    row = omnibus[
        (omnibus["scope"] == "common_raw_targets")
        & (omnibus["metric"] == metric)
    ]
    if row.empty:
        message = "No hay un contraste omnibus disponible para las ramas."
    else:
        omnibus_row = row.iloc[0]
        p_value = _fmt_p(omnibus_row["p_value"])
        alpha = _fmt_number(omnibus_row["alpha"])
        if float(omnibus_row["p_value"]) >= float(omnibus_row["alpha"]):
            message = (
                f"Omnibus no significativo: p = {p_value} >= alpha = {alpha}.\n"
                "No se calcularon comparaciones post-hoc."
            )
        else:
            message = "No hay comparaciones pareadas válidas para este panel."
    figure, axis = plt.subplots(figsize=(11.5, 5.0), facecolor=FIGURE_FACE)
    add_plot_header(
        figure,
        f"Comparaciones pareadas de ramas — {_metric_label(metric)}",
        "Sin comparaciones pareadas disponibles.",
    )
    style_axis(axis)
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(figure, output_path)
    return True


def save_branch_pairwise_plot(
    output_path: Path,
    pairwise: pd.DataFrame,
    omnibus: pd.DataFrame,
    *,
    metric: str,
) -> bool:
    """Plot branch post-hoc results or explain why none were run."""
    if save_pairwise_forest_plot(
        output_path,
        pairwise,
        scope="common_raw_targets",
        metric=metric,
        treatment_label="ramas",
    ):
        return True
    return save_empty_pairwise_plot(output_path, metric=metric, omnibus=omnibus)


def _draw_score_panel(axis: plt.Axes, scores: pd.DataFrame, *, scope: str, metric: str, title: str) -> bool:
    """Draw station score distributions for one analysis panel."""
    summary = _score_summary(scores, scope=scope, metric=metric)
    if summary.empty:
        axis.text(0.5, 0.5, "sin datos", ha="center", va="center", color=MUTED_TEXT)
        axis.set_title(title, loc="left")
        return False
    subset = scores[(scores["scope"] == scope) & (scores["metric"] == metric)]
    treatments = summary["treatment"].astype(str).tolist()
    rng = np.random.default_rng(13)
    positions = np.arange(len(treatments))
    for position, treatment in zip(positions, treatments, strict=False):
        values = _finite(subset.loc[subset["treatment"].astype(str) == treatment, "score"])
        if not values.size:
            continue
        color = SCORE_COLORS[position % len(SCORE_COLORS)]
        box = axis.boxplot(
            values,
            orientation="horizontal",
            positions=[position],
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.22, linewidth=1.1)
        for item in (*box["whiskers"], *box["caps"], *box["medians"]):
            item.set(color=color, linewidth=1.1)
        jitter = rng.normal(0.0, 0.065, size=len(values))
        axis.scatter(
            values,
            position + jitter,
            s=23,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.84,
            zorder=3,
        )
        mean = float(values.mean())
        axis.scatter([mean], [position], marker="D", s=34, color=TEXT_COLOR, zorder=4)
    style_axis(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    axis.set_yticks(positions, treatments)
    suffix = " (raw = 1; menor = mejor)" if metric.startswith("rel") else " (menor = mejor)"
    axis.set_xlabel(_metric_label(metric) + " por estación" + suffix)
    axis.set_title(title, loc="left", fontsize=11)
    axis.invert_yaxis()
    return True


def save_station_score_plot(
    output_path: Path,
    model_scores: pd.DataFrame,
    results: pd.DataFrame,
) -> bool:
    """Plot model scores and relative preprocessing scores by station."""
    model_scores = _project_foundation_raw_over_all_branches(model_scores, results)
    branch_scores = _build_relative_branch_scores(results)
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 12.0), facecolor=FIGURE_FACE)
    add_plot_header(
        figure,
        "Distribución de errores por estación",
        "Cada punto es una estación; la caja resume su distribución y el rombo marca la media. Menor error = mejor.",
    )
    jobs = (
        (
            axes[0, 0],
            model_scores,
            "all_branches",
            "mase",
            "Rendimiento medio de los modelos / MASE",
        ),
        (
            axes[0, 1],
            model_scores,
            "all_branches",
            "rmsse",
            "Rendimiento medio de los modelos / RMSSE",
        ),
        (
            axes[1, 0],
            branch_scores,
            "common_raw_targets",
            "relmae",
            "Rendimiento relativo de las ramas / RelMAE",
        ),
        (
            axes[1, 1],
            branch_scores,
            "common_raw_targets",
            "relrmse",
            "Rendimiento relativo de las ramas / RelRMSE",
        ),
    )
    saved = False
    for axis, scores, scope, metric, title in jobs:
        saved = _draw_score_panel(axis, scores, scope=scope, metric=metric, title=title) or saved
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    if not saved:
        plt.close(figure)
        return False
    save_figure(figure, output_path)
    return True


def save_branch_statistical_score_plot(
    output_path: Path,
    branch_scores: pd.DataFrame,
) -> bool:
    """Plot the absolute branch scores that enter the statistical contrasts."""
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), facecolor=FIGURE_FACE)
    add_plot_header(
        figure,
        "Scores de ramas usados en los contrastes",
        "Cada punto es una estación. Menor error es mejor.",
    )
    jobs = (
        (axes[0], "mase", "Ramas / MASE confirmatorio"),
        (axes[1], "rmsse", "Ramas / RMSSE de sensibilidad"),
    )
    saved = False
    for axis, metric, title in jobs:
        saved = _draw_score_panel(
            axis,
            branch_scores,
            scope="common_raw_targets",
            metric=metric,
            title=title,
        ) or saved
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    if not saved:
        plt.close(figure)
        return False
    save_figure(figure, output_path)
    return True


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render selected columns as a compact Markdown table."""
    if frame.empty:
        return "_Sin filas._"
    return frame.loc[:, columns].to_markdown(index=False)


def _primary_pairwise_table(pairwise: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Return the corrected-significance rows for the primary model analysis."""
    data = pairwise[
        (pairwise["scope"] == "all_branches")
        & (pairwise["metric"] == "mase")
        & (pairwise["confirmatory"].astype(str).str.lower() == "true")
    ].copy()
    if data.empty:
        return data
    data["p_holm"] = pd.to_numeric(data["p_holm"], errors="coerce")
    return data[data["p_holm"] < alpha].sort_values("p_holm")


def _branch_pairwise_table(
    pairwise: pd.DataFrame,
    *,
    metric: str = "mase",
    family: str = "vs_raw",
) -> pd.DataFrame:
    """Return one non-duplicated family of branch post-hoc comparisons."""
    data = pairwise[
        (pairwise["scope"] == "common_raw_targets")
        & (pairwise["metric"] == metric)
        & (pairwise["family"] == family)
    ].copy()
    if data.empty:
        return data
    data["p_holm"] = pd.to_numeric(data["p_holm"], errors="coerce")
    return data.sort_values(["p_holm", "improvement_pct"], ascending=[True, False])


def _format_branch_pairwise_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Format branch post-hoc results with an explicit winning treatment."""
    if pairwise.empty:
        return pairwise
    table = pairwise.copy()
    table["Comparación"] = table["left"].astype(str) + " vs " + table["right"].astype(str)
    table["Ganadora"] = "inconclusa"
    table.loc[
        table["typical_station_outcome"] == "left_lower_typical_error", "Ganadora"
    ] = table["left"].astype(str)
    table.loc[
        table["typical_station_outcome"] == "right_lower_typical_error", "Ganadora"
    ] = table["right"].astype(str)
    table["Mejora izquierda"] = table["improvement_pct"].map(
        lambda value: f"{float(value):+.1f}%"
    )
    table["Victorias-empates-derrotas"] = table.apply(
        lambda row: f"{int(row['wins'])}-{int(row['ties'])}-{int(row['losses'])}",
        axis=1,
    )
    table["p Holm"] = table["p_holm"].map(_fmt_p)
    return table


def _branch_score_summary_table(branch_scores: pd.DataFrame) -> pd.DataFrame:
    """Build one exportable summary row per branch and metric."""
    rows: list[dict[str, object]] = []
    for metric in ("mase", "rmsse"):
        summary = _score_summary(
            branch_scores, scope="common_raw_targets", metric=metric
        )
        for row in summary.to_dict("records"):
            rows.append(
                {
                    "metric": metric,
                    "branch": row["treatment"],
                    "mean_score": row["mean"],
                    "median_score": row["median"],
                    "std_score": row["std"],
                    "n_stations": row["n_stations"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "branch",
            "mean_score",
            "median_score",
            "std_score",
            "n_stations",
        ],
    )


def _branch_pairwise_export_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Build a compact, human-oriented export of branch post-hoc results."""
    columns = [
        "metric",
        "family",
        "left",
        "right",
        "winner",
        "n_stations",
        "improvement_pct",
        "median_difference",
        "wins",
        "ties",
        "losses",
        "p_value",
        "p_holm",
        "typical_station_outcome",
    ]
    if pairwise.empty:
        return pd.DataFrame(columns=columns)
    data = pairwise[pairwise["scope"] == "common_raw_targets"].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["winner"] = "inconclusa"
    data.loc[
        data["typical_station_outcome"] == "left_lower_typical_error", "winner"
    ] = data["left"]
    data.loc[
        data["typical_station_outcome"] == "right_lower_typical_error", "winner"
    ] = data["right"]
    return data.rename(columns={"winner": "winner"}).loc[:, columns].sort_values(
        ["metric", "family", "p_holm"], na_position="last"
    )


def write_branch_tables(
    run_dir: Path,
    *,
    branch_scores: pd.DataFrame,
    branch_omnibus: pd.DataFrame,
    branch_pairwise: pd.DataFrame,
) -> list[Path]:
    """Persist standalone CSV and Markdown tables for branch comparisons."""
    run_dir = Path(run_dir)
    score_table = _branch_score_summary_table(branch_scores)
    pairwise_table = _branch_pairwise_export_table(branch_pairwise)
    score_path = run_dir / "statistical_branch_scores_summary.csv"
    pairwise_path = run_dir / "statistical_branch_pairwise.csv"
    markdown_path = run_dir / "statistical_branch_tables.md"
    score_table.to_csv(score_path, index=False)
    pairwise_table.to_csv(pairwise_path, index=False)

    lines = [
        f"# Tablas estadísticas de ramas: `{run_dir.name}`",
        "",
        "Menor MASE/RMSSE es mejor. La unidad inferencial es la estación.",
        "",
        "## Scores por rama",
        "",
        _markdown_table(
            score_table.rename(
                columns={
                    "metric": "Métrica",
                    "branch": "Rama",
                    "mean_score": "Media",
                    "median_score": "Mediana",
                    "std_score": "SD",
                    "n_stations": "N estaciones",
                }
            ),
            ["Métrica", "Rama", "Media", "Mediana", "SD", "N estaciones"],
        ),
        "",
        "## Contraste omnibus",
        "",
        _markdown_table(
            branch_omnibus.rename(
                columns={
                    "metric": "Métrica",
                    "test": "Test",
                    "statistic": "Estadístico",
                    "p_value": "p-valor",
                    "alpha": "alpha",
                    "significant": "Significativo",
                    "n_stations": "N estaciones",
                    "n_treatments": "N ramas",
                }
            ),
            [
                "Métrica",
                "Test",
                "Estadístico",
                "p-valor",
                "alpha",
                "Significativo",
                "N estaciones",
                "N ramas",
            ],
        ),
        "",
        "## Comparaciones frente a raw",
        "",
    ]
    for metric in ("mase", "rmsse"):
        lines.extend([f"### {_metric_label(metric)}", ""])
        metric_pairs = pairwise_table[pairwise_table["metric"] == metric]
        if metric_pairs.empty:
            lines.append("No se generaron comparaciones porque el omnibus no fue significativo.")
        else:
            lines.append(
                _markdown_table(
                    _format_branch_pairwise_table(
                        branch_pairwise[
                            (branch_pairwise["scope"] == "common_raw_targets")
                            & (branch_pairwise["metric"] == metric)
                        ]
                    ),
                    [
                        "Comparación",
                        "Ganadora",
                        "Mejora izquierda",
                        "Victorias-empates-derrotas",
                        "p Holm",
                    ],
                )
            )
        lines.extend(["", "La columna `Ganadora` solo se rellena cuando el contraste Wilcoxon pareado corregido por Holm respalda la dirección.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return [score_path, pairwise_path, markdown_path]


def write_statistical_report(run_dir: Path) -> Path:
    """Write a Spanish interpretation report and return its path."""
    run_dir = Path(run_dir)
    tables = _load_tables(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    protocol = manifest.get("protocol", {})
    results = pd.read_csv(run_dir / "results.csv", keep_default_na=False)
    model_scores = tables["model_station_scores"]
    branch_scores = tables["branch_station_scores"]
    relative_branch_scores = _build_relative_branch_scores(results)
    model_omnibus = tables["model_omnibus"]
    branch_omnibus = tables["branch_omnibus"]
    pairwise = tables["model_pairwise"]
    branch_pairwise = tables["branch_pairwise"]
    exclusions = tables["statistical_exclusions"]
    alpha_values = _finite(model_omnibus["alpha"]) if "alpha" in model_omnibus else np.array([])
    alpha = float(alpha_values[0]) if alpha_values.size else 0.05

    primary_omnibus = model_omnibus[
        (model_omnibus["scope"] == "all_branches") & (model_omnibus["metric"] == "mase")
    ]
    primary_p = float(primary_omnibus["p_value"].iloc[0]) if not primary_omnibus.empty else float("nan")
    primary_pairs = _primary_pairwise_table(pairwise, alpha)
    primary_all_pairs = pairwise[
        (pairwise["scope"] == "all_branches") & (pairwise["metric"] == "mase")
    ]
    model_summary = _score_summary(model_scores, scope="all_branches", metric="mase")
    all_branch_model_scores = _project_foundation_raw_over_all_branches(
        model_scores, results
    )
    all_branch_model_summary = _score_summary(
        all_branch_model_scores, scope="all_branches", metric="mase"
    )
    common_model_summary = _score_summary(
        model_scores, scope="common_source_arms", metric="mase"
    )
    branch_relative_summary = _relative_branch_summary(relative_branch_scores)
    branch_mase_summary = _score_summary(
        branch_scores, scope="common_raw_targets", metric="mase"
    ).rename(
        columns={
            "mean": "mase_mean",
            "median": "mase_median",
            "std": "mase_std",
            "n_stations": "mase_n_stations",
        }
    )
    branch_rmsse_summary = _score_summary(
        branch_scores, scope="common_raw_targets", metric="rmsse"
    ).rename(
        columns={
            "mean": "rmsse_mean",
            "median": "rmsse_median",
            "std": "rmsse_std",
            "n_stations": "rmsse_n_stations",
        }
    )
    branch_absolute_summary = branch_mase_summary.merge(
        branch_rmsse_summary, on="treatment", how="outer"
    ).sort_values("mase_mean")
    branch_vs_raw = _branch_pairwise_table(branch_pairwise, family="vs_raw")
    branch_mase = branch_omnibus[
        (branch_omnibus["scope"] == "common_raw_targets") & (branch_omnibus["metric"] == "mase")
    ]
    branch_p = float(branch_mase["p_value"].iloc[0]) if not branch_mase.empty else float("nan")
    better_than_raw: list[str] = []
    for _, comparison in branch_vs_raw.iterrows():
        outcome = comparison["typical_station_outcome"]
        if outcome == "left_lower_typical_error" and comparison["right"] == "raw":
            better_than_raw.append(str(comparison["left"]))
        elif outcome == "right_lower_typical_error" and comparison["left"] == "raw":
            better_than_raw.append(str(comparison["right"]))

    lines = [
        f"# Informe estadístico del benchmark de forecasting: `{run_dir.name}`",
        "",
        "> Informe generado automáticamente a partir de las tablas CSV persistidas.\n"
        "> No vuelve a entrenar ni a ejecutar forecasts.",
        "",
        "## Resumen ejecutivo",
        "",
        f"- Run: `{run_dir}`.",
        f"- Estaciones seleccionadas: **{len(manifest.get('selected_stations', []))}**; filas de resultados: **{manifest.get('result_rows', 'NA')}**.",
        f"- Protocolo: holdout de **{protocol.get('holdout', 'NA')} h**, horizonte **{protocol.get('horizon', 'NA')} h**, stride **{protocol.get('forecast_stride', 'NA')} h**, contexto **{protocol.get('context_len', 'NA')} h**.",
        f"- Nivel de significación: `alpha = {alpha:g}`; semilla: `{manifest.get('config', {}).get('seed', 'NA')}`.",
        "",
    ]
    if np.isfinite(primary_p) and primary_p < alpha:
        lines.append(
            f"**Conclusión confirmatoria:** existen diferencias globales entre modelos usando MASE "
            f"(Friedman por permutación, `p = {_fmt_p(primary_p)}`, {len(primary_all_pairs)} comparaciones post-hoc)."
        )
    else:
        lines.append("**Conclusión confirmatoria:** no se detectaron diferencias globales entre modelos con MASE.")
    if np.isfinite(branch_p) and branch_p >= alpha:
        lines.append(
            f"**Sobre las ramas:** no hay evidencia suficiente de diferencias globales entre estrategias de limpieza/imputación "
            f"(`p = {_fmt_p(branch_p)}`). Las comparaciones predefinidas frente a `raw` se reportan por separado con Holm."
        )
    elif np.isfinite(branch_p):
        lines.append(
            f"**Sobre las ramas:** el contraste global sí es significativo (`p = {_fmt_p(branch_p)}`); consulte las comparaciones post-hoc."
        )
        if better_than_raw:
            lines.append(
                "**Ramas mejores que raw con respaldo confirmatorio:** "
                + ", ".join(f"`{branch}`" for branch in better_than_raw)
                + "."
            )
        else:
            lines.append(
                "**Comparaciones frente a raw:** ninguna rama obtiene una superioridad direccional confirmatoria."
            )
    lines.extend(
        [
            "",
            "En términos descriptivos, el menor score medio de modelo es el primero de la tabla siguiente. "
            "Esto no convierte automáticamente a ese modelo en un ganador estadístico: las comparaciones pareadas corregidas son las que sostienen las conclusiones.",
            "",
            "## Resultados principales",
            "",
            "### Modelos: panel confirmatorio MASE",
            "",
            _markdown_table(
                model_summary.assign(
                    treatment=lambda frame: frame["treatment"].astype(str),
                    mean=lambda frame: frame["mean"].map(lambda value: _fmt_number(value)),
                    median=lambda frame: frame["median"].map(lambda value: _fmt_number(value)),
                    std=lambda frame: frame["std"].map(lambda value: _fmt_number(value)),
                ).rename(columns={"treatment": "Modelo", "mean": "Media", "median": "Mediana", "std": "SD", "n_stations": "N estaciones"}),
                ["Modelo", "Media", "Mediana", "SD", "N estaciones"],
            ),
            "",
            "Menor MASE es mejor. Estos valores son medias descriptivas de los scores por estación; no son p-valores.",
            "",
            "### Modelos: todas las ramas, incluyendo foundation",
            "",
            _markdown_table(
                all_branch_model_summary.assign(
                    treatment=lambda frame: frame["treatment"].astype(str),
                    mean=lambda frame: frame["mean"].map(lambda value: _fmt_number(value)),
                    median=lambda frame: frame["median"].map(lambda value: _fmt_number(value)),
                    std=lambda frame: frame["std"].map(lambda value: _fmt_number(value)),
                ).rename(columns={"treatment": "Modelo", "mean": "Media", "median": "Mediana", "std": "SD", "n_stations": "N estaciones"}),
                ["Modelo", "Media", "Mediana", "SD", "N estaciones"],
            ),
            "",
            "Esta es la agregación que se muestra arriba en `statistical_station_scores.png`: "
            "cada modelo se resume sobre todas las ramas. Para foundation, el error de `raw` "
            "se utiliza como valor en cada rama, tal como define este benchmark.",
            "",
            "### Modelos incluyendo foundation: vistas raw comunes",
            "",
            _markdown_table(
                common_model_summary.assign(
                    treatment=lambda frame: frame["treatment"].astype(str),
                    mean=lambda frame: frame["mean"].map(lambda value: _fmt_number(value)),
                    median=lambda frame: frame["median"].map(lambda value: _fmt_number(value)),
                    std=lambda frame: frame["std"].map(lambda value: _fmt_number(value)),
                ).rename(columns={"treatment": "Modelo", "mean": "Media", "median": "Mediana", "std": "SD", "n_stations": "N estaciones"}),
                ["Modelo", "Media", "Mediana", "SD", "N estaciones"],
            ),
            "",
            "Este panel usa únicamente `raw`, la vista fuente disponible para todos los modelos, "
            "por lo que incorpora también los foundation sin repetir artificialmente su score en ramas donde no se ejecutaron. "
            "Es una comparación de sensibilidad de la fuente común, no sustituye al contraste confirmatorio de todos los brazos de entrenamiento.",
            "",
            "### Comparaciones pareadas significativas",
            "",
        ]
    )
    if primary_pairs.empty:
        lines.append("No hay comparaciones pareadas con `p_holm < 0,05`.")
    else:
        pair_table = primary_pairs.copy()
        pair_table["Comparacion"] = (
            pair_table["left"].astype(str) + " vs " + pair_table["right"].astype(str)
        )
        pair_table["Mejora_media"] = pair_table["improvement_pct"].map(
            lambda value: f"{float(value):+.1f}%"
        )
        pair_table["Resultado"] = pair_table["typical_station_outcome"].map(_outcome_label)
        pair_table["p_holm"] = pair_table["p_holm"].map(_fmt_p)
        pair_table["Victorias-empates-derrotas"] = pair_table.apply(
            lambda row: f"{int(row['wins'])}-{int(row['ties'])}-{int(row['losses'])}", axis=1
        )
        lines.append(
            _markdown_table(
                pair_table.rename(
                    columns={
                        "Comparacion": "Comparación",
                        "Mejora_media": "Mejora media",
                        "Resultado": "Lectura",
                        "p_holm": "p Holm",
                    }
                ),
                [
                    "Comparación",
                    "Mejora media",
                    "Victorias-empates-derrotas",
                    "p Holm",
                    "Lectura",
                ],
            )
        )
    lines.extend(
        [
            "",
            "La mejora media es `100 × (1 − error_izquierda / error_derecha)`: positiva favorece a la izquierda. "
            "La columna `p Holm` es el p-valor de Wilcoxon signed-rank corregido dentro de la familia de comparaciones.",
            "",
            "### Ramas: scores usados en los contrastes",
            "",
            _markdown_table(
                branch_absolute_summary.assign(
                    treatment=lambda frame: frame["treatment"].astype(str),
                    mase_mean=lambda frame: frame["mase_mean"].map(lambda value: _fmt_number(value)),
                    mase_median=lambda frame: frame["mase_median"].map(lambda value: _fmt_number(value)),
                    rmsse_mean=lambda frame: frame["rmsse_mean"].map(lambda value: _fmt_number(value)),
                    rmsse_median=lambda frame: frame["rmsse_median"].map(lambda value: _fmt_number(value)),
                ).rename(columns={"treatment": "Rama", "mase_mean": "Media MASE", "mase_median": "Mediana MASE", "rmsse_mean": "Media RMSSE", "rmsse_median": "Mediana RMSSE"}),
                ["Rama", "Media MASE", "Mediana MASE", "Media RMSSE", "Mediana RMSSE"],
            ),
            "",
            "La primera fila tiene el menor MASE medio descriptivo. Solo puede considerarse ganadora estadística cuando los post-hoc siguientes producen una conclusión direccional.",
            "",
            "### Ramas: comparaciones frente a raw",
            "",
        ]
    )
    if branch_vs_raw.empty:
        lines.append(
            "No se generaron comparaciones frente a raw: el omnibus no fue significativo o no había tratamientos suficientes."
        )
    else:
        lines.append(
            _markdown_table(
                _format_branch_pairwise_table(branch_vs_raw),
                [
                    "Comparación",
                    "Ganadora",
                    "Mejora izquierda",
                    "Victorias-empates-derrotas",
                    "p Holm",
                ],
            )
        )
    lines.extend(
        [
            "",
            "Las comparaciones frente a `raw` son la única familia inferencial de ramas. `Ganadora` solo se rellena cuando el Wilcoxon pareado corregido por Holm respalda la dirección.",
            "",
            "### Ramas: rendimiento relativo frente a raw",
            "",
            _markdown_table(
                branch_relative_summary.assign(
                    treatment=lambda frame: frame["treatment"].astype(str),
                    relmae_mean=lambda frame: frame["relmae_mean"].map(lambda value: _fmt_number(value)),
                    relmae_median=lambda frame: frame["relmae_median"].map(lambda value: _fmt_number(value)),
                    relrmse_mean=lambda frame: frame["relrmse_mean"].map(lambda value: _fmt_number(value)),
                    relrmse_median=lambda frame: frame["relrmse_median"].map(lambda value: _fmt_number(value)),
                ).rename(columns={"treatment": "Rama", "relmae_mean": "Media RelMAE", "relmae_median": "Mediana RelMAE", "relrmse_mean": "Media RelRMSE", "relrmse_median": "Mediana RelRMSE"}),
                ["Rama", "Media RelMAE", "Mediana RelMAE", "Media RelRMSE", "Mediana RelRMSE"],
            ),
            "",
            "RelMAE y RelRMSE son el error de cada rama dividido por el error de `raw` del mismo modelo y estación: "
            "`raw = 1`, valores menores que 1 indican mejora y valores mayores que 1 empeoramiento. "
            f"El omnibus estadístico original, basado en MASE, tiene `p = {_fmt_p(branch_p)}`; la decisión se toma con `alpha = {alpha:g}`.",
            "",
            "### Sensibilidad RMSSE y panel de modelos con fuentes comunes",
            "",
            _markdown_table(
                model_omnibus.assign(
                    panel=lambda frame: frame["scope"].map({"all_branches": "todos los brazos", "common_source_arms": "solo raw"}),
                    metric=lambda frame: frame["metric"].str.upper(),
                    p=lambda frame: frame["p_value"].map(_fmt_p),
                    conclusion=lambda frame: frame["significant"].astype(str).map({"True": "significativo", "False": "no significativo"}),
                ).rename(columns={"panel": "Panel", "metric": "Métrica", "p": "p-valor", "conclusion": "Resultado", "n_stations": "N estaciones", "n_treatments": "N tratamientos"}),
                ["Panel", "Métrica", "p-valor", "Resultado", "N estaciones", "N tratamientos"],
            ),
            "",
            _markdown_table(
                branch_omnibus.assign(
                    metric=lambda frame: frame["metric"].str.upper(),
                    p=lambda frame: frame["p_value"].map(_fmt_p),
                    conclusion=lambda frame: frame["significant"].astype(str).map({"True": "significativo", "False": "no significativo"}),
                ).rename(columns={"metric": "Métrica", "p": "p-valor", "conclusion": "Resultado", "n_stations": "N estaciones", "n_treatments": "N tratamientos"}),
                ["Métrica", "p-valor", "Resultado", "N estaciones", "N tratamientos"],
            ),
            "",
            "RMSSE se interpreta como análisis de sensibilidad; MASE mantiene el papel confirmatorio para modelos y ramas.",
            "",
            "## Qué significa cada salida",
            "",
            "- `model_station_scores.csv`: score MASE/RMSSE de cada modelo en cada estación, después de promediar los brazos incluidos en el panel; `rank` es el rango dentro de la estación. El scope `common_source_arms` incluye foundation y usa solo `raw`.",
            "- En `statistical_station_scores.png`, el panel superior usa `all_branches`: los foundation reciben su error `raw` en cada rama antes de promediar, por lo que aparecen junto al resto de modelos.",
            "- `model_omnibus.csv`: prueba global de igualdad entre modelos. Con dos tratamientos sería Wilcoxon; aquí es Friedman con p-valor por permutación.",
            "- `model_pairwise.csv`: comparaciones post-hoc entre pares de modelos. Incluye diferencias, ratio de errores, mejora, victorias/empates/derrotas y p-valores de Wilcoxon corregidos.",
            "- `branch_station_scores.csv`: scores absolutos MASE/RMSSE de cada rama por estación, promediados sobre los modelos entrenados/locales.",
            "- La figura y la tabla de ramas del informe usan `RelMAE` y `RelRMSE`: el denominador es el `raw` del mismo modelo y estación.",
            "- `branch_omnibus.csv`: prueba global de igualdad entre ramas. Los modelos foundation no forman parte de este panel porque no tienen brazos de entrenamiento.",
            "- `branch_pairwise.csv`: post-hoc de ramas; queda vacío cuando el omnibus no es significativo.",
            "- `statistical_exclusions.csv`: estaciones excluidas por celdas ausentes, no finitas o backtests incompletos. En este run no hay exclusiones.",
            "",
            "Columnas especialmente importantes: `p_value` es el contraste global; `p_holm` es el p-valor de Wilcoxon corregido por Holm; `error_ratio` menor que 1 favorece a la izquierda; `rank_biserial` negativo favorece a la izquierda; `typical_station_outcome` es una conclusión direccional.",
            "",
            "## Figuras",
            "",
            "- [Contrastes omnibus](statistical_omnibus.png): p-valores globales en escala `-log10(p)`.",
            "- [Scores por estación](statistical_station_scores.png): arriba, modelos promediados en todas las ramas, con foundation = error `raw`; abajo, estrategias de limpieza/imputación sobre el mismo test raw.",
            "- [Comparaciones pareadas de modelos](statistical_model_pairwise_mase.png): diferencias de medianas; la línea vertical es igualdad.",
            "- [Scores estadísticos de ramas](statistical_branch_scores.png): MASE y RMSSE absolutos por estación, exactamente los scores que entran en los contrastes.",
            "- [Comparaciones pareadas de ramas en MASE](statistical_branch_pairwise_mase.png): comparaciones de cada rama frente a `raw` cuando el omnibus MASE es significativo.",
            "- [Comparaciones pareadas de ramas en RMSSE](statistical_branch_pairwise_rmsse.png): análisis de sensibilidad frente a `raw` cuando el omnibus RMSSE es significativo.",
            "- [Tablas estadísticas de ramas](statistical_branch_tables.md): resumen legible de scores, omnibus y post-hoc.",
            "- [Resumen de scores de ramas](statistical_branch_scores_summary.csv) y [comparaciones pareadas](statistical_branch_pairwise.csv): versiones tabulares para análisis posterior.",
            "",
            "## Limitaciones para la interpretación",
            "",
            "- La unidad inferencial es la estación. Los 15 orígenes de test se solapan y no se tratan como 15 réplicas independientes.",
            "- El análisis usa métricas agregadas por estación; estos CSV no permiten reconstruir un contraste Diebold-Mariano por origen o por lead.",
            "- La selección del holdout y de las máscaras es retrospectiva/transductiva: inspecciona la serie completa. Las conclusiones describen este benchmark offline, no un protocolo online causal.",
            "- Un p-valor no significativo no demuestra equivalencia práctica. Falta definir un margen de relevancia de dominio para poder hacer una afirmación de equivalencia.",
            f"- Hay {len(manifest.get('selected_stations', []))} estaciones y {len(exclusions)} filas de exclusión.",
            "",
            "## Archivos de entrada y parámetros",
            "",
            f"- `manifest.json`: estado `{manifest.get('status', 'NA')}`, filas esperadas `{manifest.get('expected_result_rows', 'NA')}`, fallos `{len(manifest.get('failures', []))}`.",
            f"- Configuración relevante: contaminante `{config.get('pollutant', 'NA')}`, semilla `{config.get('seed', 'NA')}`, umbral MAD `{config.get('threshold_k', 'NA')}`.",
            f"- Los contrastes omnibus con más de dos tratamientos usaron `{int(model_omnibus['n_permutations'].dropna().iloc[0]) if not model_omnibus.empty and model_omnibus['n_permutations'].notna().any() else 'NA'}` permutaciones.",
        ]
    )
    if model_summary.empty:
        lines.insert(10, "No hay scores de modelo válidos para este run.")
    output_path = run_dir / "statistical_report.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def render_statistical_report(run_dir: Path) -> list[Path]:
    """Render statistical figures and the Markdown report for one run."""
    run_dir = Path(run_dir)
    tables = _load_tables(run_dir)
    results = pd.read_csv(run_dir / "results.csv", keep_default_na=False)
    jobs = (
        (
            run_dir / "statistical_omnibus.png",
            lambda path: save_omnibus_plot(path, pd.concat([tables["model_omnibus"], tables["branch_omnibus"]], ignore_index=True)),
        ),
        (
            run_dir / "statistical_model_pairwise_mase.png",
            lambda path: save_pairwise_forest_plot(path, tables["model_pairwise"]),
        ),
        (
            run_dir / "statistical_branch_pairwise_mase.png",
            lambda path: save_branch_pairwise_plot(
                path,
                tables["branch_pairwise"],
                tables["branch_omnibus"],
                metric="mase",
            ),
        ),
        (
            run_dir / "statistical_branch_pairwise_rmsse.png",
            lambda path: save_branch_pairwise_plot(
                path,
                tables["branch_pairwise"],
                tables["branch_omnibus"],
                metric="rmsse",
            ),
        ),
        (
            run_dir / "statistical_branch_scores.png",
            lambda path: save_branch_statistical_score_plot(
                path, tables["branch_station_scores"]
            ),
        ),
        (
            run_dir / "statistical_station_scores.png",
            lambda path: save_station_score_plot(
                path,
                tables["model_station_scores"],
                results,
            ),
        ),
    )
    saved: list[Path] = []
    for path, job in jobs:
        if job(path):
            saved.append(path)
    saved.extend(
        write_branch_tables(
            run_dir,
            branch_scores=tables["branch_station_scores"],
            branch_omnibus=tables["branch_omnibus"],
            branch_pairwise=tables["branch_pairwise"],
        )
    )
    saved.append(write_statistical_report(run_dir))
    return saved


def main(argv: list[str] | None = None) -> None:
    """CLI for statistical figures and the explanatory report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    for path in render_statistical_report(args.run_dir):
        print(f"[info] {path}")


if __name__ == "__main__":
    main()
