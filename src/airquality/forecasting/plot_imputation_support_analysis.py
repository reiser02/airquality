"""Plots for the separate imputation-support audit.

Purpose
-------
Visualize how much worst-case train-eligible support appears after anomaly
detection plus imputation, and how old that added support is relative to the
single holdout selected for each series. This module only reads persisted data;
it never runs detectors, imputers or forecasting models.

Required data
-------------
The input is ``training_support.csv`` inside one run directory. Obtain it with::

    uv run python -m airquality.forecasting.imputation_support_analysis

The plots require the CSV columns for ``series``, ``strategy``, ``selected``,
eligible points before/after, recovered observed points, imputed eligible
points, and median ages. The producer writes this complete schema.

Usage
-----
Render a specific run, or omit it to use the newest run containing the CSV::

    uv run python -m airquality.forecasting.plot_imputation_support_analysis \
        reports/forecasting_support/YYYYMMDD_HHMMSS
    uv run python -m airquality.forecasting.plot_imputation_support_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.anomaly.presentation import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    TEXT_COLOR,
    style_axis,
)
from airquality.forecasting.fill import _repo_root

IMPUTED_COLOR = "#cf6f1e"
RECOVERED_COLOR = "#5b8c5a"
BASE_COLOR = "#6d6258"
REQUIRED_COLUMNS = {
    "series",
    "strategy",
    "selected",
    "n_eligible_points_before_imputation",
    "n_eligible_points_after_imputation",
    "n_eligible_points_added",
    "n_observed_points_recovered",
    "n_imputed_in_eligible_blocks",
    "imputed_eligible_age_hours_median",
    "recovered_observed_age_hours_median",
}


def _selected(results: pd.DataFrame) -> pd.DataFrame:
    selected = results["selected"]
    if selected.dtype != bool:
        selected = selected.astype(str).str.lower().eq("true")
    return results.loc[selected]


def _save_support_composition(path: Path, results: pd.DataFrame) -> bool:
    selected = _selected(results)
    if selected.empty:
        return False
    totals = selected.groupby("strategy", sort=False)[
        [
            "n_eligible_points_before_imputation",
            "n_observed_points_recovered",
            "n_imputed_in_eligible_blocks",
        ]
    ].sum()
    positions = np.arange(len(totals))
    figure, axis = plt.subplots(
        figsize=(9.5, 2.8 + 0.75 * len(totals)), facecolor=FIGURE_FACE
    )
    style_axis(axis)
    left = np.zeros(len(totals))
    for column, label, color in (
        ("n_eligible_points_before_imputation", "Soporte ya elegible", BASE_COLOR),
        ("n_observed_points_recovered", "Observado recuperado", RECOVERED_COLOR),
        ("n_imputed_in_eligible_blocks", "Valores imputados", IMPUTED_COLOR),
    ):
        values = totals[column].to_numpy(dtype=float)
        bars = axis.barh(
            positions,
            values,
            left=left,
            color=color,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            label=label,
        )
        labels = [f"{int(value):,}" if value > 0 else "" for value in values]
        axis.bar_label(bars, labels=labels, label_type="center", fontsize=8, color="white")
        left += values
    axis.set_yticks(positions, totals.index)
    axis.invert_yaxis()
    axis.set_xlabel("Puntos horarios en bloques elegibles")
    axis.set_title(
        "Soporte de train antes y despues de imputar",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=30,
    )
    axis.text(
        0,
        1.015,
        "Suma entre series; umbral = peor min_train_series_length configurado.",
        transform=axis.transAxes,
        color=TEXT_COLOR,
        fontsize=9,
    )
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncols=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_support_gain_heatmap(path: Path, results: pd.DataFrame) -> bool:
    selected = _selected(results).copy()
    if selected.empty:
        return False
    added = pd.to_numeric(selected["n_eligible_points_added"], errors="coerce")
    selected["days_added"] = added / 24.0
    strategies = selected["strategy"].drop_duplicates()
    table = selected.pivot(index="series", columns="strategy", values="days_added")
    table = table.reindex(columns=strategies)
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    maximum = float(table.max().max())
    if not np.isfinite(maximum) or maximum <= 0:
        maximum = 1.0
    figure, axes = plt.subplots(
        1,
        len(strategies),
        sharex=True,
        sharey=True,
        figsize=(12.5, 2.8 + 0.42 * len(table.index)),
        facecolor=FIGURE_FACE,
    )
    axes = np.atleast_1d(axes)
    positions = np.arange(len(table))
    for index, (axis, strategy) in enumerate(zip(axes, strategies, strict=True)):
        style_axis(axis)
        values = table[strategy].to_numpy(dtype=float)
        bars = axis.barh(
            positions,
            values,
            color=IMPUTED_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
        )
        axis.bar_label(bars, fmt="+%.0f d", padding=3, fontsize=7, color=TEXT_COLOR)
        axis.set_title(strategy, fontsize=11, fontweight="bold", color=TEXT_COLOR)
        axis.set_yticks(positions, table.index)
        axis.tick_params(axis="y", labelleft=index == 0, labelsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, maximum * 1.2)
        axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    figure.suptitle(
        "Dias adicionales de entrenamiento por serie",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        x=0.08,
        y=0.99,
    )
    figure.text(
        0.08,
        0.955,
        "Una barra = soporte utilizable despues de imputar menos soporte utilizable antes.",
        color=TEXT_COLOR,
        fontsize=9,
    )
    figure.supxlabel("Dias de train ganados", color=TEXT_COLOR, fontsize=10)
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_support_age(path: Path, results: pd.DataFrame) -> bool:
    selected = _selected(results)
    if selected.empty:
        return False
    strategies = selected["strategy"].drop_duplicates().tolist()
    ages = (
        selected.groupby("strategy", sort=False)["imputed_eligible_age_hours_median"]
        .median()
        .reindex(strategies)
        / (24.0 * 30.44)
    )
    if ages.isna().all():
        return False
    ages = ages.dropna()
    strategies = ages.index.tolist()

    positions = np.arange(len(strategies))
    figure, axis = plt.subplots(figsize=(8.5, 4.5), facecolor=FIGURE_FACE)
    style_axis(axis)
    bars = axis.barh(
        positions,
        ages,
        color=IMPUTED_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
    )
    axis.bar_label(bars, fmt="%.1f meses", padding=4, fontsize=9, color=TEXT_COLOR)
    axis.set_yticks(positions, strategies)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.set_xlabel("Meses antes del inicio del holdout")
    axis.set_title(
        "Antiguedad de las horas rellenadas",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=30,
    )
    axis.text(
        0,
        1.015,
        "Mediana entre estaciones; solo se cuentan huecos rellenados dentro de bloques utilizables.",
        transform=axis.transAxes,
        color=TEXT_COLOR,
        fontsize=9,
    )
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def _save_support_age_timeline(path: Path, results: pd.DataFrame) -> bool:
    selected = _selected(results)
    if selected.empty:
        return False
    strategies = selected["strategy"].drop_duplicates().tolist()
    ages = (
        selected.groupby("strategy", sort=False)["imputed_eligible_age_hours_median"]
        .median()
        .reindex(strategies)
        / (24.0 * 30.44)
    )
    if ages.isna().all():
        return False
    ages = ages.dropna()
    strategies = ages.index.tolist()

    positions = np.arange(len(strategies))
    figure, axis = plt.subplots(figsize=(10, 4.5), facecolor=FIGURE_FACE)
    style_axis(axis)
    for position, age in zip(positions, ages, strict=True):
        axis.hlines(position, -age, 0, color=IMPUTED_COLOR, linewidth=4, alpha=0.8)
        axis.scatter(
            -age,
            position,
            s=100,
            color=IMPUTED_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            f"{age:.1f} meses antes",
            (-age, position),
            xytext=(-7, -18),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color=TEXT_COLOR,
        )
    axis.axvline(0, color=TEXT_COLOR, linewidth=2)
    axis.text(
        0.08,
        positions.mean(),
        "Inicio del holdout",
        rotation=90,
        va="center",
        color=TEXT_COLOR,
        fontsize=9,
    )
    axis.set_yticks(positions, strategies)
    axis.invert_yaxis()
    axis.set_ylim(len(strategies) - 0.35, -0.5)
    maximum = max(2, int(np.ceil(ages.max() / 2.0) * 2))
    ticks = np.arange(0, maximum + 1, 2)
    axis.set_xlim(-maximum - 1, 0.5)
    axis.set_xticks(-ticks, [f"{tick:g}" for tick in ticks])
    axis.set_xlabel("Meses antes del holdout")
    axis.set_title(
        "Ubicacion temporal de las horas rellenadas",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=30,
    )
    axis.text(
        0,
        1.015,
        "El punto marca la antiguedad mediana; la linea termina donde empieza el holdout.",
        transform=axis.transAxes,
        color=TEXT_COLOR,
        fontsize=9,
    )
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def render_plots(run_dir: Path) -> list[Path]:
    """Validate one support CSV and render all available figures beside it."""
    csv_path = run_dir / "training_support.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe {csv_path}")
    results = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(results.columns))
    if missing:
        raise ValueError(f"Faltan columnas en training_support.csv: {', '.join(missing)}")
    builders = (
        ("support_composition.png", _save_support_composition),
        ("support_gain_by_series.png", _save_support_gain_heatmap),
        ("support_age.png", _save_support_age),
        ("support_age_timeline.png", _save_support_age_timeline),
    )
    written = []
    for filename, builder in builders:
        path = run_dir / filename
        if builder(path, results):
            written.append(path)
    return written


def _latest_run() -> Path:
    root = _repo_root() / "reports" / "forecasting_support"
    runs = [path for path in root.iterdir() if (path / "training_support.csv").is_file()]
    if not runs:
        raise FileNotFoundError(f"No hay runs con training_support.csv bajo {root}")
    return max(runs, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renderiza el diagnostico de soporte desde training_support.csv"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Directorio del run; por defecto usa el mas reciente",
    )
    args = parser.parse_args()
    run_dir = args.run_dir or _latest_run()
    paths = render_plots(run_dir)
    print(f"[info] {len(paths)} figuras escritas en {run_dir}")


if __name__ == "__main__":
    main()
