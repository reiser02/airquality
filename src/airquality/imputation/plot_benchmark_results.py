"""Timing figures for the imputation benchmark.

Two per-model figures summarize the *cost* of each imputer. They are fed by
**two files**, because training costs live in different places:

1. The benchmark **results CSV** (one row per model×series×gap, from
   :func:`airquality.imputation.benchmark.execute_complete_pipeline`) holds the
   per-hole mean ``Impute_Seconds`` for every model plus ``Train_Seconds`` for
   the models trained at benchmark time (Prophet's per-hole fit, TSPulse's
   one-time load, 0 for interpolation). Pretrained Darts have ``Train_Seconds``
   NaN here.
2. The **Darts training CSV** written by ``train_global_methods``
   (``reports/metrics/training_curves_and_times.csv``, column
   ``training_time_seconds``) supplies the offline training time of the Darts
   models — merged in only at plot time.

Figures:

- ``imputation_time_by_model.png`` — grouped horizontal bars of the mean
  training time and the mean imputation (inference) time per model.
- ``imputation_time_vs_{mase,rmse,mae}.png`` — cost/accuracy scatter: mean
  imputation time (x, log) vs error (y) per model, so the bottom-left is best.

All timings are means across series; plots regenerate from the CSVs without
recomputing anything.

Run with::

    uv run python -m airquality.imputation.plot_benchmark_results results.csv \\
        [--darts-train-csv reports/metrics/training_curves_and_times.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from airquality.anomaly.presentation import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    TEXT_COLOR,
    add_plot_header,
    style_axis,
)

MUTED_TEXT = "#6d6258"

TRAIN_COL = "Train_Seconds"
IMPUTE_COL = "Impute_Seconds"

#: Default location of the Darts training-time CSV produced by
#: ``train_global_methods`` (its ``training_time_seconds`` column, constant per
#: model across epoch rows).
DEFAULT_DARTS_TRAIN_CSV = Path("reports/metrics/training_curves_and_times.csv")

#: Stable per-model palette (CVD-separable on the cream surface); assigned in
#: model order and cycled only if a run has more models than colors.
MODEL_PALETTE = (
    "#3d7ab5", "#cf6f1e", "#9b59b6", "#5b8c5a",
    "#cf6ba9", "#8c6d4b", "#d6453c", "#27313a",
)

#: Display names of the imputation error metrics (upper-case CSV columns).
METRIC_LABELS = {"MASE": "MASE", "RMSE": "RMSE", "MAE": "MAE"}


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric.upper(), metric.upper())


def _model_colors(models: list[str]) -> dict[str, str]:
    """Assign each model a fixed palette color in appearance order."""
    return {model: MODEL_PALETTE[index % len(MODEL_PALETTE)] for index, model in enumerate(models)}


def load_darts_train_seconds(path: Path | str) -> dict[str, float]:
    """Read per-model Darts training time from ``train_global_methods``' CSV.

    Accepts either the training-curves CSV (``training_time_seconds`` column,
    repeated per epoch) or a compact ``train_seconds`` file. Returns
    ``{model_name: seconds}`` (one value per model); missing/invalid files yield
    an empty mapping so plotting degrades gracefully.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return {}
    if "model_name" not in frame.columns:
        return {}
    time_col = next(
        (col for col in ("training_time_seconds", "train_seconds") if col in frame.columns),
        None,
    )
    if time_col is None:
        return {}
    frame = frame.dropna(subset=[time_col])
    grouped = frame.groupby(frame["model_name"].astype(str))[time_col].first()
    return {name: float(value) for name, value in grouped.items()}


def summarize_timing(
    results_df: pd.DataFrame,
    darts_train_seconds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Mean timing (and error metrics) per model from a raw benchmark results frame.

    Returns a frame with a ``Modelo`` column plus whichever of ``Train_Seconds`` /
    ``Impute_Seconds`` / ``MAE`` / ``RMSE`` / ``MASE`` are present, one row per
    model (input order preserved). Every value is the mean across series.

    ``darts_train_seconds`` (``{model: seconds}`` from
    :func:`load_darts_train_seconds`) overrides ``Train_Seconds`` for the Darts
    models trained offline, whose benchmark rows carry NaN there.
    """
    if results_df.empty or "Modelo" not in results_df.columns:
        return pd.DataFrame()
    value_cols = [
        col for col in (TRAIN_COL, IMPUTE_COL, "MAE", "RMSE", "MASE")
        if col in results_df.columns
    ]
    if not value_cols:
        return pd.DataFrame()
    order = list(dict.fromkeys(results_df["Modelo"].astype(str)))
    numeric = results_df.copy()
    numeric["Modelo"] = numeric["Modelo"].astype(str)
    for col in value_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    summary = numeric.groupby("Modelo", as_index=False)[value_cols].mean(numeric_only=True)
    summary["Modelo"] = pd.Categorical(summary["Modelo"], categories=order, ordered=True)
    summary = summary.sort_values("Modelo").reset_index(drop=True).assign(
        Modelo=lambda frame: frame["Modelo"].astype(str)
    )
    if darts_train_seconds:
        if TRAIN_COL not in summary.columns:
            summary[TRAIN_COL] = float("nan")
        overrides = summary["Modelo"].map(lambda name: darts_train_seconds.get(name))
        summary[TRAIN_COL] = overrides.where(overrides.notna(), summary[TRAIN_COL])
    return summary


def save_time_by_model_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    darts_train_seconds: Mapping[str, float] | None = None,
) -> bool:
    """Grouped horizontal bars: mean training vs mean imputation time per model."""
    summary = summarize_timing(results_df, darts_train_seconds)
    present = [col for col in (TRAIN_COL, IMPUTE_COL) if col in summary.columns]
    if summary.empty or not present:
        return False
    # Sort models by total measured time (fast to slow, top to bottom).
    summary = summary.assign(_total=summary[present].sum(axis=1)).sort_values(
        "_total", ascending=True
    )
    models = list(summary["Modelo"])
    colors = _model_colors(models)
    positions = np.arange(len(models))

    labels = {TRAIN_COL: "entrenamiento / carga", IMPUTE_COL: "imputacion (inferencia)"}
    hatches = {TRAIN_COL: "///", IMPUTE_COL: None}
    group = 0.8
    bar_h = group / len(present)

    figure, axis = plt.subplots(
        figsize=(11.5, 2.4 + 0.62 * len(models)), facecolor=FIGURE_FACE
    )
    add_plot_header(
        figure,
        "Coste por modelo de imputacion",
        "Barras = tiempo medio por modelo. Rayado = entrenamiento/carga; solido = imputacion.",
    )
    style_axis(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")

    all_values: list[float] = []
    for offset, col in enumerate(present):
        values = summary[col].to_numpy(dtype=float)
        y = positions - group / 2 + bar_h * (offset + 0.5)
        bars = axis.barh(
            y, np.nan_to_num(values, nan=0.0), height=bar_h * 0.9,
            color=[colors[m] for m in models], edgecolor=EDGE_COLOR, linewidth=0.7,
            hatch=hatches[col], alpha=0.9, zorder=3, label=labels[col],
        )
        for bar, value in zip(bars, values, strict=False):
            if np.isfinite(value):
                all_values.append(value)
                axis.annotate(
                    f"{value:.2f}s", (bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
                    textcoords="offset points", xytext=(3, 0),
                    va="center", fontsize=8, color=TEXT_COLOR,
                )
    if all_values and min(all_values) > 0.0:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}s"))
    axis.set_yticks(positions)
    axis.set_yticklabels(models)
    axis.set_xlabel("tiempo (s)")

    # Legend distinguishes the two bar roles (color already encodes the model).
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#ccbfae", edgecolor=EDGE_COLOR,
                      hatch=hatches[col], label=labels[col])
        for col in present
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def save_time_vs_error_plot(
    output_path: Path, results_df: pd.DataFrame, metric: str = "MASE"
) -> bool:
    """Cost/accuracy scatter: mean imputation time (x) vs mean ``metric`` (y) per model."""
    metric = metric.upper()
    metric_label = (
        f"{metric} escalado"
        if metric in {"MAE", "RMSE"} and "Scale_Std" in results_df.columns
        else _metric_label(metric)
    )
    summary = summarize_timing(results_df)
    if summary.empty or IMPUTE_COL not in summary.columns or metric not in summary.columns:
        return False
    summary = summary.dropna(subset=[IMPUTE_COL, metric])
    if summary.empty:
        return False
    models = list(summary["Modelo"])
    colors = _model_colors(models)

    figure, axis = plt.subplots(figsize=(9.5, 6.8), facecolor=FIGURE_FACE)
    add_plot_header(
        figure,
        f"Coste vs precision — {metric_label}",
        "Media por modelo: X = tiempo de imputacion, Y = error (abajo-izquierda = mejor).",
    )
    style_axis(axis)
    axis.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.5)
    for _, row in summary.iterrows():
        color = colors[row["Modelo"]]
        axis.scatter(
            row[IMPUTE_COL], row[metric], s=95, color=color,
            edgecolor="white", linewidth=0.9, alpha=0.95, zorder=3,
        )
        axis.annotate(
            row["Modelo"], (row[IMPUTE_COL], row[metric]),
            textcoords="offset points", xytext=(7, 4), fontsize=9, color=TEXT_COLOR,
        )
    times = summary[IMPUTE_COL].to_numpy(dtype=float)
    if times.size and float(np.nanmin(times)) > 0.0:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}s"))
    axis.set_xlabel("tiempo de imputacion (s)")
    axis.set_ylabel(metric_label)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def render_timing_figures(
    results_df: pd.DataFrame,
    output_dir: Path,
    darts_train_seconds: Mapping[str, float] | None = None,
) -> list[Path]:
    """Render every applicable timing figure for one results frame; return saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in ("MASE", "RMSE", "MAE") if m in results_df.columns]
    jobs: list[tuple[Path, object]] = [
        (output_dir / "imputation_time_by_model.png",
         lambda p: save_time_by_model_plot(p, results_df, darts_train_seconds)),
    ]
    for metric in metrics:
        jobs.append(
            (output_dir / f"imputation_time_vs_{metric.lower()}.png",
             lambda p, m=metric: save_time_vs_error_plot(p, results_df, m))
        )
    saved: list[Path] = []
    for path, job in jobs:
        if job(path):
            saved.append(path)
        else:
            print(f"[skip] {path.name}: sin datos aplicables")
    return saved


def main(argv: list[str] | None = None) -> None:
    """CLI: render the timing figures from a persisted benchmark ``results.csv``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_csv", help="CSV de resultados del benchmark de imputacion")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directorio de salida (por defecto, junto al CSV)",
    )
    parser.add_argument(
        "--darts-train-csv", default=str(DEFAULT_DARTS_TRAIN_CSV),
        help="CSV con el tiempo de entrenamiento de los modelos Darts "
        "(training_time_seconds); se fusiona como train de esos modelos",
    )
    args = parser.parse_args(argv)

    results_path = Path(args.results_csv)
    if not results_path.exists():
        raise SystemExit(f"No existe el CSV: {results_path}")
    results_df = pd.read_csv(results_path)
    darts_train_seconds = load_darts_train_seconds(args.darts_train_csv)
    if not darts_train_seconds:
        print(f"[info] Sin tiempos de train de Darts en {args.darts_train_csv}")
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    for path in render_timing_figures(results_df, output_dir, darts_train_seconds):
        print(f"[info] Figura guardada en {path}")


if __name__ == "__main__":
    main()
