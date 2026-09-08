"""Render forecasting benchmark figures from persisted run tables.

Reads the CSVs persisted by :mod:`airquality.forecasting.pipeline`
(``results.csv``, ``detection.csv`` and the optional paired-foundation summary)
from one run directory and renders the
report figures next to them — plots regenerate any time without recomputing
detections or backtests:

The main benchmark reports ``rmsse`` and ``mase`` plus MAE/RMSE relative to the
matching ``raw`` arm (see ``airquality.forecasting.pipeline.METRIC_COLS``).

- ``arm_error_{rmsse,mase,relmae,relrmse}.png`` — per-arm error across series (dots) with the
  mean labeled, one panel per forecast model; the ``raw`` mean is the reference.
- ``improvement_{rmsse,mase,relmae,relrmse}.png`` — series×arm heatmap of the % improvement vs
  ``raw`` (blue = the arm forecasts better, red = worse), every cell annotated.
- ``detector_selection.png`` — how often each detector ends up in the final
  mask per strategy, plus each strategy's detection-rate distribution.
- ``imputation_effect_{rmsse,mase,relmae,relrmse}.png`` — paired scatter (impute vs noimpute)
  per strategy; points below the diagonal mean imputation helped.
- ``train_time.png`` / ``inference_time.png`` — per-arm training / holdout
  inference time across series (dots) with the mean labeled, one panel per
  forecast model (uses the ``train_seconds`` / ``inference_seconds`` columns).
- ``{train,inference}_time_vs_{rmsse,mase,relmae,relrmse}.png`` — cost/accuracy scatter: mean
  training / inference time vs mean error per (arm, model), so the bottom-left
  region is fast and precise.
- ``foundation_preprocessing_recovery_{mase,rmsse}.png`` — paired error recovered
  after preprocessing synthetic context anomalies, by model, strategy and type.

Arm colors follow the strategy *family* and stay fixed across figures
(``raw`` is deliberately neutral: it is the baseline, not a competing series).

Run with::

    uv run python -m airquality.visualizations.forecasting [run_dir]

``run_dir`` defaults to the newest stamped run under ``reports/forecasting``.
"""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airquality.visualizations.anomaly import (
    EDGE_COLOR,
    FIGURE_FACE,
    GRID_COLOR,
    TEXT_COLOR,
    style_axis,
)
from airquality.forecasting.foundation_preprocessing import FOUNDATION_METRICS
from airquality.forecasting.pipeline import METRIC_COLS, RAW_ARM, RAW_FROZEN_ARM

MUTED_TEXT = "#6d6258"

#: Fixed color per strategy family (validated for CVD separation + contrast on
#: the cream surface). ``raw`` is neutral on purpose — it is the reference.
ARM_FAMILY_COLORS = {
    RAW_ARM: "#6d6258",
    RAW_FROZEN_ARM: "#2a7f76",
    "unlabeled": "#3d7ab5",
    "inject-best": "#cf6f1e",
    "inject-vote": "#9b59b6",
    "inject-soft": "#5b8c5a",
}
#: Assigned in order to strategy families beyond the known ones (never cycled
#: within one run: each new family takes the next free slot).
EXTRA_FAMILY_COLORS = ("#cf6ba9", "#8c6d4b", "#27313a")

#: Diverging map for "% improvement vs raw": blue = better, red = worse,
#: neutral cream midpoint (never a hue at the center).
IMPROVEMENT_CMAP = LinearSegmentedColormap.from_list(
    "mejora_vs_raw", ["#d6453c", "#f3ece0", "#3d7ab5"]
)

MODEL_MARKERS = ("o", "s", "^", "D", "v", "P")

#: Display names for the benchmark metrics.
METRIC_LABELS = {
    "rmsse": "RMSSE",
    "mase": "MASE",
    "relmae": "MAE relativo a raw",
    "relrmse": "RMSE relativo a raw",
}


def _metric_label(metric: str) -> str:
    """Human display name of one metric column."""
    return METRIC_LABELS.get(metric, metric.upper())


def _add_header(figure: plt.Figure, title: str, subtitle: str) -> float:
    """Left-aligned title + (multi-line) subtitle sized in inches, not fractions.

    Unlike :func:`airquality.visualizations.anomaly.add_plot_header` (tuned for
    7-inch figures), the gap is computed from the figure height so short
    figures don't overlap title and subtitle. Returns the top figure fraction
    to reserve for the axes (pass it to ``tight_layout``/``subplots_adjust``).
    """
    height = float(figure.get_size_inches()[1])
    lines = subtitle.split("\n")
    figure.suptitle(
        title, x=0.05, y=1.0 - 0.10 / height,
        ha="left", va="top", fontsize=14, fontweight="bold", color=TEXT_COLOR,
    )
    for index, line in enumerate(lines):
        figure.text(
            0.05, 1.0 - (0.44 + 0.17 * index) / height, line,
            ha="left", va="top", fontsize=9, color=MUTED_TEXT,
        )
    return 1.0 - (0.55 + 0.17 * len(lines)) / height


def _family(arm: str) -> str:
    """Strategy family of one arm name (``unlabeled+impute`` → ``unlabeled``)."""
    if arm == RAW_FROZEN_ARM:
        return arm
    return arm.split("+", 1)[0]


def family_colors(arms: list[str]) -> dict[str, str]:
    """Map every arm family to its fixed color, extending for custom families."""
    colors = dict(ARM_FAMILY_COLORS)
    extras = iter(EXTRA_FAMILY_COLORS)
    for arm in arms:
        family = _family(arm)
        if family not in colors:
            colors[family] = next(extras, TEXT_COLOR)
    return colors


def improvement_table(results_df: pd.DataFrame, metric: str, model: str) -> pd.DataFrame:
    """Per-series % improvement vs ``raw`` for one forecast model.

    Rows are series, columns the non-raw arms (input order), values
    ``100 * (raw - arm) / raw`` — positive means the arm forecasts better.
    """
    subset = results_df[results_df["model"] == model]
    wide = subset.pivot(index="series", columns="arm", values=metric)
    if RAW_ARM not in wide.columns:
        return pd.DataFrame()
    arms = [arm for arm in dict.fromkeys(subset["arm"]) if arm != RAW_ARM]
    baseline = wide[RAW_ARM]
    table = pd.DataFrame(index=wide.index)
    for arm in arms:
        table[arm] = (
            100.0 * (baseline - wide[arm]) / baseline.where(baseline != 0)
            if arm in wide
            else np.nan
        )
    return table


def selection_counts(detection_df: pd.DataFrame) -> pd.DataFrame:
    """Series count per (detector, strategy) from the ``detectors`` CSV column."""
    counts: dict[str, dict[str, int]] = {}
    for _, row in detection_df.iterrows():
        names = [name for name in str(row.get("detectors", "") or "").split(",") if name]
        for name in names:
            counts.setdefault(name, {})
            counts[name][row["strategy"]] = counts[name].get(row["strategy"], 0) + 1
    table = pd.DataFrame(counts).T.fillna(0.0)
    if table.empty:
        return table
    strategy_order = [s for s in dict.fromkeys(detection_df["strategy"]) if s in table.columns]
    return table[strategy_order].loc[table.sum(axis=1).sort_values().index]


def imputation_pairs(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Paired ``+impute`` / ``+noimpute`` metric values per (series, model, strategy)."""
    subset = results_df[results_df["strategy"] != "none"].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["variant"] = np.where(subset["imputed"], "impute", "noimpute")
    wide = subset.pivot(
        index=["series", "model", "strategy"], columns="variant", values=metric
    )
    if not {"impute", "noimpute"} <= set(wide.columns):
        return pd.DataFrame()
    return wide.dropna().reset_index()


def _finite(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _save_arm_dot_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    column: str,
    *,
    title: str,
    subtitle: str,
    xlabel: str,
    value_fmt: str = "{:.2f}",
    reference: bool = True,
) -> bool:
    """Dot plot of ``column`` per arm (points = series, tick = mean), per model.

    ``reference=True`` draws each panel's ``raw`` mean as a dashed vertical line
    (the baseline to beat); pass ``value_fmt`` to control the mean annotation.
    """
    models = list(dict.fromkeys(results_df["model"]))
    arms = list(dict.fromkeys(results_df["arm"]))
    if not models or not arms or column not in results_df.columns:
        return False
    if _finite(results_df[column]).size == 0:
        return False
    colors = family_colors(arms)

    figure, axes = plt.subplots(
        1, len(models),
        figsize=(1.0 + 5.4 * len(models), 2.4 + 0.62 * len(arms)),
        facecolor=FIGURE_FACE, sharey=True, squeeze=False,
    )
    top = _add_header(figure, title, subtitle)
    positions = np.arange(len(arms))
    for axis, model in zip(axes[0], models, strict=False):
        style_axis(axis)
        axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
        axis.grid(False, axis="y")
        subset = results_df[results_df["model"] == model]
        raw_mean = np.nan
        for position, arm in zip(positions, arms, strict=False):
            values = _finite(subset.loc[subset["arm"] == arm, column])
            if values.size == 0:
                continue
            color = colors[_family(arm)]
            filled = ("+noimpute" not in arm)
            axis.scatter(
                values,
                np.full(values.size, position),
                s=34,
                facecolor=color if filled else FIGURE_FACE,
                edgecolor=color,
                linewidths=1.2,
                alpha=0.9,
                zorder=3,
            )
            mean = float(values.mean())
            axis.scatter([mean], [position], marker="|", s=340, color=color, linewidths=2.6, zorder=4)
            axis.annotate(
                value_fmt.format(mean),
                (mean, position),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=TEXT_COLOR,
            )
            if arm == RAW_ARM:
                raw_mean = mean
        if reference and np.isfinite(raw_mean):
            axis.axvline(raw_mean, color=colors[RAW_ARM], linestyle="--", linewidth=1.1, alpha=0.8, zorder=2)
        axis.set_title(model, fontsize=11, loc="left")
        axis.set_xlabel(xlabel)
    axes[0][0].set_yticks(positions)
    axes[0][0].set_yticklabels(arms)
    axes[0][0].invert_yaxis()

    families = list(dict.fromkeys(_family(arm) for arm in arms))
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[family],
               markeredgecolor=colors[family], markersize=8, label=family)
        for family in families
    ]
    handles += [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=TEXT_COLOR,
               markeredgecolor=TEXT_COLOR, markersize=8, label="con imputacion / vistas raw"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=FIGURE_FACE,
               markeredgecolor=TEXT_COLOR, markersize=8, label="sin imputacion"),
    ]
    # Figure-level legend below the panels: inside the axes it hides points.
    figure.legend(
        handles=handles, loc="lower center", ncols=min(len(handles), 4),
        fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    height = float(figure.get_size_inches()[1])
    figure.tight_layout(rect=(0, 0.55 / height, 1, top))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def save_arm_error_plot(output_path: Path, results_df: pd.DataFrame, metric: str = "rmsse") -> bool:
    """Dot plot of ``metric`` per arm (points = series, tick = mean), per model."""
    return _save_arm_dot_plot(
        output_path,
        results_df,
        metric,
        title=f"Error de prediccion por brazo — {_metric_label(metric)}",
        subtitle=(
            "Puntos = series; barra vertical = media del brazo (menor = mejor).\n"
            "Relleno = con imputacion (o vista raw); hueco = sin imputacion. Linea discontinua = media de raw."
        ),
        xlabel=_metric_label(metric),
        value_fmt="{:.2f}",
        reference=True,
    )


def save_train_time_plot(output_path: Path, results_df: pd.DataFrame) -> bool:
    """Dot plot of training time (s) per arm (points = series, tick = mean), per model."""
    return _save_arm_dot_plot(
        output_path,
        results_df,
        "train_seconds",
        title="Tiempo de entrenamiento por brazo",
        subtitle=(
            "Puntos = series; barra vertical = media del brazo (menor = mas rapido).\n"
            "Relleno = con imputacion (o vista raw); hueco = sin imputacion."
        ),
        xlabel="tiempo de entrenamiento (s)",
        value_fmt="{:.1f}s",
        reference=False,
    )


def save_inference_time_plot(output_path: Path, results_df: pd.DataFrame) -> bool:
    """Dot plot of holdout inference time (s) per arm (points = series, tick = mean), per model."""
    return _save_arm_dot_plot(
        output_path,
        results_df,
        "inference_seconds",
        title="Tiempo de inferencia por brazo",
        subtitle=(
            "Puntos = series; barra vertical = media del brazo (menor = mas rapido).\n"
            "Tiempo del forecast sobre el holdout. Relleno = con imputacion (o vista raw); hueco = sin imputacion."
        ),
        xlabel="tiempo de inferencia (s)",
        value_fmt="{:.2f}s",
        reference=False,
    )


def _save_time_vs_error_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    time_col: str,
    metric: str,
    *,
    title: str,
    xlabel: str,
) -> bool:
    """Cost/accuracy scatter: mean ``time_col`` (x) vs mean ``metric`` (y) per arm×model.

    One point per (arm, model), averaged across series. Color = strategy family,
    marker = forecast model, filled/hollow = with/without imputation — so the
    bottom-left region is the sweet spot (fast *and* accurate).
    """
    if time_col not in results_df.columns or metric not in results_df.columns:
        return False
    df = results_df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    agg = (
        df.groupby(["arm", "model"], sort=False)[[time_col, metric]]
        .mean()
        .reset_index()
        .dropna(subset=[time_col, metric])
    )
    if agg.empty:
        return False

    arms = list(dict.fromkeys(results_df["arm"]))
    models = list(dict.fromkeys(results_df["model"]))
    colors = family_colors(arms)
    marker_by_model = {
        model: marker for model, marker in zip(models, cycle(MODEL_MARKERS))
    }

    figure, axis = plt.subplots(figsize=(8.6, 7.2), facecolor=FIGURE_FACE)
    top = _add_header(
        figure,
        title,
        f"Media por (brazo, modelo): X = {xlabel}, Y = error (abajo-izq = mejor).\n"
        "Color = estrategia, marcador = modelo, relleno/hueco = con/sin imputacion.",
    )
    style_axis(axis)
    axis.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.5)
    for _, row in agg.iterrows():
        color = colors[_family(row["arm"])]
        filled = ("+noimpute" not in row["arm"])
        axis.scatter(
            row[time_col], row[metric],
            s=78, marker=marker_by_model.get(row["model"], "o"),
            facecolor=color if filled else FIGURE_FACE,
            edgecolor=color if not filled else "white",
            linewidths=0.9, alpha=0.92, zorder=3,
        )
    times = _finite(agg[time_col])
    if times.size and float(times.min()) > 0.0:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:g}s"))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(_metric_label(metric))

    families = list(dict.fromkeys(_family(arm) for arm in arms))
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[family],
               markeredgecolor="white", markersize=8, label=family)
        for family in families
    ]
    handles += [
        Line2D([0], [0], marker=marker_by_model.get(model, "o"), linestyle="none",
               markerfacecolor=MUTED_TEXT, markeredgecolor="white", markersize=8, label=model)
        for model in models
    ]
    axis.legend(handles=handles, loc="best", fontsize=8, ncols=2)
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def save_train_time_vs_error_plot(
    output_path: Path, results_df: pd.DataFrame, metric: str = "rmsse"
) -> bool:
    """Cost/accuracy scatter: mean training time (x) vs mean ``metric`` (y) per arm×model."""
    return _save_time_vs_error_plot(
        output_path, results_df, "train_seconds", metric,
        title=f"Coste vs precision (entrenamiento) — {_metric_label(metric)}",
        xlabel="tiempo de entrenamiento (s)",
    )


def save_inference_time_vs_error_plot(
    output_path: Path, results_df: pd.DataFrame, metric: str = "rmsse"
) -> bool:
    """Cost/accuracy scatter: mean inference time (x) vs mean ``metric`` (y) per arm×model."""
    return _save_time_vs_error_plot(
        output_path, results_df, "inference_seconds", metric,
        title=f"Coste vs precision (inferencia) — {_metric_label(metric)}",
        xlabel="tiempo de inferencia (s)",
    )


def save_improvement_heatmap(output_path: Path, results_df: pd.DataFrame, metric: str) -> bool:
    """Series×arm heatmap of % improvement vs raw (blue = better), per model."""
    models = list(dict.fromkeys(results_df["model"]))
    tables = {model: improvement_table(results_df, metric, model) for model in models}
    tables = {model: table for model, table in tables.items() if not table.empty}
    if not tables:
        return False

    widest = max(len(table.columns) for table in tables.values())
    tallest = max(len(table.index) for table in tables.values())
    figure, axes = plt.subplots(
        1, len(tables),
        figsize=(2.4 + 1.5 * widest * len(tables), 3.2 + 0.5 * tallest),
        facecolor=FIGURE_FACE, sharey=False, squeeze=False,
    )
    top = _add_header(
        figure,
        f"Mejora frente a raw — {_metric_label(metric)}",
        "% de mejora del error respecto al brazo raw por serie:\n"
        "azul = el brazo predice mejor que raw, rojo = peor.",
    )
    figure.subplots_adjust(top=top - 0.02)
    magnitudes = [np.nanmax(np.abs(table.to_numpy(dtype=float))) for table in tables.values()]
    span = max([m for m in magnitudes if np.isfinite(m)] or [1.0])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)

    mesh = None
    for axis, (model, table) in zip(axes[0], tables.items(), strict=False):
        style_axis(axis)
        axis.grid(False)
        data = table.to_numpy(dtype=float)
        mesh = axis.pcolormesh(
            np.ma.masked_invalid(data),
            cmap=IMPROVEMENT_CMAP,
            norm=norm,
            edgecolors=FIGURE_FACE,
            linewidths=2.0,
        )
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data[row, col]
                if not np.isfinite(value):
                    axis.text(col + 0.5, row + 0.5, "–", ha="center", va="center",
                              fontsize=9, color=MUTED_TEXT)
                    continue
                strong = abs(value) > 0.6 * span
                axis.text(
                    col + 0.5, row + 0.5, f"{value:+.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if strong else TEXT_COLOR,
                )
        axis.set_xticks(np.arange(len(table.columns)) + 0.5)
        axis.set_xticklabels(table.columns, rotation=30, ha="right", fontsize=8.5)
        axis.set_yticks(np.arange(len(table.index)) + 0.5)
        axis.set_yticklabels(table.index, fontsize=8.5)
        axis.invert_yaxis()
        axis.set_title(model, fontsize=11, loc="left")
    if mesh is not None:
        bar = figure.colorbar(mesh, ax=axes[0].tolist(), fraction=0.04, pad=0.02)
        bar.set_label("% mejora vs raw", color=TEXT_COLOR, fontsize=9)
        bar.ax.tick_params(labelsize=8, colors=TEXT_COLOR)
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def save_detector_selection_plot(output_path: Path, detection_df: pd.DataFrame) -> bool:
    """Selection frequency per detector×strategy + detection rate per strategy."""
    if detection_df.empty:
        return False
    counts = selection_counts(detection_df)
    strategies = list(dict.fromkeys(detection_df["strategy"]))
    colors = family_colors(strategies)
    n_series = detection_df["series"].nunique()

    figure, (left, right) = plt.subplots(
        1, 2,
        figsize=(13.0, 2.6 + max(0.55 * len(counts.index), 0.8 * len(strategies), 2.5)),
        facecolor=FIGURE_FACE, gridspec_kw={"width_ratios": [3, 2]},
    )
    top = _add_header(
        figure,
        "Seleccion de detectores por estrategia",
        "Izquierda: nº de series en las que cada detector forma la mascara final "
        "(unlabeled = supervivientes del filtro; inject-* = elegidos por VUS-PR).\n"
        "Derecha: fraccion de puntos marcada como anomala por estrategia (puntos = series, barra = media).",
    )

    style_axis(left)
    left.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    left.grid(False, axis="y")
    if not counts.empty:
        group = 0.8
        bar_height = group / max(len(strategies), 1)
        base_positions = np.arange(len(counts.index))
        for offset, strategy in enumerate(strategies):
            values = counts[strategy].to_numpy(dtype=float) if strategy in counts else np.zeros(len(counts.index))
            y = base_positions - group / 2 + bar_height * (offset + 0.5)
            bars = left.barh(
                y, values, height=bar_height * 0.9,
                color=colors[strategy], edgecolor=EDGE_COLOR, linewidth=0.6,
                label=strategy, zorder=3,
            )
            for bar, value in zip(bars, values, strict=False):
                if value > 0:
                    left.annotate(
                        f"{value:g}",
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(3, 0),
                        va="center", fontsize=8, color=TEXT_COLOR,
                    )
        left.set_yticks(base_positions)
        left.set_yticklabels(counts.index, fontsize=9)
        left.set_xlabel(f"nº de series (de {n_series})")

    style_axis(right)
    right.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    right.grid(False, axis="y")
    positions = np.arange(len(strategies))
    for position, strategy in zip(positions, strategies, strict=False):
        rates = _finite(detection_df.loc[detection_df["strategy"] == strategy, "detection_rate"])
        if rates.size == 0:
            continue
        right.scatter(
            rates, np.full(rates.size, position),
            s=34, color=colors[strategy], alpha=0.75, zorder=3,
            linewidths=0.4, edgecolors="white",
        )
        mean = float(rates.mean())
        right.scatter([mean], [position], marker="|", s=320, color=colors[strategy], linewidths=2.4, zorder=4)
        right.annotate(
            f"{100.0 * mean:.2f}%", (mean, position),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8, color=TEXT_COLOR,
        )
    right.set_yticks(positions)
    right.set_yticklabels(strategies, fontsize=9)
    right.invert_yaxis()
    right.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{100.0 * value:g}%"))
    right.set_xlabel("tasa de deteccion")

    # Figure-level legend: bars can span the full axis, so any inside spot collides.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[strategy], edgecolor=EDGE_COLOR, label=strategy)
        for strategy in strategies
    ]
    figure.legend(
        handles=handles, loc="lower center", ncols=min(len(strategies), 4),
        fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    height = float(figure.get_size_inches()[1])
    figure.tight_layout(rect=(0, 0.55 / height, 1, top))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def save_imputation_effect_plot(
    output_path: Path, results_df: pd.DataFrame, metric: str = "rmsse"
) -> bool:
    """Paired scatter: ``metric`` with imputation (y) vs without (x), per strategy."""
    pairs = imputation_pairs(results_df, metric)
    if pairs.empty:
        return False
    strategies = list(dict.fromkeys(pairs["strategy"]))
    models = list(dict.fromkeys(pairs["model"]))
    colors = family_colors(strategies)
    marker_by_model = {
        model: marker for model, marker in zip(models, cycle(MODEL_MARKERS))
    }

    figure, axis = plt.subplots(figsize=(8.2, 7.4), facecolor=FIGURE_FACE)
    top = _add_header(
        figure,
        f"Efecto de la imputacion — {_metric_label(metric)}",
        "Cada punto: una (serie, modelo, estrategia). Bajo la diagonal = imputar mejora el forecast.",
    )
    style_axis(axis)
    axis.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.5)

    for strategy in strategies:
        for model in models:
            chunk = pairs[(pairs["strategy"] == strategy) & (pairs["model"] == model)]
            if chunk.empty:
                continue
            axis.scatter(
                chunk["noimpute"], chunk["impute"],
                s=52, marker=marker_by_model[model], color=colors[strategy],
                edgecolor="white", linewidths=0.6, alpha=0.9, zorder=3,
            )
    finite = np.concatenate([_finite(pairs["noimpute"]), _finite(pairs["impute"])])
    low, high = float(finite.min()), float(finite.max())
    pad = 0.06 * (high - low or 1.0)
    lims = (low - pad, high + pad)
    axis.plot(lims, lims, color=MUTED_TEXT, linestyle="--", linewidth=1.1, zorder=2)
    axis.set_xlim(lims)
    axis.set_ylim(lims)
    axis.set_aspect("equal")
    axis.text(0.97, 0.03, "imputar mejora", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=9, color=MUTED_TEXT, style="italic")
    axis.text(0.03, 0.97, "imputar empeora", transform=axis.transAxes,
              ha="left", va="top", fontsize=9, color=MUTED_TEXT, style="italic")
    axis.set_xlabel(f"{_metric_label(metric)} sin imputacion")
    axis.set_ylabel(f"{_metric_label(metric)} con imputacion")

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[strategy],
               markeredgecolor="white", markersize=8, label=strategy)
        for strategy in strategies
    ]
    handles += [
        Line2D([0], [0], marker=marker_by_model[model], linestyle="none", markerfacecolor=MUTED_TEXT,
               markeredgecolor="white", markersize=8, label=model)
        for model in models
    ]
    axis.legend(handles=handles, loc="upper left", fontsize=8, ncols=2,
                bbox_to_anchor=(0.0, 0.93))
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def save_foundation_preprocessing_plot(
    output_path: Path,
    summary_df: pd.DataFrame,
    metric: str = "mase",
) -> bool:
    """Plot mean paired recovery from corrupted synthetic contexts."""
    value_col = f"{metric}_recovery"
    required = {"model", "strategy", "anomaly_type", value_col}
    if summary_df.empty or not required <= set(summary_df.columns):
        return False

    data = summary_df.copy()
    data["row"] = (
        data["model"].astype(str)
        + " / "
        + data["strategy"].astype(str)
    )
    table = data.pivot_table(
        index="row",
        columns="anomaly_type",
        values=value_col,
        aggfunc="mean",
        sort=False,
    )
    if table.empty or not np.isfinite(table.to_numpy(dtype=float)).any():
        return False

    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(float(np.max(np.abs(finite))), 1e-9)
    figure, axis = plt.subplots(
        figsize=(max(7.5, 1.25 * len(table.columns)), max(4.5, 0.34 * len(table) + 1.8)),
        facecolor=FIGURE_FACE,
    )
    axis.set_facecolor(FIGURE_FACE)
    image = axis.imshow(
        values,
        aspect="auto",
        cmap=IMPROVEMENT_CMAP,
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(range(len(table.columns)), table.columns)
    axis.set_yticks(range(len(table.index)), table.index)
    axis.tick_params(axis="x", rotation=25)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                axis.text(
                    col,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=TEXT_COLOR,
                )
    axis.set_xlabel("Tipo de anomalia sintetica")
    axis.set_ylabel("Foundation / estrategia")
    style_axis(axis)
    top = _add_header(
        figure,
        f"Recuperacion del dano sintetico — {_metric_label(metric)}",
        "Positivo: el preprocesamiento reduce el error frente al contexto corrompido.",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Error recuperado")
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def render_run_figures(run_dir: Path) -> list[Path]:
    """Render every applicable figure for one run directory; return saved paths."""
    results_df = pd.read_csv(run_dir / "results.csv")
    detection_path = run_dir / "detection.csv"
    detection_df = pd.read_csv(detection_path) if detection_path.exists() else pd.DataFrame()
    foundation_path = run_dir / "foundation_preprocessing_summary.csv"
    foundation_df = (
        pd.read_csv(foundation_path) if foundation_path.exists() else pd.DataFrame()
    )
    if any("regime" in table.columns for table in (results_df, detection_df, foundation_df)):
        raise ValueError(
            "El run usa el esquema antiguo con regímenes; vuelve a ejecutar el benchmark"
        )

    saved: list[Path] = []
    jobs = [
        *(
            (
                run_dir / f"arm_error_{metric}.png",
                lambda path, metric=metric: save_arm_error_plot(path, results_df, metric),
            )
            for metric in METRIC_COLS
        ),
        *(
            (
                run_dir / f"improvement_{metric}.png",
                lambda path, metric=metric: save_improvement_heatmap(path, results_df, metric),
            )
            for metric in METRIC_COLS
        ),
        (run_dir / "detector_selection.png", lambda path: save_detector_selection_plot(path, detection_df)),
        *(
            (
                run_dir / f"imputation_effect_{metric}.png",
                lambda path, metric=metric: save_imputation_effect_plot(path, results_df, metric),
            )
            for metric in METRIC_COLS
        ),
        (run_dir / "train_time.png", lambda path: save_train_time_plot(path, results_df)),
        *(
            (
                run_dir / f"train_time_vs_{metric}.png",
                lambda path, metric=metric: save_train_time_vs_error_plot(path, results_df, metric),
            )
            for metric in METRIC_COLS
        ),
        (run_dir / "inference_time.png", lambda path: save_inference_time_plot(path, results_df)),
        *(
            (
                run_dir / f"inference_time_vs_{metric}.png",
                lambda path, metric=metric: save_inference_time_vs_error_plot(path, results_df, metric),
            )
            for metric in METRIC_COLS
        ),
        *(
            (
                run_dir / f"foundation_preprocessing_recovery_{metric}.png",
                lambda path, metric=metric: save_foundation_preprocessing_plot(path, foundation_df, metric),
            )
            for metric in FOUNDATION_METRICS
        ),
    ]
    for path, job in jobs:
        if job(path):
            saved.append(path)
        else:
            print(f"[skip] {path.name}: sin datos aplicables")
    return saved


def _latest_run_dir(base: Path) -> Path | None:
    """Newest stamped run directory containing a ``results.csv``."""
    candidates = sorted(
        (child for child in base.iterdir() if (child / "results.csv").exists()),
        key=lambda child: child.name,
    )
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> None:
    """CLI: render the figures for one (default: the newest) benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="Directorio del run (por defecto, el mas reciente en reports/forecasting)",
    )
    args = parser.parse_args(argv)

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        base = Path("reports") / "forecasting"
        run_dir = _latest_run_dir(base) if base.exists() else None
        if run_dir is None:
            raise SystemExit(f"No hay runs con results.csv bajo {base}")
    if not (run_dir / "results.csv").exists():
        raise SystemExit(f"{run_dir} no contiene results.csv")

    for path in render_run_figures(run_dir):
        print(f"[info] Figura guardada en {path}")


if __name__ == "__main__":
    main()
