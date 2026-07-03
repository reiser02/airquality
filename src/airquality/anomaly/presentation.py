"""Plot styling and figure builders for the anomaly benchmark reports.

Defines the shared cream/base palette, per-category model colours, and the
three benchmark figures (VUS-PR distribution, training time, VUS-PR vs.
inference time) rendered by ``plot_benchmark_results`` from a saved
``results.json``.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

from .anomalies import (
    STL_ANOMALY_TYPES,
    apply_stl_anomaly_segment,
    inject_synthetic_anomalies,
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#d0d0d0",
    }
)

FIGURE_FACE = "#f6f1e8"
AXIS_FACE = "#fffaf2"
TEXT_COLOR = "#27313a"
EDGE_COLOR = "#3c4650"
GRID_COLOR = "#d8cabb"
MODEL_CATEGORY_COLORS = {
    "Statistical": "#5b8c5a",
    "Machine Learning": "#4b79a8",
    "Deep Learning": "#f28c38",
    "Ensemble": "#9b59b6",
}
STATISTICAL_MODELS = {"ModifiedZScore", "IQR", "PCA", "Hampel_w24", "Hampel_w6", "Prophet"}
MACHINE_LEARNING_MODELS = {"IsolationForest", "LOF"}
ENSEMBLE_MODELS = {"Ensemble"}

SERIES_COLOR = "#4b79a8"
SERIES_GHOST_COLOR = "#c9bdac"
ANOMALY_POINT_COLOR = "#d6453c"
# One colour per synthetic anomaly type, used to tell them apart in the
# ``combined`` panel where one segment of every type is shown side by side.
ANOMALY_TYPE_COLORS = {
    "spikes": "#d6453c",
    "scale": "#5b8c5a",
    "noise": "#9b59b6",
    "cutoff": "#f28c38",
    "contextual": "#8c6d4b",
    "speedup": "#cf6ba9",
}
ANOMALY_TYPE_LABELS = {
    "spikes": "spikes — pico puntual (±4σ)",
    "scale": "scale — segmento ×0.25 o ×2",
    "noise": "noise — ruido gaussiano añadido (2σ)",
    "cutoff": "cutoff — segmento aplanado al cuantil 0.75",
    "contextual": "contextual — segmento invertido + desplazado",
    "speedup": "speedup — segmento comprimido (×2 velocidad)",
    "combined": "combined — un segmento de cada tipo",
}


def style_axis(axis: plt.Axes) -> None:
    """Apply the shared cream background, grid, and spine styling to one axis."""
    axis.set_facecolor(AXIS_FACE)
    axis.tick_params(labelsize=9)
    axis.tick_params(colors=TEXT_COLOR)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.title.set_color(TEXT_COLOR)
    axis.grid(True, axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="x")
    axis.spines["left"].set_color("#c2b3a3")
    axis.spines["bottom"].set_color("#c2b3a3")


def model_category(model_name: str) -> str:
    """Classify a model name into Statistical/ML/Deep Learning/Ensemble."""
    if model_name in ENSEMBLE_MODELS:
        return "Ensemble"
    if model_name in STATISTICAL_MODELS:
        return "Statistical"
    if model_name in MACHINE_LEARNING_MODELS:
        return "Machine Learning"
    return "Deep Learning"


def model_color(model_name: str) -> str:
    """Return the palette colour assigned to the model's category."""
    return MODEL_CATEGORY_COLORS[model_category(model_name)]


def add_plot_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    """Add the left-aligned title + subtitle header shared by all figures."""
    figure.suptitle(title, x=0.075, y=0.98, ha="left", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.075, 0.94, subtitle, ha="left", fontsize=9, color="#6d6258")


def category_legend_handles(alpha: float = 0.9) -> list[plt.Rectangle]:
    """Build one legend rectangle per model category with the shared colours."""
    return [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=EDGE_COLOR, alpha=alpha, label=category)
        for category, color in MODEL_CATEGORY_COLORS.items()
    ]


def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce a JSON value to float, falling back to ``default`` when invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _series_vus_pr_values(summary: dict[str, object]) -> list[float]:
    """Extract the per-case VUS-PR values from one model summary."""
    return [
        _safe_float(entry.get("metrics", {}).get("vus_pr"))
        for entry in summary.get("series_results", [])
        if isinstance(entry, dict)
    ]


def _vus_pr_order_value(summary: dict[str, object]) -> float:
    """Sort key for models: macro VUS-PR, else the median of per-case values."""
    macro_metrics = summary.get("macro_metrics", {})
    if isinstance(macro_metrics, dict) and "vus_pr" in macro_metrics:
        return _safe_float(macro_metrics["vus_pr"])
    values = _series_vus_pr_values(summary)
    return float(np.median(values)) if values else float("inf")


def _training_time_order_value(summary: dict[str, object]) -> float:
    """Sort key for models: mean fit seconds (``inf`` when unavailable)."""
    timing = summary.get("timing", {})
    if isinstance(timing, dict):
        return _safe_float(timing.get("mean_fit_seconds"), default=float("inf"))
    return float("inf")


def save_vus_pr_distribution_plot(output_path: Path, model_summaries: dict[str, dict[str, object]]) -> None:
    """Render the per-model VUS-PR distribution figure (violin + box + points)."""
    model_names = sorted(model_summaries, key=lambda model_name: _vus_pr_order_value(model_summaries[model_name]))
    figure, axis = plt.subplots(figsize=(12.5, 7.0), facecolor=FIGURE_FACE)
    add_plot_header(figure, "VUS-PR Distribution Across Series", "Models ordered from lower to higher average VUS-PR")
    style_axis(axis)
    distributions = [_series_vus_pr_values(model_summaries[model_name]) for model_name in model_names]
    positions = np.arange(1, len(model_names) + 1)
    non_empty = [(position, model_name, values) for position, model_name, values in zip(positions, model_names, distributions, strict=False) if values]
    if non_empty:
        non_empty_positions = [position for position, _, _ in non_empty]
        non_empty_distributions = [values for _, _, values in non_empty]
        violins = axis.violinplot(
            non_empty_distributions,
            positions=non_empty_positions,
            vert=False,
            widths=0.82,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, (_, model_name, _) in zip(violins["bodies"], non_empty, strict=False):
            body.set_facecolor(model_color(model_name))
            body.set_alpha(0.22)
            body.set_edgecolor("none")
        boxplot = axis.boxplot(
            non_empty_distributions,
            positions=non_empty_positions,
            patch_artist=True,
            widths=0.28,
            showfliers=False,
            vert=False,
            medianprops={"color": TEXT_COLOR, "linewidth": 1.5},
            whiskerprops={"color": EDGE_COLOR, "linewidth": 0.9},
            capprops={"color": EDGE_COLOR, "linewidth": 0.9},
        )
        for patch, (_, model_name, _) in zip(boxplot["boxes"], non_empty, strict=False):
            patch.set_facecolor(model_color(model_name))
            patch.set_alpha(0.78)
            patch.set_edgecolor(EDGE_COLOR)
            patch.set_linewidth(0.8)
    for index, (model_name, values) in enumerate(zip(model_names, distributions, strict=False), start=1):
        if not values:
            continue
        axis.scatter(
            values,
            np.full(len(values), index),
            color=model_color(model_name),
            s=20,
            alpha=0.54,
            zorder=4,
            linewidths=0.25,
            edgecolors="white",
        )
    for index, model_name in enumerate(model_names, start=1):
        value = _vus_pr_order_value(model_summaries[model_name])
        if np.isfinite(value):
            axis.text(1.015, index, f"{value:.3f}", va="center", fontsize=8.5, color=TEXT_COLOR)
    axis.set_yticks(positions)
    axis.set_yticklabels(model_names)
    axis.set_xlim(0.0, 1.08)
    axis.set_xlabel("VUS-PR")
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    axis.legend(handles=category_legend_handles(alpha=0.78), loc="lower right", ncols=3)
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_training_time_plot(output_path: Path, model_summaries: dict[str, dict[str, object]]) -> None:
    """Render the mean-training-time bar chart (log scale when all values > 0)."""
    model_names = sorted(model_summaries, key=lambda model_name: _training_time_order_value(model_summaries[model_name]))
    figure, axis = plt.subplots(figsize=(12.5, 7.0), facecolor=FIGURE_FACE)
    add_plot_header(figure, "Average Training Time by Model", "Models ordered from lower to higher training time")
    style_axis(axis)
    positions = np.arange(len(model_names))
    mean_fit_seconds = [_training_time_order_value(model_summaries[model_name]) for model_name in model_names]
    colors = [model_color(model_name) for model_name in model_names]
    bars = axis.barh(positions, mean_fit_seconds, color=colors, edgecolor=EDGE_COLOR, linewidth=0.8, alpha=0.92, zorder=3)
    positive_values = [value for value in mean_fit_seconds if np.isfinite(value) and value > 0.0]
    if positive_values and len(positive_values) == len(mean_fit_seconds):
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}s"))
    axis.set_yticks(positions)
    axis.set_yticklabels(model_names)
    axis.invert_yaxis()
    axis.set_xlabel("Average Training Time (sec)")
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    if positive_values:
        median_value = float(np.median(positive_values))
        axis.axvline(median_value, color="#7d7266", linestyle=":", linewidth=1.1, zorder=2)
        axis.text(median_value, len(model_names) - 0.45, "median", rotation=90, va="top", ha="right", fontsize=8, color="#7d7266")
    offset = 4 if positive_values else 3
    for bar, value in zip(bars, mean_fit_seconds, strict=False):
        axis.annotate(
            f"{value:.2f}s",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
            textcoords="offset points",
            xytext=(offset, 0),
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )
    axis.legend(handles=category_legend_handles(alpha=0.92), loc="lower right", ncols=3)
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_vus_pr_vs_inference_plot(output_path: Path, model_summaries: dict[str, dict[str, object]]) -> None:
    """Render the macro VUS-PR vs. mean inference-time scatter per model."""
    figure, axis = plt.subplots(figsize=(12.5, 6.5), facecolor=FIGURE_FACE)
    style_axis(axis)
    for model_name, summary in model_summaries.items():
        vus_pr = summary["macro_metrics"]["vus_pr"]
        inference_seconds = summary["timing"]["mean_inference_seconds"]
        color = model_color(model_name)
        axis.scatter(
            inference_seconds,
            vus_pr,
            s=90,
            color=color,
            edgecolor="#444444",
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )
        axis.annotate(
            model_name,
            (inference_seconds, vus_pr),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color="#222222",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Average Inference Time (sec)")
    axis.set_ylabel("Average VUS-PR")
    axis.set_title("VUS-PR vs. Inference Time", fontsize=13)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="#444444", markersize=8, label=category)
        for category, color in MODEL_CATEGORY_COLORS.items()
    ]
    axis.legend(handles=legend_handles, loc="lower right", ncols=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _inject_all_anomaly_types(
    values: np.ndarray, seed: int
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Inject one segment of *every* type, each in its own slot.

    Mirrors :func:`airquality.anomaly.anomalies.inject_synthetic_anomalies` (same
    building block and length/scale parameters) so the shapes match, but places
    one segment per equal-width slot so all types stay visible and non-overlapping.
    Returns the injected series and the ``(start, end, type)`` of each segment.
    """
    rng = np.random.default_rng(seed)
    synthetic = values.astype(np.float32, copy=True)
    scale = float(np.std(values) or 1.0)
    max_len = max(4, min(32, len(values) // 20))
    slot = max(1, len(values) // len(STL_ANOMALY_TYPES))
    segments: list[tuple[int, int, str]] = []
    for index, anomaly_type in enumerate(STL_ANOMALY_TYPES):
        length = 1 if anomaly_type == "spikes" else int(rng.integers(2, max(3, min(max_len, slot - 2)) + 1))
        low = index * slot + 1
        high = max(low + 1, (index + 1) * slot - length - 1)
        start = min(int(rng.integers(low, high)), len(values) - length)
        end = start + length
        apply_stl_anomaly_segment(synthetic, values, start, end, anomaly_type, rng, scale)
        segments.append((start, end, anomaly_type))
    return synthetic.astype(np.float32), segments


def _shade_spans(axis: plt.Axes, x_values: np.ndarray, mask: np.ndarray, color: str) -> None:
    """Shade every contiguous ``True`` run of ``mask`` along the x-axis."""
    in_span = False
    span_start = 0
    for position, flagged in enumerate(mask):
        if flagged and not in_span:
            span_start, in_span = position, True
        elif not flagged and in_span:
            axis.axvspan(x_values[span_start], x_values[position], color=color, alpha=0.10)
            in_span = False
    if in_span:
        axis.axvspan(x_values[span_start], x_values[-1], color=color, alpha=0.10)


def save_synthetic_anomaly_types_plot(
    output_path: Path,
    values: np.ndarray,
    index: np.ndarray | None = None,
    *,
    series_label: str = "",
    seed: int = 7,
) -> None:
    """Plot a clean series and each synthetic anomaly type injected into it.

    One panel shows the untouched ``values``; the rest show every type in
    :data:`airquality.anomaly.anomalies.STL_ANOMALY_TYPES` plus a ``combined``
    panel where one segment of each type is laid out side by side and coloured by
    type (see :data:`ANOMALY_TYPE_COLORS`). ``index`` supplies the x-axis (e.g. a
    ``DatetimeIndex``); a plain range is used when omitted.
    """
    values = np.asarray(values, dtype=np.float32)
    x_values = np.asarray(index) if index is not None else np.arange(len(values))
    types = [*STL_ANOMALY_TYPES, "combined"]

    ncols = 2
    nrows = (len(types) + 1 + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.7 * nrows), facecolor=FIGURE_FACE, sharex=True
    )
    flat_axes = np.asarray(axes).ravel()
    subtitle = f"Serie real: {series_label}" if series_label else "Anomalías inyectadas sobre la serie real"
    add_plot_header(figure, "Tipos de anomalías sintéticas", subtitle)

    reference_axis = flat_axes[0]
    style_axis(reference_axis)
    reference_axis.plot(x_values, values, color=SERIES_COLOR, lw=1.0)
    reference_axis.set_title("Serie normal", fontsize=11, loc="left")
    reference_axis.set_ylabel("valor")

    for axis, anomaly_type in zip(flat_axes[1:], types, strict=False):
        style_axis(axis)
        axis.plot(x_values, values, color=SERIES_GHOST_COLOR, lw=0.9, label="normal", zorder=1)
        axis.set_title(ANOMALY_TYPE_LABELS[anomaly_type], fontsize=11, loc="left")
        axis.set_ylabel("valor")

        if anomaly_type == "combined":
            injected, segments = _inject_all_anomaly_types(values, seed)
            axis.plot(x_values, injected, color=SERIES_COLOR, lw=1.0, label="con anomalía", zorder=2)
            seen: set[str] = set()
            for start, end, segment_type in segments:
                color = ANOMALY_TYPE_COLORS[segment_type]
                segment = slice(start, max(end, start + 1))
                axis.scatter(
                    x_values[segment], injected[segment], color=color, s=22, zorder=3,
                    label=segment_type if segment_type not in seen else None,
                )
                axis.axvspan(x_values[start], x_values[min(end, len(x_values) - 1)], color=color, alpha=0.12)
                seen.add(segment_type)
            axis.legend(fontsize=7, loc="best", ncols=2, title="tipo de segmento")
        else:
            injected, labels = inject_synthetic_anomalies(values, f"raw-{anomaly_type}", seed)
            axis.plot(x_values, injected, color=SERIES_COLOR, lw=1.0, label="con anomalía", zorder=2)
            mask = labels.astype(bool)
            axis.scatter(x_values[mask], injected[mask], color=ANOMALY_POINT_COLOR, s=18, zorder=3, label="anomalía")
            _shade_spans(axis, x_values, mask, ANOMALY_POINT_COLOR)
            axis.legend(fontsize=8, loc="best")

    for axis in flat_axes[len(types) + 1 :]:
        axis.set_visible(False)

    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
