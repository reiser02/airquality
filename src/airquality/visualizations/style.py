"""Shared visual language and export helpers for persisted report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FIGURE_DPI = 300
FIGURE_FACE = "#f6f1e8"
AXIS_FACE = "#fffaf2"
TEXT_COLOR = "#27313a"
MUTED_TEXT = "#6d6258"
EDGE_COLOR = "#3c4650"
SPINE_COLOR = "#c2b3a3"
GRID_COLOR = "#d8cabb"
SERIES_COLOR = "#4b79a8"
SERIES_GHOST_COLOR = "#c9bdac"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": GRID_COLOR,
        "savefig.dpi": FIGURE_DPI,
        "savefig.facecolor": FIGURE_FACE,
    }
)


def style_axis(axis: plt.Axes) -> None:
    """Apply the shared surface, typography, grid, and spine treatment."""
    axis.set_facecolor(AXIS_FACE)
    axis.tick_params(labelsize=9, colors=TEXT_COLOR)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.title.set_color(TEXT_COLOR)
    axis.grid(True, axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="x")
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)


def add_plot_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    """Add the shared left-aligned title and explanatory subtitle."""
    figure.suptitle(
        title,
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.075,
        0.94,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=MUTED_TEXT,
    )


def save_figure(figure: plt.Figure, output_path: str | Path) -> Path:
    """Save one report figure with consistent print-quality raster settings."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor=FIGURE_FACE,
        edgecolor="none",
    )
    plt.close(figure)
    return path
