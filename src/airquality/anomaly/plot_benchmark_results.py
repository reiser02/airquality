"""Standalone plot generator for anomaly-benchmark runs (genias-style).

The benchmark (:mod:`airquality.anomaly.benchmark`) only persists ``results.json``
+ ``scores.npz``; this separate script renders the three benchmark plots from a
saved ``results.json`` so plotting is decoupled from the (expensive) run.

Run::

    uv run python -m airquality.anomaly.plot_benchmark_results reports/anomaly/<run>/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import DEFAULT_MAX_DETECTION_RATE
from .presentation import (
    save_detection_rate_distribution_plot,
    save_detection_rate_vs_inference_plot,
    save_training_time_plot,
)


def save_benchmark_plots(results_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Render the three benchmark plots from a benchmark ``results.json``.

    Plots are written next to ``results.json`` unless ``output_dir`` is given.
    """
    results_path = Path(results_path)
    with results_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    plot_dir = Path(output_dir) if output_dir is not None else results_path.parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    model_summaries = summary["models"]
    max_detection_rate = float(summary.get("config", {}).get("max_detection_rate", DEFAULT_MAX_DETECTION_RATE))
    plot_paths = {
        "metrics_plot": plot_dir / summary.get("metrics_plot", "detection_rate_distribution.png"),
        "scatter_plot": plot_dir / summary.get("scatter_plot", "detection_rate_vs_inference.png"),
        "training_plot": plot_dir / summary.get("training_plot", "training_time.png"),
    }
    save_detection_rate_distribution_plot(plot_paths["metrics_plot"], model_summaries, max_detection_rate)
    save_detection_rate_vs_inference_plot(plot_paths["scatter_plot"], model_summaries, max_detection_rate)
    save_training_time_plot(plot_paths["training_plot"], model_summaries)
    return plot_paths


def main() -> None:
    """CLI entry point: render the plots for one ``results.json`` and print paths."""
    parser = argparse.ArgumentParser(description="Generate anomaly-benchmark plots from results.json")
    parser.add_argument("results", help="Path to a benchmark results.json")
    parser.add_argument("--output-dir", default=None, help="Directory for generated plot images")
    args = parser.parse_args()

    plot_paths = save_benchmark_plots(args.results, args.output_dir)
    print(json.dumps({name: str(path) for name, path in plot_paths.items()}, indent=2))


if __name__ == "__main__":
    main()
