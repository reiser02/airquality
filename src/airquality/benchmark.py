"""Zero-argument terminal entrypoint for the configured Monte Carlo benchmark."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from airquality.config import cfg_get_int
from airquality.imputation.run_benchmark import (
    run_imputation_benchmark_parallel,
    run_imputation_benchmark_parallel_montecarlo,
)


# Base palette mirrored from `airquality.anomaly.presentation` (same "base color"
# the anomaly benchmark plots use). Copied verbatim on purpose: importing that
# module would drag in the heavy STL/anomaly stack just for a handful of colours.
FIGURE_FACE = "#f6f1e8"
AXIS_FACE = "#fffaf2"
TEXT_COLOR = "#27313a"
GRID_COLOR = "#d8cabb"
SPINE_COLOR = "#c2b3a3"
BAND_ALPHA = 0.15

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
)
# Curated colours so key models stay visually stable across runs; the linear
# baseline gets the standout red so its gap=1 behaviour is easy to spot.
_MODEL_LINE_COLORS = {
    "LinearInterp": "#d6453c",
    "TSPulse": "#9b59b6",
    "TSPulse_FineTuned": "#8c6d4b",
    "Prophet": "#5b8c5a",
    "TiDE": "#4b79a8",
    "NHiTS": "#f28c38",
    "TSMixer": "#6d6258",
    "RNN": "#27313a",
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
    ax.set_facecolor(AXIS_FACE)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linestyle="--", alpha=0.65)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)


def _aggregate_metrics_by_gap(
    results_mc_df: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    """Summarize metrics per (model, gap size) with mean and P05/P95 over runs."""
    rows: list[dict[str, Any]] = []
    for (model, gap), group in results_mc_df.groupby(["Modelo", "Gap_Size"], sort=True):
        row: dict[str, Any] = {"Modelo": str(model), "Gap_Size": int(gap)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_Mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_P05"] = float(values.quantile(0.05)) if len(values) else float("nan")
            row[f"{metric}_P95"] = float(values.quantile(0.95)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _save_metric_gap_plot(
    results_mc_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path | None:
    """Plot each metric as one line per model across gap sizes (P05-P95 band).

    Persists both the aggregated table (`metrics_by_gap.csv`) and the figure
    (`metrics_by_gap.png`). Returns the image path, or None if there is nothing
    to plot.
    """
    if results_mc_df.empty:
        return None
    if not {"Modelo", "Gap_Size"}.issubset(results_mc_df.columns):
        return None

    metrics = [m for m in ("MAE", "RMSE", "MASE") if m in results_mc_df.columns]
    if not metrics:
        return None

    agg_df = _aggregate_metrics_by_gap(results_mc_df, metrics)
    if agg_df.empty:
        return None
    agg_df = agg_df.sort_values(["Modelo", "Gap_Size"]).reset_index(drop=True)
    agg_df.to_csv(output_dir / "metrics_by_gap.csv", index=False)

    gap_sizes = sorted(int(g) for g in agg_df["Gap_Size"].unique())
    models = sorted(str(m) for m in agg_df["Modelo"].unique())
    color_map = _model_color_map(models)
    x_positions = list(range(len(gap_sizes)))
    gap_to_x = {gap: pos for pos, gap in enumerate(gap_sizes)}

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(6.0 * len(metrics), 4.8),
        facecolor=FIGURE_FACE,
        squeeze=False,
    )
    axes_row = axes[0]

    for ax, metric in zip(axes_row, metrics):
        for model in models:
            model_df = agg_df[agg_df["Modelo"] == model].set_index("Gap_Size")
            xs, means, lows, highs = [], [], [], []
            for gap in gap_sizes:
                if gap not in model_df.index:
                    continue
                xs.append(gap_to_x[gap])
                means.append(model_df.loc[gap, f"{metric}_Mean"])
                lows.append(model_df.loc[gap, f"{metric}_P05"])
                highs.append(model_df.loc[gap, f"{metric}_P95"])
            if not xs:
                continue
            color = color_map[model]
            ax.plot(xs, means, marker="o", markersize=4.0, lw=1.6, color=color, label=model)
            ax.fill_between(xs, lows, highs, color=color, alpha=BAND_ALPHA, linewidth=0)

        _style_metric_axis(ax)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tamano de hueco")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(g) for g in gap_sizes])

    axes_row[0].set_ylabel("Valor de la metrica (media, banda P05-P95)")

    handles, labels = axes_row[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            frameon=True,
            framealpha=0.95,
            edgecolor="#d0d0d0",
            fontsize=9,
        )

    fig.suptitle(
        "Metricas de imputacion por tamano de hueco",
        x=0.5,
        y=0.99,
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    image_path = output_dir / "metrics_by_gap.png"
    fig.savefig(image_path, dpi=150, facecolor=FIGURE_FACE)
    plt.close(fig)
    return image_path


def _repo_root() -> Path:
    """Return the repository root (two levels above this module)."""
    return Path(__file__).resolve().parents[2]


def _sanitize_filename(text: str) -> str:
    """Reduce free-form series names to a safe filesystem identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return sanitized.strip("._") or "series"


def _build_output_dir() -> Path:
    """Create the timestamped output directory for one Monte Carlo run.

    The timestamp has 1-second resolution, so a numeric suffix disambiguates
    runs started within the same second instead of crashing on ``mkdir``.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = _repo_root() / "reports" / "benchmark"
    output_dir = base_dir / f"montecarlo_{stamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = base_dir / f"montecarlo_{stamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


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


def _save_plot_images(
    plot_store: dict[int, dict[str, Any]],
    *,
    output_dir: Path,
) -> pd.DataFrame:
    """Render one PNG per (gap size, series) with real values vs. model predictions.

    Writes the images under ``<output_dir>/plots/gap_<n>/`` plus a
    ``plot_images.csv`` manifest, which is also returned as a dataframe.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            if len(actual) > 0:
                ax.plot(actual.index, actual.values, color="black", alpha=0.45, lw=1.4, label="Real")

            mask_index = pd.DatetimeIndex([])
            naive_mase = _to_pd_series(series_payload.get("naive_mase", pd.Series(dtype=float)))
            if len(naive_mase) > 0:
                mask_index = pd.DatetimeIndex(naive_mase.index)

            for pred in preds_by_model.values():
                pred_series = _to_pd_series(pred)
                if len(pred_series) > 0:
                    mask_index = mask_index.union(pd.DatetimeIndex(pred_series.index))

            gap_real = actual.reindex(mask_index).dropna()
            if len(gap_real) > 0:
                ax.plot(
                    gap_real.index,
                    gap_real.values,
                    color="red",
                    linestyle="None",
                    marker="o",
                    markersize=4.0,
                    label="Hueco real",
                )

            for model_name, pred in sorted(preds_by_model.items()):
                pred_series = _to_pd_series(pred).dropna()
                if len(pred_series) == 0:
                    continue
                ax.plot(
                    pred_series.index,
                    pred_series.values,
                    linestyle="--",
                    marker="o",
                    markersize=3.0,
                    lw=1.1,
                    label=str(model_name),
                )

            ax.set_title(f"Serie: {series_name} | Gap={int(gap_size)}")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("NO2")
            if ax.has_data():
                ax.legend(loc="best")
            fig.autofmt_xdate()
            fig.tight_layout()

            image_path = gap_dir / f"{_sanitize_filename(str(series_name))}.png"
            fig.savefig(image_path, dpi=150)
            plt.close(fig)

            rows.append(
                {
                    "gap_size": int(gap_size),
                    "series_name": str(series_name),
                    "model_count": len(preds_by_model),
                    "image_path": str(image_path.relative_to(output_dir)),
                }
            )

    manifest_df = pd.DataFrame(
        rows,
        columns=["gap_size", "series_name", "model_count", "image_path"],
    )
    manifest_df.to_csv(output_dir / "plot_images.csv", index=False)
    return manifest_df


def run_benchmark_from_config() -> dict[str, Any]:
    """Run the configured Monte Carlo benchmark and persist CSV/image artifacts."""
    output_dir = _build_output_dir()

    results_mc_df, summary_mc_df, ranking_by_seed_df = (
        run_imputation_benchmark_parallel_montecarlo()
    )
    results_mc_df.to_csv(output_dir / "results_mc.csv", index=False)
    summary_mc_df.to_csv(output_dir / "summary_mc.csv", index=False)
    ranking_by_seed_df.to_csv(output_dir / "ranking_by_seed.csv", index=False)

    metric_gap_plot_path = _save_metric_gap_plot(results_mc_df, output_dir=output_dir)

    plot_seed = cfg_get_int("benchmark", "random_seed", 42)
    _, _, plot_store = run_imputation_benchmark_parallel(random_seed=plot_seed)
    plot_manifest_df = _save_plot_images(plot_store, output_dir=output_dir)

    return {
        "output_dir": output_dir,
        "results_mc_df": results_mc_df,
        "summary_mc_df": summary_mc_df,
        "ranking_by_seed_df": ranking_by_seed_df,
        "plot_manifest_df": plot_manifest_df,
        "metric_gap_plot_path": metric_gap_plot_path,
    }


def main() -> None:
    """Execute the Monte Carlo benchmark and print saved artifact locations."""
    artifacts = run_benchmark_from_config()
    summary_df = artifacts["summary_mc_df"]
    output_dir = artifacts["output_dir"]

    if summary_df.empty:
        print("[info] Benchmark completed with no Monte Carlo summary rows.")
    else:
        print("[info] Monte Carlo benchmark summary")
        print(summary_df.to_string(index=False))

    print(f"[info] Saved benchmark artifacts under {output_dir}")
    print(f"[info] Results CSV: {output_dir / 'results_mc.csv'}")
    print(f"[info] Summary CSV: {output_dir / 'summary_mc.csv'}")
    print(f"[info] Ranking CSV: {output_dir / 'ranking_by_seed.csv'}")
    if artifacts.get("metric_gap_plot_path") is not None:
        print(f"[info] Metrics-by-gap plot: {artifacts['metric_gap_plot_path']}")
    print(f"[info] Plot manifest CSV: {output_dir / 'plot_images.csv'}")


if __name__ == "__main__":
    main()
