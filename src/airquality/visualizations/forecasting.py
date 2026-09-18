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
- ``train_time.png`` — model-level training cost across all branches (median and
  mean, using ``train_seconds``); foundation models use 0 s and
  ``training_summary.csv`` contains the numeric summary.
- ``inference_time.png`` — per-arm holdout inference time across series (dots)
  with the mean labeled, one panel per forecast model.
- ``train_time_vs_{rmsse,mase,relmae,relrmse}.png`` — one model-level training
  cost/accuracy point across all branches.
- ``inference_time_vs_{rmsse,mase,relmae,relrmse}.png`` — cost/accuracy scatter
  per (arm, model), so the bottom-left region is fast and precise.
- ``foundation_preprocessing_recovery_{mase,rmsse}.png`` — paired error recovered
  after preprocessing synthetic context anomalies, by model, strategy and type.
- ``executive_branch_summary_{relmae,relrmse}.png`` — station-balanced branch
  improvement vs ``raw`` with interquartile bands and practical win/tie/loss rates.
- ``raw_model_ranking_{mase,rmsse}.png`` — all models ranked on the common ``raw``
  view; foundation training cost is treated as zero.
- ``executive_model_arm_{mase,rmsse}.png`` — compact model×arm mean-error matrix.
- ``executive_imputation_{mase,rmsse}.png`` — aggregated with/without-imputation
  comparison by detection strategy.
- ``executive_cost_accuracy_{mase,rmsse}.png`` — predictor training/inference cost
  vs error, with the Pareto frontier highlighted.
- ``foundation_anomaly_robustness_{mase,rmsse}.png`` — per-run synthetic anomaly
  damage before and after foundation-context preprocessing.

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
from airquality.forecasting.pipeline import METRIC_COLS, RAW_ARM

MUTED_TEXT = "#6d6258"

#: Fixed color per strategy family (validated for CVD separation + contrast on
#: the cream surface). ``raw`` is neutral on purpose — it is the reference.
ARM_FAMILY_COLORS = {
    RAW_ARM: "#6d6258",
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

MODEL_MODE_COLORS = {
    "trained": "#3d7ab5",
    "local": "#cf6f1e",
    "foundation": "#5b8c5a",
}
MODEL_MODE_MARKERS = {"trained": "o", "local": "s", "foundation": "*"}
PRACTICAL_EQUIVALENCE_PCT = 1.0

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


def _format_seconds(value: float) -> str:
    """Format short and long timings without hiding sub-second differences."""
    return f"{value:.2f}s" if abs(value) < 1.0 else f"{value:.1f}s"


def _successful_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return successful rows while remaining compatible with small plot fixtures."""
    if "status" not in results_df.columns:
        return results_df.copy()
    return results_df.loc[results_df["status"].astype(str) == "ok"].copy()


def _foundation_mask(results_df: pd.DataFrame) -> pd.Series:
    """Identify zero-shot foundation rows, or none in legacy/minimal fixtures."""
    if "model_mode" not in results_df.columns:
        return pd.Series(False, index=results_df.index)
    return results_df["model_mode"].astype(str).eq("foundation")


def training_time_summary(
    results_df: pd.DataFrame,
    metric: str | None = None,
) -> pd.DataFrame:
    """Summarize fit time by model, balancing the configured branches.

    Each branch contributes one equally weighted summary per model. This avoids
    a branch with more station rows dominating the model-level cost estimate.
    Foundation models are included with zero training cost because they are
    zero-shot.
    When ``metric`` is supplied, its branch-balanced means and medians are added
    for cost-versus-accuracy plots.
    """
    summary_columns = [
        "model",
        "n_branches",
        "n_runs",
        "train_seconds_mean",
        "train_seconds_median",
        "train_seconds_p90",
    ]
    if metric is not None:
        summary_columns.extend([f"{metric}_mean", f"{metric}_median"])

    required = {"model", "train_seconds"}
    if metric is not None:
        required.add(metric)
    if not required <= set(results_df.columns):
        return pd.DataFrame(columns=summary_columns)

    frame = _successful_results(results_df).copy()
    if frame.empty:
        return pd.DataFrame(columns=summary_columns)
    if "model_mode" not in frame.columns:
        frame["model_mode"] = "trained"
    if "arm" not in frame.columns:
        frame["arm"] = "all"

    frame["train_seconds"] = pd.to_numeric(frame["train_seconds"], errors="coerce")
    frame.loc[_foundation_mask(frame), "train_seconds"] = 0.0
    numeric_columns = ["train_seconds"]
    if metric is not None:
        numeric_columns.append(metric)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric_columns)
    if frame.empty:
        return pd.DataFrame(columns=summary_columns)

    branch_columns = ["model", "arm"]
    branch_aggregations = {
        "n_runs": ("train_seconds", "size"),
        "train_seconds_mean": ("train_seconds", "mean"),
        "train_seconds_median": ("train_seconds", "median"),
        "train_seconds_p90": ("train_seconds", lambda values: values.quantile(0.9)),
    }
    if metric is not None:
        branch_aggregations[f"{metric}_mean"] = (metric, "mean")
        branch_aggregations[f"{metric}_median"] = (metric, "median")

    by_branch = frame.groupby(branch_columns, sort=False).agg(**branch_aggregations).reset_index()
    model_aggregations = {
        "n_branches": ("arm", "nunique"),
        "n_runs": ("n_runs", "sum"),
        "train_seconds_mean": ("train_seconds_mean", "mean"),
        "train_seconds_median": ("train_seconds_median", "median"),
        "train_seconds_p90": ("train_seconds_p90", "mean"),
    }
    if metric is not None:
        model_aggregations[f"{metric}_mean"] = (f"{metric}_mean", "mean")
        model_aggregations[f"{metric}_median"] = (f"{metric}_median", "median")

    summary = by_branch.groupby("model", sort=False).agg(**model_aggregations).reset_index()
    return summary.reindex(columns=summary_columns)


def _common_nonfoundation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose arms share the primary raw targets and can be compared directly."""
    frame = _successful_results(results_df)
    return frame.loc[~_foundation_mask(frame)].copy()


def _station_arm_scores(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Balanced station×arm errors after equally averaging non-foundation models."""
    frame = _common_nonfoundation_results(results_df)
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    arms = list(dict.fromkeys(frame["arm"].astype(str)))
    models = list(dict.fromkeys(frame["model"].astype(str)))
    stations = list(dict.fromkeys(frame["series"].astype(str)))
    if RAW_ARM not in arms or not models or not stations:
        return pd.DataFrame()
    panel = frame.pivot_table(
        index=["series", "model"], columns="arm", values=metric, aggfunc="first", sort=False
    ).reindex(
        pd.MultiIndex.from_product([stations, models], names=["series", "model"]),
        columns=arms,
    )
    finite = np.isfinite(panel.to_numpy(dtype=float))
    complete = pd.Series(finite.all(axis=1), index=panel.index).groupby(level="series").all()
    kept = complete[complete].index
    if len(kept) == 0:
        return pd.DataFrame()
    panel = panel.loc[panel.index.get_level_values("series").isin(kept)]
    return panel.groupby(level="series", sort=False).mean()


def _quartile_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the interquartile spread for a station-level distribution."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(values, (0.25, 0.75))
    return float(low), float(high)


def _pareto_mask(cost: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Return points not dominated jointly on lower cost and lower error."""
    cost = np.asarray(cost, dtype=float)
    error = np.asarray(error, dtype=float)
    keep = np.zeros(len(cost), dtype=bool)
    for index, (this_cost, this_error) in enumerate(zip(cost, error, strict=True)):
        valid = np.isfinite(cost) & np.isfinite(error)
        dominates = valid & (cost <= this_cost) & (error <= this_error)
        strictly = (cost < this_cost) | (error < this_error)
        keep[index] = not bool(np.any(dominates & strictly))
    return keep


def save_executive_branch_summary_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    metric: str = "mase",
) -> bool:
    """Summarize branch improvement and practical outcomes after averaging models."""
    scores = _station_arm_scores(results_df, metric)
    if scores.empty or RAW_ARM not in scores:
        return False
    raw = scores[RAW_ARM].to_numpy(dtype=float)
    rows = []
    for index, arm in enumerate(column for column in scores.columns if column != RAW_ARM):
        values = scores[arm].to_numpy(dtype=float)
        valid = np.isfinite(raw) & np.isfinite(values) & (raw != 0.0)
        improvement = 100.0 * (raw[valid] - values[valid]) / raw[valid]
        if improvement.size == 0:
            continue
        low, high = _quartile_interval(improvement)
        rows.append(
            {
                "arm": arm,
                "mean": float(np.mean(improvement)),
                "low": low,
                "high": high,
                "better": int(np.sum(improvement > PRACTICAL_EQUIVALENCE_PCT)),
                "tie": int(np.sum(np.abs(improvement) <= PRACTICAL_EQUIVALENCE_PCT)),
                "worse": int(np.sum(improvement < -PRACTICAL_EQUIVALENCE_PCT)),
                "n": len(improvement),
            }
        )
    if not rows:
        return False
    summary = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    colors = family_colors(summary["arm"].astype(str).tolist())
    metric_name = {
        "relmae": "relMAE",
        "relrmse": "relRMSE",
    }.get(metric, _metric_label(metric))
    metric_definition = {
        "relmae": "relMAE = MAE de la rama / MAE de raw",
        "relrmse": "relRMSE = RMSE de la rama / RMSE de raw",
    }.get(metric, f"Métrica = {_metric_label(metric)}")

    figure, (effect_axis, outcome_axis) = plt.subplots(
        1,
        2,
        figsize=(13.5, 3.4 + 0.55 * len(summary)),
        facecolor=FIGURE_FACE,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    top = _add_header(
        figure,
        f"Resumen ejecutivo de ramas — {metric_name}",
        f"{metric_definition}. Positivo = mejora frente a raw.\n"
        f"Banda = intervalo intercuartílico por estación; empate práctico = ±{PRACTICAL_EQUIVALENCE_PCT:g}%.",
    )
    y = np.arange(len(summary))

    style_axis(effect_axis)
    effect_axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    effect_axis.grid(False, axis="y")
    effect_axis.axvline(0.0, color=MUTED_TEXT, linestyle="--", linewidth=1.1)
    for position, row in summary.iterrows():
        color = colors[_family(str(row["arm"]))]
        center = float(row["mean"])
        effect_axis.errorbar(
            center,
            position,
            xerr=[
                [max(0.0, center - float(row["low"]))],
                [max(0.0, float(row["high"]) - center)],
            ],
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            capsize=3,
            linewidth=1.4,
            zorder=3,
        )
        effect_axis.annotate(
            f"{center:+.1f}%",
            (center, position),
            textcoords="offset points",
            xytext=(7 if center >= 0 else -7, 0),
            ha="left" if center >= 0 else "right",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
        )
    effect_axis.set_yticks(y, summary["arm"])
    effect_axis.invert_yaxis()
    effect_axis.set_xlabel("mejora media frente a raw (%)")
    effect_axis.set_title("Efecto medio", loc="left", fontsize=11)

    style_axis(outcome_axis)
    outcome_axis.grid(False)
    left = np.zeros(len(summary))
    outcome_specs = (
        ("better", "mejora >1%", "#3d7ab5"),
        ("tie", "equivalente", "#d7cbbb"),
        ("worse", "empeora >1%", "#d6453c"),
    )
    for column, label, color in outcome_specs:
        width = 100.0 * summary[column].to_numpy(dtype=float) / summary["n"].to_numpy(dtype=float)
        bars = outcome_axis.barh(
            y, width, left=left, color=color, edgecolor=FIGURE_FACE, linewidth=1.0, label=label
        )
        for bar, value, count in zip(bars, width, summary[column], strict=True):
            if value >= 5.0:
                outcome_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%\n({int(count)}/{int(summary['n'].iloc[0])})",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if column != "tie" else TEXT_COLOR,
                )
        left += width
    outcome_axis.set_yticks(y, [])
    outcome_axis.invert_yaxis()
    outcome_axis.set_xlim(0.0, 100.0)
    outcome_axis.set_xticks([])
    outcome_axis.set_xlabel("")
    outcome_axis.spines["bottom"].set_visible(False)
    outcome_axis.set_title("Distribución entre estaciones", loc="left", fontsize=11)
    outcome_axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=8)

    figure.tight_layout(rect=(0, 0.12, 1, top))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def save_raw_model_ranking_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    metric: str = "mase",
) -> bool:
    """Rank every model on raw only, including zero-training-cost foundations."""
    frame = _successful_results(results_df)
    if metric not in frame.columns:
        return False
    frame = frame.loc[frame["arm"].astype(str).eq(RAW_ARM)].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[np.isfinite(frame[metric])]
    if frame.empty:
        return False
    if "model_mode" not in frame:
        frame["model_mode"] = "trained"
    summaries = []
    for index, (model, group) in enumerate(frame.groupby("model", sort=False)):
        values = group[metric].to_numpy(dtype=float)
        low, high = _quartile_interval(values)
        mode = str(group["model_mode"].iloc[0])
        measured_train = float(pd.to_numeric(group["train_seconds"], errors="coerce").mean())
        inference = float(pd.to_numeric(group["inference_seconds"], errors="coerce").mean())
        summaries.append(
            {
                "model": str(model),
                "mode": mode,
                "mean": float(np.mean(values)),
                "low": low,
                "high": high,
                "train": 0.0 if mode == "foundation" else measured_train,
                "inference": inference,
                "values": values,
            }
        )
    summary = sorted(summaries, key=lambda row: row["mean"])
    figure, axis = plt.subplots(
        figsize=(11.5, 3.0 + 0.52 * len(summary)), facecolor=FIGURE_FACE
    )
    top = _add_header(
        figure,
        f"Ranking de modelos usando solo raw — {_metric_label(metric)}",
        "Puntos pequeños = estaciones; punto grande = media y banda intercuartílica. Menor = mejor.",
    )
    rng = np.random.default_rng(13)
    for position, row in enumerate(summary):
        mode = str(row["mode"])
        color = MODEL_MODE_COLORS.get(mode, MUTED_TEXT)
        values = np.asarray(row["values"], dtype=float)
        axis.scatter(
            values,
            position + rng.normal(0.0, 0.055, len(values)),
            s=22,
            color=color,
            alpha=0.48,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        center = float(row["mean"])
        axis.errorbar(
            center,
            position,
            xerr=[
                [max(0.0, center - float(row["low"]))],
                [max(0.0, float(row["high"]) - center)],
            ],
            fmt=MODEL_MODE_MARKERS.get(mode, "o"),
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8 if mode == "foundation" else 6,
            capsize=3,
            linewidth=1.4,
            zorder=3,
        )
        axis.text(
            1.01,
            position,
            f"train {row['train']:.2f}s · infer {row['inference']:.2f}s",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
            clip_on=False,
        )
    style_axis(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    axis.set_yticks(np.arange(len(summary)), [row["model"] for row in summary])
    axis.invert_yaxis()
    axis.set_xlabel(_metric_label(metric))
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker=MODEL_MODE_MARKERS[mode], linestyle="none",
                markerfacecolor=color, markeredgecolor="white", markersize=8, label=mode,
            )
            for mode, color in MODEL_MODE_COLORS.items()
            if any(row["mode"] == mode for row in summary)
        ],
        loc="lower right",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 0.79, top))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def save_executive_model_arm_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    metric: str = "mase",
) -> bool:
    """Render a compact absolute-error matrix."""
    frame = _successful_results(results_df)
    if frame.empty or metric not in frame.columns:
        return False
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    if "model_mode" not in frame:
        frame["model_mode"] = "trained"
    arms = list(dict.fromkeys(frame["arm"].astype(str)))
    if RAW_ARM not in arms:
        return False
    means = frame.pivot_table(
        index="model", columns="arm", values=metric, aggfunc="mean", sort=False
    ).reindex(columns=arms)
    if means.empty:
        return False
    model_order = means[RAW_ARM].sort_values().index
    means = means.reindex(model_order)
    values = means.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    figure, axis = plt.subplots(
        figsize=(max(11.0, 1.25 * len(arms) + 3.0), 3.1 + 0.52 * len(means)),
        facecolor=FIGURE_FACE,
    )
    top = _add_header(
        figure,
        f"Mapa compacto modelo × rama — {_metric_label(metric)}",
        "Media sobre estaciones; menor = mejor.",
    )
    image = axis.imshow(values, aspect="auto", cmap="RdYlBu_r", vmin=float(finite.min()), vmax=float(finite.max()))
    for row, model in enumerate(means.index):
        for col in range(len(arms)):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=TEXT_COLOR)
    axis.set_xticks(range(len(arms)), arms, rotation=30, ha="right", fontsize=8.5)
    axis.set_yticks(range(len(means)), means.index, fontsize=9)
    axis.set_xlabel("Rama")
    axis.set_ylabel("Modelo")
    style_axis(axis)
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label(f"{_metric_label(metric)} medio")
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def save_executive_imputation_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    metric: str = "mase",
) -> bool:
    """Compare mean imputed and non-imputed errors in four readable dumbbells."""
    frame = _common_nonfoundation_results(results_df)
    if frame.empty or metric not in frame.columns:
        return False
    frame = frame.loc[frame["strategy"].astype(str).ne("none")].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    is_imputed = (
        frame["imputed"]
        if pd.api.types.is_bool_dtype(frame["imputed"])
        else frame["imputed"].astype(str).str.lower().eq("true")
    )
    frame["variant"] = np.where(is_imputed, "impute", "noimpute")
    pairs = frame.pivot_table(
        index=["series", "model", "strategy"],
        columns="variant",
        values=metric,
        aggfunc="first",
        sort=False,
    ).dropna(subset=["impute", "noimpute"])
    if pairs.empty:
        return False
    station = pairs.groupby(level=["series", "strategy"], sort=False).mean()
    rows = []
    for index, strategy in enumerate(dict.fromkeys(frame["strategy"].astype(str))):
        try:
            values = station.xs(strategy, level="strategy")
        except KeyError:
            continue
        noimpute = values["noimpute"].to_numpy(dtype=float)
        impute = values["impute"].to_numpy(dtype=float)
        no_low, no_high = _quartile_interval(noimpute)
        im_low, im_high = _quartile_interval(impute)
        valid = noimpute != 0.0
        gain = 100.0 * (noimpute[valid] - impute[valid]) / noimpute[valid]
        rows.append(
            {
                "strategy": strategy,
                "noimpute": float(noimpute.mean()),
                "impute": float(impute.mean()),
                "no_low": no_low,
                "no_high": no_high,
                "im_low": im_low,
                "im_high": im_high,
                "gain": float(gain.mean()),
            }
        )
    if not rows:
        return False
    rows.sort(key=lambda row: row["gain"], reverse=True)
    colors = family_colors([row["strategy"] for row in rows])
    figure, axis = plt.subplots(figsize=(10.2, 3.3 + 0.72 * len(rows)), facecolor=FIGURE_FACE)
    top = _add_header(
        figure,
        f"Efecto agregado de la imputación — {_metric_label(metric)}",
        "La flecha apunta de sin imputación a con imputación; hacia la izquierda significa que imputar mejora.",
    )
    for position, row in enumerate(rows):
        color = colors[_family(str(row["strategy"]))]
        axis.annotate(
            "",
            xy=(row["impute"], position),
            xytext=(row["noimpute"], position),
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.8},
            zorder=2,
        )
        for variant, filled in (("noimpute", False), ("impute", True)):
            center = float(row[variant])
            low = float(row[f"{'no' if variant == 'noimpute' else 'im'}_low"])
            high = float(row[f"{'no' if variant == 'noimpute' else 'im'}_high"])
            axis.errorbar(
                center,
                position,
                xerr=[[max(0.0, center - low)], [max(0.0, high - center)]],
                fmt="o",
                color=color,
                markerfacecolor=color if filled else FIGURE_FACE,
                markeredgecolor=color,
                capsize=3,
                linewidth=1.2,
                zorder=3,
            )
        axis.text(
            1.01,
            position,
            f"Δ {row['gain']:+.1f}%",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
            clip_on=False,
        )
    style_axis(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    axis.set_yticks(np.arange(len(rows)), [row["strategy"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel(f"{_metric_label(metric)} medio (menor = mejor)")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=FIGURE_FACE,
                   markeredgecolor=TEXT_COLOR, label="sin imputación"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=TEXT_COLOR,
                   markeredgecolor=TEXT_COLOR, label="con imputación"),
        ],
        loc="lower right",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 0.88, top))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


def save_executive_cost_accuracy_plot(
    output_path: Path,
    results_df: pd.DataFrame,
    metric: str = "mase",
) -> bool:
    """Compare all unique predictor choices on training/inference Pareto fronts."""
    frame = _successful_results(results_df)
    required = {metric, "train_seconds", "inference_seconds", "arm", "model"}
    if frame.empty or not required <= set(frame.columns):
        return False
    if "model_mode" not in frame:
        frame["model_mode"] = "trained"
    foundation = _foundation_mask(frame)
    frame = frame.loc[
        ~foundation | frame["arm"].astype(str).eq(RAW_ARM)
    ].copy()
    for column in (metric, "train_seconds", "inference_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    aggregate = (
        frame.groupby(["model", "arm", "model_mode"], sort=False)[
            [metric, "train_seconds", "inference_seconds"]
        ]
        .mean()
        .reset_index()
        .dropna()
        .reset_index(drop=True)
    )
    if aggregate.empty:
        return False
    aggregate["effective_train_seconds"] = aggregate["train_seconds"]
    aggregate.loc[
        aggregate["model_mode"].astype(str).eq("foundation"), "effective_train_seconds"
    ] = 0.0
    colors = family_colors(aggregate["arm"].astype(str).tolist())
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 7.0), facecolor=FIGURE_FACE)
    top = _add_header(
        figure,
        f"Coste del predictor frente a precisión — {_metric_label(metric)}",
        "Abajo-izquierda = mejor. Contorno = frontera de Pareto.",
    )
    foundation_order = {
        model: position
        for position, model in enumerate(
            aggregate.loc[
                aggregate["model_mode"].astype(str).eq("foundation")
            ].sort_values(metric)["model"]
        )
    }
    jobs = (
        (axes[0], "effective_train_seconds", "Entrenamiento efectivo", True),
        (axes[1], "inference_seconds", "Inferencia medida", False),
    )
    for axis, cost_column, title, includes_zero in jobs:
        style_axis(axis)
        axis.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.55)
        cost = aggregate[cost_column].to_numpy(dtype=float)
        error = aggregate[metric].to_numpy(dtype=float)
        frontier = _pareto_mask(cost, error)
        for index, row in aggregate.iterrows():
            mode = str(row["model_mode"])
            color = colors[_family(str(row["arm"]))]
            is_frontier = bool(frontier[index])
            axis.scatter(
                row[cost_column],
                row[metric],
                s=115 if mode == "foundation" else 54,
                marker=MODEL_MODE_MARKERS.get(mode, "o"),
                facecolor=color,
                edgecolor=TEXT_COLOR if is_frontier else "white",
                linewidths=2.0 if is_frontier else 0.6,
                alpha=0.95 if is_frontier or mode == "foundation" else 0.58,
                zorder=4 if is_frontier else 3,
            )
            if is_frontier or mode == "foundation":
                if mode == "foundation" and includes_zero:
                    rank = foundation_order.get(str(row["model"]), 0)
                    label_offset = (8, -10 if rank == 0 else 8)
                else:
                    label_offset = (6, 6)
                axis.annotate(
                    str(row["model"]) if mode == "foundation" else f"{row['model']}\n{row['arm']}",
                    (row[cost_column], row[metric]),
                    textcoords="offset points",
                    xytext=label_offset,
                    fontsize=7.5,
                    color=TEXT_COLOR,
                )
        if includes_zero:
            axis.set_xscale("symlog", linthresh=0.01, linscale=0.8)
        elif np.all(cost > 0.0):
            axis.set_xscale("log")
        axis.set_xlabel("tiempo (s)")
        axis.set_ylabel(_metric_label(metric))
        axis.set_title(title, loc="left", fontsize=11)
    mode_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor=MUTED_TEXT,
               markeredgecolor="white", markersize=8, label=mode)
        for mode, marker in MODEL_MODE_MARKERS.items()
        if mode in set(aggregate["model_mode"].astype(str))
    ]
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor="white", markersize=8, label=family)
        for family, color in colors.items()
        if family in {_family(arm) for arm in aggregate["arm"].astype(str)}
    ]
    figure.legend(
        handles=mode_handles + family_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=5,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.11, 1, top))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


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
    """Plot one model-level training-time summary across all benchmark arms."""
    summary = training_time_summary(results_df)
    if summary.empty:
        return False

    summary = summary.sort_values("train_seconds_median", ascending=True).reset_index(drop=True)
    positions = np.arange(len(summary))
    medians = summary["train_seconds_median"].to_numpy(dtype=float)
    means = summary["train_seconds_mean"].to_numpy(dtype=float)

    figure, axis = plt.subplots(
        figsize=(10.0, max(4.0, 2.8 + 0.42 * len(summary))),
        facecolor=FIGURE_FACE,
    )
    top = _add_header(
        figure,
        "Coste de entrenamiento por modelo",
        "Circulo = mediana; marca vertical = media.",
    )
    style_axis(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linestyle="--", alpha=0.65)
    axis.grid(False, axis="y")
    axis.scatter(
        medians,
        positions,
        s=88,
        color="#3d7ab5",
        edgecolor="white",
        linewidths=0.8,
        label="mediana",
        zorder=3,
    )
    axis.scatter(
        means,
        positions,
        marker="|",
        s=520,
        color="#27313a",
        linewidths=2.2,
        label="media",
        zorder=4,
    )
    for position, median, n_runs in zip(
        positions, medians, summary["n_runs"], strict=True
    ):
        axis.annotate(
            f"{_format_seconds(median)} (n={int(n_runs)})",
            (median, position),
            textcoords="offset points",
            xytext=(7, 0),
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
        )
    axis.set_yticks(positions, summary["model"])
    axis.invert_yaxis()
    axis.set_xlabel("tiempo de entrenamiento (s)")
    if np.all(np.r_[medians, means] >= 0):
        axis.set_xscale("symlog", linthresh=1.0)
        axis.set_xlim(left=0)
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


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
        f"X = {xlabel}; Y = error (abajo-izq = mejor).\n"
        "Color = estrategia; marcador = modelo; relleno/hueco = con/sin imputacion.",
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
    """Plot one model-level training-cost/accuracy point per model."""
    summary = training_time_summary(results_df, metric=metric)
    if summary.empty:
        return False

    plot_summary = summary.sort_values(f"{metric}_mean", ascending=False).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(11.0, 7.2), facecolor=FIGURE_FACE)
    top = _add_header(
        figure,
        f"Coste frente a precision — {_metric_label(metric)}",
        "Un punto por modelo; X = mediana del tiempo de entrenamiento, Y = media del error.",
    )
    style_axis(axis)
    axis.grid(True, axis="both", color=GRID_COLOR, linestyle="--", alpha=0.55)

    medians = plot_summary["train_seconds_median"].to_numpy(dtype=float)
    metric_values = plot_summary[f"{metric}_mean"].to_numpy(dtype=float)
    axis.scatter(
        medians,
        metric_values,
        s=92,
        color="#3d7ab5",
        edgecolor="white",
        linewidths=0.8,
        zorder=3,
    )

    time_max = float(np.nanmax(medians))
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_xlim(-1.0, max(1.0, time_max * 1.35))
    time_ticks = [0.0, 1.0, 10.0, 100.0]
    time_ticks = [tick for tick in time_ticks if tick <= max(1.0, time_max * 1.35)]
    axis.set_xticks(time_ticks)
    axis.set_xticklabels(["0" if tick == 0 else f"{tick:g}s" for tick in time_ticks])
    axis.set_xlabel("mediana del tiempo de entrenamiento (s)")
    axis.set_ylabel(f"{_metric_label(metric)} medio")

    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=top)
    # Repel labels only when necessary, keeping them close to their points.
    figure.canvas.draw()
    axis_bbox = axis.get_window_extent()
    point_pixels = axis.transData.transform(np.column_stack((medians, metric_values)))
    order = np.argsort(point_pixels[:, 1])[::-1]
    min_gap = min(18.0, (axis_bbox.height - 24.0) / max(1, len(plot_summary) - 1))
    label_y_pixels = np.array([point_pixels[index, 1] for index in order], dtype=float)
    for index in range(1, len(label_y_pixels)):
        label_y_pixels[index] = min(
            label_y_pixels[index], label_y_pixels[index - 1] - min_gap
        )
    if label_y_pixels[-1] < axis_bbox.y0 + 12.0:
        label_y_pixels += axis_bbox.y0 + 12.0 - label_y_pixels[-1]
    if label_y_pixels[0] > axis_bbox.y1 - 12.0:
        label_y_pixels -= label_y_pixels[0] - (axis_bbox.y1 - 12.0)

    offsets = [(8.0, 0.0)] * len(plot_summary)
    for rank, index in enumerate(order):
        point_y = point_pixels[index, 1]
        offsets[index] = (8.0, (label_y_pixels[rank] - point_y) * 72.0 / figure.dpi)
    annotations = []
    for median, value, model, offset in zip(
        medians, metric_values, plot_summary["model"], offsets, strict=True
    ):
        annotations.append(
            axis.annotate(
                str(model),
                (median, value),
                xytext=offset,
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": MUTED_TEXT,
                    "linewidth": 0.7,
                    "alpha": 0.75,
                    "shrinkA": 5,
                    "shrinkB": 2,
                },
            )
        )

    figure.canvas.draw()
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
    plt.close(figure)
    return True


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
        f"Recuperacion del daño sintetico — {_metric_label(metric)}",
        "Positivo: el preprocesamiento reduce el error frente al contexto corrompido.",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Error recuperado")
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def save_foundation_anomaly_robustness_plot(
    output_path: Path,
    summary_df: pd.DataFrame,
    metric: str = "mase",
) -> bool:
    """Show per-case foundation damage before and after synthetic-context cleaning."""
    required = {
        "model",
        "strategy",
        "anomaly_type",
        f"{metric}_clean",
        f"{metric}_corrupted",
        f"{metric}_processed",
    }
    if summary_df.empty or not required <= set(summary_df.columns):
        return False
    data = summary_df.copy()
    for suffix in ("clean", "corrupted", "processed"):
        column = f"{metric}_{suffix}"
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["damage"] = data[f"{metric}_corrupted"] - data[f"{metric}_clean"]
    data["residual"] = data[f"{metric}_processed"] - data[f"{metric}_clean"]
    data["panel"] = data["model"].astype(str) + " / " + data["strategy"].astype(str)
    panels = list(dict.fromkeys(data["panel"]))
    anomaly_types = list(dict.fromkeys(data["anomaly_type"].astype(str)))
    finite = np.concatenate([_finite(data["damage"]), _finite(data["residual"])])
    if not panels or not anomaly_types or finite.size == 0:
        return False

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(1.2 + 5.0 * len(panels), 6.7),
        facecolor=FIGURE_FACE,
        sharey=True,
        squeeze=False,
    )
    top = _add_header(
        figure,
        f"Robustez foundation ante anomalías sintéticas — {_metric_label(metric)}",
        "Cada punto = una ejecución/estación. Naranja: contexto corrupto − limpio; azul: tras limpiar − limpio.\n"
        "Positivo = la anomalía empeora el forecast; cero = mismo error que con contexto limpio. Rombo = mediana.",
    )
    rng = np.random.default_rng(13)
    for axis, panel in zip(axes[0], panels, strict=True):
        style_axis(axis)
        axis.grid(True, axis="y", color=GRID_COLOR, linestyle="--", alpha=0.6)
        axis.grid(False, axis="x")
        axis.axhline(0.0, color=MUTED_TEXT, linestyle="--", linewidth=1.1, zorder=2)
        subset = data[data["panel"] == panel]
        for position, anomaly_type in enumerate(anomaly_types):
            group = subset[subset["anomaly_type"].astype(str) == anomaly_type]
            for column, offset, color in (
                ("damage", -0.13, "#cf6f1e"),
                ("residual", 0.13, "#3d7ab5"),
            ):
                values = _finite(group[column])
                if values.size == 0:
                    continue
                jitter = rng.normal(0.0, 0.025, len(values))
                axis.scatter(
                    position + offset + jitter,
                    values,
                    s=27,
                    color=color,
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.62,
                    zorder=3,
                )
                axis.scatter(
                    [position + offset],
                    [float(np.median(values))],
                    marker="D",
                    s=48,
                    color=color,
                    edgecolors=TEXT_COLOR,
                    linewidths=0.6,
                    zorder=4,
                )
        axis.set_xticks(range(len(anomaly_types)), anomaly_types, rotation=25, ha="right")
        axis.set_title(panel, loc="left", fontsize=11)
        axis.set_xlabel("Tipo de anomalía")
        axis.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    axes[0][0].set_ylabel(f"Δ {_metric_label(metric)} respecto al contexto limpio")
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#cf6f1e",
                   markeredgecolor="white", markersize=8, label="daño con contexto corrupto"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#3d7ab5",
                   markeredgecolor="white", markersize=8, label="residuo tras limpieza"),
            Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=MUTED_TEXT,
                   markeredgecolor=TEXT_COLOR, markersize=7, label="mediana"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, top))
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=FIGURE_FACE)
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

    training_time_summary(results_df, metric="mase").to_csv(
        run_dir / "training_summary.csv", index=False
    )

    saved: list[Path] = []
    jobs = [
        *(
            (
                run_dir / f"executive_branch_summary_{metric}.png",
                lambda path, metric=metric: save_executive_branch_summary_plot(
                    path, results_df, metric
                ),
            )
            for metric in ("relmae", "relrmse")
        ),
        *(
            (
                run_dir / f"raw_model_ranking_{metric}.png",
                lambda path, metric=metric: save_raw_model_ranking_plot(
                    path, results_df, metric
                ),
            )
            for metric in FOUNDATION_METRICS
        ),
        *(
            (
                run_dir / f"executive_model_arm_{metric}.png",
                lambda path, metric=metric: save_executive_model_arm_plot(
                    path, results_df, metric
                ),
            )
            for metric in FOUNDATION_METRICS
        ),
        *(
            (
                run_dir / f"executive_imputation_{metric}.png",
                lambda path, metric=metric: save_executive_imputation_plot(
                    path, results_df, metric
                ),
            )
            for metric in FOUNDATION_METRICS
        ),
        *(
            (
                run_dir / f"executive_cost_accuracy_{metric}.png",
                lambda path, metric=metric: save_executive_cost_accuracy_plot(
                    path, results_df, metric
                ),
            )
            for metric in FOUNDATION_METRICS
        ),
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
        *(
            (
                run_dir / f"foundation_anomaly_robustness_{metric}.png",
                lambda path, metric=metric: save_foundation_anomaly_robustness_plot(
                    path, foundation_df, metric
                ),
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
