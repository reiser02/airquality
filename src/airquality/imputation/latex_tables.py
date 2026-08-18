"""Build summary CSV and LaTeX tables from a Monte Carlo run.

Example::

    uv run python -m airquality.imputation.latex_tables \
        reports/benchmark/montecarlo_20260730_181858
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Final

import pandas as pd


METRIC_ORDER: Final[tuple[str, ...]] = ("MAE", "RMSE", "MASE")
PROFILE_TOLERANCES: Final[tuple[float, ...]] = (0.0, 5.0, 10.0, 25.0, 50.0)


def _read_csv(run_dir: Path, filename: str, required: tuple[str, ...]) -> pd.DataFrame:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Falta el CSV requerido: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} no contiene las columnas requeridas: {missing}")
    return frame


def _metric_order(frame: pd.DataFrame) -> list[str]:
    present = [metric for metric in METRIC_ORDER if metric in set(frame["Metric"])]
    return [*present, *sorted(set(frame["Metric"]) - set(present))]


def build_overall_summary(overall: pd.DataFrame, *, top_k: int = 5) -> pd.DataFrame:
    """Return the best ``top_k`` models per metric with absolute statistics."""
    if top_k < 1:
        raise ValueError("top_k debe ser al menos 1")
    required = {"Metric", "Modelo", "Mean", "Median", "Station_SD"}
    missing = sorted(required.difference(overall.columns))
    if missing:
        raise ValueError(f"Faltan columnas en overall_model_performance.csv: {missing}")

    rows: list[pd.DataFrame] = []
    for metric in _metric_order(overall):
        group = overall[overall["Metric"] == metric].copy()
        group = group.sort_values(
            ["Mean", "Median", "Station_SD", "Modelo"], kind="stable"
        ).head(top_k)
        group.insert(1, "Rank_Mean", range(1, len(group) + 1))
        rows.append(group)
    if not rows:
        return pd.DataFrame()

    columns = [
        "Metric",
        "Rank_Mean",
        "Modelo",
        "Mean",
        "Median",
        "Station_SD",
        "Mean_Rank",
        "Winner_Percent",
        "Top_3_Percent",
        "N_Stations",
        "N_Gaps",
        "N_Contexts",
    ]
    return pd.concat(rows, ignore_index=True).reindex(columns=columns)


def build_gap_summary(by_gap: pd.DataFrame, *, top_k: int = 3) -> pd.DataFrame:
    """Return the best ``top_k`` models for every metric and gap size."""
    if top_k < 1:
        raise ValueError("top_k debe ser al menos 1")
    required = {"Metric", "Modelo", "Gap_Size", "Mean", "Median", "Station_SD"}
    missing = sorted(required.difference(by_gap.columns))
    if missing:
        raise ValueError(f"Faltan columnas en model_performance_by_gap.csv: {missing}")

    groups: list[pd.DataFrame] = []
    for (metric, gap_size), group in by_gap.groupby(
        ["Metric", "Gap_Size"], sort=False
    ):
        selected = group.sort_values(
            ["Mean", "Median", "Station_SD", "Modelo"], kind="stable"
        ).head(top_k).copy()
        selected.insert(2, "Rank_Mean", range(1, len(selected) + 1))
        groups.append(selected)
    if not groups:
        return pd.DataFrame()

    order = {metric: index for index, metric in enumerate(_metric_order(by_gap))}
    result = pd.concat(groups, ignore_index=True)
    result["_metric_order"] = result["Metric"].map(order).fillna(len(order))
    result = result.sort_values(
        ["_metric_order", "Gap_Size", "Rank_Mean"], kind="stable"
    ).drop(columns="_metric_order")
    columns = [
        "Metric",
        "Gap_Size",
        "Rank_Mean",
        "Modelo",
        "Mean",
        "Median",
        "Station_SD",
        "Mean_Rank",
        "Winner_Percent",
        "Top_3_Percent",
        "N_Stations",
    ]
    return result.reindex(columns=columns).reset_index(drop=True)


def build_profile_summary(
    profiles: pd.DataFrame,
    rank_summary: pd.DataFrame,
    overall: pd.DataFrame,
) -> pd.DataFrame:
    """Return the global-profile percentages at useful tolerance thresholds."""
    required = {"Metric", "Modelo", "Threshold_Ratio", "Context_Percent"}
    missing = sorted(required.difference(profiles.columns))
    if missing:
        raise ValueError(f"Faltan columnas en global_performance_profiles.csv: {missing}")

    work = profiles[profiles["Tolerance_Percent"].isin(PROFILE_TOLERANCES)].copy()
    if work.empty:
        return pd.DataFrame()
    table = work.pivot_table(
        index=["Metric", "Modelo"],
        columns="Tolerance_Percent",
        values="Context_Percent",
        aggfunc="first",
    ).reset_index()
    table.columns = [
        "Metric",
        "Modelo",
        *[
            "Within_Best" if tolerance == 0.0 else f"Within_{int(tolerance)}pct"
            for tolerance in table.columns[2:]
        ],
    ]
    rank_columns = rank_summary[["Metric", "Modelo", "Mean_Rank"]]
    absolute_columns = overall[["Metric", "Modelo", "Mean", "Median", "Station_SD"]]
    table = table.merge(rank_columns, on=["Metric", "Modelo"], how="left")
    table = table.merge(absolute_columns, on=["Metric", "Modelo"], how="left")
    metric_order = {metric: index for index, metric in enumerate(_metric_order(table))}
    table["_metric_order"] = table["Metric"].map(metric_order).fillna(len(metric_order))
    return table.sort_values(
        ["_metric_order", "Mean_Rank", "Modelo"], kind="stable"
    ).drop(columns="_metric_order").reset_index(drop=True)


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return re.sub(r"[\\&%$#_{}]", lambda match: replacements[match.group(0)], text)


def _format_number(value: object, decimals: int) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def _latex_frame(frame: pd.DataFrame, *, caption: str, label: str) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if column in {"Mean", "Median", "Station_SD", "Std_Stations"}:
            formatted[column] = formatted[column].map(lambda value: _format_number(value, 4))
        elif column in {
            "Mean_Rank",
            "Winner_Percent",
            "Top_3_Percent",
            "Within_Best",
            "Within_5pct",
            "Within_10pct",
            "Within_25pct",
            "Within_50pct",
        }:
            formatted[column] = formatted[column].map(lambda value: _format_number(value, 1))
        elif pd.api.types.is_object_dtype(formatted[column]):
            formatted[column] = formatted[column].map(_escape_latex)

    column_names = {
        "Metric": "Metric",
        "Rank_Mean": "Rank mean",
        "Gap_Size": "Gap (h)",
        "Modelo": "Model",
        "Mean": "Mean",
        "Median": "Median",
        "Station_SD": "SD stations",
        "Mean_Rank": "Mean rank",
        "Winner_Percent": "Winner (\\%)",
        "Top_3_Percent": "Top 3 (\\%)",
        "N_Stations": "Stations",
        "N_Gaps": "Gaps",
        "N_Contexts": "Contexts",
        "Within_Best": "Best (\\%)",
        "Within_5pct": "Within 5\\%",
        "Within_10pct": "Within 10\\%",
        "Within_25pct": "Within 25\\%",
        "Within_50pct": "Within 50\\%",
    }
    formatted = formatted.rename(columns=column_names)
    column_format = "l" + "c" * (len(formatted.columns) - 1)
    body = formatted.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        column_format=column_format,
    )
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{_escape_latex(caption)}}}\n"
        f"\\label{{{_escape_latex(label)}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def export_latex_tables(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    top_k: int = 5,
) -> dict[str, Path]:
    """Read benchmark CSVs and write summary CSV/LaTeX tables."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio del run: {run_dir}")
    output_dir = output_dir or run_dir / "latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = _read_csv(
        run_dir,
        "overall_model_performance.csv",
        ("Metric", "Modelo", "Mean", "Median", "Station_SD"),
    )
    by_gap = _read_csv(
        run_dir,
        "model_performance_by_gap.csv",
        ("Metric", "Modelo", "Gap_Size", "Mean", "Median", "Station_SD"),
    )
    rank_summary = _read_csv(
        run_dir,
        "global_rank_summary.csv",
        ("Metric", "Modelo", "Mean_Rank"),
    )
    profiles = _read_csv(
        run_dir,
        "global_performance_profiles.csv",
        ("Metric", "Modelo", "Tolerance_Percent", "Context_Percent"),
    )

    overall_summary = build_overall_summary(overall, top_k=top_k)
    gap_summary = build_gap_summary(by_gap, top_k=min(top_k, 3))
    profile_summary = build_profile_summary(profiles, rank_summary, overall)

    outputs: dict[str, Path] = {}
    csv_frames = {
        "overall_model_summary": overall_summary,
        "best_models_by_gap": gap_summary,
        "global_profile_summary": profile_summary,
    }
    for name, frame in csv_frames.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name + ".csv"] = path

    latex_tables = {
        "overall_model_summary.tex": _latex_frame(
            overall_summary,
            caption="Best imputation models by absolute global error.",
            label="tab:imputation-overall-models",
        ),
        "best_models_by_gap.tex": _latex_frame(
            gap_summary,
            caption="Best imputation models by gap size.",
            label="tab:imputation-models-by-gap",
        ),
        "global_profile_summary.tex": _latex_frame(
            profile_summary,
            caption="Global performance profile relative to the winner in each context.",
            label="tab:imputation-global-profile",
        ),
    }
    for filename, content in latex_tables.items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        outputs[filename] = path
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Directorio reports/benchmark/montecarlo_*")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida; por defecto, <run_dir>/latex_tables",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Numero de modelos globales por metrica (por defecto: 5)",
    )
    args = parser.parse_args(argv)
    outputs = export_latex_tables(
        args.run_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    for path in outputs.values():
        print(f"[info] Tabla guardada en {path}")


if __name__ == "__main__":
    main()
