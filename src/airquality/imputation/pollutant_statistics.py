"""Global paired comparison of two pollutant imputation benchmark runs.

The Monte Carlo rows are repeated gap samples, not independent stations.  This
module therefore averages the seeds first, averages the fixed model/gap catalog
second, and performs inference on one paired score per station.

Example
-------
    uv run python -m airquality.imputation.pollutant_statistics \
        reports/benchmark/montecarlo_no2_20260913_033148 \
        reports/benchmark/montecarlo_o3_20260913_060307 \
        --output-dir reports/benchmark/comparison_no2_o3
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests

from airquality.paths import create_run_dir


DEFAULT_METRICS = ("MASE", "RMSSE")
DEFAULT_STATION_ALIASES = {
    "UPCT_NO2": "UPCT",
    "UPCT_O3": "UPCT",
}
RAW_KEY_COLUMNS = (
    "Modelo",
    "Serie",
    "Gap_Size",
    "Seed",
    "MonteCarlo_Run",
)
HOLDOUT_COLUMNS = ("Test_Start", "Test_End", "Test_Block_Points")
REQUIRED_COLUMNS = frozenset(
    {
        *RAW_KEY_COLUMNS,
        "Gap_Plan_ID",
        *HOLDOUT_COLUMNS,
        "N_Gaps_Target",
        "N_Gaps_Scored",
        "N_Target_Points",
        "N_Scored_Points",
        "Support_Fraction",
        *DEFAULT_METRICS,
    }
)
OUTPUT_FILES = {
    "station_scores": "global_station_scores.csv",
    "summary": "global_statistical_summary.csv",
    "protocol": "global_protocol_overlap.csv",
    "metadata": "global_comparison_metadata.json",
    "report": "global_statistical_report.md",
}


def _label_code(label: str) -> str:
    """Convert a pollutant label to a safe suffix for output columns."""
    code = "".join(char if char.isalnum() else "_" for char in str(label).strip())
    code = code.strip("_").upper()
    if not code:
        raise ValueError("Las etiquetas de contaminante no pueden estar vacias")
    return code


def _validate_options(
    *,
    metrics: Sequence[str],
    alpha: float,
    equivalence_margin: float,
    seed: int,
) -> tuple[str, ...]:
    normalized_metrics = tuple(str(metric).strip().upper() for metric in metrics)
    if not normalized_metrics or any(metric not in DEFAULT_METRICS for metric in normalized_metrics):
        raise ValueError(f"Las metricas deben pertenecer a {DEFAULT_METRICS}")
    if len(set(normalized_metrics)) != len(normalized_metrics):
        raise ValueError("Las metricas no pueden repetirse")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha debe estar entre cero y uno")
    if not 0.0 < float(equivalence_margin) < 1.0:
        raise ValueError("equivalence_margin debe estar entre cero y uno")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed debe ser un entero")
    return normalized_metrics


def _prepare_results(
    results: pd.DataFrame,
    *,
    label: str,
    station_aliases: Mapping[str, str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Validate and normalize one persisted Monte Carlo result table."""
    missing = sorted(REQUIRED_COLUMNS - set(results.columns))
    if missing:
        raise ValueError(f"Faltan columnas en results_mc de {label}: {', '.join(missing)}")
    if results.empty:
        raise ValueError(f"results_mc de {label} no contiene resultados")

    frame = results.copy()
    for column in RAW_KEY_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"{label}: la columna {column} contiene nulos")
    frame["Serie"] = frame["Serie"].astype(str)
    frame["station"] = frame["Serie"].replace(dict(station_aliases))
    frame["Modelo"] = frame["Modelo"].astype(str)
    if (frame["Serie"].str.strip().eq("")).any() or (
        frame["Modelo"].str.strip().eq("")
    ).any():
        raise ValueError(f"{label}: Modelo y Serie no pueden estar vacios")

    for column in ("Gap_Size", "Seed", "MonteCarlo_Run"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{label}: {column} debe contener enteros finitos")
        frame[column] = values.astype(int)
    if (frame["Gap_Size"] <= 0).any() or (frame["MonteCarlo_Run"] <= 0).any():
        raise ValueError(f"{label}: Gap_Size y MonteCarlo_Run deben ser positivos")

    for column in (
        "N_Gaps_Target",
        "N_Gaps_Scored",
        "N_Target_Points",
        "N_Scored_Points",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{label}: {column} debe contener enteros finitos")
        frame[column] = values.astype(int)
    if (
        (frame["N_Gaps_Target"] <= 0).any()
        or (frame["N_Gaps_Target"] != frame["N_Gaps_Scored"]).any()
        or (frame["N_Target_Points"] <= 0).any()
        or (frame["N_Target_Points"] != frame["N_Scored_Points"]).any()
    ):
        raise ValueError(f"{label}: el soporte de gaps no es completo")

    support = pd.to_numeric(frame["Support_Fraction"], errors="coerce")
    if support.isna().any() or not np.isclose(support, 1.0, rtol=0.0, atol=1e-9).all():
        raise ValueError(f"{label}: Support_Fraction debe ser uno en todas las filas")
    frame["Support_Fraction"] = support

    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{label}: {metric} contiene valores invalidos")
        frame[metric] = values.astype(float)

    if frame["Gap_Plan_ID"].isna().any() or (
        frame["Gap_Plan_ID"].astype(str).str.strip().str.len() == 0
    ).any():
        raise ValueError(f"{label}: Gap_Plan_ID no puede estar vacio")

    starts = pd.to_datetime(frame["Test_Start"], errors="coerce", format="mixed")
    ends = pd.to_datetime(frame["Test_End"], errors="coerce", format="mixed")
    if starts.isna().any() or ends.isna().any() or (ends < starts).any():
        raise ValueError(f"{label}: las fechas de holdout no son validas")
    frame["_holdout_start"] = starts.dt.strftime("%Y-%m-%dT%H:%M:%S")
    frame["_holdout_end"] = ends.dt.strftime("%Y-%m-%dT%H:%M:%S")
    frame["_holdout_points"] = frame["Test_Block_Points"].astype(int)

    if frame.duplicated(list(RAW_KEY_COLUMNS)).any():
        raise ValueError(f"{label}: hay filas duplicadas en la clave Monte Carlo")
    seed_runs = frame[["Seed", "MonteCarlo_Run"]].drop_duplicates()
    if (
        seed_runs.groupby("Seed")["MonteCarlo_Run"].nunique().gt(1).any()
        or seed_runs.groupby("MonteCarlo_Run")["Seed"].nunique().gt(1).any()
    ):
        raise ValueError(f"{label}: Seed y MonteCarlo_Run no son una pareja uno a uno")
    normalized_keys = ["Modelo", "station", "Gap_Size", "Seed", "MonteCarlo_Run"]
    if frame.duplicated(normalized_keys).any():
        raise ValueError(
            f"{label}: los alias de estaciones generan claves duplicadas"
        )
    return frame


def build_comparison_panel(
    left_results: pd.DataFrame,
    right_results: pd.DataFrame,
    *,
    left_label: str = "NO2",
    right_label: str = "O3",
    station_aliases: Mapping[str, str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Build the paired row panel for two benchmark result tables.

    Rows are paired by station, model, gap, seed, and Monte Carlo run.  The
    station alias is applied before checking that both runs have the same panel.
    Holdout and gap-plan equality are retained as diagnostic columns rather than
    being required for the primary station pairing.
    """
    normalized_metrics = _validate_options(
        metrics=metrics,
        alpha=0.05,
        equivalence_margin=0.10,
        seed=0,
    )
    left_code = _label_code(left_label)
    right_code = _label_code(right_label)
    if left_code == right_code:
        raise ValueError("Las etiquetas de contaminante deben ser distintas")
    aliases = dict(DEFAULT_STATION_ALIASES)
    if station_aliases is not None:
        aliases.update({str(key): str(value) for key, value in station_aliases.items()})

    left = _prepare_results(
        left_results,
        label=left_code,
        station_aliases=aliases,
        metrics=normalized_metrics,
    )
    right = _prepare_results(
        right_results,
        label=right_code,
        station_aliases=aliases,
        metrics=normalized_metrics,
    )
    keys = ["Modelo", "station", "Gap_Size", "Seed", "MonteCarlo_Run"]
    left_keys = set(map(tuple, left[keys].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[keys].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise ValueError(
            "Los runs no tienen el mismo panel tras normalizar estaciones: "
            f"{len(left_keys - right_keys)} filas solo en {left_code}, "
            f"{len(right_keys - left_keys)} solo en {right_code}"
        )

    payload = keys + ["Gap_Plan_ID", "_holdout_start", "_holdout_end", "_holdout_points"]
    payload.extend(normalized_metrics)
    left_payload = left[payload].rename(
        columns={metric: f"{metric}_{left_code}" for metric in normalized_metrics}
    )
    right_payload = right[payload].rename(
        columns={metric: f"{metric}_{right_code}" for metric in normalized_metrics}
    )
    panel = left_payload.merge(
        right_payload,
        on=keys,
        how="inner",
        suffixes=(f"_{left_code}", f"_{right_code}"),
        validate="one_to_one",
    )
    panel["same_gap_plan"] = (
        panel[f"Gap_Plan_ID_{left_code}"] == panel[f"Gap_Plan_ID_{right_code}"]
    )
    panel["same_holdout"] = (
        (panel[f"_holdout_start_{left_code}"] == panel[f"_holdout_start_{right_code}"])
        & (panel[f"_holdout_end_{left_code}"] == panel[f"_holdout_end_{right_code}"])
        & (
            panel[f"_holdout_points_{left_code}"]
            == panel[f"_holdout_points_{right_code}"]
        )
    )
    panel["same_protocol"] = panel["same_gap_plan"] & panel["same_holdout"]
    return panel.sort_values(keys, kind="stable").reset_index(drop=True)


def build_global_station_scores(
    panel: pd.DataFrame,
    *,
    left_label: str = "NO2",
    right_label: str = "O3",
    metrics: Sequence[str] = DEFAULT_METRICS,
    scope: str = "all_stations",
) -> pd.DataFrame:
    """Average seeds, then the fixed model/gap catalog, into station scores."""
    normalized_metrics = tuple(str(metric).strip().upper() for metric in metrics)
    left_code = _label_code(left_label)
    right_code = _label_code(right_label)
    if scope not in {"all_stations", "same_protocol"}:
        raise ValueError("scope debe ser all_stations o same_protocol")
    selected = panel if scope == "all_stations" else panel.loc[panel["same_protocol"]].copy()
    if selected.empty:
        raise ValueError(f"No hay filas para el alcance {scope}")

    metric_columns = [
        f"{metric}_{suffix}"
        for metric in normalized_metrics
        for suffix in (left_code, right_code)
    ]
    cell = (
        selected.groupby(["station", "Modelo", "Gap_Size"], sort=True, as_index=False)[
            metric_columns
        ]
        .mean()
    )
    cell_run_counts = (
        selected.groupby(["station", "Modelo", "Gap_Size"], sort=True)["Seed"]
        .nunique()
        .rename("n_runs")
        .reset_index()
    )
    if cell_run_counts["n_runs"].nunique() != 1:
        raise ValueError("El numero de semillas no es constante por celda global")
    run_counts = cell_run_counts.groupby("station", sort=True, as_index=False).agg(
        n_runs_min=("n_runs", "min"),
        n_runs_max=("n_runs", "max"),
    )
    aggregation: dict[str, tuple[str, str]] = {}
    for metric in normalized_metrics:
        aggregation[f"{metric}_{left_code}"] = (f"{metric}_{left_code}", "mean")
        aggregation[f"{metric}_{right_code}"] = (f"{metric}_{right_code}", "mean")
    scores = cell.groupby("station", sort=True, as_index=False).agg(**aggregation)
    cell_counts = cell.groupby("station", sort=True, as_index=False).agg(
        n_model_gap_cells=("Modelo", "size"),
        n_models=("Modelo", "nunique"),
        n_gaps=("Gap_Size", "nunique"),
    )
    scores = scores.merge(cell_counts, on="station", how="left", validate="one_to_one")
    scores = scores.merge(run_counts, on="station", how="left", validate="one_to_one")

    protocol = (
        selected.groupby("station", sort=True)
        .agg(
            same_gap_plan=("same_gap_plan", "all"),
            same_holdout=("same_holdout", "all"),
            same_protocol=("same_protocol", "all"),
        )
        .reset_index()
    )
    scores = scores.merge(protocol, on="station", how="left", validate="one_to_one")
    for metric in normalized_metrics:
        left_column = f"{metric}_{left_code}"
        right_column = f"{metric}_{right_code}"
        scores[f"{metric}_difference_{right_code}_minus_{left_code}"] = (
            scores[right_column] - scores[left_column]
        )
        scores[f"{metric}_ratio_{right_code}_to_{left_code}"] = (
            scores[right_column] / scores[left_column]
        )
    scores["scope"] = scope
    return scores.sort_values("station", kind="stable").reset_index(drop=True)


def _wilcoxon_test(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    if len(differences) == 0 or not np.isfinite(differences).all():
        return float("nan"), float("nan")
    if np.allclose(differences, 0.0, rtol=0.0, atol=1e-12):
        return 0.0, 1.0
    result = wilcoxon(
        differences,
        zero_method="pratt",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _tost_log_ratio(
    log_ratios: np.ndarray,
    *,
    alpha: float,
    equivalence_margin: float,
) -> tuple[float, float, float, bool]:
    """Run a paired TOST on log ratios within a relative tolerance."""
    if len(log_ratios) < 2 or not np.isfinite(log_ratios).all():
        return float("nan"), float("nan"), float("nan"), False
    lower = float(np.log1p(-equivalence_margin))
    upper = float(np.log1p(equivalence_margin))
    lower_test = ttest_1samp(log_ratios, lower, alternative="greater")
    upper_test = ttest_1samp(log_ratios, upper, alternative="less")
    p_lower = float(lower_test.pvalue)
    p_upper = float(upper_test.pvalue)
    p_tost = max(p_lower, p_upper)
    return p_lower, p_upper, p_tost, bool(p_lower < alpha and p_upper < alpha)


def analyze_global_scores(
    station_scores: pd.DataFrame,
    *,
    left_label: str = "NO2",
    right_label: str = "O3",
    metrics: Sequence[str] = DEFAULT_METRICS,
    alpha: float = 0.05,
    equivalence_margin: float = 0.10,
    seed: int = 13,
) -> pd.DataFrame:
    """Test the global paired station scores for each requested metric."""
    normalized_metrics = _validate_options(
        metrics=metrics,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        seed=seed,
    )
    left_code = _label_code(left_label)
    right_code = _label_code(right_label)
    if station_scores.empty:
        raise ValueError("station_scores no contiene estaciones")
    rows: list[dict[str, Any]] = []
    for metric in normalized_metrics:
        left_column = f"{metric}_{left_code}"
        right_column = f"{metric}_{right_code}"
        if left_column not in station_scores or right_column not in station_scores:
            raise ValueError(f"Faltan scores globales para {metric}")
        left_values = pd.to_numeric(station_scores[left_column], errors="coerce").to_numpy()
        right_values = pd.to_numeric(station_scores[right_column], errors="coerce").to_numpy()
        valid = np.isfinite(left_values) & np.isfinite(right_values)
        left_values = left_values[valid]
        right_values = right_values[valid]
        differences = right_values - left_values
        if (left_values <= 0.0).any() or (right_values <= 0.0).any():
            log_ratios = np.full(len(differences), np.nan)
        else:
            log_ratios = np.log(right_values) - np.log(left_values)
        wilcoxon_statistic, p_wilcoxon = _wilcoxon_test(differences)
        if np.isfinite(log_ratios).all():
            p_equiv_lower, p_equiv_upper, p_tost, equivalent = _tost_log_ratio(
                log_ratios,
                alpha=alpha,
                equivalence_margin=equivalence_margin,
            )
            geometric_ratio = float(np.exp(np.mean(log_ratios)))
        else:
            p_equiv_lower = p_equiv_upper = p_tost = float("nan")
            equivalent = False
            geometric_ratio = float("nan")
        rows.append(
            {
                "scope": str(station_scores["scope"].iloc[0])
                if "scope" in station_scores
                else "unknown",
                "metric": metric,
                "primary": metric == "MASE",
                "n_stations": int(len(differences)),
                f"{left_code}_mean": float(np.mean(left_values)),
                f"{right_code}_mean": float(np.mean(right_values)),
                "mean_difference_right_minus_left": float(np.mean(differences)),
                "median_difference_right_minus_left": float(np.median(differences)),
                "geometric_mean_ratio_right_to_left": geometric_ratio,
                "wilcoxon_statistic": wilcoxon_statistic,
                "p_wilcoxon": p_wilcoxon,
                "equivalence_margin": equivalence_margin,
                "equivalence_lower_ratio": 1.0 - equivalence_margin,
                "equivalence_upper_ratio": 1.0 + equivalence_margin,
                "p_equivalence_lower": p_equiv_lower,
                "p_equivalence_upper": p_equiv_upper,
                "p_tost": p_tost,
                "equivalent_tost": equivalent,
                "alpha": alpha,
                "analysis_seed": int(seed),
            }
        )

    summary = pd.DataFrame(rows)
    for p_column, adjusted_column in (("p_wilcoxon", "p_wilcoxon_holm"),):
        values = summary[p_column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        adjusted = np.full(len(values), np.nan)
        if finite.any():
            adjusted[finite] = multipletests(values[finite], alpha=alpha, method="holm")[1]
        summary[adjusted_column] = adjusted
    summary["significant_wilcoxon"] = summary["p_wilcoxon_holm"] < alpha
    summary["decision"] = np.select(
        [summary["significant_wilcoxon"], summary["equivalent_tost"]],
        ["difference_detected", "practically_equivalent"],
        default="no_evidence_or_inconclusive",
    )
    return summary


def build_protocol_overlap(
    panel: pd.DataFrame,
    *,
    left_label: str = "NO2",
    right_label: str = "O3",
) -> pd.DataFrame:
    """Summarize same-plan and same-holdout coverage by station."""
    left_code = _label_code(left_label)
    right_code = _label_code(right_label)
    grouped = (
        panel.groupby("station", sort=True)
        .agg(
            n_rows=("station", "size"),
            same_gap_plan=("same_gap_plan", "all"),
            same_holdout=("same_holdout", "all"),
            same_protocol=("same_protocol", "all"),
            holdout_start_left=(f"_holdout_start_{left_code}", "first"),
            holdout_end_left=(f"_holdout_end_{left_code}", "first"),
            holdout_start_right=(f"_holdout_start_{right_code}", "first"),
            holdout_end_right=(f"_holdout_end_{right_code}", "first"),
        )
        .reset_index()
    )
    return grouped


def _summary_sentence(row: pd.Series) -> str:
    """Create a compact Spanish interpretation for one summary row."""
    metric = str(row["metric"])
    scope = str(row["scope"])
    p_value = float(row["p_wilcoxon_holm"])
    decision = str(row["decision"])
    difference = float(row["mean_difference_right_minus_left"])
    if decision == "difference_detected":
        return (
            f"{scope}, {metric}: se detecta una diferencia global "
            f"(p Holm={p_value:.4g}; diferencia derecha-izquierda={difference:.4g})."
        )
    if decision == "practically_equivalent":
        return (
            f"{scope}, {metric}: los resultados son equivalentes dentro del "
            f"margen practico (p TOST={float(row['p_tost']):.4g})."
        )
    return (
        f"{scope}, {metric}: no se detecta una diferencia estadistica "
        f"(p Holm={p_value:.4g}); esto no demuestra equivalencia."
    )


def _write_report(
    path: Path,
    *,
    left_run: Path,
    right_run: Path,
    summary: pd.DataFrame,
    protocol: pd.DataFrame,
    left_label: str,
    right_label: str,
    metrics: Sequence[str],
    equivalence_margin: float,
) -> None:
    lines = [
        f"# Comparacion global de imputacion: {left_label} frente a {right_label}",
        "",
        "La unidad inferencial es la estacion. Las semillas Monte Carlo se promedian "
        "antes de promediar el catalogo fijo de modelos y gaps.",
        "",
        f"- Run izquierdo: `{left_run}`",
        f"- Run derecho: `{right_run}`",
        f"- Estaciones emparejadas: {len(protocol)}",
        f"- Mismo holdout y plan de gaps: {int(protocol['same_protocol'].sum())}",
        f"- Margen de equivalencia: +/-{equivalence_margin:.0%}",
        f"- Metrica principal: `{metrics[0]}`",
        "",
        "## Resultados",
        "",
        "La diferencia se define como resultado derecho menos resultado izquierdo. "
        "Para MASE y RMSSE, un valor positivo indica mayor error en el run derecho.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(f"- {_summary_sentence(row)}")
    lines.extend(
        [
            "",
            "Un p-valor no significativo indica ausencia de evidencia suficiente, no "
            "igualdad. La equivalencia solo se declara cuando el TOST queda dentro del "
            "margen relativo prefijado.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_global_comparison(
    left_run: str | Path,
    right_run: str | Path,
    *,
    output_dir: str | Path | None = None,
    left_label: str = "NO2",
    right_label: str = "O3",
    station_aliases: Mapping[str, str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    alpha: float = 0.05,
    equivalence_margin: float = 0.10,
    seed: int = 13,
) -> dict[str, Any]:
    """Analyze two run directories and persist the global comparison artifacts."""
    normalized_metrics = _validate_options(
        metrics=metrics,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        seed=seed,
    )
    left_run = Path(left_run).expanduser().resolve()
    right_run = Path(right_run).expanduser().resolve()
    left_results_path = left_run / "results_mc.csv"
    right_results_path = right_run / "results_mc.csv"
    if not left_results_path.is_file() or not right_results_path.is_file():
        raise FileNotFoundError("Ambos runs deben contener results_mc.csv")
    left_results = pd.read_csv(left_results_path, keep_default_na=False)
    right_results = pd.read_csv(right_results_path, keep_default_na=False)
    panel = build_comparison_panel(
        left_results,
        right_results,
        left_label=left_label,
        right_label=right_label,
        station_aliases=station_aliases,
        metrics=normalized_metrics,
    )
    all_scores = build_global_station_scores(
        panel,
        left_label=left_label,
        right_label=right_label,
        metrics=normalized_metrics,
        scope="all_stations",
    )
    exact_scores = build_global_station_scores(
        panel,
        left_label=left_label,
        right_label=right_label,
        metrics=normalized_metrics,
        scope="same_protocol",
    )
    all_summary = analyze_global_scores(
        all_scores,
        left_label=left_label,
        right_label=right_label,
        metrics=normalized_metrics,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        seed=seed,
    )
    exact_summary = analyze_global_scores(
        exact_scores,
        left_label=left_label,
        right_label=right_label,
        metrics=normalized_metrics,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        seed=seed,
    )
    summary = pd.concat([all_summary, exact_summary], ignore_index=True)
    protocol = build_protocol_overlap(
        panel,
        left_label=left_label,
        right_label=right_label,
    )

    if output_dir is None:
        base_dir = left_run.parent
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = create_run_dir(base_dir, f"comparison_{_label_code(left_label).lower()}_{_label_code(right_label).lower()}_{stamp}")
    else:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

    station_scores = pd.concat([all_scores, exact_scores], ignore_index=True)
    station_scores.to_csv(output_path / OUTPUT_FILES["station_scores"], index=False)
    summary.to_csv(output_path / OUTPUT_FILES["summary"], index=False)
    protocol.to_csv(output_path / OUTPUT_FILES["protocol"], index=False)
    metadata = {
        "left_run": str(left_run),
        "right_run": str(right_run),
        "left_label": left_label,
        "right_label": right_label,
        "station_aliases": dict(DEFAULT_STATION_ALIASES)
        if station_aliases is None
        else {**DEFAULT_STATION_ALIASES, **dict(station_aliases)},
        "metrics": list(normalized_metrics),
        "alpha": float(alpha),
        "equivalence_margin": float(equivalence_margin),
        "analysis_seed": int(seed),
        "n_merged_rows": int(len(panel)),
        "n_stations": int(protocol["station"].nunique()),
        "same_protocol_stations": sorted(
            protocol.loc[protocol["same_protocol"], "station"].astype(str).tolist()
        ),
    }
    (output_path / OUTPUT_FILES["metadata"]).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_report(
        output_path / OUTPUT_FILES["report"],
        left_run=left_run,
        right_run=right_run,
        summary=summary,
        protocol=protocol,
        left_label=left_label,
        right_label=right_label,
        metrics=normalized_metrics,
        equivalence_margin=equivalence_margin,
    )
    return {
        "output_dir": output_path,
        "panel": panel,
        "station_scores": station_scores,
        "summary": summary,
        "protocol": protocol,
        "paths": {key: output_path / filename for key, filename in OUTPUT_FILES.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparacion estadistica global pareada de dos runs de imputacion"
    )
    parser.add_argument("left_run", type=Path, help="Run izquierdo, por defecto NO2")
    parser.add_argument("right_run", type=Path, help="Run derecho, por defecto O3")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--left-label", default="NO2")
    parser.add_argument("--right-label", default="O3")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--equivalence-margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    artifacts = write_global_comparison(
        args.left_run,
        args.right_run,
        output_dir=args.output_dir,
        left_label=args.left_label,
        right_label=args.right_label,
        alpha=args.alpha,
        equivalence_margin=args.equivalence_margin,
        seed=args.seed,
    )
    print(f"[info] Global comparison saved under {artifacts['output_dir']}")
    print(artifacts["summary"].to_string(index=False))
    for path in artifacts["paths"].values():
        print(f"[info] {path}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_STATION_ALIASES",
    "analyze_global_scores",
    "build_comparison_panel",
    "build_global_station_scores",
    "build_protocol_overlap",
    "write_global_comparison",
]
