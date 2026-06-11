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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sanitize_filename(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return sanitized.strip("._") or "series"


def _build_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _repo_root() / "reports" / "benchmark" / f"montecarlo_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _to_pd_series(obj: Any) -> pd.Series:
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

    plot_seed = cfg_get_int("benchmark", "random_seed", 42)
    _, _, plot_store = run_imputation_benchmark_parallel(random_seed=plot_seed)
    plot_manifest_df = _save_plot_images(plot_store, output_dir=output_dir)

    return {
        "output_dir": output_dir,
        "results_mc_df": results_mc_df,
        "summary_mc_df": summary_mc_df,
        "ranking_by_seed_df": ranking_by_seed_df,
        "plot_manifest_df": plot_manifest_df,
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
    print(f"[info] Plot manifest CSV: {output_dir / 'plot_images.csv'}")


if __name__ == "__main__":
    main()
