"""Zero-argument terminal entrypoint for the configured Monte Carlo benchmark."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from airquality.imputation.plot_montecarlo_results import (
    render_run_figures,
    save_plot_store,
)
from airquality.imputation.run_benchmark import (
    run_imputation_benchmark_parallel_montecarlo,
)
from airquality.paths import create_run_dir


def _repo_root() -> Path:
    """Return the repository root (two levels above this module)."""
    return Path(__file__).resolve().parents[2]


def _build_output_dir() -> Path:
    """Create the timestamped output directory for one Monte Carlo run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return create_run_dir(
        _repo_root() / "reports" / "benchmark", f"montecarlo_{stamp}"
    )


def run_benchmark_from_config() -> dict[str, Any]:
    """Run the configured Monte Carlo benchmark and persist all artifacts."""
    output_dir = _build_output_dir()
    results_mc_df, summary_mc_df, ranking_by_seed_df, plot_store = (
        run_imputation_benchmark_parallel_montecarlo()
    )
    results_mc_df.to_csv(output_dir / "results_mc.csv", index=False)
    summary_mc_df.to_csv(output_dir / "summary_mc.csv", index=False)
    ranking_by_seed_df.to_csv(output_dir / "ranking_by_seed.csv", index=False)

    plot_store_path = save_plot_store(plot_store, output_dir / "plot_store.csv.gz")
    plot_artifacts = render_run_figures(
        output_dir,
        results_mc_df=results_mc_df,
        plot_store=plot_store,
    )

    return {
        "output_dir": output_dir,
        "results_mc_df": results_mc_df,
        "summary_mc_df": summary_mc_df,
        "ranking_by_seed_df": ranking_by_seed_df,
        "plot_manifest_df": plot_artifacts["plot_manifest_df"],
        "metric_gap_plot_path": plot_artifacts["metric_gap_plot_path"],
        "plot_store_path": plot_store_path,
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
    print(f"[info] Plot data CSV: {artifacts['plot_store_path']}")
    if artifacts.get("metric_gap_plot_path") is not None:
        print(f"[info] Metrics-by-gap plot: {artifacts['metric_gap_plot_path']}")
    print(f"[info] Plot manifest CSV: {output_dir / 'plot_images.csv'}")


if __name__ == "__main__":
    main()
