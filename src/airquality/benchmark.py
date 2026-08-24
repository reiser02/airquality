"""Zero-argument terminal entrypoint for the configured Monte Carlo benchmark."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from airquality.visualizations.montecarlo import (
    render_run_figures,
    save_plot_store,
)
from airquality.imputation.run_benchmark import (
    run_imputation_benchmark_parallel_montecarlo,
)
from airquality.paths import create_run_dir


LOGGER = logging.getLogger(__name__)


def _build_output_dir() -> Path:
    """Create the timestamped output directory for one Monte Carlo run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return create_run_dir(
        Path(__file__).resolve().parents[2] / "reports" / "benchmark",
        f"montecarlo_{stamp}",
    )


def _configure_logging(output_dir: Path) -> Path:
    """Log benchmark progress to both the terminal and the run directory."""
    log_path = output_dir / "benchmark.log"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    package_logger = logging.getLogger("airquality")
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    package_logger.handlers.clear()
    package_logger.addHandler(stream_handler)
    package_logger.addHandler(file_handler)
    return log_path


def run_benchmark_from_config() -> dict[str, Any]:
    """Run the configured Monte Carlo benchmark and persist all artifacts."""
    output_dir = _build_output_dir()
    log_path = _configure_logging(output_dir)
    LOGGER.info("Benchmark run directory: %s", output_dir)
    results_mc_df, summary_mc_df, ranking_by_seed_df, plot_store = (
        run_imputation_benchmark_parallel_montecarlo()
    )
    results_mc_df.to_csv(output_dir / "results_mc.csv", index=False)
    summary_mc_df.to_csv(output_dir / "summary_mc.csv", index=False)
    ranking_by_seed_df.to_csv(output_dir / "ranking_by_seed.csv", index=False)
    holdout_columns = [
        column
        for column in (
            "Serie",
            "Test_Start",
            "Test_End",
            "Test_Block_Points",
            "Train_Points_Before",
            "Train_Points_After",
        )
        if column in results_mc_df.columns
    ]
    holdout_coverage_df = (
        results_mc_df[holdout_columns].drop_duplicates("Serie").sort_values("Serie")
        if "Serie" in holdout_columns
        else None
    )
    if holdout_coverage_df is not None:
        holdout_coverage_df.to_csv(output_dir / "holdout_coverage.csv", index=False)

    plot_store_path = save_plot_store(plot_store, output_dir / "plot_store.csv.gz")
    plot_artifacts = render_run_figures(
        output_dir,
        results_mc_df=results_mc_df,
        plot_store=plot_store,
    )

    return {
        "output_dir": output_dir,
        "log_path": log_path,
        "results_mc_df": results_mc_df,
        "summary_mc_df": summary_mc_df,
        "ranking_by_seed_df": ranking_by_seed_df,
        "holdout_coverage_df": holdout_coverage_df,
        **plot_artifacts,
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
    if artifacts.get("log_path") is not None:
        print(f"[info] Benchmark log: {artifacts['log_path']}")
    print(f"[info] Results CSV: {output_dir / 'results_mc.csv'}")
    print(f"[info] Summary CSV: {output_dir / 'summary_mc.csv'}")
    print(f"[info] Ranking CSV: {output_dir / 'ranking_by_seed.csv'}")
    if artifacts.get("holdout_coverage_df") is not None:
        print(f"[info] Holdout coverage CSV: {output_dir / 'holdout_coverage.csv'}")
    print(f"[info] Plot data CSV: {artifacts['plot_store_path']}")
    if artifacts.get("metric_gap_plot_path") is not None:
        print(f"[info] Metrics-by-gap plot: {artifacts['metric_gap_plot_path']}")
    summary_plot_count = sum(
        len(artifacts.get(key, {}))
        for key in (
            "model_performance_by_gap_plot_paths",
            "overall_model_performance_plot_paths",
            "global_station_error_plot_paths",
            "pairwise_win_rate_plot_paths",
            "gap_degradation_plot_paths",
            "tail_risk_plot_paths",
            "error_correlation_plot_paths",
        )
    )
    if summary_plot_count:
        print(f"[info] Summary plots: {summary_plot_count}")
    diagnostic_tables = artifacts.get("diagnostic_table_paths", {})
    if diagnostic_tables:
        print(f"[info] Diagnostic tables: {len(diagnostic_tables)}")
    print(f"[info] Plot manifest CSV: {output_dir / 'plot_images.csv'}")


if __name__ == "__main__":
    main()
