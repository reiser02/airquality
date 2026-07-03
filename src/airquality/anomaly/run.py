"""CLI entrypoint: ``uv run python -m airquality.anomaly.run``.

Defaults come from the ``[anomaly]`` section of the project config; any argument
overrides its config default.
"""

from __future__ import annotations

import argparse
import json

from airquality.config import cfg_get_int, cfg_get_str

from .benchmark import AnomalyBenchmarkConfig, run_benchmark
from .ensemble import DEFAULT_ENSEMBLE_METHOD, DEFAULT_TOP_K


def _csv_default(value: str) -> list[str] | None:
    """Split a comma-separated config value into a list (``None`` when empty)."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def build_config_from_args(args: argparse.Namespace) -> AnomalyBenchmarkConfig:
    """Map parsed CLI arguments onto an :class:`AnomalyBenchmarkConfig`."""
    return AnomalyBenchmarkConfig(
        pollutant=args.pollutant,
        raw_base_dir=args.raw_base_dir,
        models=args.models if args.models else _csv_default(cfg_get_str("anomaly", "models", "all")),
        ensemble_method=args.ensemble_method,
        ensemble_top_k=args.top_k,
        device=args.device,
        seed=args.seed,
        eval_seed=args.eval_seed,
        min_series_points=args.min_series_points,
        series_limit=args.series_limit,
        output_dir=args.output_dir,
    )


def main() -> None:
    """CLI entry point: parse args, run the benchmark, print a JSON summary."""
    parser = argparse.ArgumentParser(description="Air-quality anomaly-detection benchmark")
    parser.add_argument("--pollutant", default=cfg_get_str("anomaly", "pollutant", "NO2"))
    parser.add_argument(
        "--raw-base-dir",
        default=cfg_get_str("anomaly", "raw_base_dir", "data/raw/datos_estaciones_5m"),
    )
    parser.add_argument("--models", nargs="*", default=None, help="Model names, or 'all' (default from config)")
    parser.add_argument("--ensemble-method", default=cfg_get_str("anomaly", "ensemble_method", DEFAULT_ENSEMBLE_METHOD))
    parser.add_argument("--top-k", type=int, default=cfg_get_int("anomaly", "ensemble_top_k", DEFAULT_TOP_K))
    parser.add_argument("--device", default=cfg_get_str("anomaly", "device", "cpu"))
    parser.add_argument("--seed", type=int, default=cfg_get_int("anomaly", "seed", 13), help="Selection-injection seed")
    parser.add_argument("--eval-seed", type=int, default=cfg_get_int("anomaly", "eval_seed", 101), help="Held-out evaluation-injection seed")
    parser.add_argument("--min-series-points", type=int, default=cfg_get_int("anomaly", "min_series_points", 600))
    parser.add_argument("--series-limit", type=int, default=None, help="Limit number of stations")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: reports/anomaly/<pollutant>_<ts>)")
    args = parser.parse_args()

    config = build_config_from_args(args)
    summary = run_benchmark(config)

    payload = {
        "output_dir": summary["output_dir"],
        "model_names": summary["model_names"],
        "macro_vus_pr": {name: summary["models"][name]["macro_metrics"]["vus_pr"] for name in summary["model_names"]},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
