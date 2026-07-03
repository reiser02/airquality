"""CLI entrypoint: ``uv run python -m airquality.anomaly.run``.

Defaults come from the ``[anomaly]`` section of the project config; any argument
overrides its config default.
"""

from __future__ import annotations

import argparse
import json

from airquality.config import cfg_get_float, cfg_get_int, cfg_get_str

from .benchmark import AnomalyBenchmarkConfig, run_benchmark
from .ensemble import DEFAULT_ENSEMBLE_METHOD
from .metrics import DEFAULT_MAX_DETECTION_RATE, DEFAULT_THRESHOLD_K


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
        device=args.device,
        seed=args.seed,
        threshold_k=args.threshold_k,
        max_detection_rate=args.max_detection_rate,
        min_series_points=args.min_series_points,
        series_limit=args.series_limit,
        output_dir=args.output_dir,
    )


def main() -> None:
    """CLI entry point: parse args, run the benchmark, print a JSON summary."""
    parser = argparse.ArgumentParser(description="Air-quality anomaly-detection benchmark (label-free)")
    parser.add_argument("--pollutant", default=cfg_get_str("anomaly", "pollutant", "NO2"))
    parser.add_argument(
        "--raw-base-dir",
        default=cfg_get_str("anomaly", "raw_base_dir", "data/raw/datos_estaciones_5m"),
    )
    parser.add_argument("--models", nargs="*", default=None, help="Model names, or 'all' (default from config)")
    parser.add_argument("--ensemble-method", default=cfg_get_str("anomaly", "ensemble_method", DEFAULT_ENSEMBLE_METHOD))
    parser.add_argument("--device", default=cfg_get_str("anomaly", "device", "cpu"))
    parser.add_argument("--seed", type=int, default=cfg_get_int("anomaly", "seed", 13))
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=cfg_get_float("anomaly", "threshold_k", DEFAULT_THRESHOLD_K),
        help="k of the median + k*MAD score-binarization threshold",
    )
    parser.add_argument(
        "--max-detection-rate",
        type=float,
        default=cfg_get_float("anomaly", "max_detection_rate", DEFAULT_MAX_DETECTION_RATE),
        help="Discard detectors whose macro detection rate exceeds this fraction",
    )
    parser.add_argument("--min-series-points", type=int, default=cfg_get_int("anomaly", "min_series_points", 600))
    parser.add_argument("--series-limit", type=int, default=None, help="Limit number of stations")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: reports/anomaly/<pollutant>_<ts>)")
    args = parser.parse_args()

    config = build_config_from_args(args)
    summary = run_benchmark(config)

    payload = {
        "output_dir": summary["output_dir"],
        "model_names": summary["model_names"],
        "kept_models": summary["kept_models"],
        "discarded_models": summary["discarded_models"],
        "macro_detection_rate": {
            name: summary["models"][name]["macro_metrics"]["detection_rate"]
            for name in summary["model_names"]
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
