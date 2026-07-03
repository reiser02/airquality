"""CLI entrypoint: ``uv run python -m airquality.anomaly.run``.

Defaults come from the ``[anomaly]`` section of the project config; any argument
overrides its config default. ``--mode unlabeled`` (default) screens detectors
label-free by detection rate; ``--mode synthetic`` scores them against injected
anomalies (VUS-PR et al.).
"""

from __future__ import annotations

import argparse
import json

from airquality.config import cfg_get_float, cfg_get_int, cfg_get_str

from .benchmark import AnomalyBenchmarkConfig, run_benchmark
from .ensemble import DEFAULT_ENSEMBLE_METHOD, DEFAULT_TOP_K
from .metrics import DEFAULT_MAX_DETECTION_RATE, DEFAULT_THRESHOLD_K


def _csv_default(value: str) -> list[str] | None:
    """Split a comma-separated config value into a list (``None`` when empty)."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def build_config_from_args(args: argparse.Namespace) -> AnomalyBenchmarkConfig:
    """Map parsed CLI arguments onto an :class:`AnomalyBenchmarkConfig`."""
    return AnomalyBenchmarkConfig(
        mode=args.mode,
        pollutant=args.pollutant,
        raw_base_dir=args.raw_base_dir,
        models=args.models if args.models else _csv_default(cfg_get_str("anomaly", "models", "all")),
        ensemble_method=args.ensemble_method,
        device=args.device,
        seed=args.seed,
        threshold_k=args.threshold_k,
        max_detection_rate=args.max_detection_rate,
        eval_seed=args.eval_seed,
        ensemble_top_k=args.top_k,
        min_series_points=args.min_series_points,
        series_limit=args.series_limit,
        output_dir=args.output_dir,
    )


def main() -> None:
    """CLI entry point: parse args, run the benchmark, print a JSON summary."""
    parser = argparse.ArgumentParser(description="Air-quality anomaly-detection benchmark")
    parser.add_argument(
        "--mode",
        choices=("unlabeled", "synthetic"),
        default=cfg_get_str("anomaly", "mode", "unlabeled"),
        help="unlabeled: label-free detection-rate screening; synthetic: injected-anomaly metrics",
    )
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
        help="[unlabeled] k of the median + k*MAD score-binarization threshold",
    )
    parser.add_argument(
        "--max-detection-rate",
        type=float,
        default=cfg_get_float("anomaly", "max_detection_rate", DEFAULT_MAX_DETECTION_RATE),
        help="[unlabeled] discard detectors whose macro detection rate exceeds this fraction",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=cfg_get_int("anomaly", "eval_seed", 101),
        help="[synthetic] held-out evaluation-injection seed",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=cfg_get_int("anomaly", "ensemble_top_k", DEFAULT_TOP_K),
        help="[synthetic] ensemble size (ranked by selection VUS-PR)",
    )
    parser.add_argument("--min-series-points", type=int, default=cfg_get_int("anomaly", "min_series_points", 600))
    parser.add_argument("--series-limit", type=int, default=None, help="Limit number of stations")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: reports/anomaly/<pollutant>_<ts>)")
    args = parser.parse_args()

    config = build_config_from_args(args)
    summary = run_benchmark(config)

    headline = "vus_pr" if summary["mode"] == "synthetic" else "detection_rate"
    payload = {
        "output_dir": summary["output_dir"],
        "mode": summary["mode"],
        "model_names": summary["model_names"],
        f"macro_{headline}": {
            name: summary["models"][name]["macro_metrics"][headline]
            for name in summary["model_names"]
        },
    }
    if summary["mode"] == "unlabeled":
        payload["kept_models"] = summary["kept_models"]
        payload["discarded_models"] = summary["discarded_models"]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
