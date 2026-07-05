"""Detector registry for the anomaly pipeline.

Extracted from the original genias ``benchmark.py`` ``MODEL_REGISTRY`` (the
multi-GPU/ProcessPool machinery is intentionally dropped — the pipeline runs the
detectors sequentially). ``TSPulse`` is registered only when its optional
``tsfm_public`` dependency imports cleanly.
"""

from __future__ import annotations

import warnings

from .models import (
    CARLABase,
    CARLAGenIAS,
    COUTABase,
    COUTAGenIAS,
    Hampel6Detector,
    HampelDetector,
    IQRDetector,
    IsolationForestDetector,
    LOFDetector,
    LSTMAD,
    ModifiedZScoreDetector,
    PCADetector,
    ProphetDetector,
)
from .models import TSPULSE_AVAILABLE, TSPULSE_IMPORT_ERROR, TSPulse

MODEL_REGISTRY: dict[str, type] = {
    "ModifiedZScore": ModifiedZScoreDetector,
    "IQR": IQRDetector,
    "IsolationForest": IsolationForestDetector,
    "LOF": LOFDetector,
    "PCA": PCADetector,
    "COUTABase": COUTABase,
    "COUTAGenIAS": COUTAGenIAS,
    "CARLABase": CARLABase,
    "CARLAGenIAS": CARLAGenIAS,
    "LSTMAD": LSTMAD,
    "Hampel_w24": HampelDetector,
    "Hampel_w6": Hampel6Detector,
    "Prophet": ProphetDetector,
}

if TSPULSE_AVAILABLE:
    MODEL_REGISTRY["TSPulse"] = TSPulse


def resolve_model_class(model_name: str) -> type:
    """Return the detector class registered under ``model_name`` (KeyError if unknown)."""
    return MODEL_REGISTRY[model_name]


def resolve_model_names(model_names: list[str] | None) -> list[str]:
    """Validate requested model names, expanding ``["all"]`` to the full registry."""
    if not model_names or (len(model_names) == 1 and model_names[0].lower() == "all"):
        requested = list(MODEL_REGISTRY)
    else:
        requested = list(model_names)

    unknown = [name for name in requested if name not in MODEL_REGISTRY]
    if unknown:
        # TSPulse is the only name that can be "known but unavailable".
        if "TSPulse" in unknown and not TSPULSE_AVAILABLE:
            warnings.warn(
                f"Skipping TSPulse: optional dependency unavailable ({TSPULSE_IMPORT_ERROR}).",
                RuntimeWarning,
                stacklevel=2,
            )
            unknown = [name for name in unknown if name != "TSPulse"]
            requested = [name for name in requested if name != "TSPulse"]
        if unknown:
            raise ValueError(f"Unknown model name(s): {', '.join(unknown)}")
    return requested
