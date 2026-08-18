"""Detector registry for the anomaly pipeline.

Extracted from the original genias ``benchmark.py`` ``MODEL_REGISTRY`` (the
multi-GPU/ProcessPool machinery is intentionally dropped — the pipeline runs the
detectors sequentially). ``TSPulse`` is registered only when its optional
``tsfm_public`` dependency imports cleanly.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
import warnings

import numpy as np

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
    SubPCADetector,
    ProphetDetector,
)
from .models import TSPULSE_AVAILABLE, TSPULSE_IMPORT_ERROR, TSPulse

MODEL_REGISTRY: dict[str, type] = {
    "ModifiedZScore": ModifiedZScoreDetector,
    "IQR": IQRDetector,
    "IsolationForest": IsolationForestDetector,
    "LOF": LOFDetector,
    "Sub_PCA": SubPCADetector,
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
    """Return the detector class registered under ``model_name``."""
    return MODEL_REGISTRY[model_name]


def filter_model_kwargs(model_cls: type, kwargs: dict[str, object]) -> dict[str, object]:
    """Keep only kwargs the detector's ``__init__`` accepts (e.g. drop ``device``).

    Shared by the anomaly benchmark and the production cleaning so both build
    detectors from the same registry with the same construction rule.
    """
    parameters = inspect.signature(model_cls.__init__).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    valid = set(parameters) - {"self"}
    return {key: value for key, value in kwargs.items() if key in valid}


def fit_model_segments(
    model_cls: type,
    segments: list[np.ndarray],
    *,
    seed: int,
    model_kwargs: dict[str, object] | None = None,
    segment_indices: Sequence[object] | None = None,
):
    """Create one detector and fit it once on all station segments.

    Timestamp metadata is passed only to detectors that explicitly request it,
    so the common detector API remains array-based for every other model.
    """

    model = model_cls(seed=seed, **(model_kwargs or {}))
    fit_segments = model.fit_segments
    parameters = inspect.signature(fit_segments).parameters
    accepts_indices = "segment_indices" in parameters
    if segment_indices is not None and accepts_indices:
        fit_segments(segments, segment_indices=segment_indices)
    else:
        fit_segments(segments)
    return model


def score_model_segments(
    model, segments: list[np.ndarray]
) -> list[np.ndarray | None]:
    """Score station segments without refitting or crossing their boundaries."""
    score_segments = getattr(model, "score_segments", None)
    if score_segments is not None:
        return list(score_segments(segments))
    scores: list[np.ndarray | None] = []
    for segment in segments:
        try:
            scores.append(np.asarray(model.score(segment), dtype=float))
        except Exception:
            scores.append(None)
    return scores


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
