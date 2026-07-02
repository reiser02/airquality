"""Imputer registry mapping model names to their construction family.

Mirrors the role of ``anomaly/registry.py``: it validates requested imputation
model names and classifies each into a *family* (`darts_global`, `prophet`,
`tspulse`). The actual construction happens in ``run_benchmark.py``, which owns
the runtime configuration (artifact paths, devices, TSPulse settings).

Optional families are gated by dependency availability: ``Prophet`` requires
``darts.models.Prophet`` and ``TSPulse``/``TSPulse_FineTuned`` require
``tsfm_public``. Such names stay "known but unavailable" and are skipped with a
warning rather than raising an unknown-name error.
"""

from __future__ import annotations

import warnings

from airquality.modeling.training_config import build_model_configs
from airquality.imputation.imputers import (
    PROPHET_AVAILABLE,
    PROPHET_IMPORT_ERROR,
    TSFM_PUBLIC_AVAILABLE,
    TSFM_PUBLIC_IMPORT_ERROR,
)

# Imputer families.
DARTS_GLOBAL = "darts_global"
PROPHET = "prophet"
TSPULSE = "tspulse"

PROPHET_MODEL_NAME = "Prophet"
TSPULSE_ORIGINAL_MODEL_NAME = "TSPulse"
TSPULSE_FINETUNED_MODEL_NAME = "TSPulse_FineTuned"


def darts_global_model_names() -> list[str]:
    """Return the Darts global forecasters known to the project model catalog."""
    return list(build_model_configs().keys())


# name -> family for every known imputer (independent of dependency availability).
def _build_known_families() -> dict[str, str]:
    families: dict[str, str] = {name: DARTS_GLOBAL for name in darts_global_model_names()}
    families[PROPHET_MODEL_NAME] = PROPHET
    families[TSPULSE_ORIGINAL_MODEL_NAME] = TSPULSE
    families[TSPULSE_FINETUNED_MODEL_NAME] = TSPULSE
    return families


IMPUTER_FAMILIES: dict[str, str] = _build_known_families()

# Optional names gated by an import flag and its captured error.
_OPTIONAL_AVAILABILITY: dict[str, tuple[bool, Exception | None]] = {
    PROPHET_MODEL_NAME: (PROPHET_AVAILABLE, PROPHET_IMPORT_ERROR),
    TSPULSE_ORIGINAL_MODEL_NAME: (TSFM_PUBLIC_AVAILABLE, TSFM_PUBLIC_IMPORT_ERROR),
    TSPULSE_FINETUNED_MODEL_NAME: (TSFM_PUBLIC_AVAILABLE, TSFM_PUBLIC_IMPORT_ERROR),
}


def resolve_imputer_family(name: str) -> str:
    """Return the construction family for one imputer name."""
    try:
        return IMPUTER_FAMILIES[name]
    except KeyError as exc:
        raise ValueError(f"Modelo de imputacion desconocido: {name}") from exc


def available_imputer_names() -> list[str]:
    """Return every known imputer whose optional dependency (if any) is present."""
    names: list[str] = []
    for name in IMPUTER_FAMILIES:
        available, _ = _OPTIONAL_AVAILABILITY.get(name, (True, None))
        if available:
            names.append(name)
    return names


def resolve_imputer_names(model_names: list[str] | None) -> list[str]:
    """Validate requested names, expanding ``["all"]`` and skipping unavailable optionals."""
    if not model_names or (len(model_names) == 1 and model_names[0].lower() == "all"):
        return available_imputer_names()

    requested = list(model_names)
    resolved: list[str] = []
    unknown: list[str] = []
    for name in requested:
        if name not in IMPUTER_FAMILIES:
            unknown.append(name)
            continue
        available, import_error = _OPTIONAL_AVAILABILITY.get(name, (True, None))
        if not available:
            warnings.warn(
                f"Skipping {name}: optional dependency unavailable ({import_error}).",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        resolved.append(name)

    if unknown:
        raise ValueError(f"Unknown imputer name(s): {', '.join(unknown)}")
    return resolved


__all__ = [
    "DARTS_GLOBAL",
    "PROPHET",
    "TSPULSE",
    "PROPHET_MODEL_NAME",
    "TSPULSE_ORIGINAL_MODEL_NAME",
    "TSPULSE_FINETUNED_MODEL_NAME",
    "IMPUTER_FAMILIES",
    "darts_global_model_names",
    "available_imputer_names",
    "resolve_imputer_family",
    "resolve_imputer_names",
]
