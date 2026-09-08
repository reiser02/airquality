"""Fill NaN gaps with one imputer or a model policy keyed by gap size.

Thin helper layered on top of the unified ``GapImputer.impute_gaps`` contract
(:mod:`airquality.imputation.imputers`). The forecasting pipeline can route
each complete gap to a concrete adapter from the imputer registry by inclusive
size ranges. Supported names include ``interp``, ``LinearInterp``, ``Prophet``,
a Darts model such as ``TiDE``, and both TSPulse variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

from airquality.data.series import ensure_datetime_series
from airquality.forecasting.cache import artifact_fingerprint, effective_config
from airquality.imputation.imputers import (
    GapImputer,
    InterpolationGapImputer,
    LinearGapImputer,
    ProphetGapImputer,
)
from airquality.imputation.registry import (
    DARTS_GLOBAL,
    INTERP,
    LINEAR,
    PROPHET,
    TSPULSE,
    TSPULSE_FINETUNED_MODEL_NAME,
    TSPULSE_ORIGINAL_MODEL_NAME,
    resolve_imputer_family,
)

DEFAULT_MAX_GAP_SIZE = 5


@dataclass(frozen=True, order=True)
class GapImputationRule:
    """Inclusive gap-size range assigned to one imputation model."""

    min_size: int
    max_size: int
    model_name: str

    @property
    def spec(self) -> str:
        """Return the normalized configuration representation of this rule."""
        size = (
            str(self.min_size)
            if self.min_size == self.max_size
            else f"{self.min_size}-{self.max_size}"
        )
        return f"{size}={self.model_name}"


@dataclass(frozen=True)
class GapImputationPolicy:
    """Validated model selection policy indexed by contiguous gap size."""

    rules: tuple[GapImputationRule, ...]

    def model_for(self, gap_size: int) -> str | None:
        """Return the configured model, or ``None`` when no rule covers the gap."""
        for rule in self.rules:
            if rule.min_size <= gap_size <= rule.max_size:
                return rule.model_name
        return None

    @property
    def model_names(self) -> tuple[str, ...]:
        """Return referenced models once, preserving normalized rule order."""
        return tuple(dict.fromkeys(rule.model_name for rule in self.rules))

    @property
    def spec(self) -> str:
        """Return a stable representation suitable for manifests and cache keys."""
        return ";".join(rule.spec for rule in self.rules)

@dataclass(frozen=True)
class GapImputationOutcome:
    """Observed result of applying one configured rule to one gap."""

    start: pd.Timestamp
    end: pd.Timestamp
    hours: int
    configured_imputer: str
    effective_imputer: str
    fallback_used: bool
    filled_hours: int


@dataclass(frozen=True)
class GapImputationResult:
    """Filled series together with per-gap effective-method diagnostics."""

    series: pd.Series
    outcomes: tuple[GapImputationOutcome, ...]


def parse_imputation_gap_rules(
    value: str,
) -> GapImputationPolicy:
    """Parse ``min-max=model`` rules separated by semicolons.

    Ranges are inclusive and may be discontinuous, but cannot overlap. At least
    one explicit rule is required.
    """
    raw = str(value).strip()
    if not raw:
        raise ValueError("imputation_gap_rules no puede estar vacio")

    rules: list[GapImputationRule] = []
    for item in raw.split(";"):
        match = re.fullmatch(
            r"\s*(\d+)(?:\s*-\s*(\d+))?\s*=\s*([^=;]+?)\s*", item
        )
        if match is None:
            raise ValueError(
                "formato invalido en imputation_gap_rules; usa min-max=model "
                "y separa las reglas con ';'"
            )
        min_size = int(match.group(1))
        max_size = int(match.group(2) or min_size)
        model_name = match.group(3).strip()
        if min_size < 1 or max_size < 1:
            raise ValueError("Los tamanos de gap deben ser positivos")
        if min_size > max_size:
            raise ValueError("El limite inicial del gap no puede superar el final")
        resolve_imputer_family(model_name)
        rules.append(GapImputationRule(min_size, max_size, model_name))

    rules.sort(key=lambda rule: (rule.min_size, rule.max_size, rule.model_name))
    for previous, current in zip(rules, rules[1:]):
        if current.min_size <= previous.max_size:
            raise ValueError(
                f"Las reglas de imputacion se solapan: {previous.spec} y {current.spec}"
            )
    return GapImputationPolicy(tuple(rules))


def _repo_root() -> Path:
    """Return the repository root (three levels above this module)."""
    return Path(__file__).resolve().parents[3]


def _resolve_tspulse_model_path(model_name: str) -> str | None:
    """Select the base or validated fine-tuned checkpoint for one TSPulse name."""
    if model_name == TSPULSE_ORIGINAL_MODEL_NAME:
        return None
    if model_name != TSPULSE_FINETUNED_MODEL_NAME:
        raise ValueError(f"Modelo TSPulse no soportado: {model_name}")

    from airquality.config import cfg_get_str

    configured_path = cfg_get_str("tspulse", "finetuned_model_path", "").strip()
    if not configured_path:
        raise RuntimeError(
            "No se puede usar TSPulse_FineTuned sin "
            "`[tspulse] finetuned_model_path`."
        )
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint TSPulse fine-tuned: {path}")
    return str(path.resolve())


def imputer_cache_identity(model_name: str, *, size_k: int) -> dict[str, object]:
    """Return config and local-artifact identity for one imputation model."""
    family = resolve_imputer_family(model_name)
    config: dict[str, object] = {
        "model": model_name,
        "family": family,
        "size_k": size_k,
    }
    artifacts: dict[str, str | None] = {}
    if family == DARTS_GLOBAL:
        weights = _repo_root() / "models" / f"{model_name}_k{size_k}.pt"
        artifacts = {
            "model": artifact_fingerprint(weights),
            "checkpoint": artifact_fingerprint(Path(f"{weights}.ckpt")),
        }
    elif family == TSPULSE:
        from airquality.config import cfg_get_int, cfg_get_str

        model_path = _resolve_tspulse_model_path(model_name)
        model_id = cfg_get_str(
            "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
        )
        config["tspulse"] = effective_config(
            {
                "model_path": model_path,
                "model_id": model_id,
                "revision": cfg_get_str(
                    "tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1"
                ),
                "context_length": cfg_get_int("tspulse", "context_length", 512),
                "device": cfg_get_str("tspulse", "device", "cpu"),
            }
        )
        local_source = Path(model_path or model_id).expanduser()
        if local_source.exists():
            artifacts["model"] = artifact_fingerprint(local_source)
    return {"config": config, "artifacts": artifacts}


def imputation_policy_cache_identity(
    policy: GapImputationPolicy, *, size_k: int
) -> dict[str, object]:
    """Return the complete cache identity of a gap-dependent policy."""
    return {
        "rules": [
            {
                "min_size": rule.min_size,
                "max_size": rule.max_size,
                "model": rule.model_name,
            }
            for rule in policy.rules
        ],
        "models": {
            model_name: imputer_cache_identity(model_name, size_k=size_k)
            for model_name in policy.model_names
        },
    }


def nan_gap_windows(series: pd.Series) -> list[pd.DatetimeIndex]:
    """Split the NaN positions of ``series`` into contiguous gap windows."""
    mask = series.isna().to_numpy()
    if not mask.any():
        return []
    index = pd.DatetimeIndex(series.index)
    windows: list[pd.DatetimeIndex] = []
    start: int | None = None
    for i, is_nan in enumerate(mask):
        if is_nan and start is None:
            start = i
        elif not is_nan and start is not None:
            windows.append(index[start:i])
            start = None
    if start is not None:
        windows.append(index[start:])
    return windows


def _fit_scaler(series: pd.Series, *, freq: str) -> Scaler:
    """Fit a Darts ``Scaler`` (StandardScaler) on the gap-filled observed series."""
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    s = s.interpolate(method="time", limit_direction="both").ffill().bfill()
    scaler = Scaler(scaler=StandardScaler(), global_fit=True)
    scaler.fit(TimeSeries.from_series(s, freq=freq).astype(np.float32))
    return scaler


def _gap_mask_index(gap_windows: Sequence[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Pool one non-empty sequence of gap windows into a single index."""
    return gap_windows[0].append(list(gap_windows[1:]))


def _impute_gap_windows(
    series: pd.Series,
    imputer: GapImputer,
    gap_windows: Sequence[pd.DatetimeIndex],
    *,
    configured_imputer: str,
    freq: str,
    use_scaler: bool,
) -> GapImputationResult:
    """Fill exactly ``gap_windows`` and leave every other missing window intact."""
    if not gap_windows:
        return GapImputationResult(series, ())
    fill_index = _gap_mask_index(gap_windows)
    scaler = _fit_scaler(series, freq=freq) if use_scaler else None
    pred, _failures = imputer.impute_gaps(
        series_name=str(series.name),
        all_series_map={str(series.name): series},
        gap_windows=gap_windows,
        test_index=pd.DatetimeIndex(series.index),
        scaler=scaler,
        freq=freq,
    )

    filled = series.copy()
    if len(pred) > 0:
        filled.loc[fill_index] = pred.reindex(fill_index)

    missing = fill_index[filled.reindex(fill_index).isna()]
    fallback = pd.Series(index=missing, dtype=float)
    if len(missing) > 0:
        # Preserve the existing safety net, but only for policy-selected gaps.
        fallback = InterpolationGapImputer()._fill(series, freq=freq)
        filled.loc[missing] = fallback.reindex(missing)

    outcomes = []
    for window in gap_windows:
        model_filled = int(pred.reindex(window).notna().sum())
        fallback_filled = int(
            fallback.reindex(window.intersection(missing)).notna().sum()
        )
        effective = []
        if model_filled:
            effective.append(configured_imputer)
        if fallback_filled and "interp" not in effective:
            effective.append("interp")
        outcomes.append(
            GapImputationOutcome(
                start=pd.Timestamp(window[0]),
                end=pd.Timestamp(window[-1]),
                hours=len(window),
                configured_imputer=configured_imputer,
                effective_imputer="+".join(effective) or "none",
                fallback_used=fallback_filled > 0,
                filled_hours=int(filled.reindex(window).notna().sum()),
            )
        )
    return GapImputationResult(filled, tuple(outcomes))


def build_imputer(
    model_name: str,
    *,
    freq: str = "h",
    size_k: int = 5,
    force_cpu: bool = True,
) -> GapImputer:
    """Construct the configured :class:`GapImputer` for one model name."""
    family = resolve_imputer_family(model_name)
    if family == INTERP:
        return InterpolationGapImputer(model_name=model_name)
    if family == LINEAR:
        return LinearGapImputer(model_name=model_name)
    if family == PROPHET:
        return ProphetGapImputer(model_name=model_name)
    if family == DARTS_GLOBAL:
        # Imported lazily: loading artifacts pulls in heavy Darts/torch machinery.
        from airquality.imputation.run_benchmark import load_darts_models_from_artifacts

        loaded = load_darts_models_from_artifacts(
            repo_root=_repo_root(),
            size_k=size_k,
            model_names=[model_name],
            force_cpu=force_cpu,
            strict=True,
        )
        return loaded[model_name]
    if family == TSPULSE:
        from airquality.config import cfg_get_int, cfg_get_str
        from airquality.imputation.imputers import TSPulseGapImputer

        return TSPulseGapImputer(
            model_id=cfg_get_str(
                "tspulse", "model_id", "ibm-granite/granite-timeseries-tspulse-r1"
            ),
            revision=cfg_get_str(
                "tspulse", "revision", "tspulse-hybrid-dualhead-512-p8-r1"
            ),
            model_path=_resolve_tspulse_model_path(model_name),
            context_length=cfg_get_int("tspulse", "context_length", 512),
            freq=freq,
            device=cfg_get_str("tspulse", "device", "cpu"),
            model_name=model_name,
        )
    raise ValueError(f"Familia de imputacion no soportada: {family}")


def impute_series(
    series: pd.Series,
    imputer: GapImputer,
    *,
    freq: str = "h",
    use_scaler: bool = False,
    max_gap_size: int = DEFAULT_MAX_GAP_SIZE,
) -> pd.Series:
    """Fill complete NaN windows no longer than ``max_gap_size``.

    Any eligible point the imputer leaves unfilled falls back to seasonal
    interpolation. Longer windows remain entirely NaN.
    """
    if max_gap_size < 1:
        raise ValueError("max_gap_size debe ser positivo")
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    gap_windows = [
        window for window in nan_gap_windows(s) if len(window) <= max_gap_size
    ]
    if not gap_windows:
        return s
    return _impute_gap_windows(
        s,
        imputer,
        gap_windows,
        configured_imputer=imputer.model_name,
        freq=freq,
        use_scaler=use_scaler,
    ).series.astype(float)


def impute_series_by_gap_result(
    series: pd.Series,
    policy: GapImputationPolicy,
    get_imputer: Callable[[str], GapImputer],
    *,
    freq: str = "h",
) -> GapImputationResult:
    """Route gaps by size and return both values and effective-method details."""
    s = ensure_datetime_series(series, freq=freq, name=str(series.name or "series"))
    windows_by_model: dict[str, list[pd.DatetimeIndex]] = {}
    outcomes: list[GapImputationOutcome] = []
    for window in nan_gap_windows(s):
        model_name = policy.model_for(len(window))
        if model_name is None:
            outcomes.append(
                GapImputationOutcome(
                    start=pd.Timestamp(window[0]),
                    end=pd.Timestamp(window[-1]),
                    hours=len(window),
                    configured_imputer="none",
                    effective_imputer="none",
                    fallback_used=False,
                    filled_hours=0,
                )
            )
        else:
            windows_by_model.setdefault(model_name, []).append(window)

    filled = s.copy()
    for model_name, gap_windows in windows_by_model.items():
        model_result = _impute_gap_windows(
            s,
            get_imputer(model_name),
            gap_windows,
            configured_imputer=model_name,
            freq=freq,
            use_scaler=resolve_imputer_family(model_name) == DARTS_GLOBAL,
        )
        fill_index = _gap_mask_index(gap_windows)
        filled.loc[fill_index] = model_result.series.reindex(fill_index)
        outcomes.extend(model_result.outcomes)
    outcomes.sort(key=lambda outcome: (outcome.start, outcome.end))
    return GapImputationResult(filled.astype(float), tuple(outcomes))


def impute_series_by_gap(
    series: pd.Series,
    policy: GapImputationPolicy,
    get_imputer: Callable[[str], GapImputer],
    *,
    freq: str = "h",
) -> pd.Series:
    """Route each complete NaN window to the model selected by its size."""
    return impute_series_by_gap_result(
        series,
        policy,
        get_imputer,
        freq=freq,
    ).series


__all__ = [
    "DEFAULT_MAX_GAP_SIZE",
    "GapImputationOutcome",
    "GapImputationPolicy",
    "GapImputationResult",
    "GapImputationRule",
    "nan_gap_windows",
    "parse_imputation_gap_rules",
    "imputer_cache_identity",
    "imputation_policy_cache_identity",
    "build_imputer",
    "impute_series",
    "impute_series_by_gap",
    "impute_series_by_gap_result",
]
