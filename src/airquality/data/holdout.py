"""Station-local retrospective holdouts shared by training and evaluation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from airquality.data.segments import contiguous_observed_segments


HOLDOUT_PROTOCOL = "per_series_retrospective_v1"
EXCLUSION_REASON = "no_fixed_test_or_training_host"

EXCLUSION_COLUMNS = [
    "Serie",
    "Reason_Code",
    "Reason",
    "Observed_Points",
    "Observed_Segments",
    "Longest_Observed_Segment",
    "Required_Holdout_Points",
    "Required_Context_Points",
    "Required_Min_Train_Points",
    "Require_Post_Data",
]

HOLDOUT_METADATA_COLUMNS = [
    "Serie",
    "Test_Start",
    "Test_End",
    "Test_Block_Points",
    "Target_Test_Points",
    "Context_Points",
    "Train_Points_Before",
    "Train_Points_After",
    "Source_Block_Start",
    "Source_Block_End",
    "Source_Block_Points",
]


def _single_series(frame: pd.DataFrame) -> tuple[str, pd.Series]:
    if len(frame.columns) != 1:
        raise ValueError("Cada serie debe tener exactamente una columna.")
    name = str(frame.columns[0])
    series = frame.iloc[:, 0].astype(float).copy()
    series.name = name
    return name, series


def _candidate_starts(block_length: int, holdout_points: int, context_points: int) -> list[int]:
    first = int(context_points)
    last = int(block_length) - int(holdout_points)
    if last < first:
        return []
    return sorted({first, (first + last) // 2, last})


def _select_retrospective_holdouts(
    series_dfs: Sequence[pd.DataFrame],
    *,
    target_points: int,
    context_points: int,
    min_train_points: int,
    require_post_data: bool = True,
) -> tuple[dict[str, pd.Series], pd.DataFrame, pd.DataFrame]:
    """Select one fixed-size station-local block while retaining training data.

    Every station must provide a complete block of ``target_points``. Candidate
    values are never inspected; selection depends only on observed support and
    timestamps. Failing loudly keeps the training and benchmark station panels
    identical instead of mixing different holdout sizes.
    """
    target = int(target_points)
    context = int(context_points)
    min_train = int(min_train_points)
    if target <= 0 or context < 0 or min_train <= 0:
        raise ValueError("Las longitudes de holdout/train deben ser positivas.")

    holdouts: dict[str, pd.Series] = {}
    metadata_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    for frame in series_dfs:
        name, series = _single_series(frame)
        observed_segments = contiguous_observed_segments(series)
        selected: tuple[tuple[Any, ...], pd.Series, pd.Series] | None = None
        holdout_candidates = 0
        candidates_with_training = 0
        candidates_without_post_data = 0

        candidates: list[tuple[tuple[Any, ...], pd.Series, pd.Series]] = []
        for block in observed_segments:
            for start_pos in _candidate_starts(len(block), target, context):
                holdout = block.iloc[start_pos : start_pos + target].copy()
                if len(holdout) != target:
                    continue
                holdout_candidates += 1

                masked = series.copy()
                masked.loc[holdout.index] = np.nan
                remaining = contiguous_observed_segments(masked, min_len=min_train)
                if not remaining:
                    continue
                candidates_with_training += 1

                before = series.loc[series.index < holdout.index[0]].notna()
                after = series.loc[series.index > holdout.index[-1]].notna()
                before_points = int(before.sum())
                after_points = int(after.sum())
                if require_post_data and after_points == 0:
                    candidates_without_post_data += 1
                    continue

                before_longest = max(
                    (len(part) for part in remaining if part.index[-1] < holdout.index[0]),
                    default=0,
                )
                after_longest = max(
                    (len(part) for part in remaining if part.index[0] > holdout.index[-1]),
                    default=0,
                )
                # Prefer support on both sides, then balanced train volume and
                # finally the most recent candidate for deterministic ties.
                score = (
                    int(before_longest >= min_train and after_longest >= min_train),
                    min(before_points, after_points),
                    before_longest + after_longest,
                    len(block),
                    holdout.index[-1].value,
                )
                candidates.append((score, holdout, block))

        if candidates:
            selected = max(candidates, key=lambda item: item[0])

        if selected is None:
            longest_segment = max((len(block) for block in observed_segments), default=0)
            if not observed_segments:
                reason_code = EXCLUSION_REASON
                reason = "La serie no contiene puntos observados."
            elif holdout_candidates == 0:
                reason_code = EXCLUSION_REASON
                reason = (
                    "Ningún segmento observado alcanza el contexto y el holdout "
                    f"requeridos ({context} + {target} puntos)."
                )
            elif candidates_with_training == 0:
                reason_code = EXCLUSION_REASON
                reason = (
                    "Tras enmascarar el holdout no queda un segmento de entrenamiento "
                    f"de al menos {min_train} puntos."
                )
            elif require_post_data and candidates_without_post_data == candidates_with_training:
                reason_code = EXCLUSION_REASON
                reason = "No quedan observaciones posteriores al holdout candidato."
            else:
                reason_code = EXCLUSION_REASON
                reason = "No se encontró ningún candidato que cumpla el protocolo de holdout."
            exclusion_rows.append(
                {
                    "Serie": name,
                    "Reason_Code": reason_code,
                    "Reason": reason,
                    "Observed_Points": int(series.notna().sum()),
                    "Observed_Segments": int(len(observed_segments)),
                    "Longest_Observed_Segment": int(longest_segment),
                    "Required_Holdout_Points": target,
                    "Required_Context_Points": context,
                    "Required_Min_Train_Points": min_train,
                    "Require_Post_Data": bool(require_post_data),
                }
            )
            continue

        _, holdout, source_block = selected
        holdout.name = name
        holdouts[name] = holdout
        before_values = series.loc[series.index < holdout.index[0]]
        after_values = series.loc[series.index > holdout.index[-1]]
        metadata_rows.append(
            {
                "Serie": name,
                "Test_Start": holdout.index[0],
                "Test_End": holdout.index[-1],
                "Test_Block_Points": int(len(holdout)),
                "Target_Test_Points": target,
                "Context_Points": context,
                "Train_Points_Before": int(before_values.notna().sum()),
                "Train_Points_After": int(after_values.notna().sum()),
                "Source_Block_Start": source_block.index[0],
                "Source_Block_End": source_block.index[-1],
                "Source_Block_Points": int(len(source_block)),
            }
        )

    metadata = pd.DataFrame(metadata_rows, columns=HOLDOUT_METADATA_COLUMNS)
    if not metadata.empty:
        metadata = metadata.sort_values("Serie").reset_index(drop=True)
    exclusions = pd.DataFrame(exclusion_rows, columns=EXCLUSION_COLUMNS)
    if not exclusions.empty:
        exclusions = exclusions.sort_values("Serie").reset_index(drop=True)
    return holdouts, metadata, exclusions


def select_retrospective_holdouts(
    series_dfs: Sequence[pd.DataFrame],
    *,
    target_points: int,
    context_points: int,
    min_train_points: int,
    require_post_data: bool = True,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Select holdouts and fail if any input series is ineligible."""
    holdouts, metadata, exclusions = _select_retrospective_holdouts(
        series_dfs,
        target_points=target_points,
        context_points=context_points,
        min_train_points=min_train_points,
        require_post_data=require_post_data,
    )
    if not exclusions.empty:
        raise ValueError(
            "No se pudo seleccionar un holdout retrospectivo para: "
            + ", ".join(exclusions["Serie"].astype(str))
        )
    if not holdouts:
        raise ValueError("No se pudo seleccionar ningun holdout retrospectivo.")
    return holdouts, metadata


def select_retrospective_holdouts_with_exclusions(
    series_dfs: Sequence[pd.DataFrame],
    *,
    target_points: int,
    context_points: int,
    min_train_points: int,
    require_post_data: bool = True,
) -> tuple[dict[str, pd.Series], pd.DataFrame, pd.DataFrame]:
    """Select holdouts while reporting and omitting ineligible series."""
    holdouts, metadata, exclusions = _select_retrospective_holdouts(
        series_dfs,
        target_points=target_points,
        context_points=context_points,
        min_train_points=min_train_points,
        require_post_data=require_post_data,
    )
    if not holdouts:
        names = ", ".join(exclusions["Serie"].astype(str))
        raise ValueError(
            "No se pudo seleccionar ningun holdout retrospectivo; "
            f"series descartadas: {names}."
        )
    return holdouts, metadata, exclusions


def build_holdout_manifest(
    metadata: pd.DataFrame,
    *,
    target_points: int,
    context_points: int,
    min_train_points: int,
    freq: str,
    darts_models: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a stable JSON manifest identifying one retrospective split."""
    required = {"Serie", "Test_Start", "Test_End", "Test_Block_Points"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Faltan columnas de holdout: {sorted(missing)}")

    stations = []
    for _, row in metadata.sort_values("Serie").iterrows():
        stations.append(
            {
                "series": str(row["Serie"]),
                "test_start": pd.Timestamp(row["Test_Start"]).isoformat(),
                "test_end": pd.Timestamp(row["Test_End"]).isoformat(),
                "test_points": int(row["Test_Block_Points"]),
            }
        )
    identity = {
        "protocol": HOLDOUT_PROTOCOL,
        "frequency": str(freq),
        "target_points": int(target_points),
        "context_points": int(context_points),
        "min_train_points": int(min_train_points),
        "stations": stations,
    }
    split_id = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **identity,
        "split_id": split_id,
        "darts_models": sorted({str(name) for name in darts_models}),
    }


def write_holdout_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    """Persist a holdout manifest as human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_holdout_manifest(path: Path) -> dict[str, Any]:
    """Read and minimally validate a holdout manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("split_id"):
        raise ValueError(f"Manifest de holdout invalido: {path}")
    return payload


def manifests_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Return whether two artifacts use the same station-local split."""
    return str(expected.get("split_id", "")) == str(actual.get("split_id", ""))
