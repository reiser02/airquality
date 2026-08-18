from __future__ import annotations

import pandas as pd
import pytest

from airquality.data.holdout import (
    build_holdout_manifest,
    manifests_match,
    select_retrospective_holdouts,
)


def _frame(name: str, length: int, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=length, freq="h")
    return pd.DataFrame({name: range(length)}, index=index, dtype=float)


def test_select_retrospective_holdouts_requires_fixed_target() -> None:
    target = _frame("target", 500)

    holdouts, metadata = select_retrospective_holdouts(
        [target],
        target_points=192,
        context_points=72,
        min_train_points=125,
    )

    assert len(holdouts["target"]) == 192
    by_name = metadata.set_index("Serie")
    assert by_name.loc["target", "Test_Block_Points"] == 192


def test_select_retrospective_holdouts_rejects_station_without_target() -> None:
    with pytest.raises(ValueError, match="short"):
        select_retrospective_holdouts(
            [_frame("short", 136)],
            target_points=192,
            context_points=72,
            min_train_points=125,
        )


def test_holdout_manifest_identity_changes_with_split() -> None:
    metadata = pd.DataFrame(
        {
            "Serie": ["A"],
            "Test_Start": [pd.Timestamp("2024-01-10")],
            "Test_End": [pd.Timestamp("2024-01-11")],
            "Test_Block_Points": [25],
        }
    )
    first = build_holdout_manifest(
        metadata,
        target_points=25,
        context_points=5,
        min_train_points=20,
        freq="h",
    )
    changed = build_holdout_manifest(
        metadata.assign(Test_End=pd.Timestamp("2024-01-12")),
        target_points=25,
        context_points=5,
        min_train_points=20,
        freq="h",
    )

    assert manifests_match(first, first)
    assert not manifests_match(first, changed)
