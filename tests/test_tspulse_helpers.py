from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airquality.imputation.tspulse_finetune import (
    build_series_name,
    sanitize_name,
    split_long_train_valid,
)


def test_sanitize_name_normalizes_spaces_and_symbols() -> None:
    assert sanitize_name("  Calle Real #1  ") == "Calle_Real__1"


def test_build_series_name_avoids_duplicate_value_col_when_in_stem() -> None:
    p = Path("/tmp/estacion/Aquatec_NO2.csv")
    assert build_series_name(p, "NO2") == "estacion__Aquatec_NO2"


def test_split_long_train_valid_generates_contextual_validation() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "id": ["S"] * n,
            "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
            "y": list(range(n)),
        }
    )

    train, valid = split_long_train_valid(
        df,
        id_column="id",
        timestamp_column="ts",
        valid_fraction=0.2,
        context_length=2,
    )

    assert len(train) == 8
    assert len(valid) == 4


def test_split_long_train_valid_validates_fraction() -> None:
    df = pd.DataFrame({"id": ["S", "S"], "ts": pd.date_range("2024-01-01", periods=2, freq="h")})
    with pytest.raises(ValueError):
        split_long_train_valid(
            df,
            id_column="id",
            timestamp_column="ts",
            valid_fraction=1.0,
            context_length=2,
        )
