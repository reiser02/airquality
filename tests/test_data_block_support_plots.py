import json

import numpy as np
import pandas as pd
import pytest

import airquality.visualizations.block_support as block_support
from airquality.visualizations.block_support import render_plots


def test_gap_recovery_reconciles_support_before_and_after_imputation() -> None:
    table = pd.DataFrame(
        [
            {
                "arm": "inject-vote+impute",
                "stage": "imputed",
                "effective_training_hours": 110,
                "effective_imputed_hours": 10,
                "imputation_gain_effective_hours": 30,
            }
        ]
    )

    totals = block_support._gap_recovery_totals(table).iloc[0]

    assert totals["valid_before"] == 80
    assert totals["imputed_valid"] == 10
    assert totals["observed_unlocked"] == 20
    assert totals["effective_training_hours"] == 110


def test_effective_imputation_age_uses_only_training_points() -> None:
    table = pd.DataFrame(
        [
            {
                "series": "ST1",
                "stage": "imputed",
                "strategy": "inject-vote",
                "test_target_start": "2025-03-15 00:00:00",
                "validation_hours": 24,
                "effective_imputed_hours": 2,
                "imputation_gain_effective_hours": 3,
                "imputed_valid_age_hours_median": 100,
            }
        ]
    )
    blocks = pd.DataFrame(
        [
            {
                "series": "ST1",
                "stage": "imputed",
                "strategy": "inject-vote",
                "start": "2025-01-01 00:00:00",
                "end": "2025-03-14 23:00:00",
                "used": True,
                "validation_host": True,
            },
            {
                "series": "ST1",
                "stage": "detected",
                "strategy": "inject-vote",
                "start": "2025-01-01 00:00:00",
                "end": "2025-01-01 23:00:00",
                "used": True,
                "validation_host": False,
            },
            {
                "series": "ST1",
                "stage": "detected",
                "strategy": "inject-vote",
                "start": "2025-01-02 01:00:00",
                "end": "2025-01-02 23:00:00",
                "used": True,
                "validation_host": False,
            },
            {
                "series": "ST1",
                "stage": "detected",
                "strategy": "inject-vote",
                "start": "2025-01-03 01:00:00",
                "end": "2025-02-28 23:00:00",
                "used": True,
                "validation_host": False,
            },
            {
                "series": "ST1",
                "stage": "detected",
                "strategy": "inject-vote",
                "start": "2025-03-01 01:00:00",
                "end": "2025-03-01 23:00:00",
                "used": True,
                "validation_host": False,
            },
            {
                "series": "ST1",
                "stage": "detected",
                "strategy": "inject-vote",
                "start": "2025-03-02 00:00:00",
                "end": "2025-03-13 23:00:00",
                "used": True,
                "validation_host": False,
            },
        ]
    )
    gaps = pd.DataFrame(
        [
            {
                "series": "ST1",
                "strategy": "inject-vote",
                "start": "2025-01-02 00:00:00",
                "end": "2025-01-02 00:00:00",
                "origin": "preexisting",
                "fully_filled": True,
            },
            {
                "series": "ST1",
                "strategy": "inject-vote",
                "start": "2025-03-01 00:00:00",
                "end": "2025-03-01 00:00:00",
                "origin": "anomaly",
                "fully_filled": True,
            },
            {
                "series": "ST1",
                "strategy": "inject-vote",
                "start": "2025-03-14 12:00:00",
                "end": "2025-03-14 12:00:00",
                "origin": "anomaly",
                "fully_filled": True,
            },
        ]
    )

    points = block_support._effective_imputation_age_points(table, blocks, gaps)

    assert points["support_type"].value_counts().to_dict() == {
        "imputed": 2,
        "recovered": 1,
    }
    assert set(points.loc[points["support_type"] == "imputed", "age_bin"].astype(str)) == {
        "31-90 d",
        "0-30 d",
    }


def test_render_plots_reads_training_support_csv(tmp_path, monkeypatch) -> None:
    rows = []
    for arm, stage, strategy, valid, imputed in (
        ("raw", "raw", "none", 100, 0),
        ("inject-vote+noimpute", "detected", "inject-vote", 80, 0),
        ("inject-vote+impute", "imputed", "inject-vote", 110, 10),
        ("unlabeled+noimpute", "detected", "unlabeled", 70, 0),
        ("unlabeled+impute", "imputed", "unlabeled", 105, 5),
    ):
        rows.append(
            {
                "series": "ST1",
                "test_target_start": "2025-01-10 00:00:00",
                "arm": arm,
                "stage": stage,
                "strategy": strategy,
                "minimum_hours": 80,
                "host_minimum_hours": 128,
                "observed_hours": 120,
                "total_blocks": 3,
                "valid_blocks": 2,
                "valid_hours": valid,
                "valid_real_hours": valid - imputed,
                "valid_imputed_hours": imputed,
                "valid_imputed_anomaly_hours": imputed // 2,
                "valid_imputed_preexisting_gap_hours": imputed - imputed // 2,
                "effective_imputed_hours": imputed,
                "valid_hours_pct_raw": valid,
                "imputation_gain_valid_hours": 30 if stage == "imputed" else 0,
                "imputation_gain_effective_hours": 30 if stage == "imputed" else 0,
                "imputed_valid_age_hours_median": 96 if stage == "imputed" else np.nan,
                "used_blocks": 2,
                "effective_training_hours": valid - 8,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "series_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": arm,
                "hours": hours,
                "stage": "raw" if arm == "raw" else "detected" if "noimpute" in arm else "imputed",
                "strategy": "none" if arm == "raw" else arm.split("+")[0],
            }
            for arm, hours in (
                ("raw", 150),
                ("inject-vote+noimpute", 90),
                ("inject-vote+impute", 160),
                ("unlabeled+noimpute", 85),
                ("unlabeled+impute", 145),
            )
        ]
    ).to_csv(tmp_path / "blocks.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy": strategy,
                "observed_hours": 200,
                "scored_hours": 190,
                "flagged_hours_full": 5,
            }
            for strategy in ("inject-vote", "unlabeled")
        ]
    ).to_csv(tmp_path / "detection.csv", index=False)
    pd.DataFrame(
        columns=["series", "strategy", "start", "end", "origin", "fully_filled"]
    ).to_csv(tmp_path / "gaps.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"pollutant": "NO2"}), encoding="utf-8"
    )

    captured = {}

    def capture_support(path, table, _pollutant):
        captured["support"] = set(table["strategy"])
        path.touch()
        return True

    def capture_detection(path, table, _pollutant):
        captured["detection"] = set(table["strategy"])
        path.touch()
        return True

    def capture_age(path, _table, _blocks, _gaps, _pollutant):
        path.touch()
        return True

    monkeypatch.setattr(block_support, "_save_support_overview", capture_support)
    monkeypatch.setattr(block_support, "_save_detection_summary", capture_detection)
    monkeypatch.setattr(block_support, "_save_imputation_age", capture_age)
    paths = render_plots(tmp_path)

    assert {path.name for path in paths} == {
        "retention_overview.png",
        "block_length_distribution.png",
        "retention_overview_detected.png",
        "retention_overview_imputed.png",
        "block_length_distribution_detected.png",
        "block_length_distribution_imputed.png",
        "retained_hours_by_series_detected.png",
        "retained_hours_by_series_imputed.png",
        "usable_blocks_by_series_detected.png",
        "usable_blocks_by_series_imputed.png",
        "detected_block_length_distribution.png",
        "support_overview.png",
        "support_by_series.png",
        "valid_blocks_by_series.png",
        "block_length_survival.png",
        "gap_recovery.png",
        "detection_strategy_summary.png",
        "imputation_age.png",
    }
    assert all(path.exists() for path in paths)
    assert captured == {
        "support": {"none", "inject-vote"},
        "detection": {"inject-vote"},
    }
    assert "## Figuras" in (tmp_path / "README.md").read_text(encoding="utf-8")

    legacy = pd.DataFrame(rows).assign(regime="short")
    legacy.to_csv(tmp_path / "series_summary.csv", index=False)
    with pytest.raises(ValueError, match="esquema antiguo con regímenes"):
        render_plots(tmp_path)


def test_render_plots_handles_zero_gain_and_missing_strategy_age(tmp_path) -> None:
    rows = []
    for arm, stage, strategy in (
        ("raw", "raw", "none"),
        ("inject-vote+noimpute", "detected", "inject-vote"),
        ("inject-vote+impute", "imputed", "inject-vote"),
    ):
        rows.append(
            {
                "series": "ST1",
                "test_target_start": "2025-01-10 00:00:00",
                "arm": arm,
                "stage": stage,
                "strategy": strategy,
                "minimum_hours": 80,
                "host_minimum_hours": 128,
                "observed_hours": 100,
                "total_blocks": 1,
                "valid_blocks": 1,
                "valid_hours": 100,
                "valid_real_hours": 100,
                "valid_imputed_hours": 0,
                "valid_imputed_anomaly_hours": 0,
                "valid_imputed_preexisting_gap_hours": 0,
                "effective_imputed_hours": 0,
                "valid_hours_pct_raw": 100,
                "imputation_gain_valid_hours": 0,
                "imputation_gain_effective_hours": 0,
                "imputed_valid_age_hours_median": float("nan"),
                "used_blocks": 1,
                "effective_training_hours": 90,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "series_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": arm,
                "hours": 100,
                "stage": "raw" if arm == "raw" else "detected" if "noimpute" in arm else "imputed",
                "strategy": "none" if arm == "raw" else "inject-vote",
            }
            for arm in pd.DataFrame(rows)["arm"].unique()
        ]
    ).to_csv(tmp_path / "blocks.csv", index=False)
    pd.DataFrame(
        [{"strategy": "inject-vote", "observed_hours": 100, "scored_hours": 100, "flagged_hours_full": 0}]
    ).to_csv(tmp_path / "detection.csv", index=False)
    pd.DataFrame(
        columns=["series", "strategy", "start", "end", "origin", "fully_filled"]
    ).to_csv(tmp_path / "gaps.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"pollutant": "CO"}), encoding="utf-8"
    )

    paths = render_plots(tmp_path)

    assert {path.name for path in paths} == {
        "retention_overview.png",
        "block_length_distribution.png",
        "retention_overview_detected.png",
        "retention_overview_imputed.png",
        "block_length_distribution_detected.png",
        "block_length_distribution_imputed.png",
        "retained_hours_by_series_detected.png",
        "retained_hours_by_series_imputed.png",
        "usable_blocks_by_series_detected.png",
        "usable_blocks_by_series_imputed.png",
        "detected_block_length_distribution.png",
        "support_overview.png",
        "support_by_series.png",
        "valid_blocks_by_series.png",
        "block_length_survival.png",
        "gap_recovery.png",
        "detection_strategy_summary.png",
    }
