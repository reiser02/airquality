import numpy as np
import pandas as pd

from airquality.data.plot_block_support_analysis import render_plots


def test_render_plots_reads_training_support_csv(tmp_path) -> None:
    rows = []
    for regime, minimum in (("short", 80), ("long", 120)):
        for arm, stage, strategy, valid, imputed in (
            ("raw", "raw", "none", 100, 0),
            ("unlabeled+noimpute", "detected", "unlabeled", 80, 0),
            ("unlabeled+impute", "imputed", "unlabeled", 110, 10),
        ):
            rows.append(
                {
                    "series": "ST1",
                    "arm": arm,
                    "stage": stage,
                    "strategy": strategy,
                    "regime": regime,
                    "minimum_hours": minimum,
                    "host_minimum_hours": minimum + 48,
                    "observed_hours": 120,
                    "total_blocks": 3,
                    "valid_blocks": 2,
                    "valid_hours": valid,
                    "valid_real_hours": valid - imputed,
                    "valid_imputed_hours": imputed,
                    "valid_imputed_anomaly_hours": imputed // 2,
                    "valid_imputed_preexisting_gap_hours": imputed - imputed // 2,
                    "valid_hours_pct_raw": valid,
                    "imputation_gain_valid_hours": 30 if stage == "imputed" else 0,
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
                "strategy": "none" if arm == "raw" else "unlabeled",
            }
            for arm, hours in (
                ("raw", 150),
                ("unlabeled+noimpute", 90),
                ("unlabeled+impute", 160),
            )
        ]
    ).to_csv(tmp_path / "blocks.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy": "unlabeled",
                "observed_hours": 200,
                "scored_hours": 190,
                "flagged_hours_full": 5,
            }
        ]
    ).to_csv(tmp_path / "detection.csv", index=False)

    paths = render_plots(tmp_path)

    assert {path.name for path in paths} == {
        "retention_overview.png",
        "block_length_distribution.png",
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
    assert "## Figuras" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_render_plots_handles_zero_gain_and_missing_strategy_age(tmp_path) -> None:
    rows = []
    for regime, minimum in (("short", 80), ("long", 120)):
        for arm, stage, strategy in (
            ("raw", "raw", "none"),
            ("unlabeled+noimpute", "detected", "unlabeled"),
            ("unlabeled+impute", "imputed", "unlabeled"),
        ):
            rows.append(
                {
                    "series": "ST1",
                    "arm": arm,
                    "stage": stage,
                    "strategy": strategy,
                    "regime": regime,
                    "minimum_hours": minimum,
                    "host_minimum_hours": minimum + 48,
                    "observed_hours": 100,
                    "total_blocks": 1,
                    "valid_blocks": 1,
                    "valid_hours": 100,
                    "valid_real_hours": 100,
                    "valid_imputed_hours": 0,
                    "valid_imputed_anomaly_hours": 0,
                    "valid_imputed_preexisting_gap_hours": 0,
                    "valid_hours_pct_raw": 100,
                    "imputation_gain_valid_hours": 0,
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
                "strategy": "none" if arm == "raw" else "unlabeled",
            }
            for arm in pd.DataFrame(rows)["arm"].unique()
        ]
    ).to_csv(tmp_path / "blocks.csv", index=False)
    pd.DataFrame(
        [{"strategy": "unlabeled", "observed_hours": 100, "scored_hours": 100, "flagged_hours_full": 0}]
    ).to_csv(tmp_path / "detection.csv", index=False)

    paths = render_plots(tmp_path)

    assert {path.name for path in paths} == {
        "retention_overview.png",
        "block_length_distribution.png",
        "detected_block_length_distribution.png",
        "support_overview.png",
        "support_by_series.png",
        "valid_blocks_by_series.png",
        "block_length_survival.png",
        "gap_recovery.png",
        "detection_strategy_summary.png",
    }
