"""Tests for the forecasting-benchmark figure builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airquality.visualizations.forecasting import (
    family_colors,
    improvement_table,
    imputation_pairs,
    render_run_figures,
    save_arm_error_plot,
    save_detector_selection_plot,
    save_foundation_preprocessing_plot,
    save_imputation_effect_plot,
    save_improvement_heatmap,
    save_inference_time_plot,
    save_inference_time_vs_error_plot,
    save_train_time_plot,
    save_train_time_vs_error_plot,
    selection_counts,
)

ARMS = [
    "raw",
    "raw+frozen",
    "unlabeled+impute",
    "unlabeled+noimpute",
    "inject-vote+impute",
    "inject-vote+noimpute",
]


def _results_df(n_series: int = 3, models: tuple[str, ...] = ("NLinear", "TiDE")) -> pd.DataFrame:
    # All persisted forecasting metrics are dimensionless.
    rng = np.random.default_rng(7)
    rows = []
    for s in range(n_series):
        for model in models:
            base = 1.0 + rng.uniform(0, 0.5)
            for arm in ARMS:
                family = arm.split("+", 1)[0]
                is_raw_view = arm in {"raw", "raw+frozen"}
                rows.append(
                    {
                        "series": f"ST{s}",
                        "arm": arm,
                        "strategy": "none" if is_raw_view else family,
                        "imputed": arm.endswith("+impute"),
                        "imputation_policy": "1-5=interp" if arm.endswith("+impute") else "none",
                        "detectors": "" if is_raw_view else "IQR,Hampel_w24",
                        "n_anomalies": 0 if is_raw_view else 5,
                        "model": model,
                        "rmsse": base * (1 + rng.uniform(-0.3, 0.3)),
                        "mase": base * (1 + rng.uniform(-0.2, 0.2)),
                        "relmae": 1.0 if arm == "raw" else base * (1 + rng.uniform(-0.2, 0.2)),
                        "relrmse": 1.0 if arm == "raw" else base * (1 + rng.uniform(-0.3, 0.3)),
                        "train_seconds": 2.0 + rng.uniform(0, 8),
                        "inference_seconds": 0.1 + rng.uniform(0, 0.5),
                        "n_test_predictions": 40,
                    }
                )
    return pd.DataFrame(rows)


def _detection_df(n_series: int = 3) -> pd.DataFrame:
    rows = []
    for s in range(n_series):
        rows.append(
            {
                "series": f"ST{s}",
                "strategy": "unlabeled",
                "detectors": "IQR,Hampel_w24,ModifiedZScore",
                "discarded": "LOF",
                "n_flagged": 12,
                "detection_rate": 0.012 + 0.002 * s,
                "ranking": "",
            }
        )
        rows.append(
            {
                "series": f"ST{s}",
                "strategy": "inject-vote",
                "detectors": "ModifiedZScore,IQR" if s else "Hampel_w24,IQR",
                "discarded": "",
                "n_flagged": 9,
                "detection_rate": 0.010,
                "ranking": "IQR=0.512;Hampel_w24=0.431",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Data preparation helpers
# --------------------------------------------------------------------------- #
def test_improvement_table_is_relative_to_raw():
    df = _results_df(n_series=1, models=("NLinear",))
    df.loc[(df["arm"] == "raw") & (df["model"] == "NLinear"), "relrmse"] = 1.0
    df.loc[(df["arm"] == "unlabeled+impute") & (df["model"] == "NLinear"), "relrmse"] = 0.8

    table = improvement_table(df, "relrmse", "NLinear")

    assert "raw" not in table.columns
    assert "raw+frozen" in table.columns
    assert table.loc["ST0", "unlabeled+impute"] == pytest.approx(20.0)  # 20% better


def test_improvement_table_keeps_all_nan_arm():
    df = _results_df(n_series=1, models=("NLinear",))
    df.loc[df["arm"] == "raw+frozen", "relrmse"] = np.nan

    table = improvement_table(df, "relrmse", "NLinear")

    assert table["raw+frozen"].isna().all()


def test_selection_counts_orders_and_counts():
    counts = selection_counts(_detection_df(n_series=3))
    assert counts.loc["IQR", "unlabeled"] == 3
    assert counts.loc["IQR", "inject-vote"] == 3
    assert counts.loc["Hampel_w24", "inject-vote"] == 1
    assert list(counts.columns) == ["unlabeled", "inject-vote"]


def test_imputation_pairs_requires_both_variants():
    df = _results_df(n_series=2, models=("NLinear",))
    pairs = imputation_pairs(df, "mase")
    assert set(pairs["strategy"]) == {"unlabeled", "inject-vote"}
    assert not pairs["strategy"].str.startswith("raw").any()
    assert {"impute", "noimpute"} <= set(pairs.columns)
    # Without the noimpute arms there is nothing to pair.
    assert imputation_pairs(df[~df["arm"].str.endswith("+noimpute")], "mase").empty


def test_family_colors_fixed_and_extended():
    colors = family_colors(ARMS + ["inject-soft+impute", "custom+impute"])
    assert colors["raw"] == "#6d6258"
    assert colors["raw+frozen"] == "#2a7f76"
    assert colors["unlabeled"] == "#3d7ab5"
    assert colors["inject-soft"] == "#5b8c5a"
    assert colors["custom"] not in ("", None)


# --------------------------------------------------------------------------- #
# Figure smoke tests (files exist and are non-trivial)
# --------------------------------------------------------------------------- #
def test_save_figures_smoke(tmp_path):
    results_df = _results_df()
    detection_df = _detection_df()

    outputs = {
        "arm": tmp_path / "arm.png",
        "heat": tmp_path / "heat.png",
        "sel": tmp_path / "sel.png",
        "imp": tmp_path / "imp.png",
    }
    outputs["time"] = tmp_path / "time.png"
    outputs["cost"] = tmp_path / "cost.png"
    outputs["inf_time"] = tmp_path / "inf_time.png"
    outputs["inf_cost"] = tmp_path / "inf_cost.png"
    assert save_arm_error_plot(outputs["arm"], results_df, "mase")
    assert save_improvement_heatmap(outputs["heat"], results_df, "relrmse")
    assert save_detector_selection_plot(outputs["sel"], detection_df)
    assert save_imputation_effect_plot(outputs["imp"], results_df, "mase")
    assert save_train_time_plot(outputs["time"], results_df)
    assert save_train_time_vs_error_plot(outputs["cost"], results_df, "rmsse")
    assert save_inference_time_plot(outputs["inf_time"], results_df)
    assert save_inference_time_vs_error_plot(outputs["inf_cost"], results_df, "relmae")
    for path in outputs.values():
        assert path.exists() and path.stat().st_size > 5_000


def test_save_figures_report_empty_inputs(tmp_path):
    empty = pd.DataFrame(
        columns=[
            "series", "arm", "strategy", "imputed", "model",
            "rmsse", "mase", "relmae", "relrmse",
        ]
    )
    assert not save_arm_error_plot(tmp_path / "a.png", empty)
    assert not save_improvement_heatmap(tmp_path / "b.png", empty, "relrmse")
    assert not save_detector_selection_plot(tmp_path / "c.png", pd.DataFrame())
    assert not save_imputation_effect_plot(tmp_path / "d.png", empty)
    assert not save_train_time_plot(tmp_path / "e.png", empty)
    assert not save_train_time_vs_error_plot(tmp_path / "f.png", empty, "rmsse")
    assert not save_inference_time_plot(tmp_path / "g.png", empty)
    assert not save_inference_time_vs_error_plot(tmp_path / "h.png", empty, "relmae")
    assert not save_foundation_preprocessing_plot(tmp_path / "i.png", pd.DataFrame())
    assert not list(tmp_path.iterdir())


def test_save_foundation_preprocessing_plot(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["Chronos2"] * 4,
            "strategy": ["unlabeled"] * 4,
            "anomaly_type": ["spikes", "scale", "noise", "drift"],
            "rmsse_recovery": [0.2, -0.1, 0.3, 0.05],
        }
    )
    output = tmp_path / "foundation.png"

    assert save_foundation_preprocessing_plot(output, summary, "rmsse")
    assert output.exists() and output.stat().st_size > 5_000


def test_render_run_figures_from_csvs(tmp_path):
    _results_df().to_csv(tmp_path / "results.csv", index=False)
    _detection_df().to_csv(tmp_path / "detection.csv", index=False)
    pd.DataFrame(
        {
            "model": ["Chronos2"] * 4,
            "strategy": ["unlabeled"] * 4,
            "anomaly_type": ["spikes", "scale", "noise", "drift"],
            "rmsse_recovery": [0.2, -0.1, 0.3, 0.05],
            "mase_recovery": [0.1, -0.05, 0.2, 0.02],
        }
    ).to_csv(tmp_path / "foundation_preprocessing_summary.csv", index=False)

    saved = render_run_figures(tmp_path)

    names = {path.name for path in saved}
    assert names == {
        "arm_error_rmsse.png",
        "arm_error_mase.png",
        "arm_error_relmae.png",
        "arm_error_relrmse.png",
        "improvement_rmsse.png",
        "improvement_mase.png",
        "improvement_relmae.png",
        "improvement_relrmse.png",
        "detector_selection.png",
        "imputation_effect_rmsse.png",
        "imputation_effect_mase.png",
        "imputation_effect_relmae.png",
        "imputation_effect_relrmse.png",
        "train_time.png",
        "train_time_vs_rmsse.png",
        "train_time_vs_mase.png",
        "train_time_vs_relmae.png",
        "train_time_vs_relrmse.png",
        "inference_time.png",
        "inference_time_vs_rmsse.png",
        "inference_time_vs_mase.png",
        "inference_time_vs_relmae.png",
        "inference_time_vs_relrmse.png",
        "foundation_preprocessing_recovery_mase.png",
        "foundation_preprocessing_recovery_rmsse.png",
    }

    legacy = _results_df()
    legacy["regime"] = "short"
    legacy.to_csv(tmp_path / "results.csv", index=False)
    with pytest.raises(ValueError, match="esquema antiguo con regímenes"):
        render_run_figures(tmp_path)
