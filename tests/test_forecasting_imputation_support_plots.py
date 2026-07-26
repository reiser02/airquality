import pandas as pd

from airquality.forecasting.plot_imputation_support_analysis import render_plots


def test_render_plots_reads_training_support_csv(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "series": "ST1",
                "strategy": "unlabeled",
                "selected": True,
                "n_eligible_points_before_imputation": 100,
                "n_eligible_points_after_imputation": 130,
                "n_eligible_points_added": 30,
                "n_observed_points_recovered": 20,
                "n_imputed_in_eligible_blocks": 10,
                "imputed_eligible_age_hours_median": 48,
                "recovered_observed_age_hours_median": 72,
            },
            {
                "series": "ST2",
                "strategy": "inject-vote",
                "selected": True,
                "n_eligible_points_before_imputation": 80,
                "n_eligible_points_after_imputation": 100,
                "n_eligible_points_added": 20,
                "n_observed_points_recovered": 15,
                "n_imputed_in_eligible_blocks": 5,
                "imputed_eligible_age_hours_median": 96,
                "recovered_observed_age_hours_median": 120,
            },
        ]
    ).to_csv(tmp_path / "training_support.csv", index=False)

    paths = render_plots(tmp_path)

    assert {path.name for path in paths} == {
        "support_composition.png",
        "support_gain_by_series.png",
        "support_age.png",
        "support_age_timeline.png",
    }
    assert all(path.exists() for path in paths)


def test_render_plots_handles_zero_gain_and_missing_strategy_age(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "series": "ST1",
                "strategy": "unlabeled",
                "selected": True,
                "n_eligible_points_before_imputation": 100,
                "n_eligible_points_after_imputation": 100,
                "n_eligible_points_added": 0,
                "n_observed_points_recovered": 0,
                "n_imputed_in_eligible_blocks": 0,
                "imputed_eligible_age_hours_median": float("nan"),
                "recovered_observed_age_hours_median": float("nan"),
            },
            {
                "series": "ST1",
                "strategy": "inject-vote",
                "selected": True,
                "n_eligible_points_before_imputation": 100,
                "n_eligible_points_after_imputation": 100,
                "n_eligible_points_added": 0,
                "n_observed_points_recovered": 0,
                "n_imputed_in_eligible_blocks": 0,
                "imputed_eligible_age_hours_median": 48,
                "recovered_observed_age_hours_median": float("nan"),
            },
        ]
    ).to_csv(tmp_path / "training_support.csv", index=False)

    paths = render_plots(tmp_path)

    assert {path.name for path in paths} == {
        "support_composition.png",
        "support_gain_by_series.png",
        "support_age.png",
        "support_age_timeline.png",
    }
