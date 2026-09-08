from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import airquality.visualizations.block_analysis as block_plots


def test_block_length_distribution_uses_one_axis_with_pollutant_legend(
    tmp_path, monkeypatch
) -> None:
    blocks = pd.DataFrame(
        {
            "pollutant": ["NO2", "NO2", "O3", "O3"],
            "hours": [10, 100, 20, 200],
        }
    )
    series = pd.DataFrame(
        {
            "pollutant": ["NO2", "O3"],
            "forecast_models": ["NLinear, TiDE", "NLinear, TiDE"],
            "limiting_models": ["TCN", "TCN"],
            "minimum_hours": [84, 84],
            "host_minimum_hours": [192, 192],
        }
    )
    captured: dict[str, plt.Figure] = {}
    monkeypatch.setattr(
        block_plots.plt,
        "close",
        lambda figure: captured.setdefault("figure", figure),
    )

    block_plots.save_block_length_distribution(
        tmp_path / "block_length_distribution.png", blocks, series
    )

    figure = captured["figure"]
    assert len(figure.axes) == 1
    legend_labels = set(figure.axes[0].get_legend_handles_labels()[1])
    assert {"NO2", "O3"}.issubset(legend_labels)
    assert len(figure.axes[0].patches) == 88
    assert all(
        patch.get_alpha() == block_plots.HISTOGRAM_ALPHA
        for patch in figure.axes[0].patches
    )
    figure.clear()
