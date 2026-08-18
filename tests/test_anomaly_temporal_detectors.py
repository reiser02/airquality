"""Tests for Sub_PCA, offline Hampel, and timestamp-aware Prophet."""

from __future__ import annotations

import numpy as np
import pandas as pd

import airquality.anomaly.models.prophet_detector as prophet_module
from airquality.anomaly.models.baselines import Hampel6Detector, HampelDetector
from airquality.anomaly.models.sub_pca_detector import (
    SubPCADetector,
    find_length_rank,
)
from airquality.anomaly.registry import resolve_model_names


def test_sub_pca_resolves_the_dominant_period() -> None:
    values = np.sin(2.0 * np.pi * np.arange(600, dtype=float) / 24.0)

    assert find_length_rank(values) == 24

    model = SubPCADetector(device="cpu")
    model.fit(values)

    assert model.window_size == 24
    assert model.training_summary_["periodicity"] == 1
    assert model.score(values).shape == values.shape


def test_sub_pca_uses_tsb_ad_padding_and_prunes_constant_columns() -> None:
    values = np.sin(2.0 * np.pi * np.arange(25, dtype=float) / 8.0).astype(np.float32)
    model = SubPCADetector(device="cpu", window_size=8).fit(values)

    scores = model.score(values)

    assert scores.shape == values.shape
    assert np.isfinite(scores).all()
    assert model.training_summary_["n_active_columns"] == 8
    assert model.weighted is True
    assert model.training_summary_["weighted"] is True


def test_sub_pca_can_select_a_low_variance_component_subset() -> None:
    values = np.sin(2.0 * np.pi * np.arange(40, dtype=float) / 8.0).astype(np.float32)

    model = SubPCADetector(
        device="cpu", window_size=8, n_selected_components=4, weighted=True
    ).fit(values)

    assert model.training_summary_["n_selected_components"] == 4


def test_sub_pca_constant_series_has_zero_scores() -> None:
    values = np.ones(20, dtype=np.float32)

    scores = SubPCADetector(device="cpu", window_size=8).fit(values).score(values)

    assert np.array_equal(scores, np.zeros_like(values))


def test_sub_pca_is_the_canonical_registry_name() -> None:
    names = resolve_model_names(["all"])

    assert "Sub_PCA" in names
    assert names.count("Sub_PCA") == 1


def test_hampel_offline_requires_a_complete_window() -> None:
    values = np.arange(24, dtype=np.float32)
    model = HampelDetector(device="cpu")

    scores = model.fit(values).score(values)

    assert np.isnan(scores[:12]).all()
    assert np.isnan(scores[-11:]).all()
    assert np.isfinite(scores[12])
    assert model.score_segments([np.arange(23, dtype=np.float32)]) == [None]


def test_hampel_short_variant_has_expected_complete_support() -> None:
    values = np.arange(10, dtype=np.float32)
    scores = Hampel6Detector(device="cpu").fit(values).score(values)

    assert np.isnan(scores[:3]).all()
    assert np.isnan(scores[-2:]).all()
    assert np.isfinite(scores[3:8]).all()


class _FakeProphet:
    fit_frames: list[pd.DataFrame] = []
    predict_frames: list[pd.DataFrame] = []

    def __init__(self, *, interval_width: float) -> None:
        self.interval_width = interval_width
        self.history_dates = None

    def fit(self, frame: pd.DataFrame, seed: int) -> "_FakeProphet":
        self.fit_frames.append(frame.copy())
        self.history_dates = frame["ds"]
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.predict_frames.append(frame.copy())
        return pd.DataFrame(
            {
                "yhat_lower": np.full(len(frame), -1.0),
                "yhat_upper": np.full(len(frame), 1.0),
            }
        )


def test_prophet_uses_real_hourly_segment_timestamps(monkeypatch) -> None:
    _FakeProphet.fit_frames = []
    _FakeProphet.predict_frames = []
    monkeypatch.setattr(prophet_module, "Prophet", _FakeProphet)
    index = pd.date_range("2024-03-02 05:00", periods=8, freq="h")
    values = np.linspace(0.0, 0.7, len(index), dtype=np.float32)

    detector = prophet_module.ProphetDetector(device="cpu")
    detector.fit_segments([values], segment_indices=[index])
    detector.score_segments([values])

    fit_index = pd.DatetimeIndex(_FakeProphet.fit_frames[0]["ds"])
    predict_index = pd.DatetimeIndex(_FakeProphet.predict_frames[0]["ds"])
    assert fit_index.equals(index)
    assert predict_index.equals(index)
    assert fit_index[1] - fit_index[0] == pd.Timedelta(hours=1)
    assert detector.training_summary_["freq"] == "h"
    assert detector.training_summary_["real_timestamps"] is True
