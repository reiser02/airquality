"""Checks for anomaly detectors configured for the 80-point minimum block."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from airquality.anomaly.models import (
    CARLABase,
    CARLAGenIAS,
    COUTABase,
    COUTAGenIAS,
    LSTMAD,
    SubPCADetector,
)
from airquality.anomaly.models.carla import PretextLoss
from airquality.anomaly.models.common import pooled_windows_nd


@pytest.mark.parametrize(
    "model_cls",
    [COUTABase, COUTAGenIAS, CARLABase, CARLAGenIAS],
)
def test_couta_and_carla_variants_default_to_80_point_windows(model_cls):
    model = model_cls(device="cpu")

    assert model.window_size == 80
    assert model.minimum_series_length == 80


def test_sub_pca_and_lstmad_report_effective_minimums():
    sub_pca = SubPCADetector(device="cpu", window_size=80)
    lstmad = LSTMAD(device="cpu")

    assert sub_pca.window_size == 80
    assert sub_pca.minimum_series_length == 81
    assert lstmad.window_size == 65
    assert lstmad.horizon == 8
    assert lstmad.minimum_series_length == 80


def test_lstmad_fits_once_across_80_point_segments():
    x = np.arange(80, dtype=np.float32)
    segments = [np.sin(x / 8.0), np.cos(x / 9.0)]
    model = LSTMAD(
        device="cpu",
        hidden_size=4,
        num_layers=1,
        batch_size=16,
        num_epochs=1,
    )

    scores = model.fit_segments(segments).score_segments(segments)

    assert all(score is not None for score in scores)
    assert all(score.shape == segment.shape for score, segment in zip(scores, segments, strict=True))
    assert all(np.isfinite(score[72]) for score in scores)
    assert all(np.isnan(score[np.arange(80) != 72]).all() for score in scores)


def test_carla_pretext_loss_accepts_partial_training_batch():
    features = torch.randn(3, 8, requires_grad=True)

    loss = PretextLoss(batch_size=64)(features)

    assert torch.isfinite(loss)
    loss.backward()


def test_carla_honors_explicit_training_stride():
    model = CARLABase(device="cpu", stride=5, max_windows=100)

    assert model._resolve_training_stride(250) == 5
    assert model._resolve_training_stride(1000) == 10


def test_pooled_windows_do_not_cross_segment_boundaries():
    windows = pooled_windows_nd(
        [
            np.zeros((4, 1), dtype=np.float32),
            np.ones((4, 1), dtype=np.float32),
        ],
        window_size=3,
        stride=1,
    )

    assert windows.shape == (4, 3, 1)
    assert all(np.unique(window).size == 1 for window in windows)


def test_sub_pca_abstains_on_short_segments() -> None:
    model = SubPCADetector(device="cpu", window_size=80)
    long_segments = [
        np.sin(np.arange(81, dtype=np.float32) / 8.0),
        np.cos(np.arange(80, dtype=np.float32) / 9.0),
    ]

    model.fit_segments(long_segments)
    scores = model.score_segments([np.arange(79, dtype=np.float32), long_segments[1]])

    assert scores[0] is None
    assert scores[1] is None


def test_couta_base_fits_one_80_point_window():
    values = np.sin(np.arange(80, dtype=np.float32) / 8.0)
    model = COUTABase(device="cpu", num_epochs=1, batch_size=64)

    scores = model.fit(values).score(values)

    assert scores.shape == values.shape
    assert np.isnan(scores[:-1]).all()
    assert np.isfinite(scores[-1])


def test_carla_base_fits_one_80_point_window():
    values = np.sin(np.arange(80, dtype=np.float32) / 8.0)
    model = CARLABase(
        device="cpu",
        pretext_epochs=1,
        classification_epochs=1,
        batch_size=64,
        features_dim=8,
        mid_channels=2,
        num_clusters=2,
        num_heads=1,
        num_neighbors=1,
    )

    scores = model.fit(values).score(values)

    assert scores.shape == values.shape
    assert np.isfinite(scores).all()
