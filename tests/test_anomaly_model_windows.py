"""Checks for anomaly detectors configured for the 80-point minimum block."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from airquality.anomaly.models import (
    CARLABase,
    CARLAGenIAS,
    COUTABase,
    COUTAGenIAS,
    LSTMAD,
    SubPCADetector,
)
from airquality.anomaly.models.carla import (
    PretextLoss,
    RepositoryAugmentedDataset,
    TSRepository,
)
from airquality.anomaly.models.common import pooled_windows_nd


def _legacy_build_train_repository_dataset(model, pretext_dataset):
    """Reference the removed CARLA repository flow for differential tests."""
    loader = DataLoader(pretext_dataset, batch_size=model.batch_size, shuffle=False, drop_last=False)
    repository_base = TSRepository(
        len(pretext_dataset), model.features_dim, model.num_clusters, model.temperature
    )
    repository_aug = TSRepository(
        len(pretext_dataset) * 2, model.features_dim, model.num_clusters, model.temperature
    )
    model.pretext_model.eval()
    repository_base.reset()
    repository_aug.reset()
    repository_base.resize(3)
    data_parts = []
    target_parts = []
    with torch.no_grad():
        for batch in loader:
            ts_org = batch["ts_org"].float().to(model.device)
            targets = batch["target"].to(model.device)
            outputs = model.pretext_model(ts_org.transpose(1, 2))
            repository_base.update(outputs, targets)
            repository_aug.update(outputs, targets)
            data_parts.append(ts_org.cpu())
            target_parts.append(targets.cpu())

            ts_w_augment = batch["ts_w_augment"].float().to(model.device)
            weak_targets = torch.full(
                (ts_w_augment.shape[0],), 2, dtype=torch.long, device=model.device
            )
            weak_outputs = model.pretext_model(ts_w_augment.transpose(1, 2))
            repository_base.update(weak_outputs, weak_targets)

            ts_ss_augment = batch["ts_ss_augment"].float().to(model.device)
            subseq_targets = torch.full(
                (ts_ss_augment.shape[0],), 4, dtype=torch.long, device=model.device
            )
            data_parts.append(ts_ss_augment.cpu())
            target_parts.append(subseq_targets.cpu())
            subseq_outputs = model.pretext_model(ts_ss_augment.transpose(1, 2))
            repository_base.update(subseq_outputs, subseq_targets)
            repository_aug.update(subseq_outputs, subseq_targets)

    repository_dataset = RepositoryAugmentedDataset(
        torch.cat(data_parts, dim=0), torch.cat(target_parts, dim=0)
    )
    furthest, nearest = repository_aug.furthest_nearest_neighbors(10)
    return repository_dataset, nearest, furthest


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


def test_carla_repository_keeps_only_original_and_subsequence_forwards():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, inputs):
            self.inputs.append(inputs.detach().clone())
            return inputs.mean(dim=(1, 2)).unsqueeze(1).repeat(1, 8)

    samples = []
    for index in range(3):
        original = torch.full((4, 1), float(index + 1))
        samples.append(
            {
                "ts_org": original,
                "ts_w_augment": original + 100.0,
                "ts_ss_augment": original + 10.0,
                "target": torch.tensor(0, dtype=torch.long),
            }
        )

    model = CARLABase(device="cpu", batch_size=2, features_dim=8)
    recorder = RecordingModel()
    model.pretext_model = recorder

    repository_dataset, nearest, furthest = model._build_train_repository_dataset(samples)
    legacy_model = CARLABase(device="cpu", batch_size=2, features_dim=8)
    legacy_recorder = RecordingModel()
    legacy_model.pretext_model = legacy_recorder
    legacy_dataset, legacy_nearest, legacy_furthest = _legacy_build_train_repository_dataset(
        legacy_model, samples
    )

    expected_inputs = [
        torch.stack([samples[0]["ts_org"], samples[1]["ts_org"]]).transpose(1, 2),
        torch.stack([samples[0]["ts_ss_augment"], samples[1]["ts_ss_augment"]]).transpose(1, 2),
        samples[2]["ts_org"].unsqueeze(0).transpose(1, 2),
        samples[2]["ts_ss_augment"].unsqueeze(0).transpose(1, 2),
    ]
    assert len(recorder.inputs) == len(expected_inputs)
    for actual, expected in zip(recorder.inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual, expected)
    assert len(legacy_recorder.inputs) == 6

    expected_data = torch.stack(
        [
            samples[0]["ts_org"],
            samples[1]["ts_org"],
            samples[0]["ts_ss_augment"],
            samples[1]["ts_ss_augment"],
            samples[2]["ts_org"],
            samples[2]["ts_ss_augment"],
        ]
    )
    torch.testing.assert_close(repository_dataset.data, expected_data)
    assert repository_dataset.targets.tolist() == [0, 0, 4, 4, 0, 4]
    torch.testing.assert_close(repository_dataset.data, legacy_dataset.data)
    torch.testing.assert_close(repository_dataset.targets, legacy_dataset.targets)
    np.testing.assert_array_equal(nearest, legacy_nearest)
    np.testing.assert_array_equal(furthest, legacy_furthest)
    assert nearest.shape == (6, 5)
    assert furthest.shape == (6, 5)
    assert not recorder.training


def test_carla_refactor_matches_legacy_fit_and_score():
    class LegacyCARLA(CARLABase):
        def _build_train_repository_dataset(self, pretext_dataset):
            return _legacy_build_train_repository_dataset(self, pretext_dataset)

    kwargs = {
        "device": "cpu",
        "pretext_epochs": 1,
        "classification_epochs": 1,
        "batch_size": 64,
        "features_dim": 8,
        "mid_channels": 2,
        "num_clusters": 2,
        "num_heads": 1,
        "num_neighbors": 1,
        "seed": 23,
    }
    values = np.sin(np.arange(80, dtype=np.float32) / 8.0)
    current = CARLABase(**kwargs).fit(values)
    legacy = LegacyCARLA(**kwargs).fit(values)

    np.testing.assert_array_equal(current.score(values), legacy.score(values))
    assert current.selected_head_ == legacy.selected_head_
    assert current.majority_label_ == legacy.majority_label_
    assert current.training_summary_ == legacy.training_summary_
    for current_state, legacy_state in zip(
        current.classification_model.state_dict().values(),
        legacy.classification_model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(current_state, legacy_state, rtol=0.0, atol=0.0)


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
