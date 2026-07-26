"""Focused tests for mask-aware TSPulse anomaly reconstruction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

tspulse_module = pytest.importorskip("airquality.anomaly.models.tspulse")
TSPulse = tspulse_module.TSPulse
clear_tspulse_model_cache = tspulse_module.clear_tspulse_model_cache


class FakeTSPulse(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            context_length=512,
            patch_length=8,
            patch_stride=8,
            num_input_channels=1,
            mask_type="user",
        )
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, past_values, past_observed_mask, **kwargs):
        self.calls.append((past_values.detach().cpu(), past_observed_mask.detach().cpu()))
        position = torch.linspace(0.0, 1.0, 512, device=past_values.device).view(1, 512, 1)
        return {
            "reconstruction_outputs": past_values + position,
            "reconstructed_ts_from_fft": past_values + 2.0 * position,
        }


@pytest.fixture
def fake_checkpoint(monkeypatch):
    clear_tspulse_model_cache()
    model = FakeTSPulse()
    loads: list[dict[str, object]] = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return model

    monkeypatch.setattr(
        "airquality.anomaly.models.tspulse.TSPulseForReconstruction.from_pretrained",
        load,
    )
    yield model, loads
    clear_tspulse_model_cache()


def test_rejects_79_finite_points_and_loads_checkpoint_once(fake_checkpoint):
    _, loads = fake_checkpoint
    detector = TSPulse(device="cpu")

    with pytest.raises(ValueError, match="at least 80 finite points"):
        detector.fit(np.arange(79, dtype=np.float32))

    values = np.arange(80, dtype=np.float32)
    detector.fit(values)
    detector.fit(values)
    assert len(loads) == 1
    assert detector.minimum_series_length == 80


def test_checkpoint_is_shared_but_station_scalers_are_not(fake_checkpoint):
    model, loads = fake_checkpoint
    first = TSPulse(device="cpu").fit(np.arange(80, dtype=np.float32))
    second = TSPulse(device="cpu").fit(np.arange(80, dtype=np.float32) + 100.0)

    assert len(loads) == 1
    assert first.model_ is second.model_ is model
    assert first.scaler_ is not second.scaler_
    assert not np.array_equal(first.mean_, second.mean_)


def test_exactly_80_points_are_center_padded_and_patch_masked(fake_checkpoint):
    model, _ = fake_checkpoint
    values = np.arange(80, dtype=np.float32)
    detector = TSPulse(device="cpu", batch_size=3, smoothing_length=1).fit(values)

    scores = detector.score(values)

    assert scores.shape == (80,)
    assert np.isfinite(scores).all()
    assert model.calls
    for past_values, observed_mask in model.calls:
        assert past_values.shape[1:] == (512, 1)
        assert observed_mask.shape == past_values.shape
        assert torch.count_nonzero(past_values[:, :216]) == 0
        assert torch.count_nonzero(past_values[:, 296:]) == 0
        assert not observed_mask[:, :216].any()
        assert not observed_mask[:, 296:].any()
        assert torch.all(observed_mask.sum(dim=(1, 2)) == 72)
        for row in range(len(observed_mask)):
            target = np.flatnonzero(~observed_mask[row, 216:296, 0].numpy()) + 216
            assert np.array_equal(target, np.arange(target[0], target[0] + 8))
            assert target[0] % 8 == 0


def test_natural_nans_are_zero_masked_and_never_scored(fake_checkpoint):
    model, _ = fake_checkpoint
    values = np.arange(82, dtype=np.float32)
    values[[10, 30]] = np.nan
    detector = TSPulse(device="cpu", batch_size=256, smoothing_length=1).fit(values)

    scores = detector.score(values)

    assert scores.shape == values.shape
    assert np.isnan(scores[[10, 30]]).all()
    assert np.isfinite(np.delete(scores, [10, 30])).all()
    left_padding = (512 - len(values)) // 2
    expected = detector.scaler_.transform(values[:, None])[:, 0]
    for past_values, observed_mask in model.calls:
        assert torch.isfinite(past_values).all()
        for missing in (10, 30):
            position = left_padding + missing
            assert torch.all(past_values[:, position] == 0)
            assert not observed_mask[:, position].any()
        untouched = observed_mask[0, :, 0].numpy()
        actual = past_values[0, :, 0].numpy()
        original_positions = np.arange(left_padding, left_padding + len(values))
        comparable = original_positions[untouched[original_positions]]
        assert np.allclose(actual[comparable], expected[comparable - left_padding])


def test_long_series_keeps_original_alignment(fake_checkpoint):
    model, _ = fake_checkpoint
    values = np.linspace(-2.0, 3.0, 620, dtype=np.float32)
    detector = TSPulse(device="cpu", batch_size=128).fit(values)

    scores = detector.score(values)

    assert scores.shape == (620,)
    assert np.isfinite(scores).all()
    assert model.calls
    assert all(past_values.shape[1] == 512 for past_values, _ in model.calls)
