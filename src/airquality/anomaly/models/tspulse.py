"""Zero-shot TSPulse anomaly detector (IBM Granite TSFM wrapper).

The pretrained model is implemented by IBM Granite TSFM:
https://github.com/ibm-granite/granite-tsfm

This module does not implement the TSPulse backbone. It provides project-specific
loading, masking, reconstruction-error scoring, smoothing, and aggregation.
"""

from __future__ import annotations

from threading import Lock

import numpy as np
import torch
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

from .common import BaseTimeSeriesAnomalyDetector

_MODEL_CACHE: dict[tuple[str, str, str], TSPulseForReconstruction] = {}
_MODEL_CACHE_LOCK = Lock()


def clear_tspulse_model_cache() -> None:
    """Drop checkpoint references cached by this process (primarily for tests)."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _canonical_device(device: str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(resolved)


class TSPulse(BaseTimeSeriesAnomalyDetector):
    """Score reconstruction errors with IBM's frozen pretrained TSPulse."""

    minimum_series_length = 80
    _CONTEXT_LENGTH = 512
    _PATCH_LENGTH = 8
    _OUTPUT_BY_MODE = {
        AnomalyScoreMethods.TIME_RECONSTRUCTION.value: "reconstruction_outputs",
        AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value: "reconstructed_ts_from_fft",
    }

    def __init__(
        self,
        checkpoint: str = "ibm-granite/granite-timeseries-tspulse-r1",
        revision: str = "main",
        window_size: int = 512,
        batch_size: int = 256,
        aggregation_length: int = 64,
        aggr_function: str = "mean",
        smoothing_length: int = 8,
        prediction_modes: tuple[str, ...] = (
            AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
            AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
        ),
        device: str | None = None,
        seed: int = 13,
    ) -> None:
        if int(window_size) != self._CONTEXT_LENGTH:
            raise ValueError("The pretrained TSPulse context length is fixed at 512")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if int(aggregation_length) < 1 or int(aggregation_length) > self._CONTEXT_LENGTH:
            raise ValueError("aggregation_length must be between 1 and 512")
        if int(aggregation_length) % self._PATCH_LENGTH:
            raise ValueError(
                "aggregation_length must be divisible by TSPulse's 8-point patch length"
            )
        if int(smoothing_length) < 1:
            raise ValueError("smoothing_length must be positive")
        if aggr_function.lower() not in {"min", "mean", "max"}:
            raise ValueError("aggr_function must be one of: min, mean, max")
        if not prediction_modes or any(
            mode not in self._OUTPUT_BY_MODE for mode in prediction_modes
        ):
            raise ValueError(
                "prediction_modes supports only TSPulse 'time' and 'fft' reconstruction"
            )

        super().__init__(window_size=self._CONTEXT_LENGTH, device=device, seed=seed)
        self.minimum_series_length = 80
        self.checkpoint = checkpoint
        self.revision = revision
        self.batch_size = int(batch_size)
        self.aggregation_length = int(aggregation_length)
        self.aggr_function = aggr_function.lower()
        self.smoothing_length = int(smoothing_length)
        self.prediction_modes = tuple(prediction_modes)
        self.model_: TSPulseForReconstruction | None = None

    def _validate_series(self, values: np.ndarray) -> None:
        if values.shape[1] != 1:
            raise ValueError("TSPulse currently supports only univariate series")
        finite_count = int(np.isfinite(values[:, 0]).sum())
        if finite_count < self.minimum_series_length:
            raise ValueError(
                f"TSPulse requires at least {self.minimum_series_length} finite points; "
                f"got {finite_count}"
            )

    def _validate_model(self, model: TSPulseForReconstruction) -> None:
        config = model.config
        context_length = int(config.context_length)
        patch_length = int(config.patch_length)
        patch_stride = int(config.patch_stride)
        channels = int(config.num_input_channels)
        if context_length != self._CONTEXT_LENGTH:
            raise ValueError(
                f"TSPulse checkpoint context_length must be 512, got {context_length}"
            )
        if patch_length != self._PATCH_LENGTH or patch_stride != self._PATCH_LENGTH:
            raise ValueError(
                "TSPulse checkpoint must use non-overlapping 8-point patches; "
                f"got patch_length={patch_length}, patch_stride={patch_stride}"
            )
        if context_length % patch_length or channels != 1 or config.mask_type != "user":
            raise ValueError(
                "TSPulse checkpoint has incompatible patch, channel, or user-mask geometry"
            )

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Reuse one frozen checkpoint per process/device."""
        self._validate_series(train_values)
        if self.model_ is None:
            device = _canonical_device(self.device)
            key = (self.checkpoint, self.revision, device)
            with _MODEL_CACHE_LOCK:
                model = _MODEL_CACHE.get(key)
                if model is None:
                    model = TSPulseForReconstruction.from_pretrained(
                        self.checkpoint,
                        num_input_channels=1,
                        revision=self.revision,
                        mask_type="user",
                    )
                    self._validate_model(model)
                    model = model.to(device).eval()
                    _MODEL_CACHE[key] = model
                self.model_ = model
        self.model_.eval()
        self.training_summary_ = {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "context_length": self._CONTEXT_LENGTH,
            "zero_shot": True,
        }

    def _contexts(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return zero-filled contexts, natural masks, and source indices."""
        series = values[:, 0]
        length = len(series)
        observed = np.isfinite(series)

        if length < self._CONTEXT_LENGTH:
            left = (self._CONTEXT_LENGTH - length) // 2
            context_values = np.zeros((1, self._CONTEXT_LENGTH), dtype=np.float32)
            context_observed = np.zeros((1, self._CONTEXT_LENGTH), dtype=bool)
            source_indices = np.full((1, self._CONTEXT_LENGTH), -1, dtype=np.int64)
            context_values[0, left : left + length] = np.where(observed, series, 0.0)
            context_observed[0, left : left + length] = observed
            source_indices[0, left : left + length] = np.arange(length)
            return context_values, context_observed, source_indices

        final_start = length - self._CONTEXT_LENGTH
        starts = list(range(0, final_start + 1, self.aggregation_length))
        if starts[-1] != final_start:
            starts.append(final_start)
        offsets = np.arange(self._CONTEXT_LENGTH)
        source_indices = np.stack([start + offsets for start in starts])
        context_observed = observed[source_indices]
        context_values = np.where(
            context_observed, series[source_indices], 0.0
        ).astype(np.float32)
        return context_values, context_observed, source_indices

    @staticmethod
    def _normalize_finite(scores: np.ndarray) -> np.ndarray:
        normalized = np.full(scores.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(scores)
        if not finite.any():
            return normalized
        minimum = float(np.min(scores[finite]))
        maximum = float(np.max(scores[finite]))
        if maximum <= minimum:
            normalized[finite] = 0.0
        else:
            normalized[finite] = (scores[finite] - minimum) / (maximum - minimum)
        return normalized

    def _smooth_finite(self, scores: np.ndarray) -> np.ndarray:
        if self.smoothing_length < 2:
            return scores
        finite = np.isfinite(scores)
        kernel = np.ones(self.smoothing_length, dtype=np.float64)
        numerator = np.convolve(np.where(finite, scores, 0.0), kernel, mode="full")
        denominator = np.convolve(finite.astype(np.float64), kernel, mode="full")
        start = (self.smoothing_length - 1) // 2
        numerator = numerator[start : start + len(scores)]
        denominator = denominator[start : start + len(scores)]
        smoothed = np.full(scores.shape, np.nan, dtype=np.float64)
        supported = finite & (denominator > 0)
        smoothed[supported] = numerator[supported] / denominator[supported]
        return smoothed

    def _aggregate_modes(self, mode_scores: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(mode_scores)
        finite = np.isfinite(stacked)
        supported = finite.any(axis=0)
        if self.aggr_function == "mean":
            result = np.divide(
                np.where(finite, stacked, 0.0).sum(axis=0),
                finite.sum(axis=0),
                out=np.full(stacked.shape[1], np.nan, dtype=np.float64),
                where=supported,
            )
        elif self.aggr_function == "min":
            result = np.where(finite, stacked, np.inf).min(axis=0)
        else:
            result = np.where(finite, stacked, -np.inf).max(axis=0)
        result[~supported] = np.nan
        return result.astype(np.float32)

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Mask each model patch and aggregate finite reconstruction errors."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        self._validate_series(values)
        context_values, context_observed, source_indices = self._contexts(values)
        targets = [
            (context_index, patch_start)
            for context_index in range(len(context_values))
            for patch_start in range(0, self._CONTEXT_LENGTH, self._PATCH_LENGTH)
            if context_observed[
                context_index, patch_start : patch_start + self._PATCH_LENGTH
            ].any()
        ]
        totals = {
            mode: np.zeros(len(values), dtype=np.float64)
            for mode in self.prediction_modes
        }
        counts = {
            mode: np.zeros(len(values), dtype=np.int64)
            for mode in self.prediction_modes
        }

        self.model_.eval()
        with torch.no_grad():
            for batch_start in range(0, len(targets), self.batch_size):
                batch_targets = targets[batch_start : batch_start + self.batch_size]
                context_ids = np.asarray([target[0] for target in batch_targets])
                batch_values = context_values[context_ids].copy()
                batch_mask = context_observed[context_ids].copy()
                for row, (_, patch_start) in enumerate(batch_targets):
                    batch_mask[row, patch_start : patch_start + self._PATCH_LENGTH] = False

                past_values = torch.from_numpy(batch_values[:, :, None]).to(self.device)
                past_observed_mask = torch.from_numpy(batch_mask[:, :, None]).to(
                    self.device
                )
                output = self.model_(
                    past_values=past_values,
                    past_observed_mask=past_observed_mask,
                    return_loss=False,
                )

                for mode in self.prediction_modes:
                    key = self._OUTPUT_BY_MODE[mode]
                    reconstruction = output[key]
                    if tuple(reconstruction.shape) != tuple(past_values.shape):
                        raise ValueError(
                            f"TSPulse {key} shape {tuple(reconstruction.shape)} does not "
                            f"match input shape {tuple(past_values.shape)}"
                        )
                    reconstructed = (
                        reconstruction.detach().float().cpu().numpy()[:, :, 0]
                    )
                    for row, (context_index, patch_start) in enumerate(batch_targets):
                        patch_end = patch_start + self._PATCH_LENGTH
                        original = source_indices[
                            context_index, patch_start:patch_end
                        ]
                        observed_target = context_observed[
                            context_index, patch_start:patch_end
                        ]
                        error = (
                            reconstructed[row, patch_start:patch_end]
                            - context_values[context_index, patch_start:patch_end]
                        ) ** 2
                        valid = observed_target & np.isfinite(error)
                        np.add.at(totals[mode], original[valid], error[valid])
                        np.add.at(counts[mode], original[valid], 1)

        mode_scores = []
        for mode in self.prediction_modes:
            averaged = np.divide(
                totals[mode],
                counts[mode],
                out=np.full(len(values), np.nan, dtype=np.float64),
                where=counts[mode] > 0,
            )
            mode_scores.append(
                self._smooth_finite(self._normalize_finite(averaged))
            )
        return self._aggregate_modes(mode_scores)
