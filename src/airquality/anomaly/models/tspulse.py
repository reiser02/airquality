"""Zero-shot TSPulse anomaly detector (IBM Granite TSFM wrapper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline

from .common import BaseTimeSeriesAnomalyDetector


def _build_frame(series: np.ndarray) -> pd.DataFrame:
    """Wrap a value array in the timestamp/value frame the TSFM pipeline expects."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2000-01-01", periods=len(series), freq="s"),
            "value": np.ascontiguousarray(series, dtype=np.float64),
        }
    )


class TSPulse(BaseTimeSeriesAnomalyDetector):
    """Zero-shot anomaly detector wrapping IBM's pretrained TSPulse checkpoint.

    Unlike the other detectors here, TSPulse is not trained per series: `fit` only
    loads the frozen pretrained weights and builds the dual time/frequency-domain
    reconstruction-error scoring pipeline (see ibm-granite/granite-timeseries-tspulse-r1).
    """

    def __init__(
        self,
        checkpoint: str = "ibm-granite/granite-timeseries-tspulse-r1",
        revision: str = "main",
        window_size: int = 512,
        batch_size: int = 256,
        aggregation_length: int = 64,
        aggr_function: str = "max",
        smoothing_length: int = 8,
        least_significant_scale: float = 0.01,
        least_significant_score: float = 0.1,
        prediction_modes: tuple[str, ...] = (AnomalyScoreMethods.TIME_RECONSTRUCTION.value, AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value),
        device: str | None = None,
        seed: int = 13,
    ) -> None:
        super().__init__(window_size=window_size, device=device, seed=seed)
        self.checkpoint = checkpoint
        self.revision = revision
        self.batch_size = batch_size
        self.aggregation_length = aggregation_length
        self.aggr_function = aggr_function
        self.smoothing_length = smoothing_length
        self.least_significant_scale = least_significant_scale
        self.least_significant_score = least_significant_score
        self.prediction_modes = prediction_modes
        self.model_: TSPulseForReconstruction | None = None
        self.pipeline_: TimeSeriesAnomalyDetectionPipeline | None = None

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Load the frozen pretrained checkpoint and build the scoring pipeline."""
        if train_values.shape[1] != 1:
            raise ValueError("TSPulse currently supports only univariate series")
        model = TSPulseForReconstruction.from_pretrained(
            self.checkpoint,
            num_input_channels=1,
            revision=self.revision,
            mask_type="user",
        )
        self.window_size = int(model.config.context_length)
        if train_values.shape[0] < self.window_size:
            raise ValueError(f"Series length {train_values.shape[0]} is shorter than TSPulse's context length {self.window_size}")
        self.model_ = model
        self.pipeline_ = TimeSeriesAnomalyDetectionPipeline(
            model,
            timestamp_column="timestamp",
            target_columns=["value"],
            prediction_mode=list(self.prediction_modes),
            aggregation_length=self.aggregation_length,
            aggr_function=self.aggr_function,
            smoothing_length=self.smoothing_length,
            least_significant_scale=self.least_significant_scale,
            least_significant_score=self.least_significant_score,
            device=self.device,
        )
        self.training_summary_ = {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "context_length": self.window_size,
            "zero_shot": True,
        }

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Run the TSFM anomaly pipeline and return its per-timestep scores."""
        if self.pipeline_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        if values.shape[0] < self.window_size:
            raise ValueError(f"Series length {values.shape[0]} is shorter than TSPulse's context length {self.window_size}")
        frame = _build_frame(values[:, 0])
        result = self.pipeline_(frame, batch_size=self.batch_size)
        return result["anomaly_score"].to_numpy(dtype=np.float32)
