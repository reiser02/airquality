"""Detector implementations for the anomaly pipeline (statistical + deep).

Re-exports every detector class plus the shared windowing/standardization
helpers from :mod:`.common`. ``TSPulse`` is optional: it is only exported when
``tsfm_public`` imports cleanly (see ``TSPULSE_AVAILABLE`` below).
"""

from .baselines import Hampel6Detector, HampelDetector, IQRDetector, IsolationForestDetector, LOFDetector, ModifiedZScoreDetector
from .carla import (
    CARLABase,
    CARLAGenIAS,
    CarlaAnomalyGenerator,
    CarlaGenIASGenerator,
    CarlaNativeGenerator,
    ClassificationLoss,
    ClusteringModel,
    ContrastiveModel,
    Conv1dSamePadding,
    ConvBlock,
    NeighborDataset,
    PretextDataset,
    PretextLoss,
    RepositoryAugmentedDataset,
    ResNetBlock,
    ResNetRepresentation,
    TSRepository,
    entropy,
)
from .common import (
    BaseTimeSeriesAnomalyDetector,
    GenIASWindowGenerator,
    aggregate_tail_scores,
    ensure_2d,
    fit_standardizer_nd,
    log_epoch,
    progress_enabled,
    rolling_windows_nd,
    set_progress_settings,
    set_random_seed,
    transform_standardize_nd,
)
from .couta import (
    COUTABase,
    COUTAGenIAS,
    COUTAGenIASGenerator,
    COUTAGenerator,
    COUTANativeGenerator,
    COUTANet,
    COUTATemporalBlock,
    DSVDDLoss,
    DSVDDUncLoss,
)
from .lstmad import LSTMAD, LSTMADNet
from .pca_detector import PCADetector
from .prophet_detector import ProphetDetector

# TSPulse pulls in `tsfm_public` (granite-tsfm). The version pinned by this
# project may not expose the APIs the wrapper imports; guard so an incompatible
# install degrades to "TSPulse unavailable" instead of breaking the whole
# detector package. `TSPULSE_AVAILABLE`/`TSPULSE_IMPORT_ERROR` let the registry
# skip it with a warning.
try:
    from .tspulse import TSPulse

    TSPULSE_AVAILABLE = True
    TSPULSE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional heavy deps
    TSPulse = None  # type: ignore[assignment]
    TSPULSE_AVAILABLE = False
    TSPULSE_IMPORT_ERROR = exc

__all__ = [
    "BaseTimeSeriesAnomalyDetector",
    "Hampel6Detector",
    "HampelDetector",
    "IQRDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "ModifiedZScoreDetector",
    "PCADetector",
    "ProphetDetector",
    "CARLABase",
    "CARLAGenIAS",
    "COUTABase",
    "COUTAGenIAS",
    "COUTAGenIASGenerator",
    "COUTAGenerator",
    "COUTANativeGenerator",
    "COUTANet",
    "COUTATemporalBlock",
    "CarlaAnomalyGenerator",
    "CarlaGenIASGenerator",
    "CarlaNativeGenerator",
    "ClassificationLoss",
    "ClusteringModel",
    "ContrastiveModel",
    "Conv1dSamePadding",
    "ConvBlock",
    "DSVDDLoss",
    "DSVDDUncLoss",
    "GenIASWindowGenerator",
    "LSTMAD",
    "LSTMADNet",
    "NeighborDataset",
    "PretextDataset",
    "PretextLoss",
    "RepositoryAugmentedDataset",
    "ResNetBlock",
    "ResNetRepresentation",
    "TSPulse",
    "TSRepository",
    "aggregate_tail_scores",
    "ensure_2d",
    "entropy",
    "fit_standardizer_nd",
    "log_epoch",
    "progress_enabled",
    "rolling_windows_nd",
    "set_progress_settings",
    "set_random_seed",
    "transform_standardize_nd",
]
