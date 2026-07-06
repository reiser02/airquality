"""CARLA detector (contrastive pretext + classification) and its GenIAS variant.

Self-supervised windowed detector: a ResNet encoder is pretrained contrastively
against synthetically corrupted windows (native CARLA rules or GenIAS-generated
anomalies), then a classification head separates normal from anomalous windows;
the per-window anomaly probability is folded back onto the timeline.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..windowing import aggregate_window_scores
from .common import (
    BaseTimeSeriesAnomalyDetector,
    GenIASWindowGenerator,
    log_epoch,
    resize_delta,
    resolve_training_stride,
    rolling_windows_nd,
    subsample_windows,
)


class CarlaAnomalyGenerator:
    """Interface for CARLA anomaly generators used in pretext augmentation."""

    def fit(self, train_windows: np.ndarray) -> None:
        """Optional training hook; stateless generators keep the no-op default."""
        return None

    def generate_window(self, window: Tensor) -> Tensor:
        """Return an anomalous copy of ``window``; implemented by subclasses."""
        raise NotImplementedError


class CarlaNativeGenerator(CarlaAnomalyGenerator):
    """Native CARLA window corruption rules for creating anomalous subsequences."""

    def __init__(self, portion_len: float = 0.1) -> None:
        self.portion_len = portion_len

    def inject_frequency_anomaly(
        self,
        window: Tensor,
        subsequence_length: int | None = None,
        compression_factor: int | None = None,
        scale_factor: float | np.ndarray | None = None,
        trend_factor: float | None = None,
        shapelet_factor: bool = False,
        trend_end: bool = False,
        start_index: int | None = None,
    ) -> Tensor:
        window = window.clone()
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)
        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)
        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])
        if start_index is None:
            start_index = np.random.randint(0, len(window) - subsequence_length)
        end_index = min(start_index + subsequence_length, window.shape[0])
        if trend_end:
            end_index = window.shape[0]
        anomalous_subsequence = window[start_index:end_index]
        anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)
        anomalous_subsequence = anomalous_subsequence[::compression_factor]
        scale_tensor = torch.as_tensor(scale_factor, dtype=window.dtype, device=window.device)
        anomalous_subsequence = anomalous_subsequence * scale_tensor
        if trend_factor is None:
            trend_factor = float(np.random.normal(1.0, 0.5))
        coefficient = -1.0 if np.random.uniform() < 0.5 else 1.0
        anomalous_subsequence = anomalous_subsequence + (coefficient * trend_factor)
        if shapelet_factor:
            anomalous_subsequence = window[start_index] + (torch.rand_like(window[start_index]) * 0.1)
        window[start_index:end_index] = anomalous_subsequence
        return window.squeeze(-1)

    def generate_window(self, window: Tensor) -> Tensor:
        base_window = window.clone()
        anomaly_seasonal = base_window.clone()
        anomaly_trend = base_window.clone()
        anomaly_global = base_window.clone()
        anomaly_contextual = base_window.clone()
        anomaly_shapelet = base_window.clone()
        min_len = int(base_window.shape[0] * 0.1)
        max_len = int(base_window.shape[0] * 0.9)
        subsequence_length = np.random.randint(min_len, max_len)
        start_index = np.random.randint(0, len(base_window) - subsequence_length)
        if base_window.ndim > 1:
            num_features = base_window.shape[1]
            min_dims = max(1, int(num_features / 10))
            max_dims = max(min_dims + 1, int(num_features / 2))
            num_dims = np.random.randint(min_dims, max_dims)
            for _ in range(num_dims):
                feature_index = np.random.randint(0, num_features)
                temp_window = base_window[:, feature_index].reshape(base_window.shape[0], 1)
                anomaly_seasonal[:, feature_index] = self.inject_frequency_anomaly(
                    temp_window,
                    scale_factor=1,
                    trend_factor=0,
                    subsequence_length=subsequence_length,
                    start_index=start_index,
                )
                anomaly_trend[:, feature_index] = self.inject_frequency_anomaly(
                    temp_window,
                    compression_factor=1,
                    scale_factor=1,
                    trend_end=True,
                    subsequence_length=subsequence_length,
                    start_index=start_index,
                )
                anomaly_global[:, feature_index] = self.inject_frequency_anomaly(
                    temp_window,
                    subsequence_length=2,
                    compression_factor=1,
                    scale_factor=8,
                    trend_factor=0,
                    start_index=start_index,
                )
                anomaly_contextual[:, feature_index] = self.inject_frequency_anomaly(
                    temp_window,
                    subsequence_length=4,
                    compression_factor=1,
                    scale_factor=3,
                    trend_factor=0,
                    start_index=start_index,
                )
                anomaly_shapelet[:, feature_index] = self.inject_frequency_anomaly(
                    temp_window,
                    compression_factor=1,
                    scale_factor=1,
                    trend_factor=0,
                    shapelet_factor=True,
                    subsequence_length=subsequence_length,
                    start_index=start_index,
                )
        else:
            temp_window = base_window.reshape(base_window.shape[0], 1)
            anomaly_seasonal = self.inject_frequency_anomaly(
                temp_window,
                scale_factor=1,
                trend_factor=0,
                subsequence_length=subsequence_length,
                start_index=start_index,
            )
            anomaly_trend = self.inject_frequency_anomaly(
                temp_window,
                compression_factor=1,
                scale_factor=1,
                trend_end=True,
                subsequence_length=subsequence_length,
                start_index=start_index,
            )
            anomaly_global = self.inject_frequency_anomaly(
                temp_window,
                subsequence_length=3,
                compression_factor=1,
                scale_factor=8,
                trend_factor=0,
                start_index=start_index,
            )
            anomaly_contextual = self.inject_frequency_anomaly(
                temp_window,
                subsequence_length=5,
                compression_factor=1,
                scale_factor=3,
                trend_factor=0,
                start_index=start_index,
            )
            anomaly_shapelet = self.inject_frequency_anomaly(
                temp_window,
                compression_factor=1,
                scale_factor=1,
                trend_factor=0,
                shapelet_factor=True,
                subsequence_length=subsequence_length,
                start_index=start_index,
            )
        anomalies = [anomaly_seasonal, anomaly_trend, anomaly_global, anomaly_contextual, anomaly_shapelet]
        anomalous_window = random.choice(anomalies)
        return anomalous_window.reshape_as(window)


class CarlaGenIASGenerator(CarlaAnomalyGenerator):
    """Adapts the GenIAS window generator to the CARLA generator interface."""

    def __init__(self, backend: GenIASWindowGenerator, fallback: CarlaAnomalyGenerator | None = None, max_windows: int = 256) -> None:
        self.backend = backend
        self.fallback = fallback or CarlaNativeGenerator()
        self.max_windows = max_windows

    def fit(self, train_windows: np.ndarray) -> None:
        """Train the GenIAS backend on a subsample of the normal training windows."""
        self.backend.fit(subsample_windows(train_windows, self.max_windows, self.backend.seed))

    @staticmethod
    def _select_target_span(window_length: int) -> tuple[int, int]:
        min_length = max(1, int(window_length * 0.1))
        max_length = min(window_length - 1, max(min_length, int(window_length * 0.9)))
        if max_length <= min_length:
            target_length = min_length
        else:
            target_length = int(np.random.randint(min_length, max_length + 1))
        max_start = max(window_length - target_length, 0)
        start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        return start, start + target_length

    def generate_window(self, window: Tensor) -> Tensor:
        batch = window.unsqueeze(0)
        patched_batch, mask_batch = self.backend.generate_batch(batch)
        patched = patched_batch.squeeze(0)
        changed = torch.nonzero(torch.any(mask_batch.squeeze(0), dim=-1), as_tuple=False).flatten()
        if changed.numel() == 0:
            return self.fallback.generate_window(window)
        source_start = int(changed[0].item())
        source_end = int(changed[-1].item()) + 1
        delta = patched[source_start:source_end] - window[source_start:source_end]
        target_start, target_end = self._select_target_span(int(window.shape[0]))
        resized_delta = resize_delta(delta, target_end - target_start)
        anomalous = window.clone()
        anomalous[target_start:target_end] = anomalous[target_start:target_end] + resized_delta
        return anomalous


class Conv1dSamePadding(nn.Conv1d):
    """1D convolution with runtime-computed same padding for sequence models."""

    def forward(self, input: Tensor) -> Tensor:
        kernel = self.weight.size(2)
        dilation = self.dilation[0]
        stride = self.stride[0]
        length = input.size(2)
        padding = ((length - 1) * stride) - length + (dilation * (kernel - 1)) + 1
        if padding % 2 != 0:
            input = F.pad(input, [0, 1])
        return F.conv1d(input, self.weight, self.bias, self.stride, padding // 2, self.dilation, self.groups)


class ConvBlock(nn.Module):
    """Small conv-batchnorm-ReLU block used inside the CARLA backbone."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ResNetBlock(nn.Module):
    """Residual 1D convolution block for extracting temporal features in CARLA."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        channels = [in_channels, out_channels, out_channels, out_channels]
        kernels = [8, 5, 3]
        self.layers = nn.Sequential(*[ConvBlock(channels[i], channels[i + 1], kernels[i], 1) for i in range(len(kernels))])
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.residual is None:
            return self.layers(x)
        return self.layers(x) + self.residual(x)


class ResNetRepresentation(nn.Module):
    """ResNet-style backbone that converts a window into a fixed-size representation."""

    def __init__(self, in_channels: int, mid_channels: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ResNetBlock(in_channels, mid_channels),
            ResNetBlock(mid_channels, mid_channels * 2),
            ResNetBlock(mid_channels * 2, mid_channels * 2),
        )
        self.output_dim = mid_channels * 2

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x).mean(dim=-1)


class ContrastiveModel(nn.Module):
    """Backbone plus projection head used during CARLA contrastive pretraining."""

    def __init__(self, backbone: nn.Module, backbone_dim: int, features_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.contrastive_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, features_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.contrastive_head(self.backbone(x)), dim=1)


class ClusteringModel(nn.Module):
    """Backbone plus cluster head used in CARLA's clustering stage."""

    def __init__(self, backbone: nn.Module, backbone_dim: int, nclusters: int, nheads: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.cluster_head = nn.ModuleList([nn.Linear(backbone_dim, nclusters) for _ in range(nheads)])

    def forward(self, x: Tensor, forward_pass: str = "default") -> Any:
        if forward_pass == "backbone":
            return self.backbone(x)
        if forward_pass == "head":
            return [cluster_head(x) for cluster_head in self.cluster_head]
        if forward_pass == "return_all":
            features = self.backbone(x)
            return {"features": features, "output": [cluster_head(features) for cluster_head in self.cluster_head]}
        features = self.backbone(x)
        return [cluster_head(features) for cluster_head in self.cluster_head]


class PretextLoss(nn.Module):
    """Margin-based contrastive loss for CARLA's pretext augmentation task."""

    def __init__(self, batch_size: int, temperature: float = 0.2, initial_margin: float = 1.0, adjust_factor: float = 0.1) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.margin = initial_margin
        self.adjust_factor = adjust_factor

    def forward(self, features: Tensor, current_loss: float | None = None) -> Tensor:
        features_org, features_pos, features_subseq = torch.split(features, self.batch_size, dim=0)
        anchor = F.normalize(features_org, dim=-1)
        positive = F.normalize(features_pos, dim=-1)
        negative = F.normalize(features_subseq, dim=-1)
        if current_loss is not None:
            self.margin = max(0.01, self.margin - self.adjust_factor * current_loss)
        positive_distance = torch.sum((anchor - positive) ** 2, dim=-1) / self.temperature
        negative_distance = torch.sum(torch.pow(anchor.unsqueeze(1) - negative, 2), dim=-1) / self.temperature
        hard_negative_distance = torch.min(negative_distance, dim=1)[0]
        return torch.mean(torch.clamp(self.margin + positive_distance - hard_negative_distance, min=0.0))


def entropy(x: Tensor, input_as_probabilities: bool) -> Tensor:
    """Compute entropy from probabilities or logits for CARLA regularization."""

    eps = 1e-8
    if input_as_probabilities:
        x_clamped = torch.clamp(x, min=eps)
        values = x_clamped * torch.log(x_clamped)
    else:
        values = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    return -values.sum(dim=1).mean() if values.ndim == 2 else -values.sum()


class ClassificationLoss(nn.Module):
    """Neighbor-consistency clustering loss used in CARLA's second training stage."""

    def __init__(self, entropy_weight: float = 2.0, inconsistency_weight: float = 0.0) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight
        self.inconsistency_weight = inconsistency_weight

    def forward(self, anchors: Tensor, nneighbors: Tensor, fneighbors: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_classes = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(nneighbors)
        negatives_prob = self.softmax(fneighbors)
        similarity = torch.bmm(anchors_prob.view(batch_size, 1, num_classes), positives_prob.view(batch_size, num_classes, 1)).squeeze()
        consistency_loss = self.bce(similarity, torch.ones_like(similarity))
        neg_similarity = torch.bmm(anchors_prob.view(batch_size, 1, num_classes), negatives_prob.view(batch_size, num_classes, 1)).squeeze()
        inconsistency_loss = self.bce(neg_similarity, torch.zeros_like(neg_similarity))
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)
        total = consistency_loss - self.entropy_weight * entropy_loss + self.inconsistency_weight * inconsistency_loss
        return total, consistency_loss, inconsistency_loss, entropy_loss


class PretextDataset(Dataset):
    """Builds original, weak, and anomalous window triplets for CARLA pretraining."""

    def __init__(
        self,
        windows: np.ndarray,
        generator: CarlaAnomalyGenerator,
        mean: np.ndarray,
        std: np.ndarray,
        noise_sigma: float = 0.1,
    ) -> None:
        self.windows = torch.from_numpy(windows)
        self.generator = generator
        self.noise_sigma = noise_sigma
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(np.where(std == 0.0, 1.0, std), dtype=torch.float32)
        self.samples = [self._build_sample(index) for index in range(self.windows.shape[0])]

    def __len__(self) -> int:
        return len(self.samples)

    def _build_sample(self, index: int) -> dict[str, Tensor]:
        ts_org = self.windows[index].clone()
        if index > 10:
            neighbor_index = np.random.randint(index - 10, index)
            ts_w_augment = self.windows[neighbor_index].clone()
        else:
            ts_w_augment = ts_org + torch.randn_like(ts_org) * self.noise_sigma
        ts_ss_augment = self.generator.generate_window(ts_org)
        mean = self.mean
        std = self.std
        return {
            "ts_org": (ts_org - mean) / std,
            "ts_w_augment": (ts_w_augment - mean) / std,
            "ts_ss_augment": (ts_ss_augment - mean) / std,
            "target": torch.tensor(0, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.samples[index]


class RepositoryAugmentedDataset(Dataset):
    """Stores CARLA's materialized original and subsequence-augmented windows."""

    def __init__(self, data: Tensor, targets: Tensor) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "ts_org": self.data[index],
            "target": self.targets[index],
        }


class NeighborDataset(Dataset):
    """Supplies anchor, near-neighbor, and far-neighbor windows for CARLA clustering."""

    def __init__(self, windows: np.ndarray, nearest_indices: np.ndarray, furthest_indices: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows)
        self.nearest_indices = nearest_indices
        self.furthest_indices = furthest_indices

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        nearest_index = int(np.random.choice(self.nearest_indices[index]))
        furthest_index = int(np.random.choice(self.furthest_indices[index]))
        return {
            "anchor": self.windows[index],
            "NNeighbor": self.windows[nearest_index],
            "FNeighbor": self.windows[furthest_index],
            "possible_nneighbors": torch.from_numpy(self.nearest_indices[index]),
            "possible_fneighbors": torch.from_numpy(self.furthest_indices[index]),
        }


def chunked_furthest_nearest_neighbors(feature_array: np.ndarray, neighbor_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k nearest/furthest neighbor search without materializing the full (N, N) pairwise matrix at once.

    Computes squared Euclidean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b in row chunks, so peak
    memory is bounded by chunk_size * N instead of N * N (and avoids an (N, N, dim) intermediate entirely).
    """

    size = feature_array.shape[0]
    squared_norms = np.sum(feature_array ** 2, axis=1)
    target_elements_per_chunk = 50_000_000
    chunk_size = max(1, min(size, target_elements_per_chunk // max(size, 1)))

    nearest = np.empty((size, neighbor_count), dtype=np.int64)
    furthest = np.empty((size, neighbor_count), dtype=np.int64)
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        cross_term = feature_array[start:end] @ feature_array.T
        distances = squared_norms[start:end, None] + squared_norms[None, :] - 2.0 * cross_term
        distances[np.arange(end - start), np.arange(start, end)] = np.inf

        nearest_candidates = np.argpartition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
        nearest_order = np.argsort(np.take_along_axis(distances, nearest_candidates, axis=1), axis=1)
        nearest[start:end] = np.take_along_axis(nearest_candidates, nearest_order, axis=1)

        furthest_candidates = np.argpartition(distances, size - neighbor_count, axis=1)[:, -neighbor_count:]
        furthest_order = np.argsort(np.take_along_axis(distances, furthest_candidates, axis=1), axis=1)[:, ::-1]
        furthest[start:end] = np.take_along_axis(furthest_candidates, furthest_order, axis=1)

    return nearest, furthest


class TSRepository:
    """Feature repository matching the upstream CARLA storage and mining flow."""

    def __init__(self, n: int, dim: int, num_classes: int, temperature: float) -> None:
        self.n = n
        self.dim = dim
        self.features = torch.zeros((self.n, self.dim), dtype=torch.float32)
        self.targets = torch.zeros(self.n, dtype=torch.long)
        self.ptr = 0
        self.device = "cpu"
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def reset(self) -> None:
        self.ptr = 0

    def resize(self, sz: int) -> None:
        self.n = sz * self.n
        self.features = torch.zeros((self.n, self.dim), dtype=torch.float32)
        self.targets = torch.zeros(self.n, dtype=torch.long)
        self.ptr = 0

    def update(self, features: Tensor, targets: Tensor | np.ndarray) -> None:
        batch_size = int(features.size(0))
        assert batch_size + self.ptr <= self.n
        self.features[self.ptr:self.ptr + batch_size].copy_(features.detach().cpu())
        if not torch.is_tensor(targets):
            targets = torch.from_numpy(np.asarray(targets))
        self.targets[self.ptr:self.ptr + batch_size].copy_(targets.detach().cpu())
        self.ptr += batch_size

    def to(self, device: str) -> None:
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def furthest_nearest_neighbors(self, topk: int) -> tuple[np.ndarray, np.ndarray]:
        feature_array = self.features[:self.ptr].cpu().numpy().astype(np.float32, copy=False)
        size = feature_array.shape[0]
        if size == 0:
            return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.int64)
        if size == 1:
            return np.zeros((1, 1), dtype=np.int64), np.zeros((1, 1), dtype=np.int64)

        neighbor_count = max(1, min(topk, size - 1))
        nearest, furthest = chunked_furthest_nearest_neighbors(feature_array, neighbor_count)
        return furthest.astype(np.int64, copy=False), nearest.astype(np.int64, copy=False)


class CARLABase(BaseTimeSeriesAnomalyDetector):
    """Baseline CARLA detector with native anomalous subsequence generation."""

    def __init__(
        self,
        window_size: int = 200,
        stride: int = 1,
        batch_size: int = 64,
        pretext_epochs: int = 20,
        classification_epochs: int = 20,
        learning_rate: float = 1e-3,
        features_dim: int = 32,
        mid_channels: int = 4,
        num_clusters: int = 4,
        num_heads: int = 5,
        num_neighbors: int = 5,
        temperature: float = 0.2,
        validation_ratio: float = 0.2,
        max_windows: int = 4096,
        device: str | None = None,
        seed: int = 13,
    ) -> None:
        super().__init__(window_size=window_size, stride=stride, device=device, seed=seed)
        self.batch_size = batch_size
        self.pretext_epochs = pretext_epochs
        self.classification_epochs = classification_epochs
        self.learning_rate = learning_rate
        self.features_dim = features_dim
        self.mid_channels = mid_channels
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.num_neighbors = num_neighbors
        self.temperature = temperature
        self.validation_ratio = validation_ratio
        self.max_windows = max_windows
        self.generator = CarlaNativeGenerator()
        self.pretext_model: ContrastiveModel | None = None
        self.classification_model: ClusteringModel | None = None
        self.majority_label_: int = 0
        self.selected_head_: int = 0

    def _build_generator(self) -> CarlaAnomalyGenerator:
        return self.generator

    def _resolve_training_stride(self, num_raw_windows: int) -> int:
        """Stride that keeps the training-window count near ``max_windows``."""
        return resolve_training_stride(num_raw_windows, self.max_windows)

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Run the two CARLA stages: contrastive pretext, then head classification.

        Works on de-normalized windows (the pretext dataset re-standardizes
        internally), trains the ResNet encoder against generator-corrupted
        windows, and finally picks the majority (normal) class per head.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model normalization statistics are unavailable")
        raw_train_values = (train_values * self.std_) + self.mean_
        num_raw_windows = raw_train_values.shape[0] - self.window_size + 1
        training_stride = self._resolve_training_stride(num_raw_windows)
        raw_windows = rolling_windows_nd(raw_train_values, self.window_size, stride=training_stride)
        raw_windows = subsample_windows(raw_windows, self.max_windows, self.seed)
        train_windows, val_windows = self._split_windows(raw_windows)
        generator = self._build_generator()
        generator.fit(train_windows)
        backbone = ResNetRepresentation(in_channels=raw_train_values.shape[1], mid_channels=self.mid_channels)
        self.pretext_model = ContrastiveModel(backbone, backbone.output_dim, features_dim=self.features_dim).to(self.device)
        pretext_dataset = PretextDataset(train_windows, generator, self.mean_, self.std_)
        pretext_loader = DataLoader(pretext_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        pretext_criterion = PretextLoss(self.batch_size, temperature=self.temperature)
        pretext_optimizer = torch.optim.Adam(self.pretext_model.parameters(), lr=self.learning_rate)
        previous_loss = None

        for epoch in range(self.pretext_epochs):
            batch_losses = []
            self.pretext_model.train()
            for batch in pretext_loader:
                ts_org = batch["ts_org"].float().to(self.device)
                ts_w_augment = batch["ts_w_augment"].float().to(self.device)
                ts_ss_augment = batch["ts_ss_augment"].float().to(self.device)
                batch_size = ts_org.shape[0]
                input_tensor = torch.cat([ts_org, ts_w_augment, ts_ss_augment], dim=0).transpose(1, 2)
                output = self.pretext_model(input_tensor)
                loss = pretext_criterion(output, previous_loss)
                pretext_optimizer.zero_grad()
                loss.backward()
                pretext_optimizer.step()
                previous_loss = float(loss.detach().cpu())
                batch_losses.append(previous_loss)
            log_epoch(
                f"{self.__class__.__name__} pretext",
                epoch,
                self.pretext_epochs,
                float(np.mean(batch_losses)) if batch_losses else 0.0,
            )

        train_repo_dataset, nearest_indices, furthest_indices = self._build_train_repository_dataset(pretext_dataset)
        classification_backbone = ResNetRepresentation(in_channels=raw_train_values.shape[1], mid_channels=self.mid_channels)
        classification_backbone.load_state_dict(self.pretext_model.backbone.state_dict())
        self.classification_model = ClusteringModel(
            classification_backbone,
            classification_backbone.output_dim,
            self.num_clusters,
            nheads=self.num_heads,
        ).to(self.device)
        classification_dataset = NeighborDataset(train_repo_dataset.data.cpu().numpy(), nearest_indices, furthest_indices)
        classification_loader = DataLoader(classification_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        classification_criterion = ClassificationLoss()
        classification_optimizer = torch.optim.Adam(self.classification_model.parameters(), lr=self.learning_rate)
        best_state = None
        best_loss = float("inf")
        best_head = 0

        validation_loader = None
        if len(val_windows) >= max(self.batch_size, self.num_neighbors + 1):
            normalized_val_windows = self._normalize_window_array(val_windows)
            val_nearest_indices, val_furthest_indices = self._build_validation_repository_neighbors(normalized_val_windows)
            validation_dataset = NeighborDataset(normalized_val_windows, val_nearest_indices, val_furthest_indices)
            validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        for epoch in range(self.classification_epochs):
            losses = []
            self.classification_model.train()
            for batch in classification_loader:
                anchors = batch["anchor"].float().to(self.device).transpose(1, 2)
                nneighbors = batch["NNeighbor"].float().to(self.device).transpose(1, 2)
                fneighbors = batch["FNeighbor"].float().to(self.device).transpose(1, 2)
                anchors_output = self.classification_model(anchors)
                nneighbors_output = self.classification_model(nneighbors)
                fneighbors_output = self.classification_model(fneighbors)
                total_loss_values = []
                for anchor_logits, nneighbor_logits, fneighbor_logits in zip(anchors_output, nneighbors_output, fneighbors_output):
                    total_loss, _, _, _ = classification_criterion(anchor_logits, nneighbor_logits, fneighbor_logits)
                    total_loss_values.append(total_loss)
                total_loss = torch.sum(torch.stack(total_loss_values, dim=0))
                classification_optimizer.zero_grad()
                total_loss.backward()
                classification_optimizer.step()
                losses.append(float(total_loss.detach().cpu()))

            head_losses = self._evaluate_classification_heads(validation_loader or classification_loader, classification_criterion)
            epoch_best_head = int(np.argmin(head_losses))
            epoch_best_loss = float(head_losses[epoch_best_head])
            log_epoch(
                f"{self.__class__.__name__} classification",
                epoch,
                self.classification_epochs,
                epoch_best_loss,
            )
            if epoch_best_loss < best_loss:
                best_loss = epoch_best_loss
                best_head = epoch_best_head
                best_state = {key: value.detach().cpu().clone() for key, value in self.classification_model.state_dict().items()}

        if best_state is not None:
            self.classification_model.load_state_dict(best_state)
        self.selected_head_ = best_head

        normalized_train_windows = self._normalize_window_array(train_windows)
        train_probabilities = self._window_probabilities(normalized_train_windows)
        self.majority_label_ = int(np.bincount(np.argmax(train_probabilities, axis=1), minlength=self.num_clusters).argmax())
        self.training_summary_ = {
            "pretext_epochs": self.pretext_epochs,
            "classification_epochs": self.classification_epochs,
            "selected_head": self.selected_head_,
            "selection_loss": float(best_loss) if np.isfinite(best_loss) else None,
            "majority_label": self.majority_label_,
        }

    def _split_windows(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(windows) < 4 or self.validation_ratio <= 0.0:
            return windows, windows[:0]
        permutation = np.random.RandomState(self.seed).permutation(len(windows))
        split_index = max(1, int(round(len(windows) * (1.0 - self.validation_ratio))))
        split_index = min(split_index, len(windows) - 1)
        train_indices = permutation[:split_index]
        val_indices = permutation[split_index:]
        return windows[train_indices], windows[val_indices]

    def _fill_ts_repository(
        self,
        loader: DataLoader,
        model: ContrastiveModel,
        ts_repository: TSRepository,
        *,
        real_aug: bool = False,
        ts_repository_aug: TSRepository | None = None,
    ) -> RepositoryAugmentedDataset | None:
        model.eval()
        ts_repository.reset()
        if ts_repository_aug is not None:
            ts_repository_aug.reset()
        if real_aug:
            ts_repository.resize(3)
        # Collect per-batch pieces and concatenate once at the end: repeated
        # `torch.cat` on the accumulator copies everything gathered so far on
        # every batch (quadratic time and memory churn).
        con_data_parts: list[Tensor] = []
        con_target_parts: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                ts_org = batch["ts_org"].float().to(self.device)
                targets = batch["target"].to(self.device)
                outputs = model(ts_org.transpose(1, 2))
                ts_repository.update(outputs, targets)
                if ts_repository_aug is not None:
                    ts_repository_aug.update(outputs, targets)
                if real_aug:
                    con_data_parts.append(ts_org.cpu())
                    con_target_parts.append(targets.cpu())
                    ts_w_augment = batch["ts_w_augment"].float().to(self.device)
                    weak_targets = torch.full((ts_w_augment.shape[0],), 2, dtype=torch.long, device=self.device)
                    weak_outputs = model(ts_w_augment.transpose(1, 2))
                    ts_repository.update(weak_outputs, weak_targets)
                    ts_ss_augment = batch["ts_ss_augment"].float().to(self.device)
                    subseq_targets = torch.full((ts_ss_augment.shape[0],), 4, dtype=torch.long, device=self.device)
                    con_data_parts.append(ts_ss_augment.cpu())
                    con_target_parts.append(subseq_targets.cpu())
                    subseq_outputs = model(ts_ss_augment.transpose(1, 2))
                    ts_repository.update(subseq_outputs, subseq_targets)
                    if ts_repository_aug is not None:
                        ts_repository_aug.update(subseq_outputs, subseq_targets)
        if real_aug:
            sample_shape = tuple(loader.dataset[0]["ts_org"].shape)
            con_data = (
                torch.cat(con_data_parts, dim=0)
                if con_data_parts
                else torch.empty((0, *sample_shape), dtype=torch.float32)
            )
            con_target = (
                torch.cat(con_target_parts, dim=0)
                if con_target_parts
                else torch.empty(0, dtype=torch.long)
            )
            return RepositoryAugmentedDataset(con_data, con_target)
        return None

    def _build_train_repository_dataset(self, pretext_dataset: PretextDataset) -> tuple[RepositoryAugmentedDataset, np.ndarray, np.ndarray]:
        base_loader = DataLoader(pretext_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        # Repositories stay on CPU: `update()` stores detached CPU copies and
        # neighbor mining runs in NumPy, so device residency only wasted VRAM
        # and transfers.
        repository_base = TSRepository(len(pretext_dataset), self.features_dim, self.num_clusters, self.temperature)
        repository_aug = TSRepository(len(pretext_dataset) * 2, self.features_dim, self.num_clusters, self.temperature)
        repo_dataset = self._fill_ts_repository(
            base_loader,
            self.pretext_model,
            repository_base,
            real_aug=True,
            ts_repository_aug=repository_aug,
        )
        if repo_dataset is None:
            raise RuntimeError("Expected repository dataset to be created for CARLA classification")
        furthest_indices, nearest_indices = repository_aug.furthest_nearest_neighbors(10)
        return repo_dataset, nearest_indices, furthest_indices

    def _build_validation_repository_neighbors(self, val_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.pretext_model is None:
            raise RuntimeError("Pretext model must be fitted before repository mining")
        targets = torch.zeros(len(val_windows), dtype=torch.long)
        dataset = RepositoryAugmentedDataset(torch.from_numpy(val_windows), targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        repository_val = TSRepository(len(val_windows), self.features_dim, self.num_clusters, self.temperature)
        self._fill_ts_repository(loader, self.pretext_model, repository_val, real_aug=False, ts_repository_aug=None)
        furthest_indices, nearest_indices = repository_val.furthest_nearest_neighbors(10)
        return nearest_indices, furthest_indices

    def _normalize_window_array(self, windows: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model must be fitted before normalizing windows")
        return ((windows - self.mean_[None, None, :]) / self.std_[None, None, :]).astype(np.float32)

    def _evaluate_classification_heads(self, loader: DataLoader, criterion: ClassificationLoss) -> np.ndarray:
        if self.classification_model is None:
            raise RuntimeError("Model must be fitted before evaluation")
        head_losses = [[] for _ in range(self.num_heads)]
        self.classification_model.eval()
        with torch.no_grad():
            for batch in loader:
                anchors = batch["anchor"].float().to(self.device).transpose(1, 2)
                nneighbors = batch["NNeighbor"].float().to(self.device).transpose(1, 2)
                fneighbors = batch["FNeighbor"].float().to(self.device).transpose(1, 2)
                anchors_output = self.classification_model(anchors)
                nneighbors_output = self.classification_model(nneighbors)
                fneighbors_output = self.classification_model(fneighbors)
                for index, (anchor_logits, nneighbor_logits, fneighbor_logits) in enumerate(zip(anchors_output, nneighbors_output, fneighbors_output)):
                    total_loss, _, _, _ = criterion(anchor_logits, nneighbor_logits, fneighbor_logits)
                    head_losses[index].append(float(total_loss.detach().cpu()))
        return np.array([np.mean(losses) if losses else float("inf") for losses in head_losses], dtype=np.float64)

    def _window_probabilities(self, windows: np.ndarray) -> np.ndarray:
        if self.classification_model is None:
            raise RuntimeError("Model must be fitted before scoring")
        dataset = torch.from_numpy(windows)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        probabilities = []
        self.classification_model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.float().to(self.device).transpose(1, 2)
                logits = self.classification_model(batch)[self.selected_head_]
                probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def _score_with_head_and_label(self, values: np.ndarray, head_index: int, majority_label: int) -> np.ndarray:
        """Score with one classification head: ``1 − P(majority class)`` per window."""
        if self.classification_model is None:
            raise RuntimeError("Model must be fitted before scoring")
        windows = rolling_windows_nd(values, self.window_size, stride=1)
        dataset = torch.from_numpy(windows)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        probabilities = []
        self.classification_model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.float().to(self.device).transpose(1, 2)
                logits = self.classification_model(batch)[head_index]
                probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        window_probabilities = np.concatenate(probabilities, axis=0)
        window_scores = 1.0 - window_probabilities[:, majority_label]
        return aggregate_window_scores(window_scores, values.shape[0], self.window_size)

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Score sliding windows with the selected head and fold onto the timeline."""
        windows = rolling_windows_nd(values, self.window_size, stride=1)
        probabilities = self._window_probabilities(windows)
        window_scores = 1.0 - probabilities[:, self.majority_label_]
        return aggregate_window_scores(window_scores, values.shape[0], self.window_size)


class CARLAGenIAS(CARLABase):
    """CARLA variant that uses GenIAS instead of native anomaly generation."""

    def __init__(self, *args: Any, latent_dim: int = 50, genias_learning_rate: float = 1e-4, genias_batch_size: int = 100, genias_max_epochs: int = 60, genias_patience: int = 12, genias_max_windows: int = 256, tau: float = 0.2, anomaly_scale: float = 2.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        backend = GenIASWindowGenerator(
            window_size=self.window_size,
            latent_dim=latent_dim,
            learning_rate=genias_learning_rate,
            batch_size=genias_batch_size,
            max_epochs=genias_max_epochs,
            patience=genias_patience,
            tau=tau,
            anomaly_scale=anomaly_scale,
            device=self.device,
            seed=self.seed,
        )
        self.generator = CarlaGenIASGenerator(backend, max_windows=genias_max_windows)
