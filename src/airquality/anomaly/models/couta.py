"""COUTA detector (calibrated one-class TCN) and its GenIAS variant.

Windowed deep-SVDD-style detector: a TCN maps each window to an embedding and
the anomaly score is the squared distance to a fixed center, with a
self-supervised pretext head trained on synthetically corrupted windows
(native COUTA rules or GenIAS-generated anomalies).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader

from .genias import Chomp1d
from .common import (
    BaseTimeSeriesAnomalyDetector,
    GenIASWindowGenerator,
    aggregate_tail_scores,
    log_epoch,
    rolling_windows_nd,
    subsample_windows,
)


class COUTAGenerator:
    """Interface for COUTA anomaly generators used during self-supervised training."""

    def fit(self, train_windows: np.ndarray) -> None:
        """Optional training hook; stateless generators keep the no-op default."""
        return None

    def generate_batch(self, batch_seqs: Tensor, seed: int) -> tuple[Tensor, Tensor]:
        """Return ``(corrupted_batch, labels)``; implemented by subclasses."""
        raise NotImplementedError


class COUTANativeGenerator(COUTAGenerator):
    """Native COUTA corruption rules for synthesizing anomalous training windows."""

    def __init__(self, max_cut_ratio: float = 0.5, ss_type: str = "FULL") -> None:
        self.max_cut_ratio = max_cut_ratio
        self.ss_type = ss_type

    def generate_batch(self, batch_seqs: Tensor, seed: int) -> tuple[Tensor, Tensor]:
        rng = np.random.RandomState(seed=seed)
        batch_size, length, dim = batch_seqs.shape
        cut_start = length - rng.randint(1, max(2, int(self.max_cut_ratio * length)), size=batch_size)
        n_cut_dim = rng.randint(1, dim + 1, size=batch_size)
        cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]
        batch_neg = batch_seqs.clone()
        flags = rng.randint(1e5, size=batch_size)
        n_types = 6
        for index in range(batch_size):
            flag = flags[index] % n_types
            dims = cut_dim[index]
            if flag == 0:
                batch_neg[index, cut_start[index]:, dims] = 0
            elif flag == 1:
                batch_neg[index, cut_start[index]:, dims] = 1
            elif flag == 2:
                mean = torch.mean(batch_neg[index, -10:, dims], dim=0)
                batch_neg[index, -1, dims] = mean + 0.5
            elif flag == 3:
                mean = torch.mean(batch_neg[index, -10:, dims], dim=0)
                batch_neg[index, -1, dims] = mean - 0.5
            elif flag == 4:
                batch_neg[index, -1, dims] = 2
            else:
                batch_neg[index, -1, dims] = -2
        labels = torch.ones(batch_size, device=batch_seqs.device)
        return batch_neg, labels


class COUTAGenIASGenerator(COUTAGenerator):
    """Adapts the GenIAS window generator to the COUTA generator interface."""

    def __init__(self, backend: GenIASWindowGenerator, fallback: COUTAGenerator | None = None, max_cut_ratio: float = 0.5, max_windows: int = 256) -> None:
        self.backend = backend
        self.fallback = fallback or COUTANativeGenerator(max_cut_ratio=max_cut_ratio)
        self.max_cut_ratio = max_cut_ratio
        self.max_windows = max_windows

    def fit(self, train_windows: np.ndarray) -> None:
        """Train the GenIAS backend on a subsample of the normal training windows."""
        self.backend.fit(subsample_windows(train_windows, self.max_windows, self.backend.seed))

    @staticmethod
    def _resize_delta(delta: Tensor, target_length: int) -> Tensor:
        source_length = int(delta.shape[0])
        if source_length == target_length:
            return delta
        if source_length == 1:
            return delta.repeat(target_length, 1)
        resized = F.interpolate(delta.transpose(0, 1).unsqueeze(0), size=target_length, mode="linear", align_corners=False)
        return resized.squeeze(0).transpose(0, 1)

    def generate_batch(self, batch_seqs: Tensor, seed: int) -> tuple[Tensor, Tensor]:
        batch_neg, mask = self.backend.generate_batch(batch_seqs)
        aligned_batch = batch_seqs.clone()
        fallback_batch, fallback_labels = self.fallback.generate_batch(batch_seqs, seed=seed)
        rng = np.random.RandomState(seed=seed)
        batch_size, length, _ = batch_seqs.shape
        cut_start = length - rng.randint(1, max(2, int(self.max_cut_ratio * length)), size=batch_size)
        for index in range(batch_seqs.shape[0]):
            changed = torch.nonzero(torch.any(mask[index], dim=-1), as_tuple=False).flatten()
            if changed.numel() == 0:
                aligned_batch[index] = fallback_batch[index]
                continue
            start = int(changed[0].item())
            end = int(changed[-1].item()) + 1
            delta = batch_neg[index, start:end] - batch_seqs[index, start:end]
            target_start = int(cut_start[index])
            target_length = length - target_start
            resized_delta = self._resize_delta(delta, target_length)
            if torch.max(torch.abs(resized_delta)) <= 1e-6:
                aligned_batch[index] = fallback_batch[index]
                continue
            if torch.max(torch.abs(resized_delta[-1])) <= 1e-6:
                resized_delta[-1] = delta[-1]
            aligned_batch[index, target_start:] = batch_seqs[index, target_start:] + resized_delta
        return aligned_batch, fallback_labels.to(batch_seqs.device)


class COUTATemporalBlock(nn.Module):
    """Residual dilated temporal convolution block used in the COUTA encoder."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, bias: bool, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias))
        self.net = nn.Sequential(
            self.conv1,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


class COUTANet(nn.Module):
    """Temporal convolution network that produces COUTA representations and pretext scores."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | list[int] = 16,
        rep_hidden: int = 16,
        pretext_hidden: int = 16,
        emb_dim: int = 16,
        kernel_size: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        dup: bool = True,
        pretext: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers = []
        for index, out_channels in enumerate(hidden_dims):
            in_channels = input_dim if index == 0 else hidden_dims[index - 1]
            layers.append(COUTATemporalBlock(in_channels, out_channels, kernel_size, 2 ** index, bias, dropout))
        self.network = nn.Sequential(*layers)
        self.l1 = nn.Linear(hidden_dims[-1], rep_hidden, bias=bias)
        self.l2 = nn.Linear(rep_hidden, emb_dim, bias=bias)
        self.act = nn.LeakyReLU()
        self.dup = dup
        self.pretext = pretext
        if dup:
            self.l1_dup = nn.Linear(hidden_dims[-1], rep_hidden, bias=bias)
        if pretext:
            self.pretext_l1 = nn.Linear(hidden_dims[-1], pretext_hidden, bias=bias)
            self.pretext_l2 = nn.Linear(pretext_hidden, 1, bias=bias)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = out[:, -1]
        rep = self.l2(self.act(self.l1(out)))
        score = self.pretext_l2(self.act(self.pretext_l1(out))) if self.pretext else None
        rep_dup = self.l2(self.act(self.l1_dup(out))) if self.dup else None
        outputs = [rep]
        if rep_dup is not None:
            outputs.append(rep_dup)
        if score is not None:
            outputs.append(score)
        return tuple(outputs)


class DSVDDLoss(nn.Module):
    """Basic Deep SVDD loss measuring distance to the learned center."""

    def __init__(self, c: Tensor) -> None:
        super().__init__()
        self.c = c

    def forward(self, rep: Tensor) -> Tensor:
        return torch.mean(torch.sum((rep - self.c) ** 2, dim=1))


class DSVDDUncLoss(nn.Module):
    """Uncertainty-aware SVDD loss using two representations of the same window."""

    def __init__(self, c: Tensor) -> None:
        super().__init__()
        self.c = c

    def forward(self, rep: Tensor, rep2: Tensor) -> Tensor:
        dis1 = torch.sum((rep - self.c) ** 2, dim=1)
        dis2 = torch.sum((rep2 - self.c) ** 2, dim=1)
        var = (dis1 - dis2) ** 2
        return torch.mean(0.5 * torch.exp(-var) * (dis1 + dis2) + 0.5 * var)


class COUTABase(BaseTimeSeriesAnomalyDetector):
    """Baseline COUTA detector with native synthetic anomaly generation."""

    def __init__(
        self,
        window_size: int = 100,
        stride: int = 1,
        batch_size: int = 64,
        num_epochs: int = 40,
        train_val_pc: float = 0.25,
        learning_rate: float = 1e-4,
        hidden_dims: int | list[int] = 16,
        emb_dim: int = 16,
        rep_hidden: int = 16,
        pretext_hidden: int = 16,
        kernel_size: int = 2,
        dropout: float = 0.0,
        alpha: float = 0.1,
        neg_batch_ratio: float = 0.2,
        max_windows: int = 4096,
        device: str | None = None,
        seed: int = 13,
    ) -> None:
        super().__init__(window_size=window_size, stride=stride, device=device, seed=seed)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_val_pc = train_val_pc
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.emb_dim = emb_dim
        self.rep_hidden = rep_hidden
        self.pretext_hidden = pretext_hidden
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.alpha = alpha
        self.neg_batch_ratio = neg_batch_ratio
        self.max_windows = max_windows
        self.generator = COUTANativeGenerator()
        self.net: COUTANet | None = None
        self.c: Tensor | None = None

    def _build_generator(self) -> COUTAGenerator:
        return self.generator

    def _resolve_training_stride(self, num_raw_windows: int) -> int:
        """Mirror CARLABase's window-count cap so training cost stops scaling with series length."""
        return max(1, -(-num_raw_windows // self.max_windows))

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Train the one-class TCN: fix the SVDD center, then optimize both losses."""
        num_raw_windows = train_values.shape[0] - self.window_size + 1
        training_stride = self._resolve_training_stride(num_raw_windows)
        windows = rolling_windows_nd(train_values, self.window_size, training_stride)
        windows = subsample_windows(windows, self.max_windows, self.seed)
        windows = windows[np.random.RandomState(42).permutation(len(windows))]
        split_index = len(windows) - int(self.train_val_pc * len(windows))
        train_windows = windows[:split_index] if split_index > 0 else windows
        generator = self._build_generator()
        generator.fit(train_windows)
        dataset = torch.from_numpy(train_windows)
        self.net = COUTANet(
            input_dim=train_values.shape[1],
            hidden_dims=self.hidden_dims,
            emb_dim=self.emb_dim,
            rep_hidden=self.rep_hidden,
            pretext_hidden=self.pretext_hidden,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            dup=True,
            pretext=True,
        ).to(self.device)
        self.c = self._set_center(dataset)
        criterion_oc = DSVDDUncLoss(self.c)
        criterion_ssl = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        neg_batch_size = max(1, int(self.neg_batch_ratio * self.batch_size))
        for epoch in range(self.num_epochs):
            loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)
            epoch_losses = []
            epoch_oc_losses = []
            epoch_ssl_losses = []
            for batch_index, x0 in enumerate(loader):
                x0 = x0.float().to(self.device)
                x0_output = self.net(x0)
                rep_x0, rep_x0_dup, pred_x0 = x0_output
                loss_oc = criterion_oc(rep_x0, rep_x0_dup)

                neg_candidate_idx = np.random.RandomState(self.seed + epoch + batch_index).randint(0, self.batch_size, neg_batch_size)
                x1, y1 = generator.generate_batch(x0[neg_candidate_idx], seed=self.seed + epoch + batch_index)
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)
                y0 = -torch.ones(self.batch_size, device=self.device)
                _, _, pred_x1 = self.net(x1)
                ssl_targets = torch.cat([y0, y1], dim=0)
                ssl_outputs = torch.cat([pred_x0.view(-1), pred_x1.view(-1)], dim=0)
                loss_ssl = criterion_ssl(ssl_outputs, ssl_targets)
                loss = loss_oc + self.alpha * loss_ssl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(float(loss.detach().cpu()))
                epoch_oc_losses.append(float(loss_oc.detach().cpu()))
                epoch_ssl_losses.append(float(loss_ssl.detach().cpu()))
            self.training_summary_ = {
                "loss": float(np.mean(epoch_losses)),
                "oc_loss": float(np.mean(epoch_oc_losses)),
                "ssl_loss": float(np.mean(epoch_ssl_losses)),
                "epochs_trained": epoch + 1,
            }
            log_epoch(self.__class__.__name__, epoch, self.num_epochs, self.training_summary_["loss"])

    def _set_center(self, dataset: Tensor, eps: float = 0.1) -> Tensor:
        if self.net is None:
            raise RuntimeError("Network is not initialized")
        loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)
        representations = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.float().to(self.device)
                representations.append(self.net(batch)[0].detach())
        c = torch.mean(torch.cat(representations, dim=0), dim=0)
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c > 0)] = eps
        return c

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Score each timestep by its window's distance to the SVDD center."""
        window_scores, _ = self._window_components_normalized(values)
        prefix = np.zeros(values.shape[0] - window_scores.shape[0], dtype=np.float32)
        return np.concatenate([prefix, window_scores.astype(np.float32)], axis=0)

    def _window_components_normalized(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.net is None or self.c is None:
            raise RuntimeError("Model must be fitted before scoring")
        windows = rolling_windows_nd(values, self.window_size, stride=1)
        dataset = torch.from_numpy(windows)
        loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        distances = []
        pretext_scores = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.float().to(self.device)
                rep, rep_dup, pred = self.net(batch)
                dis = torch.sum((rep - self.c) ** 2, dim=1) + torch.sum((rep_dup - self.c) ** 2, dim=1)
                distances.append(dis.cpu().numpy())
                pretext_scores.append(pred.view(-1).cpu().numpy())
        return np.concatenate(distances, axis=0), np.concatenate(pretext_scores, axis=0)


class COUTAGenIAS(COUTABase):
    """COUTA variant that swaps native augmentations for GenIAS-generated anomalies."""

    def __init__(self, *args: Any, latent_dim: int = 50, genias_learning_rate: float = 1e-4, genias_batch_size: int = 100, genias_max_epochs: int = 60, genias_patience: int = 12, genias_max_windows: int = 256, tau: float = 0.2, anomaly_scale: float = 1.0, ssl_score_weight: float = 0.5, **kwargs: Any) -> None:
        kwargs.setdefault("alpha", 0.2)
        super().__init__(*args, **kwargs)
        self.ssl_score_weight = ssl_score_weight
        self.train_distance_std_ = 1.0
        self.train_pretext_mean_ = 0.0
        self.train_pretext_std_ = 1.0
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
        self.generator = COUTAGenIASGenerator(backend, max_windows=genias_max_windows)

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Fit COUTA, then record train-score statistics to calibrate the SSL boost."""
        super()._fit_normalized(train_values)
        train_distances, train_pretext = self._window_components_normalized(train_values)
        self.train_distance_std_ = float(max(train_distances.std(), 1e-6))
        self.train_pretext_mean_ = float(train_pretext.mean())
        self.train_pretext_std_ = float(max(train_pretext.std(), 1e-6))

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Combine SVDD distances with a calibrated pretext-score tail boost."""
        distances, pretext_scores = self._window_components_normalized(values)
        pretext_excess = np.maximum(pretext_scores - self.train_pretext_mean_, 0.0)
        pretext_boost = pretext_excess * (self.train_distance_std_ / self.train_pretext_std_)
        prefix = np.zeros(values.shape[0] - distances.shape[0], dtype=np.float32)
        distance_scores = np.concatenate([prefix, distances.astype(np.float32)], axis=0)
        tail_length = max(1, int(self.window_size * self.generator.max_cut_ratio * 0.5))
        pretext_tail_scores = aggregate_tail_scores(pretext_boost, values.shape[0], self.window_size, tail_length)
        return distance_scores + self.ssl_score_weight * pretext_tail_scores
