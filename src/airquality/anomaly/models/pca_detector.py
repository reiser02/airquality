"""Sliding-window PCA anomaly detector (pyod-style, Shyu et al. 2003)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from ..windowing import aggregate_window_scores
from .common import BaseTimeSeriesAnomalyDetector, rolling_windows_nd


class PCADetector(BaseTimeSeriesAnomalyDetector):
    """Sliding-window adaptation of pyod's `PCA` detector (Shyu et al., 2003).

    Each window is standardized and treated as a point in `window_size`-dimensional
    space. PCA finds the eigenvectors of that window matrix; a window's anomaly score
    is the sum of its distances to the `n_selected_components_` eigenvectors with the
    smallest eigenvalues (the directions normal windows vary in least), each distance
    weighted by the inverse of that eigenvector's explained-variance ratio so rarer
    directions of variation contribute more to the score. Window scores are averaged
    back onto the timeline via `aggregate_window_scores`, since PCA itself has no
    native notion of "one score per timestep."
    """

    def __init__(
        self,
        *args: Any,
        window_size: int = 100,
        n_components: int | float | None = None,
        n_selected_components: int | None = None,
        whiten: bool = False,
        svd_solver: str = "auto",
        weighted: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, window_size=window_size, **kwargs)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.weighted = weighted
        self.window_scaler_: StandardScaler | None = None
        self.model_: SklearnPCA | None = None
        self.selected_components_: np.ndarray | None = None
        self.selected_weights_: np.ndarray | None = None

    def _fit_normalized(self, train_values: np.ndarray) -> None:
        """Fit PCA on standardized windows and keep the smallest-variance components."""
        if train_values.shape[1] != 1:
            raise ValueError("PCADetector currently supports only univariate series")
        windows = rolling_windows_nd(train_values, self.window_size, stride=1)[:, :, 0]

        self.window_scaler_ = StandardScaler().fit(windows)
        standardized = self.window_scaler_.transform(windows)

        self.model_ = SklearnPCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.seed,
        )
        self.model_.fit(standardized)

        n_components = self.model_.n_components_
        n_selected = max(1, min(self.n_selected_components or n_components, n_components))
        weights = self.model_.explained_variance_ratio_ if self.weighted else np.ones(n_components, dtype=np.float64)

        self.selected_components_ = self.model_.components_[-n_selected:, :]
        self.selected_weights_ = weights[-n_selected:]
        self.training_summary_ = {
            "window_size": self.window_size,
            "n_components": int(n_components),
            "n_selected_components": int(n_selected),
        }

    def _score_normalized(self, values: np.ndarray) -> np.ndarray:
        """Score windows by weighted distance to the selected components, then average."""
        if self.model_ is None or self.window_scaler_ is None or self.selected_components_ is None:
            raise RuntimeError("Model must be fitted before scoring")
        windows = rolling_windows_nd(values, self.window_size, stride=1)[:, :, 0]
        standardized = self.window_scaler_.transform(windows)
        window_scores = np.sum(
            cdist(standardized, self.selected_components_) / self.selected_weights_,
            axis=1,
        )
        return aggregate_window_scores(window_scores.astype(np.float32), values.shape[0], self.window_size)
