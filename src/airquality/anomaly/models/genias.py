"""GenIAS: TCN-based VAE that generates realistic anomalous time-series windows.

The VAE reconstructs normal windows (``x_hat``) and, by widening the latent
scale with a learned ``psi`` factor, produces perturbed variants (``x_tilde``)
that :func:`patch_anomalies` splices into the original window. The ``*GenIAS``
detector variants use these patched windows as synthetic anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Trims convolution padding so temporal outputs keep the intended length."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Dilated causal convolution block with residual connection (TCN unit)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.residual(x))


class TCNEncoder(nn.Module):
    """Stacked TemporalBlocks with doubling dilation, average-pooled to a vector."""

    def __init__(self, channels: list[int], kernel_size: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        for idx, out_channels in enumerate(channels):
            in_channels = 1 if idx == 0 else channels[idx - 1]
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** idx,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        pooled = self.pool(features)
        return pooled.squeeze(-1)


class ConvTransposeDecoder(nn.Module):
    """Upsampling decoder: latent vector → reconstructed window of ``window_size``."""

    def __init__(self, latent_dim: int, window_size: int, base_channels: int = 128) -> None:
        super().__init__()
        self.window_size = window_size
        self.base_length = window_size // 8
        self.proj = nn.Linear(latent_dim, base_channels * self.base_length)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        x = x.view(z.shape[0], 128, self.base_length)
        x = self.decoder(x)
        if x.shape[-1] != self.window_size:
            x = F.interpolate(x, size=self.window_size, mode="linear", align_corners=False)
        return x


@dataclass
class GeniasLossConfig:
    """Weights and margins for the GenIAS training loss (see ``compute_losses``)."""

    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.0
    zeta: float = 0.1
    delta_min: float = 0.1
    delta_max: float = 0.2
    sigma_prior: float = 0.5
    tau: float = 0.2


class GenIAS(nn.Module):
    """VAE over windows whose ``psi``-widened latent samples yield perturbed outputs.

    ``forward`` returns both the faithful reconstruction ``x_hat`` (from ``z``)
    and the perturbed variant ``x_tilde`` (from ``z_tilde``, sampled with the
    learned variance-amplification factor ``psi``).
    """

    def __init__(
        self,
        window_size: int = 200,
        latent_dim: int = 50,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        self.window_size = window_size
        self.encoder = TCNEncoder(channels=channels, kernel_size=kernel_size, dropout=dropout)
        hidden_dim = channels[-1]
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.psi_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = ConvTransposeDecoder(latent_dim=latent_dim, window_size=window_size, base_channels=hidden_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden).clamp(min=-8.0, max=8.0)
        psi = 1.0 + F.softplus(self.psi_head(hidden))
        return mu, logvar, psi

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps, std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar, psi = self.encode(x)
        z, std = self.reparameterize(mu, logvar)
        eps = torch.randn_like(std)
        z_tilde = mu + psi * (std * eps)
        x_hat = self.decoder(z)
        x_tilde = self.decoder(z_tilde)
        return {
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "psi": psi,
            "z": z,
            "z_tilde": z_tilde,
            "x_hat": x_hat,
            "x_tilde": x_tilde,
        }


def mse_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error per batch element (flattened over all other dims)."""
    return ((a - b) ** 2).flatten(1).mean(dim=1)


def patch_anomalies(x: torch.Tensor, x_tilde: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Splice perturbed values into ``x`` where they deviate strongly.

    Returns ``(patched_window, patch_mask)``; a timestep is patched when its
    squared deviation exceeds ``tau`` times the window amplitude.
    """
    # Compare reconstruction error at each time step so we patch local regions,
    # not entire windows whenever the total deviation is large.
    deviation = ((x - x_tilde) ** 2).sum(dim=1, keepdim=True)
    amplitude = x.amax(dim=-1, keepdim=True) - x.amin(dim=-1, keepdim=True)
    patch_mask = deviation > (tau * amplitude)
    return torch.where(patch_mask, x_tilde, x), patch_mask


def compute_losses(outputs: dict[str, torch.Tensor], x: torch.Tensor, config: GeniasLossConfig) -> dict[str, torch.Tensor]:
    """GenIAS objective: reconstruction + perturbation margins + zero-run + KL terms."""
    x_hat = outputs["x_hat"]
    x_tilde = outputs["x_tilde"]
    recon_loss = F.mse_loss(x_hat, x)

    d_x_hat = mse_per_sample(x, x_hat)
    d_x_tilde = mse_per_sample(x, x_tilde)
    triplet = F.relu(d_x_hat - d_x_tilde + config.delta_min).mean()
    realism = F.relu(d_x_tilde - config.delta_max).mean()
    perturb_loss = triplet + realism

    zero_dimensions = torch.all(x == 0.0, dim=-1, keepdim=True)
    if zero_dimensions.any():
        zero_terms = 1.0 / (((x_tilde - x) ** 2) + 1.0)
        zero_loss = zero_terms[zero_dimensions.expand_as(zero_terms)].mean()
    else:
        zero_loss = x.new_tensor(0.0)

    mu = outputs["mu"]
    logvar = outputs["logvar"]
    var = torch.exp(logvar)
    sigma_prior_sq = config.sigma_prior ** 2
    kl_terms = 1.0 + logvar - mu.pow(2) - (var / sigma_prior_sq) + 2.0 * math.log(config.sigma_prior)
    kl_loss = -0.5 * kl_terms.sum(dim=1).mean()

    total = (
        config.alpha * recon_loss
        + config.beta * perturb_loss
        + config.gamma * zero_loss
        + config.zeta * kl_loss
    )
    return {
        "loss": total,
        "recon_loss": recon_loss,
        "perturb_loss": perturb_loss,
        "zero_loss": zero_loss,
        "kl_loss": kl_loss,
    }
