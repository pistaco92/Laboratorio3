"""Arquitecturas CAE, CVAE y CAAE para rostros 64x64 en escala de grises."""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_conv_encoder_backbone() -> nn.Sequential:
    """Bloque convolucional compartido: (1, 64, 64) -> 4096."""
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # (16, 32, 32)
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 16, 16)
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 8)
        nn.ReLU(),
        nn.Flatten(),                                           # 4096
    )


def _build_conv_decoder(latent_dim: int) -> nn.Sequential:
    """Decodificador compartido (láminas 24-25 y A5-A6): z (d) -> (1, 64, 64)."""
    return nn.Sequential(
        nn.Linear(latent_dim, 64 * 8 * 8),
        nn.ReLU(),
        nn.Unflatten(1, (64, 8, 8)),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (16, 32, 32)
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # (1, 64, 64)
        nn.Sigmoid(),
    )


class CAE(nn.Module):
    """Convolutional Autoencoder determinista."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            _build_conv_encoder_backbone(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )
        self.decoder = _build_conv_decoder(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class CVAE(nn.Module):
    """Convolutional Variational Autoencoder (láminas 33-38, A10-A12)."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = _build_conv_encoder_backbone()
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.decoder = _build_conv_decoder(latent_dim)

    def encode_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Código determinista mu(x) para clasificación (lámina 33)."""
        mu, _ = self.encode_params(x)
        return mu

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def loss_cvae(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pérdida del CVAE según lámina A12: suma por imagen, media del lote."""
    rec = (x_hat - x).square().flatten(1).sum(dim=1)
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1)
    total = (rec + beta * kl).mean()
    return total, rec.mean(), kl.mean()


class LatentDiscriminator(nn.Module):
    """Discriminador MLP sobre el vector latente z (lámina A13)."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CAAE(nn.Module):
    """Convolutional Adversarial Autoencoder (láminas 44-47, A13-A16)."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            _build_conv_encoder_backbone(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )
        self.decoder = _build_conv_decoder(latent_dim)
        self.discriminator = LatentDiscriminator(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def build_autoencoder(kind: str, latent_dim: int) -> nn.Module:
    """Crea CAE, CVAE o CAAE para la dimensión latente indicada."""
    kind = kind.lower()
    if kind == "cae":
        return CAE(latent_dim=latent_dim)
    if kind == "cvae":
        return CVAE(latent_dim=latent_dim)
    if kind == "caae":
        return CAAE(latent_dim=latent_dim)
    raise ValueError(f"Codificador desconocido: {kind}. Usa 'cae', 'cvae' o 'caae'.")
