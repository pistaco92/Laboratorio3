"""Clasificadores 1D sobre códigos faciales z: CNN 1D propia y FCN-1D (tsai)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """CNN 1D propia según láminas 51-52, 66 y A17.

    Recibe [B, 1, d] y produce [B, n_classes] logits.
    """

    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        channels: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.extend(
            [
                nn.Conv1d(channels, channels * 2, kernel_size=kernel_size, padding=pad),
                nn.ReLU(),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(channels * 2 * latent_dim, n_classes),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        return self.net(z_seq)


class PurePyTorchFCN1D(nn.Module):
    """Implementación equivalente a FCN de tsai (lámina 56) por si tsai no está instalada."""

    def __init__(self, n_classes: int, width: int = 16):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(width * 2, width, kernel_size=3, padding=1),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width, n_classes)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        h = self.block1(z_seq)
        h = self.block2(h)
        h = self.block3(h)
        h = self.gap(h).flatten(1)
        return self.fc(h)


def build_fcn1d(n_classes: int, width: int = 16) -> tuple[nn.Module, str]:
    """Carga FCN de tsai (lámina A18) o usa PurePyTorchFCN1D si tsai no está disponible."""
    layers = [width, width * 2, width]
    kss = [7, 5, 3]
    try:
        from tsai.models.FCN import FCN

        return FCN(c_in=1, c_out=n_classes, layers=layers, kss=kss), "tsai"
    except ImportError:
        return PurePyTorchFCN1D(n_classes=n_classes, width=width), "pytorch_fallback"
