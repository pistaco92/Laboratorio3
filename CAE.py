#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:33:04 2026

@author: helo
"""

import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder: (1, 64, 64) -> (64, 8, 8) -> 4096 -> z (32)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),                                           # 4096 valores
            nn.Linear(64 * 8 * 8, latent_dim),                      # z de 32
        )

        # Decoder: z (32) -> 4096 -> (64, 8, 8) -> (1, 64, 64)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),                      # 4096 valores
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),                            # (64, 8, 8)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (1, 64, 64)
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    