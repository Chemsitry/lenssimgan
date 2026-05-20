"""Refiner and Discriminator architectures for SimGAN.

Both networks are intentionally small. With only thousands of training
images, a large network would overfit the discriminator instantly and
collapse training. Four residual blocks give the refiner enough capacity
to learn local texture/noise corrections without being able to invent
new structure that would destroy the lens.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN with an additive skip connection.

    The skip means the block learns a *correction* to its input, so a
    randomly initialised refiner starts close to the identity map and
    L_preservation is small from the very first iteration.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Refiner(nn.Module):
    """Image-to-image refiner: 125x125 sim -> 125x125 refined, both in [0, 1]."""

    def __init__(self, n_residual_blocks: int = 4, channels: int = 64):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_residual_blocks)]
        )
        self.exit = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.residuals(x)
        x = self.exit(x)
        return x


class Discriminator(nn.Module):
    """Classifies a 125x125 image as real (1) or refined (0).

    Three stride-2 convolutions take the spatial resolution from
    125 -> 62 -> 31 -> 15, then a single 15x15 convolution collapses
    it to a scalar logit. Sigmoid maps it to a probability in [0, 1]
    so we can pair it with the BCE losses defined in losses.py.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),    # 62
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # 31
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 15
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(128, 1, kernel_size=15),                        # 1
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


if __name__ == "__main__":
    # Sanity check: forward a random batch through both networks.
    x = torch.randn(4, 1, 125, 125)
    R = Refiner()
    D = Discriminator()
    refined = R(x)
    score = D(refined)
    print(f"Refiner out: {refined.shape}, range [{refined.min():.3f}, {refined.max():.3f}]")
    print(f"Discriminator out: {score.shape}, range [{score.min():.3f}, {score.max():.3f}]")
    n_R = sum(p.numel() for p in R.parameters())
    n_D = sum(p.numel() for p in D.parameters())
    print(f"Params: Refiner={n_R:,}, Discriminator={n_D:,}")
