"""Loss functions for SimGAN.

Discriminator loss: standard binary cross-entropy on real (label 1) vs
refined (label 0) images.

Refiner loss: adversarial term (BCE pushing D(refined) -> 1) plus a
preservation term that keeps the refined image close to the original
simulation. Two preservation options:

  MSE  — mean squared pixel error. Simple, but penalises noise and
          brightness changes as heavily as structural changes.
  SSIM — Structural Similarity Index. Measures luminance, contrast,
          and local structure separately. Tolerant of added noise and
          overall brightness shifts, so the refiner can adjust those
          without being penalised, while still protecting the lens arc.

The relative weight `lam` is the single most important hyperparameter.
With SSIM the preservation loss is bounded [0, 1], so lam=10 is
appropriate. With MSE the raw values are much smaller (~0.005), so
lam=200 is needed to get a comparable contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

bce = nn.BCELoss()

# --- SSIM helpers -----------------------------------------------------------

def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


_WINDOW = _gaussian_window()  # cached; moved to device on first use
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - mean SSIM over the batch.  Range [0, 1]: 0 = identical, 1 = totally different."""
    global _WINDOW
    if _WINDOW.device != x.device:
        _WINDOW = _WINDOW.to(x.device)
    w = _WINDOW.expand(x.shape[1], 1, -1, -1)  # one filter per channel
    pad = w.shape[-1] // 2

    mu_x = F.conv2d(x, w, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, w, padding=pad, groups=x.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sig_x2 = F.conv2d(x * x, w, padding=pad, groups=x.shape[1]) - mu_x2
    sig_y2 = F.conv2d(y * y, w, padding=pad, groups=x.shape[1]) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=pad, groups=x.shape[1]) - mu_xy

    ssim_map = (
        (2 * mu_xy + _C1) * (2 * sig_xy + _C2)
    ) / (
        (mu_x2 + mu_y2 + _C1) * (sig_x2 + sig_y2 + _C2)
    )
    return 1.0 - ssim_map.mean()

# ----------------------------------------------------------------------------


def discriminator_loss(D, real_images: torch.Tensor, refined_images: torch.Tensor) -> torch.Tensor:
    real_labels = torch.ones(len(real_images), device=real_images.device)
    fake_labels = torch.zeros(len(refined_images), device=refined_images.device)
    loss_real = bce(D(real_images), real_labels)
    loss_fake = bce(D(refined_images), fake_labels)
    return 0.5 * (loss_real + loss_fake)


def refiner_loss(
    D,
    refined_images: torch.Tensor,
    simulated_images: torch.Tensor,
    lam: float = 10.0,
    use_ssim: bool = True,
):
    """Returns (total, adversarial, preservation) so we can log them separately."""
    real_labels = torch.ones(len(refined_images), device=refined_images.device)
    L_adversarial = bce(D(refined_images), real_labels)
    if use_ssim:
        L_preservation = ssim_loss(refined_images, simulated_images)
    else:
        L_preservation = torch.mean((refined_images - simulated_images) ** 2)
    return L_adversarial + lam * L_preservation, L_adversarial, L_preservation
