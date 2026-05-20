"""Checkpointing, loss plotting, and side-by-side image grids."""

import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe for headless SLURM nodes
import matplotlib.pyplot as plt


def save_checkpoint(R, D, epoch: int, outdir: str = "checkpoints") -> None:
    os.makedirs(outdir, exist_ok=True)
    torch.save(R.state_dict(), f"{outdir}/refiner_epoch{epoch}.pt")
    torch.save(D.state_dict(), f"{outdir}/discriminator_epoch{epoch}.pt")


def plot_losses(history: dict, epoch: int, outdir: str = "plots") -> None:
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (key, vals) in zip(axes, history.items()):
        ax.plot(vals)
        ax.set_title(key)
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(f"{outdir}/losses_epoch{epoch}.png", dpi=120)
    plt.close(fig)


def plot_sample_grid(sim_batch, refined_batch, real_batch, epoch: int,
                     outdir: str = "plots", n: int = 6) -> None:
    """Save a grid: top row simulated, middle row refined, bottom row real.

    Inputs are torch tensors of shape (B, 1, 125, 125). Only the first `n`
    in each batch are shown so the figure stays readable.
    """
    os.makedirs(outdir, exist_ok=True)

    def _to_np(t):
        return t.detach().cpu().numpy()[:n, 0]

    sim = _to_np(sim_batch)
    ref = _to_np(refined_batch)
    real = _to_np(real_batch)

    rows = [("simulated", sim), ("refined", ref), ("real JWST", real)]
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{outdir}/samples_epoch{epoch}.png", dpi=120)
    plt.close(fig)
