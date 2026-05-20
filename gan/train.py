"""SimGAN training loop.

Each iteration does:
  1. Generate a batch of refined images from the simulator output.
  2. Update the discriminator on (real, refined.detach()).
  3. Recompute refined (with grad this time) and update the refiner
     using L_adversarial + lam * L_preservation.

The two `.detach()` / re-forward steps look redundant but they are the
standard pattern: detaching prevents the discriminator update from
flowing gradients back into the refiner, and re-forwarding gives the
refiner update a fresh graph to backprop through.

CLI flags let you run a fast smoke test (`--smoke`) without hitting
the full datasets. This is what to use when validating that the code
runs at all before submitting a real SLURM job.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make sibling imports work whether run as `python gan/train.py` or `python train.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_real_jwst_cutouts, load_simulated_images, load_vela_images
from networks import Refiner, Discriminator
from losses import discriminator_loss, refiner_loss
from utils import save_checkpoint, plot_losses, plot_sample_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=1e-5,
                   help="Discriminator learning rate. Defaults to --lr / 4.")
    p.add_argument("--lam", type=float, default=10.0,
                   help="Weight on the lens-preservation loss. "
                        "10 is appropriate for SSIM (bounded 0-1); "
                        "use ~200 if switching to --pres-loss mse.")
    p.add_argument("--real-source", choices=["jwst", "vela"], default="vela",
                   help="Which dataset to use as the 'real' target for the discriminator. "
                        "'vela' uses VELA cosmological sim images (cleaner, no blank frames). "
                        "'jwst' uses raw GOODS-S JWST cutouts (original behaviour).")
    p.add_argument("--pres-loss", choices=["ssim", "mse"], default="ssim",
                   help="Preservation loss type. 'ssim' is tolerant of noise and brightness "
                        "shifts; 'mse' penalises every pixel change equally.")
    p.add_argument("--max-real", type=int, default=5000)
    p.add_argument("--max-sim", type=int, default=10000)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--outdir", type=str, default=".",
                   help="Where to write checkpoints/ and plots/.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run (2 epochs, 64 images each) to validate the pipeline.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke:
        args.epochs = 2
        args.max_real = 64
        args.max_sim = 64
        args.batch_size = 16
        args.checkpoint_every = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: training on CPU will be extremely slow. Submit a GPU SLURM job.")

    use_ssim = args.pres_loss == "ssim"
    print(f"real source:       {args.real_source}")
    print(f"preservation loss: {args.pres_loss}  (lam={args.lam})")
    print("Loading data...")
    if args.real_source == "vela":
        real_np = load_vela_images(max_images=args.max_real)
    else:
        real_np = load_real_jwst_cutouts(max_cutouts=args.max_real, stride=10)
    sim_np = load_simulated_images(max_images=args.max_sim)
    print(f"  real:      {real_np.shape}")
    print(f"  simulated: {sim_np.shape}")

    real_t = torch.from_numpy(real_np).unsqueeze(1)  # (N, 1, 125, 125)
    sim_t = torch.from_numpy(sim_np).unsqueeze(1)

    real_loader = DataLoader(TensorDataset(real_t), batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    sim_loader = DataLoader(TensorDataset(sim_t), batch_size=args.batch_size,
                            shuffle=True, drop_last=True)

    R = Refiner().to(device)
    D = Discriminator().to(device)

    lr_d = args.lr_d if args.lr_d is not None else args.lr / 4
    opt_R = torch.optim.Adam(R.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    history = {"D_loss": [], "R_adv": [], "R_pres": []}

    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    plot_dir = os.path.join(args.outdir, "plots")

    for epoch in range(1, args.epochs + 1):
        # The two loaders may have different lengths; zip stops at the shorter.
        # That is fine — we just see slightly fewer of one set per epoch.
        for (real_batch,), (sim_batch,) in zip(real_loader, sim_loader):
            real_batch = real_batch.to(device)
            sim_batch = sim_batch.to(device)

            # --- Discriminator update ---
            with torch.no_grad():
                refined_detached = R(sim_batch)
            opt_D.zero_grad()
            d_loss = discriminator_loss(D, real_batch, refined_detached)
            d_loss.backward()
            opt_D.step()

            # --- Refiner update ---
            opt_R.zero_grad()
            refined = R(sim_batch)
            r_loss, r_adv, r_pres = refiner_loss(D, refined, sim_batch, args.lam, use_ssim=use_ssim)
            r_loss.backward()
            opt_R.step()

            history["D_loss"].append(d_loss.item())
            history["R_adv"].append(r_adv.item())
            history["R_pres"].append(r_pres.item())

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"D_loss={d_loss.item():.4f} | "
              f"R_adv={r_adv.item():.4f} | "
              f"R_pres={r_pres.item():.4f}")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(R, D, epoch, outdir=ckpt_dir)
            plot_losses(history, epoch, outdir=plot_dir)
            # Side-by-side qualitative check.
            with torch.no_grad():
                R.eval()
                sample_sim = sim_batch[:6]
                sample_refined = R(sample_sim)
                R.train()
            plot_sample_grid(sample_sim, sample_refined, real_batch[:6],
                             epoch, outdir=plot_dir)

    print("Training complete.")


if __name__ == "__main__":
    main()
