# GAN Implementation Plan

## What we are building

A **SimGAN** (Simulation + Generative Adversarial Network). The existing physics simulator already generates realistic gravitational lens images. Rather than replacing it with a neural network, we keep it and add a small **Refiner** network on top that learns to make the simulated images match real JWST observations. A **Discriminator** network then judges whether a given image is real JWST data or a refined simulation, and the loss from this judgment drives the Refiner's training.

```
Simulator (physics, fixed) → Refiner (trainable) → Discriminator (trainable)
                                                              ↕
                                              Real JWST images (from NERSC)
```

The total loss for training the Refiner is:

```
L_refiner = L_adversarial + λ × L_lens_preservation
```

- `L_adversarial`: how well the Discriminator is fooled (pushes refined images toward real JWST)
- `L_lens_preservation`: pixel-level difference between refined and original simulated image (prevents the Refiner from destroying the lens structure)
- `λ`: a tunable weighting factor (start with `λ = 10`)

---

## Files to create

```
small-lens-forecast-sims/
├── gan/
│   ├── data_loader.py        # Load real JWST images and simulated images
│   ├── networks.py           # Refiner and Discriminator architectures
│   ├── losses.py             # Loss functions
│   ├── train.py              # Main training loop
│   └── utils.py              # Saving checkpoints, plotting losses
├── GAN_plan.md               # This file
└── GAN_explanation.md        # Beginner explanation document
```

---

## Step-by-step implementation

### Step 1: Set up the environment

Check that you have the required packages. The existing `environment.yml` already covers `numpy`, `astropy`, `scipy`, `lenstronomy`. You will need to add:

```bash
conda install pytorch torchvision -c pytorch
conda install matplotlib
```

Or if PyTorch is already available on NERSC modules:

```bash
module load pytorch
```

NERSC tip: run all training jobs via SLURM batch scripts, not interactively. There is a template in Step 6.

---

### Step 2: Load real JWST data (`gan/data_loader.py`)

Real JWST images live at:

```
/global/cfs/projectdirs/deepsrch/jwst_sims/data/
```

From the instructions files in this repo, the relevant JWST image is the **Williams/JADES field** FITS file. We need to cut it into 125×125 pixel postage stamps (same size as the simulations) centred on detected sources in the JADES catalog.

```python
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

JWST_IMAGE_PATH = "/global/cfs/projectdirs/deepsrch/jwst_sims/data/JWST/goods_s_F115W_2018_08_29.fits"
JADES_CATALOG_PATH = "/global/cfs/projectdirs/deepsrch/jwst_sims/data/JWST/JADES_SF_mock_r1_v1.1.fits"
SIM_DATA_PATH = "/global/cfs/projectdirs/deepsrch/jwst_sims/sims.15-.5/images.npy"

CUTOUT_SIZE = 125  # pixels, matches simulations

def load_real_jwst_cutouts(max_cutouts=5000):
    """Cut 125x125 postage stamps from the Williams JWST image."""
    hdu = fits.open(JWST_IMAGE_PATH)[0]
    wcs = WCS(hdu.header)
    image_data = hdu.data

    cat = fits.open(JADES_CATALOG_PATH)[1].data
    ras, decs = cat['ra'], cat['dec']

    cutouts = []
    for ra, dec in zip(ras, decs):
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        try:
            cutout = Cutout2D(image_data, coord, CUTOUT_SIZE, wcs=wcs)
            stamp = cutout.data
            if stamp.shape == (CUTOUT_SIZE, CUTOUT_SIZE):
                cutouts.append(stamp)
        except Exception:
            continue
        if len(cutouts) >= max_cutouts:
            break

    cutouts = np.array(cutouts, dtype=np.float32)
    # Normalise each stamp to [0, 1]
    for i in range(len(cutouts)):
        lo, hi = cutouts[i].min(), cutouts[i].max()
        if hi > lo:
            cutouts[i] = (cutouts[i] - lo) / (hi - lo)
    return cutouts

def load_simulated_images(lens_only=True):
    """Load the simulator output, returning only lensed images."""
    images = np.load(SIM_DATA_PATH)       # shape (20000, 125, 125)
    labels = np.load(SIM_DATA_PATH.replace("images", "lensed"))  # shape (20000,)
    if lens_only:
        images = images[labels == 1]
    # Normalise
    for i in range(len(images)):
        lo, hi = images[i].min(), images[i].max()
        if hi > lo:
            images[i] = (images[i] - lo) / (hi - lo)
    return images.astype(np.float32)
```

The catalog has 302,515 sources with `RA` and `DEC` columns. The image is a 14060×16160 pixel mosaic — plenty of sky to cut thousands of 125×125 stamps from. There is also a `JADES_Q_mock_r1_v1.1.fits` (quiescent galaxies) if you want to include both galaxy types.

---

### Step 3: Build the networks (`gan/networks.py`)

Keep both networks small. Bigger is not better when starting out.

#### Refiner

A simple residual convolutional network. It takes a 125×125 simulated image and outputs a 125×125 refined image of the same size. Residual connections mean it learns *corrections* rather than a complete new image, which makes `L_lens_preservation` easy to satisfy.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)   # skip connection: output = input + correction


class Refiner(nn.Module):
    def __init__(self, n_residual_blocks=4):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(n_residual_blocks)]
        )
        self.exit = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),   # keep output in [0, 1]
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.residuals(x)
        x = self.exit(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 62x62
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 31x31
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 15x15
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=15),                       # 1x1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)   # returns a score per image in [0,1]
```

---

### Step 4: Define the losses (`gan/losses.py`)

```python
import torch
import torch.nn as nn

bce = nn.BCELoss()   # Binary Cross-Entropy

def discriminator_loss(D, real_images, refined_images):
    """Train D to output 1 for real, 0 for refined."""
    real_labels  = torch.ones(len(real_images),    device=real_images.device)
    fake_labels  = torch.zeros(len(refined_images), device=refined_images.device)

    loss_real = bce(D(real_images),    real_labels)
    loss_fake = bce(D(refined_images), fake_labels)
    return (loss_real + loss_fake) / 2


def refiner_loss(D, refined_images, simulated_images, lam=10.0):
    """
    Train R to fool D, while staying close to the original simulation.

    L_refiner = L_adversarial + lam * L_lens_preservation
    """
    real_labels = torch.ones(len(refined_images), device=refined_images.device)

    # Adversarial: the refiner wants D to call its output "real"
    L_adversarial = bce(D(refined_images), real_labels)

    # Preservation: keep the lens structure intact
    L_preservation = torch.mean((refined_images - simulated_images) ** 2)

    return L_adversarial + lam * L_preservation, L_adversarial, L_preservation
```

---

### Step 5: Write the training loop (`gan/train.py`)

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from networks import Refiner, Discriminator
from losses import discriminator_loss, refiner_loss
from utils import save_checkpoint, plot_losses

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA = 10.0        # weight of the lens preservation term
N_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4

def main():
    # --- data ---
    from data_loader import load_real_jwst_cutouts, load_simulated_images
    real_np = load_real_jwst_cutouts()
    sim_np  = load_simulated_images()

    real_t = torch.tensor(real_np[:, None, :, :])   # add channel dim
    sim_t  = torch.tensor(sim_np[:, None,  :, :])

    real_loader = DataLoader(TensorDataset(real_t), batch_size=BATCH_SIZE, shuffle=True)
    sim_loader  = DataLoader(TensorDataset(sim_t),  batch_size=BATCH_SIZE, shuffle=True)

    # --- models ---
    R = Refiner().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_R = torch.optim.Adam(R.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    history = {"D_loss": [], "R_adv": [], "R_pres": []}

    for epoch in range(N_EPOCHS):
        for (real_batch,), (sim_batch,) in zip(real_loader, sim_loader):
            real_batch = real_batch.to(DEVICE)
            sim_batch  = sim_batch.to(DEVICE)

            refined = R(sim_batch).detach()   # don't backprop through R yet

            # --- update Discriminator ---
            opt_D.zero_grad()
            d_loss = discriminator_loss(D, real_batch, refined)
            d_loss.backward()
            opt_D.step()

            # --- update Refiner ---
            opt_R.zero_grad()
            refined = R(sim_batch)            # recompute with grad
            r_loss, r_adv, r_pres = refiner_loss(D, refined, sim_batch, LAMBDA)
            r_loss.backward()
            opt_R.step()

            history["D_loss"].append(d_loss.item())
            history["R_adv"].append(r_adv.item())
            history["R_pres"].append(r_pres.item())

        print(f"Epoch {epoch+1}/{N_EPOCHS} | D_loss={d_loss.item():.4f} "
              f"| R_adv={r_adv.item():.4f} | R_pres={r_pres.item():.4f}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(R, D, epoch + 1)
            plot_losses(history, epoch + 1)

if __name__ == "__main__":
    main()
```

---

### Step 6: SLURM batch script for NERSC

Save this as `gan/run_gan.sh`:

```bash
#!/bin/bash
#SBATCH --account=deepsrch
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=simgan
#SBATCH --output=logs/simgan_%j.out

module load python
conda activate <your-env-name>

cd /global/u2/f/forrestc/small-lens-forecast-sims/gan
python train.py
```

Submit with `sbatch run_gan.sh`.

---

### Step 7: Utility functions (`gan/utils.py`)

```python
import torch
import matplotlib.pyplot as plt
import os

def save_checkpoint(R, D, epoch, outdir="checkpoints"):
    os.makedirs(outdir, exist_ok=True)
    torch.save(R.state_dict(), f"{outdir}/refiner_epoch{epoch}.pt")
    torch.save(D.state_dict(), f"{outdir}/discriminator_epoch{epoch}.pt")

def plot_losses(history, epoch, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (key, vals) in zip(axes, history.items()):
        ax.plot(vals)
        ax.set_title(key)
        ax.set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(f"{outdir}/losses_epoch{epoch}.png")
    plt.close()
```

---

## Decisions you will need to make

| Decision | Suggested starting value | Why |
|---|---|---|
| `λ` (preservation weight) | 10 | Strong enough to keep lens structure, room to reduce if images look too "simulated" |
| Number of residual blocks | 4 | Enough capacity without overfitting small datasets |
| Batch size | 32 | Standard for image GANs; reduce to 16 if GPU runs out of memory |
| Learning rate | 1e-4 | Conservative starting point for Adam with GANs |
| Epochs | 50 | GANs often converge within 30-100 epochs on this scale |
| Real images source | JADES/Williams F115W | Same band as simulations |

---

## Known challenges and how to handle them

**GAN training instability** — If the discriminator loss collapses to 0 quickly, the refiner is getting no useful gradient. Solution: reduce the discriminator learning rate, or briefly freeze D for a few iterations to let R catch up.

**Blank or noisy refined images** — The refiner has collapsed or exploded. Check that `λ` is large enough to act as a stabilising anchor.

**Real images look very different from simulated** — The normalisation approach matters a lot. Both datasets must be normalised consistently. If real JWST images have a very different flux distribution, consider matching histograms before training.

**Memory** — 5000 real cutouts × 125 × 125 × 4 bytes ≈ 310 MB. The simulated set is ~2.4 GB on disk; only load what fits in RAM or stream from disk.

---

## How to verify it is working

1. After every 10 epochs, look at `plots/losses_epoch*.png`. The discriminator loss should stay near 0.5 (neither player is winning decisively). If D_loss → 0, D is winning; if D_loss → 1, R is winning.
2. Visually compare: simulated image → refined image → real JWST image. The refined image should look less "perfect" and more textured/noisy than the raw simulation.
3. Optionally re-run the classifier from the original paper on the refined images. If the precision/recall improves, the GAN is adding information.
