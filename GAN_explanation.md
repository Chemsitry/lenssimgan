# Understanding the Simulation and the GAN

*Written for a physics undergraduate who is new to both coding and machine learning.*

---

## Part 1: How the existing simulation works

### The physical picture

A **gravitational lens** forms when a massive galaxy (the "lens") sits between us and a more distant galaxy (the "source"). The lens galaxy's gravity bends the light from the source, distorting it into arcs, rings, or multiple images. Einstein's general relativity predicts the bending angle precisely. The key number is the **Einstein radius** θ_E: the angular scale at which the lensing effect is most pronounced. If the source sits perfectly behind the lens, you get a full Einstein ring. In practice, the alignment is rarely perfect, so you get arcs.

The simulation in `simulations.ipynb` is a **forward model**: given the physical parameters of a lens-source system, it uses the `lenstronomy` library to compute what the JWST camera would observe. This is essentially ray-tracing — it traces light rays backwards from the detector through the lens plane to the source plane.

### The lens model: Singular Isothermal Ellipse (SIE)

The galaxy is modelled as a mass distribution with a density that falls off as 1/radius (like the isothermal sphere in thermodynamics). The ellipse part means it can be squashed in one direction. The only parameter controlling the strength of lensing is the Einstein radius θ_E, which is derived from the **velocity dispersion** σ_v of the lens galaxy (how fast stars move around inside it):

```
θ_E ∝ σ_v² × D_LS / D_S
```

Where D_LS and D_S are angular diameter distances between lens-source and observer-source respectively. The code calculates these using `lenstronomy`'s `LensCosmo` class.

### Where the parameters come from

The simulation does not invent galaxy properties: it samples them from real surveys.

- **Lens galaxy properties** (mass, ellipticity, size, redshift): drawn from the **Cosmo DC2** cosmological simulation catalog, which models realistic galaxy populations.
- **Source galaxy appearance**: taken from **VELA** simulations, which are mock JWST images of high-redshift star-forming galaxies rendered with JWST's actual pixel scale and filters.
- **Environmental galaxies** (random field objects in the image): sampled from the **JADES** catalog of real JWST detections.

### What the simulation outputs

For each simulated system, the code saves a 125 × 125 pixel image. This is the number of pixels that fits a JWST NIRCam field at 0.032 arcseconds per pixel. The code also saves several diagnostic images and physical parameters (θ_E, z_lens, z_source, halo mass).

For each lens it saves two versions: the full image with the lens, and the same image without the lensing distortion (the "no-lens" version). These pairs are useful for measuring how much the lensing actually changes the image.

---

## Part 2: What a GAN is

### The intuitive picture (the forger and the detective)

A **Generative Adversarial Network** (GAN) is a framework involving two competing networks:

- The **Generator** (G): tries to produce fake data that looks real.
- The **Discriminator** (D): tries to tell real data apart from fakes.

They play a game. G gets better at fooling D. D gets better at catching G. In the end, G produces outputs so realistic that D cannot distinguish them from real data.

This is analogous to a forger trying to paint counterfeit Renaissance paintings, and an art detective learning to spot fakes. As the detective improves, the forger must improve too. If the game reaches equilibrium, the forger's paintings are indistinguishable from originals.

In math terms, you train D to minimise:

```
L_D = -[ log D(real) + log(1 - D(G(noise))) ]
```

And train G to minimise (i.e. maximise D's error):

```
L_G = -log D(G(noise))
```

Where log is the natural logarithm. These two loss functions fight each other.

### Why GANs matter for physics

Simulations like the one in this repository are expensive computationally (each image takes seconds to generate) and may not perfectly match the real telescope data. GANs offer two things:

1. **Domain adaptation**: closing the gap between simulated and real data ("sim-to-real transfer").
2. **Fast surrogate generation**: once trained, a neural network can generate new images in milliseconds instead of seconds.

---

## Part 3: The GAN we are building (SimGAN)

### The core idea

Rather than replacing the physics simulator with a neural network, we keep the simulator exactly as it is and add a small **Refiner** network on top of it. The Refiner learns to make simulator outputs look more like real JWST images. The Discriminator then judges whether a given 125 × 125 image is a real JWST cutout or a refined simulation.

```
[Physics Simulator] → raw simulated image
                           ↓
                      [Refiner R]   ← trained by adversarial loss
                           ↓
                    refined image   → [Discriminator D] ← also trained
                                              ↑
                             real JWST images (from /global/cfs/...)
```

This design is called **SimGAN** (Shrivastava et al., 2017, Apple Research). The advantage is that the physics is preserved: the Refiner is only allowed to make small corrections to the image, so the Einstein ring or arc will still be there.

### The loss function in detail

The Refiner is trained to minimise:

```
L_refiner = L_adversarial + λ × L_preservation
```

**L_adversarial** measures how well the Refiner fools the Discriminator. Specifically:

```
L_adversarial = -log D(R(x_sim))
```

This is the Binary Cross-Entropy loss asking "what probability does D assign to R(x_sim) being real?". If D is fooled into saying probability ≈ 1, the loss is near zero. If D correctly identifies it as fake (probability ≈ 0), the loss is large. So minimising this pushes the Refiner toward images that D cannot distinguish from JWST observations.

**L_preservation** measures how much the Refiner has changed the simulated image:

```
L_preservation = mean( (R(x_sim) - x_sim)² )
```

This is the mean squared error (MSE) between the refined and original images. Minimising this forces the Refiner to make only small adjustments — preserving the lens arc, ring, or Einstein cross that the physics simulation produced. Without this term, the Refiner could just output a random JWST image, which would fool the Discriminator but would no longer be a lens image.

**λ** (lambda) is a number you choose (start with λ = 10). It controls the trade-off: a large λ means "preserve the lens structure very strictly"; a small λ means "make it look real even if you have to change it quite a bit".

This exactly matches the requirement: *"the simulator loss should be a function of the discriminator loss against real JWST data, plus a factor that ensures the images continue looking like lenses."*

---

## Part 4: Advice for a physics undergraduate learning coding and ML

### On learning Python and numerical computing

If you are not yet comfortable with Python, the absolute minimum you need for this project is:

1. **NumPy** — think of it as doing physics calculations on arrays. A JWST image is just a 2D NumPy array of pixel values (floating point numbers). Everything in the simulation is NumPy under the hood.
2. **Matplotlib** — for plotting. `plt.imshow(image)` is your best debugging tool. If something is wrong, look at the image before reading any code.
3. **Astropy** — for reading FITS files (the standard format in astronomy). `fits.open(filename)[0].data` gives you the NumPy array.

Good free resources:
- *Software Carpentry* (software-carpentry.org) — Python and NumPy, taught to scientists
- *Astropy tutorials* (learn.astropy.org) — practical examples in an astronomy context
- *fast.ai* (course.fast.ai) — the best practical ML course for scientists. Uses PyTorch. Start here before reading any paper.

### On understanding PyTorch (what the GAN is coded in)

PyTorch is the main framework for deep learning in research. The key idea is the **computational graph**: when you call operations on `torch.Tensor` objects, PyTorch records all the operations. When you call `.backward()`, it automatically computes the gradient of a loss with respect to every parameter. This is called **automatic differentiation**. You do not need to derive gradients by hand.

The two things you must understand before touching the training code:

1. **`tensor.backward()`**: computes all gradients. Call this once per loss per update step.
2. **`optimizer.step()`**: updates the parameters using the computed gradients (e.g. gradient descent with momentum). Call this after `backward()`.
3. **`optimizer.zero_grad()`**: clears the gradients from the previous step. Call this before `backward()` every time, or gradients will accumulate.

If you mix up the order of these three calls, the network will train incorrectly. The order is always: `zero_grad()` → compute loss → `backward()` → `step()`.

### On understanding what the networks are learning

The **Discriminator** is just an image classifier. It takes a 125×125 image and outputs a single number between 0 and 1: the estimated probability that the image is real JWST data. Internally it applies a series of convolution operations that extract features (edges, textures, brightness patterns) and progressively summarises the image down to a single score.

The **Refiner** is an image-to-image network. It takes a 125×125 image and outputs a 125×125 image. The residual architecture means it learns *what to add* to the input image rather than learning the full output image from scratch. Mathematically: `output = input + correction`. This is stable to train and naturally satisfies the preservation constraint.

### On GAN training instability

GANs are notoriously difficult to train. The two most common failure modes are:

- **Mode collapse**: the Generator produces only one or two types of images that fool the Discriminator, ignoring the full variety in the training data.
- **Discriminator winning too fast**: if D becomes perfect early in training, it gives G/R a gradient signal of zero, so R stops learning.

Watch the loss curves carefully:
- Healthy training: D_loss ≈ 0.5, R_adv ≈ 0.5-0.7 (neither winning decisively).
- D winning: D_loss → 0.
- R winning: D_loss → 1.

If training collapses, the first things to try (in order): reduce the discriminator learning rate, add dropout to the Discriminator, or increase λ.

### On running code at NERSC

NERSC is a supercomputer at Lawrence Berkeley National Lab. Your files live in two places:

- Your home directory: `/global/homes/f/forrestc/` (small, 40 GB limit)
- Project directory: `/global/cfs/projectdirs/deepsrch/` (large, shared with the group)

The data (simulated images, JWST images) is in the project directory. You should never run Python directly on a login node for a training job — the login nodes are shared and running heavy computation there is against the rules and will be killed anyway. Instead, submit a **SLURM batch job** using the `sbatch` command. There is a template script in `gan/run_gan.sh`.

When developing and debugging, you can use an **interactive job**:

```bash
salloc --account=deepsrch --constraint=gpu --gpus=1 --time=01:00:00
```

This gives you a GPU node to work on interactively for 1 hour.

### On reading the existing notebooks

Open `simulations.ipynb` in Jupyter (or JupyterLab, available via the NERSC web portal at jupyter.nersc.gov). Run the cells in order. The key function to read carefully is `simulate()`. It is long, but follow the physical steps:

1. Sample redshifts and mass.
2. Calculate σ_v and θ_E.
3. Pick a VELA source image.
4. Build the lenstronomy model.
5. Generate the image.

Once you understand that one function, you understand the entire simulation.

### On the gap between the simulation and real data

The simulation is physically accurate in the lens geometry but imperfect in other ways:

- The PSF model is a Gaussian — real JWST PSFs have diffraction spikes and asymmetric structure.
- The noise model is simplified — real JWST data has correlated noise from the detector electronics.
- The source morphologies from VELA may not represent all real high-z galaxies.

These are precisely the gaps that the GAN Refiner is trying to close. It learns the empirical difference between the simulation and real data without us having to model each discrepancy explicitly. That is the power of the adversarial approach.

---

## Summary in one paragraph

The existing simulator is a physics engine: given realistic galaxy properties drawn from surveys, it traces light rays through a gravitational lens and renders a 125×125 pixel JWST-like image. The GAN we are building wraps a small neural network (the Refiner) around the simulator output. The Refiner learns to make those images look more like real JWST observations by competing with a Discriminator network. The Refiner's loss has two parts: one that rewards fooling the Discriminator (pushing toward JWST realism), and one that penalises changing the image too much (preserving the lens physics). The result is a pipeline that can generate gravitational lens images that are both physically correct and statistically indistinguishable from real JWST data.

---

## Part 5: What was actually built (and how to defend it)

This section documents the concrete code that lives in `gan/`. It is meant to be the version you read just before the presentation.

### Files

- `gan/data_loader.py` — loads real JWST cutouts and simulated images.
- `gan/networks.py` — the Refiner and Discriminator architectures.
- `gan/losses.py` — the BCE-based GAN losses + the preservation MSE term.
- `gan/utils.py` — checkpoint saving, loss plotting, side-by-side image grids.
- `gan/train.py` — the training loop with a CLI.
- `gan/run_gan.sh` — SLURM batch script for NERSC GPU nodes.

### Implementation choices and why

**Real-image source.** Cutouts are taken from `goods_s_F115W_2018_08_29.fits` (JWST NIRCam F115W, GOODS-S field) at positions from the JADES SF mock catalog (`JADES_SF_mock_r1_v1.1.fits`, 302,515 sources). I pull 5,000 cutouts by default with a stride of 10 over the catalog so the cutouts are spread across the field rather than clustered. The catalog columns are uppercase `RA`/`DEC` (the original plan said lowercase; this would have failed silently).

**Catalog choice.** I use the SF (star-forming) mock — these are the high-z galaxies that dominate the JWST observations the simulator is designed to imitate. The Q (quiescent) catalog would also work; combining both is a reasonable extension if the refiner shows obvious mode collapse later.

**Normalisation.** Per-image min-max to [0, 1]. This puts both real and simulated images on the same dynamic range without committing to a flux calibration. The simulator output tends to be very dark (mean ≈ 0.008 after normalisation); the real cutouts are brighter on average (mean ≈ 0.087) because most cutouts contain a real source. That is the gap the refiner is supposed to close.

**Refiner architecture.** Entry conv → 4 residual blocks (64 channels) → 1×1 exit conv → sigmoid. ~297K parameters. Residual blocks mean the network learns a *correction* on top of the input, so even at random initialisation the refined image is close to the simulated input. This is the structural reason `L_preservation` is small from iteration 1 — the network is biased toward the identity map, which is a deliberate defence against the refiner destroying the lens arc.

**Discriminator architecture.** Three stride-2 convolutions (32→64→128 channels) collapse the image from 125×125 down to 15×15, then a single 15×15 conv produces a scalar logit, then sigmoid. ~194K parameters. LeakyReLU (slope 0.2) and BatchNorm follow standard DCGAN practice.

**Why both networks are deliberately small.** With only ~5,000 real cutouts and ~10,000 lensed simulations, a large discriminator would memorise the real set within a couple of epochs and start outputting near-1 for real and near-0 for refined for every input — at which point the refiner's gradient vanishes and training stops. The first lever to pull if D wins too quickly is to lower its learning rate or add dropout; the second is to shrink it further.

**Optimiser.** Adam, learning rate 1e-4, betas (0.5, 0.999). The β₁ of 0.5 is the standard DCGAN choice — the default 0.9 momentum is too high for the noisy gradient signal in adversarial training and tends to make training oscillate.

**Loss weight λ.** Default 10. The single most important hyperparameter. If refined images look identical to simulated, λ is too high — drop to 1. If refined images look like generic noise that has lost the lens, λ is too low — raise to 50.

**Training/discriminator update order.** Per iteration: compute `R(sim).detach()` → step the discriminator on (real, refined.detach()) → recompute `R(sim)` with grad → step the refiner. The detach prevents discriminator gradients flowing into the refiner, and the recompute gives the refiner update a clean computational graph.

### The `lenssim` conda env

The simulator notebooks already use a conda env called `lenssim` at `/global/homes/f/forrestc/.conda/envs/lenssim`. It had everything needed for data loading (astropy, numpy, matplotlib) but not torch. I installed `torch` and `torchvision` into it via `pip` so the same env can run both the simulator notebooks and the GAN. Activate with:

```bash
module load conda
conda activate lenssim
```

### How to run

**Smoke test (~30 seconds, no SLURM, validates the pipeline):**

```bash
cd /global/u2/f/forrestc/small-lens-forecast-sims/gan
python train.py --smoke --outdir _smoke_out
```

This trains for 2 epochs on 64 real and 64 simulated images and writes sample images and loss plots to `_smoke_out/`. Use this whenever you change the code, before submitting a real job. Already verified to run end-to-end on this system.

**Full training (SLURM, ~30 min – 2 hours on a single GPU):**

```bash
cd /global/u2/f/forrestc/small-lens-forecast-sims/gan
sbatch run_gan.sh
```

To override defaults from the SLURM submission:

```bash
sbatch run_gan.sh --epochs 30 --lam 20
```

Logs land in `gan/logs/simgan_<jobid>.out`.

### What to look at after training

1. `plots/losses_epoch*.png` — three subplots: D_loss, R_adv, R_pres.
   - Healthy: D_loss bouncing around 0.5, R_adv around 0.5–0.7, R_pres slowly drifting down.
   - D winning (loss → 0): refiner gets no signal. Lower D's learning rate or add dropout.
   - R winning (D_loss → 1): D is too weak. Train D for more steps per refiner step.

2. `plots/samples_epoch*.png` — three rows of 6 images each: simulated, refined, real. The refined row should look noisier/less idealised than the simulated row but should still clearly show the lens arc. If the refined images look identical to simulated, λ is too high. If they no longer show a lens at all, λ is too low.

3. `checkpoints/refiner_epoch*.pt` — the network weights. Load with `R = Refiner(); R.load_state_dict(torch.load(path))` if you want to apply the refiner to new simulator output.

### Honest limitations to mention in the presentation

- **Small real-image set.** 5,000 cutouts of mostly faint sources is a small training distribution. The refiner can only learn the kinds of noise/PSF effects that show up frequently in those cutouts.
- **Single filter.** Only F115W. A multi-filter setup would need multi-channel inputs and a matched simulator output.
- **No quantitative metric yet.** Beyond loss curves and qualitative side-by-side image grids, this implementation does not yet report a quantitative gain (e.g. running the original lens classifier on refined vs raw simulated images to see whether precision/recall on real data improves). This is the most natural next experiment.
- **The discriminator does not validate that the lens is preserved.** The preservation term in the loss is the *only* thing keeping the arc intact. If λ is misjudged, the refined images can be physically meaningless even if they look real.

### What I would change with more time

1. Use a percentile-based intensity stretch (clip to [1st, 99th] percentile, then scale) instead of pure min-max — robust to single bright pixels in the cutouts.
2. Train the discriminator for k steps per refiner step (k ≈ 2 or 3) if training is unstable — this is a standard GAN-training trick.
3. Add a "history buffer" of past refined images (Shrivastava et al.'s original SimGAN trick) to stop the refiner from cycling through tricks that exploit a stale discriminator.
4. Replace BCE with the WGAN-GP loss for more stable training. Worth doing if the BCE version oscillates badly.
