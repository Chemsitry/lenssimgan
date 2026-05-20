# How to Run the SimGAN Training

*Written for a physics undergraduate who is new to coding. Read every section before running anything.*

---

## Before you type anything: supercomputer rules

NERSC (this machine) is a shared supercomputer used by hundreds of scientists simultaneously. There are two types of nodes (computers within the cluster):

- **Login nodes** — the machine you land on when you SSH in. Shared by everyone. Think of this as the lobby of a library. You can browse, navigate, and type short commands here. You must **never** run training or heavy computation here; it slows down the machine for everyone and NERSC will kill your process.
- **Compute nodes** — dedicated machines you request from the job scheduler (SLURM). This is where the actual work happens. You request one, it is yours alone for the duration of your job.

The safe workflow is always:
1. Navigate and set things up on the login node (fast, light commands only).
2. Submit a job to SLURM, which runs the heavy work on a compute node.

---

## Step 1: Load your conda environment

Every time you start a new terminal session, you need to activate the software environment that has all the necessary libraries installed. Think of it like loading a specific set of lab instruments before starting an experiment.

```bash
module load conda
conda activate lenssim
```

**What these do:**
- `module load conda` — makes the `conda` command available. NERSC organises software into "modules"; you have to load them before using them.
- `conda activate lenssim` — switches your Python installation to the `lenssim` environment, which has `lenstronomy`, `astropy`, `numpy`, `matplotlib`, and `torch` (PyTorch) installed. Without this, `import torch` would fail because Python would not know where to find the library.

You should see your command prompt change to include `(lenssim)` at the start, confirming the environment is active.

---

## Step 2: Navigate to the GAN folder

```bash
cd /global/u2/f/forrestc/small-lens-forecast-sims/gan
```

**What this does:** `cd` stands for "change directory". This moves your terminal's working location into the `gan/` folder where all the training code lives. Everything from this point on assumes you are in this folder.

You can confirm you are in the right place with:

```bash
ls
```

You should see: `train.py`, `run_gan.sh`, `networks.py`, `data_loader.py`, `losses.py`, `utils.py`, and folders `checkpoints/`, `logs/`, `plots/`.

---

## Step 3: Run the smoke test first (ALWAYS do this before submitting a real job)

```bash
python train.py --smoke --outdir _smoke_out
```

**What this does, word by word:**
- `python` — runs the Python interpreter (the program that reads and executes Python code).
- `train.py` — the training script written for this project.
- `--smoke` — a special flag that tells the script to run in "smoke test" mode: only 2 training epochs, only 64 images of each type, batch size of 16. This is designed to finish in about 30 seconds and verify that the entire pipeline runs without crashing before you commit to a real multi-hour job.
- `--outdir _smoke_out` — tells the script to write all its output (sample images, loss plots, checkpoints) into a folder called `_smoke_out` so it does not clutter the main output folders.

**Why always run the smoke test first?** If there is a typo in the code, a missing file, or a broken library, you will find out in 30 seconds rather than discovering the job crashed after 2 hours on a GPU node that you waited 20 minutes to get.

**What you should see while it runs:**

```
Using device: cpu
Loading data...
  real:      (64, 125, 125)
  simulated: (64, 125, 125)
Epoch   1/2 | D_loss=0.6932 | R_adv=0.6931 | R_pres=0.0041
Epoch   2/2 | D_loss=0.6843 | R_adv=0.7201 | R_pres=0.0038
Training complete.
```

The exact numbers will differ, but it should complete without error. If it does, the pipeline is working.

**If you see an error about files not found**, check that the FITS data files are still in their expected locations. The script looks for the JADES catalog and the GOODS-S image in `/global/cfs/projectdirs/deepsrch/`. Ask your supervisor if those paths have changed.

---

## Step 4: Submit the real training job to SLURM

Once the smoke test passes, you submit the actual training run. This runs on a dedicated GPU node instead of your laptop or the login node, so it is fast and does not bother anyone.

```bash
sbatch run_gan.sh
```

**What this does:**
- `sbatch` — the SLURM command for submitting a batch job. SLURM is the job scheduling system. It queues your request and runs it when a GPU node is available.
- `run_gan.sh` — the batch script that describes what compute resources to request and what Python command to run. It asks for 1 GPU, 1 node, for up to 4 hours, under the `deepsrch` account.

**What you get back immediately:**

```
Submitted batch job 12345678
```

That number is your **job ID**. Write it down — you will need it to check status and find your log file.

**To check if your job is running:**

```bash
squeue -u forrestc
```

This shows all your currently queued or running jobs. The `ST` (state) column will say `PD` (pending, waiting in queue) or `R` (running).

**To check the estimated start time (when it is still pending):**

```bash
squeue -u forrestc --start
```

This adds an `START_TIME` column showing when SLURM estimates your job will begin. The estimate can be inaccurate — it shifts as other jobs finish or are cancelled — but it gives you a rough sense of how long to wait.

**To check the live output as the job runs:**

```bash
tail -f logs/simgan_12345678.out
```

Replace `12345678` with your actual job ID. `tail -f` prints the end of the file and keeps updating as new lines are added — like watching a live log. Press `Ctrl+C` to stop watching (this does not cancel the job, just stops the display).

---

## Optional: Changing training settings

You can override the default settings when you submit. Add options after `run_gan.sh`:

```bash
sbatch run_gan.sh --epochs 30 --lam 20
```

**What the options mean:**
- `--epochs 30` — train for 30 full passes through the dataset instead of the default 50. One epoch = the network sees every training image once.
- `--lam 20` — increase the preservation weight λ to 20. This makes the Refiner more conservative — it will change the simulated images less. Use this if your results are losing the lens arc.

Other options you can set:
- `--batch-size 16` — how many images to process at once per training step. Smaller = slower but uses less GPU memory.
- `--lr 5e-5` — the Refiner learning rate. How big a step the Refiner network takes when it updates its weights.
- `--lr-d 1e-5` — the Discriminator learning rate specifically. Defaults to `lr / 4` (one quarter of the Refiner rate). The Discriminator is deliberately trained slower than the Refiner so it does not "win" the game too quickly and stop giving the Refiner useful feedback.

---

## Step 5: Check the results after training

When the job finishes, SLURM writes "Training complete." to the log. Then look at:

### Loss plots

```
plots/losses_epoch50.png
```

This image has three sub-plots. Here is what each number means in plain physics terms:

| Plot | What it measures | Healthy range | Problem |
|------|-----------------|---------------|---------|
| **D_loss** | How well the Discriminator tells real JWST images from refined ones. 0 = perfect, 1 = completely fooled. | ~0.5 | If it goes to 0, D has "won" and the Refiner is getting no signal to improve. |
| **R_adv** | How well the Refiner fools the Discriminator. Higher = D is more confused by refined images. | ~0.5–0.7 | If it goes to 1, the Refiner is dominating and D has collapsed. |
| **R_pres** | How much the Refiner changed the simulated image (mean squared pixel change). Lower = smaller corrections. | Slowly decreasing | If it is very high, the Refiner is destroying the lens. Increase λ. |

A healthy training run looks like D_loss and R_adv hovering around 0.5–0.7, neither going to 0 nor 1. If they diverge dramatically, see the troubleshooting section at the bottom.

### Sample images

```
plots/samples_epoch50.png
```

This is a grid with three rows:
- **Row 1 (Simulated):** The raw output from the physics simulator. Clean, idealised, may look a bit fake.
- **Row 2 (Refined):** What the Refiner produced. Should look noisier and more realistic, but you should still clearly see the lens arc or Einstein ring.
- **Row 3 (Real):** Actual JWST cutouts from the GOODS-S field. The "ground truth" the Refiner is trying to match.

**What you want to see:** Row 2 looks clearly different from Row 1 (the Refiner is doing something) and plausibly similar to Row 3 (it is making images look more like real JWST data).

**Warning signs:**
- Row 2 looks identical to Row 1 → λ is too high, the Refiner is too constrained.
- Row 2 has lost the lens arc and just looks like noise → λ is too low.
- Row 2 looks like a single repeating pattern → mode collapse (see troubleshooting).

### Saved model weights

```
checkpoints/refiner_epoch50.pt
checkpoints/discriminator_epoch50.pt
```

These files store the trained neural network weights. If you want to apply the trained Refiner to new simulator output later, you load this file. You do not need to do anything with these right now — just know they are there.

---

## Troubleshooting

### "No such file or directory" when loading data

The script expects the JADES catalog and GOODS-S FITS image in `/global/cfs/projectdirs/deepsrch/`. If those paths have changed or the files are missing, check with your supervisor. Do not try to copy or move multi-gigabyte FITS files yourself.

### Training on CPU (very slow)

If the output says `Using device: cpu` during a real (non-smoke) job, the GPU was not allocated correctly. Cancel the job with:

```bash
scancel 12345678
```

and check the SLURM script `run_gan.sh` to make sure `--constraint=gpu` and `--gpus=1` are still present. Then resubmit.

### D_loss goes to 0 quickly (Discriminator winning)

The Discriminator is learning too fast and the Refiner can no longer improve. The Discriminator already has dropout (30% of neurons randomly switched off each step) and runs at 1/4 of the Refiner's learning rate by default. If it is still winning too fast, slow it down further:

```bash
sbatch run_gan.sh --lr-d 1e-5
```

This sets the Discriminator's learning rate to 1e-5 (half of the default 2.5e-5) without touching the Refiner.

### Refined images lose the lens arc

λ is too low. The Refiner is allowed to change the image too freely. Try:

```bash
sbatch run_gan.sh --lam 50
```

### Job was cancelled before finishing

NERSC may cancel jobs that exceed their time limit. The default limit in `run_gan.sh` is 4 hours. If you need more, ask your supervisor — changing the time limit requires an approved allocation.

---

## Quick reference (all the commands in order)

```bash
# 1. Load environment (every new terminal session)
module load conda
conda activate lenssim

# 2. Go to the GAN folder
cd /global/u2/f/forrestc/small-lens-forecast-sims/gan

# 3. Smoke test — always run this first
python train.py --smoke --outdir _smoke_out

# 4. Submit real training job
sbatch run_gan.sh

# 5. Check job status
squeue -u forrestc

# 5b. Check estimated start time (if job is still pending)
squeue -u forrestc --start

# 6. Watch the live log (replace 12345678 with your job ID)
tail -f logs/simgan_12345678.out

# 7. Cancel a job if something is wrong
scancel 12345678
```

---

## Glossary (physics analogues)

| Term | What it means | Physics analogy |
|------|--------------|-----------------|
| **Epoch** | One full pass through all training images | One full scan of a detector array |
| **Batch** | A small subset of images processed together in one step | A single exposure |
| **Loss** | A number measuring how wrong the network currently is | A chi-squared residual |
| **Learning rate** | How big a step the network takes when updating weights | Step size in a gradient descent minimisation |
| **Checkpoint** | Saved copy of the network weights at a given epoch | An intermediate data file saved mid-pipeline |
| **SLURM** | The job scheduler that manages compute resources | The queue for telescope time |
| **GPU** | The specialised hardware that runs matrix operations fast | The spectrograph — designed for one task, very fast at it |
| **`--flag value`** | A command-line option passed to a script | A parameter value in a config file |
