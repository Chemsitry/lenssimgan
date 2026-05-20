#!/bin/bash
#SBATCH --account=deepsrch
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=simgan
#SBATCH --output=logs/simgan_%j.out
#SBATCH --error=logs/simgan_%j.err

cd /global/u2/f/forrestc/small-lens-forecast-sims/gan

# Pass extra arguments through, e.g.  sbatch run_gan.sh --epochs 30 --lam 20
/global/homes/f/forrestc/.conda/envs/lenssim/bin/python -u train.py "$@"
