#!/bin/bash
#SBATCH --account=deepsrch
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=simgen
#SBATCH --output=logs/simgen_%j.out
#SBATCH --error=logs/simgen_%j.err

cd /global/u2/f/forrestc/small-lens-forecast-sims

/global/homes/f/forrestc/.conda/envs/lenssim/bin/jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=28800 \
    --output simulations_wide_pos_output.ipynb \
    simulations_wide_pos.ipynb
