#!/usr/bin/env bash
# Compatibility job name for the classic-novice comparison; checkpoint selection
# is supplied through FULL_SUITE_CHECKPOINTS like every other ScreenSim eval.
#SBATCH --job-name=screensim-novice-full-eval
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=screensim-novice-full-eval-%A_%a.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the Slime checkout}"
exec bash examples/interact/evaluation/screensim.sbatch
