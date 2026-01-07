#!/bin/bash
#SBATCH --job-name=hpo_worker
#SBATCH --output=outputs/hpo/slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG=${1:-configs/hpo_inference.yaml}
STUDY=${2:-sc_inference}
STORAGE=${3:-sqlite:///outputs/hpo/optuna.db}
TRIALS=${4:-50}
SEED=${5:-42}

python scripts/hpo_inference.py --config "$CONFIG" --study_name "$STUDY" --storage "$STORAGE" --n_trials "$TRIALS" --seed "$SEED"
