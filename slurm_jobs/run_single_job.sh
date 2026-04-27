#!/bin/bash
#SBATCH --job-name=train
#SBATCH -p serial_requeue
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

OPTIMIZER=$1
LEARNING_RATE=$2

mkdir -p slurm_logs
source .venv/bin/activate
python3 src/main.py --optimizer $OPTIMIZER --learning_rate $LEARNING_RATE
