#!/bin/bash
# Usage: ./slurm_jobs/submit_all.sh
OPTIMIZERS=("SGD" "Adam" "AdamW")
LEARNING_RATES=(0.1 0.01 0.001 0.0001)

mkdir -p slurm_logs
job_count=0

echo "Submitting jobs ..."
echo "==============================================================="

for optimizer in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "  Optimizer=$optimizer  lr=$lr"
        sbatch --parsable ./slurm_jobs/run_single_job.sh "$optimizer" "$lr"
        job_count=$((job_count + 1))
    done
done

echo "==============================================================="
echo "Submitted $job_count jobs"
