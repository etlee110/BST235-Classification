---
name: experiment
description: Design and run a hyperparameter grid search, either locally (sequential) or via SLURM. Use when user says "run a grid search", "sweep hyperparameters", "try all learning rates", "submit SLURM jobs", "run experiments", or "compare optimizers".
argument-hint: "[local | slurm] [optional: --optimizers SGD,Adam --lrs 0.1,0.01,0.001]"
allowed-tools: ["Bash", "Read", "Edit"]
---

# /experiment — Hyperparameter Grid Search

Design and run a systematic sweep over optimizers and learning rates.

## Step 1: Parse arguments

Default grid if `$ARGUMENTS` doesn't specify:
- Optimizers: `SGD Adam AdamW`
- Learning rates: `0.1 0.01 0.001 0.0001`
- Mode: `local` (sequential) or `slurm` (cluster)

## Step 2: Pre-flight check

```bash
source .venv/bin/activate
cd src && python3 -c "import config, dataset, model, train, utils; print('Imports OK')" && cd ..
ls data/ 2>/dev/null || echo "WARNING: data/ is empty"
```

Stop if imports fail.

## Step 3A: Run locally (sequential)

```bash
source .venv/bin/activate

OPTIMIZERS=("SGD" "Adam" "AdamW")
LEARNING_RATES=(0.1 0.01 0.001 0.0001)

for opt in "${OPTIMIZERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo ">>> Running: optimizer=$opt lr=$lr"
        python3 src/main.py --optimizer "$opt" --learning_rate "$lr"
        echo ">>> Done: $opt lr=$lr"
    done
done
```

Report progress after each run completes.

## Step 3B: Run on SLURM cluster

```bash
# Verify the chdir path in slurm_jobs/run_single_job.sh is correct
grep "chdir" slurm_jobs/run_single_job.sh

# Submit all jobs
./slurm_jobs/submit_all.sh
```

After submission, show the job count and explain how to monitor:
```bash
squeue -u $USER | head -20
```

## Step 4: Wait and collect results (local only)

After all runs complete:
```bash
python3 src/parse_results.py --log_dir output/logs
```

## Step 5: Report

```
🔬 Experiment Grid
─────────────────────────────────
Optimizers: [list]
Learning rates: [list]
Total runs: [N]
Mode: [local | slurm]

[If local and complete:]
Best configuration:
  Optimizer: [value]  LR: [value]  Best Val: [value]

Full results table:
[paste parse_results.py output]

→ Say "/evaluate" to get a detailed diagnosis of the best run.
→ Open hparam_search.ipynb for interactive visualization.
```

## Notes

- Local sequential can take a long time for large grids — warn user if >12 runs
- For SLURM: user must update `--chdir` in `slurm_jobs/run_single_job.sh` with their cluster path
- Each run's log is timestamped and saved to `output/logs/` — nothing is overwritten
