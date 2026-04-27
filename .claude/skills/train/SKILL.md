---
name: train
description: Run a training experiment with the current config and verify it completed cleanly. Use when user says "train", "run the model", "start training", "run an experiment", "let's train", or "run with these settings".
argument-hint: "[optional: --optimizer Adam --learning_rate 0.001]"
allowed-tools: ["Bash", "Read"]
---

# /train — Run a Training Experiment

Run training with the current config (or overrides from `$ARGUMENTS`), verify it completes, and report results.

## Step 1: Pre-flight check

```bash
# Verify venv is active and imports work
source .venv/bin/activate
cd src && python3 -c "import config, dataset, model, train, utils; print('Imports OK')" && cd ..

# Show current config
grep -E "^(NUM_EPOCHS|BATCH_SIZE|LEARNING_RATE|OPTIMIZER|NUM_CLASSES)" src/config.py
```

If imports fail, stop and report the error — do not proceed.

## Step 2: Check data is available

```bash
ls data/ 2>/dev/null || echo "WARNING: data/ directory is empty"
```

If `dataset.py` raises `NotImplementedError`, remind user that `CustomDataset` must be implemented first.

## Step 3: Run training

```bash
source .venv/bin/activate
python3 src/main.py $ARGUMENTS 2>&1 | tee /tmp/train_output.txt
```

If `$ARGUMENTS` is empty, run with config defaults.

## Step 4: Verify completion

```bash
# Check that training completed (not crashed)
grep -c "Training completed" /tmp/train_output.txt || echo "WARNING: 'Training completed' not found — training may have crashed"

# Find the log file
ls -lt output/logs/*.txt 2>/dev/null | head -1

# Find the best checkpoint
ls -lt output/checkpoints/best_model_*.pth.tar 2>/dev/null | head -1

# Find the loss curve
ls -lt output/results/*.png 2>/dev/null | head -1
```

## Step 5: Extract key metrics

```bash
LOGFILE=$(ls -lt output/logs/*.txt 2>/dev/null | head -1 | awk '{print $NF}')
if [ -n "$LOGFILE" ]; then
    echo "=== Hyperparameters ==="
    grep -A 20 "HYPERPARAMETERS" "$LOGFILE" | head -20
    echo "=== Final Metrics ==="
    grep -A 5 "FINAL METRICS" "$LOGFILE"
    echo "=== Best Val Loss ==="
    grep "reach best val loss" "$LOGFILE" | tail -1
fi
```

## Step 6: Report

Report in this format:
```
✅ Training completed
─────────────────────────────────
Optimizer:    [value]
LR:           [value]
Epochs:       [value]
Final Train:  [value]
Final Val:    [value]
Best Val:     [value]

Log:       output/logs/[filename]
Checkpoint: output/checkpoints/best_model_[...].pth.tar
Plot:       output/results/[filename]

→ Say "what do the results say" to get diagnosis and next experiment recommendation.
```

## Common failure modes

- **ModuleNotFoundError**: venv not activated or requirements not installed
- **NotImplementedError from dataset**: `CustomDataset` not yet implemented
- **CUDA out of memory**: reduce `BATCH_SIZE` in config.py
- **Loss is NaN**: LR too high — try 10x smaller
- **FileNotFoundError for data**: check `data/` directory
