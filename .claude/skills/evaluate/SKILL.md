---
name: evaluate
description: Analyze training results from logs and recommend the next experiment. Use when user says "what do the results say", "analyze the logs", "training finished", "what should I try next", "evaluate the run", or "what happened".
argument-hint: "[optional: path to specific log file]"
allowed-tools: ["Read", "Bash"]
---

# /evaluate — Analyze Training Results

Read the most recent training log, diagnose what happened, and recommend the single most impactful next experiment.

## Step 1: Find the log

```bash
# If $ARGUMENTS specifies a path, use that. Otherwise use the most recent.
if [ -n "$ARGUMENTS" ]; then
    LOGFILE="$ARGUMENTS"
else
    LOGFILE=$(ls -lt output/logs/*.txt 2>/dev/null | head -1 | awk '{print $NF}')
fi
echo "Analyzing: $LOGFILE"
```

## Step 2: Extract metrics

```bash
echo "--- Hyperparameters ---"
grep -A 20 "HYPERPARAMETERS" "$LOGFILE" | grep -E "^(NUM_EPOCHS|BATCH_SIZE|LEARNING_RATE|OPTIMIZER|NUM_CLASSES)"

echo "--- Final Metrics ---"
grep -A 5 "FINAL METRICS" "$LOGFILE"

echo "--- Loss Trajectory (all best-val events) ---"
grep "reach best val loss" "$LOGFILE"

echo "--- Epoch Progression (sample every 10) ---"
grep "Epoch " "$LOGFILE" | awk 'NR % 10 == 0'

echo "--- Last 5 epochs ---"
grep "Epoch " "$LOGFILE" | tail -5
```

## Step 3: Diagnose

Use this table to identify the pattern:

| Pattern | Diagnosis |
|---|---|
| train ↓, val ↓ (both still falling at end) | Undertrained — train longer or raise LR |
| train ↓↓, val → (plateau or rising) | Overfitting — add dropout/augmentation, reduce model size, add weight decay |
| both flat from epoch 1 | LR too low OR bad data pipeline — raise LR 10x, check dataset |
| loss oscillates wildly | LR too high — lower 10x |
| train ↓, val >> train from start | Data leakage or preprocessing mismatch |
| loss = NaN | LR way too high, or division by zero in data |
| val loss decreasing then spike | Scheduler too aggressive |

## Step 4: Report

Format:

```
📊 Run Summary
─────────────────────────────────
Optimizer: [value]  LR: [value]  Epochs: [value]
Final Train: [value]  Final Val: [value]  Best Val: [value]

🔍 Diagnosis
[1-2 sentences describing what happened during training]

🎯 Recommended Next Experiment
[Specific: name the exact parameter change and why]
e.g., "Reduce LR from 0.01 to 0.001 — loss oscillation in early epochs
suggests the current rate is overshooting. Keep all other settings fixed."

📋 Experiments to try after that (ranked by expected impact)
1. [highest impact]
2. [medium impact]
3. [lower impact]

→ Say "/train --optimizer [X] --learning_rate [Y]" to run the next experiment.
```

## Step 5: Compare against other runs (if multiple logs exist)

```bash
python3 src/parse_results.py --log_dir output/logs 2>/dev/null | head -30
```

If the parse_results table shows multiple runs, highlight how this run compares.
