---
name: debug-training
description: Diagnose and fix a training failure or anomaly. Use when user says "training crashed", "loss is NaN", "loss isn't decreasing", "something is wrong with training", "debug this", "loss exploded", or "model isn't learning".
argument-hint: "[optional: description of the symptom]"
allowed-tools: ["Read", "Bash", "Edit"]
---

# /debug-training — Diagnose and Fix Training Failures

Systematically diagnose the failure, identify root cause, fix it, and verify the fix.

## Step 1: Gather evidence

```bash
# Most recent log
LOGFILE=$(ls -lt output/logs/*.txt 2>/dev/null | head -1 | awk '{print $NF}')
echo "Log: $LOGFILE"
[ -n "$LOGFILE" ] && tail -50 "$LOGFILE"

# Any Python error in the last run
[ -n "$LOGFILE" ] && grep -i "error\|traceback\|exception\|nan\|inf" "$LOGFILE" | head -20
```

Also read `src/model.py`, `src/dataset.py`, and `src/train.py` to understand current state.

## Step 2: Identify the symptom class

| Symptom | Check first |
|---|---|
| Traceback / ImportError | Syntax or missing dependency |
| `loss = nan` from epoch 1 | LR too high, or division by zero in data |
| `loss = nan` after several epochs | Gradient explosion — add gradient clipping or lower LR |
| Loss constant every epoch | `zero_grad()` missing, or model output is constant |
| Loss decreasing, then suddenly NaN | LR schedule too aggressive |
| Accuracy stuck at 50% (binary) | Model always predicting same class — check class balance |
| Val loss >> Train loss immediately | Data leakage or preprocessing mismatch between train/val |
| OOM (out of memory) | Reduce `BATCH_SIZE` in config.py |
| Shape mismatch error | Trace tensor shapes through forward pass |

## Step 3: Run targeted diagnostics

For NaN/constant loss:
```bash
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'src')
from dataset import get_loaders
from config import BATCH_SIZE
train, val = get_loaders(batch_size=BATCH_SIZE)
x, y = next(iter(train))
print('x stats: min', x.min().item(), 'max', x.max().item(), 'mean', x.mean().item())
print('y values:', y[:10], 'dtype:', y.dtype)
print('any NaN in x:', x.isnan().any().item())
"
```

For shape mismatch:
```bash
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'src')
from model import CNN
from config import NUM_CLASSES
import torch
m = CNN(NUM_CLASSES)
x = torch.randn(4, 3, 32, 32)
print('Input:', x.shape)
out = m(x)
print('Output:', out.shape)
"
```

## Step 4: Apply fix

Based on diagnosis, apply the minimal fix:

- **LR too high** → update `LEARNING_RATE` in `src/config.py`
- **Missing `zero_grad()`** → fix in `src/train.py` `_train_epoch`
- **Wrong label dtype** → fix `__getitem__` in `src/dataset.py` to return `y.long()`
- **Data NaN** → add normalization in `src/dataset.py`
- **Gradient explosion** → add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` in `_train_epoch` before `optimizer.step()`
- **Softmax + CrossEntropy** → remove `Softmax` from model's final layer

## Step 5: Verify fix

Run the verification protocol:
```bash
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'src')
from model import CNN; from config import NUM_CLASSES
import torch
m = CNN(NUM_CLASSES)
x = torch.randn(4, 3, 32, 32)
out = m(x)
assert not out.isnan().any(), 'NaN in output!'
print('Forward pass clean. Output shape:', out.shape)
"
```

Then run a 3-epoch smoke test:
```bash
python3 src/main.py 2>&1 | head -40
```

## Step 6: Report

```
🔧 Debug Report
─────────────────────────────────
Symptom: [what was observed]
Root cause: [what caused it]
Fix applied: [file:line — what was changed]
Verification: [PASS/FAIL + evidence]

→ Say "/train" to run a full experiment with the fix applied.
```
