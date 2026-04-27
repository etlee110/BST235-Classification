# Task Completion Verification Protocol

**At the end of EVERY code task, verify the output works correctly before reporting done.**

## For Model / Architecture Changes (`src/model.py`)

1. Run a dummy forward pass to confirm shape is correct:
   ```bash
   cd /Users/ethanlee/Desktop/bst235-classification && \
   source .venv/bin/activate && \
   python3 -c "
   import sys; sys.path.insert(0, 'src')
   from model import CNN
   import torch
   x = torch.randn(4, 3, 32, 32)
   m = CNN(num_classes=2)
   out = m(x)
   print('Output shape:', out.shape)
   assert out.shape == (4, 2), f'Expected (4,2) got {out.shape}'
   print('PASS')
   "
   ```
2. Verify no `softmax` in the model (conflicts with `CrossEntropyLoss`).
3. Confirm `BatchNorm` is placed before `ReLU` in each block.

## For Dataset Changes (`src/dataset.py`)

1. Instantiate the dataset and check `len()` and `__getitem__` output:
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, 'src')
   from dataset import get_loaders
   train, val = get_loaders(batch_size=8)
   x, y = next(iter(train))
   print('Input shape:', x.shape)
   print('Label shape:', y.shape, 'dtype:', y.dtype)
   print('Label range:', y.min().item(), '-', y.max().item())
   "
   ```
2. Confirm label dtype is `torch.long` (required by `CrossEntropyLoss`).
3. Confirm label values are in `[0, NUM_CLASSES)`.
4. Confirm `shuffle=True` on train loader, `shuffle=False` on val loader.

## For Training Loop Changes (`src/train.py`)

1. Run a 2-epoch smoke test:
   ```bash
   python3 src/main.py --optimizer Adam --learning_rate 0.001
   ```
   (Stop after 2 epochs if needed — just confirm it runs without error.)
2. Verify loss is decreasing (not NaN, not constant).
3. Verify checkpoint is saved to `output/checkpoints/`.
4. Verify log file is created in `output/logs/`.

## For Config Changes (`src/config.py`)

1. Verify `main.py` still references only variables that exist in config.
2. Run `python3 -c "from config import *"` from `src/` to catch import errors.

## Common ML Pitfalls

- **Loss is NaN from epoch 1**: LR too high, or data not normalized
- **Loss not decreasing at all**: LR too low, or wrong loss function
- **Val loss much higher than train from epoch 1**: Data leakage or train/val preprocessing mismatch
- **Loss identical every epoch**: `optimizer.zero_grad()` missing, or model output is constant (bad init)
- **Shape mismatch error**: Batch dimension wrong, or wrong `squeeze()`/`unsqueeze()`

## Verification Checklist

```
[ ] Forward pass runs without error
[ ] Output shape is correct (batch_size, num_classes)
[ ] Label dtype is torch.long
[ ] Loss decreases over first few epochs
[ ] No NaN in loss
[ ] Checkpoint saved
[ ] Log file created
[ ] Reported results to user
```
