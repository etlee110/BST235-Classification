---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
---

# Python Conventions

## Imports

- No star imports (`from module import *`) — import explicitly
- Standard library first, then third-party (torch, numpy), then local imports
- Local imports use explicit module names: `from model import CNN`, not `from src.model import CNN` (when running from `src/`)

## Code style

- No hardcoded values outside `src/config.py` — all hyperparameters live there
- No absolute paths — use `os.path.join()` with relative roots or `CLAUDE_PROJECT_DIR`
- Functions and classes get one blank line of separation, not docblock paragraphs
- Comments only when the WHY is non-obvious — never describe what the code does

## PyTorch specifics

- Always call `model.train()` at the start of a training epoch
- Always call `model.eval()` before evaluation
- Always use `torch.no_grad()` context manager during evaluation
- `optimizer.zero_grad()` before `loss.backward()` — never after
- Never put `softmax` in the model when using `CrossEntropyLoss`
- `BatchNorm2d` goes BEFORE `ReLU` in conv blocks
- Labels for `CrossEntropyLoss` must be `torch.long` dtype

## Reproducibility

- Set `torch.manual_seed(SEED)`, `np.random.seed(SEED)`, `random.seed(SEED)` at the top of `main()`
- SEED lives in `src/config.py` — never inline it
- DataLoader: always set `num_workers` (0 for local, 4 for cluster)

## File organization

- Model architecture → `src/model.py` only
- Hyperparameters → `src/config.py` only
- Data loading → `src/dataset.py` only
- One-time data conversion scripts → `scripts/convert_*.py`, not inside `src/`
- Do not modify `src/train.py` unless the task explicitly requires it
