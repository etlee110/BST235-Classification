# Quality Gates

> **Framing:** Thresholds are advisory — the `/commit` skill enforces them and halts on failure. A direct `git commit` bypasses review. "Gate" means "checkpoint enforced by a specific skill," not a repo-wide block.

## Thresholds

- **80/100 = Commit** — good enough to save
- **90/100 = Merge** — ready for the main branch
- **95/100 = Excellence** — aspirational

## Python / ML Code Rubric

| Severity | Issue | Deduction |
|---|---|---|
| Critical | Syntax error / import error | -100 |
| Critical | Data leakage (val data used in training) | -40 |
| Critical | Wrong loss dtype (float vs long targets) | -30 |
| Critical | `model.eval()` not called during evaluation | -25 |
| Critical | `torch.no_grad()` missing during evaluation | -20 |
| Critical | Hardcoded absolute paths | -20 |
| Major | `optimizer.zero_grad()` missing or in wrong place | -15 |
| Major | `shuffle=True` on validation loader | -10 |
| Major | `model.train()` missing at top of training loop | -10 |
| Major | Softmax in model + CrossEntropyLoss (double-counts) | -10 |
| Major | `SEED` not set before data split | -10 |
| Major | Best model saved on train loss instead of val loss | -8 |
| Minor | Hyperparameter hardcoded outside `src/config.py` | -5 |
| Minor | No `num_workers` set in DataLoader | -2 |
| Minor | Uncommented `#TODO` left in production code | -3 |

## Enforcement

- **Score < 80 in `/commit`:** Halt. List blocking issues. User may override with an explicit reason — logged in the commit body.
- **Score < 90 in `/commit`:** Allow, warn. List recommendations.
- **Direct `git commit`:** No enforcement. To add hard enforcement, install a git pre-commit hook.

## Tolerance Thresholds (ML Experiments)

| Metric | Tolerance | Notes |
|---|---|---|
| Final val loss (same config, re-run) | ±1% | Due to random seed variation |
| Best val loss reproducibility | ±2% | Stochastic training |
| Accuracy (classification) | ±0.5% | Acceptable run-to-run variance |
