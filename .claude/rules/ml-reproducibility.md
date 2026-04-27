# ML Reproducibility Protocol

Every experiment must be reproducible from a git commit hash.

## What must be logged per run

The `Logger` in `src/utils.py` automatically captures:
- Git branch and commit hash
- All hyperparameters from the `hyperparams` dict
- Final train and val loss
- Training duration

**Never rely on memory for experiment results.** If it's not in `output/logs/`, it didn't happen.

## Experiment identity

A run is uniquely identified by:
1. Git commit hash (code state)
2. Hyperparameter set (optimizer, lr, batch_size, architecture, seed)
3. Data split (fixed by SEED)

If any of these three change, it is a new experiment.

## Before submitting a training run

- [ ] `SEED` is set in `src/config.py` and passed to `main()`
- [ ] `use_logger=True` in Trainer instantiation
- [ ] Output directories exist (auto-created by config.py on import)
- [ ] Git working tree is clean (or at minimum: note the commit hash)

## Data splits

- Train/val split must be deterministic given `SEED`
- Never shuffle the validation loader
- Never touch validation data during training (no val-guided early stopping that influences architecture)

## Checkpoint naming

Checkpoints include optimizer and lr in the filename:
```
best_model_{OPTIMIZER}_lr{LEARNING_RATE}.pth.tar
```

To reproduce a result: find the log file, read the hyperparams block, check out the git commit hash, run with those params.

## When comparing experiments

Use `src/parse_results.py --log_dir output/logs` to compare all runs.
Use `hparam_search.ipynb` for interactive visualization.
Never compare runs from different git commits unless the architecture is known to be identical.
