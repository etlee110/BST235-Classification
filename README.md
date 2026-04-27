# Deep Learning Project Template

A reusable PyTorch project skeleton with training infrastructure, logging, checkpointing, and SLURM-based hyperparameter search.

## Structure

```
src/
├── config.py          # All hyperparameters — edit this first
├── dataset.py         # CustomDataset placeholder + get_loaders()
├── model.py           # MLP placeholder — swap in your architecture
├── train.py           # Trainer class (train loop, checkpoints, logging)
├── utils.py           # save/load checkpoint, loss curve plots, Logger
├── main.py            # Entry point with argparse (--optimizer, --learning_rate)
└── parse_results.py   # Parse training logs into a comparison table
slurm_jobs/
├── run_single_job.sh  # Single SLURM job (optimizer + lr as args)
└── submit_all.sh      # Grid search: submits all optimizer × lr combos
hparam_search.ipynb    # Interactive notebook to compare training runs
output/
├── checkpoints/
├── results/
└── logs/
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Edit src/config.py, src/dataset.py, src/model.py for your project
python3 src/main.py
python3 src/main.py --optimizer Adam --learning_rate 0.001
```

## Hyperparameter Search

**Locally:**
```bash
for opt in SGD Adam AdamW; do
  for lr in 0.1 0.01 0.001 0.0001; do
    python3 src/main.py --optimizer $opt --learning_rate $lr
  done
done
```

**SLURM cluster:**
```bash
# Update --chdir in slurm_jobs/run_single_job.sh to your cluster path
./slurm_jobs/submit_all.sh
```

**Analyze results** in `hparam_search.ipynb` or:
```bash
python3 src/parse_results.py --log_dir output/logs
```

## Adapting to a New Project

1. **`src/config.py`** — set your hyperparameters and model dims
2. **`src/dataset.py`** — implement `CustomDataset.__init__`, `__len__`, `__getitem__`
3. **`src/model.py`** — replace `MLP` with your architecture
4. **`src/main.py`** — update the optimizer choices and loss function if needed


## Dataset specifics

You're classifying truck(1) or non-truck(0).

```
Dimensions
  x_train : (7500, 3, 32, 32)  dtype=float32  range=[0.000, 1.000]
  x_test  : (2500, 3, 32, 32)  dtype=float32
  y_train : (7500,)  dtype=int64
  labels  : {np.int64(0): np.int64(3714), np.int64(1): np.int64(3786)}
```