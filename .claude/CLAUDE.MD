# bst235-classification

**Project:** Binary image classification — car vs. truck (CIFAR-10 subset)
**Stack:** PyTorch · Python 3.10 · SLURM (cluster)
**Branch:** main

---

## Core Principles

- **Plan first** — enter plan mode before non-trivial tasks; save plans to `quality_reports/plans/`
- **Verify after** — run a forward pass or smoke-test after every code change
- **Single source of truth** — all hyperparameters live in `src/config.py`; never hardcode elsewhere
- **Quality gates** — nothing commits below 80/100 (via `/commit` skill)
- **[LEARN] tags** — when corrected, save `[LEARN:category] wrong → right` to `MEMORY.md`

Cross-session context lives in `MEMORY.md`; past plans, specs, and session logs are in `quality_reports/`.

---

## Folder Structure

```
bst235-classification/
├── .claude/                    # Rules, skills, agents, hooks, settings
│   ├── agents/                 # coder, reviewer, evaluator, data-explorer
│   ├── hooks/                  # pre-compact, post-compact-restore, context-monitor,
│   │                           #   verify-reminder, log-reminder, notify
│   ├── rules/                  # plan-first, session-logging, quality-gates,
│   │                           #   verification-protocol, orchestrator, python-conventions,
│   │                           #   ml-reproducibility
│   ├── scripts/statusline.sh
│   ├── skills/                 # commit, context-status, train, evaluate,
│   │                           #   experiment, debug-training
│   └── settings.json
├── src/
│   ├── config.py               ← hyperparameters (edit this first)
│   ├── dataset.py              ← CustomDataset + get_loaders()
│   ├── model.py                ← CNN architecture
│   ├── train.py                ← Trainer class
│   ├── utils.py                ← checkpointing, logging, loss curve plots
│   ├── main.py                 ← entry point
│   └── parse_results.py        ← parse logs into comparison table
├── scripts/                    ← one-time utilities (data conversion, etc.)
├── slurm_jobs/                 ← SLURM grid search scripts
├── data/                       ← raw data (gitignored)
├── output/                     ← checkpoints, logs, results (gitignored)
├── quality_reports/            ← plans, session logs, merge reports, specs
├── templates/                  ← session-log.md, requirements-spec.md, quality-report.md
├── hparam_search.ipynb         ← interactive notebook for comparing runs
└── MEMORY.md                   ← cross-session learnings ([LEARN] entries)
```

---

## Commands

```bash
# Setup
source .venv/bin/activate
pip install -r requirements.txt

# Training
python3 src/main.py                                        # config.py defaults
python3 src/main.py --optimizer Adam --learning_rate 0.001

# Analysis
python3 src/parse_results.py --log_dir output/logs

# SLURM grid search
./slurm_jobs/submit_all.sh
```

---

## Key Conventions

- `src/config.py`: `IMAGE_SIZE=32`, `NUM_CHANNELS=3`, `NUM_CLASSES=2` (car vs truck)
- Loss: `CrossEntropyLoss` — do NOT add softmax to the model
- Labels must be `torch.long`
- Seeds: `SEED=42` set via `torch.manual_seed`, `np.random.seed`, `random.seed` in `main()`
- Output dirs are gitignored and auto-created by `config.py` on import

---

## Skills Quick Reference

| Command | What It Does |
|---|---|
| `/train [--optimizer X --learning_rate Y]` | Run a training experiment |
| `/evaluate [log_file]` | Analyze results, recommend next experiment |
| `/experiment [local\|slurm]` | Run a full hyperparameter grid search |
| `/debug-training` | Diagnose and fix training failures |
| `/commit [msg]` | Quality-gate, stage, commit, PR, merge |
| `/context-status` | Show session health + context usage |

---

## Agent Workflow

```
Describe change → coder implements → reviewer checks → /train → evaluator reads logs → recommends → repeat
```

Invoke naturally:
- "implement X" → coder
- "review the changes" → reviewer
- "explore the data" → data-explorer
- "training finished / what do results say" → evaluator

---

## Current Experiment State

| Config | Best Val Loss | Notes |
|---|---|---|
| *(add rows as experiments complete)* | | |
