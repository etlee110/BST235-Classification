# Project Memory

Corrections and learned facts that persist across sessions.
When a mistake is corrected or a non-obvious thing is learned, append a `[LEARN:category]` entry below.

---

<!-- Append new entries below. Most recent at bottom. -->

## Workflow Patterns

[LEARN:workflow] Requirements specification before planning catches ambiguity early → reduces rework. Use spec-then-plan for complex/ambiguous tasks (>1 hour or >3 files). Template: `templates/requirements-spec.md`.

[LEARN:workflow] Plans and session logs must live on disk to survive context compression. Pre-compact hook saves state; post-compact-restore surfaces it. Always save plan before major work starts.

[LEARN:workflow] Context survival checklist before compression: (1) MEMORY.md updated with [LEARN] entries, (2) session log current (last 10 min), (3) active plan saved to disk with unchecked items visible.

## Agent Consistency

[LEARN:agents] Coder agent must run consistency checks after every edit: model.py changed → verify main.py import and instantiation match; config.py changed → verify hyperparams dict has no dangling references. This class of miss caused a manual fix in session 1 (MLP → CNN rename not propagated to main.py).

## ML Conventions

[LEARN:ml] CrossEntropyLoss includes softmax internally. Never add Softmax/LogSoftmax to the model's final layer — doing so applies softmax twice and produces wrong gradients. Silent bug — no error, just bad results.

[LEARN:ml] For binary classification with CrossEntropyLoss and 2 classes: labels must be torch.long (int64), not torch.float. Shape: [batch_size] not [batch_size, 1]. Type mismatch gives a runtime error that looks confusing.

[LEARN:ml] BatchNorm2d before ReLU is the conventional order in conv blocks. BatchNorm after ReLU is also used but less common. The project uses before-ReLU as the standard.

## Code Quality

[LEARN:code] Star imports (from config import *) work at runtime but Ruff/Pyrefly flag them as undefined names. This is a linter false alarm for this codebase's intentional pattern — not a real error. Do not convert to explicit imports unless the user asks.
