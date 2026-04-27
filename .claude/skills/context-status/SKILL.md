---
name: context-status
description: Show current context status and session health. Use to check how much context has been used, whether auto-compact is approaching, and what state will be preserved.
allowed-tools: ["Read", "Bash"]
---

# /context-status — Check Session Health

Show the current session status: context usage estimate, active plan, session log, preservation state.

## Step 1: Check Context Monitor Cache

```bash
cat ~/.claude/sessions/*/context-monitor-cache.json 2>/dev/null | head -20
```

## Step 2: Find Active Plan

```bash
ls -lt quality_reports/plans/*.md 2>/dev/null | head -3
```

## Step 3: Find Session Log

```bash
ls -lt quality_reports/session_logs/*.md 2>/dev/null | head -1
```

## Step 4: Check Git State

```bash
git status --short
git log --oneline -3
```

## Step 5: Report Status

Format the output:

```
📊 Session Status
─────────────────────────────────
Context Usage:  ~XX% (estimated)
Auto-compact:   [approaching | not imminent]

📋 Active Plan
File:   quality_reports/plans/YYYY-MM-DD_description.md
Status: [draft | approved | in_progress | completed]
Task:   [current unchecked task or "none"]

📝 Session Log
File:   quality_reports/session_logs/YYYY-MM-DD_description.md

🔬 Experiment State
Last run:    [optimizer, lr, epochs — from most recent log file]
Best val:    [from log]

✓ Preservation Check
  • Pre-compact hook:    [configured | missing]
  • Post-compact restore: [configured | missing]
  • Git state:           [clean | N files modified]
```

## Notes

- Context % is an estimate based on tool call count (heuristic, not exact)
- All important state is saved to disk (plans, logs, MEMORY.md)
- If context is approaching 80%, consider running `/commit` to save work
