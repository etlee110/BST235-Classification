# Session Logging

**Location:** `quality_reports/session_logs/YYYY-MM-DD_description.md`
**Template:** `templates/session-log.md`

## Three Triggers (all proactive)

### 1. Post-Plan Log
After plan approval, immediately capture: goal, approach, rationale, key context.

### 2. Incremental Logging
Append 1-3 lines whenever: a design decision is made, a problem is solved, the user corrects something, or the approach changes. Do not batch.

### 3. End-of-Session Log
When wrapping up: high-level summary, experiment results, open questions, blockers, next steps.

## What to Capture for ML Sessions

- What experiment was run (optimizer, lr, architecture)
- What the results showed (final train/val loss, any observed patterns)
- What was decided next (based on evaluator recommendation or user direction)
- Any corrections made (e.g., "was using wrong loss dtype — fixed to torch.long")

## Quality Reports

Generated **only at merge time** — not at every commit.
Save to `quality_reports/merges/YYYY-MM-DD_[branch-name].md` using `templates/quality-report.md`.
