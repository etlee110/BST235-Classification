# Orchestrator Protocol

**This rule describes the loop that skills implement.** The pattern is not automatic — specific skills run it internally when invoked.

## The Loop (pattern)

```
Task starts (plan approved or skill invoked)
  │
  Step 1: IMPLEMENT — Execute plan steps
  │
  Step 2: VERIFY — Run verification protocol (forward pass, smoke test, etc.)
  │         If verification fails → fix → re-verify (max 2 retries)
  │
  Step 3: REVIEW — Spawn reviewer agent for ML bug check
  │
  Step 4: FIX — Apply fixes (critical → major → minor)
  │
  Step 5: RE-VERIFY — Confirm fixes are clean
  │
  Step 6: SCORE — Apply quality-gates.md rubric
  │
  └── Score >= threshold?
        YES → Present summary to user
        NO  → Loop back to Step 3 (max 3 rounds)
              After max rounds → present with remaining issues
```

## Where the pattern is implemented

| Skill | Steps covered | Notes |
|---|---|---|
| `/commit` | 2 (quality score), 6 (gate) | Halts on <80; user can override |
| `/train` | 1, 2 | Runs training and verifies logs exist |
| `/evaluate` | 3–5 (evaluator agent) | Reads logs, diagnoses, recommends |
| `/experiment` | 1, 2 (grid search) | Submits jobs, tracks results |
| `/debug-training` | 3–5 (coder + reviewer loop) | Diagnoses and fixes training failures |

## What is NOT automatic

- **No post-plan-approval trigger.** Exiting plan mode does not launch this loop. The user (or a skill invocation) starts it.
- **No git-hook enforcement of Step 6.** Quality gate runs inside `/commit`; a direct `git commit` bypasses it.

## Limits

- **Main loop:** max 3 review-fix rounds
- **Verification retries:** max 2 attempts
- Never loop indefinitely

## Agent roles in the loop

- **coder** → Step 1 (implement), Step 4 (fix)
- **reviewer** → Step 3 (ML bug check)
- **evaluator** → Step 3 (results analysis, post-training)
- **data-explorer** → Step 1 pre-check (confirm data is sane before training)

## Cross-references

- `.claude/rules/plan-first-workflow.md` — when to enter plan mode
- `.claude/rules/quality-gates.md` — threshold definitions
- `.claude/rules/verification-protocol.md` — what to verify and how
