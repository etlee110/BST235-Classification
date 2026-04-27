# Plan-First Workflow

**For any non-trivial task, enter plan mode before writing code.**

## The Protocol

1. **Enter Plan Mode** — use `EnterPlanMode`
2. **Check MEMORY.md** — read any `[LEARN]` entries relevant to this task
3. **Requirements Specification (for complex/ambiguous tasks)** — see below
4. **Draft the plan** — what changes, which files, in what order
5. **Save to disk** — write to `quality_reports/plans/YYYY-MM-DD_short-description.md`
6. **Present to user** — wait for approval
7. **Exit plan mode** — only after approval
8. **Save initial session log** — capture goal and key context while fresh
9. **Implement** — see `orchestrator-protocol.md`

## Step 3: Requirements Specification (For Complex/Ambiguous Tasks)

**When to use:**
- Task is high-level or vague ("improve the model", "add augmentation")
- Multiple valid interpretations exist
- Significant effort required (>1 hour or >3 files)

**When to skip:**
- Task is clear and specific ("fix batch norm before ReLU")
- Simple single-file edit
- User has already provided detailed requirements

**Protocol:**
1. Use AskUserQuestion to clarify ambiguities (max 3-5 questions)
2. Create `quality_reports/specs/YYYY-MM-DD_description.md` using `templates/requirements-spec.md`
3. Mark each requirement: **MUST** / **SHOULD** / **MAY**
4. Declare clarity status: **CLEAR** / **ASSUMED** / **BLOCKED**
5. Get user approval on spec, then proceed to plan

## Plans on Disk

Plans survive context compression. Save every plan to:
```
quality_reports/plans/YYYY-MM-DD_short-description.md
```
Format: Status (DRAFT/APPROVED/COMPLETED), approach, files to modify, verification steps.

## Context Survival Strategy

Before auto-compression, ensure:
1. MEMORY.md has all `[LEARN]` entries from this session
2. Session log is current (updated within 10 minutes)
3. Active plan is saved to disk with unchecked items visible

After compression, first message:
"Resuming after compression. Last task: [read most recent plan + git log]. Status: [next step]."
