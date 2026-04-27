---
name: commit
description: Stage, commit, push, open a PR, and merge to main. Use ONLY on explicit commit intent — user says "commit", "ship it", "push this", "open a PR", "merge to main", "let's commit this", or prefixes with `/commit`. Do NOT auto-invoke on vague end-of-task phrases ("we're done", "wrap up") — those require explicit confirmation first. Never force-pushes or skips hooks.
argument-hint: "[optional: commit message]"
allowed-tools: ["Bash", "Read", "Glob"]
---

# Commit, PR, and Merge

Stage changes, verify quality gates, commit with a descriptive message, create a PR, and merge.

## Step 0: Quality Gate (Pre-Commit)

Run the quality rubric mentally against all changed `.py` files using `quality-gates.md`.

Check for the most common critical issues:
```bash
# Check for syntax errors
python3 -m py_compile src/*.py && echo "Syntax OK"

# Check for obvious import errors
cd src && python3 -c "import config, dataset, model, train, utils" && echo "Imports OK" && cd ..
```

- If any critical issue exists (score < 80), halt and report. User may override with "commit anyway" + reason — log the override in the commit body.
- If all checks pass, continue.

## Step 1: Check current state

```bash
git status
git diff --stat
git log --oneline -5
```

## Step 2: Create a branch

```bash
git checkout -b <short-descriptive-branch-name>
```

## Step 3: Stage files

Add specific files (never `git add -A`):
```bash
git add src/<file1> src/<file2> ...
```

Do NOT stage `.claude/settings.local.json`, `output/`, `data/`, or `.venv/`.

## Step 4: Commit with a descriptive message

If `$ARGUMENTS` is provided, use it as the commit message. Otherwise analyze the changes and write a message explaining WHY, not just WHAT.

```bash
git commit -m "$(cat <<'EOF'
<commit message here>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

## Step 5: Push and create PR

```bash
git push -u origin <branch-name>
gh pr create --title "<short title>" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Test plan
- [ ] Forward pass verified
- [ ] Smoke-test training run completed
- [ ] No regressions in other src/ files

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Step 6: Merge and clean up

```bash
gh pr merge <pr-number> --merge --delete-branch
git checkout main
git pull
```

## Step 7: Report

Report the PR URL and what was merged.

## Important

- **Never commit directly to main.** Always create a branch.
- **Never stage `output/`, `data/`, or `.venv/`** — these are gitignored for good reason.
- If the user provides a commit message via `$ARGUMENTS`, use it exactly.
