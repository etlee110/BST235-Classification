# Quality Report: [Branch Name]

**Date:** [YYYY-MM-DD]
**Branch:** [branch-name]
**Reviewer:** Claude

---

## Summary

[1-2 sentence description of what this merge contains]

---

## Quality Score

| File | Score | Blocking Issues |
|------|-------|----------------|
| `src/model.py` | [N]/100 | [none / list] |
| `src/dataset.py` | [N]/100 | [none / list] |
| `src/train.py` | [N]/100 | [none / list] |
| `src/main.py` | [N]/100 | [none / list] |

**Overall:** [N]/100 — [PASS ≥80 / FAIL <80]

---

## Verification Results

| Check | Result | Status |
|-------|--------|--------|
| Syntax check (all src/*.py) | | PASS / FAIL |
| Import check | | PASS / FAIL |
| Forward pass (dummy input) | output shape [4,2] | PASS / FAIL |
| Smoke test (2 epochs) | loss decreasing | PASS / FAIL |

---

## Issues Found

### Critical (must fix before merge)
- [none]

### Major (should fix)
- [none]

### Minor (optional)
- [none]

---

## Decision

[ ] **MERGE** — all critical issues resolved, score ≥ 80
[ ] **HOLD** — blocking issues listed above must be fixed first
