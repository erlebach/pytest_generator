---
phase: 04-signature-standardization
plan: "01"
subsystem: assert_utilities
tags: [signature, annotations, CheckResult, REQ-CONS-01, REQ-CONS-02]
dependency_graph:
  requires: []
  provides: [scripts/audit_signatures.py, annotated-assert-utilities]
  affects: [src/pytest_generator/assert_utilities.py]
tech_stack:
  added: []
  patterns: [AST-based audit, targeted line-range rename]
key_files:
  created:
    - scripts/audit_signatures.py
  modified:
    - src/pytest_generator/assert_utilities.py
decisions:
  - "Fixed check_structure_dict_any param naming (student_dict->student_answer) as auto-fix: required for audit to exit 0"
  - "Used AST node line ranges for scoped rename to avoid cross-function contamination"
  - "Handled both single-line and multi-line def signatures for annotation replacement"
metrics:
  duration_minutes: 10
  completed_date: "2026-03-11"
  tasks_completed: 3
  files_modified: 2
---

# Phase 4 Plan 1: Signature Standardization — Annotation and Naming Fixes Summary

**One-liner:** AST audit script added; all 91 checker functions now use CheckResult return type alias and standard parameter names.

## What Was Done

### Task 1: Write scripts/audit_signatures.py

Created an AST-based audit tool that:
- Walks only `tree.body` (module-level FunctionDef nodes), avoiding nested functions like `check_grid_status`
- Classifies functions by `check_structure_*` / `check_answer_*` prefix
- Checks return annotation equals `"CheckResult"` (not bare `tuple[bool, str]`, not missing)
- Checks `param[0] == "student_answer"` and (for `check_answer_*`) `param[1] == "instructor_answer"`
- Exits 1 with violation list; exits 0 with `"OK: N functions audited, 0 violations"`

Before fixes: reported 95 violations. After all fixes: exits 0 with 91 functions audited.

### Task 2: Fix annotations (REQ-CONS-02)

Applied two-phase fix to `assert_utilities.py`:

**Phase A** — Added `-> CheckResult` to 4 scatterplot functions that had no return annotation:
- `check_structure_scatterplot2d`, `check_structure_scatterplot3d`
- `check_answer_scatterplot2d`, `check_answer_scatterplot3d`

**Phase B** — Replaced `-> tuple[bool, str]:` with `-> CheckResult:` on all remaining 87 `check_structure_*` / `check_answer_*` functions. Required handling both single-line and multi-line function signatures (used AST node line ranges to locate the annotation line within each function header).

Result: 91 insertions, 91 deletions in assert_utilities.py.

### Task 3: Fix parameter naming (REQ-CONS-01)

Renamed parameters within scoped line ranges using AST `node.lineno`/`node.end_lineno`:

- `check_structure_dendrogram` (L713–753): `student_dendro` -> `student_answer` (4 occurrences: def, type annotation, docstring, body)
- `check_answer_dendrogram` (L2792–2837): `student_dendro` -> `student_answer` and `instructor_dendro` -> `instructor_answer` (8 occurrences total)
- `check_structure_dict_any` (L756–796): `student_dict` -> `student_answer`, `instructor_dict` -> `instructor_answer` (auto-fix, see Deviations)

## Verification Results

```
$ uv run pytest tests/ -q
183 passed in 0.90s

$ python3 scripts/audit_signatures.py
OK: 91 check_structure_* / check_answer_* functions audited, 0 violations.
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Fix] Renamed check_structure_dict_any parameters**
- **Found during:** Task 3 verification (audit still showed 1 violation after dendrogram fixes)
- **Issue:** `check_structure_dict_any` used `student_dict` / `instructor_dict` instead of standard `student_answer` / `instructor_answer`. The plan's RESEARCH.md did not enumerate this violation, but the plan's success criterion requires "audit script exits 0 after fixes" — which required fixing this.
- **Fix:** Renamed `student_dict` -> `student_answer` and `instructor_dict` -> `instructor_answer` within the function's line range (L756–796)
- **Files modified:** `src/pytest_generator/assert_utilities.py`
- **Commit:** 41d379f

## Self-Check: PASSED

- scripts/audit_signatures.py: FOUND
- src/pytest_generator/assert_utilities.py: FOUND
- Commit d71f714 (Task 1): FOUND
- Commit c8468e4 (Task 2): FOUND
- Commit 41d379f (Task 3): FOUND
