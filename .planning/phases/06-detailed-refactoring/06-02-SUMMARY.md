---
phase: 06-detailed-refactoring
plan: 02
subsystem: assert_utilities
tags: [cleanup, debug-removal, dead-code, refactoring]
dependency_graph:
  requires: [06-01]
  provides: [clean-assert-utilities-no-debug-output]
  affects: [src/pytest_generator/assert_utilities.py]
tech_stack:
  added: []
  patterns: [grep-verify-before-remove, line-number-targeted-deletion]
key_files:
  created: []
  modified:
    - src/pytest_generator/assert_utilities.py
decisions:
  - "Converted 3 TODO markers to plain explanatory comments rather than deleting (preserves known-gap documentation)"
  - "Used line-number targeted Python scripts for atomic batch deletion rather than sed/awk to avoid line-shift errors"
metrics:
  duration: "8 minutes"
  completed_date: "2026-03-10"
  tasks_completed: 2
  files_modified: 1
---

# Phase 06 Plan 02: Debug Print and Dead Code Removal Summary

**One-liner:** Removed all 67 print() calls, 2 pprint() calls, pprint import, 14 # ! dead-code comment lines, and resolved 3 TODO markers in assert_utilities.py — zero debug output remains.

## What Was Built

Cleaned assert_utilities.py of all debug output and dead commented-out code. The file went from 5071 lines to 4986 lines (85 lines removed). All 183 regression tests continue to pass.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Remove all print() and pprint() calls, remove pprint import | 394972e | src/pytest_generator/assert_utilities.py |
| 2 | Remove commented-out dead code and resolve TODO markers | 9c694b0 | src/pytest_generator/assert_utilities.py |

## Verification Results

```
grep -c "^\s*print(" src/pytest_generator/assert_utilities.py  → 0
grep -c "pprint" src/pytest_generator/assert_utilities.py      → 0
grep -c "# !" src/pytest_generator/assert_utilities.py         → 0
grep -ic "TODO" src/pytest_generator/assert_utilities.py       → 0
uv run pytest tests/test_assert_utilities.py -q                → 183 passed
```

## Deviations from Plan

None - plan executed exactly as written.

The 3 TODO markers were handled as follows:
1. `## ARE WE CHECKING THAT THE KEYS ARE int ! TODO` (line 810) — converted to `# Known gap: top-level key type (int) is not explicitly validated.` because the function does not validate int key types.
2. `# ! TODO: check that keys are in the instructor answer` (line 3617) — converted to `# Known gap: student keys are not verified against instructor_answer keys.` because the function iterates instructor keys without checking student key presence first.
3. `# ! TODO: Explicitly state the indices considered for grading.` (line 4368) — converted to `# Note: the indices considered for grading are not explicitly stated in the message.`

## Self-Check: PASSED

- src/pytest_generator/assert_utilities.py — FOUND
- commit 394972e (Task 1) — FOUND
- commit 9c694b0 (Task 2) — FOUND
