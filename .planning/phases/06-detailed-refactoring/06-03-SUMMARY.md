---
phase: 06-detailed-refactoring
plan: 03
subsystem: assert_utilities
tags: [dead-code, documentation, requirements, cleanup]
dependency_graph:
  requires: [06-01-PLAN, 06-02-PLAN]
  provides: [REQ-AUDIT-01-defined, all-dead-code-removed]
  affects: [REQUIREMENTS.md, assert_utilities.py]
tech_stack:
  added: []
  patterns: [bottom-to-top-deletion]
key_files:
  created: []
  modified:
    - .planning/REQUIREMENTS.md
    - src/pytest_generator/assert_utilities.py
decisions:
  - "REQ-AUDIT-01 added as AUDIT section requirement with Phase 6 Complete traceability"
  - "Deleted 10 # msg_list lines and 7 dead-code lines in check_structure_lineplot"
metrics:
  duration: "~5 minutes"
  completed: "2026-03-10"
  tasks_completed: 3
  files_modified: 2
---

# Phase 6 Plan 03: Gap Closure — REQ-AUDIT-01 and Remaining Dead Code Summary

**One-liner:** Added REQ-AUDIT-01 requirement definition to REQUIREMENTS.md and removed 17 remaining dead-code lines from assert_utilities.py identified in Phase 6 verification.

## What Was Built

This plan closed three verification gaps from Phase 6 execution (which scored 7/10 must-haves):

1. **REQ-AUDIT-01 documentation gap:** The requirement was referenced in plans 06-01 and 06-02 but never defined in REQUIREMENTS.md. Added under a new "AUDIT — Code Quality" section with a traceability row marking Phase 6 as Complete.

2. **# msg_list dead-code lines (10 lines):** Four clusters of commented-out `msg_list.append()` and `msg_list +=` lines across the file — deleted in reverse line-number order to avoid shift errors.

3. **check_structure_lineplot dead code (7 lines):** Two commented import lines (already imported at file top) and a bare triple-quoted string block containing commented print statements — both deleted.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add REQ-AUDIT-01 definition and traceability | 7d8bc8e | .planning/REQUIREMENTS.md |
| 2 | Delete # msg_list dead-code lines | 5fa8b9a | src/pytest_generator/assert_utilities.py |
| 3 | Delete dead commented imports and bare triple-quoted block | 8172e57 | src/pytest_generator/assert_utilities.py |

## Verification Results

- `grep -n "REQ-AUDIT-01" REQUIREMENTS.md` — shows definition (line 34) and traceability row (line 65)
- `grep -c "# msg_list" assert_utilities.py` — returns 0
- `uv run pytest tests/test_assert_utilities.py -q` — 183 passed in 1.25s

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- REQUIREMENTS.md contains REQ-AUDIT-01 at lines 34 and 65
- assert_utilities.py has 0 `# msg_list` lines
- 183 regression tests pass
- Commits 7d8bc8e, 5fa8b9a, 8172e57 exist in git log
