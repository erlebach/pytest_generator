---
phase: 03-file-cleanup
plan: 01
subsystem: testing
tags: [python, type-aliases, refactoring, assert_utilities]

requires:
  - phase: 02-duplicate-removal
    provides: clean single-definition file with no duplicate functions

provides:
  - SECTION 1 TYPE ALIASES block with CheckResult and PartialScoreDict
  - SECTION 2 CONSTANTS block around FLOAT_TOL
  - SECTION 3 UTILITY PRIMITIVES block consolidating all helpers before checker zone

affects:
  - 03-02: checker reorganization depends on stable SECTION 3 foundation

tech-stack:
  added: []
  patterns:
    - "TypeAlias annotations via `from typing import TypeAlias` for semantic type aliases"
    - "Section delimiters using 70-char `# ===...` banners to separate logical zones"

key-files:
  created: []
  modified:
    - src/pytest_generator/assert_utilities.py

key-decisions:
  - "Do NOT apply type aliases as replacements throughout file — alias definition only (Phase 4 concern)"
  - "Existing utilities block (lines 43-560) falls under SECTION 3 header without needing a second banner"
  - "Three scattered helpers moved verbatim, zero logic changes"

patterns-established:
  - "Section headers use 70-char === banners: SECTION 1, SECTION 2, SECTION 3"
  - "All utility primitives must precede checker zone (check_structure_X, check_answer_X)"

requirements-completed:
  - REQ-CLEAN-01
  - REQ-CLEAN-02

duration: 12min
completed: 2026-03-10
---

# Phase 3 Plan 01: File Cleanup — Type Aliases and Utility Primitives Summary

**TypeAlias import plus CheckResult/PartialScoreDict aliases added; check_key_structure, convert_to_set_of_sets, and is_sequence_but_not_str consolidated into SECTION 3 before all checker functions**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-10T23:53:00Z
- **Completed:** 2026-03-10T23:55:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `TypeAlias` to the `from typing import` line and defined `CheckResult` and `PartialScoreDict` aliases in SECTION 1
- Wrapped `FLOAT_TOL` in a SECTION 2: CONSTANTS header
- Inserted SECTION 3: UTILITY PRIMITIVES header after `are_sets_equal` and moved three previously scattered helpers there

## Section Line Ranges (after edits)

| Section | Header Line | Content |
|---------|------------|---------|
| SECTION 1: TYPE ALIASES | 41 | CheckResult (line 43), PartialScoreDict (line 44) |
| SECTION 2: CONSTANTS | 47 | FLOAT_TOL (line 50) |
| SECTION 3: UTILITY PRIMITIVES | 596 | all helpers through is_sequence_but_not_str (lines 600-668) |

## Three Helpers Moved

| Function | Old Line | New Line | Notes |
|----------|----------|----------|-------|
| check_key_structure | ~1160 | 600 | No callers above old location; moved first |
| convert_to_set_of_sets | ~3208 | 640 | Callers at ~3219-3220 remain in place |
| is_sequence_but_not_str | ~3271 | 655 | Callers at ~3269, 3278 remain in place |

## Task Commits

1. **Task 1: Add type aliases and section headers for Sections 1-2** - `890fc97` (feat)
2. **Task 2: Consolidate scattered helpers into SECTION 3 UTILITY PRIMITIVES** - `6d8409e` (feat)

## Files Created/Modified

- `src/pytest_generator/assert_utilities.py` - Added 3 section headers, 2 type aliases, TypeAlias import; moved 3 helper functions to Section 3

## Decisions Made

- Type aliases are defined only — not applied as replacements throughout file (deferred to Phase 4)
- No second SECTION 3 banner needed for the original utilities block (init_partial_score_dict through are_sets_equal); the single header at line 596 covers the entire section

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- SECTION 1, 2, 3 are complete and stable — Plan 03-02 (checker reorganization) can proceed
- All 183 regression tests pass at plan completion

---
*Phase: 03-file-cleanup*
*Completed: 2026-03-10*
