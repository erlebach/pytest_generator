---
phase: 01-regression-safety-net
plan: "03"
subsystem: testing
tags: [regression, tests, answer-checkers, inventory]
dependency_graph:
  requires: [01-02]
  provides: [complete-regression-safety-net]
  affects: [assert_utilities.py]
tech_stack:
  added: []
  patterns: [ast-parse-at-runtime, regression-baseline-documentation]
key_files:
  created: []
  modified:
    - tests/test_assert_utilities.py
decisions:
  - "Document pre-existing bugs as regression baselines rather than fixing them (consistent with Phase 1 policy)"
  - "check_answer_str requires positional str_choices and remove_spaces arguments"
  - "check_answer_dict_str_int always returns True due to keys-is-empty bug — baseline documented"
  - "check_answer_scatterplot2d/3d fail cases document NameError (mcolors not imported)"
metrics:
  duration: "~25 minutes"
  completed: "2026-03-10"
  tasks_completed: 2
  files_modified: 1
---

# Phase 1 Plan 03: Answer Checker Tests and Inventory Summary

**One-liner:** 88 check_answer_* regression tests plus AST-based drift-detector inventory test completing the Phase 1 safety net.

## What Was Built

Appended to `tests/test_assert_utilities.py`:

- 88 test functions (2 per function) covering all 44 `check_answer_*` functions
- 1 `test_inventory_completeness` function using `ast.parse` at runtime to verify no checker function lacks a test
- Total: 183 tests passing (`uv run pytest tests/` exits 0)

## Deviations from Plan

### Auto-fixed Issues

None — all deviations are pre-existing bugs documented as regression baselines.

### Pre-existing Bugs Documented as Regression Baselines

**1. check_answer_str signature**
- **Found during:** Task 1
- **Issue:** Requires two additional positional arguments: `str_choices` and `remove_spaces`
- **Fix:** Tests pass `["hello", "world"], False` as additional args
- **Files modified:** tests/test_assert_utilities.py

**2. check_answer_dict_str_int always returns True**
- **Found during:** Task 1
- **Issue:** Pre-existing bug — `keys is []` identity check always False, so keys stays empty list, loop body never executes
- **Fix:** Test asserts `status is True` with comment documenting bug

**3. check_answer_dict_str_dict_str_float ZeroDivisionError**
- **Found during:** Task 1
- **Issue:** `ps_dict["nb_total"]` remains 0, causing division by zero
- **Fix:** Tests assert `pytest.raises(ZeroDivisionError)`

**4. check_answer_dict_tuple_int_ndarray ZeroDivisionError**
- **Found during:** Task 1
- **Issue:** `ps_dict` variable referenced but never initialized in the function
- **Fix:** Tests assert `pytest.raises((ZeroDivisionError, NameError))`

**5. check_answer_scatterplot2d NameError**
- **Found during:** Task 1
- **Issue:** `mcolors` used but not imported in assert_utilities.py
- **Fix:** Tests assert `pytest.raises(NameError)`

**6. check_answer_explain_str, check_answer_function, check_answer_lineplot always return True**
- **Found during:** Task 1
- **Issue:** These functions are not graded (by design or unimplemented comparison)
- **Fix:** Fail tests assert `status is True` with regression baseline comments

**7. check_answer_list_float, check_answer_list_int: partial_score_frac required**
- **Found during:** Task 1
- **Issue:** Despite optional type hint, code crashes without a non-empty list
- **Fix:** Tests pass `partial_score_frac=[0.0]` explicitly

## Deferred Items

None.

## Self-Check

- [x] tests/test_assert_utilities.py exists and has 183 passing tests
- [x] test_inventory_completeness passes
- [x] Commit e714aaf exists with all changes
