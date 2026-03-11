---
phase: 03-file-cleanup
plan: 02
subsystem: assert_utilities
tags: [reorganization, sections, structure-checks, answer-checks]
dependency_graph:
  requires: [03-01]
  provides: [SECTION4-structure-checks, SECTION5-answer-checks]
  affects: [assert_utilities.py]
tech_stack:
  added: []
  patterns: [5-section-layout, alphabetical-ordering]
key_files:
  created: []
  modified:
    - src/pytest_generator/assert_utilities.py
decisions:
  - Reorganized all checker functions into two named sections (4 and 5) alphabetically
  - Used a Python script to extract, sort, and reassemble functions atomically to avoid line-shift errors
metrics:
  duration: ~5 minutes
  completed_date: "2026-03-11"
  tasks_completed: 3
  files_modified: 1
---

# Phase 03 Plan 02: Section 4 and 5 Checker Reorganization Summary

One-liner: All 91 check_structure/check_answer functions reorganized into two named sections with alphabetical ordering.

## What Was Built

Reorganized `assert_utilities.py` to have a complete 5-section layout:

| Section | Header | Line Range | Contents |
|---------|--------|------------|----------|
| SECTION 1 | TYPE ALIASES | 40-45 | CheckResult, PartialScoreDict |
| SECTION 2 | CONSTANTS | 46-594 | Configuration constants |
| SECTION 3 | UTILITY PRIMITIVES | 595-668 | Helper functions (check_key_structure, etc.) |
| SECTION 4 | STRUCTURE CHECKS | 669-2722 | 47 check_structure_X functions |
| SECTION 5 | ANSWER CHECKS | 2723-5077 | 44 check_answer_X functions |

## Checker Function Inventory

### SECTION 4: Structure Checks (47 functions, lines 672-2720)
Alphabetically ordered by type suffix:
`bool`, `decisiontreeclassifier`, `dendrogram`, `dict_any`, `dict_int_dict_str_any`, `dict_int_float`, `dict_int_list`, `dict_int_list_float`, `dict_int_ndarray`, `dict_str_any`, `dict_str_dict_str_float`, `dict_str_float`, `dict_str_int`, `dict_str_list_int`, `dict_str_list_str`, `dict_str_ndarray`, `dict_str_set`, `dict_str_set_int`, `dict_str_tuple_ndarray`, `dict_tuple_int_ndarray`, `eval_float`, `explain_str`, `float`, `function`, `gridsearchcv`, `int`, `kfold`, `lineplot`, `list_float`, `list_int`, `list_list_float`, `list_ndarray`, `list_set`, `list_str`, `list_tuple_float`, `logisticregression`, `ndarray`, `randomforestclassifier`, `scatterplot2d`, `scatterplot3d`, `set_set_int`, `set_str`, `set_tuple_int`, `shufflesplit`, `str`, `stratifiedkfold`, `svc`

### SECTION 5: Answer Checks (44 functions, lines 2726-5077)
Alphabetically ordered by type suffix (same sort key):
`bool`, `decisiontreeclassifier`, `dendrogram`, `dict_int_dict_str_any`, `dict_int_float`, `dict_int_list_float`, `dict_int_ndarray`, `dict_str_any`, `dict_str_dict_str_float`, `dict_str_float`, `dict_str_int`, `dict_str_list_int`, `dict_str_list_str`, `dict_str_ndarray`, `dict_str_set_int`, `dict_str_tuple_ndarray`, `dict_tuple_int_ndarray`, `eval_float`, `explain_str`, `float`, `function`, `gridsearchcv`, `int`, `kfold`, `lineplot`, `list_float`, `list_int`, `list_list_float`, `list_ndarray`, `list_set`, `list_str`, `list_tuple_float`, `logisticregression`, `ndarray`, `randomforestclassifier`, `scatterplot2d`, `scatterplot3d`, `set_set_int`, `set_str`, `set_tuple_int`, `shufflesplit`, `str`, `stratifiedkfold`, `svc`

Note: `check_structure_dict_any` and `check_structure_dict_str_set` (no corresponding answer function) are structure-only types.

## File Metrics

- Before: 5301 lines (interleaved, no section headers for 4/5)
- After: 5077 lines (5 sections, alphabetically ordered)
- Net reduction: 224 lines (whitespace normalization)

## Execution Approach

Used a Python script to atomically reorganize:
1. Parse all function boundaries (def to next def/separator)
2. Separate into check_structure_X and check_answer_X groups
3. Sort each group alphabetically by type suffix
4. Write pre-section4 content + SECTION 4 header + structure funcs + SECTION 5 header + answer funcs

This atomic approach avoided line-shift errors from sequential edits.

## Deviations from Plan

None — plan executed exactly as written, using a single-script atomic rewrite instead of sequential moves (cleaner and safer for 91 functions).

## Test Results

- Import: OK
- Regression suite: 183/183 passed
- Duration: 0.89s

## Self-Check: PASSED
