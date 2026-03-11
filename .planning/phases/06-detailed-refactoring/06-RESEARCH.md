# Phase 6: Detailed Refactoring - Research

**Researched:** 2026-03-10
**Domain:** Python code quality audit — debug statements, dead commented code, logic bugs, misleading comments
**Confidence:** HIGH (direct grep/AST analysis of the actual source file)

## Summary

Phases 1-4 cleaned the structural skeleton of `assert_utilities.py` (deduplication, section organization, signature standardization, type annotations). What remains is cosmetic-but-consequential: 67 live `print(...)` debug statements scattered across utility helpers and checker functions, plus `pprint` calls in one function; a stale multi-line string block holding commented-out imports at module scope; at least 28 commented-out code lines using `# !` and `# msg_list.append` conventions; three active TODO markers; and one confirmed logic inversion bug in `check_structure_dict_str_set`.

The file is 5,077 lines as of this research. The regression suite (183 tests) runs in about 1.2 seconds with `uv run pytest`.

**Primary recommendation:** Work in strict order — logic bug fix first (REQ-AUDIT-01), then print removal (highest regression risk), then commented-out code cleanup, then TODO annotation or resolution.

## Issue Inventory

### Category A: Debug print() statements (67 occurrences)

Every one of these is a debug artifact — none logs to a proper handler. They must all be deleted.

Key clusters by function:

| Function | Lines | Count | Notes |
|----------|-------|-------|-------|
| `check_float` | 115, 120 | 2 | Prints on every call |
| `check_list_int` | 217 | 1 | Prints `ps_dict` on every call |
| `check_answer_bool` (helper near 546) | 546–547 | 2 | |
| `check_structure_bool` | 684 | 1 | |
| `check_structure_dict_str_int` | 1362–1363 | 2 | |
| `check_structure_dict_str_list_int` | 1418–1419 | 2 | |
| `check_structure_list_tuple_float` | 2365–2387 | 6 | Dense cluster |
| `check_structure_set_str` | 2572, 2604, 2609, 2614, 2619, 2623 | 6 | |
| `check_answer_dict_int_dict_str_any` | 2869–2875 | 6 | Also uses `pprint` |
| `check_answer_dict_int_float` | 2978, 2981 | 2 | Prints ZeroDivisionError as plain text |
| `check_answer_dict_str_float` | 3314, 3330, 3332, 3394, 3399 | 5 | |
| `check_answer_dict_str_int` | 3414–3415, 3438, 3442 | 4 | |
| `check_answer_dict_str_list_int` | 3471, 3487, 3496 | 3 | |
| `check_answer_dict_str_list_str` | 3559–3560, 3562, 3574, 3577, 3588, 3592–3594 | 8 | |
| `check_answer_list_str` | 4423 | 1 | |
| `check_structure_list_tuple_float` / `check_answer_list_tuple_float` | 4464–4467, 4474, 4499, 4501–4502, 4504, 4512 | 10 | Mixed structure/answer functions |
| `check_answer_set_str` | 4832, 4835, 4866 | 3 | |
| `check_answer_set_tuple_int` | 4894–4896, 4902 | 4 | |

Additionally `pprint` is called at lines 2871 and 2873 inside `check_answer_dict_int_dict_str_any`. The `pprint` import at line 21 becomes unused after removal; delete it too. (The commented duplicate `# from pprint import pprint` at line 36 is in the stale block described below.)

### Category B: Logic inversion bug in check_structure_dict_str_set

**Location:** Lines 1630–1643

**Bug:** The condition on line 1633 is:
```python
if k in keys and isinstance(v, set | list):
```
This fires an ERROR message and sets `status = False` when the value IS a `set` or `list` — the opposite of the intended behavior. The sibling function `check_structure_dict_str_set_int` (line 1687) has the correct condition `not isinstance(v, set | list)`. The bug means `check_structure_dict_str_set` always reports an error for correctly-typed values and passes for incorrectly-typed values.

**Fix:** Insert `not` before `isinstance` on line 1633.

There is also a stale/broken comment on line 1636:
```python
# msg_list.append(f"- Answer[repr(k)r}] must be of type 'set' or 'list'.")
```
This has a syntax error in the f-string (`repr(k)r}` is malformed). Delete it.

### Category C: Stale module-level multi-line string block (lines 35–38)

Lines 35–38 are a bare string literal (not assigned, not a docstring) that wraps two commented-out import lines:
```python
"""
# from pprint import pprint
# import matplotlib.pyplot as plt
"""
```
This is dead code masquerading as a comment block. Delete the entire block (lines 35–38 inclusive).

### Category D: Commented-out code lines (28 occurrences)

These use two conventions: `# !` prefix (for lines that were live code) and `# msg_list.append(...)` for debug appends. All are dead code.

Key examples:
- Lines 24, 32: `# ! import matplotlib.pyplot as plt` and `# ! from matplotlib.collections import PathCollection` — commented-out imports at module scope. If they are truly unused, delete. If `PathCollection` may be needed, leave a note; otherwise delete.
- Lines 515–516: `# ! return repr(x)...` and `# ! if isinstance(...)` — dead alternative logic in a helper.
- Lines 543–544: `# msg_list.append(...)` — disabled debug appends.
- Lines 1264–1266: `# ! print(...)` block — already commented-out prints.
- Lines 2055–2059: `# ! i_xlabel = ...` block (3 lines) — dead assignments.
- Lines 2669: `# ! choices = ...` — dead assignment.
- Lines 3215–3217: `# msg_list.append(f"DEBUG: ...")` cluster (3 lines).
- Lines 3403–3405: identical DEBUG cluster (3 lines).
- Lines 3675: `# ! TODO: check that keys are in the instructor answer`
- Lines 3739: `# ! print(f"{i_norms=}...")` — already commented-out print.
- Lines 3796–3797: `# ! print(...)` and `# ! return False, ""` — dead return path.
- Lines 4429–4430: `# msg_list += [...]` pair — dead debug lines.
- Line 4981: `# ! print(f"check_answer_str, {remove_spaces=}")`

### Category E: Active TODO markers (3 occurrences)

| Line | Content | Action |
|------|---------|--------|
| 822 | `## ARE WE CHECKING THAT THE KEYS ARE int ! TODO` | Investigate: does `check_structure_dict_int_dict_str_any` actually verify top-level keys are `int`? Either fix or convert to a plain comment explaining the known gap. |
| 3675 | `# ! TODO: check that keys are in the instructor answer` | Investigate same gap in `check_answer_dict_str_set_int`. Fix or document. |
| 4427 | `# ! TODO: Explicitly state the indices considered for grading.` | Low priority; convert to a plain comment or open a follow-up issue. |

### Category F: Misleading/stale comments

- Line 1636: Broken f-string in a comment (described in Category B).
- The `"instructor+answer"` typo in print at line 2872 is not a standalone comment but a print argument — removed as part of Category A.
- The `## ARE WE CHECKING...` at line 822 uses `##` (double-hash) inconsistently with the rest of the file style.

## Architecture Patterns

No new patterns are introduced by Phase 6. All changes are deletions or one-line fixes. The existing section structure from Phase 3 remains intact.

### Recommended Change Order

1. Fix logic bug (line 1633) — highest correctness impact.
2. Delete stale module-level string block (lines 35–38).
3. Delete all `print(...)` and `pprint(...)` statements (67 + 2 calls).
4. Remove `from pprint import pprint` import (line 21) after step 3.
5. Delete commented-out code lines (Category D).
6. Resolve or annotate TODO markers (Category E).
7. Run regression suite after each step to isolate regressions.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Confirming zero prints remain | Manual grep | `grep -n "^\s*print(" src/pytest_generator/assert_utilities.py` — must return empty |
| Confirming pprint removed | Manual grep | `grep -n "pprint" src/pytest_generator/assert_utilities.py` — must return 0 live lines |
| Confirming logic bug is fixed | New test case | Add test: pass correctly-typed dict, expect `True` from `check_structure_dict_str_set` |

## Common Pitfalls

### Pitfall 1: Removing a print but forgetting pprint
**What goes wrong:** `pprint` calls at lines 2871 and 2873 are not matched by `print(` grep.
**How to avoid:** Run a separate grep for `pprint\b` after print removal.

### Pitfall 2: Deleting a commented-out import that is actually needed
**What goes wrong:** `matplotlib.pyplot` and `PathCollection` are commented out at lines 24 and 32. If any remaining function silently depends on pyplot being side-effect imported, removal may cause failures.
**How to avoid:** Run full test suite after removing each commented import. Both are already commented out so the risk is low, but verify.

### Pitfall 3: The logic bug fix changes existing (incorrect) test baselines
**What goes wrong:** Phase 1 regression tests document pre-existing behavior as baselines. If a test was written to match the buggy behavior of `check_structure_dict_str_set`, fixing the bug will break that test.
**How to avoid:** Check the test file for `check_structure_dict_str_set` before fixing. Update the test to reflect correct behavior, not the bug.

### Pitfall 4: Stale string block deletion shifts line numbers
**What goes wrong:** Deleting lines 35–38 shifts all subsequent line numbers down by 4, breaking any external references.
**How to avoid:** This is internal cleanup; no external references to specific line numbers exist. Safe to delete.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (via uv) |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/test_assert_utilities.py -q` |
| Full suite command | `uv run pytest -q` |

### Phase Requirements to Test Map

| ID | Behavior | Test Type | Automated Command |
|----|----------|-----------|-------------------|
| REQ-AUDIT-01 | No debug prints in file | grep smoke | `grep -c "^\s*print(" src/pytest_generator/assert_utilities.py` must output `0` |
| REQ-AUDIT-01 | No pprint calls | grep smoke | `grep -c "pprint(" src/pytest_generator/assert_utilities.py` must output `0` |
| REQ-AUDIT-01 | Logic bug in check_structure_dict_str_set fixed | unit | Add test: correctly-typed dict must return `(True, ...)` |
| REQ-AUDIT-01 | Regression tests still pass | regression | `uv run pytest tests/test_assert_utilities.py -q` — 183 passed |

### Per-Fix Verification Steps

1. **Print removal:** `grep -c "^\s*print(" src/pytest_generator/assert_utilities.py` → `0`
2. **pprint removal:** `grep -c "pprint" src/pytest_generator/assert_utilities.py` → `0`
3. **Logic bug fix:** new unit test for `check_structure_dict_str_set` with a correct-type input returns `True`
4. **Commented-out code:** `grep -c "# !" src/pytest_generator/assert_utilities.py` → `0`
5. **Module-level stale block:** `grep -n "from pprint import pprint" src/pytest_generator/assert_utilities.py` returns at most 0 live lines
6. **Full regression:** `uv run pytest -q` → `183 passed` (or more if new tests added)

### Sampling Rate

- **Per fix commit:** `uv run pytest tests/test_assert_utilities.py -q`
- **Phase gate:** Full suite green before marking phase complete

### Wave 0 Gaps

- [ ] Add test case for `check_structure_dict_str_set` with correctly-typed input (currently the bug means passing inputs return `False` — the existing test baseline may encode the broken behavior)

## Sources

### Primary (HIGH confidence)

- Direct grep analysis of `/src/pytest_generator/assert_utilities.py` (5,077 lines, 2026-03-10)
- AST analysis via `python3 -c "import ast ..."` — no unreachable-after-return code found
- `uv run pytest --collect-only -q` — 183 tests collected

### Secondary (MEDIUM confidence)

- `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/REQUIREMENTS.md` — project decisions and constraints

## Metadata

**Confidence breakdown:**
- Issue inventory (prints, comments, logic bug): HIGH — direct source grep
- TODO resolution strategy: MEDIUM — requires reading function context to confirm fix correctness
- Test impact of logic bug fix: MEDIUM — depends on whether Phase 1 tests encoded the buggy behavior

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable codebase, no external dependencies)
