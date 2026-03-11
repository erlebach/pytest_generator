# Phase 1: Regression Safety Net - Research

**Researched:** 2026-03-10
**Domain:** pytest, Python testing, assert_utilities.py checker inventory
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REQ-INFRA-01 | Comprehensive regression tests for all existing checker functions (written BEFORE any refactoring begins) | Checker inventory complete: 103 unique names (44 check_answer_*, 47 check_structure_*, 12 other check_* helpers). pytest 8.3.5 is already installed. uv run pytest is the invocation. |
</phase_requirements>

---

## Summary

`assert_utilities.py` contains 103 unique checker function definitions (confirmed via Python AST parsing). Breakdown: 44 `check_answer_*`, 47 `check_structure_*`, and 12 helper `check_*` functions (e.g. `check_float`, `check_int`, `check_str`, `check_list_float`, etc.). There are 2 true duplicate definitions where the same name appears twice as real Python code: `check_answer_dict_str_int` (lines 1796, 2348) and `check_structure_dict_str_list_str` (lines 826, 1009) — Python silently uses the last definition. The other apparent duplicates found by grep were inside `'''...'''` triple-quoted comment blocks and are not active code. Tests must cover all 103 unique names, targeting the last-defined version for the 2 real duplicates.

All functions return `tuple[bool, str]`. The `bool` component indicates correctness; the `str` component is a human-readable message. Tests must assert both components. The package is managed with `uv` and `pyproject.toml`; `pytest 8.3.5` is already a declared dependency. The correct invocation is `uv run pytest`.

No `tests/` directory exists yet. No `pytest.ini` or `[tool.pytest.ini_options]` section exists. A `conftest.py` exists only inside `template/tests/` (not at root). Wave 0 must create the test directory and test file before any implementation work begins.

**Primary recommendation:** Create `tests/test_regression_checkers.py` that imports from `src.pytest_generator.assert_utilities` and exercises every unique checker function with at least one valid-pass case and one valid-fail case, asserting the `(bool, str)` tuple returned.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | 8.3.5 (declared in pyproject.toml) | Test runner | Already declared dependency; used throughout project |
| numpy | 2.2.3 (declared in pyproject.toml) | ndarray construction for checker fixtures | Required by assert_utilities.py itself |
| matplotlib | 3.10.1 (declared in pyproject.toml) | Line2D / PathCollection fixtures for plot checkers | Required by assert_utilities.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | NOT installed | DecisionTreeClassifier, SVC, KFold, etc. fixtures | Only if ML checker tests are in scope; currently missing from pyproject.toml dependencies |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest parametrize | Manual test-per-function | parametrize reduces boilerplate for pass/fail case pairs but makes individual failures harder to identify — direct named test functions are clearer for this inventory |

**Installation:**
```bash
# All core deps already installed via uv
uv sync
# If sklearn checkers are to be tested:
uv add scikit-learn
```

---

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── conftest.py              # shared fixtures (numpy arrays, matplotlib objects, etc.)
└── test_regression_checkers.py  # one test function per checker (pass + fail)
```

### Pattern 1: Named test functions with explicit bool + str assertions
**What:** One test function per checker, with a valid-pass case and a valid-fail case, each asserting the full `(bool, str)` return tuple.
**When to use:** Always — matches success criterion 4 exactly.
**Example:**
```python
# Source: assert_utilities.py lines 621-641 (check_structure_float)
def test_check_structure_float_valid():
    status, msg = check_structure_float(3.14)
    assert status is True
    assert isinstance(msg, str)
    assert len(msg) > 0

def test_check_structure_float_invalid():
    status, msg = check_structure_float("not a float")
    assert status is False
    assert isinstance(msg, str)
    assert len(msg) > 0
```

### Pattern 2: conftest.py shared fixtures
**What:** Reusable pytest fixtures for expensive or complex objects (numpy arrays, matplotlib figures, sklearn models).
**When to use:** Any fixture used by 3+ test functions.
**Example:**
```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_ndarray():
    return np.array([1.0, 2.0, 3.0])

@pytest.fixture
def sample_ndarray_2d():
    return np.random.rand(4, 3)
```

### Pattern 3: Inventory-driven approach
**What:** The plan must first enumerate all unique checker names by introspecting the source file, then generate tests for each. Python duplicate definitions mean the last definition wins — test the last-defined version.
**When to use:** Required given the 7 known duplicated names.

### Anti-Patterns to Avoid
- **Testing the import only:** Calling a function and not asserting both the bool and the str violates success criterion 4.
- **Mocking assert_utilities internals:** Tests must call real functions against real inputs to serve as regression protection.
- **Single case per function:** A function that only passes is not a regression test — you need a failing case to confirm the function can distinguish bad input.
- **Relying on sklearn without declaring it:** scikit-learn is NOT in pyproject.toml. ML checker tests will fail unless the dependency is added.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Test parametrization | Manual loop over cases | `@pytest.mark.parametrize` | Built into pytest 8.x; cleaner failure reporting |
| Float comparison in tests | Custom tolerance check | `pytest.approx` | Handles relative/absolute tolerance correctly |
| Fixture management | Global test state | `@pytest.fixture` with appropriate scope | Ensures isolation between tests |

---

## Common Pitfalls

### Pitfall 1: Duplicate function names — testing the wrong definition
**What goes wrong:** Python silently shadows earlier definitions with later ones. If you test by importing by name, you always get the last definition. But if the plan says "cover all definitions," a naive reader might think 7 extra tests are needed.
**Why it happens:** The file has 7 names defined 2+ times (check_answer_dict_str_float x2, check_answer_dict_str_int x2, check_answer_randomforestclassifier x2, check_structure_dict_str_float x2, check_structure_dict_str_list_str x2, check_structure_kfold x2, check_structure_shufflesplit x2).
**How to avoid:** Import by name — Python resolves to last definition. The regression test covers the currently active behavior. Phase 2 (duplicate removal) will reconcile which version to keep; the Phase 1 test will catch if behavior changes during that reconciliation.
**Warning signs:** If a test passes when the duplicate is removed but behavior changed, the test was testing the wrong variant.

### Pitfall 2: sklearn not installed
**What goes wrong:** Tests for DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, SVC, KFold, StratifiedKFold, ShuffleSplit, GridSearchCV checkers will fail with ImportError at collection time.
**Why it happens:** scikit-learn is not declared in pyproject.toml.
**How to avoid:** Either (a) add scikit-learn to pyproject.toml dependencies and run `uv sync`, or (b) mark ML checker tests with `@pytest.mark.skipif(not sklearn_available, ...)` with a note that full coverage requires sklearn.
**Warning signs:** `ModuleNotFoundError: No module named 'sklearn'` during `uv run pytest`.

### Pitfall 3: matplotlib plot objects require figure context
**What goes wrong:** `check_structure_lineplot`, `check_structure_scatterplot2d`, `check_structure_scatterplot3d`, and their answer counterparts require `Line2D` or `PathCollection` objects that must be created via matplotlib API calls, not constructed directly.
**Why it happens:** These objects are produced by `plt.plot()`, `plt.scatter()`, `ax.scatter3D()` — not constructable from scratch trivially.
**How to avoid:** In conftest.py, create fixtures using matplotlib's non-interactive backend (`matplotlib.use('Agg')`) that generate actual plot objects.
**Warning signs:** `AttributeError` on `get_offsets()` or `_offsets3d` when passing wrong object type.

### Pitfall 4: Asymmetric structure/answer pairs
**What goes wrong:** `check_structure_dict_str_set` and `check_structure_dict_any` exist without a corresponding `check_answer_*`. Tests for these structure-only checkers cannot test an answer pair.
**Why it happens:** Some types are partially implemented.
**How to avoid:** Test structure-only checkers as structure-only (pass/fail type check). Do not skip them — they must still be covered.

### Pitfall 5: check_structure_dict_int_list exists but no check_answer_dict_int_list
**What goes wrong:** Similar to above — structure checker exists (line 2442) but the `check_answer_dict_int_list` (line 2515) is actually named `check_answer_dict_int_list_float` in the unique list.
**How to avoid:** Cross-check the full list before writing tests; do not assume symmetry.

---

## Code Examples

### Importing from installed package
```python
# Source: confirmed via uv run python3 invocation
from src.pytest_generator.assert_utilities import (
    check_structure_float,
    check_answer_float,
    check_structure_int,
    check_answer_int,
    # ... etc
)
```

### Non-interactive matplotlib fixture (conftest.py)
```python
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import
import matplotlib.pyplot as plt
import pytest

@pytest.fixture
def line2d_fixture():
    fig, ax = plt.subplots()
    lines = ax.plot([1, 2, 3], [4, 5, 6])
    yield lines[0]
    plt.close(fig)

@pytest.fixture
def scatter2d_fixture():
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [4, 5, 6])
    yield sc
    plt.close(fig)
```

### Testing both tuple components
```python
def test_check_answer_float_correct():
    status, msg = check_answer_float(
        student_answer=1.0,
        instructor_answer=1.0,
        rel_tol=1e-5,
        abs_tol=1e-8,
    )
    assert status is True
    assert isinstance(msg, str)

def test_check_answer_float_incorrect():
    status, msg = check_answer_float(
        student_answer=999.0,
        instructor_answer=1.0,
        rel_tol=1e-5,
        abs_tol=1e-8,
    )
    assert status is False
    assert isinstance(msg, str)
    assert len(msg) > 0
```

---

## Checker Inventory

Complete list of unique checker names to cover (91 total):

**check_answer_* (44):**
check_answer_bool, check_answer_decisiontreeclassifier, check_answer_dendrogram,
check_answer_dict_int_dict_str_any, check_answer_dict_int_float,
check_answer_dict_int_list_float, check_answer_dict_int_ndarray,
check_answer_dict_str_any, check_answer_dict_str_dict_str_float,
check_answer_dict_str_float, check_answer_dict_str_int,
check_answer_dict_str_list_int, check_answer_dict_str_list_str,
check_answer_dict_str_ndarray, check_answer_dict_str_set_int,
check_answer_dict_str_tuple_ndarray, check_answer_dict_tuple_int_ndarray,
check_answer_eval_float, check_answer_explain_str, check_answer_float,
check_answer_function, check_answer_gridsearchcv, check_answer_int,
check_answer_kfold, check_answer_lineplot, check_answer_list_float,
check_answer_list_int, check_answer_list_list_float, check_answer_list_ndarray,
check_answer_list_set, check_answer_list_str, check_answer_list_tuple_float,
check_answer_logisticregression, check_answer_ndarray,
check_answer_randomforestclassifier, check_answer_scatterplot2d,
check_answer_scatterplot3d, check_answer_set_set_int, check_answer_set_str,
check_answer_set_tuple_int, check_answer_shufflesplit, check_answer_str,
check_answer_stratifiedkfold, check_answer_svc

**check_structure_* (45):**
check_structure_bool, check_structure_decisiontreeclassifier,
check_structure_dendrogram, check_structure_dict_any,
check_structure_dict_int_dict_str_any, check_structure_dict_int_float,
check_structure_dict_int_list, check_structure_dict_int_list_float,
check_structure_dict_int_ndarray, check_structure_dict_str_any,
check_structure_dict_str_dict_str_float, check_structure_dict_str_float,
check_structure_dict_str_int, check_structure_dict_str_list_int,
check_structure_dict_str_list_str, check_structure_dict_str_ndarray,
check_structure_dict_str_set, check_structure_dict_str_set_int,
check_structure_dict_str_tuple_ndarray, check_structure_dict_tuple_int_ndarray,
check_structure_eval_float, check_structure_explain_str, check_structure_float,
check_structure_function, check_structure_gridsearchcv, check_structure_int,
check_structure_kfold, check_structure_lineplot, check_structure_list_float,
check_structure_list_int, check_structure_list_list_float,
check_structure_list_ndarray, check_structure_list_set, check_structure_list_str,
check_structure_list_tuple_float, check_structure_logisticregression,
check_structure_ndarray, check_structure_randomforestclassifier,
check_structure_scatterplot2d, check_structure_scatterplot3d,
check_structure_set_set_int, check_structure_set_str, check_structure_set_tuple_int,
check_structure_shufflesplit, check_structure_str, check_structure_stratifiedkfold,
check_structure_svc

**Known duplicate definitions (last definition wins in Python):**
- check_answer_dict_str_float (lines 4673, 4734)
- check_answer_dict_str_int (lines 1796, 2348)
- check_answer_randomforestclassifier (lines 4470, 4512)
- check_structure_dict_str_float (lines 4562, 4620)
- check_structure_dict_str_list_str (lines 826, 1009)
- check_structure_kfold (lines 4207, 4225)
- check_structure_shufflesplit (lines 4336, 4354)

**Structure-only (no answer counterpart):**
- check_structure_dict_str_set
- check_structure_dict_any

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3.5 |
| Config file | none — Wave 0 adds `[tool.pytest.ini_options]` to `pyproject.toml` |
| Quick run command | `uv run pytest tests/test_regression_checkers.py -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-INFRA-01 | Every check_structure_* and check_answer_* function is covered by at least one pass-case and one fail-case test, each asserting both tuple components | unit | `uv run pytest tests/test_regression_checkers.py -q` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_regression_checkers.py -x -q`
- **Per wave merge:** `uv run pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/__init__.py` — makes tests/ a package
- [ ] `tests/conftest.py` — shared numpy, matplotlib, sklearn fixtures
- [ ] `tests/test_regression_checkers.py` — covers REQ-INFRA-01
- [ ] `pyproject.toml` update — add `[tool.pytest.ini_options]` with `testpaths = ["tests"]`
- [ ] sklearn dependency decision: `uv add scikit-learn` OR skip-markers for ML checkers

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pytest 7.x conftest patterns | pytest 8.x (no breaking changes for this use case) | 2024 | None — same API for fixtures and parametrize |

---

## Open Questions

1. **sklearn dependency**
   - What we know: scikit-learn is not in pyproject.toml; 8 checker functions depend on it (DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, SVC, KFold, StratifiedKFold, ShuffleSplit, GridSearchCV)
   - What's unclear: Whether the project intends to add sklearn as a test dependency or skip those tests
   - Recommendation: Add `scikit-learn` to pyproject.toml dependencies and include full ML checker tests; the regression safety net should be complete

2. **check_answer_dict_str_set — missing**
   - What we know: `check_structure_dict_str_set` exists at line 1623, but no `check_answer_dict_str_set` is defined
   - What's unclear: Is this intentional (partial implementation) or a bug?
   - Recommendation: Note in test file as a known gap; do not fabricate a test for a non-existent function

3. **check_answer_dict_int_list — missing**
   - What we know: `check_structure_dict_int_list` at line 2442 exists; the answer-side is `check_answer_dict_int_list_float` (not `check_answer_dict_int_list`)
   - Recommendation: Test `check_structure_dict_int_list` as structure-only; no answer counterpart to test

---

## Sources

### Primary (HIGH confidence)
- Direct read of `/src/pytest_generator/assert_utilities.py` — full function inventory, signature inspection, return pattern inspection
- `/pyproject.toml` — pytest 8.3.5, numpy 2.2.3, matplotlib 3.10.1 confirmed declared
- `uv run python3` invocation — confirmed import path `from src.pytest_generator.assert_utilities import ...` works, returns `tuple[bool, str]`

### Secondary (MEDIUM confidence)
- `.planning/REQUIREMENTS.md` — REQ-INFRA-01 confirmed as Phase 1 target
- `.planning/STATE.md` — confirmed blocker: "full enumeration of all checker functions needed"

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pytest/numpy/matplotlib versions confirmed in pyproject.toml; import confirmed via uv run
- Architecture: HIGH — based on direct codebase inspection; function inventory is complete
- Pitfalls: HIGH for sklearn/duplicate/matplotlib issues (confirmed by inspection); MEDIUM for edge cases within specific checker implementations

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable codebase; no external dependency changes expected)
