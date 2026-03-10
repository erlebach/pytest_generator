# Phase 1: Regression Safety Net - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Write regression tests covering all existing `check_structure_X` and `check_answer_X` functions in `assert_utilities.py` before any refactoring begins. This is a pure test-writing phase — no production code changes.

</domain>

<decisions>
## Implementation Decisions

### Coverage scope
- Cover `check_structure_X` and `check_answer_X` functions only (91 unique names: 47 structure + 44 answer)
- The 12 helper `check_*` functions (check_float, check_int, check_str, etc.) are NOT explicitly tested — they are implementation details covered implicitly
- For the 2 true duplicate definitions, test the last (active) definition only — Python silently uses it anyway
- Include an automated inventory test that enumerates all check_structure_X / check_answer_X names from assert_utilities.py at runtime and fails if any are untested — catches future drift

### scikit-learn dependency
- Add `scikit-learn` to `pyproject.toml` dependencies (`uv add scikit-learn`)
- All 8 ML checker functions (KFold, SVC, GridSearchCV, RandomForestClassifier, ShuffleSplit, etc.) must be tested with real sklearn objects — no skips, no stubs

### Test organization
- Single test file: `tests/test_assert_utilities.py`
- One test function per checker: `def test_check_answer_float()`, `def test_check_structure_dict_str_int()`, etc.
- No test classes, no parametrize over checker names
- `tests/__init__.py` to make it a package
- `tests/conftest.py` for shared fixtures

### Test depth
- Each test function contains exactly two cases: one pass (returns `True, <msg>`) and one fail (returns `False, <msg>`)
- The `str` component is validated as a non-empty string only — no assertion on specific message content (keeps tests stable through message wording changes during refactoring)
- Plot checker fixtures (matplotlib Line2D, PathCollection) go in `tests/conftest.py` with `matplotlib.use('Agg')` set globally

### Claude's Discretion
- Exact fixture names and structure in conftest.py
- Order of test functions within the file
- How to construct valid/invalid inputs for each specific checker type

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `template/tests/conftest.py`: Exists but serves a different purpose (dynamic student/instructor module loading) — not directly reusable, but shows the project's conftest pattern
- `template/tests/my_fixtures.py`: Module-loading utilities — not relevant to regression tests

### Established Patterns
- `uv run pytest` is the project invocation
- `pyproject.toml` manages deps — add sklearn via `uv add scikit-learn`
- No existing `[tool.pytest.ini_options]` section in pyproject.toml — Wave 0 may add `testpaths = ["tests"]`

### Integration Points
- Import path: `from pytest_generator.assert_utilities import check_answer_float` (via `src/` layout)
- AST-based inventory: `ast.parse` on assert_utilities.py source to enumerate check_* names at test time

</code_context>

<specifics>
## Specific Ideas

- Inventory test should use Python's `ast` module (already validated in research) to enumerate functions — same approach used to confirm the 103-function count
- matplotlib fixtures need `matplotlib.use('Agg')` called before any import of `matplotlib.pyplot` — safest in `conftest.py` at module level or in a session-scoped fixture

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-regression-safety-net*
*Context gathered: 2026-03-10*
