# Phase 4: Signature Standardization - Research

**Researched:** 2026-03-10
**Domain:** Python type annotations, AST-based auditing, function signature conventions
**Confidence:** HIGH

---

## Summary

Phase 4 targets two locked requirements: REQ-CONS-01 (standardize parameter order and naming across all `check_structure_X` / `check_answer_X` functions) and REQ-CONS-02 (confirm every public function carries a `-> tuple[bool, str]` return annotation and actually returns nothing else).

A direct AST audit of `assert_utilities.py` reveals exactly **4 functions missing return annotations** — all scatterplot variants. Every other checker (91 functions) already carries `-> tuple[bool, str]`. Parameter naming is almost entirely consistent already: `student_answer` / `instructor_answer` are universal. The two known deviations are the `dendrogram` pair (`student_dendro` / `instructor_dendro`) and the scatterplot `check_answer` pair (`student_answer` / `instructor_answer` but with extra positional params `options`, `validation_functions` that differ from the dominant pattern). There are also 12 lower-level helper functions (`check_float`, `check_int`, etc.) that intentionally have different signatures and return types — these are private utilities, not public checkers, and must NOT be altered.

The type alias `CheckResult = tuple[bool, str]` is already defined in SECTION 1. Phase 3 explicitly deferred applying it as a replacement annotation throughout the file — Phase 4 is the right time to apply it.

**Primary recommendation:** Fix the 4 missing annotations on the scatterplot functions first. Then apply `CheckResult` as the annotation alias uniformly. Then fix the `dendrogram` parameter naming deviation. Validate all changes with the Phase 1 regression suite.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REQ-CONS-01 | Standardize `check_structure_X` / `check_answer_X` signature conventions across all types | Audit confirms 2 deviations in param naming; full inventory below |
| REQ-CONS-02 | Audit and confirm all public functions return `tuple[bool, str]`; fix any that don't | Audit confirms exactly 4 functions missing return annotations — all scatterplot variants |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `ast` (stdlib) | 3.12 | Parse `assert_utilities.py` into AST to enumerate function signatures without executing code | Zero dependencies; already imported in the file |
| `mypy` | latest in .venv | Static type-check; can verify all public functions return `tuple[bool, str]` | Standard Python type checker |
| `pytest` | in .venv | Run the Phase 1 regression suite after every change | Already configured via `tests/test_assert_utilities.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `typing.TypeAlias` | 3.10+ | `CheckResult: TypeAlias = tuple[bool, str]` is already defined | Apply as replacement for bare `tuple[bool, str]` annotations |

### Installation
No new dependencies. Everything is already installed.

---

## Architecture Patterns

### Recommended Audit Script Pattern

Write a standalone Python script (not a test) that uses `ast` to enumerate all `def` nodes, classify them, and report violations. This is the mechanism for the "audit script" mentioned in the success criteria.

```python
# Source: Python stdlib ast module
import ast

with open("src/pytest_generator/assert_utilities.py") as f:
    tree = ast.parse(f.read())

violations = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name.startswith("check_"):
        # Only public checker functions (not nested helpers like check_grid_status)
        ret = ast.unparse(node.returns) if node.returns else None
        if ret not in ("tuple[bool, str]", "CheckResult"):
            violations.append((node.name, ret, node.lineno))

for name, ret, line in violations:
    print(f"L{line}: {name} -> {ret!r}  (MISSING or WRONG)")
```

### Parameter Naming Convention (from actual file audit)

**Standard for `check_structure_X`:**
```
check_structure_X(student_answer, ...) -> tuple[bool, str]
```
Extra positional params (e.g., `instructor_answer`, `keys`) come AFTER `student_answer`.

**Standard for `check_answer_X`:**
```
check_answer_X(student_answer, instructor_answer, ...) -> tuple[bool, str]
```
Extra params (e.g., `rel_tol`, `keys`, `partial_score_frac`) come after the two mandatory positional params.

### Anti-Patterns to Avoid
- **Modifying the 12 private helper functions** (`check_float`, `check_int`, `check_list_float`, etc.): These are internal primitives with intentionally different return types (`tuple[bool, list[str]]` for some) and different param ordering. They must not be touched.
- **Renaming `check_grid_status`**: This is a nested function inside `check_answer_scatterplot2d` — not a public function. The AST audit must filter nested functions by checking their parent scope.
- **Changing `type_handlers.yaml`-referenced names**: Function names called from YAML must stay callable. This phase only changes annotations and parameter names, not function names.

---

## Findings: Current State of Violations

### REQ-CONS-02 — Missing Return Annotations (exactly 4 functions)

| Function | Line | Current annotation | Fix |
|----------|------|--------------------|-----|
| `check_structure_scatterplot2d` | 2447 | MISSING | `-> CheckResult` |
| `check_structure_scatterplot3d` | 2478 | MISSING | `-> CheckResult` |
| `check_answer_scatterplot2d` | 4644 | MISSING | `-> CheckResult` |
| `check_answer_scatterplot3d` | 4726 | MISSING | `-> CheckResult` |

These functions DO return `(status, "\n".join(msg_list))` or call `return_value(...)` which returns `tuple[bool, str]` — so the return type is correct at runtime. Only the annotation is missing.

### REQ-CONS-01 — Parameter Naming Deviations

| Function | Deviation | Standard name | Fix |
|----------|-----------|---------------|-----|
| `check_structure_dendrogram` | param `student_dendro` | `student_answer` | rename |
| `check_answer_dendrogram` | params `student_dendro`, `instructor_dendro` | `student_answer`, `instructor_answer` | rename |

**All other check_structure_X and check_answer_X functions already use `student_answer` / `instructor_answer`.**

### REQ-CONS-01 — CheckResult Alias Application

Phase 3 defined `CheckResult: TypeAlias = tuple[bool, str]` in SECTION 1 but explicitly deferred applying it as a replacement annotation. Phase 4 applies `CheckResult` as the annotation on all 91 public checker functions instead of the bare `tuple[bool, str]` spelling. This is cosmetic standardization only — no runtime behavior changes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Enumerate all checker signatures | Manual grepping | `ast.walk` script | AST is exact; grep misses multiline defs |
| Verify return types statically | Custom checker | `mypy --check-untyped-defs` | mypy follows call chains across helpers |
| Run regression gate | Manual test runs | `pytest tests/test_assert_utilities.py -x` | Existing suite covers all 91 checkers |

---

## Common Pitfalls

### Pitfall 1: Nested Functions Matching `check_*`
**What goes wrong:** `check_grid_status` is defined inside `check_answer_scatterplot2d`. A naive `ast.walk` will find it and flag it as a public function missing a return annotation.
**Why it happens:** `ast.walk` does not track depth or parent scope.
**How to avoid:** Walk `tree.body` (module-level only) for top-level `FunctionDef` nodes. Or track parent: nested `FunctionDef` nodes appear inside other `FunctionDef` bodies.
**Warning signs:** Audit script reports `check_grid_status` as a violation.

### Pitfall 2: Renaming Parameters Breaks Internal References
**What goes wrong:** In `check_answer_dendrogram`, the body references `student_dendro` and `instructor_dendro` by name. Renaming the parameters without updating the body references produces a `NameError` at runtime.
**How to avoid:** When renaming params, search and replace ALL occurrences of the old name within the function body. Use the AST or a careful string replace bounded by the function's line range.
**Warning signs:** Regression tests for `dendrogram` fail with `NameError`.

### Pitfall 3: Treating `check_str` / `check_dict_str_str` as Public Checkers
**What goes wrong:** The 12 `check_*` helpers that start with `check_` but are not `check_structure_` or `check_answer_` have different return types (`tuple[bool, list[str]]` for some). They MUST NOT receive `-> CheckResult` annotations — that would be incorrect.
**How to avoid:** Scope the audit and fixes strictly to functions matching `check_structure_` or `check_answer_` prefixes.

### Pitfall 4: `check_key_structure` Returns `bool`, Not `tuple[bool, str]`
**What goes wrong:** `check_key_structure` (line 600) returns plain `bool`. It IS annotated `-> bool` already. Do not add `-> CheckResult` to it.
**Warning signs:** Over-broad regex on function names.

---

## Code Examples

### Correct Annotation Pattern (already seen in file)
```python
# Source: assert_utilities.py line 672
def check_structure_bool(student_answer) -> tuple[bool, str]:
    ...
```

After Phase 4 (with CheckResult alias applied):
```python
def check_structure_bool(student_answer) -> CheckResult:
    ...
```

### Scatterplot Fix (adding missing annotation)
```python
# Before (line 2447):
def check_structure_scatterplot2d(student_answer):

# After:
def check_structure_scatterplot2d(student_answer) -> CheckResult:
```

### Dendrogram Fix (parameter rename)
```python
# Before (line 2792):
def check_answer_dendrogram(student_dendro, instructor_dendro, rel_tol) -> tuple[bool, str]:

# After:
def check_answer_dendrogram(student_answer, instructor_answer, rel_tol) -> CheckResult:
    # All body references to student_dendro -> student_answer
    # All body references to instructor_dendro -> instructor_answer
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Bare `tuple[bool, str]` annotation | `CheckResult` alias | Phase 3 defined alias; Phase 4 applies it | Consistent, grep-able return type |
| No annotation on scatterplot functions | `-> CheckResult` | Phase 4 | Passes mypy, passes audit script |

---

## Open Questions

1. **Should `check_answer_scatterplot2d` and `check_answer_scatterplot3d` keep the `options` / `validation_functions` parameters?**
   - What we know: These two functions take `(student_answer, instructor_answer, options, validation_functions)`. All other `check_answer_X` functions take `(student_answer, instructor_answer, ...)` where extras are named typed values, not a generic `options` dict.
   - What's unclear: Whether standardizing parameter names here means collapsing `options` into named params, or just accepting the pattern for complex plot types.
   - Recommendation: Do NOT change the parameter structure in Phase 4 — that is deeper refactoring. Phase 4 only adds the annotation. Parameter structure rationalization belongs to Phase 6 (detailed refactoring).

2. **Apply `CheckResult` alias or keep bare `tuple[bool, str]`?**
   - What we know: Phase 3 decision explicitly deferred this to Phase 4. `CheckResult` is in SECTION 1.
   - Recommendation: Apply `CheckResult` uniformly — this is the point of the alias. The planner should include a task for global replacement.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (installed in .venv) |
| Config file | `pyproject.toml` or `pytest.ini` (check project root) |
| Quick run command | `uv run pytest tests/test_assert_utilities.py -x -q` |
| Full suite command | `uv run pytest tests/ -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-CONS-01 | `check_structure_dendrogram` uses `student_answer` param name | unit | `uv run pytest tests/test_assert_utilities.py -k dendrogram -x` | Yes (existing test) |
| REQ-CONS-01 | `check_answer_dendrogram` uses `student_answer`/`instructor_answer` param names | unit | `uv run pytest tests/test_assert_utilities.py -k dendrogram -x` | Yes |
| REQ-CONS-02 | Audit script reports zero violations | smoke | `python scripts/audit_signatures.py` (Wave 0 gap) | No — Wave 0 |
| REQ-CONS-02 | All regression tests still pass after annotation changes | regression | `uv run pytest tests/test_assert_utilities.py -x -q` | Yes |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_assert_utilities.py -x -q`
- **Per wave merge:** `uv run pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `scripts/audit_signatures.py` — covers REQ-CONS-02 (audit script that reports functions with missing or wrong return annotations)

---

## Sources

### Primary (HIGH confidence)
- Direct AST parse of `src/pytest_generator/assert_utilities.py` — all function signatures, annotations, line numbers enumerated programmatically
- `.planning/REQUIREMENTS.md` — REQ-CONS-01, REQ-CONS-02 definitions
- `.planning/STATE.md` — Phase 3 decision: CheckResult alias deferred to Phase 4

### Secondary (MEDIUM confidence)
- `tests/test_assert_utilities.py` — existing regression coverage for structure/answer functions

---

## Metadata

**Confidence breakdown:**
- Violation inventory: HIGH — derived from AST parse, not grep or memory
- Architecture: HIGH — ast module is stdlib, patterns are well-established
- Pitfalls: HIGH — scatterplot nesting and dendrogram naming deviations are confirmed by actual file inspection

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (file is frozen during this milestone; no external dependencies)
