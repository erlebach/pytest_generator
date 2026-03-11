# Phase 3: File Cleanup - Research

**Researched:** 2026-03-10
**Domain:** Python source reorganization, type aliases, in-place file restructuring
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REQ-CLEAN-01 | Extract shared primitives into a clearly delineated utilities section | Utility functions inventoried below; all live in lines 43–592 but are interspersed with no clear boundary |
| REQ-CLEAN-02 | Add type aliases (`PartialScoreDict`, `CheckResult`, etc.) for clarity | No type aliases exist today; `tuple[bool, str]` and `dict[str, float | int]` repeated ~80+ times |
| REQ-CLEAN-03 | Reorganize file into logical sections: utilities → primitives → structure checks → answer checks | Current ordering is chaotic; check_answer and check_structure functions interleaved randomly |
</phase_requirements>

---

## Summary

`assert_utilities.py` is currently 5290 lines (after Phase 2 duplicate removal). It has no formal section structure: utility/primitive helper functions (lines 43–592) are intermixed with no boundary markers, and `check_structure_X` / `check_answer_X` functions appear in no consistent order throughout lines 587–5290. The two types are often paired by topic (e.g., `check_answer_float` at 587 followed by `check_structure_float` at 621) but sometimes reversed or separated by hundreds of lines.

The goal of Phase 3 is purely structural: reorder existing function bodies into four clearly-delimited sections, add a type-alias block at the top, and ensure the regression suite remains green throughout. No function logic is modified.

The central risk is **reference ordering**: if a function is moved above another function it calls, Python will raise a `NameError` at module load time. Every move must be preceded by a forward-dependency check. The safe strategy is a two-pass approach — first establish the target order on paper with dependency analysis, then execute the moves in a single carefully-sequenced edit session.

**Primary recommendation:** Build a dependency graph of the utility functions first. Move functions in dependency order (callees before callers). Run `python -m pytest tests/test_assert_utilities.py -x -q` after each section is finalized, not after each individual move.

---

## Current File Anatomy

### Module-level constants and singletons (lines 1–42)

| Item | Line | Type |
|------|------|------|
| Module docstring | 1 | docstring |
| Imports | 14–33 | import |
| `FLOAT_TOL = 1.0e-5` | 40 | constant |

### Utility / Primitive helpers (lines 43–592) — NO SECTION HEADER TODAY

These functions are helpers called by `check_answer_X` / `check_structure_X`. They must stay above all callers.

| Function | Line | Role |
|----------|------|------|
| `init_partial_score_dict` | 43 | partial-score dict factory |
| `check_missing_keys` | 57 | dict-key validation helper |
| `check_float` | 83 | float comparison primitive |
| `check_int` | 125 | int comparison primitive |
| `check_list_float` | 149 | list-of-float comparison |
| `check_list_int` | 187 | list-of-int comparison |
| `check_set_int` | 222 | set-of-int comparison |
| `check_str` | 269 | string comparison |
| `check_list_str` | 320 | list-of-str comparison |
| `check_dict_str_str` | 350 | dict-str-str comparison |
| `update_score` | 382 | partial score updater |
| `check_dict_str_float` | 395 | dict-str-float comparison |
| `clean_str_answer` | 442 | string normalization |
| `load_yaml_file` | 460 | YAML loader |
| `extract_config_dict` | 475 | config dict builder |
| `config_dict` (singleton) | 492 | module-level config |
| `fmt_ifstr` | 496 | format helper |
| `return_value` | 512 | return-format helper |
| `are_sets_equal` | 560 | set equality helper |

Additional helpers scattered among checker functions:

| Function | Line | Role |
|----------|------|------|
| `check_key_structure` | 1151 | key-structure validator |
| `convert_to_set_of_sets` | 3167 | type converter |
| `is_sequence_but_not_str` | 3230 | type predicate |

### check_answer_X / check_structure_X functions (lines 587–5290)

Functions are grouped loosely by answer type but the pair order is inconsistent:
- Some types: `check_answer_X` appears before `check_structure_X`
- Some types: `check_structure_X` appears before `check_answer_X`
- `check_key_structure`, `convert_to_set_of_sets`, `is_sequence_but_not_str` are helpers buried inside this zone

---

## Standard Stack

No external libraries needed for this phase. All work is:

- Python `typing` module — `TypeAlias` (Python 3.10+), `type` statement (Python 3.12+)
- Standard file editing with the `Edit` / `Write` tools
- `pytest` for regression validation

### Python Version

Per the file header: "Use Python 3.10 or above." The project uses `pyproject.toml`. Use `TypeAlias` from `typing` (Python 3.10 compatible) rather than the `type X = ...` syntax (Python 3.12+).

```python
# Python 3.10-compatible type alias syntax
from typing import TypeAlias

CheckResult: TypeAlias = tuple[bool, str]
PartialScoreDict: TypeAlias = dict[str, float | int]
```

---

## Architecture Patterns

### Target Section Layout

```
assert_utilities.py
├── Module docstring
├── Imports
│
├── # ======================================================================
├── # SECTION 1: TYPE ALIASES
├── # ======================================================================
│   CheckResult, PartialScoreDict, (others as needed)
│
├── # ======================================================================
├── # SECTION 2: CONSTANTS
├── # ======================================================================
│   FLOAT_TOL
│
├── # ======================================================================
├── # SECTION 3: UTILITY PRIMITIVES
├── # ======================================================================
│   init_partial_score_dict, check_missing_keys, check_float, check_int,
│   check_list_float, check_list_int, check_set_int, check_str, check_list_str,
│   check_dict_str_str, update_score, check_dict_str_float, clean_str_answer,
│   load_yaml_file, extract_config_dict, config_dict, fmt_ifstr, return_value,
│   are_sets_equal, check_key_structure, convert_to_set_of_sets, is_sequence_but_not_str
│
├── # ======================================================================
├── # SECTION 4: STRUCTURE CHECKS  (check_structure_X)
├── # ======================================================================
│   All check_structure_X functions, grouped alphabetically by type name
│
└── # ======================================================================
    # SECTION 5: ANSWER CHECKS  (check_answer_X)
    # ======================================================================
    All check_answer_X functions, grouped alphabetically by type name
```

### Why Alphabetical Within Sections

- Deterministic — no judgment calls during execution
- Easy to find a function by type name
- Consistent with how type_handlers.yaml is likely organized

### Section Delimiter Pattern

Use the existing `# ======================================================================` style already present in the file. Add a label line immediately after:

```python
# ======================================================================
# SECTION 3: UTILITY PRIMITIVES
# ======================================================================
```

### Execution Strategy: Move Scattered Helpers First

The three helpers buried in the checker zone (`check_key_structure` at 1151, `convert_to_set_of_sets` at 3167, `is_sequence_but_not_str` at 3230) must be moved to Section 3 before any check functions are reordered. They are called by checker functions, so they must precede all callers.

### Anti-Patterns to Avoid

- **Move callee after caller:** Causes `NameError` at import. Always verify call graph before moving.
- **Edit logic while reordering:** Zero logic changes in Phase 3. Move blocks verbatim.
- **Rename section delimiters mid-file:** Choose the delimiter format once and use it uniformly.
- **Batch all moves in one edit:** Too risky. Finalize one section, test, then the next.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finding all callers of a helper | Manual grep | `grep -n "func_name" assert_utilities.py` | Fast, accurate |
| Verifying import still works | Manual inspection | `python -c "import src.pytest_generator.assert_utilities"` | Catches NameError immediately |
| Type alias syntax | Custom comment convention | `typing.TypeAlias` | PEP 613 — stdlib, IDE-understood |

---

## Type Aliases to Introduce

Based on repeating patterns in the file:

| Alias | Expansion | Occurrences (approx) |
|-------|-----------|---------------------|
| `CheckResult` | `tuple[bool, str]` | 80+ function return types |
| `PartialScoreDict` | `dict[str, float \| int]` | 15+ parameter types |

These two aliases cover the vast majority of type annotation repetition. Additional aliases (e.g., `MsgList = list[str]`) are optional and at discretion — do not add them unless they appear 10+ times.

---

## Common Pitfalls

### Pitfall 1: Forward reference / NameError at import

**What goes wrong:** A function is moved to a section that appears before the function it calls.

**Why it happens:** Python executes module top-level statements in order. If `check_answer_X` is placed before `return_value`, Python raises `NameError: name 'return_value' is not defined`.

**How to avoid:** Before moving any function, run:
```bash
grep -n "return_value\|check_float\|check_key_structure" \
  /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py
```
Confirm the callee will land above all callers after the move.

**Warning signs:** Import fails immediately with `NameError`.

### Pitfall 2: config_dict singleton position

**What goes wrong:** `extract_config_dict()` is called at module load on line 492 (`config_dict = extract_config_dict()`). If this line is moved before `load_yaml_file` or `extract_config_dict`, it raises `NameError`.

**How to avoid:** Keep `config_dict` assignment immediately after `extract_config_dict`. They travel as a unit.

### Pitfall 3: Type alias import placement

**What goes wrong:** `TypeAlias` is imported but `from typing import TypeAlias` is not added to the imports block.

**How to avoid:** Add `TypeAlias` to the existing `from typing import Any, cast` line. Confirm no duplicate import.

### Pitfall 4: Accidentally modifying function bodies

**What goes wrong:** When cutting/pasting large blocks, a line gets dropped, duplicated, or indented incorrectly.

**How to avoid:** After moving each section, run `python -m pytest tests/test_assert_utilities.py -x -q` immediately. Any body corruption will surface as a test failure or syntax error.

### Pitfall 5: Section header comment style inconsistency

**What goes wrong:** Some section headers use `# ===` (80 chars), some use `# ---`, some use `# ...`. The file has 3 existing delimiter styles, leading to visual noise.

**How to avoid:** Use only `# ======================================================================` (80 chars) for top-level section headers. Leave existing `# ---` and `# ...` sub-delimiters within functions untouched.

---

## Code Examples

### Adding type aliases (Python 3.10+)

```python
# Source: https://peps.python.org/pep-0613/ (PEP 613, Python 3.10)
from typing import TypeAlias

CheckResult: TypeAlias = tuple[bool, str]
PartialScoreDict: TypeAlias = dict[str, float | int]
```

### Verifying no NameError after move

```bash
# Run after each section finalization
cd /Users/erlebach/src/2026/pytest_generator
python -c "import src.pytest_generator.assert_utilities; print('import OK')"
```

### Regression suite

```bash
cd /Users/erlebach/src/2026/pytest_generator
python -m pytest tests/test_assert_utilities.py -x -q
```

### Finding all callers of a helper before moving it

```bash
grep -n "check_key_structure\b" \
  /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py
grep -n "convert_to_set_of_sets\b" \
  /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py
grep -n "is_sequence_but_not_str\b" \
  /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py
```

### Counting type annotation occurrences

```bash
grep -c "tuple\[bool, str\]" \
  /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py
grep -c "dict\[str, float | int\]" \
  /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Ad-hoc `# ---` dividers between functions | Named, consistent `# === SECTION N: NAME ===` blocks | Navigable by search and eye |
| Repeated `tuple[bool, str]` inline | `CheckResult` type alias | Single definition, IDE hover shows meaning |
| Helpers scattered at point of first use | Helpers consolidated in Utilities section | Predictable location for all callers |

---

## Open Questions

1. **Pair order within sections: check_structure before or after check_answer?**
   - What we know: The file is inconsistent today. REQ-CLEAN-03 says structure checks in one section, answer checks in another — so they will no longer be paired.
   - What's unclear: Within the `STRUCTURE CHECKS` section, should functions be ordered alphabetically by type name? Or by some other principle?
   - Recommendation: Alphabetical by type name (the suffix after `check_structure_`) within each section. Deterministic, easy to verify.

2. **`check_key_structure` naming — is it a primitive or a structure-check?**
   - What we know: Its name starts with `check_` but it does not match the `check_structure_X` / `check_answer_X` pattern. It is a helper called by other checker functions.
   - What's unclear: Whether it belongs in Section 3 (Primitives) or Section 4 (Structure Checks).
   - Recommendation: Place in Section 3 (Primitives) because it does not correspond to a single answer type and is an internal helper, not a public entry point.

3. **`config_dict` singleton — does it belong in constants or utilities?**
   - What we know: It is assigned at module load and used by `extract_config_dict` callers.
   - Recommendation: Keep it at the end of Section 3 (Utilities), immediately after `extract_config_dict`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (pyproject.toml config) |
| Config file | pyproject.toml |
| Quick run command | `python -m pytest tests/test_assert_utilities.py -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-CLEAN-01 | Utilities section exists with clear delimiter | smoke (grep) | `grep "SECTION.*UTIL" src/pytest_generator/assert_utilities.py` | ❌ Wave 0 (grep, not a test file) |
| REQ-CLEAN-02 | CheckResult alias defined | smoke (grep) | `grep "CheckResult" src/pytest_generator/assert_utilities.py` | ❌ Wave 0 |
| REQ-CLEAN-03 | Sections in correct order (aliases → constants → utils → structure → answer) | smoke (grep+awk) | verify section header line numbers are ascending | ❌ Wave 0 |
| REQ-CLEAN-01/02/03 | All Phase 1 regression tests pass after reorganization | regression | `python -m pytest tests/test_assert_utilities.py -x -q` | ✅ exists |
| REQ-CLEAN-01/02/03 | Module imports without error | smoke | `python -c "import src.pytest_generator.assert_utilities"` | N/A (command) |

### Sampling Rate

- **Per task commit:** `python -c "import src.pytest_generator.assert_utilities" && python -m pytest tests/test_assert_utilities.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_file_structure.py` — optional structural smoke tests (grep-based checks for section headers and type alias presence). Can be skipped if regression suite alone is deemed sufficient — the structure is verifiable by inspection.

None for regression testing — `tests/test_assert_utilities.py` already covers behavioral correctness.

---

## Sources

### Primary (HIGH confidence)

- Direct file read of `src/pytest_generator/assert_utilities.py` — function inventory, line numbers, section delimiter styles verified by Read tool and grep
- `.planning/REQUIREMENTS.md` — REQ-CLEAN-01/02/03 definitions read directly
- `.planning/STATE.md` — project constraints (single file, functional only, no module splitting) read directly
- `.planning/phases/02-duplicate-removal/02-RESEARCH.md` — Phase 2 outcomes and post-cleanup line counts

### Secondary (MEDIUM confidence)

- PEP 613 (TypeAlias) — Python 3.10 compatible type alias syntax; well-known stdlib feature

---

## Metadata

**Confidence breakdown:**
- File inventory (utility functions, their lines): HIGH — verified by direct grep and Read
- Type alias candidates: HIGH — verified by grep count
- Dependency ordering risk: HIGH — verified by grep of caller sites
- Section layout recommendation: HIGH — directly derived from REQUIREMENTS.md success criteria
- Python 3.10 TypeAlias syntax: HIGH — PEP 613 stdlib feature

**Research date:** 2026-03-10
**Valid until:** Until assert_utilities.py is modified (30-day shelf life)
