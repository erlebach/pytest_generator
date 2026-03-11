# pytest-generator: assert_utilities Refactoring

## What This Is

A refactoring of `assert_utilities.py` — the core grading library in the pytest-generator
system — to eliminate duplication, establish a consistent pattern across all answer-type
checkers, and make adding new types a mechanical, obvious process. The external interface
of `test_generator.py` is frozen; only `assert_utilities.py` and its internal boundary
with `test_generator.py` are in scope.

## Core Value

Adding a new answer type (e.g., `list[ndarray]`, `dict[str, DataFrame]`) must require
only filling in a well-defined template — no hunting for where to add code, no copy-paste
of boilerplate, no guessing at conventions.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Establish a consistent internal structure/answer check pattern across all types
- [ ] Eliminate duplicated comparison logic (float tolerance, partial scoring, return formatting)
- [ ] Standardize function signatures and return formats within assert_utilities.py
- [ ] Reorganize file so related functions are grouped (utilities → structure checks → answer checks)
- [ ] Comprehensive regression test suite verifying all existing type checkers are unbroken
- [ ] Document the pattern so adding a new type is self-evident (template or guide)
- [ ] Preserve the public API surface callable from test_generator.py (no breaking changes to names/signatures used there)
- [ ] tuple[bool, str] return format maintained for all public check functions

### Out of Scope

- Refactoring test_generator.py — future phase
- Splitting assert_utilities.py into multiple files — deferred; revisit when auto-generation is considered
- Adding new answer types — no new types in this phase
- Converting to class-based design — explicitly excluded (keep functional)
- Changing YAML format (type_handlers.yaml, generator_config.yaml) — frozen

## Context

- `assert_utilities.py` is ~2000+ lines with ~30+ functions following two patterns per type:
  `check_structure_<type>` and `check_answer_<type>`, plus shared utilities
- `test_generator.py` calls these via `eval()` on strings from `type_handlers.yaml`
- The internal boundary (which function names are referenced in type_handlers.yaml) can change
  as long as test_generator.py's own public interface is untouched
- Future: new types expected; auto-generation of checker code is a longer-term possibility
- Partial scoring (`ps_dict`) is a cross-cutting concern used by many checkers

## Constraints

- **Architecture**: Functional only — no classes or OOP patterns
- **File structure**: Single file for now — no module splitting
- **Compatibility**: All function names/signatures referenced by type_handlers.yaml must remain
  callable (or be updated consistently in that file)
- **Tests**: Must have comprehensive tests before and after refactoring to verify no regression

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep single file | Splitting by type is non-obvious for composite types (list[float] vs dict[str, list[float]]) | — Pending |
| Functional approach | User preference; class hierarchies add complexity without clear benefit here | — Pending |
| Freeze test_generator.py | External callers must not break; that file is a future refactoring phase | — Pending |

---
*Last updated: 2026-03-10 after initialization*
