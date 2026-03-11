# REQUIREMENTS

> All Active requirements are hypotheses until shipped and validated.

## Validated

*(None yet — ship to validate)*

## Active

### INFRA — Foundation

- **REQ-INFRA-01**: Comprehensive regression tests for all existing checker functions (written BEFORE any refactoring begins)
- **REQ-INFRA-02**: Identify and resolve all duplicate function definitions

### CLEANUP — File Health

- **REQ-CLEAN-01**: Extract shared primitives into a clearly delineated utilities section
- **REQ-CLEAN-02**: Add type aliases (`PartialScoreDict`, `CheckResult`, etc.) for clarity
- **REQ-CLEAN-03**: Reorganize file into logical sections: utilities → primitives → structure checks → answer checks

### CONSISTENCY — Standardization

- **REQ-CONS-01**: Standardize `check_structure_X` / `check_answer_X` signature conventions across all types
- **REQ-CONS-02**: Audit and confirm all public functions return `tuple[bool, str]`; fix any that don't

### EXTENSIBILITY — Future-Proofing

- **REQ-EXT-01**: Document template/recipe for adding a new answer type
- **REQ-EXT-02**: Apply template retroactively to one existing type as the canonical reference implementation

## Out of Scope

- Refactoring `test_generator.py` — future phase
- Splitting `assert_utilities.py` into multiple files — deferred
- Adding new answer types — no new types in this milestone
- Converting to class-based design — explicitly excluded (keep functional)
- Changing YAML format (`type_handlers.yaml`, `generator_config.yaml`) — frozen

## Constraints

- Functional only — no classes or OOP patterns
- Single file — no module splitting
- All function names/signatures referenced by `type_handlers.yaml` must remain callable (or be updated consistently in that file)
- `test_generator.py` public interface must not break
- `tuple[bool, str]` return format maintained for all public `check_structure_X` / `check_answer_X` functions

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REQ-INFRA-01 | Phase 1 | Complete |
| REQ-INFRA-02 | Phase 2 | Complete |
| REQ-CLEAN-01 | Phase 3 | Complete |
| REQ-CLEAN-02 | Phase 3 | Complete |
| REQ-CLEAN-03 | Phase 3 | Pending |
| REQ-CONS-01 | Phase 4 | Pending |
| REQ-CONS-02 | Phase 4 | Pending |
| REQ-EXT-01 | Phase 5 | Pending |
| REQ-EXT-02 | Phase 5 | Pending |
