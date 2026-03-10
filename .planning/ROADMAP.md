# Roadmap: assert_utilities Refactoring

## Overview

This milestone transforms `assert_utilities.py` from a 5660-line, partially-duplicated monolith into
a clean, consistently-structured grading library. The work proceeds in strict dependency order:
regression tests first (safety net), then duplication removal, then structural cleanup, then
signature standardization, and finally a documented template so adding a new answer type is
mechanical. The external interface of `test_generator.py` does not change.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Regression Safety Net** - Write comprehensive tests covering all existing checker functions before touching any code
- [ ] **Phase 2: Duplicate Removal** - Identify and eliminate all duplicate function definitions
- [ ] **Phase 3: File Cleanup** - Extract utilities, add type aliases, reorganize file into logical sections
- [ ] **Phase 4: Signature Standardization** - Standardize all checker signatures and confirm uniform return types
- [ ] **Phase 5: Extensibility Template** - Document and demonstrate the canonical pattern for adding a new answer type

## Phase Details

### Phase 1: Regression Safety Net
**Goal**: Every existing checker function is covered by tests that will catch any regression introduced during refactoring
**Depends on**: Nothing (first phase)
**Requirements**: REQ-INFRA-01
**Success Criteria** (what must be TRUE):
  1. A test file exists that imports and exercises every `check_structure_X` and `check_answer_X` function currently in assert_utilities.py
  2. All tests pass against the unmodified assert_utilities.py
  3. The test suite can be run with a single pytest invocation and produces a clear pass/fail result
  4. Each test validates both the `bool` and `str` components of the returned tuple
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Test infrastructure (sklearn dep, pyproject.toml config, conftest fixtures)
- [ ] 01-02-PLAN.md — check_structure_* regression tests (47 functions, 94 test cases)
- [ ] 01-03-PLAN.md — check_answer_* regression tests + AST inventory completeness test

### Phase 2: Duplicate Removal
**Goal**: Each function in assert_utilities.py is defined exactly once; no shadowed definitions remain
**Depends on**: Phase 1
**Requirements**: REQ-INFRA-02
**Success Criteria** (what must be TRUE):
  1. A search for duplicate `def check_` names returns zero results
  2. All Phase 1 regression tests continue to pass after duplicate removal
  3. type_handlers.yaml references remain intact (no broken handler lookups)
**Plans**: TBD

### Phase 3: File Cleanup
**Goal**: assert_utilities.py has a clear, navigable structure with named sections, type aliases, and shared primitives in a dedicated area
**Depends on**: Phase 2
**Requirements**: REQ-CLEAN-01, REQ-CLEAN-02, REQ-CLEAN-03
**Success Criteria** (what must be TRUE):
  1. The file opens with a section containing only type aliases (`CheckResult`, `PartialScoreDict`, etc.)
  2. A clearly delimited utilities/primitives section contains all shared comparison helpers (float tolerance, partial scoring, return formatting)
  3. Structure check functions (`check_structure_X`) are grouped together in their own section
  4. Answer check functions (`check_answer_X`) are grouped together in their own section
  5. All Phase 1 regression tests continue to pass
**Plans**: TBD

### Phase 4: Signature Standardization
**Goal**: Every public checker function follows the same signature convention and every public function returns `tuple[bool, str]`
**Depends on**: Phase 3
**Requirements**: REQ-CONS-01, REQ-CONS-02
**Success Criteria** (what must be TRUE):
  1. All `check_structure_X` functions share a consistent parameter order and naming scheme
  2. All `check_answer_X` functions share a consistent parameter order and naming scheme
  3. Running a type-check or audit script confirms every public function has a `tuple[bool, str]` return annotation and returns nothing else
  4. All Phase 1 regression tests continue to pass
**Plans**: TBD

### Phase 5: Extensibility Template
**Goal**: Adding a new answer type requires only filling in a documented template; one existing type serves as the reference implementation
**Depends on**: Phase 4
**Requirements**: REQ-EXT-01, REQ-EXT-02
**Success Criteria** (what must be TRUE):
  1. A template file (or clearly marked inline template block) exists that shows the exact structure required for a new type's `check_structure_X` and `check_answer_X` functions
  2. One existing type (e.g., `float`) is annotated or documented as the canonical reference that matches the template
  3. A developer can read the template and the reference implementation and understand exactly where to add code for a new type without reading the rest of the file
  4. All Phase 1 regression tests continue to pass
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Regression Safety Net | 2/3 | In Progress|  |
| 2. Duplicate Removal | 0/? | Not started | - |
| 3. File Cleanup | 0/? | Not started | - |
| 4. Signature Standardization | 0/? | Not started | - |
| 5. Extensibility Template | 0/? | Not started | - |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REQ-INFRA-01 | Phase 1 | Pending |
| REQ-INFRA-02 | Phase 2 | Pending |
| REQ-CLEAN-01 | Phase 3 | Pending |
| REQ-CLEAN-02 | Phase 3 | Pending |
| REQ-CLEAN-03 | Phase 3 | Pending |
| REQ-CONS-01 | Phase 4 | Pending |
| REQ-CONS-02 | Phase 4 | Pending |
| REQ-EXT-01 | Phase 5 | Pending |
| REQ-EXT-02 | Phase 5 | Pending |
