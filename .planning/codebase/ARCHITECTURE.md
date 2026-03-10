# Architecture

## Pattern

Multi-layer code generation pipeline: Configuration → Generation → Assertion → Validation

YAML-driven question and answer type system. The system reads configuration files to understand what types of student answers are expected, then generates pytest test files that validate those answers.

## Layers

### 1. Configuration Layer
- YAML files define question types, validation rules, and answer encodings
- `generator_config.yaml` — top-level generation settings
- `type_handlers.yaml` — maps answer types to handler functions
- `validations.yaml` — validation rules per type
- Homework-specific YAML files with encoded instructor answers

### 2. Generation Layer
- `src/pytest_generator/test_generator.py` — orchestrates test file generation
- `src/pytest_generator/generate_test_functions.py` — builds individual test function strings
- Templates in `template/` — user setup scaffold
- Entry points: `generate_all_tests.x`, `generate_homework_tests.x`, `run_test_generator.x`

### 3. Assertion Layer
- `src/pytest_generator/assert_utilities.py` — 5660-line monolith, core validation logic
- Type-specific `check_TYPE()` functions for each answer type
- Returns `(bool, message)` tuples: `(is_success, explanation_str)`

### 4. Validation Layer
- Pytest integration via decorators in `pytest_utils/`
- `pytest_plugin.py` captures test metadata and scores
- `@max_score()`, `@visibility()`, `@hide_errors()`, `@partial_score()` decorators

## Data Flow

```
YAML config files
    ↓
test_generator.py (reads config, encodes answers)
    ↓
generate_test_functions.py (builds test function strings)
    ↓
Generated .py test file (dynamic string-based code generation)
    ↓
pytest execution
    ↓
assert_utilities.py (validates student answers vs instructor answers)
    ↓
pytest_plugin.py (captures scores and metadata)
    ↓
JSON results output
```

## Entry Points

- `generate_all_tests.x` — generate tests for all homeworks
- `generate_homework_tests.x` — generate tests for a specific homework
- `run_test_generator.x` — run the full generation + test pipeline

## Key Abstractions

- **Type handlers** — functions that know how to validate a specific answer type
- **Encoded answers** — instructor answers stored encoded in YAML to prevent cheating
- **Score decorators** — pytest decorators that attach scoring metadata to test functions
- **Plugin** — pytest plugin that aggregates scores across all tests

---
*Mapped: 2026-03-10*
