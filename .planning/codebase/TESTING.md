# Testing

## Framework

**pytest 8.3.5**

## Key Pattern

Tests are **dynamically generated** from YAML configuration files, not written by hand. The codebase IS the test generator — it generates test files for student homework grading.

## Generated Test Structure

```python
@max_score(10.0)
@hide_errors('')
def test_answers_question1_part_a():
    student_answer = student_function()
    instructor_answer = decode_data(encoded_answer)
    is_success, message = check_TYPE(student_answer, instructor_answer)
    assert is_success, message
```

## Custom Decorators

Defined in `pytest_utils/decorators.py`:

| Decorator | Purpose |
|-----------|---------|
| `@max_score(points)` | Attach point value to test |
| `@visibility(level)` | Control visibility in grader output |
| `@hide_errors(msg)` | Suppress error details from students |
| `@partial_score(fn)` | Enable partial credit scoring |

## Plugin

`pytest_utils/pytest_plugin.py` — pytest plugin that:
- Hooks into test collection and reporting
- Captures `max_score`, `visibility`, `hide_errors` metadata per test
- Aggregates scores across all tests
- Outputs JSON results for grading system

## Configuration

Test generation driven by:
- `generator_config.yaml` — which questions/types to generate
- `type_handlers.yaml` — which `check_*` function handles each type
- `validations.yaml` — additional validation rules

## Test Coverage of the Generator Itself

**No unit tests found** for the generator codebase itself. The repo is a test generator — it is not itself tested with automated tests.

**Coverage gaps:**
- `assert_utilities.py` validation logic — untested
- Code generation string building — untested
- YAML parsing and config loading — untested
- Integration flows (YAML → generated file → pytest run) — untested

---
*Mapped: 2026-03-10*
