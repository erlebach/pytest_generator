# Conventions

## Code Style

- **Language**: Python (PEP 8 informal — no explicit linter config)
- **Indentation**: 4 spaces
- **File naming**: All snake_case (`test_generator.py`, `assert_utilities.py`)

## Naming

| Pattern | Usage | Example |
|---------|-------|---------|
| `generate_*` | Code generation functions | `generate_test_functions()` |
| `check_*` | Type-specific validators | `check_list_float()`, `check_dict_str_int()` |
| `decode_*` | Answer decoding utilities | `decode_data()` |
| `is_*` | Boolean predicates | `is_success` |

## Type Hints

Used throughout codebase. Return type pattern for validators:

```python
def check_list_float(student_answer, instructor_answer, ...) -> tuple[bool, str]:
    # Returns (is_success, explanation_message)
```

## Error Handling

- Try/except for expected failures (student code can raise anything)
- Assertions (`assert`) for test failures within generated tests
- **Known issue**: Some places use `raise "string"` (syntax error in Python 3)

## Logging

- `print()` statements only — no logging framework
- No structured logging

## Comments

- Block separators: `# ------` or `# ===`
- TODO/FIXME markers present
- Inline comments explain non-obvious logic

## Patterns

- **Tuple returns**: `(bool, message)` for all validators — `(True, "correct")` or `(False, "expected X, got Y")`
- **String-based code gen**: Tests are built as strings then written to `.py` files
- **YAML-driven dispatch**: Type names in YAML map to handler function names

---
*Mapped: 2026-03-10*
