# Adding a New Answer Type

This document describes the four files you must touch to add a new answer type to the
pytest-generator framework.  Use `float` as the reference implementation throughout.

---

## Overview

Each type requires two functions in `assert_utilities.py` and one entry each in
`type_handlers.yaml` and `types_list.py`.  The test generator reads
`type_handlers.yaml` at runtime and emits the correct assertion calls; it gates on
`types_list.py` to decide whether a type name found in an exercise YAML file is
recognised.

---

## File 1 — `src/pytest_generator/assert_utilities.py`

### 1a. `check_structure_<type>`

Validates that `student_answer` has the right Python type or shape.
The instructor answer is **not** passed — structure checking never needs a reference value.

**Reference implementation** (`check_structure_float`, line 1862):

```python
def check_structure_float(student_answer: float) -> CheckResult:
    """Check the structure of the student answer.

    Args:
        student_answer (float): The student answer

    Returns:
        tuple[bool, str]: The status and message

    """
    if isinstance(student_answer, float | np.floating):
        status = True
        msg_list = ["Answer is of type float as expected."]
    else:
        status = False
        msg_list = [
            "Answer should be of type float. It is of type ",
            f"{type(student_answer).__name__!r}",
        ]
    return status, "\n".join(msg_list)
```

Rules:
- Return type is always `CheckResult` (`tuple[bool, str]`).
- On success: `status = True`, one positive confirmation message.
- On failure: `status = False`, message says what was expected and what was received.
- For simple scalar types an `isinstance` check is sufficient.
- For compound types (e.g. `dict[str, float]`) iterate the keys/values and check each
  element; see `check_structure_dict_str_float` (line 1267) as an example.

**Placement:** insert alphabetically inside the `# SECTION 4: STRUCTURE CHECKS` block.

---

### 1b. `check_answer_<type>`

Validates the *value* of the answer, assuming the structure check already passed.
Receives both `student_answer` and `instructor_answer`, plus any type-specific
tolerance or option parameters.

**Reference implementation** (`check_answer_float`, line 3795):

```python
def check_answer_float(
    student_answer: float,
    instructor_answer: float,
    rel_tol: float,
    abs_tol: float,
) -> CheckResult:
    """Check answer correctness. Assume the structure is correct.

    Args:
        student_answer (float): The student answer
        instructor_answer (float): The instructor answer
        rel_tol (float): The relative tolerance
        abs_tol (float): The absolute tolerance

    Returns:
        tuple[bool, str]: The status and message

    """
    status, msg = check_float(
        instructor_answer,
        student_answer,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    if not msg:
        msg = "Float value is as expected."

    return return_value(status, [msg], student_answer, instructor_answer)
```

Rules:
- Always return via `return_value(status, msg_list, student_answer, instructor_answer)`.
  Never construct the final tuple directly — `return_value` appends the student and
  instructor values to the message string, which aids debugging.
- If a primitive comparison helper already exists in Section 3 (e.g. `check_float`,
  `check_int`), call it rather than duplicating the comparison logic.
- Provide a positive fallback message when `msg_list` is empty (the `if not msg:` guard
  in the float example).

**Placement:** insert alphabetically in the `check_answer_*` block (starts around
line 2649).

---

## File 2 — `type_handlers.yaml`

Add one YAML entry.  The key is exactly the type name that will appear in exercise YAML
files.

**Minimal entry (simple built-in type):**

```yaml
float:
    import: ""
    assert_answer: "assert_utilities.check_answer_float(student_answer, instructor_answer, rel_tol, abs_tol)"
    assert_structure: "assert_utilities.check_structure_float(student_answer)"
```

**Entry with a third-party import:**

```yaml
SVC:
    import: "from sklearn.svm import SVC"
    assert_answer: "assert_utilities.check_answer_svc(student_answer, instructor_answer)"
    assert_structure: "assert_utilities.check_structure_svc(student_answer)"
    struct_msg: Answer should be an instance of SVC.
```

Fields:
- `import` — injected verbatim as an import line at the top of the generated pytest
  file.  Leave as `""` for built-in Python types.
- `assert_structure` — the exact function call that will appear in the generated
  structure-check test.  List every parameter the function signature requires.
- `assert_answer` — the exact function call for the answer-check test.  Include all
  parameters beyond `student_answer` and `instructor_answer` (e.g. `rel_tol`,
  `partial_score_frac_l`).
- `struct_msg` *(optional)* — human-readable description used in generated output.

---

## File 3 — `src/pytest_generator/types_list.py`

Add the type name string to `types_list`.  This is the gate used in `test_generator.py`
to decide whether a type found in an exercise YAML file is handled by the framework.

```python
types_list = [
    ...
    "float",   # ← add here, keeping rough alphabetical order
    ...
]
```

The string must match the key used in `type_handlers.yaml` exactly.

---

## Checklist

| Step | File | What to add |
|------|------|-------------|
| 1a | `src/pytest_generator/assert_utilities.py` | `check_structure_<type>` in Section 4 |
| 1b | `src/pytest_generator/assert_utilities.py` | `check_answer_<type>` in the answer-checker block |
| 2 | `type_handlers.yaml` | YAML entry with `import`, `assert_structure`, `assert_answer` |
| 3 | `src/pytest_generator/types_list.py` | Type name string |

No other files need to change.
