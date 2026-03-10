# Concerns

## Critical Issues

### Syntax Errors
- `raise "string"` pattern found — invalid in Python 3 (must be `raise ValueError("string")`)
- These will cause runtime errors when that code path is hit

### Unsafe eval() Usage
- `eval()` used with student-submitted code
- **Security risk**: student code could execute arbitrary commands
- Needs sandboxing or restricted execution environment

### Unguarded Dictionary Access
- Direct `dict[key]` access without `.get()` or key existence checks
- Will raise `KeyError` on unexpected input types

## Technical Debt

### Monolithic assert_utilities.py
- **5660 lines** in a single file
- Contains all type-specific validators (`check_list_float`, `check_dict_str_int`, etc.)
- Difficult to navigate, test, or extend
- Should be split into per-type modules

### Global Variables
- Some global state used across module boundaries
- Makes testing and reasoning about state difficult

### Print-Based Logging
- Pervasive `print()` for debugging/status
- No log levels, no ability to silence or redirect
- Should use `logging` module

### String-Based Code Generation
- Tests built as string concatenation → written to `.py` files
- Fragile: easy to generate syntactically invalid Python
- No AST-based generation or validation before writing

## Known Bugs

### Dangling Brace Formatting
- Formatting error producing dangling `}` in some generated output

### Type Name Inconsistencies
- Same concept referred to by different names in different YAML files
- Causes lookup failures for some type handlers

### Missing Field Handling
- Some YAML fields assumed present without checking
- Causes `KeyError` on malformed configs

## Performance

- Large datasets (many questions × many students) run slowly
- Repeated type handler lookups on every validation call (no caching)
- No parallelization of test generation across questions

## Security

- `eval()` with student code — highest risk item
- YAML loaded with `yaml.load()` (not `yaml.safe_load()`) in some places — allows arbitrary Python object deserialization
- Encoded answers in YAML provide some obfuscation but not real security

## Missing Features / Design Gaps

- No type extension mechanism — adding new answer types requires editing the monolith
- No partial credit framework beyond the decorator stub
- No pedagogical feedback — errors say "wrong" but not "here's a hint"
- No integration test suite for the generator itself

---
*Mapped: 2026-03-10*
