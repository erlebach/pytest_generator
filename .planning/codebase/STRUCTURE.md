# Structure

## Directory Layout

```
pytest_generator/
├── src/
│   └── pytest_generator/          # Core library (11 Python modules)
│       ├── assert_utilities.py    # Type-specific answer validators (5660 lines)
│       ├── test_generator.py      # Main generation orchestrator
│       ├── generate_test_functions.py  # Test function string builder
│       ├── type_handlers.py       # Answer type dispatch
│       └── ...                    # Other core modules
├── pytest_utils/                  # Pytest integration
│   ├── decorators.py              # @max_score, @visibility, @hide_errors, @partial_score
│   └── pytest_plugin.py          # Score capture plugin
├── template/                      # User setup scaffold
├── generator_config.yaml          # Top-level generation settings
├── type_handlers.yaml             # Type → handler mappings
├── validations.yaml               # Validation rules per type
├── generate_all_tests.x           # Entry point: generate all
├── generate_homework_tests.x      # Entry point: generate one homework
└── run_test_generator.x           # Entry point: full pipeline
```

## Key Locations

| Location | Purpose |
|----------|---------|
| `src/pytest_generator/assert_utilities.py` | Core validation logic — most complex file |
| `src/pytest_generator/test_generator.py` | Generation orchestrator — start here |
| `pytest_utils/decorators.py` | Scoring decorators for generated tests |
| `pytest_utils/pytest_plugin.py` | Score aggregation plugin |
| `generator_config.yaml` | Primary configuration |
| `template/` | Scaffold for new homework setups |

## Naming Conventions

- **Modules**: snake_case.py (`test_generator.py`, `assert_utilities.py`)
- **Functions**: snake_case with semantic prefixes:
  - `generate_` — code generation functions
  - `check_` — type-specific validators in assert_utilities
  - `decode_` — answer decoding utilities
- **Entry scripts**: `.x` extension (executable scripts)
- **Config files**: `_config.yaml` or descriptive names

---
*Mapped: 2026-03-10*
