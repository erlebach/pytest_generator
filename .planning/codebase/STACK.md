# Technology Stack

**Analysis Date:** 2026-03-10

## Languages

**Primary:**
- Python 3.10+ - Full codebase for test generation, assertions, and utilities

## Runtime

**Environment:**
- Python 3.10 or above (specified in `pyproject.toml` requires-python = ">=3.10")

**Package Manager:**
- uv (modern Python package manager)
- Lockfile: `uv.lock` (present)
- Also supports Pipenv (legacy): `Pipfile`

## Frameworks

**Core:**
- pytest 8.3.5 - Unit testing framework for executing generated tests

**Testing & Validation:**
- pytest 8.3.5 - Primary test execution engine

**Build/Dev:**
- hatchling - Build system backend specified in `pyproject.toml`

## Key Dependencies

**Critical:**
- matplotlib 3.10.1 - Visualization library for plot comparison assertions
  - Depends on: contourpy, cycler, fonttools, kiwisolver, pillow, pyparsing, python-dateutil
- numpy 2.2.3 - Numerical computing for array operations and numerical assertions
- PyYAML 6.0.2 - YAML parsing for test configuration files (type_handlers.yaml, generator_config.yaml, etc.)

**Supporting:**
- colorama 0.4.6 - Cross-platform colored terminal output
- pillow 11.1.0 - Image processing (dependency of matplotlib)
- python-dateutil 2.9.0 - Date utilities

**Structuring:**
- packaging 24.2 - Version parsing utilities
- iniconfig 2.0.0 - INI file parsing (pytest dependency)
- pluggy 1.5.0 - Plugin system (pytest dependency)
- tomli 2.2.1 - TOML file parsing

## Configuration

**Environment:**
- No .env files detected - Configuration via YAML files only
- Key configuration files:
  - `generator_config.yaml` - Main test generator settings
  - `type_handlers.yaml` - Type-specific assertion handlers
  - `validations.yaml` - Validation rules and options
  - `homework_template.yaml` - Template for homework test generation
  - `test_generator.yaml` - Test generation configuration
  - `test_generator_with_code.yaml` - Alternative test generation config

**Build:**
- `pyproject.toml` - Project metadata and dependencies
- Build target: `src/pytest_generator` (wheel package)

## Platform Requirements

**Development:**
- macOS (development platform confirmed)
- Linux/Unix compatible (cross-platform Python)

**Production/Deployment:**
- No special deployment infrastructure required
- Runs as standalone Python package
- Can be imported as module via `from pytest_generator import ...`
- Requires `src/pytest_generator` module installed

## Notable Absence

**Not Used:**
- Database integration (no SQL, ORM, or database libraries)
- External API clients (no requests, boto, supabase, etc.)
- Web frameworks (no FastAPI, Flask, Django)
- Authentication libraries (no auth0, JWT libraries)
- Environment variable management (no python-dotenv)
- Logging frameworks (uses Python built-in only)
- Docker (no Docker-specific dependencies)
- Cloud SDKs (AWS, Azure, GCP)

---

*Stack analysis: 2026-03-10*
