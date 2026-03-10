# External Integrations

**Analysis Date:** 2026-03-10

## APIs & External Services

**Not Applicable** - This is a standalone test generation framework with no external API integrations.

## Data Storage

**Databases:**
- Not used - Project does not use any database systems

**File Storage:**
- Local filesystem only
- Input: YAML configuration files in project root
  - `generator_config.yaml` - Master configuration
  - `type_handlers.yaml` - Type handlers
  - `homework_template.yaml` - Homework templates
  - `test_generator.yaml`, `test_generator_with_code.yaml` - Test configs
  - `validations.yaml` - Validation rules
  - `all_questions.yaml` - Question definitions

- Output: Generated Python test files to local filesystem
  - Student code folder: configured via `student_folder_name` (default: `student_code_with_answers/`)
  - Instructor code folder: configured via `instructor_folder_name` (default: `instructor_code_with_answers/`)

**Caching:**
- None - No caching layer detected

## Authentication & Identity

**Auth Provider:**
- Not applicable - Standalone command-line tool, no user authentication required

## Monitoring & Observability

**Error Tracking:**
- None - No error tracking service integration

**Logs:**
- Standard output only via `print()` statements
- No structured logging framework
- Debug messages logged to stdout during test generation

## CI/CD & Deployment

**Hosting:**
- Standalone package - Runs locally or in educational grading environments
- Can be used in GitHub Classrooms or similar platforms

**CI Pipeline:**
- Not configured in codebase
- Deployment: Manual via `python setup.py` or `pip install -e .`

## Environment Configuration

**Required env vars:**
- None - No environment variables required

**Secrets location:**
- Not applicable - No secrets required

## File System Expectations

**Input Directories (must exist):**
- Folder configured via `student_folder_name` setting in `generator_config.yaml`
- Folder configured via `instructor_folder_name` setting in `generator_config.yaml`

**Output Locations:**
- Generated test files written to same directory structure
- Test files generated for each homework/question defined in YAML

## Configuration Integration Points

**YAML-Driven Configuration:**
1. `generator_config.yaml` defines:
   - `all_tests` section: Student/instructor folder names, partial scoring settings
   - `test_structure` section: Test organization and formatting
   - `test_answers` section: Tolerance settings (rel_tol, abs_tol), answer validation rules
   - `types` section: Type-specific handlers and validation options
   - `option_defaults` section: Default values for all validation options

2. `type_handlers.yaml` defines:
   - Custom type handlers for Python types (dict[str,float], list[ndarray], etc.)
   - Handler implementations for 20+ custom types

3. `validations.yaml` defines:
   - Validation rules for different assertion types

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Flow

**Input Pipeline:**
1. User creates YAML files defining questions and answers
2. Test generator loads YAML configuration
3. Generator creates Python test files with pytest format
4. Test files import student and instructor code from specified folders
5. pytest runs generated tests

**Output:**
- Standalone Python test files suitable for grading with pytest

---

*Integration audit: 2026-03-10*
