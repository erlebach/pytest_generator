---
phase: 1
slug: regression-safety-net
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.5 |
| **Config file** | pyproject.toml (Wave 0 installs pytest config section) |
| **Quick run command** | `uv run pytest tests/test_assert_utilities.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_assert_utilities.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | REQ-INFRA-01 | infra | `uv run pytest --collect-only` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | REQ-INFRA-01 | unit | `uv run pytest tests/test_assert_utilities.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — make tests a package
- [ ] `tests/conftest.py` — shared fixtures (matplotlib Agg backend, numpy arrays, sklearn objects)
- [ ] `tests/test_assert_utilities.py` — test stubs for all 91 unique checker functions

*If scikit-learn absent from pyproject.toml: Wave 0 must add it or use `pytest.mark.skip` with Manual-Only entry.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| scikit-learn dependency resolution | REQ-INFRA-01 | Requires decision: add dep or skip ML tests | Run `uv run pytest tests/ -v` and review any sklearn-related skip/fail |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] No `@pytest.mark.skip` / `xfail` / `skipIf` without a corresponding entry in Manual-Only Verifications
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
