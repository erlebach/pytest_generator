---
phase: 4
slug: signature-standardization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (installed in .venv) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/test_assert_utilities.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_assert_utilities.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-W0-01 | 01 | 0 | REQ-CONS-02 | smoke | `python scripts/audit_signatures.py` | ❌ W0 | ⬜ pending |
| 4-01-01 | 01 | 1 | REQ-CONS-01 | unit | `uv run pytest tests/test_assert_utilities.py -k dendrogram -x` | ✅ | ⬜ pending |
| 4-01-02 | 01 | 1 | REQ-CONS-02 | unit | `uv run pytest tests/test_assert_utilities.py -x -q` | ✅ | ⬜ pending |
| 4-01-03 | 01 | 1 | REQ-CONS-02 | smoke | `python scripts/audit_signatures.py` | ❌ W0 | ⬜ pending |
| 4-01-04 | 01 | 2 | REQ-CONS-01, REQ-CONS-02 | regression | `uv run pytest tests/ -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/audit_signatures.py` — AST-based audit script that enumerates module-level `check_structure_*` and `check_answer_*` functions, reports missing `-> tuple[bool, str]` annotations and non-standard parameter names. Covers REQ-CONS-02.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] No `@pytest.mark.skip` / `xfail` / `skipIf` without a corresponding entry in Manual-Only Verifications
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
