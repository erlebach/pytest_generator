---
phase: 6
slug: detailed-refactoring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest -q tests/` |
| **Full suite command** | `uv run pytest -q tests/` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -q tests/`
- **After every plan wave:** Run `uv run pytest -q tests/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 1 | REQ-AUDIT-01 | grep | `grep -n "print(" src/pytest_generator/assert_utilities.py \| wc -l` (should be 0) | ✅ | ⬜ pending |
| 6-01-02 | 01 | 1 | REQ-AUDIT-01 | unit | `uv run pytest -q tests/` | ✅ | ⬜ pending |
| 6-01-03 | 01 | 1 | REQ-AUDIT-01 | grep | `grep -n "# !" src/pytest_generator/assert_utilities.py \| wc -l` (should be 0) | ✅ | ⬜ pending |
| 6-01-04 | 01 | 1 | REQ-AUDIT-01 | unit | `uv run pytest -q tests/ -k check_structure_dict_str_set` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No stale TODO/FIXME remain unresolved | REQ-AUDIT-01 | Requires judgment call on each TODO | Review each remaining TODO comment and confirm it's either resolved or annotated with rationale |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] No `@pytest.mark.skip` / `xfail` / `skipIf` without a corresponding entry in Manual-Only Verifications
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
