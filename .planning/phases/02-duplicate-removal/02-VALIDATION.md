---
phase: 2
slug: duplicate-removal
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | REQ-INFRA-02 | regression | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 2-01-02 | 01 | 1 | REQ-INFRA-02 | regression | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 2-01-03 | 01 | 1 | REQ-INFRA-02 | regression | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 2-01-04 | 01 | 1 | REQ-INFRA-02 | regression | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 2-01-05 | 01 | 1 | REQ-INFRA-02 | regression | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 2-01-06 | 01 | 1 | REQ-INFRA-02 | regression | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 2-01-07 | 01 | 1 | REQ-INFRA-02 | regression | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 2-01-08 | 01 | 1 | REQ-INFRA-02 | audit | `grep -c "^def check_" assert_utilities.py` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* Phase 1 already established the full regression test suite; no new test files needed for Phase 2.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No duplicate `def check_` names in assert_utilities.py | REQ-INFRA-02 | Static structural check | Run `grep "^def check_" assert_utilities.py \| sort \| uniq -d` — expect zero output |
| type_handlers.yaml references still valid | REQ-INFRA-02 | YAML/Python cross-file check | Verify all function names in type_handlers.yaml still exist in assert_utilities.py |

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
