---
phase: 03
slug: file-cleanup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | REQ-CLEAN-02 | regression | `uv run pytest tests/ -x -q` | ✅ | ⬜ pending |
| 03-01-02 | 01 | 1 | REQ-CLEAN-01 | regression | `uv run pytest tests/ -x -q` | ✅ | ⬜ pending |
| 03-01-03 | 01 | 1 | REQ-CLEAN-03 | regression | `uv run pytest tests/ -x -q` | ✅ | ⬜ pending |
| 03-01-04 | 01 | 1 | REQ-CLEAN-03 | regression | `uv run pytest tests/ -x -q` | ✅ | ⬜ pending |
| 03-01-05 | 01 | 1 | REQ-CLEAN-01 | regression | `uv run pytest tests/ -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* All 183 Phase 1 regression tests already cover behavioral correctness. No new test stubs needed — file restructuring is behavioral-parity refactoring.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| File opens with type aliases section | REQ-CLEAN-02 | Structural layout (not behavioral) | `head -60 src/pytest_generator/assert_utilities.py` — verify `# === Type Aliases` section appears first |
| Named sections present and in order | REQ-CLEAN-01 | Section ordering not tested by pytest | `grep "^# ===" src/pytest_generator/assert_utilities.py` — verify 5 sections in order |

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
