# Phase 2: Duplicate Removal - Research

**Researched:** 2026-03-10
**Domain:** Python source analysis, dead-code removal, file editing
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REQ-INFRA-02 | Identify and resolve all duplicate function definitions | Duplicates fully enumerated below; removal strategy and keeper selection documented |
</phase_requirements>

---

## Summary

`assert_utilities.py` (5660 lines, 119 top-level `def` statements) contains exactly 7 function names that are defined more than once. Python silently uses only the last definition, so every earlier definition of a shadowed name is dead code regardless of its quality.

Two distinct shadowing mechanisms are present. In three cases (kfold, shufflesplit, randomforestclassifier) the earlier definition is already commented out using triple-quote string literals (`"""..."""` or `'''...'''`), making the shadow harmless but messy. In four cases (dict_str_float, dict_str_list_str, check_answer_dict_str_float, check_answer_dict_str_int) two live `def` blocks exist at different line numbers; Python uses the later one.

The correct strategy for each case is: delete the dead/earlier definition and leave the later (live) one intact. No logic changes are needed. The regression test suite from Phase 1 is the safety net.

**Primary recommendation:** Delete each shadowed definition block (and its surrounding comment delimiters where applicable). Never edit the keeper definition. Run Phase 1 tests after each deletion to confirm no regression.

---

## Duplicate Inventory

### Category A — Earlier definition already commented out (triple-quote string)

These are dead code wrapped in triple-quote strings. The live definition follows immediately after. Remove the string literal block entirely (including the opening/closing `"""` or `'''` delimiters).

| Function | Dead block lines | Live block lines | Delimiter |
|----------|-----------------|-----------------|-----------|
| `check_structure_kfold` | 4206–4222 | 4225–4239 | `"""..."""` |
| `check_structure_shufflesplit` | 4335–4351 | 4354–4368 | `"""..."""` (closing `"""` is on the same line as last `return`) |
| `check_answer_randomforestclassifier` | 4469–4509 | 4512–4551 | `'''...'''` |

**Key observation for kfold/shufflesplit:** The two versions are identical in logic. For randomforestclassifier the live version (line 4512) uses `return_value()` and appends to `msg_list` differently from the dead version — the live one is the correct keeper.

### Category B — Two live `def` blocks (no comment wrapper)

Python uses the later definition. The earlier definition is semantically dead from runtime's perspective, but both blocks are syntactically active. Remove the earlier block.

| Function | Earlier (dead) lines | Later (live) lines | Behavioral difference |
|----------|---------------------|-------------------|-----------------------|
| `check_structure_dict_str_list_str` | 826–~1007 | 1009–~1100 | Earlier version uses `clean_str_answer()` and different key-matching algorithm; later version is simpler and uses set-based matching |
| `check_answer_dict_str_int` | 1796–1884 | 2348–2436 | Both versions appear identical in logic; remove earlier |
| `check_structure_dict_str_float` | 4562–4617 (inside `'''`) | 4620–4669 | Earlier wrapped in `'''`, later is live; see Category A note — this is actually Category A |
| `check_answer_dict_str_float` | 4673–4731 (inside `'''`) | 4734–~4830 | Earlier wrapped in `'''`, later is live; Category A |

**Corrected category assignments:**

After reading the file:

- `check_structure_dict_str_float`: dead block at 4561–4618 is inside `'''`, live block at 4620 — Category A
- `check_answer_dict_str_float`: dead block at 4672–4731 is inside `'''`, live block at 4734 — Category A
- `check_structure_dict_str_list_str`: two live `def` blocks at 826 and 1009 — Category B (genuine live shadow)
- `check_answer_dict_str_int`: two live `def` blocks at 1796 and 2348 — Category B (genuine live shadow)

### Final Consolidated Duplicate Table

| Function | Dead block start | Dead block end | Mechanism | Keeper line |
|----------|-----------------|----------------|-----------|------------|
| `check_structure_kfold` | 4206 | 4222 | `"""` string | 4225 |
| `check_structure_shufflesplit` | 4335 | 4351 | `"""` string | 4354 |
| `check_answer_randomforestclassifier` | 4469 | 4509 | `'''` string | 4512 |
| `check_structure_dict_str_float` | 4561 | 4618 | `'''` string | 4620 |
| `check_answer_dict_str_float` | 4672 | 4731 | `'''` string | 4734 |
| `check_structure_dict_str_list_str` | 826 | ~1007 | live shadow | 1009 |
| `check_answer_dict_str_int` | 1796 | ~1884 | live shadow | 2348 |

---

## Architecture Patterns

### Recommended Removal Approach

**One function at a time.** Remove one dead block, run the full Phase 1 test suite, commit. Repeat. Never batch multiple removals before testing.

**Use line-number-anchored reads** to verify block boundaries before each deletion. The exact end line of a block is defined by the line immediately before the next top-level `def` or section comment.

**Do not touch the keeper.** The only edit is to remove the dead block (and its surrounding delimiters if Category A).

### Determining Block End Line

For Category A (string-wrapped), the block ends at the closing `"""` or `'''`. For Category B (live shadow), the block ends at the blank line or comment immediately before the next `def` statement.

For `check_structure_dict_str_list_str` (line 826), the dead block runs until approximately line 1007 where a `# ---` comment separator appears before the live definition at 1009.

### Test Execution Pattern

```bash
cd /Users/erlebach/src/2026/pytest_generator
python -m pytest tests/test_assert_utilities.py -x -q
```

Run this after each deletion. All tests must pass before proceeding to the next duplicate.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Finding duplicate defs | Custom AST walker | `grep -n "^def "` piped to `sort | uniq -d` |
| Verifying deletion correctness | Manual inspection only | Phase 1 pytest suite |
| Boundary detection | Guessing | Read tool with offset+limit to inspect actual lines |

---

## Common Pitfalls

### Pitfall 1: Deleting the wrong copy

**What goes wrong:** Removing the live (later) definition instead of the dead (earlier) one.

**Why it happens:** Line numbers look similar; confusion about which copy Python actually uses.

**How to avoid:** Python always uses the LAST definition in a file. Always delete the EARLIER block. Verify with `grep -n "^def FUNCNAME"` before and after deletion.

### Pitfall 2: Incomplete string-literal removal

**What goes wrong:** Removing the function body inside a `'''` block but leaving the opening or closing delimiter as a bare string expression.

**Why it happens:** The delimiters are on their own lines and look like comments.

**How to avoid:** Remove the opening delimiter line, all body lines, and the closing delimiter line together as one contiguous block.

### Pitfall 3: Off-by-one on block boundaries

**What goes wrong:** Accidentally including the first line of the keeper (the `def` statement at the live block start) in the deletion range.

**How to avoid:** Always read 3–5 lines past the expected end of the dead block before deleting. The keeper's `def` line must remain.

### Pitfall 4: Breaking type_handlers.yaml lookups

**What goes wrong:** If a function name were renamed or removed without updating type_handlers.yaml, runtime handler lookups fail silently or raise KeyError.

**Why it won't happen here:** We are only deleting dead code. The keeper functions retain their exact names and signatures. No YAML changes needed.

### Pitfall 5: check_structure_dict_str_list_str behavioral difference

**What goes wrong:** The earlier live definition (line 826) uses `clean_str_answer()` for key normalization — a behavior the later definition (line 1009) does not replicate. If tests depended on the earlier behavior, they would fail after removal.

**How to avoid:** The Phase 1 tests were written against the actual runtime behavior (which is the later definition at line 1009, since Python uses the last one). Removing the earlier definition changes nothing at runtime. Tests should stay green.

---

## Code Examples

### Verify duplicates before starting

```bash
grep -n "^def " /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py \
  | awk -F'[( ]' '{print $1}' | sed 's/:[[:space:]]*def//' | sort | uniq -d
```

### Confirm zero duplicates after completion

```bash
grep -n "^def " /Users/erlebach/src/2026/pytest_generator/src/pytest_generator/assert_utilities.py \
  | awk -F'[(]' '{print $1}' | sort | uniq -d
# Expected output: (empty)
```

### Run regression suite (quick check)

```bash
cd /Users/erlebach/src/2026/pytest_generator && python -m pytest tests/test_assert_utilities.py -x -q
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Commenting out old code with triple-quoted strings | Deleting the block entirely | Cleaner file, no ambiguity about which version is active |

---

## Open Questions

1. **check_structure_dict_str_list_str: which version was intended?**
   - What we know: The later version (line 1009) is the live one and is what Phase 1 tests exercise.
   - What's unclear: Whether the earlier version's `clean_str_answer()` normalization was intentional behavior that should be preserved elsewhere.
   - Recommendation: Remove earlier version as planned. If the normalization feature is needed, it can be added to the keeper in a later phase. Do not block Phase 2 on this question.

2. **check_answer_dict_str_int: identical or slightly different?**
   - What we know: Both versions appear to have the same body. The later one is the keeper.
   - What's unclear: Whether there is any subtle difference (e.g., a debug print statement).
   - Recommendation: Read both blocks side-by-side during execution to confirm before deleting.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (detected via pyproject.toml) |
| Config file | pyproject.toml |
| Quick run command | `python -m pytest tests/test_assert_utilities.py -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-INFRA-02 | Zero duplicate `def check_` names after removal | smoke (grep) | `grep -n "^def " src/pytest_generator/assert_utilities.py \| awk -F'[(]' '{print $1}' \| sort \| uniq -d \| wc -l` (expect 0) | N/A (command) |
| REQ-INFRA-02 | All Phase 1 regression tests still pass | regression | `python -m pytest tests/test_assert_utilities.py -x -q` | ✅ exists |
| REQ-INFRA-02 | type_handlers.yaml references intact | smoke (import) | `python -c "import src.pytest_generator.assert_utilities"` | N/A (command) |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/test_assert_utilities.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

None — existing test infrastructure from Phase 1 covers all phase requirements.

---

## Sources

### Primary (HIGH confidence)

- Direct file read of `src/pytest_generator/assert_utilities.py` (lines 826, 1009, 1796, 2348, 4207–4734) — duplicate locations verified by grep and manual read
- Direct file read of `type_handlers.yaml` — confirmed all 7 affected function names are referenced; all references point to names that survive in the keeper definitions

### Secondary (MEDIUM confidence)

- Python language specification: last `def` in module scope wins (well-known, no source needed)

---

## Metadata

**Confidence breakdown:**
- Duplicate inventory: HIGH — verified by grep and manual line reads
- Keeper selection: HIGH — Python semantics are unambiguous (last def wins)
- type_handlers.yaml safety: HIGH — verified no references to function names being removed, only to names being kept
- Behavioral difference in dict_str_list_str: MEDIUM — observed structural difference; runtime impact is zero (later version already active)

**Research date:** 2026-03-10
**Valid until:** Until assert_utilities.py is modified (stable file, 60-day shelf life otherwise)
