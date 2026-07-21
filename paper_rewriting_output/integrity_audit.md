# Integrity Audit

- Output directory: `paper_rewriting_output`
- Total findings: 5
- LaTeX gate: READY

> This report teaches, not just checks. Each finding includes a root cause, a concrete fix, what happens downstream if unfixed, and why this pattern matters.

## Summary

| Dimension | Status | Findings |
|---|---|---|
| Artifact Chain | CLEAN | 1 |
| Reasoning Depth | WARNINGS | 2 |
| Evidence Chain | CLEAN | 1 |
| Integrity Patterns | WARNINGS | 1 |
| Process-Language Leak | CLEAN | 0 |

## Artifact Chain

**ART-000** ✅ All 11 required artifacts present

---

## Reasoning Depth

### ⚠️ RSN-003 — WARNING

**What was found:** First rationale row is shallow (135 chars). The whole-work framework justification should be at least 300 chars.

**Root cause:** The writing step spent insufficient effort on the controlling structure justification. This row should explain *why* the chosen framework fits the confirmed motivation.

**Fix:** Expand the first data row to include: (a) why this structure was chosen over alternatives, (b) how SOTA examples informed it, (c) how it serves the confirmed motivation, (d) which user evidence anchors it, and (e) how the final text will be checked against it.

**Downstream impact:** A weak first row means the entire paper structure is unjustified. The structured review will flag this as a fundamental weakness.

**Why this matters:** The whole-work framework row is the most important row in the matrix. It's not a summary — it's a design justification.

### ⚠️ RSN-005 — WARNING

**What was found:** 1 shallow rows: [15]

**Root cause:** Minor gaps in reasoning depth — likely rows that were left as placeholders.

**Fix:** Review each shallow row and add at least motivation link + planned change.

**Downstream impact:** These rows will produce weaker-than-necessary writing units.

**Why this matters:** A rationale row doesn't need to be long, but it needs to be specific.

---

## Evidence Chain

**EVD-000** ✅ Claims are adequately linked to evidence

---

## Integrity Patterns

### ⚠️ INT-001 — WARNING

**What was found:** No manuscript text to scan for integrity patterns

**Root cause:** The manuscript hasn't been written yet or final_paper/ is empty.

**Fix:** Proceed with writing, then re-run this audit.

**Downstream impact:** Cannot verify integrity of unwritten text.

---
