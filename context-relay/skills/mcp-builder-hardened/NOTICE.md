# NOTICE

This skill is a modified copy of Anthropic's `mcp-builder` example skill, licensed under the Apache License 2.0 (see `LICENSE.txt`).

Modifications (2026-10-01, for 9x25dillon):

| File | Change |
|---|---|
| `SKILL.md` | Renamed to `mcp-builder-hardened`; description mentions the review; added section 2.4 "Contain What the Model Can Reach", section 3.3 "Blast-Radius Review", and the new references in the index |
| `reference/mcp_best_practices.md` | Added "The Caller Is a Model" under Security Best Practices |
| `reference/blast_radius_review.md` | New: patterns BR001–BR005, untrusted text, egress, per-tool checklist |
| `scripts/blast_radius_check.py` | New: standard-library checker for those patterns in Python and JS/TS servers |
| `scripts/test_blast_radius_check.py` | New: tests for the checker |

The patterns come from a review of the kgirl harness MCP server, fixed in 9x25dillon/kgirl#55. On that server, the checker reports exactly the four confirmed issues at commit 26748c4, and none after the fix.

All other files are unchanged from the original.
