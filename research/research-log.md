# Research log

Chronological decisions. Newest last.

## 2026-09-25 — bootstrap

- Read existing paper (output/pdf/hybridmind-retrieval-research-20260822.pdf), companion
  report, claim ledger, status, protocol, benchmark report, design-space program §7–10.
- Constraint from user: laptop is weak; no heavy local tests.
- LoCoMo structure probe (tmp/research/locality_probe2.py): category 1 (282 q) is
  cross-session aggregation (3% same-session, 38% span >=3 sessions); category 4 single-hop
  multi-evidence is local (83% within ±2 turns). Mean conversation ~13.4k words, so
  full-context is a cheap strong control on LoCoMo.
- Launched audit workflow (pipeline trace + claim recomputation) and SOTA research workflow.
