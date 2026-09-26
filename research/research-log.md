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

## 2026-09-25 — E1 run (exploratory, LoCoMo)

- Protocol locked in b0379e2 before running. Result: ±1 neighbour window +0.08–0.11 complete
  coverage at equal budget; beats 2× budget top-k. Session-diversity cap refuted. Multi-hop
  (cross-session aggregation) complete coverage 0.12–0.29 under every unit: the open failure.

## 2026-09-25 — audit results, E2, E3

- Audit workflow (6 agents): every published number recomputes, but all offline LoCoMo scripts
  use a permuted category map (1=single-hop, 3=multi-hop, 4=world-knowledge); the "MiniLM hurts
  multi-hop −0.091" and "PPR multi-hop Δ0" findings are really open-domain (cat 3, n≈42).
  Real multi-hop improved with MiniLM (+0.067). Degree sham ≈ random. 1,977 includes 446
  adversarial questions. No production context assembly/answer path; /nodes neighbour edges
  start from sentence chunks (i+1 unreachable for 90.4% of turns); no LLM judge; adversarial
  scored against trap answer; no authorized live budget.
- E2 (diagnostic): dataset observation keys +0.05–0.07 multi-hop complete coverage; reveals a
  depth/breadth budget conflict.
- E3: neighbour score propagation λ=0.7 (dev-tuned) — held-out narrow fail on production keys at
  2k (tie), pass with fact keys; wins at other budgets. Neighbourhood = mechanism; propagation =
  refinement.
