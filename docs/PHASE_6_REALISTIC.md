# PHASE 6 — REALISTIC: Loss-Ledger-Driven Optimization

**Status:** Supersedes `ROADMAP_TO_SOTA.md` §Phase 6 in its entirety.
**Prime directive:** Every intervention in this phase must be justified by a *measured* loss, executed with a *deterministic* procedure, and accepted or rejected by a *pre-registered* statistical gate. No component is built because it is fashionable. No component is built because it appeared in a paper about a system 1000× larger than this one.

---

## 0. Why the old Phase 6 is void

The previous Phase 6 (IVF-PQ, SimCSE contrastive fine-tuning, GraphSAGE+BPR, Fusion MLP) was written for a system that does not exist. The system that *does* exist has these measured properties:

| Property | Measured value | Source |
|---|---|---|
| Corpus size | ~7,510 nodes (multi-domain eval) | `docs/MULTI_DOMAIN_EVAL.md` |
| Retrieval Hit@10 (LoCoMo) | 60% | `docs/LOCOMO_BENCHMARK_REPORT.md` |
| End-to-end answer accuracy (LoCoMo) | 48% | same |
| Single-hop accuracy | 0% — attributed to **LLM answer-extraction failure**, not retrieval | same |
| Graph signal behavior | Binary: Recall@3 = 1.0 with edges, 0.0 without | `scripts/targeted_graph_benchmark.py` results |
| p95 vector search latency | < 50ms up to ~8–10k nodes (HNSW) | `STRESS_TEST_REPORT.md` |
| Auto-edge provenance | Cosine threshold on the *current* embedder + entity co-occurrence | `engine/edge_inference.py` |

Four disqualifications follow directly:

1. **IVF-PQ (old 6a):** Product quantization trades recall for memory and throughput at ≥10⁶ vectors. At 7.5×10³ vectors, exact flat search over the entire corpus costs single-digit milliseconds. IVF-PQ here strictly *reduces* recall for zero latency benefit. **Disqualified by scale.**
2. **Contrastive fine-tuning on graph edges (old 6b):** Positive pairs sampled from edges whose creation criterion was cosine similarity under the current embedder constitutes training the embedder on its own output distribution. The gradient signal contains no information the model did not already have; the failure mode is anisotropic collapse (embedding space contracts around existing clusters, hurting out-of-cluster recall — the exact regime where you currently lose). Additionally, fine-tuning the query-side model while stored vectors remain frozen breaks the shared-space assumption; fixing it costs a full corpus re-embed through an 8B model per experiment iteration. **Disqualified by circularity and by data volume** (7.5k nodes yields at most ~10⁴ pairs; contrastive methods need 10⁵–10⁶ to beat their base model out-of-domain).
3. **GraphSAGE reranker (old 6c):** The graph benchmark proves the graph channel's failure mode is *missing edges*, not *misranked edges*. A GNN is a function of `(x, edge_index)`; when `edge_index` is empty for the relevant subgraph, no parameterization of the function recovers signal. **Disqualified by input sparsity.** (Coverage must be fixed first — see 6.2 — and only then does learned graph ranking become well-posed.)
4. **Fusion MLP as specified (old 6d):** Direction is right, capacity and protocol are wrong. A 9-input MLP with ~3k parameters trained on a few thousand examples, with no train/test split policy and labels drawn from the same benchmark used for final reporting, will "gain" points via leakage and lose them on any held-out set. **Salvaged in reduced form as 6.3.**

The old phase asked: *"what can we train on an A100?"*
The correct question is: *"where, exactly, are the points dying?"*

---

## 1. The Loss Ledger

Accounting identity for end-to-end accuracy:

```
Answer accuracy = P(gold evidence retrieved) × P(correct answer | evidence retrieved)
```

Current: `0.48 ≈ 0.60 × 0.80` (approximately; the ledger's first task is to make this decomposition exact per question type).

Three loss pools, ordered by size:

| Pool | Size | Nature |
|---|---|---|
| **L1 — Reading loss** | ~12+ pts (60% Hit@10 vs 48% E2E; single-hop 0% from extraction failure) | The answering LLM fails to extract/format the answer even when gold evidence is in context |
| **L2 — Retrieval loss** | 40 pts of Hit@10 headroom | Gold evidence never enters top-10: ingestion granularity, multi-hop query mismatch, missing graph edges |
| **L3 — Fusion loss** | Unknown, bounded small | Gold evidence in candidate pool but ranked below cutoff by static RRF weights |

**Rule: no work on pool Lₙ₊₁ before pool Lₙ has a quantified residual.** L3 (the only pool the old Phase 6 addressed) is the smallest and is gated behind the other two.

---

## 2. Phase 6.0 — Measurement Integrity (prerequisite, ~half a day, $0 GPU)

Nothing below is meaningful until the harness itself is trustworthy.

**6.0.1 — Exact ledger instrumentation.** The eval scripts (`eval_locomo_retrieval.py`, `eval_longmemeval_retrieval.py`, `eval_musique_retrieval.py`) must emit, per question: question type, gold evidence IDs, retrieved IDs at k∈{1,3,5,10,25}, whether gold ⊆ candidate pool pre-rerank, final rank of gold post-rerank, the raw LLM answer string, the judged verdict, and the judge's rationale. This single JSONL artifact is the substrate for every decision in this phase. Without per-question attribution, "accuracy went up 2 points" is indistinguishable from noise.

**6.0.2 — Statistical protocol (pre-registered, immutable for the phase):**
- Every reported metric carries a 95% bootstrap CI (10⁴ resamples over questions).
- Every A/B comparison uses a paired permutation test on per-question outcomes; significance threshold p < 0.05.
- LoCoMo has ~1,540 questions → the minimum detectable effect at reasonable power is roughly ±2.5 pts. **Any claimed gain under 2.5 pts on LoCoMo is noise and must be reported as such.** This single rule retroactively invalidates most of the old Phase 6's "+2–4 pts" line items as unverifiable at this sample size — which is itself a finding.
- **Split policy (anti-leakage, non-negotiable):** anything *trained* or *tuned* (fusion weights, thresholds, prompts) uses LongMemEval-S as development set. LoCoMo is touched exactly twice: once before Phase 6 (baseline) and once after (final). MuSiQue is the multi-hop probe, same two-touch rule. Tuning on the reporting benchmark is the null hypothesis for every fake SOTA claim in this field; the two-touch rule is the cure.
- Fixed seeds for any stochastic component; seeds recorded in the run artifact.

**6.0.3 — Judge validity.** The LLM-as-judge must be spot-validated: hand-grade 50 random judged answers; if judge-human agreement < 90%, fix the judge prompt before anything else, because every downstream number inherits its error.

**Acceptance gate for 6.0:** one command reproduces the baseline ledger end-to-end from a clean checkout, twice, with identical numbers. If two consecutive runs differ, find the nondeterminism (embedding batching, cache state, LLM temperature) and eliminate it. *This is the "100% precision, no second retries" property: it is achieved by determinism in the harness, not by heroism in the model.*

---

## 3. Phase 6.1 — Reading Loss (L1): the 12-point hole (~1–2 days, ~$5 GPU)

The largest measured loss is *after* retrieval. Single-hop at 0% with correct evidence retrieved means the answer stage — prompt, extraction format, or judge — is broken, not the memory system.

**6.1.1 — Failure taxonomy.** From the 6.0.1 ledger, take every question where gold evidence was in the final context but the answer was judged wrong. Classify each (this is a reading task, use the RunPod Qwen3.5 LLM as first-pass classifier, human-verify a 20% sample) into: (a) answer present in context but not extracted; (b) answer extracted but format-mismatched with judge expectations; (c) answer requires composition across ≥2 retrieved chunks and the prompt does not elicit composition; (d) genuine model incapability; (e) gold label is wrong (benchmarks contain label noise — count it, don't tune around it).

**6.1.2 — Interventions, strictly per-category:** (a)→rewrite the answering prompt with explicit evidence citation-then-answer structure; (b)→normalize answer format (dates, entities, numerics) in a deterministic post-processor, never in the LLM; (c)→switch multi-hop answering to iterative evidence-then-conclude prompting; (d)→document as model ceiling; (e)→document as benchmark noise with a per-benchmark noise-rate estimate.

**Why this is first:** it is the cheapest work in the phase (prompt and harness changes, near-zero GPU), it addresses the single largest loss pool, and — critically — every retrieval improvement in 6.2 is *invisible* through a broken reading stage. Improving retrieval under a broken reader is optimizing a signal through a dead channel.

**Acceptance gate:** single-hop accuracy on the dev set (LongMemEval) moves from its current floor to within 5 pts of its own Hit@10 — i.e., the reader stops being the bottleneck for questions retrieval already solves. Expected E2E effect: high single-digit points, the cheapest points in the entire roadmap.

---

## 4. Phase 6.2 — Retrieval Loss (L2): Hit@10 60% → target ≥80% (~3–5 days, ~$10 GPU)

40 points of headroom. The ledger (6.0.1) partitions the misses; attack in measured-size order. The expected partition, given known system behavior:

**6.2.1 — Ingestion granularity.** LoCoMo/LongMemEval evidence is conversational session data. If session-fact extraction (the `/ingest/session-facts` path) drops or merges facts, retrieval cannot recover them — the loss is at write time. Audit: for each missed question, does *any* stored node contain the gold fact? If no → ingestion loss. Fix is in `engine/fact_extractor.py` prompting and chunk policy (fact granularity: one atomic proposition per node; retain speaker + session timestamp as metadata, since LongMemEval's temporal questions are unanswerable without them). This is likely the largest sub-pool, and it is embarrassingly unglamorous, which is precisely why the old Phase 6 skipped it for GNNs.

**6.2.2 — Multi-hop query decomposition.** MuSiQue-style questions embed two hops in one query string; a single dense query lands between the two evidence clusters and hits neither (this is a geometric fact about mean-pooled query embeddings, not a tuning problem). Intervention: the query router (`engine/query_router.py` — currently classified but *not integrated*, per audit) routes multi-hop queries to a decompose-retrieve-then-retrieve loop using the RunPod LLM: extract sub-questions, retrieve per sub-question, take the union as the candidate pool. Edge cases that must be specified in the implementation: decomposition returning 1 sub-question (fall through to standard path); decomposition hallucinating entities not in the query (constrain sub-questions to be substrings-or-paraphrases, reject otherwise); latency budget (two retrieval rounds + one LLM call ≈ +300–800ms — acceptable for eval, gate behind router so single-hop pays nothing).

**6.2.3 — Edge coverage (the real graph fix).** The graph channel is binary on edge existence. Therefore the highest-leverage graph work is *coverage*, not ranking: enable auto-edges by default with the cosine threshold *tuned on the dev set* (sweep threshold ∈ {0.60…0.85} step 0.05; report edge count and Hit@10 at each; pick the knee, not the max — an over-dense graph turns traversal into noise injection). Enable entity co-occurrence edges (spaCy path already implemented, gated off). Acceptance is *channel-level*: fraction of multi-hop dev questions where the gold second-hop node is graph-reachable within 2 hops of a first-hop retrieval hit. That reachability number is the ceiling on any future graph learning — measure it, because it is also the trigger condition for the deferred GNN (§6.5).

**6.2.4 — Candidate pool audit.** Trivial but mandatory: verify the reranker's top-25 pool actually contains gold for questions where fusion "lost" it. If gold is absent at 25 but present at 100, the fix is a pool-size constant, not a learned model. Check this before ever training anything.

**Acceptance gate:** Hit@10 ≥ 75–80% on dev, with per-sub-pool attribution showing which intervention paid. Interventions that do not individually clear the permutation test get *reverted* — accumulated dead config is how systems rot.

---

## 5. Phase 6.3 — Fusion Loss (L3): learned fusion, right-sized (~1 day, ~$2 GPU)

Only now — with the reader fixed and the candidate pools honest — is learned fusion well-posed. And at this data scale the correct model is not an MLP.

**6.3.1 — Model: per-query-type logistic regression.** Features per (query, candidate): the four channel scores (dense, sparse BM25, graph, reranker — each rank-normalized to [0,1] within the candidate pool, *never* raw scores, which are not comparable across queries), plus the router's query-type. Model: one logistic regression per query type (5 types × 5 weights = 25 parameters), or equivalently a single LR with type interactions. Rationale: with ~500 dev questions × ~25 candidates you have ~12k weakly-labeled rows with heavy per-query correlation; effective sample size supports tens of parameters, not thousands. An LR is also *inspectable* — the learned weights per query type are themselves a research finding ("temporal queries load 0.6 on sparse, multi-hop queries load 0.5 on graph"), which an MLP hides.

**6.3.2 — Protocol:** train on LongMemEval dev per the split policy; 5-fold CV *grouped by question* (never split candidates of one question across folds — that is intra-question leakage, the subtlest bug in learning-to-rank); accept only if CV gain ≥ 2 pts Hit@5 over tuned-static RRF *and* survives the permutation test. If static RRF with per-type weights (a 15-constant grid search, no learning at all) matches the LR, **ship the constants and delete the model** — a config file needs no inference path, no serialization, no versioning, and cannot drift.

**6.3.3 — The honest expected outcome:** this buys 1–3 points, maybe zero. That is fine. The point of doing it *properly* is that when it buys zero you *know* fusion is not your bottleneck, permanently, with a CI attached — which is worth more than a fake +4 from leakage.

---

## 6. Phase 6.4 — RunPod asset utilization (runs alongside 6.1–6.3)

You have two standing GPU assets. Their realistic roles:

**Qwen3-Embedding-8B (TEI, 4096-dim):** already the primary embedder. Its Phase-6 role is *fixed inference*, not fine-tuning. One decision to pre-register: any experiment that changes ingestion (6.2.1) requires re-embedding affected nodes — batch these re-embeds per experiment *round*, not per experiment, or TEI throughput becomes the phase's wall-clock bottleneck. Corpus-scale re-embed at ~7.5k nodes is minutes, not hours; this is a luxury of your scale — exploit it while it lasts.

**Qwen3.5-9B (vLLM):** four roles, all labeler/judge-side, none trained: (1) answer generation in evals; (2) failure classification in 6.1.1; (3) query decomposition in 6.2.2; (4) *label distillation*: judge relevance of (query, candidate) pairs on the dev set to densify 6.3's training labels beyond gold-only (gold-only labels mark exactly one positive per question; the LLM judge recovers valid-but-unlabeled positives, which otherwise poison LR training as false negatives — this is the single most important data-quality trick in the phase). Judge prompts are versioned artifacts; a changed judge prompt invalidates all prior labels — treat prompts like schema migrations.

**Cost ceiling for the entire phase: ~$20–30**, dominated by eval-answering tokens. The old Phase 6's budget was similar but bought training runs of components disqualified above; this budget buys *measurements*, which are the scarce good.

---

## 7. Phase 6.5 — Deferred components with explicit trigger conditions

Nothing here is deleted; it is *gated*. Each item re-enters the roadmap automatically when its trigger fires. This converts "we didn't build it" from a judgment call into a measurement.

| Component | Trigger condition (all must hold) | Rationale |
|---|---|---|
| **FAISS IVF-PQ / GPU index** | Corpus ≥ 500k vectors, or p95 search latency > 100ms at production k | Below this, flat/HNSW is exact and fast; PQ only destroys recall |
| **Contrastive embedder fine-tuning** | ≥ 50k nodes; ≥ 100k positive pairs from *non-cosine* provenance (human edges, entity edges, LLM-verified); dev-set evidence that embedding recall (not fusion, not reading) is the binding constraint | Kills the circularity: pairs must carry information the embedder doesn't already encode |
| **GNN reranker (GraphSAGE or better)** | Graph reachability (6.2.3 metric) ≥ 60% on multi-hop dev *and* graph channel shows ranking errors (gold reachable but ranked low) rather than coverage errors | A GNN ranks paths; it cannot invent them |
| **Fusion MLP (upgrade from LR)** | LR-vs-static shows learned fusion pays *and* labeled rows ≥ 10⁵ (via 6.4 distillation at scale) | Capacity should trail data by an order of magnitude, always |

**Kill criteria** (equally binding): if after 6.1+6.2 the E2E dev accuracy has not improved ≥ 8 pts over baseline, stop and re-derive the ledger — the model of where points die is wrong, and continuing to execute a wrong model is the definition of LARP.

---

## 8. Delivery runbook (deterministic, one pass)

Ordering is a dependency chain; do not parallelize across steps, do parallelize within.

1. **D0:** Implement 6.0.1 ledger emission → run baseline twice → confirm bit-identical → commit `benchmarks/results/phase6_baseline.jsonl` + config hash. LoCoMo touch #1.
2. **D1:** 6.1 failure taxonomy on dev → interventions (a)–(c) → re-run dev → paired test → commit or revert each intervention *individually*.
3. **D2–D4:** 6.2 sub-pools in ledger-measured size order. Each intervention: dev run → paired test → commit/revert. Edge-threshold sweep and pool-size audit are half-day items; ingestion granularity is the long pole.
4. **D5:** 6.3 fusion: static per-type grid first, LR second, ship whichever wins, delete the loser.
5. **D6:** Freeze config. Full re-embed + re-index from clean state (guards against index/store drift — the documented BUG-1 class). Run LongMemEval final, MuSiQue final, LoCoMo touch #2. Report all three with CIs, including the sub-2.5-pt "not significant" honesty rows.
6. **Artifact set:** baseline ledger, final ledger, per-intervention diff table (accepted/reverted, p-values), judge prompt versions, config hash, seed record. This set *is* the research contribution — it is what separates a finding from a claim.

No step has a retry path because every step is deterministic given (config hash, seed, corpus snapshot). Where an LLM is in the loop (judging, decomposition), temperature is 0 and prompts are pinned. That is what "no second retries" means in practice: not confidence — *reproducibility*.

---

## 9. What this phase is, philosophically

The old Phase 6 was a list of techniques. This one is a *ledger discipline*: measure where points die, buy the cheapest points first, let data volume dictate model capacity, gate the glamorous components behind trigger conditions that make their future adoption a measurement rather than a mood, and keep the reporting benchmark two-touch clean so the final number means something.

A system at 7.5k nodes that reports honest CIs, a reproducible harness, per-intervention attribution, and trigger-gated deferral of the fashionable stuff is *rarer* in this field than a system with a GNN. That is the actual finding.
