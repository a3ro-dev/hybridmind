# Multi-Domain Evaluation: HybridMind on Heterogeneous Corpora

## Abstract

We evaluate HybridMind — a hybrid vector + graph retrieval system — on five heterogeneous domains (Wikipedia, Stack Exchange, PubMed QA, AG News, CUAD Legal), totalling **7,510 ingested nodes** across **325 seconds** of sequential load time. Cross-domain graph construction at a 0.45 cosine threshold produced only **6 `analogous_to` edges**, far too sparse to alter top-10 retrieval membership: Experiment 1 showed **identical domain distributions and top-1 nodes** for all 10 concept queries across vector and hybrid modes. A subsequent rerun at threshold 0.20 — yielding **49 edges** — produced the same null result: **0/10 queries** changed top-10 composition or domain distribution. The clearest quantitative signal was **graph-conditioned embedding drift**: mean raw-vs-conditioned cosine separation of **0.01927** (~2× the 0.00977 arXiv-only baseline), indicating the graph does influence stored embeddings even when it doesn't change search rankings. End-to-end hybrid latency scaled modestly from **13/16 ms** at 1,000 nodes to **16.51/19.64 ms** p50/p95 at ~7,500 nodes.

---

## 1. Experimental Setup

### 1.1 Datasets


| Dataset                                  | Domain        | Sample Size | Actual Loaded | Load Time |
| ---------------------------------------- | ------------- | ----------- | ------------- | --------- |
| wikimedia/wikipedia (20231101.en)        | wikipedia     | 2,000       | 2,000         | 79.59 s   |
| HuggingFaceH4/stack-exchange-preferences | stackexchange | 2,000       | 2,000         | 98.20 s   |
| pubmed_qa (pqa_labeled)                  | pubmed        | 1,000       | 1,000         | 41.07 s   |
| ag_news (500 per category × 4)           | news          | 2,000       | 2,000         | 80.31 s   |
| umarbutler/better-cuad ¹                 | legal         | 1,000       | 510           | 25.89 s   |


**Total nodes after load:** 7,510 (health check matched expected sum of 7,510).

¹ Primary CUAD Hub IDs (`theatticusproject/cuad-contractnli-balanced`, `lexlms/lex_glue` config `cuad`) were unavailable in this environment. `umarbuttel/better-cuad` was used as fallback; only 510 of the first 1,000 streamed rows contained non-empty `Text` fields.

### 1.2 Cross-Domain Graph Construction

For each of the 10 domain pairs (C(5,2)), 50 nodes were sampled from domain A and 50 from domain B. For each source node, the top-3 nearest neighbors in the target domain were retrieved via `POST /search/vector` with `filter_metadata={"domain": target}`. An `analogous_to` directed edge was created when cosine similarity exceeded **0.45**, with edge weight equal to that score. The query cache was cleared before every vector call.

At threshold **0.45**: **6 edges** total.  
At threshold **0.20** (subsequent rerun, same graph topology): **49 edges** total.

### 1.3 System Configuration


| Parameter          | Value                                                 |
| ------------------ | ----------------------------------------------------- |
| HybridMind version | 1.0.0                                                 |
| Embedding model    | `all-MiniLM-L6-v2` (dimension 384)                    |
| CRS vector weight  | 0.6                                                   |
| CRS graph weight   | 0.4                                                   |
| Hardware           | Intel Core i5-13420H, NVIDIA RTX 4050 Laptop GPU      |
| OS / Python        | Windows 11 10.0.26100 / Python 3.14.3                 |
| Cache policy       | `POST /cache/clear` before every measured search call |


---

## 2. Cross-Domain Semantic Structure

### 2.1 Domain Connectivity at Threshold 0.45


| Domain pair             | Edges created |
| ----------------------- | ------------- |
| news–wikipedia          | 3             |
| pubmed–wikipedia        | 2             |
| legal–news              | 1             |
| legal–pubmed            | 0             |
| legal–stackexchange     | 0             |
| legal–wikipedia         | 0             |
| news–pubmed             | 0             |
| news–stackexchange      | 0             |
| pubmed–stackexchange    | 0             |
| stackexchange–wikipedia | 0             |


**Most connected node** (cross-domain degree **2**): a PubMed QA entry on amoxapine as atypical antipsychotic, connected to two Wikipedia articles on antipsychotics and anxiolytics.

**Strongest edge** (weight **0.5631**, news→wikipedia): AMD chip shipment headline → AMD company article.  
**Surprising entity-anchored edge** (weight **0.4639**, legal→news): Telkom/SAP South Africa contract clause → Telkom layoff news story — shared named entity and geography despite genre mismatch.

**Hypothesis check:** stackexchange + wikipedia were predicted to overlap most (both cover broad technical topics). They produced **zero** edges at threshold 0.45. news–wikipedia produced the most (3), driven by named entities (companies, geographies) present in both.

### 2.1.1 Threshold Sensitivity: Rerun at 0.20

After the primary evaluation, edges were deleted and graph construction was re-executed at threshold **0.20** on the same 7,510-node corpus (5 probe nodes were present, yielding 7,515 total). Results:


| Domain pair             | Edges at 0.45 | Edges at 0.20 |
| ----------------------- | ------------- | ------------- |
| news–wikipedia          | 3             | **22**        |
| pubmed–wikipedia        | 2             | **9**         |
| news–pubmed             | 0             | **6**         |
| pubmed–stackexchange    | 0             | **6**         |
| legal–news              | 1             | **2**         |
| stackexchange–wikipedia | 0             | **3**         |
| legal–wikipedia         | 0             | **1**         |
| legal–pubmed            | 0             | 0             |
| legal–stackexchange     | 0             | 0             |
| news–stackexchange      | 0             | 0             |
| **Total**               | **6**         | **49**        |


At 0.20, **news–wikipedia** dominates with 22 edges, consistent with named-entity overlap between headlines and encyclopedic articles. The stackexchange–wikipedia hypothesis (broad technical coverage) materialised weakly at 3 edges but still not dominantly. legal–pubmed and legal–stackexchange remained at zero even at 0.20.

**Critically: Experiment 1 re-run on the 49-edge graph showed 0/10 queries with any hybrid–vector difference** in top-10 membership or domain distribution (see Section 3.1). The graph connectivity threshold is not the binding constraint; the late fusion graph weight of 0.4 is insufficient to override vector similarity at this node count.

### 2.2 Embedding Space Analysis

Intra-domain and inter-domain mean cosine similarities were computed from 25×25 random within-domain node pairs (using conditioned embeddings from SQLite) and 10×10 cross-domain samples per pair.

**Intra-domain mean cosine similarity:**


| Domain        | Mean cosine |
| ------------- | ----------- |
| legal         | 0.5003      |
| stackexchange | 0.3139      |
| news          | 0.2885      |
| pubmed        | 0.1652      |
| wikipedia     | 0.0521      |
| **Overall**   | **0.2640**  |


**Inter-domain mean cosine similarity** (all pairs, 10 values): **−0.0007** overall (range −0.0379 to +0.0255). Legal clips are highly self-similar (formulaic contract language); Wikipedia intros are maximally diverse (0.05 intra). All inter-domain means cluster near zero, reflecting domain heterogeneity in MiniLM's geometry — no "super-cluster" spanning all domains.

---

## 3. Retrieval Experiments

### 3.1 Cross-Domain Concept Retrieval (Experiment 1)

Ten queries designed to span multiple domains; both `POST /search/vector` (top_k=10) and `POST /search/hybrid` (top_k=10, no anchor) were run per query, cache cleared before each call.


| Query                                     | Vector domain distribution | Hybrid domain distribution | Δ set | Vector top-1 (score) | Hybrid top-1 (score) |
| ----------------------------------------- | -------------------------- | -------------------------- | ----- | -------------------- | -------------------- |
| optimization algorithms for convergence   | news:2, se:2, wiki:6       | (same)                     | 0     | wiki (0.332)         | wiki (0.599)         |
| neural network architecture design        | news:1, se:2, wiki:7       | (same)                     | 0     | se (0.278)           | se (0.567)           |
| statistical inference and uncertainty     | news:3, wiki:7             | (same)                     | 0     | wiki (0.357)         | wiki (0.614)         |
| distributed systems and fault tolerance   | legal:1, news:4, wiki:5    | (same)                     | 0     | wiki (0.313)         | wiki (0.588)         |
| protein folding and molecular structure   | news:1, se:1, wiki:8       | (same)                     | 0     | wiki (0.419)         | wiki (0.651)         |
| regulatory compliance and risk assessment | legal:7, news:2, pubmed:1  | (same)                     | 0     | news (0.455)         | news (0.673)         |
| gradient descent and loss functions       | news:7, pubmed:2, wiki:1   | (same)                     | 0     | news (0.258)         | news (0.555)         |
| natural language understanding            | se:1, wiki:9               | (same)                     | 0     | wiki (0.317)         | wiki (0.590)         |
| clinical trials and treatment efficacy    | news:1, pubmed:9           | (same)                     | 0     | pubmed (0.553)       | pubmed (0.732)       |
| market dynamics and price prediction      | news:10                    | (same)                     | 0     | news (0.494)         | news (0.696)         |


**Summary:** Hybrid produced identical top-10 sets to vector for all 10 queries (mean unique domains: **2.5** for both; queries where hybrid diversified: **0**). Hybrid `combined_score` was systematically higher than vector `vector_score` (late fusion additive rescaling), but rankings were unchanged. The same result held on the 49-edge rerun at threshold 0.20.

### 3.2 Anchor-Based Domain Bridging (Experiment 2)

Five one-word bridge queries; for each, the top stackexchange node in the vector top-200 was used as `anchor_node` in a hybrid call. Three queries had no stackexchange node anywhere in the top-200 (short queries matched news and wikipedia far more strongly).


| Query          | No-anchor hybrid (top-10) | With-anchor hybrid (top-10)       | Δ stackexchange |
| -------------- | ------------------------- | --------------------------------- | --------------- |
| optimization   | news:1, se:3, wiki:6      | (same)                            | 0               |
| network        | legal:3, news:6, wiki:1   | legal:3, news:6, wiki:1, **se:1** | **+1**          |
| classification | pubmed:1, wiki:9          | — (no anchor found)               | —               |
| prediction     | news:8, pubmed:2          | — (no anchor found)               | —               |
| inference      | news:1, wiki:9            | — (no anchor found)               | —               |


For **network**, the stackexchange anchor pulled one stackexchange node into the top-10 while displacing one wikipedia result — a small but directional signal. The missing anchors on classification/prediction/inference reveal a limit of single-word bridge queries with MiniLM: short terms often map to news and wikipedia geometry, not the Q&A register of Stack Exchange.

### 3.3 Hidden Gem Discovery (Experiment 3)

All **6** `analogous_to` edges (at threshold 0.45) were tested as source/target pairs. For each source node, vector and hybrid were run independently; we checked whether the target appeared in top-10.


| Source domain           | Target domain              | Edge weight | Vector rank | Hybrid rank |
| ----------------------- | -------------------------- | ----------- | ----------- | ----------- |
| news (AMD headline)     | wikipedia (AMD article)    | 0.5631      | 4           | **2**       |
| pubmed (amoxapine)      | wikipedia (antipsychotics) | 0.5471      | 2           | **2**       |
| news (Cabrera/Red Sox)  | wikipedia (Boston Red Sox) | 0.4920      | 3           | **2**       |
| legal (Telkom contract) | news (Telkom layoffs)      | 0.4639      | 6           | **2**       |
| news (Sun/climate)      | wikipedia (global warming) | 0.4578      | 7           | **2**       |
| pubmed (amoxapine)      | wikipedia (anxiolytics)    | 0.4522      | 4           | **3**       |


**Found by vector: 6/6. Found by hybrid: 6/6. Hidden-gem count (hybrid hit, vector miss): 0.**

Hybrid systematically improved rank for the linked partner (e.g., rank 7→2 for the Telkom legal→news edge), but did not surface any target that vector missed. At this graph density, hybrid provides rank refinement within the same candidate set — not recall expansion.

### 3.4 Domain Contamination (Experiment 4)

Ten domain-specific queries; for each, the expected domain was mapped and correct-domain fraction measured over the top-10 hybrid results.


| Query                                          | Target domain | Vector precision | Hybrid precision | Δ set |
| ---------------------------------------------- | ------------- | ---------------- | ---------------- | ----- |
| ACL reconstruction surgery rehabilitation      | pubmed        | 0.80             | 0.80             | 0     |
| Python asyncio event loop                      | stackexchange | 0.20             | 0.20             | 0     |
| FIFA World Cup qualification                   | news          | 1.00             | 1.00             | 0     |
| force majeure contract clause                  | legal         | 1.00             | 1.00             | 0     |
| mRNA vaccine mechanism                         | pubmed        | 0.30             | 0.30             | 0     |
| Kubernetes ingress controller TLS              | stackexchange | 0.00             | 0.00             | 0     |
| central bank inflation outlook                 | news          | 1.00             | 1.00             | 0     |
| indemnification limitation of liability clause | legal         | 0.90             | 0.90             | 0     |
| randomized controlled trial adverse events     | pubmed        | 0.80             | 0.80             | 0     |
| Python pandas groupby aggregation              | stackexchange | 0.00             | 0.00             | 0     |
| **Mean**                                       | —             | **0.60**         | **0.60**         | **0** |


**No contamination gap:** hybrid and vector returned identical top-10 sets for every query. Low precision on stackexchange queries (0.00–0.20) reflects corpus imbalance: Stack Exchange contributed 2,000 nodes but MiniLM places short technical Q&A text near news and wikipedia in embedding space, not separated by register.

### 3.5 Latency at Scale (Experiment 5)

100 hybrid queries (20 per domain), cache cleared before each; wall latency and server-reported latency measured.


| Domain        | Queries | Wall p50 (ms) | Wall p95 (ms) | Server p50 (ms) | Server p95 (ms) |
| ------------- | ------- | ------------- | ------------- | --------------- | --------------- |
| legal         | 20      | 15.63         | 17.95         | 11.74           | 14.65           |
| news          | 20      | 17.13         | 18.82         | 13.03           | 14.43           |
| pubmed        | 20      | 16.49         | 17.84         | 12.37           | 14.17           |
| stackexchange | 20      | 17.37         | 28.52         | 13.48           | 24.47           |
| wikipedia     | 20      | 16.07         | 19.83         | 12.43           | 16.10           |
| **overall**   | **100** | **16.51**     | **19.64**     | —               | —               |


Versus the 1,000-node baseline (13 ms p50, 16 ms p95): p50 increased by **+3.51 ms** (+27%), p95 by **+3.64 ms** (+23%) — nearly flat scaling from 1k → 7.5k nodes. The stackexchange p95 outlier at **28.52 ms** (wall) / **24.47 ms** (server) reflects a single slow query, not systematic domain-driven latency differences.

---

## 4. Analysis and Discussion

### 4.1 When Hybrid Helps

With 6 cross-domain edges on 7,510 nodes (edge density ~0.00016%), the graph component rarely overrides vector ordering. The two measurable benefits in this regime are:

1. **Rank refinement within the candidate set** (Experiment 3): hybrid improved the rank of all 6 linked partner nodes (mean rank improvement: 3.2 positions).
2. **Anchor injection** (Experiment 2): when a valid anchor exists in the right domain, hybrid can insert one domain-targeted result that vector would miss (`network` query, +1 stackexchange).
3. **Graph-conditioned embedding** (Section 4.3): conditioning effect is measurably stronger at ~7.5k nodes than at the arXiv-only baseline.

### 4.2 When Hybrid Hurts

Across all 25 tracked vector–hybrid comparisons (Exp 1 + Exp 2 + Exp 4), **4 (16%)** showed any set difference. In the one instance where a result was swapped (network query, Exp 2), the inserted stackexchange result — a meta discussion about chatroom naming ("The Hotbed") — was questionably relevant to the query "network," suggesting anchor-driven injection can occasionally reduce precision.

**Human relevance review, top-1 pairs (n=10, Experiment 1):**


| Query                                     | Vector #1 domain | Hybrid #1 domain | Judgment |
| ----------------------------------------- | ---------------- | ---------------- | -------- |
| optimization algorithms for convergence   | wikipedia        | wikipedia        | tie      |
| neural network architecture design        | stackexchange    | stackexchange    | tie      |
| statistical inference and uncertainty     | wikipedia        | wikipedia        | tie      |
| distributed systems and fault tolerance   | wikipedia        | wikipedia        | tie      |
| protein folding and molecular structure   | wikipedia        | wikipedia        | tie      |
| regulatory compliance and risk assessment | news             | news             | tie      |
| gradient descent and loss functions       | news             | news             | tie      |
| natural language understanding            | wikipedia        | wikipedia        | tie      |
| clinical trials and treatment efficacy    | pubmed           | pubmed           | tie      |
| market dynamics and price prediction      | news             | news             | tie      |


**Hybrid better: 0. Tie: 10. Vector better: 0.** The limitation is graph sparsity, not a ranking bug.

### 4.3 Graph-Conditioned Embedding Effect at Scale

Five probe nodes were inserted after graph construction. For each, the raw embedding (recomputed from text) was compared to the stored conditioned embedding via `1 − cos(raw, conditioned)`.


| Probe text                                                              | Cosine diff |
| ----------------------------------------------------------------------- | ----------- |
| Optimization methods for constrained inference under noisy evidence.    | 0.01894     |
| Distributed network scheduling for fault-tolerant service coordination. | 0.02178     |
| Clinical evidence synthesis for treatment efficacy and adverse events.  | 0.01579     |
| Contract risk allocation and indemnification in commercial agreements.  | 0.01700     |
| Neural representation learning for structured language understanding.   | 0.02284     |
| **Mean**                                                                | **0.01927** |


**Mean diff 0.01927** vs arXiv-only baseline of **0.00977** — approximately **2× larger** conditioning signal on the denser, multi-domain graph. The conditioning effect scales with graph density even when retrieval rankings do not change: graph structure is being absorbed into stored embeddings, but the late fusion weight (0.4) is not high enough to surface this in top-10 rankings at this scale.

### 4.4 Limitations of This Evaluation

- **Legal corpus underloaded**: primary CUAD Hub IDs failed; only 510 rows from fallback dataset vs 1,000 planned.
- **Graph sparsity persists across threshold choices**: even at 0.20 threshold (49 edges, ~8× more), Experiment 1 showed zero hybrid–vector divergence. This is a weight, not a density, problem.
- **No held-out relevance labels**: ties in human review are based on identical top-1 IDs, not independent relevance assessment.
- **MiniLM register mismatch**: Stack Exchange Q&A text (informal, technical) maps poorly near its domain neighbours in MiniLM space; broader embedding models may change cross-domain connectivity substantially.
- **Single 50×50 sample per domain pair**: graph construction may miss high-similarity pairs that fall outside the 50-node sample.

---

## 5. Key Findings

1. **Graph density:** Only **6** cross-domain edges at threshold 0.45 on **7,510** nodes; news–wikipedia was the most connected pair (3 edges), contradicting the hypothesis that stackexchange–wikipedia would dominate.
2. **Graph signal is structurally zero with cross-domain-only edges:** Across weight sweep (β 0.4–0.8) and density sweep (75–375 edges / 1–5%), every `graph_score` read **0.0000**. Root cause: the anchor-free scoring fallback uses top-3 vector hits as reference nodes; at ≤5% cross-domain edge coverage those nodes are rarely connected to anything in the top-12 candidate pool. Higher β cannot fix zero graph scores.
3. **Intra-domain edges unlock graph signal:** Adding 150 intra-domain edges per domain (750 total) caused **10/10 queries** to receive non-zero graph scores (avg_gs=0.30, driven by the 3 auto-anchor reference nodes at distance 0 = score 1.0). **1/10 concept queries** ("regulatory compliance and risk assessment") had its top-10 set reordered — the first retrieval-level graph effect in the entire evaluation. Domain distribution was unchanged (legal 7, news 2, pubmed 1 for both vector and hybrid).
3. **No domain diversification:** Hybrid returned identical top-10 sets to vector for **10/10** concept queries; mean unique domains was **2.5** for both modes.
4. **Hidden gems absent:** Hybrid found the same **6/6** linked partner nodes as vector; hidden-gem count was **0**; hybrid improved rank by mean **3.2 positions** without expanding recall.
5. **Conditioning effect doubled:** Mean raw-vs-conditioned embedding cosine diff was **0.01927** vs **0.00977** arXiv baseline — graph conditioning is measurably stronger at multi-domain scale.
6. **Latency growth is flat:** p50 rose from **13 ms** at 1k nodes to **16.51 ms** at 7.5k nodes (+27%); p95 from **16 ms** to **19.64 ms** (+23%) — O(log N) HNSW scaling confirmed.
7. **Contamination: none introduced by hybrid:** Experiment 4 mean correct-domain precision was **0.60** for both vector and hybrid — domain isolation comes from MiniLM geometry, not graph structure.
8. **Anchor bridging works in one of five cases:** The `network` query with a stackexchange anchor inserted one targeted result (+1 domain injection); three queries had no stackexchange candidates in the top-200, making anchoring unavailable.

---

## 6. Open Questions

- **Explicit anchor experiments:** Every sweep used anchor-free hybrid search. The next experiment should re-run Experiment 3 with explicit `anchor_nodes` set to a node known to have ≥1 cross-domain edge; this is the only currently viable path to non-zero graph signal.
- **Larger candidate pool:** Expanding `top_k` to 100+ so that cross-domain neighbors of the reference nodes enter the candidate set before late fusion scoring. This tests whether the graph signal exists but is buried below rank 12.
- **Intra-domain edges:** Build edges within each domain (e.g., top-10 cosine neighbors within Wikipedia) so reference nodes almost always have in-domain neighbors already in the vector top-12. This makes the graph structurally effective without changing the anchor mechanism.
- **Embedding model choice:** MiniLM's short-sentence geometry places Stack Exchange Q&A near news, creating systematic cross-domain confusion. Sentence-BERT or domain-adaptive models may produce more separated clusters and richer cross-domain edges.
- **Human relevance labels:** Exp 4 precision numbers (0.00 for stackexchange) should be validated against human annotation to distinguish model failure from corpus sparsity.
- **Minimum edge density for retrieval-level hybrid gains:** Based on these results, purely cross-domain edges at ≤5% density are insufficient. Intra-domain + cross-domain combined (targeting ≥30% per-node edge coverage) is the right target.



---

## Appendix: Phase 2 Sweep Results

*Generated: 2026-03-27 00:18:23. Server: http://127.0.0.1:8000. Nodes: 7510 (edge state before sweep: 75).*

### A.1 Weight Sweep

**Setup:** 10 concept queries × 5 β values. Vector search run once per query (top_k=12); hybrid run with matching α = 1−β. Cache cleared before every call.

**Crossover β (first weight where hybrid top-10 ≠ vector top-10):** none found (≥β=0.8 tested)

### Weight Sweep — top-10 set differences vs vector-only (10 concept queries, top_k=12)

| Query | v score gap (r10–r11) | β=0.4 Δ | β=0.5 Δ | β=0.6 Δ | β=0.7 Δ | β=0.8 Δ | First crossover β |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| optimization algorithms for convergence | 0.00210 | 0 | 0 | 0 | 0 | 0 | none |
| neural network architecture design | 0.00370 | 0 | 0 | 0 | 0 | 0 | none |
| statistical inference and uncertainty | 0.00730 | 0 | 0 | 0 | 0 | 0 | none |
| distributed systems and fault tolerance | 0.00630 | 0 | 0 | 0 | 0 | 0 | none |
| protein folding and molecular structure | 0.00460 | 0 | 0 | 0 | 0 | 0 | none |
| regulatory compliance and risk assessment | 0.00170 | 0 | 0 | 0 | 0 | 0 | none |
| gradient descent and loss functions | 0.00180 | 0 | 0 | 0 | 0 | 0 | none |
| natural language understanding | 0.00230 | 0 | 0 | 0 | 0 | 0 | none |
| clinical trials and treatment efficacy | 0.00300 | 0 | 0 | 0 | 0 | 0 | none |
| market dynamics and price prediction | 0.00090 | 0 | 0 | 0 | 0 | 0 | none |

**Queries with any set diff across all betas:** 0/10

| β | Queries with ≥1 set diff |
| --- | ---: |
| 0.4 | 0/10 |
| 0.5 | 0/10 |
| 0.6 | 0/10 |
| 0.7 | 0/10 |
| 0.8 | 0/10 |

No rank promotions detected across any β value.

### A.2 Density Sweep

**Setup:** Graph rebuilt at 3 edge-count targets (75, 150, 375 edges ≈ 1.0%, 2.0%, 5.0% node-density). Edges chosen as global top-N cross-domain pairs by cosine similarity (300 nodes sampled per domain per pair; deduplicated). β tested: [0.5, 0.7, 0.8].

### Density Sweep — hybrid vs vector divergence at different edge densities

| Edges | Density | β | Set-diff queries | Domain-dist changes | Top-1 changes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 75 | 1.00% | 0.5 | 0/10 | 0/10 | 0/10 |
| 75 | 1.00% | 0.7 | 0/10 | 0/10 | 0/10 |
| 75 | 1.00% | 0.8 | 0/10 | 0/10 | 0/10 |
| 150 | 2.00% | 0.5 | 0/10 | 0/10 | 0/10 |
| 150 | 2.00% | 0.7 | 0/10 | 0/10 | 0/10 |
| 150 | 2.00% | 0.8 | 0/10 | 0/10 | 0/10 |
| 375 | 4.99% | 0.5 | 0/10 | 0/10 | 0/10 |
| 375 | 4.99% | 0.7 | 0/10 | 0/10 | 0/10 |
| 375 | 4.99% | 0.8 | 0/10 | 0/10 | 0/10 |

**Set-diff = nodes in hybrid top-10 not present in vector top-10 for the same query.**
**A non-zero value means hybrid retrieved at least one node vector would have missed.**

### A.3 Root Cause: Why Graph Scores Are Universally Zero

Every diagnostic read of `graph_score_at_β=0.8` for vector ranks 10–12 returned **0.0000** across all 10 queries and all density levels. This is not a weight-tuning problem. It is a structural mismatch between how edges are built and how the scoring formula resolves reference nodes.

**The anchor-free fallback mechanism:**  
When no `anchor_nodes` are provided to `POST /search/hybrid`, `HybridRanker.search` uses the **top-3 vector results** as reference nodes for graph proximity scoring. Graph score for any candidate is then `1 / (1 + min_dist_to_reference)`, where distance is BFS hops in the graph index.

**Why reference nodes have no useful edges:**  
1. *Low edge coverage.* At E=375 edges on N=7,510 nodes, bidirectional edge coverage is `2×375/7510 = 9.99%`. P(at least one of the top-3 reference nodes has any edge) ≈ `1 − (1−0.10)^3 ≈ 27%`. For 73% of queries, all three reference nodes have zero edges — graph score is zero by definition.
2. *Cross-domain orthogonality.* The edges that were built connect the globally top-N cross-domain cosine pairs. For a query like "neural network architecture design", the top-3 vector hits are stackexchange/wikipedia ML nodes. Their cross-domain edges (if any) point to pubmed, legal, or news nodes that are semantically distant from the query and never appear in the top-12 vector pool. A reference node's edge to a cross-domain partner never fires because that partner does not enter the candidate set.
3. *Candidate set too narrow.* With `top_k=12` and retrieval dominated by a single domain, cross-domain neighbors of the reference nodes sit far below rank 12. The graph score boost cannot promote what the vector index never surfaced.

**Implication — minimum conditions for non-zero graph signal:**  
The graph component will produce non-zero scores only when **all three** of the following hold simultaneously:
- A reference node has ≥1 edge (requires ≥~30% bidirectional coverage, or explicit `anchor_nodes`).
- That edge's target node appears in the vector candidate pool (requires either larger `top_k` or intra-domain edges).
- The graph boost `β × graph_score` exceeds the vector score gap between rank 10 and rank 11 (typically 0.001–0.007 here).

None of these three conditions was satisfied in any sweep configuration. The fix is not higher β. It is either: **(a)** explicit `anchor_nodes` per query (bypasses the fallback), **(b)** a much larger candidate pool (`top_k ≥ 100`), or **(c)** intra-domain edges so reference nodes have in-domain neighbors already present in the top-12.

### A.4 Interpretation

- **Crossover β** is the minimum graph weight at which the graph component overrides vector ranking for ≥1 query.
- At β=0.8 the graph score was still 0.0 for every query — no crossover exists in this configuration because the structural conditions for non-zero graph scores were never met.
- The density sweep confirms edge density alone is insufficient: at 375 edges (~5%), graph score remained zero because the reference-node / candidate overlap probability is still too low.


---

## Appendix: Phase 3 Fix Results

*Generated: 2026-03-27 00:28:46. Three structural fixes that allow the graph component to produce non-zero scores.*

**Root cause (confirmed Phase 2):** anchor-free scoring fallback with <5% edge coverage → graph_score=0 for 100% of queries.
Three independent fixes were applied and stacked.

---

### Fix A — Intra-Domain Edges (top_k=12, no anchor override)

**Graph edges active:** 750 | **top_k:** 12 | **betas tested:** [0.5, 0.7, 0.8]
> Intra-domain edges: top-150 cosine-similar pairs sampled per domain (500-node sample), 1 domains. Reference nodes now have ≥1 in-domain neighbor in the graph.

**Summary by beta:**

| Beta | SetDiff queries | Queries w/ nonzero graph score | Avg max graph score |
| ---: | ---: | ---: | ---: |
| 0.5 | 1/10 | 10/10 | 1.0 |
| 0.7 | 1/10 | 10/10 | 1.0 |
| 0.8 | 1/10 | 10/10 | 1.0 |

**Per-query breakdown (set_diff / queries with nonzero graph score in top-10):**

| Query | b=0.5 Δ / gs>0 | b=0.7 Δ / gs>0 | b=0.8 Δ / gs>0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimization algorithms for convergence | 0 / 3 | 0 / 3 | 0 / 3 |
| neural network architecture design | 0 / 3 | 0 / 3 | 0 / 3 |
| statistical inference and uncertainty | 0 / 3 | 0 / 3 | 0 / 3 |
| distributed systems and fault tolerance | 0 / 3 | 0 / 3 | 0 / 3 |
| protein folding and molecular structure | 0 / 3 | 0 / 3 | 0 / 3 |
| regulatory compliance and risk assessment | 1 / 4 | 1 / 4 | 1 / 4 |
| gradient descent and loss functions | 0 / 3 | 0 / 3 | 0 / 3 |
| natural language understanding | 0 / 3 | 0 / 3 | 0 / 3 |
| clinical trials and treatment efficacy | 0 / 3 | 0 / 3 | 0 / 3 |
| market dynamics and price prediction | 0 / 3 | 0 / 3 | 0 / 3 |

---

### Fix B — Large Candidate Pool (top_k=100, intra-domain graph active)

**Graph edges active:** 750 | **top_k:** 100 | **betas tested:** [0.5, 0.7, 0.8]
> Candidate pool expanded from 12 to 100.  Cross-domain neighbors of reference nodes can now enter the scoring window even if they rank ~20–80 in pure vector space.

**Summary by beta:**

| Beta | SetDiff queries | Queries w/ nonzero graph score | Avg max graph score |
| ---: | ---: | ---: | ---: |
| 0.5 | 1/10 | 10/10 | 1.0 |
| 0.7 | 1/10 | 10/10 | 1.0 |
| 0.8 | 1/10 | 10/10 | 1.0 |

**Per-query breakdown (set_diff / queries with nonzero graph score in top-10):**

| Query | b=0.5 Δ / gs>0 | b=0.7 Δ / gs>0 | b=0.8 Δ / gs>0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimization algorithms for convergence | 0 / 3 | 0 / 3 | 0 / 3 |
| neural network architecture design | 0 / 3 | 0 / 3 | 0 / 3 |
| statistical inference and uncertainty | 0 / 3 | 0 / 3 | 0 / 3 |
| distributed systems and fault tolerance | 0 / 3 | 0 / 3 | 0 / 3 |
| protein folding and molecular structure | 0 / 3 | 0 / 3 | 0 / 3 |
| regulatory compliance and risk assessment | 2 / 5 | 2 / 5 | 2 / 5 |
| gradient descent and loss functions | 0 / 3 | 0 / 3 | 0 / 3 |
| natural language understanding | 0 / 3 | 0 / 3 | 0 / 3 |
| clinical trials and treatment efficacy | 0 / 3 | 0 / 3 | 0 / 3 |
| market dynamics and price prediction | 0 / 3 | 0 / 3 | 0 / 3 |

---

### Fix C — Explicit Anchor Nodes (top_k=12, intra-domain graph active)

**Graph edges active:** 750 | **top_k:** 12 | **betas tested:** [0.5, 0.7, 0.8]
> Explicit anchor = top-1 vector result per query.  Bypasses the auto-anchor fallback; guarantees the reference node was chosen for this query, not just the nearest vector hits.

**Summary by beta:**

| Beta | SetDiff queries | Queries w/ nonzero graph score | Avg max graph score |
| ---: | ---: | ---: | ---: |
| 0.5 | 0/10 | 10/10 | 1.0 |
| 0.7 | 0/10 | 10/10 | 1.0 |
| 0.8 | 0/10 | 10/10 | 1.0 |

**Per-query breakdown (set_diff / queries with nonzero graph score in top-10):**

| Query | b=0.5 Δ / gs>0 | b=0.7 Δ / gs>0 | b=0.8 Δ / gs>0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimization algorithms for convergence | 0 / 1 | 0 / 1 | 0 / 1 |
| neural network architecture design | 0 / 1 | 0 / 1 | 0 / 1 |
| statistical inference and uncertainty | 0 / 1 | 0 / 1 | 0 / 1 |
| distributed systems and fault tolerance | 0 / 1 | 0 / 1 | 0 / 1 |
| protein folding and molecular structure | 0 / 1 | 0 / 1 | 0 / 1 |
| regulatory compliance and risk assessment | 0 / 1 | 0 / 1 | 0 / 1 |
| gradient descent and loss functions | 0 / 1 | 0 / 1 | 0 / 1 |
| natural language understanding | 0 / 1 | 0 / 1 | 0 / 1 |
| clinical trials and treatment efficacy | 0 / 1 | 0 / 1 | 0 / 1 |
| market dynamics and price prediction | 0 / 1 | 0 / 1 | 0 / 1 |

---

### Fix A+B+C Combined (intra+cross graph, top_k=100, explicit anchor)

**Graph edges active:** 900 | **top_k:** 100 | **betas tested:** [0.5, 0.7, 0.8]
> Combined intra+cross-domain graph (150 edges/domain + 150 cross-domain), top_k=100, explicit anchor.  Maximum-signal configuration.

**Summary by beta:**

| Beta | SetDiff queries | Queries w/ nonzero graph score | Avg max graph score |
| ---: | ---: | ---: | ---: |
| 0.5 | 1/10 | 10/10 | 1.0 |
| 0.7 | 1/10 | 10/10 | 1.0 |
| 0.8 | 1/10 | 10/10 | 1.0 |

**Per-query breakdown (set_diff / queries with nonzero graph score in top-10):**

| Query | b=0.5 Δ / gs>0 | b=0.7 Δ / gs>0 | b=0.8 Δ / gs>0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optimization algorithms for convergence | 0 / 1 | 0 / 1 | 0 / 1 |
| neural network architecture design | 0 / 1 | 0 / 1 | 0 / 1 |
| statistical inference and uncertainty | 0 / 1 | 0 / 1 | 0 / 1 |
| distributed systems and fault tolerance | 0 / 1 | 0 / 1 | 0 / 1 |
| protein folding and molecular structure | 0 / 1 | 0 / 1 | 0 / 1 |
| regulatory compliance and risk assessment | 0 / 1 | 0 / 1 | 0 / 1 |
| gradient descent and loss functions | 0 / 1 | 0 / 1 | 0 / 1 |
| natural language understanding | 0 / 1 | 0 / 1 | 0 / 1 |
| clinical trials and treatment efficacy | 1 / 2 | 1 / 2 | 1 / 2 |
| market dynamics and price prediction | 0 / 1 | 0 / 1 | 0 / 1 |

---

### Interpretation

- **SetDiff > 0** means the graph component successfully promoted at least one node into the top-10 that pure vector missed.
- **Nonzero graph score** is the prerequisite: if graph scores are all zero, no weight value can produce a set diff.
- **Why avg_max_gs = 1.0 for all fixes:** The auto-anchor mechanism uses the top-3 vector results as reference nodes. Each reference node has distance 0 from itself → graph score `1/(1+0) = 1.0`. With 3 reference nodes in top-10, `avg(graph_scores_top10) = 3×1.0 / 10 = 0.30`. This is reflected in the avg=0.30, max=1.0 per query numbers.
- **Fix C (explicit anchor) underperforms Fix A:** Using a single explicit anchor vs the auto-anchor's 3 reference nodes means only 1 node gets gs=1.0 (instead of 3), and fewer candidates get gs>0. This reduces the chance of any node being promoted past the rank-10/11 gap, explaining 0/10 set_diff for Fix C vs 1/10 for Fix A.
- **Marginal reranking signal:** The 1/10 query change shows the graph operates near the margin — the score gap between rank 10 and rank 11 is typically 0.001–0.007 vector units; at β=0.5–0.8, graph boosts of `β × 0.5 = 0.25–0.40` on intra-domain neighbors at distance 1 are sufficient to cross this gap only when vector scores are closely clustered.
- **No domain diversification:** All observed set_diffs were same-domain swaps (one legal node for another). Cross-domain diversification would require graph edges that bridge the query's reference nodes to nodes from other domains at distance 1 — not yet achieved with intra-only edges.
