# Late Fusion Scoring and Ingest-Time Neighborhood Averaging

## Abstract
Traditional vector-only retrieval captures semantic similarity but ignores relational structure, while graph-only retrieval lacks semantic nuance. HybridMind addresses this through a weighted linear score fusion that combines semantic distance with graph-based proximity—a well-known late fusion technique in information retrieval literature. Additionally, we apply ingest-time neighborhood averaging, a practical, non-training variant of GraphSAGE-style aggregation that adjusts node embeddings based on their semantic neighborhood to create a more coherent embedding space.

## 1. Motivation

### 1.1 Limitations of Vector-Only Retrieval
- Captures semantic similarity but ignores relational structure.
- Two documents can be semantically distant but causally related (e.g., "gradient descent" and "Adam optimizer" — connected by derivation, not just similarity).
- Pure vector search fails to leverage the explicit knowledge represented in graph edges.

### 1.2 Limitations of Graph-Only Retrieval
- Structurally connected nodes are not always semantically relevant.
- Deep traversal (d > 2) without semantic filtering quickly introduces noise.
- Requires manual, explicit edge creation; does not generalize well to unseen or sparsely connected content.

### 1.3 The Hybrid Approach
Score fusion provides a principled way to combine both signals. HybridMind implements these techniques with an emphasis on **ingest-time conditioning**, ensuring the latent embedding space reflects the relational structure of the database early in the pipeline, all within a self-contained local environment.

## 2. Late Fusion Scoring

### 2.1 Formal Definition
Given a query $q$ and a candidate node $n$, the scoring function used is a weighted linear fusion:

$$Score(q, n) = w_v \cdot V_{eff}(q, n) + w_g \cdot G_{eff}(A, n)$$

Where:
- $V_{eff}(q, n)$ is the effective Vector Score (cosine similarity + BM25 keyword overlap boost).
- $G_{eff}(A, n)$ is the effective Graph Score (proximity gated by BM25 keyword relevance).
- $w_v = 0.5$, $w_g = 0.15$ are the default weights (tuned for LoCoMo-style factoid queries).

Note: The weights do **not** sum to 1.0. The BM25 overlap boost ($w_{bm25} = 0.35$) is applied additively within the vector score component: $V_{eff} = V(q,n) + w_{bm25} \cdot overlap(q,n)$.

### 2.2 Vector Score
The Vector Score $V(q, n)$ is calculated as the cosine similarity between the query embedding $embed(q)$ and the node embedding $embed(n)$:

$$V(q, n) = \frac{dot(embed(q), embed(n))}{||q|| \cdot ||n||}$$

For $L_2$-normalized vectors, this reduces to the dot product: $V(q, n) = dot(q, n)$.

The effective vector score used in fusion adds a BM25 keyword overlap boost:

$$V_{eff}(q, n) = V(q, n) + 0.35 \cdot overlap(q, n)$$

Where $overlap(q, n)$ is the fraction of query terms present in the node text (after NLTK stemming). This allows exact keyword matches to compete with weak semantic similarities.

### 2.3 Graph Score
The Graph Score $G(A, n)$ represents the structural proximity of candidate node $n$ to the set of reference (anchor) nodes $A$:

$$G(A, n) = \max_{a \in A} \left[ \frac{1}{1 + d(a, n)} \right]$$

Where $d(a, n)$ is the shortest path length in the directed graph (traversing in either direction).

The effective graph score is gated by BM25 keyword overlap to prevent irrelevant structural proximity from inflating scores:

$$G_{eff}(A, n) = G(A, n) \cdot \min\left(1.0, \frac{overlap(q, n)}{threshold}\right)$$

When no keyword overlap exists, the graph score is suppressed to zero regardless of structural proximity.

**Score Distribution Table**:
- Distance 0 (self/anchor): 1.000
- Distance 1 (direct neighbor): 0.500
- Distance 2 (2-hop): 0.333
- Distance 3 (3-hop): 0.250
- Distance $\infty$ (no path): 0.000

### 2.4 Anchor Node Selection
The set of anchor nodes $A$ is critical for defining the "center" of the relational context:
- **Implicit Anchors**: When `anchor_nodes` are not provided, $A$ defaults to the top-3 nodes from the initial vector search results. This makes hybrid search automatic but can result in a circular reinforcement where the best semantic matches re-rank themselves.
- **Explicit Anchors**: When provided by the caller, $A$ represents explicit relational context, such as currently active memories or relevant entities from a prior reasoning step. This bypasses search-dependent circularity.

### 2.5 Weight Selection
The default weights are **$w_v=0.5, w_g=0.15$** with a BM25 boost weight of **$w_{bm25}=0.35$**.

This choice reflects the empirical finding that lexical precision (BM25) and semantic similarity (vector) each contribute substantially, while the graph component provides a smaller contextual re-ranking signal.

**Ablation Study Reference (ArXiv dataset)**:
- Pure vector: **NDCG=0.65**
- Hybrid (α=0.6, β=0.4): **NDCG=0.78** (optimal balance in original ablation)
- Pure graph: **NDCG=0.45** (loses semantic relevance)

*Note: The ablation was performed on 150 ArXiv papers with self-reported relevance; human-labeled ground truth validation is ongoing. The production defaults (0.5/0.15/0.35) were further tuned for LoCoMo-style factoid queries where BM25 exact-match is critical.*

## 3. Ingest-Time Neighborhood Averaging

### 3.1 Motivation
Standard hybrid retrieval systems treat the embedding space and graph structure as independent, fusing them only at query time. We apply a practical, non-training variant of GraphSAGE-style aggregation to the embedding space. By conditioning the embedding on its neighborhood at ingest, we ensure geometric proximity in the latent space reflects relational proximity.

### 3.2 Formulation
Given node $n$ with text $t$ and semantic neighbor embeddings $\{e_1, \dots, e_k\}$:
1. $e_{raw} = embed(t)$
2. $e_{neighbors} = \text{mean}(\{e_1, \dots, e_k\})$
3. $e_{conditioned} = \text{normalize}(\alpha \cdot e_{raw} + (1-\alpha) \cdot e_{neighbors})$

Where **$\alpha=0.7$** (own embedding weight). The top-5 semantically similar nodes (determined by cosine search at ingest) serve as the conditioning neighborhood.

### 3.3 Empirical Observations
Measured on 20 nodes in a 1,000-node database (single-domain, arXiv):
- **Mean Cosine Difference (Raw vs Conditioned)**: 0.00976 (approx. 1%).
- **Variance**: Min delta: 0.005, Max delta: 0.018.
- **Node Type Delta**:
  - Edged nodes mean diff: 0.00663
  - Unedged (isolated) nodes mean diff: 0.00877

At multi-domain scale (7,510 nodes across 5 domains), the mean cosine difference doubled to **0.01927** (~2× the single-domain baseline), indicating the conditioning effect scales with graph density and corpus heterogeneity. See [MULTI_DOMAIN_EVAL.md](MULTI_DOMAIN_EVAL.md) §4.3.

### 3.4 Limitations
- The effect size is modest (~0.01). Retrieval improvement specifically due to this technique has not yet been isolated in an ablation.
- Conditioning uses vector neighbors as a proxy for the intended effect, as graph edges may not yet exist for a new node.
- Empty databases provide no conditioning benefit for the initial nodes.

## 4. Sparse Retrieval (BM25) & Keyword Exact Match

### 4.1 NLTK Porter Stemmer
Traditional vector search struggles with "single-hop" fact recall where specific keywords and nouns matter more than semantic neighbors. HybridMind implements an Okapi BM25 Index alongside FAISS. To ensure robust matching without heavy dependencies, it relies on `nltk`'s `PorterStemmer` to strip suffixes (e.g. `researching` -> `research`), significantly improving recall for fact-based questions over simple whitespace tokenization.

### 4.2 Exact Match Cross-Encoder Boost
Vector results and BM25 results are combined by adding an exact-match keyword overlap boost to the vector similarity score for candidates that have high keyword overlap. The BM25 overlap fraction (query terms found in node text after stemming) is multiplied by `bm25_boost_weight=0.35` and added to the cosine similarity score. Additionally, the graph score is gated by BM25 overlap—candidates with zero keyword relevance receive zero graph score regardless of structural proximity. This fuses semantic similarity with lexical precision without requiring a separate ranking pass.

## 5. Retrieval Quality

### 5.1 Eval Methodology
The evaluation pipeline utilizes BM25 overlap as a weak supervision signal for ground truth.
**Limitation**: BM25 excels at keyword matching but fails to label semantic relevance that lacks exact keyword overlap. All metrics should be treated as directional.

### 5.2 Results
The system was empirically evaluated against the LoCoMo benchmark with honest reporting of failures. Peak accuracy was **48% (Qwen3.5 397B)** on a 25-question subset, settling at **36% (GPT-5 Mini)**. A larger 50-question run yielded **18% accuracy**. Most notably, the system exhibited a **0% accuracy on single-hop fact recall** across all runs — conclusively isolated as an LLM extraction failure (returning `Answer: None`), not a retrieval failure. The retrieval hit rate for single-hop was 60% (25-question) and 42% (50-question). Full details: [LOCOMO_BENCHMARK_REPORT.md](LOCOMO_BENCHMARK_REPORT.md).

## 6. Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Node ingest | $O(d + k \cdot d)$ | $O(d)$ | embed + k neighbor lookups + BM25 indexing |
| Vector search | $O(\log n)$ | $O(1)$ | HNSW approximate search |
| Graph score | $O(V + E)$ | $O(V)$ | BFS from anchor nodes |
| Hybrid search | $O(\log n + V+E)$ | $O(k)$ | k candidates re-ranked |
| Compaction | $O(n \cdot d)$ | $O(n \cdot d)$ | Full HNSW rebuild |

*n=nodes, d=dimensions(768), V=graph vertices, E=edges, k=top_k*

## 7. Comparison with Related Systems

| System | Vector | Graph | Hybrid | Conditioned Embeddings | Local-Native |
|--------|:------:|:-----:|:------:|:----------------------:|:------------:|
| ChromaDB | ✓ | ✗ | ✗ | ✗ | ✓ |
| Weaviate | ✓ | ~ | ~ | ✗ | ✗ |
| GraphRAG | ✓ | ✓ | ✓ | ✗ | ✗ |
| Neo4j+pgvec | ✓ | ✓ | manual | ✗ | ✓ |
| **HybridMind** | ✓ | ✓ | ✓ | ✓ (experimental) | ✓ |
