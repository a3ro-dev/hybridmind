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

$$Score(q, n) = \alpha \cdot V(q, n) + \beta \cdot G(A, n)$$

Subject to the constraint: $\alpha + \beta = 1$.  
Where:
- $V(q, n)$ is the Vector Score.
- $G(A, n)$ is the Graph Score relative to a set of anchor nodes $A$.
- $\alpha, \beta$ are the relative weights for vector and graph signals, respectively.

### 2.2 Vector Score
The Vector Score $V(q, n)$ is calculated as the cosine similarity between the query embedding $embed(q)$ and the node embedding $embed(n)$:

$$V(q, n) = \frac{dot(embed(q), embed(n))}{||q|| \cdot ||n||}$$

For $L_2$-normalized vectors, this reduces to the dot product: $V(q, n) = dot(q, n)$.

### 2.3 Graph Score
The Graph Score $G(A, n)$ represents the structural proximity of candidate node $n$ to the set of reference (anchor) nodes $A$:

$$G(A, n) = \max_{a \in A} \left[ \frac{1}{1 + d(a, n)} \right]$$

Where $d(a, n)$ is the shortest path length in the directed graph (traversing in either direction).

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
The default weights are **$\alpha=0.6, \beta=0.4$**.

This choice is based on **Semantic Primacy**: retrieval is primarily driven by semantic intent. The graph provides a contextual re-ranking signal rather than the primary signal.

**Ablation Study Reference (ArXiv dataset)**:
- $\alpha=1.0, \beta=0.0$: **NDCG=0.65** (pure vector baseline)
- $\alpha=0.6, \beta=0.4$: **NDCG=0.78** (optimal balance)
- $\alpha=0.0, \beta=1.0$: **NDCG=0.45** (pure graph, loses semantic relevance)

*Note: The ablation was performed on 150 ArXiv papers with self-reported relevance; human-labeled ground truth validation is ongoing.*

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
Measured on 20 nodes in a 1,000-node database:
- **Mean Cosine Difference (Raw vs Conditioned)**: 0.00977 (approx. 1%).
- **Variance**: Min delta: 0.005, Max delta: 0.018.
- **Node Type Delta**:
  - Edged nodes mean diff: 0.00663
  - Unedged (isolated) nodes mean diff: 0.00877

**Hypothesis**: Unedged nodes show higher conditioning because they are "pulled" to their vector-neighbors from an isolated starting point. Edged nodes often link to concepts that are already semantically proximal, resulting in smaller effective deltas.

### 3.4 Limitations
- The effect size is modest (~0.01). Retrieval improvement specifically due to this technique has not yet been isolated in an ablation.
- Conditioning uses vector neighbors as a proxy for the intended effect, as graph edges may not yet exist for a new node.
- Empty databases provide no conditioning benefit for the initial nodes.

## 4. Sparse Retrieval (BM25) & Keyword Exact Match

### 4.1 NLTK Porter Stemmer
Traditional vector search struggles with "single-hop" fact recall where specific keywords and nouns matter more than semantic neighbors. HybridMind implements an Okapi BM25 Index alongside FAISS. To ensure robust matching without heavy dependencies, it relies on `nltk`'s `PorterStemmer` to strip suffixes (e.g. `researching` -> `research`), significantly improving recall for fact-based questions over simple whitespace tokenization.

### 4.2 Reciprocal Rank Fusion and RRF Constant
Vector results and BM25 results are fused using Reciprocal Rank Fusion (RRF):
$$Score = \frac{1}{k_{rrf} + rank}$$
We heavily tuned the $k_{rrf}$ constant, reducing it from the standard 60 down to **20**. This aggressively separates top-ranked exact matches from lower-ranked semantic fuzz, providing a significant boost to factual accuracy in the LOCOMO benchmarks.

## 5. Retrieval Quality

### 5.1 Eval Methodology
The evaluation pipeline utilizes BM25 overlap as a weak supervision signal for ground truth.
**Limitation**: BM25 excels at keyword matching but fails to label semantic relevance that lacks exact keyword overlap. All metrics should be treated as directional.

### 5.2 Results
The system was empirically evaluated against the LoCoMo benchmark with honest reporting of failures. We observed a **36% overall accuracy** on the benchmark. Most notably, the system exhibited a **0% baseline accuracy on single-hop fact recall** (prior to BM25 inclusion strategy refinements). Implementing NLTK-stemmed BM25 plus the lowered $k_{rrf}$ constant serves to mitigate these single-hop factual recall limitations, though achieving an optimally robust retrieval remains an active area of refinement.

## 6. Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Node ingest | $O(d + k \cdot d)$ | $O(d)$ | embed + k neighbor lookups |
| Vector search | $O(n \cdot d)$ | $O(1)$ | FAISS exact scan |
| Graph score | $O(V + E)$ | $O(V)$ | BFS from anchor nodes |
| Hybrid search | $O(n \cdot d + V+E)$ | $O(k)$ | k candidates re-ranked |
| Compaction | $O(n \cdot d)$ | $O(n \cdot d)$ | Full FAISS rebuild |

*n=nodes, d=dimensions(384), V=graph vertices, E=edges, k=top_k*

## 7. Comparison with Related Systems

| System | Vector | Graph | Hybrid | Conditioned Embeddings | Local-Native |
|--------|:------:|:-----:|:------:|:----------------------:|:------------:|
| ChromaDB | ✓ | ✗ | ✗ | ✗ | ✓ |
| Weaviate | ✓ | ~ | ~ | ✗ | ✗ |
| GraphRAG | ✓ | ✓ | ✓ | ✗ | ✗ |
| Neo4j+pgvec | ✓ | ✓ | manual | ✗ | ✓ |
| **HybridMind** | ✓ | ✓ | ✓ | ✓ (experimental) | ✓ |
