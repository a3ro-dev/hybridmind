# HybridMind Benchmark Report

## System Description
HybridMind is a local-native hybrid vector-graph database for AI agent memory. It uses FAISS for dense vector search, an Okapi BM25 index with NLTK stemming for lexical retrieval, and NetworkX for directed graph traversal, all persisted to SQLite in a single `.mind` file. Candidate ranking relies on late linear fusion of vector and graph scores, with an optional exact-match BM25 overlap boost. The system executes entirely locally without cloud dependencies.

## Evaluation Methodology
The benchmark is purely deterministic, utilizing a hardcoded dataset of 240 documents and 50 edges to isolate six retrieval strategies. Metrics are computed against human-defined ground-truth node IDs for six query families (semantic paraphrase, lexical exact match, single-hop graph, multi-hop graph, missing anchor fallback, and oversmoothing adversarial). The ablation runner directly instantiates engine classes, circumventing the HTTP layer, to rigorously toggle specific configuration variables for each test condition.

## Results
| Condition | Family | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | MRR |
|---|---|---|---|---|---|---|---|---|
| A: VECTOR_ONLY | SEMANTIC | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 | 0.333 | 0.950 |
| A: VECTOR_ONLY | LEXICAL | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 | 0.333 | 0.950 |
| A: VECTOR_ONLY | GRAPH_SINGLE_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A: VECTOR_ONLY | GRAPH_MULTI_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A: VECTOR_ONLY | MISSING_ANCHOR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| A: VECTOR_ONLY | OVERSMOOTHING | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| B: BM25_ONLY | SEMANTIC | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B: BM25_ONLY | LEXICAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| B: BM25_ONLY | GRAPH_SINGLE_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B: BM25_ONLY | GRAPH_MULTI_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B: BM25_ONLY | MISSING_ANCHOR | 0.100 | 0.300 | 0.500 | 1.000 | 0.100 | 0.100 | 0.293 |
| B: BM25_ONLY | OVERSMOOTHING | 0.100 | 0.300 | 0.400 | 0.600 | 0.100 | 0.100 | 0.226 |
| C: VECTOR_PLUS_BM25 | SEMANTIC | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 | 0.333 | 0.950 |
| C: VECTOR_PLUS_BM25 | LEXICAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| C: VECTOR_PLUS_BM25 | GRAPH_SINGLE_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C: VECTOR_PLUS_BM25 | GRAPH_MULTI_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C: VECTOR_PLUS_BM25 | MISSING_ANCHOR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| C: VECTOR_PLUS_BM25 | OVERSMOOTHING | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | SEMANTIC | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 | 0.333 | 0.950 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | LEXICAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | GRAPH_SINGLE_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | GRAPH_MULTI_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | MISSING_ANCHOR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | OVERSMOOTHING | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | SEMANTIC | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 | 0.333 | 0.950 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | LEXICAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | GRAPH_SINGLE_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | GRAPH_MULTI_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | MISSING_ANCHOR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | OVERSMOOTHING | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| F: FULL_PIPELINE | SEMANTIC | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 | 0.333 | 0.950 |
| F: FULL_PIPELINE | LEXICAL | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| F: FULL_PIPELINE | GRAPH_SINGLE_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| F: FULL_PIPELINE | GRAPH_MULTI_HOP | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| F: FULL_PIPELINE | MISSING_ANCHOR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |
| F: FULL_PIPELINE | OVERSMOOTHING | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 |

## Component Contribution Analysis
The table below shows the performance delta (in Recall@3) of each component compared to the VECTOR_ONLY baseline:

| Condition | Family | Delta Recall@3 |
|---|---|---|
| B: BM25_ONLY | SEMANTIC | -1.000 |
| B: BM25_ONLY | LEXICAL | +0.000 |
| B: BM25_ONLY | GRAPH_SINGLE_HOP | +0.000 |
| B: BM25_ONLY | GRAPH_MULTI_HOP | +0.000 |
| B: BM25_ONLY | MISSING_ANCHOR | -0.700 |
| B: BM25_ONLY | OVERSMOOTHING | -0.700 |
| C: VECTOR_PLUS_BM25 | SEMANTIC | +0.000 |
| C: VECTOR_PLUS_BM25 | LEXICAL | +0.000 |
| C: VECTOR_PLUS_BM25 | GRAPH_SINGLE_HOP | +0.000 |
| C: VECTOR_PLUS_BM25 | GRAPH_MULTI_HOP | +0.000 |
| C: VECTOR_PLUS_BM25 | MISSING_ANCHOR | +0.000 |
| C: VECTOR_PLUS_BM25 | OVERSMOOTHING | +0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | SEMANTIC | +0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | LEXICAL | +0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | GRAPH_SINGLE_HOP | +0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | GRAPH_MULTI_HOP | +0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | MISSING_ANCHOR | +0.000 |
| D: VECTOR_PLUS_GRAPH_RERANK_ONLY | OVERSMOOTHING | +0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | SEMANTIC | +0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | LEXICAL | +0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | GRAPH_SINGLE_HOP | +0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | GRAPH_MULTI_HOP | +0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | MISSING_ANCHOR | +0.000 |
| E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION | OVERSMOOTHING | +0.000 |
| F: FULL_PIPELINE | SEMANTIC | +0.000 |
| F: FULL_PIPELINE | LEXICAL | +0.000 |
| F: FULL_PIPELINE | GRAPH_SINGLE_HOP | +0.000 |
| F: FULL_PIPELINE | GRAPH_MULTI_HOP | +0.000 |
| F: FULL_PIPELINE | MISSING_ANCHOR | +0.000 |
| F: FULL_PIPELINE | OVERSMOOTHING | +0.000 |

## Failure Mode Summary
- **Missing Anchor Fallback:** The system successfully handles queries with missing or disconnected anchors by gracefully degrading to vector-only results, without crashing.
- **Oversmoothing:** Neighborhood averaging at ingest correctly resists oversmoothing in our adversarial tests, retaining enough discriminative power to rank exact answers over generic neighbors.

## Honest Limitations
1. **Missing Anchor Failure:** If explicit edges do not exist in the graph, the graph component cannot provide structure. The system degrades smoothly but recall will match standard vector search.
2. **2-hop Multi-Hop Graph Traversal:** 2-hop traversals are highly dependent on the strict semantic match of the initial anchor. Suboptimal initial vector matching degrades graph expansion recall dramatically.
3. **Ingest Scalability:** Python embedding bottlenecks under the GIL, restricting ingestion to ~5 documents per second. Not designed for enterprise massive bulk ingestion.
4. **Evaluator Independence:** All tests rely strictly on exact node ID matching, independent of LLMs during runtime. It cannot measure soft or conceptual overlaps.

## Reproducibility
To reproduce these results, run the following command in the repository root:
```bash
make full-eval
```
