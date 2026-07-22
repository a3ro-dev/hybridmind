# Algorithmic Foundations of HybridMind Retrieval

## Abstract
HybridMind combines **Tri-Signal Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, **Multi-Hop Query Decomposition**, and **Ingest-Time Auto-Edges** into a single unified retrieval system for AI memory. This document details the mathematical formulations and algorithms behind each stage of the pipeline.

---

## 1. Tri-Signal Reciprocal Rank Fusion (RRF)

### 1.1 Formula
RRF combines ranked candidate lists from multiple distinct retrieval signals without requiring cross-signal score calibration:

$$RRF(d) = \sum_{s \in \{dense, sparse, graph\}} w_s \cdot \frac{1}{k + r_s(d)}$$

Where:
- $k = 60$ (standard RRF smoothing constant)
- $r_s(d)$ is the 1-based rank position of document/node $d$ within retrieval signal $s$
- $w_s$ is the signal weight for signal $s$:
  - $w_{dense}$: Dense vector similarity rank weight (`vector_weight`)
  - $w_{sparse}$: BM25 lexical rank weight (`bm25_boost_weight`)
  - $w_{graph}$: Graph proximity rank weight (`graph_weight`)

### 1.2 Query-Type Weight Routing
Signal weights are dynamically set per-query via `route_query()`:
- **Default Queries**: `vector_weight=0.5, graph_weight=0.15, bm25_boost_weight=0.35`
- **Multi-Hop / Structural Queries**: `vector_weight=0.2, graph_weight=0.60, bm25_boost_weight=0.20`
- **Keyword / Exact Matches**: `vector_weight=0.2, graph_weight=0.10, bm25_boost_weight=0.70`

---

## 2. Pre-trained Cross-Encoder Reranking

### 2.1 Candidate Selection & Reranking
The top $N$ candidates (configured by `config.rerank_pool_size`, default 25) output by RRF fusion are passed to `mixedbread-ai/mxbai-rerank-large-v2`.

### 2.2 Normalized Score Blending
To prevent text-only cross-encoders from discarding Graph-discovered candidates that lack direct query term matches, both RRF and Cross-Encoder scores are independently min-max normalized to $[0, 1]$ before linear blending:

$$Norm(S) = \frac{S - \min(S)}{\max(S) - \min(S) + \epsilon}$$

$$Score_{final}(d) = 0.70 \cdot Norm(RRF(d)) + 0.30 \cdot Norm(Score_{reranker}(d))$$

---

## 3. Multi-Hop Query Decomposition

Multi-hop questions are decomposed into targeted sub-questions using `engine/query_decomposition.py`:

```
FUNCTION decompose_multihop_query(query_text, llm_engine):
    prompt = BUILD_DECOMPOSITION_PROMPT(query_text)
    sub_questions = llm_engine.generate_json(prompt, schema=DECOMP_SCHEMA)
    
    # Guard 1: Single Sub-Question Guard
    IF LENGTH(sub_questions) <= 1:
        RETURN [query_text]  # Retain original query
        
    # Guard 2: Novel Entity Guard
    FOR EACH sq IN sub_questions:
        IF sq CONTAINS entities NOT IN query_text:
            FILTER sq OUT
            
    RETURN sub_questions
```

Sub-questions are retrieved sequentially, accumulating candidate nodes into a unified context pool before final QA execution.

---

## 4. Answer Normalization & Citation Prompting

### 4.1 Citation Prompting
Answering prompts require the LLM to output explicit node citations:
`[Citation: <node_id>] Fact description... Answer: <exact_fact>`

### 4.2 Answer Normalization (`normalize_answer()`)
Before evaluation scoring, candidate answers undergo deterministic normalization:
- Lowercase conversion and punctuation removal
- Strip leading/trailing articles (*a*, *an*, *the*)
- Whitespace collapse

---

## 5. Auto-Edge Inference & Threshold Sweeping

### 5.1 Auto-Edge Cosine Thresholding
During node ingestion (`HYBRIDMIND_AUTO_EDGES_ENABLED=true`), cosine similarity edges (`similar_to`) are automatically inferred for vector pairs satisfying:

$$\text{cos}(\mathbf{e}_i, \mathbf{e}_j) \ge \tau_{auto} \quad (\text{default } \tau_{auto} = 0.75)$$

### 5.2 Reachability Sweeping
`scripts/sweep_edge_threshold.py` evaluates auto-edge reachability by sweeping $\tau_{auto} \in [0.60, 0.90]$ and measuring:
1. Total edge count added to graph
2. 2-hop graph path reachability between multi-hop entity pairs
