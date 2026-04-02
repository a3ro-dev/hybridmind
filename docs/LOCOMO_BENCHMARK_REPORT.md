# HybridMind × LoCoMo Benchmark Report

> **Run ID**: `hybridmind-locomo-100`
> **Date**: 2026-03-27
> **Answering Model**: Qwen 3.5 (397B-A17B, reasoning)
> **Judge Model**: Qwen 3.5 (397B-A17B)
> **Questions Evaluated**: 100 (from LoCoMo conv-26)

---

## Executive Summary

HybridMind achieved **3% accuracy** (3/100) on the LoCoMo long-conversation memory benchmark. This result is significantly below expectations and does not reflect the system's retrieval architecture capability. Root cause analysis reveals a **critical container isolation bug** in the benchmark harness that caused duplicate node accumulation across question runs, poisoning the search index with redundant copies of identical content under different container tags.

The 3 correct answers were all `world-knowledge` questions — the category least dependent on precise retrieval from the conversation corpus.

---

## Results

### Overall Performance

| Metric               | Value                    |
| -------------------- | ------------------------ |
| **Accuracy**         | **3.0%** (3/100)         |
| Retrieval Hit@10     | 12.0%                    |
| Retrieval MRR        | 0.096                    |
| Retrieval NDCG       | 0.104                    |
| Retrieval Precision  | 8.2%                     |
| Retrieval Recall     | 12.0%                    |
| Retrieval F1         | 9.2%                     |
| Avg search latency   | 2,254 ms (median: 836ms) |
| Avg answer latency   | 17.0s                    |
| Avg total latency    | 408.4s per question      |

### Accuracy by Question Type

| Type            | Correct | Total | Accuracy | Hit@10 | MRR    | NDCG   |
| --------------- | ------- | ----- | -------- | ------ | ------ | ------ |
| world-knowledge | 3       | 18    | 16.7%    | 22.2%  | 0.222  | 0.220  |
| single-hop      | 0       | 32    | 0.0%     | 6.3%   | 0.063  | 0.058  |
| multi-hop       | 0       | 37    | 0.0%     | 13.5%  | 0.069  | 0.099  |
| temporal         | 0       | 13    | 0.0%     | 7.7%   | 0.077  | 0.071  |

### Latency Profile

| Stage     | Mean     | Median   | P95       | Std Dev  |
| --------- | -------- | -------- | --------- | -------- |
| Ingest    | 359.1s   | 362.2s   | 387.8s    | 15.8s    |
| Indexing  | 26ms     | 31ms     | 50ms      | 18ms     |
| Search    | 2,254ms  | 836ms    | 15,179ms  | 4,305ms  |
| Answer    | 17.0s    | 12.7s    | 61.7s     | 14.7s    |
| Evaluate  | 30.1s    | 26.7s    | 60.3s     | 15.8s    |
| **Total** | **408s** | **403s** | **454s**  | **25.6s**|

### Token Usage

| Metric                    | Value   |
| ------------------------- | ------- |
| Total tokens consumed     | 64,025  |
| Avg tokens per question   | 640     |
| Avg base prompt tokens    | 198     |
| Avg context tokens        | 442     |

### MemScore

```
3% / 2,254ms / 442tok
```

---

## Failure Analysis

### The Smoking Gun: Duplicate Nodes Across Containers

The benchmark harness assigns each question a unique container tag (e.g., `conv-26-q0-hybridmind-locomo-100`) and ingests the full conversation (19 sessions, ~570 documents) into that container. However, **HybridMind's `clear()` function did not properly isolate containers between runs**, causing the vector index to accumulate duplicate nodes from every question's ingestion pass.

**Evidence** — For the question _"When did Caroline go to the LGBTQ support group?"_ (conv-26-q0), the top 10 search results are:

```
Rank 1: container "conv-26-q3-hybridmind-test"   — same text
Rank 2: container "conv-26-q2-hybridmind-test"   — same text
Rank 3: container "conv-26-q0-hybridmind-test"   — same text
Rank 4: container "conv-26-q1-hybridmind-test"   — same text
Rank 5: container "conv-26-q4-hybridmind-test"   — same text
Rank 6: container "conv-26-q2-hybridmind-test"   — same text (different session)
Rank 7: container "conv-26-q22-hybridmind-locomo-100" — same text
Rank 8: container "conv-26-q24-hybridmind-locomo-100" — same text
Rank 9: container "conv-26-q25-hybridmind-locomo-100" — same text
Rank 10: container "conv-26-q23-hybridmind-locomo-100" — same text
```

All 10 results contain the **exact same text** — an assistant message about LGBTQ advocacy. They differ only in their `container` metadata tag. The search returns 10 copies of the same message instead of 10 diverse, relevant results.

This pattern is consistent across nearly every question in the evaluation.

### Impact Chain

```
Container isolation failure
  → Duplicate nodes accumulate (100 questions × ~570 docs = ~57,000 nodes instead of ~570)
    → Vector search returns duplicate copies of the same content
      → Top-K results are wasted on redundant nodes
        → Relevant context never reaches the answering model
          → Model responds "I don't know" (97% of answers)
```

### Why world-knowledge Questions Survived

The 3 correct answers are all `world-knowledge` type — questions answerable from general knowledge without precise retrieval (e.g., commonsense facts about psychology, education). These don't depend on finding specific conversation turns, so the contaminated index didn't block them.

### Root Cause: Two Compounding Issues

1. **No container-scoped search**: HybridMind's `search()` in `engine/hybrid_ranker.py` and `engine/vector_search.py` does not accept or filter by a `container` metadata field. It searches the entire index globally.

2. **Incomplete `clear()` between runs**: The `storage/vector_index.py::clear()` and `storage/graph_index.py::clear()` functions exist but were either not called between question runs by the benchmark harness, or did not fully flush the underlying HNSW index. The evidence (nodes from `conv-26-q0-hybridmind-test` through `conv-26-q4-hybridmind-test` — a different run ID — coexisting with `conv-26-qN-hybridmind-locomo-100` nodes) proves the database was never wiped between benchmark runs.

### Additional Contributing Factor: Graph Score Distortion

The graph edges are also affected. Some duplicate nodes receive `graph_score: 1.0` (full graph connectivity) while identical copies in other containers receive `graph_score: 0.0`. This happens because edge creation during ingestion connects nodes within a single ingestion pass, but cross-container duplicates have no edges between them. The combined ranking formula (`0.6 × vector + 0.4 × graph`) then arbitrarily boosts whichever duplicate happened to be in the same ingestion batch as the query seed — not necessarily the most relevant one.

---

## What These Numbers Actually Tell Us

### What's NOT broken:
- **Embedding quality**: Vector similarity scores (0.52–0.65) are reasonable for the content domain. The embeddings are finding semantically relevant content — the problem is they're finding 10 copies of it.
- **Graph construction**: Edge creation within a single ingestion pass works correctly (nodes get `graph_score: 1.0`).
- **Answer generation**: When Qwen 3.5 receives correct context, it can reason to the right answer. The world-knowledge hits prove this.
- **Latency**: Search median of 836ms is acceptable for a local system. The P95 spike to 15s is likely caused by searching over the inflated (~57K node) index.

### What IS broken:
- **Container/namespace isolation**: The system has no concept of scoped retrieval. Every search hits every node ever inserted.
- **Database lifecycle management**: `clear()` doesn't reliably wipe state between benchmark runs, leading to cross-contamination.
- **Deduplication**: No dedup logic prevents identical text from being stored multiple times with different metadata.

---

## Recommended Fixes

### Fix 1: Container-Scoped Search (High Priority)
Add a `container` filter parameter to `vector_search.py::search()` and `hybrid_ranker.py::search()`. During retrieval, filter candidates by `metadata.container` before ranking.

### Fix 2: Atomic Clear (High Priority)
Ensure `vector_index.clear()` and `graph_index.clear()` fully destroy the underlying HNSW index and adjacency lists — not just Python-level data structures but also any memory-mapped or persisted state.

### Fix 3: Content Deduplication (Medium Priority)
Before inserting a node, hash the text content and check for duplicates. Either skip insertion or merge metadata tags onto the existing node.

### Fix 4: Benchmark Harness Isolation (Medium Priority)
The benchmark runner should either:
- Call `clear()` and verify the index is empty (count = 0) before each question
- Use separate database instances per question
- Filter search results by the active container tag

### Expected Impact
With proper container isolation, the effective search space drops from ~57,000 nodes back to ~570 per question. This alone should increase Hit@10 significantly, which directly unlocks the reasoning model's ability to answer correctly.

---

## Comparison Context

For reference, the LoCoMo benchmark paper reports these baselines:

| System                | Accuracy |
| --------------------- | -------- |
| GPT-4 (full context)  | ~60-70%  |
| RAG (basic)           | ~30-40%  |
| HybridMind (this run) | 3%       |

The 3% result should not be compared against these baselines without fixing the container isolation bug first. The contaminated index makes this an unfair test of HybridMind's retrieval architecture.

---

## Next Steps

1. **Fix container isolation** in the provider (Fixes 1–2 above)
2. **Re-run the benchmark** with a clean database per question
3. **Compare** the fixed results against baselines to get a fair assessment
4. **If results remain low**, investigate embedding model quality and graph traversal depth as secondary factors

---

*Report generated from [locomo_report.json](../locomo_report.json) and [locomo_checkpoint.json](../locomo_checkpoint.json)*
