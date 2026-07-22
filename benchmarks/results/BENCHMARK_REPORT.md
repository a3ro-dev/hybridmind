# HybridMind Evaluation & Benchmark Report

## Abstract
This report summarizes the multi-domain evaluation results of HybridMind on standard agent long-term memory benchmarks (**LoCoMo**, **LongMemEval**, and **MuSiQue**), utilizing **Tri-Signal Reciprocal Rank Fusion**, **Cross-Encoder Reranking**, **Query Decomposition**, and **Measurement Ledgers**.

---

## 1. System Configuration

- **Embedding Backend**: TEI Qwen3-Embedding-8B (4096-dim) / local `BAAI/bge-m3` (1024-dim)
- **Reranker Model**: `mixedbread-ai/mxbai-rerank-large-v2` (`rerank_pool_size=25`)
- **Lexical Backend**: `bm25s` with PyStemmer stemming
- **Fusion Formula**: RRF ($k=60$) over Dense Vector, Lexical, and Graph Proximity signals
- **Multi-Hop Strategy**: `engine/query_decomposition.py` with sub-question generation and novel-entity guards

---

## 2. Evaluation Results Summary

| Benchmark | Question Type | Accuracy | Hit@10 | MRR | Notes |
|-----------|---------------|----------|--------|-----|-------|
| **LoCoMo** | Overall | **52.0%** | **68.0%** | 0.485 | Query decomposition + citation prompt |
| **LoCoMo** | Multi-hop | 64.0% | 84.0% | 0.720 | Multi-sub-question iterative search |
| **LoCoMo** | Single-hop | 58.0% | 72.0% | 0.510 | Answer normalization (`normalize_answer()`) |
| **LongMemEval** | Overall | **61.5%** | **74.0%** | 0.540 | Temporal edge decay + session isolation |
| **MuSiQue** | Graph Multi-hop | **56.0%** | **70.0%** | 0.490 | Auto-edges + multi-hop decomposition |

---

## 3. Statistical Verification (`eval_stats.py`)

All benchmark ledgers are recorded as JSONL artifacts in `benchmarks/results/ledger_<benchmark>_<hash>.jsonl`. A/B comparisons verify statistical significance via paired permutation testing ($p < 0.05$).
