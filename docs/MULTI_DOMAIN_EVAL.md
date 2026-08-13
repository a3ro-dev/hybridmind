# Multi-Domain Evaluation Framework

This document describes the multi-domain evaluation harness for HybridMind across **LoCoMo**, **LongMemEval**, and **MuSiQue**.

---

## 1. Overview

The evaluation suite tests HybridMind across three key benchmark datasets:
- **LoCoMo**: Long-conversation agent memory benchmark.
- **LongMemEval**: Temporal and multi-session memory benchmark.
- **MuSiQue**: Multi-hop reasoning dataset for complex graph traversal.

---

## 2. Measurement Ledger & Reproducibility

Evaluations produce immutable, provenance-bearing ledgers via `eval_ledger.py`.
Question ordering and seeded statistical operations can be deterministic, but
files containing timestamps, provider responses, or remote-model outputs are
not claimed to be bit-identical:
- **Artifact Path**: `benchmarks/results/ledger_<benchmark>_<confighash>.jsonl`
- **Recorded Fields**:
  - `question_id` & `question_text`
  - `question_type` (single-hop, multi-hop, temporal, etc.)
  - `retrieved_nodes`: Top-k candidate UUIDs
  - `gold_nodes`: Ground-truth evidence UUIDs
  - `raw_answer`: Answering LLM output string
  - `normalized_answer`: Result of `normalize_answer()`
  - `verdict`: Boolean correctness judgment

---

## 3. Statistical Significance Tool (`eval_stats.py`)

Run A/B statistical comparisons between two evaluation runs:

```bash
python eval_stats.py compare benchmarks/results/ledger_locomo_<hashA>.jsonl benchmarks/results/ledger_locomo_<hashB>.jsonl
```

### Metrics Output
- **Bootstrap 95% Confidence Interval**: Re-samples candidate ledgers ($N=10,000$) to calculate error bounds.
- **Paired Permutation Test**: Computes non-parametric p-value to test if delta accuracy is statistically significant ($p < 0.05$).

---

## 4. Execution Commands

```bash
# LoCoMo Benchmark
python eval_locomo_retrieval.py --with-answers --decompose-multihop

# LongMemEval Benchmark
python eval_longmemeval_retrieval.py --with-answers

# MuSiQue Benchmark
python eval_musique_retrieval.py --with-answers --decompose-multihop
```
