# PHASE 6 — REALISTIC: Loss-Ledger-Driven Optimization

**Status:** Active & Implemented (Parts A, B, C Completed)  
**Prime directive:** Every intervention in this phase is justified by a *measured* loss, executed with a *deterministic* procedure, and accepted or rejected by a *pre-registered* statistical gate.

---

## 0. Current Implementation State

The Phase 6 prep punch list is fully completed across Parts A, B, and C:

| Sub-Phase | Component | Status | Evidence |
|-----------|-----------|--------|----------|
| **Part A** | Single source reranker truth | ✅ **COMPLETED** | `engine/reranker.py` loads strictly from `settings.reranker_model` |
| **Part A** | Lifespan dimension assertion | ✅ **COMPLETED** | `main.py` lifespan fails fast on FAISS/embedding dimension mismatch |
| **Part A** | Scaffolding documentation | ✅ **COMPLETED** | `README.md` & `PHASE_IMPLEMENTATION_STATUS.md` updated to clarify untrained scaffolds |
| **Part B** | Measurement Ledger | ✅ **COMPLETED** | `eval_ledger.py` emits per-question JSONL ledgers |
| **Part B** | Bootstrap CI & Permutation Test | ✅ **COMPLETED** | `eval_stats.py` provides bootstrap 95% CIs and paired permutation tests |
| **Part C** | Answer Normalization & Prompting | ✅ **COMPLETED** | `eval_common.py` citation prompt, iterative multihop, and `normalize_answer()` |
| **Part C** | Query Decomposition Module | ✅ **COMPLETED** | `engine/query_decomposition.py` multi-hop sub-question generation |
| **Part C** | Candidate Pool Size Knob | ✅ **COMPLETED** | `config.rerank_pool_size` controls rerank candidate depth |
| **Part C** | Auto-Edge Threshold Sweeper | ✅ **COMPLETED** | `scripts/sweep_edge_threshold.py` auto-edge reachability tool |
| **Part D** | Per-query-type Fusion LR | ⚠️ **OPTIONAL** | Gated by project budget; unstarted |

---

## 1. The Loss Ledger Architecture

End-to-end accuracy accounting:

$$\text{Accuracy} = P(\text{gold evidence retrieved}) \times P(\text{correct answer} \mid \text{evidence retrieved})$$

### Loss Pools
1. **L1 — Reading Loss**: Reduced via `normalize_answer()`, explicit citation prompting (`[Citation: <node_id>]`), and iterative multi-hop execution in `eval_common.py`.
2. **L2 — Retrieval Loss**: Addressed via `engine/query_decomposition.py` (decomposing multi-step questions into sub-questions) and auto-edge threshold sweeping.
3. **L3 — Fusion Loss**: Controlled via `config.rerank_pool_size` and RRF $k=60$ fusion with query-type routing.

---

## 2. Measurement Harness Usage

### 1. Emit Evaluation Ledger
```bash
python eval_locomo_retrieval.py --with-answers --decompose-multihop
```
Emits: `benchmarks/results/ledger_locomo_<confighash>.jsonl`

### 2. Statistical Significance Testing
```bash
python eval_stats.py compare benchmarks/results/ledger_locomo_<hashA>.jsonl benchmarks/results/ledger_locomo_<hashB>.jsonl
```
Outputs bootstrap 95% confidence intervals and paired permutation p-values.

### 3. Sweep Auto-Edge Thresholds
```bash
python scripts/sweep_edge_threshold.py
```
Sweeps cosine similarity thresholds for auto-edge creation against 2-hop graph reachability.
