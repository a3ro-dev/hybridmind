# Evaluation harness guide

How to run, record, and compare HybridMind evaluations. This document is the
operational how-to; the *rules that make a result admissible* live in
`docs/RETRIEVAL_RESEARCH_PROTOCOL.md`, and the *current inventory of valid
results* lives in `benchmarks/results/BENCHMARK_REPORT.md`.

## 1. Retrieval evaluators

Three benchmark drivers share one contract: exact evidence IDs,
corpus/session scoping, per-question ledger rows (failures included), and
immutable run manifests.

```bash
python eval_locomo_retrieval.py --with-answers            # LoCoMo long conversations
python eval_longmemeval_retrieval.py --with-answers       # LongMemEval sessions
python eval_musique_retrieval.py --with-answers --decompose-multihop
```

Common controls: `--vector-weight`, `--graph-weight`, `--bm25-boost`,
`--top-k`, `--search-mode` (`vector_only`, `sparse_only`, `vector_sparse`,
`graph_only`, `hybrid`), `--rerank-pool`, `--with-answers`, `--decompose-multihop`.
A signal-ablation invocation must also disable every other stage explicitly
(e.g. `--rerank-pool 0 --no-route-weights --no-track-access`); graph-only
additionally requires explicit, gold-independent anchors via
`--anchor-node-id`. A positive rerank pool must produce execution evidence in
the response trace; pool `0` means the reranker is off.

These drivers require a running API. Offline (zero-provider) baselines are
separate scripts under `scripts/offline_*` — see
`experiments/reports/baseline.md`.

## 2. Ledgers

Every run appends one JSONL row per question to
`benchmarks/results/ledger_<benchmark>_<confighash>.jsonl`
(`eval_ledger.py`). Row schema highlights:

- identity: `schema`, `run_id`, `manifest_sha256`, `config_hash`, `seed`,
  `sequence`, `question_id`, `question_type`;
- retrieval: `gold_evidence_ids`, `retrieved_ids_at_k`, `metric_basis`
  (`exact_evidence_id`), `gold_in_pool_pre_rerank`,
  `gold_rank_pre_rerank`, `gold_rank_post_rerank`, `hit_at_k`;
- answering: `raw_llm_answer`, `answer_status`, `judged_correct`,
  `judge_method`, `judge_rationale`, `prompt_version`;
- failure accounting: `status` (`completed`/failed variants), `error_type`,
  `error_message`, plus budget provenance.

Duplicate `question_id`s are rejected; a run is sealed by `finalize()` with a
checksum and row count, producing completed or failed receipts. Failed
questions remain rows in the denominators; they are never skipped.

The deterministic answer judge is lexical/normalized string matching — it is
**not** an LLM judge, and answer-string overlap is **not** evidence recall.

## 3. Answer prompting conventions

`eval_common.py` version answer prompts so a changed prompt cannot silently
mix into an existing artifact: `qa_v1` (single-shot),
`qa_citation_v1` (citation-then-answer), `qa_multihop_v1` (iterative
evidence-then-conclude). The active `prompt_version` is recorded per row.
Changing prompt text requires bumping the version constant.

Loss decomposition when reading results:

- **L1 reading loss** — answer normalization (`normalize_answer()`:
  lowercase, strip punctuation/articles, collapse whitespace) and citation
  prompting address it.
- **L2 retrieval loss** — candidate generation; decomposition and sparse-key
  experiments target it. Rerankers only reorder what the pool already
  contains, so report candidate-pool oracle recall alongside final recall.
- **L3 fusion loss** — weight/routing/fusion-stage effects, measured with
  equal candidate budgets across arms.

## 4. Statistical comparison

```bash
python eval_stats.py ci      <ledger.jsonl>                 # bootstrap 95% CI
python eval_stats.py compare <ledger_A.jsonl> <ledger_B.jsonl>
```

`compare` reports a bootstrap 95% confidence interval (10,000 resamples) and a
paired permutation test. Treat any apparent gain under ~2.5 points as noise
even when p < 0.05: ledgers this small regularly produce smaller spurious
deltas. Selection on development data and confirmation on held-out data must
remain separate runs; see the protocol for the full statistical rules.

## 5. Ablation planning

`scripts/ablation_matrix.py --list` / `--dry-run` plan condition matrices
without network calls. Plan-only output is not an experiment: a completed
condition needs request attestation plus external server/corpus attestation.
Graph-only conditions stay plan-only until a gold-independent anchor manifest
exists.

## Legacy note

The retired AG-news "multi-domain" harness era (weight/density sweeps writing
into this file) is gone: `scripts/multi_domain_eval.py` survives only as a
quarantined plan-only stub whose live execution is rejected by
`tests/test_eval_benchmark_integrity.py`. Its historical outputs
(`benchmarks/multi_domain_results*.json`) predate evidence-ID metrics and are
not comparable with anything above.
