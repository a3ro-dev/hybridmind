# HybridMind KV-Cache Reduction Research

## Objective

Determine whether HybridMind can reduce the prompt-side KV working set for
long-horizon memory without losing the evidence required for correct answers.
External retrieval is not itself a KV-cache replacement: it must first show a
competitive quality, memory, latency, and throughput frontier against full
context and cache-compression baselines.

## Current Architecture

HybridMind currently functions primarily as persistent external retrieval:

- Conversation turns and extracted facts are stored as SQLite nodes with dense
  embeddings, a BM25 index, and optional graph edges.
- Hybrid search generates dense and lexical candidates, expands from graph
  anchors, fuses ranked signals with RRF, and optionally applies a cross-encoder.
- Session consolidation and importance pruning exist, but they are offline
  lifecycle operations rather than a learned working-memory policy.
- The answering model receives retrieved text. HybridMind does not modify,
  evict, quantize, page, or refresh the model's internal KV cache.

The system is therefore an external persistent-memory layer, not yet intrinsic
working memory or a technically defensible KV-cache replacement.

## Ranked Blockers

1. **No direct quality-versus-KV measurement.** Existing evaluations report
   retrieval or answer metrics but do not compare full-history token/KV cost
   with retrieved-context quality.
2. **Unsupported benchmark claims.** The full checked-in LoCoMo ledger is a
   retrieval-only run. It contains no answer accuracy and yields Hit@10 far
   below the headline report.
3. **Weak retrieval ground truth.** The ledger's `gold_rank_post_rerank` is
   based on answer-text overlap, not exact LoCoMo evidence identifiers.
4. **No full-context or cache-compression baseline.** HybridMind is not compared
   against uncompressed attention, SnapKV-style eviction, or model-managed
   cache policies on identical prompts.
5. **Graph scale and provenance risk.** The current store has far more edges
   than nodes, while checked-in graph ablations show no demonstrated gain.
6. **Serving metrics are incomplete.** Retrieval latency is measured separately
   from end-to-end prefill, decode, peak memory, and throughput.
7. **Lifecycle policies are heuristic.** Consolidation, contradiction handling,
   and pruning are not trained or validated against downstream forgetting.

## Literature Principles

- [PagedAttention](https://arxiv.org/abs/2309.06180) reduces allocation waste and
  enables cache sharing, but it preserves the logical cache rather than deciding
  what long-term information should be forgotten.
- [SnapKV](https://arxiv.org/abs/2404.14469) demonstrates that head-specific KV
  selection can materially reduce memory while retaining long-context quality.
- [Infini-attention](https://arxiv.org/abs/2404.07143) integrates bounded
  compressive memory into attention, which is architecturally closer to cache
  replacement than an external database.
- [MemGPT](https://arxiv.org/abs/2310.08560) frames external memory as tiered
  virtual context, but requires explicit memory movement and control policy.
- [LongMemEval](https://arxiv.org/abs/2410.10813) separates indexing, retrieval,
  and reading; HybridMind must measure all three rather than retrieval alone.
- [SideQuest](https://arxiv.org/abs/2602.22603) reports up to 65% peak-token
  reduction on long-horizon agentic tasks using model-driven cache management.
- [MSA](https://arxiv.org/abs/2603.23516) and
  [PReM](https://arxiv.org/abs/2607.14327) indicate that the strongest current
  direction is adaptive, end-to-end memory selection and refresh rather than a
  fixed one-shot retrieval decision.

## Hypothesis 1

> At `k=10`, current HybridMind retrieval preserves answer-bearing context for
> at least 80% of LoCoMo questions with gold evidence while reducing
> memory-context tokens, and therefore proportional KV-cache bytes, by at least
> 90% versus full-history prompting.

The smallest experiment is `benchmarks/kv_reduction_eval.py`. It joins the
existing LoCoMo ledger to the original annotated conversations and resolves
retrieved node text from SQLite. It can also stream a self-contained MemoryBench
checkpoint. Historical checkpoints are normalized back to relevance order by
`combined_score` because the former provider sorted results chronologically
before persistence. The benchmark reports:

- full-history and retrieved-context token counts;
- proportional prompt-side KV reduction;
- the existing answer-overlap proxy Hit@k;
- exact annotated-source recall, any-hit, and all-hit rates;
- input hashes, node-resolution coverage, bootstrap confidence intervals, and
  explicit limitations.

Absolute KV bytes require explicit model architecture parameters because the
hosted GLM-4.6 KV layout is not documented in this repository. For a standard
cache, the benchmark uses:

```text
bytes_per_token = 2 * layers * kv_heads * head_dim * element_bytes
```

Run:

```powershell
.\.venv\Scripts\python.exe benchmarks\kv_reduction_eval.py `
  --checkpoint memorybench\data\runs\hybridmind-locomo-fixed-20260726\checkpoint.json `
  --k-values 1,3,5,10,25,50,100
```

## Hypothesis 1 Result

**Result: failed.** The self-contained checkpoint contains 755 completed
searches, 1 failed search, and 1,230 pending searches. Answering and judging are
pending for every question, so this experiment measures retrieval evidence, not
downstream QA accuracy.

| k | Mean context/KV reduction | Answer-overlap proxy hit | Exact source recall |
|---:|---:|---:|---:|
| 1 | 99.90% | 2.79% | 2.53% |
| 3 | 99.69% | 5.98% | 6.29% |
| 5 | 99.47% | 8.91% | 11.09% |
| 10 | **98.93%** | **11.97%** | **15.71%** |
| 25 | 97.26% | 19.28% | 29.71% |
| 50 | 94.36% | 24.60% | 42.82% |
| 100 | 88.41% | 30.32% | 55.78% |

At `k=10`, mean context reduction is 98.93% with a bootstrap 95% CI of
98.90%-98.97%, but the answer-overlap proxy is only 11.97% with a 95% CI of
9.71%-14.49%. Exact annotated-source recall is 15.71%. The memory-reduction
gate passes and the quality gate fails by a large margin.

The failure persists beyond reranking depth. At `k=100`, exact source recall is
only 55.78% and context reduction has already fallen to 88.41%, below the 90%
target. Candidate generation and memory representation are therefore major
blockers; reranking alone cannot make the current system a defensible KV-cache
substitute.

The checkpoint reports mean search latency of 14.18 seconds and p95 latency of
29.74 seconds. These values are not competitive serving results and may include
remote embedding, query decomposition, and serverless cold-start effects. They
must be decomposed in a live preflight-verified run.

The reproducible result is stored in
`benchmarks/results/kv_reduction_locomo_checkpoint.json`.

## Hypothesis 2

> At `k=10`, a 50/50 reciprocal-rank fusion of the existing candidate order
> and a query-local lexical order improves exact annotated-source recall by at
> least 5 percentage points.

The query-local lexical score uses query-term inverse document frequency within
a bounded candidate pool and square-root document-length normalization. Its
rank is fused with the existing rank using RRF (`k=60`). The production stage
runs before cross-encoder truncation so that the neural reranker receives the
stronger candidate subset. The benchmark replays the same operation over the
stored 500-candidate checkpoint pool.

Run:

```powershell
.\.venv\Scripts\python.exe benchmarks\kv_reduction_eval.py `
  --checkpoint memorybench\data\runs\hybridmind-locomo-fixed-20260726\checkpoint.json `
  --checkpoint-ranking local-lexical-rrf `
  --local-lexical-pool-size 500 `
  --local-lexical-weight 0.5 `
  --k-values 1,3,5,10,25,50,100 `
  --output benchmarks\results\kv_reduction_locomo_lexical_rrf.json
```

## Hypothesis 2 Result

**Retrospective threshold result: passed. Independent confirmation: pending.**

| k | Baseline exact-source recall | Lexical-RRF recall | Difference |
|---:|---:|---:|---:|
| 1 | 2.53% | 5.85% | +3.32 pp |
| 3 | 6.29% | 17.33% | +11.04 pp |
| 5 | 11.09% | 25.82% | +14.73 pp |
| 10 | **15.71%** | **33.77%** | **+18.05 pp** |
| 25 | 29.71% | 45.58% | +15.88 pp |
| 50 | 42.82% | 54.94% | +12.12 pp |
| 100 | 55.78% | 64.32% | +8.54 pp |

At `k=10`, the paired mean improvement is 18.05 percentage points with a
bootstrap 95% CI of 15.27-20.85 points. Of 752 evidence-bearing questions, 154
improve, 6 regress, and 592 are unchanged. The answer-overlap proxy rises from
11.97% to 21.01%. Mean retrieved context falls from 290 to 231 tokens, so mean
prompt-side context reduction is 99.14%.

The offline lexical stage costs 59.31 ms mean and 73.63 ms p95 for the
500-candidate replay. This is small relative to the historical 14.18 second
search latency but is not negligible for a low-latency serving target. Live
candidate counts may be lower than 500, and must be measured separately.

The feature remains configuration-gated and is disabled by default. The
historical effect was selected and measured on the same partial checkpoint, so
it cannot justify a production default or a confirmatory scientific claim.

The result artifact is
`benchmarks/results/kv_reduction_locomo_lexical_rrf.json`.

## Critical Review

- The checkpoint is partial and predates current uncommitted provider/ranker
  fixes. It is useful as a baseline, not as the final system score.
- Historical provider code sorted checkpointed results chronologically. The
  benchmark reconstructs relevance order from persisted `combined_score` and
  records this normalization in the result artifact.
- `BAAI/bge-m3` tokenization is a fixed token-count proxy, not GLM-4.6's exact
  tokenizer. Percentage reduction is expected to be robust, but absolute token
  counts are model-dependent.
- Exact source recall does not credit extracted facts that paraphrase evidence.
  The answer-overlap proxy provides that complementary but weaker signal.
- No answer accuracy, accelerator memory, time to first token, or decode
  throughput is available from this checkpoint.
- Hypothesis 2 was selected after exploratory analysis of the same 755-search
  checkpoint. Its paired confidence interval quantifies the observed effect but
  does not remove selection bias. The 1,230 pending searches or another dataset
  are required for independent confirmation.
- The checkpoint contains no non-null cross-encoder scores. The lexical result
  validates pre-cross-encoder candidate ordering, not the final interaction
  with `mixedbread-ai/mxbai-rerank-large-v2`.

## Required Next Baselines

The next confirmatory experiment should use a preregistered held-out LoCoMo
split with lexical reranking disabled and enabled, then compare exact-evidence
retrieval and explicitly identified reader outcomes. The current deterministic
overlap heuristic must not be called a Z.AI or official LoCoMo judge. After
that gate, compare identical
questions under full history, HybridMind retrieval, and a supported
cache-compression implementation. Record answer accuracy, prompt tokens, peak
accelerator memory, time to first token, decode throughput, and total latency.
RunPod TEI and vLLM must pass the default-deny,
`scripts/preflight.py --plan <validated-live-plan.json>` gate before any live
run. The plan must select only required providers and include preflight usage.
