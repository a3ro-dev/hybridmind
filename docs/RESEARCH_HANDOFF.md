# HybridMind Research Handoff

Last updated: 2026-08-05 (Asia/Calcutta)

## Active objective

Demonstrate that HybridMind can substantially reduce or replace prompt-side
transformer KV-cache requirements for long-horizon memory while preserving
quality, latency, and throughput. Until full replacement is defensible, improve
HybridMind as an empirically supported external long-term memory system.

## Repository constraints

- Use `D:\hybridmind\.venv` for Python.
- Use Z.AI `glm-4.6` for canonical QA and judging.
- Run `python scripts/preflight.py` before any live TEI/vLLM evaluation.
- Do not restore HackClub proxy configuration.
- Preserve pre-existing user edits in:
  - `engine/fact_extractor.py`
  - `engine/hybrid_ranker.py`
  - `eval_common.py`
  - `main.py`
  - `memorybench/memorybench/src/providers/hybridmind/index.ts`
- Keep exactly one measurable hypothesis active per research iteration.

## Completed iteration: Hypothesis 1

Hypothesis: at `k=10`, HybridMind retains answer-bearing context for at least
80% of evidence-bearing LoCoMo questions while reducing prompt-side context and
proportional KV working set by at least 90%.

Result: failed.

- Source checkpoint: 755 completed searches, 1 failed, 1,230 pending.
- Baseline exact-source Recall@10: 15.71%.
- Answer-overlap proxy at 10: 11.97%.
- Mean context reduction at 10: 98.93%.
- Mean historical search latency: 14.18 s; p95: 29.74 s.
- Artifact: `benchmarks/results/kv_reduction_locomo_checkpoint.json`.

Conclusion: current HybridMind is persistent external retrieval, not a
defensible KV-cache replacement.

## Completed iteration: Hypothesis 2

Hypothesis: a 50/50 RRF of existing candidate rank and query-local lexical rank
improves exact-source Recall@10 by at least 5 percentage points.

Implementation:

- Added `engine/lexical_reranker.py`.
- Added configuration in `config.py`:
  - `local_lexical_rerank_enabled = True`
  - `local_lexical_rerank_weight = 0.5`
  - `local_lexical_rerank_pool_size = 500`
- Integrated lexical reranking before the cross-encoder pool in
  `engine/hybrid_ranker.py`.
- Extended `benchmarks/kv_reduction_eval.py` with checkpoint lexical replay,
  paired bootstrap comparison, and offline rerank latency.
- Added unit and integration coverage in `tests/test_lexical_reranker.py` and
  `tests/test_kv_reduction_eval.py`.

Formal retrospective result:

- Baseline exact-source Recall@10: 15.71%.
- Lexical-RRF exact-source Recall@10: 33.77%.
- Paired improvement: +18.05 percentage points.
- Bootstrap 95% CI for improvement: +15.27 to +20.85 points.
- Evidence-bearing questions: 752.
- Improved / regressed / unchanged: 154 / 6 / 592.
- Answer-overlap proxy: 11.97% to 21.01%.
- Mean offline rerank latency over 500 candidates: 59.31 ms.
- P95 offline rerank latency: 73.63 ms.
- Artifact: `benchmarks/results/kv_reduction_locomo_lexical_rrf.json`.
- Artifact SHA-256:
  `5382ccda509f1245e058c3962f0e24328db6cc61ec081962d05f90dbbfaf3df9`.

Interpretation: the effect clears the 5-point threshold on the reused partial
checkpoint. It is not independent confirmation because the same checkpoint
informed hypothesis selection. The checkpoint has no non-null cross-encoder
scores, so it measures pre-cross-encoder ordering only.

Verification completed:

```text
9 passed in 0.37s
python -m py_compile: passed
git diff --check: passed
```

Focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_lexical_reranker.py `
  tests\test_kv_reduction_eval.py -q
```

Formal benchmark command:

```powershell
.\.venv\Scripts\python.exe benchmarks\kv_reduction_eval.py `
  --checkpoint memorybench\data\runs\hybridmind-locomo-fixed-20260726\checkpoint.json `
  --checkpoint-ranking local-lexical-rrf `
  --local-lexical-pool-size 500 `
  --local-lexical-weight 0.5 `
  --k-values 1,3,5,10,25,50,100 `
  --output benchmarks\results\kv_reduction_locomo_lexical_rrf.json
```

## GPU environment status

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB VRAM.
- Driver: 595.97.
- Cached model: `mixedbread-ai/mxbai-rerank-large-v2` is present locally.
- Current `.venv` Torch: `2.11.0+cpu`; `torch.cuda.is_available()` is false.
- CUDA wheel `torch==2.11.0+cu128` exists on the official PyTorch index.
- Two pip install attempts timed out and left child processes; the exact pip
  processes were terminated. Torch remained unchanged and importable.
- Do not claim GPU results until both the Torch build tag and a real CUDA model
  forward pass are verified.

Latest preflight (2026-08-05):

- Z.AI GLM-4.6: OK, HTTP 200.
- RunPod vLLM decomposition: OK, HTTP 200, one worker ready.
- RunPod TEI embedding: DOWN, `ReadTimeout` after three cold-start attempts.
- Held-out evaluation is blocked until TEI warms or is repaired.

## Current limitations and next experiment

Highest-priority scientific gap: independent end-to-end validation of lexical
reranking, including the Mixedbread cross-encoder and Z.AI answer/judge stages.

Before a live run:

1. Make the `.venv` CUDA-capable without leaving orphaned pip processes.
2. Verify a CUDA tensor allocation and one cached cross-encoder forward pass.
3. Load repository `.env` for the evaluation process without printing secrets.
4. Run `scripts/preflight.py`; do not proceed if TEI/vLLM/Z.AI checks fail.
5. Use held-out or previously pending LoCoMo questions for an A/B comparison.

Candidate next hypothesis, not yet started:

> With lexical reranking enabled, GPU cross-encoder reranking improves
> exact-source Recall@10 or downstream judged answer accuracy without adding
> more than an explicitly measured latency budget.

The threshold must be fixed before running the experiment.

If TEI remains unavailable, the bounded local fallback hypothesis is:

> A thread-safe bounded cache of candidate token sets reduces mean offline
> lexical rerank latency by at least 3x while producing identical rankings.

## Worktree state

Task-related modified or new files currently include:

- `README.md`
- `benchmarks/__init__.py`
- `benchmarks/kv_reduction_eval.py`
- `benchmarks/results/BENCHMARK_REPORT.md`
- `benchmarks/results/kv_reduction_locomo_checkpoint.json`
- `benchmarks/results/kv_reduction_locomo_lexical_rrf.json`
- `config.py`
- `docs/KV_CACHE_RESEARCH.md`
- `docs/RESEARCH_HANDOFF.md`
- `engine/hybrid_ranker.py`
- `engine/lexical_reranker.py`
- `tests/test_kv_reduction_eval.py`
- `tests/test_lexical_reranker.py`

No commit has been created. Inspect `git diff` carefully because
`engine/hybrid_ranker.py` contains both pre-existing user work and research-loop
changes.
