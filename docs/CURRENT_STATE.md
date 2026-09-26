# CURRENT STATE — rolling handoff

Read this first when arriving mid-stream; update it before ending a session.
Keep it short: this is a whiteboard, not a report. Historical narrative
belongs in `docs/DECISIONS.md` and `docs/research/`.

- **Last updated:** 2026-09-26
- **Branch/commit:** `research/evidence-budget-20260925` @ `a201d8f` plus
  uncommitted tri-signal engine work (nothing committed by the agent).
  `README.md` and `docs/assets/` also carry the owner's earlier uncommitted
  edits; README received only a surgical `/retrieve` + reranker update.
- **Provider calls made this session:** 0. All runs were offline or on EMG's
  shipped reference artifacts.

## Last verified (2026-09-26)

- Full offline suite `pytest tests/ -q`: **525 passed / 3 skipped** (64.8 s).
- `python -m compileall` over app, eval, scripts, tests and verify: pass.
- Suite runs no longer touch `data/backups/` (conftest isolation fix).

## Active focus

A tri-signal retrieval engine built from ported SOTA mechanisms (see
`docs/REPRODUCTION_MAP.md`):

- `POST /retrieve` (`engine/trisignal.py`) runs independent scope-local
  channels:
  - dense: exact or HNSW;
  - sparse: BM25S;
  - graph: an EMG entity-memory graph with exact upstream parity, plus
    HippoRAG-2 PPR.
- On top of those it adds RRF/DBSF/z-score fusion, fixed-pool rerank
  (including a TEI bge-reranker), and E-series evidence packing.
- Surfaces: SDK `retrieve()` and the MCP `retrieve` tool.
- Harness: `eval_trisignal.py` runs engine-driven ablations on LoCoMo and
  LongMemEval-S cleaned.
- Results: `benchmarks/results/BENCHMARK_REPORT.md` (tri-signal section).

**Next:** the 4096-d dense arm on a GPU VPS. The steps are in
`docs/EVALUATION.md` ("Filling the 4096-d cache"):

1. Serve TEI with Qwen3-Embedding-8B.
2. Run `scripts/fill_embedding_cache.py` (plan-gated).
3. Run `eval_trisignal.py --dense cache:...` for both datasets.
4. Add the instruction and bge-reranker arms.
5. Then set the `/retrieve` defaults from the evidence.

## Open questions / pending decisions

- Default channels and weights are provisional: equal-weight RRF over all
  three. The two datasets disagree on which channels help:
  - LoCoMo (reference track): dense+graph or PPR-seeded fusion help; sparse
    in equal RRF hurts.
  - LongMemEval (lexical graph): the graph hurts; BM25S alone is best.
- The graph channel on LongMemEval may need LLM extraction (EMG extractor
  ported, never run) or down-weighting. Measure before promoting.
- Carried over, still open: the claim-ledger errata for mislabelled LoCoMo
  categories; the RunPod TEI endpoint being down; the `/nodes` previous-turn
  chunk bug.
- To commit the work, split it (owner's choice):
  1. engine + API;
  2. harness + results;
  3. docs.

## Gotchas for the next agent

- `.venv` (not system Python) for everything; `pnpm` inside `memorybench/`
  only if touching the benchmark harness.
- Never edit files via PowerShell string pipelines (`Get-Content |
  Set-Content`). They corrupt non-ASCII bytes (bitten twice; see the
  DECISIONS log).
- The repo has **mixed CRLF/LF files at HEAD**, and editors normalize whole
  files. Run `tmp/fix_eol.py` (gitignored helper) or keep HEAD's line
  endings so diffs stay real.
- `cd` inside a Bash call moves the session cwd; use absolute paths.
- BM25S tie order depends on the requested k; break ties explicitly (the
  engine uses the corpus chronological index).
- The EMG reproduction needs the upstream clone at
  `tmp/upstream/em_graph_memory` @ `f020e855` and NLTK stopwords in
  `tmp/nltk_data`. The parity test skips without them.
- Reference-track (1536-d EMG) numbers are never runtime numbers; label them.
- The owner's laptop is weak: no local neural encoders or long sweeps.
  `eval_trisignal.py` on LoCoMo takes about 9 min and LongMemEval about 6 min.
- `memorybench/` is gitignored except two tracked provider files. Its data
  includes a 347 MB checkpoint JSON; do not commit or "clean" it blindly.
  Its answer/judge path is not used for research runs (see DECISIONS).
- Business material lives in gitignored `local/`; do not cite it as technical
  evidence.
