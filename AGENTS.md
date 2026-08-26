# HybridMind — agent entry point

Updated: August 2026. Read top to bottom before changing anything; it is
short on purpose. `README.md` is the human-facing front door; this file is
the working contract for anyone editing the code.

## What this is

HybridMind is a local dense + sparse + graph retrieval service for AI-memory
experiments: FastAPI over an authoritative SQLite store, with FAISS HNSW,
BM25/BM25S, and NetworkX indexes derived from it. It is built to show *what*
it retrieved, *why*, and *whether that evidence helped*.

It is **not** a transformer KV-cache replacement and **not** a validated
10M–100M-token context system. Those are preregistered research targets
governed by `docs/RETRIEVAL_RESEARCH_PROTOCOL.md`. Do not let claims drift:
the honest-status table is `PHASE_IMPLEMENTATION_STATUS.md`, and the research
evidence ledger is `docs/research/design-space-experiment-program.md`.

## Cold start

```powershell
.\.venv\Scripts\Activate.ps1            # .venv only, never system Python
pip install -r requirements.txt         # pnpm only inside memorybench/
pytest tests/ -q                        # full offline suite (~400 tests, zero provider calls)
python -m compileall -q main.py config.py api engine storage models cli sdk mcp_server scripts benchmarks tests verify
```

- The offline suite makes **zero** provider calls. If your change needs one,
  see rule 5 below first.
- `make test` / `make compile` wrap the same checks (`make verify` runs the
  legacy opt-in suite separately).
- API: `python -m uvicorn main:app --host 127.0.0.1 --port 8000`. Startup
  does not wake remote providers by default.
- Arriving mid-stream? Read `docs/CURRENT_STATE.md` (and update it when you
  finish; see Covenants).

## Load-bearing map (do not break casually)

| Path | Role | Breakage radius |
|---|---|---|
| `config.py` | Single configuration source of truth | Every subsystem reads it; add fields here, never raw `os.getenv` |
| `storage/sqlite_store.py` | Authoritative store (SQLite/WAL), bitemporal fields | Schema/write-path changes need migration thinking |
| `storage/vector_index.py`, `bm25_index.py`, `graph_index.py` | Derived rebuildable indexes | Must stay rebuildable-from-SQLite; never authoritative |
| `storage/mindfile.py` | `.mind` snapshot publish/restore validation | Security-sensitive path/checksum/semantic gates |
| `engine/hybrid_ranker.py` | Candidate generation, temporal filtering, RRF fusion, optional rerank stages | Retrieval semantics + measured numbers depend on exact behavior |
| `engine/fusion.py` | RRF implementation (`k=60`, weight validation) | Fusion contract |
| `engine/embedding.py`, `provider_policy.py` | Remote-only native 4096-d embeddings; endpoint-bound keys | Fail-closed contract, rule 3/4 |
| `engine/llm_client.py` (+ `llm.py`, `runpod_llm.py`) | Centralized LLM policy chain | Provider routing/spend discipline |
| `main.py` + `api/*.py` | App assembly, security middleware, routers | Auth/rate/limit logic lives in `main.py` ASGI middleware |
| `eval_ledger.py`, `eval_common.py`, `eval_stats.py`, `eval_*.py` | Evaluation ledgers, prompts, statistics | Recorded numbers must stay interpretable/comparable |
| `scripts/preflight.py` | Default-deny live-provider admission gate | Spend safety |
| `memorybench/src/providers/hybridmind/index.ts` | Benchmark harness provider (gitignored tree; these two files tracked) | Contract test: `experiments/harnesses/memorybench_provider_contract.test.ts` |

Real-vs-scaffolded, per module: `PHASE_IMPLEMENTATION_STATUS.md`. Short
version: dense/sparse/graph retrieval, RRF fusion, snapshots, ledgers are
real and tested; GNN/fusion training, ColBERT/visual paths, decomposition,
salience, consolidation are opt-in experiments whose benefit is unproven or
absent. A scaffold, config field, or untrained checkpoint is not a result.

## Required engineering rules

1. Use `.venv` for Python and `pnpm` for `memorybench` frontend dependencies.
2. `config.py` is the configuration source of truth.
3. Runtime embeddings are remote and must be native, finite, exactly 4096
   dimensions. Never add a local, padded, projected, truncated, or mixed-width
   fallback. Offline tests may inject an explicit deterministic 4096-d test
   double (`tests/embedding_double.py`).
4. Z.AI is the hosted production LLM policy, RunPod is the self-hosted policy,
   and the Hack Club proxy is research-only behind
   `HYBRIDMIND_ALLOW_RESEARCH_PROXY=true`. Bind each endpoint to its own key;
   never infer credentials from whichever key happens to exist.
5. Starting the API must not wake a remote provider by default. Before any live
   warm-up or evaluation, generate and validate the offline resource report,
   bind a priced usage-limited live plan to its SHA-256, and run
   `scripts/preflight.py --plan <plan> --validate-only`. Only then may an
   operator omit `--validate-only`.
6. Preserve SQLite as the authoritative source. Primary mutations must be
   atomic or compensate by rebuilding every derived index. Fail closed on
   malformed provider output, corrupt persistence, partial batches, or invalid
   benchmark provenance.
7. Keep experimental/heavy modules opt-in. A scaffold, configuration field, or
   untrained checkpoint path is not an implemented result.

## Retrieval and evaluation contracts

- Default fusion is weighted reciprocal-rank fusion with `k=60`. Explicit
  request weights must not be silently overridden by routing.
- `search_mode` controls `vector_only`, `sparse_only`, `vector_sparse`,
  `graph_only`, and `hybrid`. Graph-only requires explicit, gold-independent
  anchors. A positive rerank pool must produce execution evidence; pool `0`
  means off.
- Exact evidence IDs and corpus/session scoping are required for retrieval
  metrics. Answer-string overlap is not evidence recall and the deterministic
  answer judge is not an LLM judge.
- Evaluation failures must be ledger rows/failed receipts, not skipped
  questions. Ledgers need immutable manifests and completion records.
- `scripts/ablation_matrix.py` may plan conditions, but plan-only modes are not
  completed experiments without request/server/corpus attestation.
- Current valid historical results do not establish the 70–80% prompt-source
  substitution target. Run fresh evidence-ID ingestion and the preregistered
  scale gates before making that claim.

## Persistence contract

Live `.mind` directories contain SQLite plus runtime-derived state. Portable v2
`.mind.zip` snapshots contain only `store.db`, `vectors.json`, `graph.jsonl`,
`bm25.jsonl`, and `manifest.json`. Archives are path-checked, checksum-verified,
and semantically checked against SQLite before restore. Never deserialize
untrusted legacy `.pkl`, `.nx`, or vector metadata files. Plain checksums detect
corruption but are not authenticity signatures unless the deployment adds a
trusted signing key.

## Documentation map (who owns what)

Every tracked `.md` is registered here; `tests/test_doc_integrity.py` fails if
a doc goes missing, unregistered, or links rot. When you change X, update the
doc in its row — do not leave prose describing code that no longer exists.

| Document | Owns | Update when |
|---|---|---|
| `AGENTS.md` | This contract: rules, map, covenants | Any rule/architecture/doc-map change |
| `README.md` | Human front door: overview, quick start, API summary | User-visible behavior/install changes |
| `PHASE_IMPLEMENTATION_STATUS.md` | Honest real-vs-scaffolded inventory | Any module's implementation status changes |
| `docs/CURRENT_STATE.md` | Rolling session handoff | Start AND end of every work session |
| `docs/DECISIONS.md` | Append-only judgment-call log | Whenever you make a call someone might reverse |
| `docs/ARCHITECTURE.md` | Request/data flow, storage, security posture | Architectural behavior changes |
| `docs/ALGORITHM.md` | RRF math, reranker blending, decomposition guards | Ranking/scoring algorithm changes |
| `docs/EVALUATION.md` | Evaluator usage, ledger schema, stats conventions | Eval harness/ledger/statistics changes |
| `docs/RETRIEVAL_RESEARCH_PROTOCOL.md` | Preregistered quality/scale/cost gates | Only with deliberate protocol revision |
| `docs/RESOURCE_SPEED_TOKENOMICS.md` | Offline resource measurement + live-plan gate | Resource accounting changes |
| `benchmarks/results/BENCHMARK_REPORT.md` | Which results are currently valid vs deprecated | New valid result, or an artifact invalidated |
| `experiments/reports/baseline.md` | Pre-change measurement snapshot | Never edited; superseded snapshots get new files |
| `docs/KV_CACHE_RESEARCH.md` | KV-hypothesis history and failed results | New hypothesis evidence |
| `docs/research/*` | Research program, prior-art ledger, claim ledger | New experiments/claims (append; don't rewrite verdicts) |
| `docs/ADVERSARIAL_AUDIT_REMEDIATION.md` | Historical audit record | Never (frozen audit) |
| `docs/AGENT_INTEGRATION.md` | SDK/MCP/API integration contracts | SDK/MCP/request-schema changes |
| `cli/README.md` | CLI command surfaces | cli/* changes |
| `demos/techspec.md`, `deploy/README_image_server.md` | Demo spec; visual-backend deploy | Respective scope changes |

## Covenants (what keeps this true over time)

1. **Session bracket.** On arrival, read `docs/CURRENT_STATE.md`. Before you
   finish: update it (branch/commit, last verified suite results, active
   focus, gotchas). This is the cross-agent memory; stale entries are worse
   than none.
2. **Decision trail.** Made a call a reasonable person might reverse?
   Append it to `docs/DECISIONS.md` with context and reversal notes.
   Burying rationale in commit messages is not enough — commits get squashed,
   this log doesn't.
3. **Doc integrity is tested.** `tests/test_doc_integrity.py` enforces the
   documentation map, resolves README/docs links, and asserts AGENTS.md is
   the single agent entry point. If it fails after your change, fix docs as
   part of the change, not "later".
4. **Verification honesty.** Run focused tests while editing, then the full
   offline suite and compilation before claiming done. Record which provider
   calls were made; offline runs must report zero. Test names and config
   assertions are not feature evidence.
5. **No shell string-surgery on files.** PowerShell `Get-Content |
   Set-Content` pipelines corrupt non-ASCII bytes and BOMs; use proper edit
   tooling. (This has already bitten once.)

## Known debts (documented, deliberately unfixed)

Do not "clean these up" silently; each needs a small design decision:

- `main.py` mixes error-response shapes (`{"status","message"}`, `{"error"}`,
  raised `HTTPException`) across endpoints; clients tolerate both today.
- Node metadata carries dual camel/snake keys (`containerTag`/`container_tag`,
  `sessionId`/`session_id`) for backward compatibility with existing stores
  and the TS provider.
- `memorybench/src/providers/hybridmind/index.ts` re-implements the query
  router regexes from `engine/query_router.py`; they must be updated in
  lockstep (contract test covers part of this).
- Legacy direct index `load()` paths remain trusted-local tooling; they must
  never be pointed at untrusted artifacts.
- Multi-process deployment is unsupported/uncoordinated (single uvicorn
  worker assumed); process-local rate limits and caches assume one process.
