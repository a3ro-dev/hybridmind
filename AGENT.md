# HybridMind repository instructions

Updated: August 2026

HybridMind is a local dense, sparse, and graph retrieval service for memory
experiments. Do not describe it as a transformer KV-cache replacement or as a
validated 10M–100M context system. Those are research targets governed by
`docs/RETRIEVAL_RESEARCH_PROTOCOL.md`.

## Required engineering rules

1. Use `.venv` for Python and `pnpm` for `memorybench` frontend dependencies.
2. `config.py` is the configuration source of truth.
3. Runtime embeddings are remote and must be native, finite, exactly 4096
   dimensions. Never add a local, padded, projected, truncated, or mixed-width
   fallback. Offline tests may inject an explicit deterministic 4096-d test
   double.
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

## Executable architecture

- API: FastAPI routers in `api/`, application/security/lifecycle in `main.py`.
- Authoritative store: SQLite/WAL in `storage/sqlite_store.py`.
- Derived retrieval indexes: FAISS HNSW (`storage/vector_index.py`), BM25/BM25S
  (`storage/bm25_index.py`), NetworkX `MultiDiGraph`
  (`storage/graph_index.py`). The service rebuilds them from validated SQLite.
- Retrieval: `engine/hybrid_ranker.py` generates independently controlled
  dense, sparse, and graph candidates, applies temporal validity, fuses ranks,
  and optionally invokes the configured cross-encoder.
- Structured ingestion: `engine/fact_extractor.py` plus
  `POST /ingest/session-facts`. Extracted fields are conservative heuristics,
  not general causal or temporal reasoning.
- Query decomposition: optional bounded LLM subqueries in
  `engine/query_decomposition.py`; usefulness remains an empirical question.
- Salience: optional recency/access/degree heuristic in `engine/salience.py`.
- Consolidation: optional lossy, provenance-linked derived summaries in
  `engine/consolidation.py`. It is not an Observer/Reflector architecture and
  source facts may not be archived by this path.

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

## Verification

Run focused tests while editing, then the full offline suite and compilation.
Do not count test names or configuration assertions as feature evidence. Prefer
behavioral, invariant, crash/rollback, and provenance tests. Record explicitly
which provider calls were made; offline runs must report zero.
