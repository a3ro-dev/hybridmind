# HybridMind architecture

## Scope

HybridMind is a single-process FastAPI service over an authoritative SQLite
store and three derived retrieval structures. The default path is conventional
hybrid retrieval; optional fact extraction, query decomposition, rerankers,
salience, and consolidation are configuration-gated. Their presence is not
evidence that they improve a workload.

## Request and data flow

```text
write request
  -> validate text, metadata, time, confidence, and exact 4096-d embedding
  -> SQLite transaction (source of truth)
  -> FAISS + BM25/BM25S + MultiDiGraph projections
  -> commit, or rollback SQL and rebuild every projection

search request
  -> exact 4096-d query embedding when dense is enabled
  -> independent dense / sparse / explicitly anchored graph candidates
  -> hard validity filtering and optional time/salience signals
  -> weighted rank fusion (RRF by default)
  -> optional hybrid-only lexical/learned/cross-encoder stages
  -> bounded final list with reranker execution metadata
```

`search_mode` is the isolation boundary for `vector_only`, `sparse_only`,
`vector_sparse`, `graph_only`, and `hybrid`. Controlled modes bypass downstream
rerankers so they remain signal ablations. Graph-only requires explicit anchors;
an anchor recovered by dense search makes the overall procedure vector+graph.

## Storage

- `storage/sqlite_store.py`: nodes, versions, edges, metadata, embeddings,
  validity/assertion times, access/archive state. SQLite is authoritative.
- `storage/vector_index.py`: in-memory FAISS HNSW plus a rebuild copy.
- `storage/bm25_index.py`: lexical candidate generation.
- `storage/graph_index.py`: typed parallel directed edges with edge-ID lookup,
  confidence, direction, and half-open temporal validity.

The application initializes runtime indexes with `index_path=None` and rebuilds
them from SQLite. Legacy direct index `load()` methods remain a trusted-local
tooling boundary and must not be used on untrusted artifacts.

## Portable persistence

Version-2 `.mind.zip` snapshots contain exactly:

- `store.db`
- `vectors.json`
- `graph.jsonl`
- `bm25.jsonl`
- `manifest.json`

The archive publisher uses a SQLite online backup, a staged archive, file
flush/fsync, verification, and atomic file replacement. Import checks paths,
the exact file allowlist, sizes, SHA-256 values, SQLite integrity, embedding
width, and agreement between derived descriptors and SQLite. Restore publishes
the validated SQLite file; runtime indexes are rebuilt. No archive pickle is
accepted.

SHA-256 detects accidental corruption, not malicious repackaging. Authenticity
needs a deployment-held signing/HMAC key. Directory-entry durability on Windows
is filesystem-dependent, and recovery must occur before live SQLite connections
open.

## Provider and network policy

`config.py` owns provider selection. Z.AI, RunPod, and the research proxy have
separate endpoint/key bindings and host validation. Research-proxy use requires
an explicit flag. API startup is offline by default; optional embedding warm-up
must follow the priced, checksum-bound live-plan preflight. Health probes are
non-billable unless remote checks are explicitly enabled.

The default bind is loopback. A non-loopback bind requires an API key. CORS and
trusted hosts are allowlisted, and costly endpoints use process-local rate
limits. This is not multi-tenant authorization: remote production still needs
TLS, identity/roles, centralized rate limiting, and audit logging.

## Memory semantics

Extracted facts use retry-stable session/container-qualified IDs. Event time,
valid time, and assertion time are distinct fields. Conservative slot conflicts
can create `supersedes` or `contradicts` relations; this is not general natural
language inference. Consolidation produces a lossy derived summary with source
provenance and does not archive source facts. Salience is a bounded heuristic,
not a learned cognitive model.

## Evaluation boundary

Valid retrieval evaluation requires stable exact evidence IDs, corpus/session
scoping, failure rows, immutable run manifests, completed-run receipts, and
request/server/corpus attestation for ablations. The deterministic answer judge
is lexical/normalized, not an LLM judge. See
`docs/RETRIEVAL_RESEARCH_PROTOCOL.md` for scale, quality, latency, and cost gates.
