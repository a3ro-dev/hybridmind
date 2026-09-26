# HybridMind implementation status

Updated: September 2026

This is a code-status inventory, not a quality or production-readiness claim.

| Area | Current status | Evidence boundary |
|---|---|---|
| Tri-signal retrieval (`POST /retrieve`) | Implemented | Independent scope-local dense/sparse/graph channels, RRF/DBSF/z-score/min-max fusion, fixed-pool rerank, evidence packing, hashed trace; API/SDK/MCP tested offline. Defaults are provisional until the 4096-d arm runs. |
| Graph channel: EMG entity-memory graph | Implemented, parity-verified | Exact ranking parity with upstream EMG on its shipped LoCoMo artifacts (19,860/19,860). Reproduces the paper's fusion gain qualitatively (+4.65 vs +4.74 R@25, reference 1536-d vectors). |
| Graph channel: HippoRAG-2 PPR | Implemented arm | PPR matches networkx within 1e-8; below EMG as graph-only on the reference track; not the default. |
| Entity extraction | `lexical-v1` implemented (offline); EMG LLM extractor ported, tested with fakes only | Lexical graph matched the LLM graph on LoCoMo graph-only recall (66.06 vs 66.09); LLM extraction has not been run here. |
| Dense exact vs HNSW | Implemented + audited on reference vectors | Scope-local exact search; HNSW recall audit tooling; 4096-d audit pending a real store. |
| TEI reranker / keyless loopback TEI | Implemented, mocked tests only | No live TEI call has been made. |
| Engine-driven benchmark harness (`eval_trisignal.py`) | Implemented | Offline and reference-track runs recorded; 4096-d dense arms need a filled embedding cache. |
| Remote 4096-d embeddings | Implemented, fail-closed | Responses, batches, SQLite blobs, rebuilds, and queries validate finite exact width; live provider health still needs bounded preflight. |
| Dense retrieval | Implemented | FAISS HNSW candidate generation; real-corpus quality and scale are unproven. |
| Sparse retrieval | Implemented | BM25/BM25S independent candidate path. |
| Typed graph retrieval | Implemented | Directed parallel edges, edge IDs, confidence, validity and explicit anchors affect traversal; graph usefulness remains benchmark-dependent. |
| RRF fusion | Implemented | Validated numeric inputs and explicit-weight semantics. |
| Cross-encoder reranker | Optional, off by default | Search exposes attempted/applied/failure state; a positive evaluator pool must prove execution. |
| Query routing | Implemented heuristic | Regex classification changes omitted weights; it is not an LLM or learned router. |
| Query decomposition | Optional | Bounded centralized-LLM path with guards; improvement is unproven. |
| Temporal handling | Partially implemented | Event/valid/assertion times, half-open filtering, latest/previous and conservative supersession exist; this is not general temporal reasoning. |
| Structured facts | Optional | Provider-extracted fields, stable IDs, provenance and conservative slot updates; extraction accuracy requires evaluation. |
| Salience/access | Optional heuristic | Recency/access/degree multiplier; not learned and disabled by default. |
| Consolidation | Optional lossy summary | Provenance-linked derived summaries; not Observer/Reflector and sources are retained. |
| Portable snapshots | Implemented v2 | Safe allowlisted files, SQLite backup, integrity/semantic verification and staged restore; unsigned manifests do not prove authenticity. |
| API security | Local-safe baseline | Loopback default, remote-bind key requirement, CORS/host allowlists and process-local rate limits; not multi-tenant auth. |
| LoCoMo/LongMemEval/MuSiQue retrieval evaluators | Hardened but awaiting fresh live runs | Exact evidence metrics, scoping, failed receipts and manifests; historical legacy results are not valid current evidence. |
| Ablation matrix | Partial | Vector/sparse conditions can be request-attested; graph/hybrid/stateful conditions remain plan-only without external manifests/server attestation. |
| GNN/fusion training scripts | Scaffolded/untrained | No shipped trained checkpoints or validated gains. |
| ColBERT/visual retrieval | Experimental/optional | Not part of the default validated path. |
| 10M–100M effective context | Research target | Protocol and capacity arithmetic exist; target quality has not been demonstrated. |
| Transformer KV-cache replacement | Not implemented | Requires model-integrated attention/KV access, not semantic retrieval alone. |

Current test results should always be reported from a fresh full-suite run. Do
not reuse counts from this document.
