# Reproduction map: upstream mechanism → HybridMind

Each row traces one mechanism: where it came from (pinned commit and licence),
where it lives in HybridMind, how it deviates, how equivalence was checked,
and what was measured. Licence texts are in `THIRD_PARTY_NOTICES.md`.

**Status words**
- **port-parity**: identical outputs to the upstream code on identical inputs.
- **reproduced**: the upstream result was re-run in our environment.
- **measured**: HybridMind results.

"Reference track" means EMG's shipped 1536-d `text-embedding-3-small`
vectors. It is not the 4096-d runtime model, and no reference-track number
transfers to the runtime configuration.

## Mechanism table

| Upstream mechanism | Source / licence | HybridMind | Deviations | Verification | Measured result |
|---|---|---|---|---|---|
| EMG entity–memory graph: entity BM25 soft match, degree-discounted entity strength, Who-only dampening, ±1 sequence expansion | Sun668/em_graph_memory `f020e855`, MIT | `engine/entity_graph.py` (`EntityMemoryGraph`, `emg_rank`) | Chronological tie-break by default (`upstream_order=True` restores EMG's dia_id order); node ids are HybridMind ids; stopword list vendored (198 words) | **port-parity**: 19,860/19,860 rankings identical to upstream on EMG's shipped LoCoMo artifacts (arms A, B, B_entity, B_noseq × k); metric equal in every cell (`scripts/reproduce_emg_locomo.py --port`, `tests/test_emg_port_parity.py`) | Graph-only R@25 0.662 through `/retrieve` code on the reference track, vs EMG B_entity 0.661 |
| EMG gated fusion `0.30·E + 0.70·S` with dense fill | same | `EntityMemoryGraph.emg_fused_rank`, `engine/fusion.py` (`EMG_LINEAR_WEIGHTS`) | none in logic | **port-parity** (as above) | **reproduced** (qualitative): B − A = +4.65 R@25 vs +4.74 published; ordering B > B_noseq > A > B_entity holds. Absolute values 0.9–2.6 points low because the shipped vectors embed a legacy text version (0/5,882 digest matches); see `research/experiments/r1-emg-reproduction/analysis.md` |
| EMG LLM entity extraction (typed Who/What/When/…) | same | `engine/entity_extraction.py` (`LLMEntityExtractor`), persisted in `node_entity_extractions` | LLM call injected; never run in this repo so far | Prompt and entity types equal upstream `build/config.py` v4; tested with fake completions | not run (needs spend) |
| HippoRAG 2 seeding + personalized PageRank (damping 0.5, top-5 entity seeds, node specificity `1/\|P_e\|`, passage seeds `minmax·0.05`) | OSU-NLP-Group/HippoRAG paper code `191f281`, MIT | `EntityMemoryGraph.ppr_rank`, `personalized_pagerank` | scipy power iteration instead of igraph PRPACK (GPL); walk runs on the EMG graph (entities + turns + sequence edges), not OpenIE triples; no recognition-memory LLM filter | PPR equals `networkx.pagerank` within 1e-8 (random graphs, dangling nodes, parallel edges); igraph not compared | Reference track: PPR graph-only R@25 0.628; `all-ppr-seeded` 0.824 (+3.7 vs dense) |
| HybridMind `lexical-v1` extractor (speakers, capitalized spans, time/number expressions, RAKE-style phrases) | HybridMind original | `engine/entity_extraction.py` (`LexicalEntityExtractor`) | not a reproduction | deterministic unit tests | Graph-only on EMG artifacts: 66.06 vs 66.09 with the LLM graph; fused 0.30/0.70: +4.10 vs dense [3.14, 4.93] |
| BM25S Lucene scoring | xhluca/bm25s 0.3.9 (tag `c37c81c`), MIT | `storage/bm25_index.BM25SBackend` via `engine/trisignal.SparseChannel` | scope-local IDF; stopwords "en" + Snowball stemmer | library used directly (identical by construction) | LoCoMo reference track sparse R@25 0.679; LongMemEval-S recall_all@10 0.713, complete @4k 0.798 (matches E4's independent 0.791) |
| LongMemEval official retrieval metrics (recall_any/all, ndcg_any, turn2session) | xiaowu0162/LongMemEval `9e0b455`, MIT | `scripts/reproduce_longmemeval_retrieval.py`, imported by `benchmarks/conversational_metrics.py` | numpy-2 `asfarray` change | exact equality on 8 real questions × k ∈ {1,3,5,10,30,50}; official denominators reproduced (419 official, 470 any-role) | E4 official BM25 reproduction (session recall_any@5 0.888) |
| LoCoMo evidence recall | snap-research/locomo `3eb6f2c`, CC-BY-NC-4.0 | `benchmarks/conversational_metrics.locomo_recall_acc` | reimplemented from the definition (no code copied); our means exclude empty-evidence rows (EMG/official divide by all 1,986) | all 13 EMG formal `stats.json` cells replayed to 1e-16 (repro harness) | — |
| Reciprocal rank fusion `Σ w/(k+rank)`, k=60 | Cormack et al. 2009 | `engine/fusion.rrf_fuse` | none | unit tests | default fusion |
| Distribution-based score fusion | qdrant/qdrant `6ab21ca`, Apache-2.0 | `engine/fusion.dbsf_fuse` | reimplemented from the Rust source | hand-computed tests, including σ=0 and singleton | Reference track `all-dbsf` R@25 0.815 |
| z-score convex fusion | Chrislysen/opsem `68186a4`, MIT | `engine/fusion.zscore_fuse` | a missing score is filled with the channel minimum | unit tests | Reference track `all-zscore` R@25 0.800 |
| E-series evidence operators: ±w window, λ=0.7 propagation, greedy skip-not-truncate packing, session units | HybridMind `scripts/offline_budgeted_evidence.py` (E1–E4) | `engine/evidence.py` (`assemble_evidence`) | node-id keyed; signed scores min-max mapped before propagation | 53,776 selection checks against the E-series harness on LoCoMo, 0 mismatches | complete coverage @ budgets in `eval_trisignal.py` outputs |
| Emergence "Simple" session grouping (turn-rank DCG) | blog description only (repository unlicensed) | `engine/evidence.rank_sessions(method="dcg")` | clean-room; DCG uses `1/log2(rank+1)` (the blog's formula divides by zero at rank 1) | unit tests | not yet measured |
| Exact vs HNSW dense search | faiss 1.13.2, MIT | `engine/dense_channel.py`, `scripts/ann_audit.py` | HNSW built single-threaded for determinism | exact == brute-force numpy; audit tooling | Reference vectors: efSearch 64 gives R@100 0.997 vs exact (worst conversation 0.994); 4096-d audit pending |
| Qwen3-Embedding query instruction `Instruct: {task}\nQuery:{q}` | Qwen/Qwen3-Embedding-8B model card `1d8ad4c` | `engine/embedding.format_query_for_embedding` | none (no space after `Query:`, per the Python helper and ST config) | string tests | not yet measured (VPS arm) |
| TEI `/rerank` with bge-reranker-v2-m3 | TEI OpenAPI 1.9.4; BAAI model, Apache-2.0 | `engine/reranker.TEIReranker` | chunks of 32 texts per request | mocked transport tests | not yet measured (VPS arm) |

## How to re-run

```
python scripts/reproduce_emg_locomo.py            # upstream EMG, offline, shipped artifacts
python scripts/reproduce_emg_locomo.py --port     # HybridMind port vs upstream, identical inputs
python eval_trisignal.py --dataset locomo --dense emg-reference --arms dense,sparse,graph,dense+graph,all --baseline dense --label <label>
python eval_trisignal.py --dataset longmemeval --arms sparse,graph,graph-ppr,sparse+graph --label <label>
```

The EMG scripts need the upstream clone at `tmp/upstream/em_graph_memory`
(commit `f020e855`) and NLTK stopwords in `tmp/nltk_data`. The harness checks
the commit and makes zero provider calls.

## Not reproduced yet (and why)

- **EMG absolute published numbers.** These need the formal post-refactor
  embedding index and graphs. Rebuilding them costs paid calls.
- **HippoRAG 2's published MuSiQue/2Wiki recall.** This needs NV-Embed-v2 on a
  GPU plus about 1,000 recognition-filter LLM calls.
- **Every 4096-d runtime number.** This needs the embedding cache filled on a
  GPU host (`scripts/fill_embedding_cache.py`, plan-gated).
