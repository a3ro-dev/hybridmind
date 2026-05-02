# FINAL STATUS

## 1. Repo/Doc Mismatches Fixed
- Removed references to RRF (Reciprocal Rank Fusion) and replaced the hybrid scoring implementation to honestly follow the declared linear late fusion scoring: `Score = α·V + β·G`.
- Removed dead knobs like `edge_type_weights` and `gamma` / `R` (Relationship bonuses) which were confusing, improperly wired, and convoluted the code unnecessarily.
- Fixed constraints in `ALGORITHM.md`, `ARCHITECTURE.md` and `AGENT.md` to reflect the removed configurations and honest methodology.

## 2. Tightened Hybrid Implementation
- Discarded RRF for fusing Vector and Graph and implemented the linear scoring `Score = α·V + β·G` as claimed.
- BM25 overlap score is now explicitly added as an exact-match boost to the base vector score during candidate generation.
- **Graph-Aware Candidate Expansion**: The hybrid ranker now traverses explicit anchors to dynamically inject graph-neighbors into the candidate pool. This fixes a critical flaw where graph scoring previously could not rescue nodes that missed the initial Vector `top_k` cut.

## 3. Ablation Findings & Benchmarks
- Created `retrieval_ablation.py`: Isolates vector-only, BM25-only, and hybrid regimes.
- Created `targeted_graph_benchmark.py`: Evaluates semantic, lexical, edge-dependent multi-hop, and missing-anchor retrieval scenarios.
- Created `ingest_ablation.py`: Stress-tests ingest-time neighborhood averaging.
- **Findings**:
  - In semantic and lexical retrieval, Vector+BM25 performs flawlessly (100% recall@3) making graph traversal unnecessary.
  - In explicit edge-dependent multi-hop scenarios (e.g. asking a question where the answer shares no semantic overlap but relies on a structural edge), Vector-only fails (0% recall), but Graph-heavy Hybrid retrieves the answer correctly (100% recall).
  - Graph-heavy search fails completely if an anchor is missing (Missing Anchor Failure), demonstrating the hard limit of graph-assisted retrieval.
  - Ingest-time neighborhood averaging improves retrieval of a query connected semantically to neighbors (improves from 66% to 100% probe recall in our tests).

## 4. Removed Dead Knobs
- Deleted unused `edge_type_weights` implementation and configuration from the database, api models, graph search engine, and agent specs.
- Replaced the flawed RRF fusion with direct linear combination.

## 5. Reviewer-Grade Limitations Added
- Added an honest evaluation of graph sparsity failure, embedding model domain-separation problems, and ingest scalability bounds to the `README.md`.

## 6. Next Steps
- Validate multi-hop semantic expansion: Currently, missing anchors lead to zero graph signal. An iterative approach where vector search informs the anchors could be added if needed.
- Expose an asynchronous embedding ingest pipeline to improve scalability beyond single-threaded GIL constraints.
