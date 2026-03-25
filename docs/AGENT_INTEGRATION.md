# Integrating HybridMind into a Research Agent

## What this document covers
Using the Python SDK to provide a research agent with structured, relational memory. This guide focuses on the practical application of HybridMind within an agent reasoning loop.

## Mental Model
HybridMind is more than a vector store; it is a memory layer that understands relationships.

Three core questions it answers:
1. **"What do I know that is similar to X?"** → `recall(x, mode="vector")`
2. **"What do I know that is related to X through any path?"** → `trace(x)`
3. **"How does concept A connect to concept B?"** → `relate()` then `trace()`

## The Edge Vocabulary
Using the correct edge types is critical. Mislabeling a relationship will result in inaccurate graph-based re-ranking.

| Edge Type | Use Case |
|-----------|----------|
| **supports** | Evidence A strengthens claim B. |
| **contradicts** | Finding A conflicts with finding B. |
| **led_to** | Reasoning about A caused you to think of B. |
| **derived_from**| B is a specific case or extension of A. |
| **depends_on** | B requires A to be true/valid. |
| **invalidated_by**| B shows A was incorrect. |
| **refined_by** | B is a more precise version of A. |
| **analogous_to** | A and B are similar across different domains. |
| **caused_by** | A is the direct cause of B. |
| **retrieved_during**| A was retrieved while reasoning about B (provenance). |

## Integration Patterns

### Pattern 1: Read-then-store
Every time the agent reads a source, store it to establish a base memory.
```python
node_id = memory.store(
    text=paper_abstract,
    metadata={"source": url, "domain": "ml", "read_at": "2026-03-25"}
)
```

### Pattern 2: Conclusion Linking
When the agent forms a conclusion from evidence, link them explicitly.
```python
conclusion_id = memory.store("transformers outperform RNNs on long sequences")
memory.relate(conclusion_id, evidence_id_1, "derived_from")
memory.relate(conclusion_id, evidence_id_2, "supports")
```

### Pattern 3: Contradiction Detection
Explicitly link conflicting nodes to enable relational re-ranking that surfaces conflicts.
```python
memory.relate(new_finding_id, old_finding_id, "contradicts")
```

### Pattern 4: Context Recall
Retrieve a combined semantic and structural context before starting a reasoning step.
```python
context = memory.recall(current_topic, top_k=10, mode="hybrid")
neighborhood = memory.trace(current_topic, depth=2)
# Combine both into the LLM prompt context window
```

### Pattern 5: Provenance Tracking
Link retrieved items to the current session or task node.
```python
memory.relate(retrieved_node_id, task_id, "retrieved_during")
```

## Performance Guidance
Single-threaded p50 latencies (1,000-node database):
- `recall()`: ~13ms (hybrid mode)
- `store()`: ~200ms (embedding dominant)
- `trace()`: ~17ms (~15ms anchor search + ~2ms BFS)
- `relate()`: ~2ms (SQLite write)
- `forget()`: ~2ms (Soft delete)
- `compact()`: ~1-2s per 1,000 nodes (Full FAISS rebuild)

**Critical Limitation**: Call SDK methods sequentially. Concurrent calls cause extreme latency degradation (~100-240x) due to Python GIL contention on the embedding model. Ensure memory operations reside within a single-threaded section.

## Operational Recommendations

### Snapshotting
- Do **not** snapshot after every `store()`. Snapshot after bulk ingestion or at the end of an agent session.
- **Latencies**: p50 is 4ms; p99 is 162ms (due to occasional WAL checkpoints).

### Compaction
- Run `compact()` periodically (e.g., at session end) if you frequently use `forget()`. 
- This is an O(n) operation that rebuilds the FAISS index.

### Memory and Scale
- **Memory Growth**: ~1.6KB per node for FAISS; ~0.2KB per node for the graph.
- **Scale Ceiling**: HybridMind provides optimal performance (sub-30ms p95) up to ~1,000 nodes. Estimated ceiling for sub-50ms p95 is ~5k-8k nodes.

## Minimal Working Example

```python
from sdk.memory import HybridMemory

# Initialize connection
memory = HybridMemory(base_url="http://127.0.0.1:8000")

# 1. Store a memory from a source
paper_id = memory.store(
    "Attention is All You Need introduced the transformer architecture, "
    "replacing recurrent networks with self-attention mechanisms.",
    metadata={"source": "arxiv:1706.03762", "year": 2017}
)

# 2. Store a conclusion drawn by the agent
conclusion_id = memory.store(
    "Self-attention enables parallel computation, unlike sequential RNNs.",
    metadata={"type": "conclusion", "confidence": 0.95}
)

# 3. Relate the conclusion to the source
memory.relate(conclusion_id, paper_id, "derived_from", weight=0.9)

# 4. Recall context for a new query
context = memory.recall("attention mechanisms", top_k=5)
neighborhood = memory.trace("transformer architecture", depth=2)

# 5. Cleanup
memory.forget(old_node_id)
memory.compact()  # Permanent removal
```

## Known Gotchas
1. **Empty Database Cold Start**: The first ~10 nodes receive raw embeddings with no conditioning pull (no neighbors yet). The relational coherence builds as the database fills.
2. **store() Latency**: Inserting is ~15x slower than searching (200ms vs 13ms). Avoid inserting in hot reasoning loops; use batch ingestion.
3. **Semantic Anchors**: `trace("BERT")` identifies the node most semantically similar to "BERT" and traverses from it. It does not require an exact string match to find a starting point.
4. **Graph Score Sparse Benefits**: Isolated nodes receive a `graph_score` of 0.0 and will be ranked solely by vector similarity. Explicit edge creation is necessary to leverage the hybrid advantage.
5. **Snapshot version**: Monotonically increases across all experiments sharing the same `.mind` directory.
