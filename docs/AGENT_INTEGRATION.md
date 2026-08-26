# HybridMind agent integration guide

Three supported integration surfaces: the Python SDK (`sdk/memory.py`), the
MCP server (`mcp_server/main.py`), and the REST API. All of them require a
running HybridMind API process (`python -m uvicorn main:app --host 127.0.0.1
--port 8000`); none of them may bypass the API's security or provider policy.

---

## 1. Python SDK

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")

# Sessions scope recall and fact extraction.
session = memory.session.create(name="lit-review", metadata={"goal": "survey"})
sid = session["session_id"]

# Store returns a node ID. Temporal fields are ISO-8601 strings; memory_kind
# is one of world / experience / observation / opinion when supplied.
nid = memory.store(
    text="Paper A proposes diffusion policies for visuomotor control.",
    metadata={"domain": "robotics"},
    session_id=sid,
    entities=["Paper A"],
    event_time="2026-08-26T10:00:00Z",
)

memory.relate(nid, "target-node-id", "supports")     # explicit graph edge
results = memory.recall("diffusion policies", top_k=5, mode="hybrid")
```

Other `HybridMemory` methods: `store_batch`, `store_with_auto_edges`,
`recall_stream`, `trace(concept, depth)` (graph traversal),
`forget(node_id)` (soft delete by ID — there is no text-matching forget),
`compact()`, `stats()`, and `session.{recall,list,archive}`.

Mode contract: `hybrid`, `vector_only`, `sparse_only`, `vector_sparse`, and
`graph_only`. `graph_only` raises unless you pass explicit `anchor_nodes`;
anchors must come from somewhere other than gold labels. Explicit weights you
pass are never overridden by server-side query routing.

## 2. MCP server

`mcp_server/main.py` is a stdio FastMCP adapter over the API:

| Tool | Signature | Notes |
|---|---|---|
| `remember` | `(text, metadata=None) -> dict` | Stores one node. |
| `recall` | `(query, top_k=10, mode="hybrid") -> list` | Same mode contract as the SDK. |
| `relate` | `(source_id, target_id, relationship) -> dict` | Creates an edge. |
| `forget` | `(node_id) -> dict` | Soft-deletes that exact node ID. |
| `health` | `() -> dict` | Server health probe. |

Client configuration (Claude Desktop / Claude Code / Cursor / Windsurf):

```json
{
  "mcpServers": {
    "hybridmind": {
      "command": "python",
      "args": ["d:/hybridmind/mcp_server/main.py"],
      "env": { "HYBRIDMIND_API_URL": "http://127.0.0.1:8000" }
    }
  }
}
```

`HYBRIDMIND_MCP_TIMEOUT_SECONDS` (default 60) bounds each request.

## 3. Structured session ingestion

`POST /ingest/session-facts` performs server-side LLM fact extraction over raw
conversation turns (clients do not send pre-extracted facts):

```json
{
  "session_id": "<session-id>",
  "container_tag": "optional-scope-tag",
  "turns": [
    {"speaker": "alice", "text": "I prefer dark mode.", "date": "2026-08-26"}
  ]
}
```

Extraction is opt-in and fail-closed: it requires
`HYBRIDMIND_FACT_EXTRACTION_ENABLED=true` plus a configured LLM provider, and
malformed provider output produces an error rather than an empty success.
Extracted fields are conservative heuristics, not general causal/temporal
reasoning.

## 4. Operational rules

1. Scope every write with `session_id` (and `container_tag` where relevant);
   recall scoping depends on it.
2. Live embedding/LLM/reranker use is default-deny. Warm-up or evaluation
   requires the offline resource report + priced plan +
   `scripts/preflight.py --plan <plan> --validate-only` flow (see README).
3. The 4096-dimensional embedding contract has no fallback: if no remote
   backend is configured and healthy, ingestion/search fail rather than
   degrade to another width or a local model.
