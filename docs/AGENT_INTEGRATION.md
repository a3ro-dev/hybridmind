# HybridMind Agent Integration Guide

This guide details how to integrate HybridMind into AI agents using the Python SDK (`sdk/memory.py`), the FastMCP server (`mcp_server/main.py`), and the REST API endpoints.

---

## 1. Quick Start (Python SDK)

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")

# 1. Create a session
session = memory.session.create(
    name="robotics_lit_review",
    metadata={"owner": "agent", "goal": "survey manipulation papers"}
)
session_id = session["session_id"]

# 2. Store findings
paper_id = memory.store(
    text="Paper A proposes diffusion policies for visuomotor control.",
    metadata={"domain": "robotics", "source_url": "https://arxiv.org/abs/2303.04137"},
    session_id=session_id,
)

# 3. Relate nodes explicitly
memory.relate(paper_id, "target-uuid-2", "supports")

# 4. Recall memory (Tri-signal RRF + mxbai Reranker)
results = memory.recall("diffusion control policy", top_k=5, mode="hybrid")
```

---

## 2. MCP Server Integration (Model Context Protocol)

HybridMind ships with a native FastMCP server (`mcp_server/main.py`) for AI tools like Claude Desktop, Cursor, and Windsurf:

### Tools Exposed

- **`remember(text: str, metadata: dict = None)`**: Store text node into HybridMind.
- **`recall(query: str, top_k: int = 5, mode: str = "hybrid")`**: Search memories via RRF fusion + cross-encoder reranking.
- **`relate(source_id: str, target_id: str, relationship: str)`**: Create explicit graph edge between memories.
- **`forget(text: str)`**: Find and soft-delete nearest node matching text.

### Claude Code / MCP Config (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "hybridmind": {
      "command": "python",
      "args": ["d:/hybridmind/mcp_server/main.py"],
      "env": {
        "HYBRIDMIND_API_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

---

## 3. Structured Ingestion Endpoint (`/ingest/session-facts`)

For agent workflow fact extraction with contradiction handling:

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/ingest/session-facts",
    json={
        "session_id": session_id,
        "facts": [
            {"fact": "User prefers Dark Mode UI.", "entities": ["User", "Dark Mode"]},
            {"fact": "User works at OpenAI.", "entities": ["User", "OpenAI"]}
        ]
    }
)
```

---

## 4. Operational Best Practices

1. **Session Lifecycles**: Isolate conversation contexts using `session_id` parameters to enable scoped memory recall.
2. **Auto-Edge Linkage**: Enable `HYBRIDMIND_AUTO_EDGES_ENABLED=true` to infer graph edges automatically on ingestion.
3. **Environment Setup**: Set `RUNPOD_TEI_EMBEDDING_URL` and `RUNPOD_LLM_ENDPOINT_ID` for GPU-accelerated serverless processing.
