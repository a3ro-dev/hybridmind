# HybridMind Agent Integration Guide

This guide shows how to integrate `sdk/memory.py` into a production research agent.
It is focused on reliable memory workflows, session isolation, provenance, and graph-aware retrieval.

## Quick Start

```python
from sdk.memory import HybridMemory

memory = HybridMemory(base_url="http://127.0.0.1:8000")

# 1) Create a session for scoped work
session = memory.session.create(
    name="robotics_lit_review",
    metadata={"owner": "agent", "goal": "survey manipulation papers"}
)
session_id = session["session_id"]

# 2) Store findings in that session
paper_id = memory.store(
    text="Paper A proposes diffusion policies for visuomotor control.",
    metadata={
        "domain": "robotics",
        "source_url": "https://arxiv.org/abs/2303.04137",
        "query": "diffusion policy robotics",
        "kind": "paper_summary",
    },
    session_id=session_id,
)

# 3) Recall context in-session (strict metadata filtering)
context = memory.session.recall("diffusion control policy", session_id=session_id, top_k=8)

# 4) Optional graph exploration around a concept
neighborhood = memory.trace("diffusion policy", depth=2)

# 5) Get health of memory state
stats = memory.stats()
print(stats["node_count"], stats["edge_count"])
```

---

## Core Concepts

- `store()` writes one node.
- `store_batch()` writes many nodes + optional edges in one API call (`/bulk/import`).
- `store_with_auto_edges()` writes a node, then auto-links to nearest neighbors using vector similarity.
- `recall()` retrieves via `hybrid` or `vector`.
- `session.*` provides scoped memory spaces with lifecycle controls.
- `tools.get_schema()` exposes JSON schemas for LLM function-calling.

Edge taxonomy is enforced by the server. Supported types include:
`led_to`, `contradicts`, `supports`, `caused_by`, `retrieved_during`,
`refined_by`, `depends_on`, `analogous_to`, `invalidated_by`, `derived_from`.

---

## Pattern 1: Read a Paper -> Store Summary -> Auto-Edge to Related Work

Use this when ingesting many papers where linking to similar prior work should be automatic.

```python
from sdk.memory import AutoEdgeConfig, HybridMemory

memory = HybridMemory("http://127.0.0.1:8000")
session_id = memory.session.create("paper_ingest")["session_id"]

paper_text = (
    "This paper introduces a retrieval-augmented planner for robotic manipulation "
    "with uncertainty calibration."
)

result = memory.store_with_auto_edges(
    text=paper_text,
    metadata={
        "domain": "robotics",
        "source_url": "https://arxiv.org/abs/2401.00001",
        "kind": "paper_summary",
    },
    session_id=session_id,
    auto_edge_config=AutoEdgeConfig(
        threshold=0.7,
        max_edges=5,
        allowed_edge_types=("derived_from", "analogous_to"),
    ),
)

print("node:", result["node_id"])
print("auto_edges:", result["auto_edges_created"])
```

Notes:
- Similarity threshold controls precision/recall of automatic links.
- The alias `related_to` maps to `analogous_to` in SDK for compatibility.

---

## Pattern 2: Answer a Research Question (Decompose -> Search -> Store -> Synthesize)

Use a deterministic loop where each sub-answer is persisted with provenance.

```python
question = "How do diffusion policies compare to behavior cloning in low-data regimes?"
session_id = memory.session.create("diffusion_vs_bc")["session_id"]

sub_questions = [
    "sample efficiency of diffusion policies",
    "behavior cloning performance in sparse data",
    "robustness under distribution shift",
]

for sq in sub_questions:
    # Retrieve memory context before external search/tooling
    prior = memory.session.recall(sq, session_id=session_id, top_k=6)

    # ... run your web/arxiv/tools pipeline here ...
    synthesized_finding = f"Synthesized answer for: {sq}"
    source_url = "https://example.org/source"

    finding_id = memory.store(
        text=synthesized_finding,
        metadata={
            "domain": "ml",
            "query": sq,
            "source_url": source_url,
            "kind": "finding",
            "prior_context_count": len(prior),
        },
        session_id=session_id,
    )

# Final synthesis node
final_id = memory.store(
    text="Final synthesis for diffusion vs BC question.",
    metadata={"kind": "final_answer", "query": question},
    session_id=session_id,
)
```

Why this works:
- The agent keeps an auditable chain of intermediate claims.
- Every claim carries source/query context.

---

## Pattern 3: Multi-Session Research (Carry Findings Across Sessions)

Use sessions for isolated tasks while still allowing controlled cross-session transfer.

```python
# Session A: foundation reading
sess_a = memory.session.create("foundation_models")
sid_a = sess_a["session_id"]

node_a = memory.store(
    "Scaling laws indicate smooth loss improvements with compute.",
    metadata={"domain": "ml", "source_url": "https://arxiv.org/abs/2001.08361"},
    session_id=sid_a,
)

# Session B: product strategy
sess_b = memory.session.create("product_strategy")
sid_b = sess_b["session_id"]

node_b = memory.store(
    "Need model choice strategy for cost-latency-quality frontier.",
    metadata={"domain": "strategy"},
    session_id=sid_b,
)

# Bridge sessions explicitly with an edge
memory.relate(node_b, node_a, "depends_on", weight=0.85)
```

Recommended workflow:
- Keep sessions narrow and goal-specific.
- Add explicit cross-session edges when transferring assumptions/facts.
- Archive stale sessions to reduce recall noise.

---

## Pattern 4: Contradiction Detection

Represent conflicting claims explicitly so graph traversal surfaces disagreements.

```python
claim_1 = memory.store(
    "Method X outperforms baseline Y on benchmark Z.",
    metadata={"source_url": "https://paper-a.org", "kind": "claim"},
)
claim_2 = memory.store(
    "Reproduction study finds Method X underperforms baseline Y on benchmark Z.",
    metadata={"source_url": "https://paper-b.org", "kind": "claim"},
)

memory.relate(claim_2, claim_1, "contradicts", weight=0.95)
```

During answer generation:
- Query recall for the topic.
- Trace local graph neighborhood.
- If contradictory nodes exist, force the agent to produce uncertainty-aware output.

---

## Pattern 5: Provenance-First Memory

Every stored node should contain at least:
- `source_url`
- `query` (what retrieval/search prompt produced this evidence)
- `kind` (`paper_summary`, `finding`, `hypothesis`, `final_answer`, etc.)

Example:

```python
memory.store(
    text="Fact extracted from source.",
    metadata={
        "source_url": "https://example.com/doc",
        "query": "specific sub-question here",
        "kind": "finding",
        "domain": "legal",
    },
)
```

This makes downstream auditing and citation generation straightforward.

---

## Session Operations

### Create
```python
session = memory.session.create("my_session", {"owner": "agent-v1"})
```

### Recall within session
```python
rows = memory.session.recall("contract liability cap", session["session_id"], top_k=10)
```

### List sessions with stats
```python
for s in memory.session.list():
    print(s["session_id"], s["node_count"], s["edge_count"], s["status"])
```

### Archive session
```python
archive_result = memory.session.archive(session["session_id"])
print(archive_result)
```

Archiving creates one summary node and soft-deletes older session nodes.

---

## Batch Ingestion

Use `store_batch()` for high-throughput imports where each item can include outgoing edges.

```python
batch = [
    {
        "id": "paper_1",
        "text": "Paper 1 summary",
        "metadata": {"domain": "ml", "source_url": "https://a.org"},
        "edges": [{"target_id": "paper_2", "type": "analogous_to", "weight": 0.8}],
    },
    {
        "id": "paper_2",
        "text": "Paper 2 summary",
        "metadata": {"domain": "ml", "source_url": "https://b.org"},
        "edges": [],
    },
]

result = memory.store_batch(batch)
print(result["created_nodes"], result["created_edges"])
```

---

## Streaming Recall for Long Context Pipelines

When post-processing many hits, stream in batches:

```python
def handle_batch(batch):
    # process batch with your agent pipeline
    pass

for batch in memory.recall_stream(
    query="long-form retrieval query",
    top_k=100,
    batch_size=20,
    mode="hybrid",
    on_batch_callback=handle_batch,
):
    # batch already processed in callback; optional extra logic here
    pass
```

---

## Tool Interface for LLM Function Calling

SDK exposes tool schemas compatible with OpenAI function-calling style:

```python
schemas = memory.tools.get_schema()
for tool in schemas:
    print(tool["function"]["name"])
```

Each definition includes:
- `name`
- `description`
- `parameters` (JSON Schema)
- `x-return-type`

---

## Full Working Example: Minimal Research Agent Loop

```python
from sdk.memory import HybridMemory, AutoEdgeConfig


def run_research_agent(question: str) -> str:
    memory = HybridMemory("http://127.0.0.1:8000")
    session = memory.session.create(
        name="research_loop",
        metadata={"question": question, "agent_version": "v1"},
    )
    sid = session["session_id"]

    # Step 1: retrieve existing memory context
    context_rows = memory.session.recall(question, session_id=sid, top_k=8)

    # Step 2: external research step (replace with real tools)
    external_findings = [
        {
            "text": "Finding A from source 1",
            "source_url": "https://example.org/1",
            "query": question,
        },
        {
            "text": "Finding B from source 2",
            "source_url": "https://example.org/2",
            "query": question,
        },
    ]

    # Step 3: persist findings with auto-linking
    for f in external_findings:
        memory.store_with_auto_edges(
            text=f["text"],
            metadata={
                "kind": "finding",
                "domain": "research",
                "source_url": f["source_url"],
                "query": f["query"],
                "context_seed_count": len(context_rows),
            },
            session_id=sid,
            auto_edge_config=AutoEdgeConfig(threshold=0.7, max_edges=3),
        )

    # Step 4: synthesize answer from memory
    synthesis_context = memory.session.recall(question, session_id=sid, top_k=12)
    answer = f"Synthesized answer with {len(synthesis_context)} supporting memory nodes."

    memory.store(
        text=answer,
        metadata={"kind": "final_answer", "query": question},
        session_id=sid,
    )

    # Optional: archive at end of run
    # memory.session.archive(sid)

    return answer
```

---

## Operational Recommendations

- Keep memory writes deterministic and metadata-rich.
- Use `store_batch` for ingestion phases, `store_with_auto_edges` for iterative discovery phases.
- Run `compact()` periodically if many nodes are soft-deleted.
- Prefer session-scoped recall for agent loops to reduce context contamination.
- Keep contradiction edges explicit (`contradicts`) to avoid silent conflicts in final outputs.
