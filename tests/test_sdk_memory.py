import json
from typing import Any, Dict, List

import httpx

from sdk.memory import AutoEdgeConfig, HybridMemory


class MockHybridMindTransport:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.created_edges: List[Dict[str, Any]] = []
        self.nodes_all = [
            {
                "id": "n_session_root",
                "text": "Session root: test",
                "metadata": {
                    "_hm_session_id": "sess_1",
                    "_hm_kind": "session_root",
                    "session_name": "test",
                },
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": "n_member_1",
                "text": "Finding A",
                "metadata": {"_hm_session_id": "sess_1", "domain": "ml"},
                "created_at": "2026-01-01T00:01:00Z",
            },
        ]
        self.edges_all = [
            {
                "id": "e1",
                "source_id": "n_session_root",
                "target_id": "n_member_1",
                "type": "supports",
                "weight": 0.8,
            }
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method = request.method
        path = request.url.path
        content = request.content.decode() if request.content else ""
        payload = json.loads(content) if content else None
        self.calls.append({"method": method, "path": path, "json": payload, "query": str(request.url.query)})

        if method == "POST" and path == "/nodes":
            return httpx.Response(201, json={"id": f"node_{len(self.calls)}"})

        if method == "POST" and path == "/search/hybrid":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"node_id": "h1", "metadata": {"domain": "ml"}, "combined_score": 0.9},
                        {"node_id": "h2", "metadata": {"domain": "bio"}, "combined_score": 0.8},
                    ]
                },
            )

        if method == "POST" and path == "/search/vector":
            filter_metadata = (payload or {}).get("filter_metadata")
            if filter_metadata:
                return httpx.Response(
                    200,
                    json={"results": [{"node_id": "s1", "metadata": filter_metadata, "vector_score": 0.92}]},
                )
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"node_id": "node_1", "vector_score": 0.99, "metadata": {"domain": "ml"}},
                        {"node_id": "node_2", "vector_score": 0.88, "metadata": {"domain": "ml"}},
                        {"node_id": "node_3", "vector_score": 0.76, "metadata": {"domain": "ml"}},
                        {"node_id": "node_4", "vector_score": 0.61, "metadata": {"domain": "bio"}},
                    ]
                },
            )

        if method == "GET" and path == "/search/graph":
            return httpx.Response(200, json={"results": [{"node_id": "g1", "graph_score": 0.5}]})

        if method == "POST" and path == "/edges":
            edge_id = f"edge_{len(self.created_edges) + 1}"
            self.created_edges.append(payload or {})
            return httpx.Response(201, json={"id": edge_id})

        if method == "DELETE" and path.startswith("/nodes/"):
            return httpx.Response(200, json={"deleted": True})

        if method == "POST" and path == "/admin/compact":
            return httpx.Response(200, json={"compacted": True, "deleted_nodes": 2})

        if method == "POST" and path == "/bulk/import":
            node_count = len((payload or {}).get("nodes", []))
            edge_count = len((payload or {}).get("edges", []))
            return httpx.Response(
                200,
                json={
                    "nodes": {"created": node_count, "failed": 0},
                    "edges": {"created": edge_count, "failed": 0},
                    "total_elapsed_ms": 11.1,
                },
            )

        if method == "GET" and path == "/search/stats":
            return httpx.Response(
                200,
                json={
                    "total_nodes": 2,
                    "total_edges": 1,
                    "edge_types": {"supports": 1},
                    "avg_edges_per_node": 0.5,
                    "vector_index_size": 2,
                    "database_size_bytes": 1024,
                },
            )

        if method == "GET" and path == "/nodes":
            skip = int(request.url.params.get("skip", "0"))
            limit = int(request.url.params.get("limit", "100"))
            page = self.nodes_all[skip: skip + limit]
            return httpx.Response(200, json=page)

        if method == "GET" and path == "/edges":
            skip = int(request.url.params.get("skip", "0"))
            limit = int(request.url.params.get("limit", "100"))
            page = self.edges_all[skip: skip + limit]
            return httpx.Response(200, json=page)

        return httpx.Response(404, json={"detail": f"Unhandled route {method} {path}"})


def make_memory(transport: MockHybridMindTransport) -> HybridMemory:
    client = httpx.Client(transport=httpx.MockTransport(transport), base_url="http://127.0.0.1:8000")
    return HybridMemory(base_url="http://127.0.0.1:8000", client=client)


def test_backward_compat_store_recall_trace_relate_forget_compact():
    transport = MockHybridMindTransport()
    memory = make_memory(transport)

    node_id = memory.store("A node", {"domain": "ml"})
    assert node_id.startswith("node_")

    hybrid_rows = memory.recall("query", top_k=2, mode="hybrid")
    assert len(hybrid_rows) == 2
    assert hybrid_rows[0]["node_id"] == "h1"

    trace_rows = memory.trace("concept", depth=2)
    assert trace_rows["anchor"]["node_id"] == "node_1"
    assert trace_rows["neighbors"][0]["node_id"] == "g1"

    edge_id = memory.relate("node_a", "node_b", "supports", weight=0.8)
    assert edge_id == "edge_1"

    memory.forget(node_id)
    compacted = memory.compact()
    assert compacted["compacted"] is True

    called_paths = [c["path"] for c in transport.calls]
    assert "/nodes" in called_paths
    assert "/search/hybrid" in called_paths
    assert "/search/graph" in called_paths
    assert "/admin/compact" in called_paths


def test_store_batch_uses_bulk_import_and_embeds_edge_payload():
    transport = MockHybridMindTransport()
    memory = make_memory(transport)

    result = memory.store_batch(
        [
            {
                "id": "n1",
                "text": "Node 1",
                "metadata": {"domain": "ml"},
                "edges": [{"target_id": "n2", "type": "derived_from", "weight": 0.9}],
            },
            {"id": "n2", "text": "Node 2", "metadata": {"domain": "ml"}, "edges": []},
        ]
    )

    assert result["created_nodes"] == 2
    assert result["created_edges"] == 1
    bulk_call = next(c for c in transport.calls if c["path"] == "/bulk/import")
    assert len(bulk_call["json"]["nodes"]) == 2
    assert bulk_call["json"]["edges"][0]["source_id"] == "n1"
    assert bulk_call["json"]["edges"][0]["target_id"] == "n2"


def test_store_with_auto_edges_respects_threshold_and_edge_selection():
    transport = MockHybridMindTransport()
    memory = make_memory(transport)

    result = memory.store_with_auto_edges(
        text="new finding",
        metadata={"domain": "ml"},
        auto_edge_config=AutoEdgeConfig(
            threshold=0.75,
            max_edges=3,
            allowed_edge_types=("derived_from", "related_to"),
        ),
    )

    assert result["node_id"].startswith("node_")
    # vector candidates: 0.99 self (skipped), 0.88 kept, 0.76 kept, 0.61 skipped
    assert result["auto_edges_created"] == 2
    edge_types = [e["type"] for e in result["edges"]]
    assert edge_types[0] == "derived_from"
    assert all(t in ("derived_from", "analogous_to") for t in edge_types)


def test_session_lifecycle_and_session_recall():
    transport = MockHybridMindTransport()
    memory = make_memory(transport)

    session = memory.session.create("test session", {"topic": "retrieval"})
    assert session["session_id"].startswith("sess_")

    recalls = memory.session.recall("query", session["session_id"], top_k=5)
    assert len(recalls) == 1
    assert recalls[0]["metadata"]["_hm_session_id"] == session["session_id"]

    sessions = memory.session.list()
    assert sessions
    assert sessions[0]["session_id"] == "sess_1"

    archived = memory.session.archive("sess_1")
    assert archived["archived"] is True
    assert archived["archived_nodes"] >= 1


def test_recall_stream_batches_and_callback_and_tool_schema():
    transport = MockHybridMindTransport()
    memory = make_memory(transport)

    seen_batches: List[List[str]] = []

    def on_batch(batch: List[Dict[str, Any]]) -> None:
        seen_batches.append([item["node_id"] for item in batch])

    batches = list(memory.recall_stream("query", top_k=2, batch_size=1, on_batch_callback=on_batch))
    assert len(batches) == 2
    assert seen_batches == [["h1"], ["h2"]]

    schema = memory.tools.get_schema()
    tool_names = {tool["function"]["name"] for tool in schema}
    assert "store" in tool_names
    assert "store_batch" in tool_names
    assert "session_create" in tool_names
    assert "session_archive" in tool_names
