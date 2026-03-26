import uuid
from typing import Any, Dict, List

import pytest

from sdk.memory import AutoEdgeConfig, HybridMemory


BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture
def memory() -> HybridMemory:
    m = HybridMemory(base_url=BASE_URL, timeout=30.0)
    try:
        health = m._get("/health")
        if health.get("status") != "healthy":
            pytest.skip("HybridMind server is not healthy")
    except Exception as exc:  # pragma: no cover - only for missing local server
        pytest.skip(f"HybridMind server not reachable at {BASE_URL}: {exc}")
    return m


def _safe_forget(memory: HybridMemory, node_ids: List[str]) -> None:
    for node_id in node_ids:
        try:
            memory.forget(node_id)
        except Exception:
            pass


def test_sdk_backward_compat_live(memory: HybridMemory):
    created: List[str] = []
    token = f"sdk_live_{uuid.uuid4().hex[:8]}"
    try:
        node_id = memory.store(f"{token} backward compat node", {"domain": "sdk_test"})
        created.append(node_id)
        assert node_id

        recalls = memory.recall(token, top_k=5, mode="vector")
        assert isinstance(recalls, list)
        assert any(r.get("node_id") == node_id for r in recalls)

        trace = memory.trace(token, depth=1)
        assert "anchor" in trace and "neighbors" in trace
        assert trace["anchor"] is not None

        other_id = memory.store(f"{token} other node", {"domain": "sdk_test"})
        created.append(other_id)
        edge_id = memory.relate(node_id, other_id, "supports", weight=0.75)
        assert edge_id

        compact_info = memory.compact()
        assert isinstance(compact_info, dict)
        stats = memory.stats()
        assert "node_count" in stats and "edge_count" in stats
    finally:
        _safe_forget(memory, created)


def test_session_management_live(memory: HybridMemory):
    created: List[str] = []
    token = f"sdk_session_{uuid.uuid4().hex[:8]}"
    try:
        session = memory.session.create(
            name=f"{token}_session",
            metadata={"purpose": "integration_test"},
        )
        session_id = session["session_id"]
        created.append(session["root_node_id"])

        n1 = memory.store(
            f"{token} finding one",
            metadata={"domain": "sdk_test", "kind": "finding"},
            session_id=session_id,
        )
        n2 = memory.store(
            f"{token} finding two",
            metadata={"domain": "sdk_test", "kind": "finding"},
            session_id=session_id,
        )
        outside = memory.store(
            f"{token} unrelated outside session",
            metadata={"domain": "sdk_test", "kind": "finding"},
        )
        created.extend([n1, n2, outside])

        session_rows = memory.session.recall(token, session_id=session_id, top_k=10)
        assert len(session_rows) >= 1
        assert all(
            (r.get("metadata") or {}).get("_hm_session_id") == session_id
            for r in session_rows
        )

        sessions = memory.session.list()
        assert any(s["session_id"] == session_id for s in sessions)

        archived = memory.session.archive(session_id)
        assert archived["archived"] is True
        assert archived["summary_node_id"]
        created.append(archived["summary_node_id"])
    finally:
        _safe_forget(memory, created)


def test_store_batch_auto_edges_and_stream_live(memory: HybridMemory):
    created: List[str] = []
    token = f"sdk_batch_{uuid.uuid4().hex[:8]}"
    try:
        seed_id = memory.store(
            f"{token} seed memory for auto edge linking",
            metadata={"domain": "sdk_test"},
        )
        created.append(seed_id)

        batch_nodes = [
            {
                "id": f"{token}_n1",
                "text": f"{token} batch node one about retrieval systems",
                "metadata": {"domain": "sdk_test", "source": "batch"},
                "edges": [{"target_id": seed_id, "type": "derived_from", "weight": 0.9}],
            },
            {
                "id": f"{token}_n2",
                "text": f"{token} batch node two about retrieval systems",
                "metadata": {"domain": "sdk_test", "source": "batch"},
                "edges": [],
            },
        ]

        batch_result = memory.store_batch(batch_nodes)
        assert batch_result["created_nodes"] == 2
        assert batch_result["created_edges"] >= 1
        created.extend(batch_result["node_ids"])

        auto = memory.store_with_auto_edges(
            text=f"{token} retrieval systems with semantic linking",
            metadata={"domain": "sdk_test", "source": "auto"},
            auto_edge_config=AutoEdgeConfig(
                threshold=0.0,  # force creation for deterministic live test
                max_edges=2,
                allowed_edge_types=("derived_from", "related_to"),
            ),
        )
        created.append(auto["node_id"])
        assert auto["auto_edges_created"] >= 1
        assert all(e["type"] in ("derived_from", "analogous_to") for e in auto["edges"])

        callback_batches: List[int] = []

        def on_batch(batch: List[Dict[str, Any]]) -> None:
            callback_batches.append(len(batch))

        streamed = list(
            memory.recall_stream(
                query=token,
                top_k=20,
                batch_size=6,
                mode="vector",
                on_batch_callback=on_batch,
            )
        )
        assert len(streamed) >= 1
        assert all(len(batch) <= 6 for batch in streamed)
        assert sum(callback_batches) == sum(len(batch) for batch in streamed)

        schemas = memory.tools.get_schema()
        names = {tool["function"]["name"] for tool in schemas}
        assert "store_batch" in names
        assert "session_archive" in names
        assert "stats" in names
    finally:
        _safe_forget(memory, created)
