from datetime import datetime, timezone

import numpy as np
import pytest

import main
from config import settings
from engine.temporal import extract_time_range, validity_relevance
from storage.bm25_index import BM25Index
from storage.sqlite_store import SQLiteStore


def test_sqlite_rejects_wrong_width_and_non_finite_embeddings(tmp_path):
    store = SQLiteStore(str(tmp_path / "store.db"))
    with pytest.raises(ValueError, match=r"shape \(4096,\)"):
        store.create_node("short", "bad", {}, np.zeros(3, dtype=np.float32))

    invalid = np.zeros(4096, dtype=np.float32)
    invalid[17] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        store.create_node("infinite", "bad", {}, invalid)


def test_sqlite_rejects_corrupt_persisted_embedding_width(tmp_path):
    store = SQLiteStore(str(tmp_path / "store.db"))
    with store._cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO nodes (id, text, metadata, embedding, raw_embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            ("corrupt", "bad", "{}", np.zeros(3, dtype=np.float32).tobytes(), None),
        )
    with pytest.raises(ValueError, match="persisted embedding has"):
        store.get_node("corrupt")


def test_python_bm25_replacement_and_removal_do_not_duplicate_documents():
    index = BM25Index()
    index.add("node", "oldtoken only")
    index.add("node", "newtoken only")

    assert index.doc_count == 1
    assert index.doc_ids == ["node"]
    assert index.search("oldtoken") == []
    assert index.search("newtoken")[0][0] == "node"
    assert index.remove("node") is True
    assert index.remove("node") is False
    assert index.doc_count == 0


def test_temporal_before_after_calendar_periods_and_hard_validity():
    now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    before = extract_time_range("What was true before 2025?", now=now)
    assert before is not None
    assert before.end == datetime(2025, 1, 1, tzinfo=timezone.utc)

    after = extract_time_range("What happened after March 2025?", now=now)
    assert after is not None
    assert after.start == datetime(2025, 4, 1, tzinfo=timezone.utc)

    last_month = extract_time_range("What changed last month?", now=now)
    assert last_month is not None
    assert last_month.start == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert last_month.end == datetime(2026, 8, 1, tzinfo=timezone.utc)

    expired = {"valid_from": "2024-01-01", "valid_until": "2025-01-01"}
    future = {"valid_from": "2027-01-01", "valid_until": None}
    assert validity_relevance(expired, None, now=now) == 0.0
    assert validity_relevance(future, None, now=now) == 0.0


def test_single_sentence_parent_is_not_duplicated_as_a_child(client):
    response = client.post("/nodes", json={"text": "A single atomic memory."})
    assert response.status_code == 201
    node_id = response.json()["id"]

    from api.dependencies import get_sqlite_store

    assert get_sqlite_store().get_sentence_children(node_id) == []
    assert client.delete(f"/nodes/{node_id}").status_code == 200


def test_update_and_delete_keep_sentence_indexes_synchronized(client):
    created = client.post(
        "/nodes",
        json={"text": "Oldalpha first statement. Oldbeta second statement."},
    )
    assert created.status_code == 201
    node_id = created.json()["id"]

    from api.dependencies import get_bm25_index, get_sqlite_store

    store = get_sqlite_store()
    assert [node["text"] for node in store.get_sentence_children(node_id)] == [
        "Oldalpha first statement.",
        "Oldbeta second statement.",
    ]

    updated = client.put(
        f"/nodes/{node_id}",
        json={"text": "Newgamma first statement. Newdelta second statement."},
    )
    assert updated.status_code == 200
    children = store.get_sentence_children(node_id)
    assert [node["text"] for node in children] == [
        "Newgamma first statement.",
        "Newdelta second statement.",
    ]

    sparse = get_bm25_index()
    old_ids = {result_id for result_id, _ in sparse.search("Oldalpha", top_k=20)}
    new_ids = {result_id for result_id, _ in sparse.search("Newgamma", top_k=20)}
    assert node_id not in old_ids
    assert not old_ids.intersection({f"{node_id}_0", f"{node_id}_1"})
    assert node_id in new_ids or f"{node_id}_0" in new_ids

    deleted = client.delete(f"/nodes/{node_id}")
    assert deleted.status_code == 200
    assert store.get_sentence_children(node_id) == []
    remaining = {result_id for result_id, _ in sparse.search("Newgamma", top_k=20)}
    assert node_id not in remaining
    assert not remaining.intersection({f"{node_id}_0", f"{node_id}_1"})


def test_sqlite_outer_transaction_rolls_back_nested_store_operations(tmp_path):
    store = SQLiteStore(str(tmp_path / "transaction.db"))
    embedding = np.zeros(4096, dtype=np.float32)
    with pytest.raises(RuntimeError, match="abort"):
        with store.transaction():
            store.create_node("rolled-back", "temporary", {}, embedding)
            raise RuntimeError("abort")
    assert store.get_node("rolled-back") is None


def test_sqlite_session_predecessor_uses_container_and_turn_order(tmp_path):
    store = SQLiteStore(str(tmp_path / "session-order.db"))
    for node_id, container, turn_index in (
        ("a-2", "a", 2),
        ("b-1", "b", 1),
        ("a-0", "a", 0),
    ):
        store.create_node(
            node_id,
            node_id,
            {
                "session_id": "shared",
                "container_tag": container,
                "turn_index": turn_index,
            },
        )

    previous = store.get_latest_node_by_session(
        "shared", container_tag="a", before_turn_index=2
    )
    assert previous["id"] == "a-0"


def test_untagged_session_predecessor_never_wildcards_tagged_containers(tmp_path):
    store = SQLiteStore(str(tmp_path / "default-session-scope.db"))
    store.create_node(
        "untagged",
        "untagged",
        {"session_id": "shared", "container_tag": "", "turn_index": 0},
    )
    store.create_node(
        "tagged",
        "tagged",
        {"session_id": "shared", "container_tag": "other", "turn_index": 1},
    )

    previous = store.get_latest_node_by_session(
        "shared", container_tag=None, before_turn_index=2
    )
    assert previous["id"] == "untagged"


@pytest.mark.parametrize(
    ("valid_from", "valid_until"),
    [
        ("2026-02-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        ("not-a-time", "2026-01-01T00:00:00+00:00"),
    ],
)
def test_sqlite_rejects_invalid_validity_intervals(
    tmp_path, valid_from, valid_until
):
    store = SQLiteStore(str(tmp_path / "validity.db"))
    with pytest.raises(ValueError):
        store.create_node(
            "invalid",
            "invalid temporal fact",
            {},
            valid_from=valid_from,
            valid_until=valid_until,
        )
    assert store.get_node("invalid") is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_time", "on 2026-01-01"),
        ("valid_from", "2026-01-01 trailing prose"),
        ("valid_until", "not-a-time"),
    ],
)
def test_sqlite_rejects_each_malformed_one_sided_temporal_value(
    tmp_path, field, value
):
    store = SQLiteStore(str(tmp_path / f"strict-{field}.db"))
    with pytest.raises(ValueError, match="complete ISO-8601"):
        store.create_node("invalid", "invalid time", {}, **{field: value})
    assert store.get_node("invalid") is None


def test_sqlite_edge_rejects_malformed_one_sided_validity(tmp_path):
    store = SQLiteStore(str(tmp_path / "strict-edge-time.db"))
    store.create_node("source", "source", {})
    store.create_node("target", "target", {})
    with pytest.raises(ValueError, match="complete ISO-8601"):
        store.create_edge(
            "invalid-edge",
            "source",
            "target",
            "supports",
            valid_until="2026-01-01 and later",
        )
    assert store.count_edges() == 0


def test_sqlite_canonicalizes_temporal_values_to_utc(tmp_path):
    store = SQLiteStore(str(tmp_path / "canonical-time.db"))
    result = store.create_node(
        "canonical",
        "canonical time",
        {},
        event_time="2026-08-13T12:30:00+05:30",
        valid_from="2026-08-13",
    )
    assert result["event_time"] == "2026-08-13T07:00:00+00:00"
    assert result["valid_from"] == "2026-08-13T00:00:00+00:00"


def test_node_api_rejects_malformed_time_before_storage(client):
    response = client.post(
        "/nodes",
        json={"text": "invalid API time", "event_time": "sometime in 2026-01-01"},
    )
    assert response.status_code == 422


def test_soft_delete_retains_history_until_hard_delete_erases_it(tmp_path):
    store = SQLiteStore(str(tmp_path / "hard-erasure.db"))
    store.create_node("memory", "sensitive memory", {})
    assert store.soft_delete_node("memory") is True
    with store._cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM node_versions WHERE node_id = 'memory'")
        assert cursor.fetchone()[0] == 1

    assert store.hard_delete_soft_deleted_nodes() == 1
    with store._cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM node_versions WHERE node_id = 'memory'")
        assert cursor.fetchone()[0] == 0


def test_forget_erases_summaries_derived_from_the_source(tmp_path):
    store = SQLiteStore(str(tmp_path / "derived-erasure.db"))
    store.create_node("source", "sensitive source", {})
    store.create_node("summary", "summary containing sensitive source", {})
    store.create_edge(
        "provenance",
        "summary",
        "source",
        "derived_from",
        1.0,
    )

    erased_ids, _ = store.erase_node_family("source")
    assert set(erased_ids) == {"source", "summary"}
    with store._cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM nodes")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM node_versions")
        assert cursor.fetchone()[0] == 0


def test_sdk_forget_endpoint_erases_current_and_versioned_text(client):
    created = client.post(
        "/nodes", json={"text": "First sensitive sentence. Second sensitive sentence."}
    )
    assert created.status_code == 201
    node_id = created.json()["id"]

    from api.dependencies import get_sqlite_store

    store = get_sqlite_store()
    child_ids = [node["id"] for node in store.get_sentence_children(node_id)]
    assert client.delete(f"/nodes/{node_id}").status_code == 200
    erased_ids = [node_id, *child_ids]
    placeholders = ",".join("?" for _ in erased_ids)
    with store._cursor() as cursor:
        cursor.execute(
            f"SELECT COUNT(*) FROM nodes WHERE id IN ({placeholders})", erased_ids
        )
        assert cursor.fetchone()[0] == 0
        cursor.execute(
            f"SELECT COUNT(*) FROM node_versions WHERE node_id IN ({placeholders})",
            erased_ids,
        )
        assert cursor.fetchone()[0] == 0


def test_bulk_clear_erases_version_history(client):
    created = client.post("/nodes", json={"text": "clear this historical text"})
    assert created.status_code == 201
    response = client.delete("/bulk/clear")
    assert response.status_code == 200

    from api.dependencies import get_sqlite_store

    with get_sqlite_store()._cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM nodes")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM node_versions")
        assert cursor.fetchone()[0] == 0


def test_sqlite_preserves_valid_and_asserted_time_history(tmp_path):
    store = SQLiteStore(str(tmp_path / "bitemporal.db"))
    store.create_node(
        "fact",
        "Akshat lives in Delhi",
        {"location": "Delhi"},
        valid_from="2025-01-01T00:00:00+00:00",
        valid_until="2026-02-01T00:00:00+00:00",
    )
    with store._cursor() as cursor:
        cursor.execute(
            "SELECT asserted_from FROM node_versions WHERE node_id = ? AND version = 1",
            ("fact",),
        )
        first_assertion = cursor.fetchone()[0]

    store.update_node(
        "fact",
        text="Akshat lives in Bengaluru",
        metadata={"location": "Bengaluru"},
        valid_from="2026-02-01T00:00:00+00:00",
        valid_until="2030-01-01T00:00:00+00:00",
    )

    historical = store.get_node_version(
        "fact",
        valid_at="2025-06-01T00:00:00+00:00",
        asserted_at=first_assertion,
    )
    current = store.get_node_version(
        "fact", valid_at="2026-06-01T00:00:00+00:00"
    )
    assert historical["text"] == "Akshat lives in Delhi"
    assert historical["metadata"]["location"] == "Delhi"
    assert current["text"] == "Akshat lives in Bengaluru"
    assert current["metadata"]["location"] == "Bengaluru"


def test_fact_ingestion_rolls_back_sql_and_indexes_on_mid_batch_failure(
    client, monkeypatch
):
    manager = main.get_db_manager()
    before = {
        "sql": manager.sqlite_store.count_nodes(),
        "vector": manager.vector_index.size,
        "graph": manager.graph_index.node_count,
        "bm25": manager.bm25_index.size,
    }

    async def extracted(_session_id, _turns):
        return [
            {
                "fact": "Akshat prefers exact evidence identifiers.",
                "entities": ["Akshat"],
                "memory_kind": "observation",
                "confidence": 0.9,
            },
            {
                "fact": "Akshat measures retrieval latency distributions.",
                "entities": ["Akshat"],
                "memory_kind": "observation",
                "confidence": 0.9,
            },
        ]

    original_add = manager.bm25_index.add
    add_calls = 0

    def fail_second_add(node_id, text):
        nonlocal add_calls
        add_calls += 1
        if add_calls == 2:
            raise RuntimeError("private backend failure")
        return original_add(node_id, text)

    monkeypatch.setattr(settings, "fact_extraction_enabled", True)
    monkeypatch.setattr("engine.llm_client.is_configured", lambda: True)
    monkeypatch.setattr(main, "_get_or_extract_facts", extracted)
    monkeypatch.setattr(manager.bm25_index, "add", fail_second_add)

    response = client.post(
        "/ingest/session-facts",
        json={
            "session_id": "atomic-facts",
            "turns": [{"speaker": "user", "text": "remember both facts"}],
        },
    )

    assert response.status_code == 500
    assert "private backend failure" not in response.text
    assert manager.sqlite_store.count_nodes() == before["sql"]
    assert manager.vector_index.size == before["vector"]
    assert manager.graph_index.node_count == before["graph"]
    assert manager.bm25_index.size == before["bm25"]


def test_edge_creation_rolls_back_sql_when_graph_projection_fails(
    client, monkeypatch
):
    manager = main.get_db_manager()
    first = client.post("/nodes", json={"text": "Atomic edge source."})
    second = client.post("/nodes", json={"text": "Atomic edge target."})
    assert first.status_code == second.status_code == 201
    source_id = first.json()["id"]
    target_id = second.json()["id"]
    before_sql = manager.sqlite_store.count_edges()
    before_graph = manager.graph_index.edge_count

    original_add = manager.graph_index.add_edge
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("private graph failure")
        return original_add(*args, **kwargs)

    monkeypatch.setattr(manager.graph_index, "add_edge", fail_once)
    response = client.post(
        "/edges",
        json={
            "source_id": source_id,
            "target_id": target_id,
            "type": "supports",
            "weight": 1.0,
        },
    )

    assert response.status_code == 500
    assert "private graph failure" not in response.text
    assert manager.sqlite_store.count_edges() == before_sql
    assert manager.graph_index.edge_count == before_graph

    assert client.delete(f"/nodes/{source_id}").status_code == 200
    assert client.delete(f"/nodes/{target_id}").status_code == 200


def test_oversized_image_is_rejected_before_remote_provider_call(
    client, monkeypatch
):
    provider_called = False

    def forbidden_provider():
        nonlocal provider_called
        provider_called = True
        raise AssertionError("remote provider must not be resolved")

    monkeypatch.setattr(settings, "image_ingest_max_base64_chars", 3)
    monkeypatch.setattr(
        "engine.image_embedding.get_image_embedding_engine", forbidden_provider
    )
    response = client.post(
        "/nodes/image",
        json={"image_b64": "AAAA", "caption": "bounded image"},
    )

    assert response.status_code == 413
    assert provider_called is False
