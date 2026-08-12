from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from config import Settings
from engine.edge_inference import infer_temporal_edges
from engine.salience import compute_salience
from engine.temporal import extract_time_range, temporal_relevance
from storage.graph_index import GraphIndex
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex


def _node(store, node_id, *, event_time=None, entities=None):
    embedding = np.zeros(4096, dtype=np.float32)
    store.create_node(
        node_id,
        f"memory {node_id}",
        {"entities": entities or []},
        embedding,
        embedding,
        event_time=event_time,
        memory_kind="world",
    )
    store.upsert_node_entities(node_id, entities or [])


def test_settings_reject_every_dimension_except_4096():
    assert Settings(embedding_dimension=4096).embedding_dimension == 4096
    with pytest.raises(ValueError, match="requires embedding_dimension=4096"):
        Settings(embedding_dimension=1024)


def test_vector_index_rejects_wrong_shape_and_non_finite_values():
    index = VectorIndex(dimension=4096)
    with pytest.raises(ValueError, match="requires exactly"):
        index.add("wrong", np.zeros(1024, dtype=np.float32))
    invalid = np.zeros(4096, dtype=np.float32)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        index.add("nan", invalid)


def test_normalized_entity_lookup_is_case_and_punctuation_insensitive(tmp_path):
    store = SQLiteStore(str(tmp_path / "store.db"))
    _node(store, "a", entities=["Acme Corp.", "Alice"])
    _node(store, "b", entities=["ACME   CORP"])

    assert store.search_nodes_by_entity("acme corp", exclude_id="b") == ["a"]
    assert [item["entity_key"] for item in store.get_node_entities("a")] == [
        "acme corp",
        "alice",
    ]


def test_parallel_graph_relations_are_preserved_and_rebuilt(tmp_path):
    store = SQLiteStore(str(tmp_path / "store.db"))
    _node(store, "a")
    _node(store, "b")
    store.create_edge("e1", "a", "b", "next_turn", 1.0)
    store.create_edge("e2", "a", "b", "same_session", 0.5)

    graph = GraphIndex()
    graph.rebuild_from_edges(store.get_all_edges())
    edges = graph.get_node_edges("a", direction="outgoing")

    assert {edge["type"] for edge in edges} == {"next_turn", "same_session"}
    assert graph.edge_count == 2


def test_graph_proximity_uses_edge_weight_not_only_hop_count():
    graph = GraphIndex()
    graph.add_edge("anchor", "strong", "supports", weight=0.9, edge_id="strong-edge")
    graph.add_edge("anchor", "weak", "supports", weight=0.1, edge_id="weak-edge")

    strong = graph.compute_weighted_proximity_score(
        "strong", ["anchor"], edge_type_weights={"supports": 1.0}
    )
    weak = graph.compute_weighted_proximity_score(
        "weak", ["anchor"], edge_type_weights={"supports": 1.0}
    )
    assert strong == pytest.approx(0.45)
    assert weak == pytest.approx(0.05)


def test_temporal_edges_decay_by_event_distance(tmp_path, monkeypatch):
    store = SQLiteStore(str(tmp_path / "store.db"))
    graph = GraphIndex()
    first = "2026-08-01T00:00:00+00:00"
    second = "2026-08-03T00:00:00+00:00"
    _node(store, "a", event_time=first)
    _node(store, "b", event_time=second)
    graph.add_node("a")
    graph.add_node("b")

    from config import settings

    monkeypatch.setattr(settings, "temporal_edges_enabled", True)
    monkeypatch.setattr(settings, "temporal_edge_window_days", 10.0)
    monkeypatch.setattr(settings, "temporal_edge_half_life_days", 2.0)
    monkeypatch.setattr(settings, "temporal_edge_max_per_node", 5)

    assert infer_temporal_edges("b", second, store, graph) == 1
    edge = store.get_node_edges("b", direction="outgoing")[0]
    assert edge["type"] == "temporally_near"
    assert edge["weight"] == pytest.approx(0.5, abs=1e-4)


def test_temporal_query_range_and_salience_access_state(tmp_path):
    target = extract_time_range(
        "What happened last week?", now=datetime(2026, 8, 12, tzinfo=timezone.utc)
    )
    assert target is not None
    assert temporal_relevance("2026-08-10", target) == 1.0
    assert temporal_relevance("2025-01-01", target) < 0.001

    store = SQLiteStore(str(tmp_path / "store.db"))
    graph = GraphIndex()
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _node(store, "recent", event_time=recent)
    _node(store, "old", event_time="2020-01-01T00:00:00+00:00")
    graph.add_node("recent")
    graph.add_node("old")
    assert store.record_access(["recent", "recent"]) == 1

    from config import settings

    assert compute_salience(store.get_node("recent"), graph, settings) > compute_salience(
        store.get_node("old"), graph, settings
    )


def test_archiving_preserves_provenance_but_removes_retrieval(tmp_path):
    store = SQLiteStore(str(tmp_path / "store.db"))
    _node(store, "source")
    _node(store, "summary")

    assert store.archive_nodes(["source"], "summary") == 1
    assert store.get_node("source")["archived_by"] == "summary"
    assert not store.is_node_retrievable("source")
    assert store.is_node_retrievable("summary")
    assert [node_id for node_id, _ in store.get_all_node_embeddings()] == ["summary"]
