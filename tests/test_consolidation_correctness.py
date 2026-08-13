import json
from types import SimpleNamespace

import numpy as np

from engine import consolidation
from storage.sqlite_store import SQLiteStore


class _RecordingIndex:
    def __init__(self):
        self.added = []

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))

    def remove(self, node_id):
        before = len(self.added)
        self.added = [entry for entry in self.added if not entry[0] or entry[0][0] != node_id]
        return len(self.added) != before


class _RecordingGraph(_RecordingIndex):
    def add_node(self, *args, **kwargs):
        self.add(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        self.add(*args, **kwargs)

    def remove_node(self, node_id):
        before = len(self.added)
        self.added = [
            entry
            for entry in self.added
            if not (
                (entry[0] and entry[0][0] == node_id)
                or entry[1].get("source_id") == node_id
                or entry[1].get("target_id") == node_id
            )
        ]
        return len(self.added) != before


class _EmbeddingEngine:
    def embed(self, _text):
        return np.zeros(4096, dtype=np.float32)


def test_summary_uses_every_fact_and_honors_model(monkeypatch):
    calls = []

    def fake_call(messages, max_tokens=512, model=None):
        calls.append((messages, max_tokens, model))
        return f"summary-{len(calls)}"

    monkeypatch.setattr(consolidation, "_call_llm", fake_call)
    facts = [f"fact-{index}" for index in range(51)]

    result = consolidation.llm_summarize(facts, model="explicit-model")

    assert result == "summary-4"
    assert len(calls) == 4
    first_stage_text = "\n".join(call[0][1]["content"] for call in calls[:3])
    assert all(fact in first_stage_text for fact in facts)
    assert {call[2] for call in calls} == {"explicit-model"}


def test_summary_failure_is_not_disguised_as_truncated_success(monkeypatch):
    monkeypatch.setattr(consolidation, "_call_llm", lambda *args, **kwargs: None)
    facts = [f"fact-{index}" for index in range(30)]
    assert consolidation.llm_summarize(facts) == ""


def test_similarity_alone_never_means_contradiction():
    class MustNotEmbed:
        def embed(self, _text):
            raise AssertionError("semantic similarity is not contradiction evidence")

    nodes = [
        {
            "id": "same-meaning",
            "text": "Akshat enjoys building retrieval systems.",
            "embedding": np.ones(4096, dtype=np.float32),
        }
    ]
    assert (
        consolidation.check_contradiction(
            "Akshat likes building retrieval systems.", nodes, MustNotEmbed()
        )
        is None
    )


def test_structured_slot_update_is_detected_without_embedding():
    nodes = [{"id": "old", "text": "Akshat location is Delhi"}]
    assert (
        consolidation.check_contradiction(
            "Akshat location is Bengaluru", nodes, embedding_engine=None
        )
        == "old"
    )


def test_consolidation_scopes_same_session_name_by_container(tmp_path, monkeypatch):
    store = SQLiteStore(str(tmp_path / "store.db"))
    for container, suffix in (("corpus-a", "a"), ("corpus-b", "b")):
        for index in range(2):
            store.create_node(
                node_id=f"{suffix}-{index}",
                text=f"fact {suffix} {index}",
                metadata={
                    "type": "extracted_fact",
                    "session_id": "shared-name",
                    "container_tag": container,
                },
            )

    monkeypatch.setattr(
        consolidation,
        "llm_summarize",
        lambda facts, model=None: " | ".join(facts),
    )
    manager = SimpleNamespace(
        sqlite_store=store,
        embedding_engine=_EmbeddingEngine(),
        vector_index=_RecordingIndex(),
        graph_index=_RecordingGraph(),
        bm25_index=_RecordingIndex(),
    )

    result = consolidation.consolidate_sessions(
        manager, min_facts=2, max_age_hours=0
    )

    assert result["summaries_created"] == 2
    summaries = []
    with store._cursor() as cursor:
        cursor.execute(
            "SELECT text, metadata FROM nodes "
            "WHERE json_extract(metadata, '$.type') = 'session_summary'"
        )
        summaries = cursor.fetchall()
    assert len(summaries) == 2
    summary_metadata = [json.loads(row["metadata"]) for row in summaries]
    assert {metadata["container_tag"] for metadata in summary_metadata} == {
        "corpus-a",
        "corpus-b",
    }
    assert all(metadata["source_count"] == 2 for metadata in summary_metadata)
    for metadata in summary_metadata:
        summary = store.get_node(
            next(
                row["id"]
                for row in store.list_nodes(limit=100)
                if row["metadata"].get("source_fingerprint_sha256")
                == metadata["source_fingerprint_sha256"]
            )
        )
        edges = store.get_node_edges(summary["id"], direction="outgoing")
        assert len(edges) == metadata["source_count"]
        assert {edge["type"] for edge in edges} == {"derived_from"}
        assert all(edge["metadata"]["lossy"] is True for edge in edges)


def test_consolidation_rolls_back_sql_and_projections_when_provenance_fails(
    tmp_path, monkeypatch
):
    store = SQLiteStore(str(tmp_path / "store.db"))
    for index in range(2):
        store.create_node(
            node_id=f"source-{index}",
            text=f"source fact {index}",
            metadata={"type": "extracted_fact", "session_id": "session"},
        )

    monkeypatch.setattr(consolidation, "llm_summarize", lambda *_args, **_kwargs: "summary")
    vector = _RecordingIndex()
    graph = _RecordingGraph()
    bm25 = _RecordingIndex()
    original_add_edge = graph.add_edge
    calls = 0

    def fail_second_edge(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("projection failure")
        return original_add_edge(*args, **kwargs)

    graph.add_edge = fail_second_edge
    manager = SimpleNamespace(
        sqlite_store=store,
        embedding_engine=_EmbeddingEngine(),
        vector_index=vector,
        graph_index=graph,
        bm25_index=bm25,
    )

    result = consolidation.consolidate_sessions(manager, min_facts=2, max_age_hours=0)

    assert result["summaries_created"] == 0
    assert len(result["failures"]) == 1
    summaries = [
        node
        for node in store.list_nodes(limit=100)
        if node["metadata"].get("type") == "session_summary"
    ]
    assert summaries == []
    assert vector.added == graph.added == bm25.added == []


def test_consolidation_refuses_lossy_source_archival(tmp_path):
    manager = SimpleNamespace(sqlite_store=SQLiteStore(str(tmp_path / "store.db")))
    try:
        consolidation.consolidate_sessions(manager, archive_sources=True)
    except ValueError as exc:
        assert "cannot replace exact source facts" in str(exc)
    else:
        raise AssertionError("lossy source archival must be rejected")


def test_importance_score_uses_precomputed_degree_without_full_graph_scan():
    class Graph:
        def has_node(self, node_id):
            return True

        def degree(self, node_id=None):
            if node_id is None:
                raise AssertionError("full graph degree scan must not occur")
            return 2

    node = {
        "id": "node",
        "event_time": "2026-08-13T00:00:00+00:00",
        "created_at": "2026-08-13T00:00:00+00:00",
        "access_count": 1,
    }
    manager = SimpleNamespace(
        sqlite_store=SimpleNamespace(get_node=lambda node_id: node),
        graph_index=SimpleNamespace(graph=Graph()),
    )
    score = consolidation.importance_score(
        "node", manager, max_graph_degree=4
    )
    assert 0.0 < score <= 1.0
