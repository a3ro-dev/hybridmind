"""Offline failure-injection tests for all-or-nothing bulk mutations."""

import asyncio

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.bulk import (
    BulkEdgeCreate,
    BulkEdgesRequest,
    BulkNodeCreate,
    BulkNodesRequest,
    BulkImportRequest,
    UnstructuredDataRequest,
    bulk_create_edges,
    bulk_create_nodes,
    bulk_import,
    process_unstructured_data,
)
from engine.edge_inference import infer_cosine_edges
from storage.bm25_index import BM25Index
from storage.graph_index import GraphIndex
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from config import settings


class Embeddings:
    def __init__(self):
        self.batch_sizes = []

    def embed_batch(self, texts, *_args):
        self.batch_sizes.append(_args[1])
        return np.ones((len(texts), 4096), dtype=np.float32)


class NonFiniteEmbeddings:
    def embed_batch(self, texts, *_args):
        result = np.ones((len(texts), 4096), dtype=np.float32)
        result[0, 0] = np.nan
        return result


class FailingBM25(BM25Index):
    def add_batch(self, batch):
        super().add_batch(batch[:1])
        raise RuntimeError("sensitive backend detail")


class FailingEdgeGraph(GraphIndex):
    def __init__(self):
        super().__init__()
        self.add_edge_calls = 0

    def add_edge(self, *args, **kwargs):
        self.add_edge_calls += 1
        if self.add_edge_calls >= 2:
            raise RuntimeError("sensitive graph detail")
        return super().add_edge(*args, **kwargs)


def stores(tmp_path, *, graph=None, bm25=None):
    return (
        SQLiteStore(str(tmp_path / "bulk.db")),
        VectorIndex(dimension=4096),
        graph or GraphIndex(),
        bm25 or BM25Index(),
    )


def test_duplicate_ids_are_rejected_before_mutation():
    with pytest.raises(ValidationError, match="Duplicate explicit node IDs"):
        BulkNodesRequest(nodes=[
            BulkNodeCreate(id="same", text="first"),
            BulkNodeCreate(id="same", text="second"),
        ])
    with pytest.raises(ValidationError, match="Duplicate explicit edge IDs"):
        BulkEdgesRequest(edges=[
            BulkEdgeCreate(id="same", source_id="a", target_id="b", type="x"),
            BulkEdgeCreate(id="same", source_id="a", target_id="b", type="x"),
        ])


def test_total_character_ceiling_rejects_before_embedding():
    embedding_engine = Embeddings()
    nodes = [BulkNodeCreate(text="x" * 50_000) for _ in range(21)]
    with pytest.raises(ValidationError, match="total character limit"):
        BulkNodesRequest(nodes=nodes)
    assert embedding_engine.batch_sizes == []


def test_nonfinite_embedding_batch_fails_before_sql_mutation(tmp_path):
    sqlite, vector, graph, bm25 = stores(tmp_path)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(bulk_create_nodes(
            BulkNodesRequest(nodes=[BulkNodeCreate(id="n1", text="finite required")]),
            sqlite_store=sqlite,
            vector_index=vector,
            graph_index=graph,
            embedding_engine=NonFiniteEmbeddings(),
            bm25_index=bm25,
        ))
    assert raised.value.status_code == 503
    assert sqlite.count_nodes() == vector.size == graph.node_count == bm25.size == 0


def test_bm25_failure_compensates_sql_vector_and_graph(tmp_path):
    sqlite, vector, graph, bm25 = stores(tmp_path, bm25=FailingBM25())
    with pytest.raises(HTTPException) as raised:
        asyncio.run(bulk_create_nodes(
            BulkNodesRequest(nodes=[
                BulkNodeCreate(id="n1", text="first consistent node"),
                BulkNodeCreate(id="n2", text="second consistent node"),
            ]),
            sqlite_store=sqlite,
            vector_index=vector,
            graph_index=graph,
            embedding_engine=Embeddings(),
            bm25_index=bm25,
        ))
    assert raised.value.status_code == 500
    assert "sensitive backend detail" not in raised.value.detail
    assert sqlite.count_nodes() == vector.size == graph.node_count == bm25.size == 0
    with sqlite._cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM node_versions")
        assert cursor.fetchone()[0] == 0


def test_bulk_rejects_malformed_temporal_metadata_before_embedding(tmp_path):
    sqlite, vector, graph, bm25 = stores(tmp_path)
    embeddings = Embeddings()
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            bulk_create_nodes(
                BulkNodesRequest(
                    nodes=[
                        BulkNodeCreate(
                            id="bad-time",
                            text="invalid temporal node",
                            metadata={"event_time": "2026-01-01 trailing prose"},
                        )
                    ]
                ),
                sqlite_store=sqlite,
                vector_index=vector,
                graph_index=graph,
                embedding_engine=embeddings,
                bm25_index=bm25,
            )
        )
    assert raised.value.status_code == 422
    assert embeddings.batch_sizes == []
    assert sqlite.count_nodes() == 0


def test_auto_edge_graph_failure_removes_sql_edge_and_propagates(tmp_path):
    class CandidateVector:
        @staticmethod
        def search(_embedding, top_k=10):
            return [("target", 0.99)]

    class RejectingGraph(GraphIndex):
        def add_edge(self, *args, **kwargs):
            raise RuntimeError("projection unavailable")

    sqlite = SQLiteStore(str(tmp_path / "auto-edge.db"))
    embedding = np.ones(4096, dtype=np.float32)
    sqlite.create_node("source", "source", {}, embedding)
    sqlite.create_node("target", "target", {}, embedding)

    with pytest.raises(RuntimeError, match="projection unavailable"):
        infer_cosine_edges(
            "source",
            embedding,
            CandidateVector(),
            sqlite,
            RejectingGraph(),
            threshold=0.5,
            max_edges=1,
        )
    assert sqlite.count_edges() == 0


def test_success_is_reported_only_after_every_primary_index_is_updated(monkeypatch, tmp_path):
    sqlite, vector, graph, bm25 = stores(tmp_path)
    embeddings = Embeddings()
    monkeypatch.setattr(settings, "embedding_batch_size", 7)
    result = asyncio.run(bulk_create_nodes(
        BulkNodesRequest(nodes=[
            BulkNodeCreate(id="n1", text="first consistent node", metadata={"session_id": "s"}),
            BulkNodeCreate(id="n2", text="second consistent node", metadata={"session_id": "s"}),
        ]),
        sqlite_store=sqlite,
        vector_index=vector,
        graph_index=graph,
        embedding_engine=embeddings,
        bm25_index=bm25,
    ))
    assert result.success and result.created == 2 and result.failed == 0
    assert sqlite.count_nodes() == vector.size == graph.node_count == bm25.size == 2
    assert sqlite.count_edges() == graph.edge_count == 2
    assert embeddings.batch_sizes == [7]


def test_enabled_colbert_failure_is_not_reported_as_success(monkeypatch, tmp_path):
    class EmptyColbertStore:
        def remove(self, _node_id):
            return False

    import api.dependencies
    monkeypatch.setattr(settings, "colbert_enabled", True)
    monkeypatch.setattr(api.dependencies, "get_colbert_store", lambda: EmptyColbertStore())
    sqlite, vector, graph, bm25 = stores(tmp_path)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(bulk_create_nodes(
            BulkNodesRequest(nodes=[BulkNodeCreate(id="n1", text="colbert required node")]),
            sqlite_store=sqlite,
            vector_index=vector,
            graph_index=graph,
            embedding_engine=Embeddings(),
            bm25_index=bm25,
        ))
    assert raised.value.status_code == 500
    assert sqlite.count_nodes() == vector.size == graph.node_count == bm25.size == 0


def test_compensation_rebuild_excludes_archived_nodes(tmp_path):
    bm25 = FailingBM25()
    sqlite, vector, graph, bm25 = stores(tmp_path, bm25=bm25)
    embedding = np.ones(4096, dtype=np.float32)
    sqlite.create_node("archived", "archived memory", {}, embedding)
    with sqlite._cursor() as cursor:
        cursor.execute("UPDATE nodes SET archived_at = CURRENT_TIMESTAMP WHERE id = ?", ("archived",))
    vector.add("archived", embedding)
    graph.add_node("archived")
    bm25.add("archived", "archived memory")

    with pytest.raises(HTTPException):
        asyncio.run(bulk_create_nodes(
            BulkNodesRequest(nodes=[BulkNodeCreate(id="new", text="new memory")]),
            sqlite_store=sqlite,
            vector_index=vector,
            graph_index=graph,
            embedding_engine=Embeddings(),
            bm25_index=bm25,
        ))
    assert sqlite.count_retrievable_nodes() == 0
    assert vector.size == graph.node_count == bm25.size == 0


def test_graph_failure_compensates_every_bulk_edge(tmp_path):
    graph = FailingEdgeGraph()
    sqlite, _vector, graph, _bm25 = stores(tmp_path, graph=graph)
    for node_id in ("a", "b"):
        sqlite.create_node(node_id, node_id, {}, np.ones(4096, dtype=np.float32))
        graph.add_node(node_id)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(bulk_create_edges(
            BulkEdgesRequest(edges=[
                BulkEdgeCreate(id="e1", source_id="a", target_id="b", type="x"),
                BulkEdgeCreate(id="e2", source_id="b", target_id="a", type="x"),
            ]),
            sqlite_store=sqlite,
            graph_index=graph,
        ))
    assert raised.value.status_code == 500
    assert "sensitive graph detail" not in raised.value.detail
    assert sqlite.count_edges() == graph.edge_count == 0


def test_combined_import_rolls_back_nodes_when_edges_are_invalid(tmp_path):
    sqlite, vector, graph, bm25 = stores(tmp_path)
    request = BulkImportRequest(
        nodes=[BulkNodeCreate(id="new", text="new consistent node")],
        edges=[BulkEdgeCreate(id="bad", source_id="new", target_id="missing", type="x")],
    )
    with pytest.raises(HTTPException) as raised:
        asyncio.run(bulk_import(
            request,
            sqlite_store=sqlite,
            vector_index=vector,
            graph_index=graph,
            embedding_engine=Embeddings(),
            bm25_index=bm25,
        ))
    assert raised.value.status_code == 500
    assert sqlite.count_nodes() == vector.size == graph.node_count == bm25.size == 0


def test_unstructured_edge_failure_rolls_back_nodes(monkeypatch, tmp_path):
    class FakeLLM:
        def process_unstructured(self, _text):
            return {
                "summary": "safe summary",
                "nodes": [
                    {"text": "first extracted node", "metadata": {}},
                    {"text": "second extracted node", "metadata": {}},
                ],
                "edges": [{"source_index": 0, "target_index": 1, "type": "relates_to", "weight": 0.5}],
            }

    import engine.llm
    monkeypatch.setattr(engine.llm, "LLMEngine", FakeLLM)
    sqlite, vector, graph, bm25 = stores(tmp_path, graph=FailingEdgeGraph())
    # Fail the first explicit relationship, not node registration.
    graph.add_edge_calls = 1
    with pytest.raises(HTTPException) as raised:
        asyncio.run(process_unstructured_data(
            UnstructuredDataRequest(text="long enough unstructured input"),
            sqlite_store=sqlite,
            vector_index=vector,
            graph_index=graph,
            embedding_engine=Embeddings(),
            bm25_index=bm25,
        ))
    assert raised.value.status_code == 500
    assert sqlite.count_nodes() == sqlite.count_edges() == 0
    assert vector.size == graph.node_count == graph.edge_count == bm25.size == 0


def test_unstructured_request_and_provider_errors_fail_closed(monkeypatch, tmp_path):
    with pytest.raises(ValidationError):
        UnstructuredDataRequest(text="x" * 12001)

    class FailingLLM:
        def process_unstructured(self, _text):
            raise RuntimeError("secret provider body")

    import engine.llm
    monkeypatch.setattr(engine.llm, "LLMEngine", FailingLLM)
    sqlite, vector, graph, bm25 = stores(tmp_path)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(process_unstructured_data(
            UnstructuredDataRequest(text="long enough unstructured input"),
            sqlite_store=sqlite,
            vector_index=vector,
            graph_index=graph,
            embedding_engine=Embeddings(),
            bm25_index=bm25,
        ))
    assert raised.value.status_code == 502
    assert "secret provider body" not in raised.value.detail
    assert sqlite.count_nodes() == 0
