"""Behavioral contracts for source-preserving sparse document keys."""

import numpy as np
import pytest

from storage.bm25_index import BM25Index, sparse_document_text
from storage.sqlite_store import SQLiteStore


def test_sparse_document_text_is_explicit_and_fail_closed():
    assert sparse_document_text("authoritative source", {}) == "authoritative source"
    assert (
        sparse_document_text(
            "authoritative source", {"sparse_text": "  Speaker: source  "}
        )
        == "Speaker: source"
    )

    with pytest.raises(ValueError, match="source text"):
        sparse_document_text("", {})
    with pytest.raises(ValueError, match="metadata.sparse_text"):
        sparse_document_text("source", {"sparse_text": "  "})
    with pytest.raises(ValueError, match="metadata.sparse_text"):
        sparse_document_text("source", {"sparse_text": 7})


def test_rebuild_uses_alternate_key_without_changing_authoritative_source(tmp_path):
    store = SQLiteStore(tmp_path / "source.db")
    embedding = np.zeros(4096, dtype=np.float32)
    store.create_node(
        node_id="turn-1",
        text="The expedition reached the alpine observatory.",
        metadata={
            "speaker": "Avery",
            "evidence_id": "D1:1",
            "sparse_text": "Avery: The expedition reached the alpine observatory.",
        },
        embedding=embedding,
    )

    node = store.get_node("turn-1")
    index = BM25Index()
    index.rebuild_from_nodes([node])

    assert index.search("Avery", top_k=1)[0][0] == "turn-1"
    assert node["text"] == "The expedition reached the alpine observatory."
    assert node["metadata"]["evidence_id"] == "D1:1"


def test_api_create_and_metadata_update_refresh_sparse_key_but_return_source(client):
    source = "The expedition reached the alpine observatory."
    created = client.post(
        "/nodes",
        json={
            "text": source,
            "metadata": {
                "benchmark_sample_id": "sparse-key-contract",
                "evidence_id": "D1:1",
                "sparse_text": "Avery sourcekeyalpha",
            },
        },
    )
    assert created.status_code == 201
    node_id = created.json()["id"]

    try:
        first = client.post(
            "/search/hybrid",
            json={
                "query_text": "sourcekeyalpha",
                "top_k": 5,
                "rerank_pool": 0,
                "search_mode": "sparse_only",
                "route_weights": False,
                "filter_metadata": {"benchmark_sample_id": "sparse-key-contract"},
            },
        )
        assert first.status_code == 200
        assert first.json()["results"][0]["node_id"] == node_id
        assert first.json()["results"][0]["text"] == source

        updated = client.put(
            f"/nodes/{node_id}",
            json={
                "metadata": {
                    "benchmark_sample_id": "sparse-key-contract",
                    "evidence_id": "D1:1",
                    "sparse_text": "Basil sourcekeybeta",
                }
            },
        )
        assert updated.status_code == 200
        assert updated.json()["text"] == source

        old_key = client.post(
            "/search/hybrid",
            json={
                "query_text": "sourcekeyalpha",
                "top_k": 5,
                "rerank_pool": 0,
                "search_mode": "sparse_only",
                "route_weights": False,
                "filter_metadata": {"benchmark_sample_id": "sparse-key-contract"},
            },
        )
        new_key = client.post(
            "/search/hybrid",
            json={
                "query_text": "sourcekeybeta",
                "top_k": 5,
                "rerank_pool": 0,
                "search_mode": "sparse_only",
                "route_weights": False,
                "filter_metadata": {"benchmark_sample_id": "sparse-key-contract"},
            },
        )
        assert old_key.status_code == 200
        assert all(row["node_id"] != node_id for row in old_key.json()["results"])
        assert new_key.status_code == 200
        assert new_key.json()["results"][0]["node_id"] == node_id
        assert new_key.json()["results"][0]["text"] == source
    finally:
        client.delete(f"/nodes/{node_id}")
