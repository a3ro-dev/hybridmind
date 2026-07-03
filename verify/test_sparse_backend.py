"""
Verify sparse search backend (BM25SBackend).
"""
import pytest
from storage.bm25_index import BM25SBackend, _HAS_BM25S, _HAS_PYSTEMMER

def test_sparse_backend_type(db_manager):
    if _HAS_BM25S:
        assert isinstance(db_manager.bm25_index, BM25SBackend)
    else:
        pytest.skip("bm25s is not installed in the environment (should be installed)")

def test_sparse_search(client):
    # Clear DB first
    client.post("/admin/clear")

    # Ingest some nodes
    client.post("/nodes", json={"text": "The quick brown fox jumps over the lazy dog."})
    client.post("/nodes", json={"text": "Artificial intelligence and machine learning are growing fields."})

    # Search with keyword matching
    resp = client.post("/search/hybrid", json={
        "query_text": "machine learning",
        "top_k": 5
    })
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0
    # The machine learning node should be first
    assert "Artificial intelligence" in results[0]["text"]
