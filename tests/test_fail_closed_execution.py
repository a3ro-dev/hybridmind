"""Requested retrieval/index features may not silently downgrade or disappear."""

import numpy as np
import pytest

from engine.edge_inference import infer_cosine_edges, infer_temporal_edges
import storage.bm25_index as sparse_module
from storage.vector_index import VectorIndex


def test_sparse_factory_rejects_missing_requested_backend(monkeypatch):
    monkeypatch.setattr(sparse_module, "_HAS_BM25S", False)
    with pytest.raises(RuntimeError, match="bm25s.*requested"):
        sparse_module.create_sparse_index("bm25s")

    monkeypatch.setattr(sparse_module, "_HAS_FASTEMBED", False)
    with pytest.raises(RuntimeError, match="splade.*requested"):
        sparse_module.create_sparse_index("splade")

    with pytest.raises(ValueError, match="must be one of"):
        sparse_module.create_sparse_index("mystery")


def test_enabled_cosine_edge_inference_propagates_vector_failure():
    class BrokenVectorIndex:
        @staticmethod
        def search(*_args, **_kwargs):
            raise RuntimeError("vector unavailable")

    with pytest.raises(RuntimeError, match="vector unavailable"):
        infer_cosine_edges(
            "node", np.zeros(4096, dtype=np.float32), BrokenVectorIndex(),
            object(), object(),
        )


def test_enabled_temporal_edge_inference_propagates_lookup_failure(monkeypatch):
    class Config:
        temporal_edges_enabled = True
        temporal_edge_window_days = 30.0
        temporal_edge_max_per_node = 5
        temporal_edge_half_life_days = 7.0

    class BrokenStore:
        @staticmethod
        def find_temporal_neighbors(*_args, **_kwargs):
            raise RuntimeError("temporal index unavailable")

    monkeypatch.setattr("engine.edge_inference._settings", lambda: Config())
    with pytest.raises(RuntimeError, match="temporal index unavailable"):
        infer_temporal_edges(
            "node", "2026-01-01T00:00:00+00:00", BrokenStore(), object(),
        )


def test_faiss_compaction_refuses_to_drop_active_row_with_missing_cache():
    index = VectorIndex(dimension=4096, deletion_threshold=0.0)
    if not index._use_faiss:
        pytest.skip("raw reconstruction cache is specific to FAISS HNSW")
    first = np.zeros(4096, dtype=np.float32)
    first[0] = 1.0
    second = np.zeros(4096, dtype=np.float32)
    second[1] = 1.0
    index.add("first", first)
    index.add("second", second)
    second_row = index.reverse_map["second"]
    index._raw_vectors.pop(second_row)

    with pytest.raises(RuntimeError, match="refusing to drop"):
        index.remove("first")

    assert index.id_map[second_row] == "second"
