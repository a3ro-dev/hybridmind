from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from config import Settings, settings
import engine.hybrid_ranker as hybrid_ranker_module
from engine.hybrid_ranker import HybridRanker
from engine.fusion import rrf_fuse
from engine.salience import compute_salience
from models.search import HybridSearchRequest
from storage.graph_index import GraphIndex
from storage.vector_index import VectorIndex


class _Store:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def is_node_retrievable(self, node_id):
        return node_id in self.nodes

    def record_access(self, *_args, **_kwargs):
        return None


class _VectorEngine:
    def __init__(self, results):
        self.results = results
        self.sqlite_store = _Store({r["node_id"]: {"id": r["node_id"], **r} for r in results})
        self.filters = []

    def search(self, *, filter_metadata=None, **_kwargs):
        self.filters.append(filter_metadata)
        results = [
            r for r in self.results
            if not filter_metadata or self._matches_filter(r["metadata"], filter_metadata)
        ]
        return results, 0.0, len(results)

    @staticmethod
    def _matches_filter(metadata, requested):
        return all(metadata.get(key) == value for key, value in requested.items())


class _GraphEngine:
    graph_index = GraphIndex()

    def __init__(self):
        self.proximity_calls = []

    def traverse(self, *_args, **_kwargs):
        return [], 0.0, 0

    def compute_proximity_scores(self, *, node_ids, **kwargs):
        self.proximity_calls.append(kwargs)
        return {node_id: 0.0 for node_id in node_ids}


class _BM25:
    def search(self, *_args, **_kwargs):
        return []

    def tokenize(self, text):
        return text.lower().split()


class _RankedBM25(_BM25):
    def __init__(self, hits, *, forbid_tokenize=False):
        self.hits = hits
        self.forbid_tokenize = forbid_tokenize

    def search(self, *_args, **_kwargs):
        return list(self.hits)

    def tokenize(self, text):
        if self.forbid_tokenize:
            raise AssertionError("sparse_only must not use token-overlap heuristics")
        return super().tokenize(text)


class _RecordingReranker:
    enabled = True

    def __init__(self):
        self.calls = 0
        self.pool_size = 0

    def rerank(self, _query, candidates, top_k=None):
        self.calls += 1
        self.pool_size = len(candidates)
        for candidate in candidates:
            candidate["rerank_score"] = 1.0
        return candidates


def _result(node_id, score, metadata=None, **node_fields):
    return {
        "node_id": node_id,
        "text": f"memory {node_id}",
        "metadata": metadata or {},
        "vector_score": score,
        **node_fields,
    }


def test_graph_conditioning_is_opt_in_by_default():
    assert Settings().use_graph_conditioned_embeddings is False


def test_salience_uses_precomputed_max_degree_without_full_scan():
    class _DegreeView:
        def __call__(self, node_id=None):
            if node_id is None:
                raise AssertionError("full degree scan must not run per candidate")
            return 2

    class _Graph:
        degree = _DegreeView()

        @staticmethod
        def has_node(_node_id):
            return True

    graph_index = type("Index", (), {"graph": _Graph()})()
    score = compute_salience(
        {"id": "node", "created_at": datetime.now(timezone.utc).isoformat()},
        graph_index,
        settings,
        max_degree=4,
    )
    assert 0.0 <= score <= 1.0


def test_hybrid_search_scans_graph_degree_once_for_all_candidates(monkeypatch):
    calls = {"full": 0, "node": 0}

    class _DegreeView:
        def __call__(self, node_id=None):
            if node_id is None:
                calls["full"] += 1
                return [("a", 3), ("b", 2), ("c", 1)]
            calls["node"] += 1
            return {"a": 3, "b": 2, "c": 1}[node_id]

    class _Graph:
        degree = _DegreeView()

        @staticmethod
        def has_node(_node_id):
            return True

    class _CountingGraphEngine(_GraphEngine):
        graph_index = type("Index", (), {"graph": _Graph()})()

    monkeypatch.setattr(settings, "salience_enabled", True)
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", False)
    vector = _VectorEngine([_result(node_id, 1.0) for node_id in ("a", "b", "c")])
    ranker = HybridRanker(vector, _CountingGraphEngine(), _BM25(), query_routing_enabled=False)
    ranker.search("query", search_mode="hybrid", track_access=False)
    assert calls == {"full": 1, "node": 3}


def test_vector_batch_and_rebuild_reject_before_mutation():
    index = VectorIndex(dimension=4096)
    good = np.zeros(4096, dtype=np.float32)
    index.add("existing", good)
    bad = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="requires exactly"):
        index.add_batch([("new", good), ("bad", bad)])
    assert index.reverse_map.keys() == {"existing"}

    invalid = good.copy()
    invalid[10] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        index.rebuild_from_embeddings([("bad", invalid)])
    assert index.reverse_map.keys() == {"existing"}


def test_vector_rebuild_backend_failure_keeps_previous_live_index(monkeypatch):
    import types
    import storage.vector_index as vector_module

    if not vector_module.FAISS_AVAILABLE:
        pytest.skip("FAISS replacement failure injection requires FAISS")
    index = VectorIndex(dimension=4096)
    existing = np.zeros(4096, dtype=np.float32)
    existing[0] = 1.0
    replacement = np.zeros(4096, dtype=np.float32)
    replacement[1] = 1.0
    index.add("existing", existing)
    old_backend = index.index

    class FailingReplacement:
        hnsw = types.SimpleNamespace(efSearch=0)

        @staticmethod
        def add(_vectors):
            raise RuntimeError("replacement allocation failed")

    monkeypatch.setattr(
        vector_module.faiss,
        "IndexHNSWFlat",
        lambda *_args, **_kwargs: FailingReplacement(),
    )
    with pytest.raises(RuntimeError, match="replacement allocation failed"):
        index.rebuild_from_embeddings([("replacement", replacement)])

    assert index.index is old_backend
    assert index.reverse_map == {"existing": 0}
    assert index.search(existing, top_k=1)[0][0] == "existing"


def test_vector_update_tombstones_only_the_old_generation():
    index = VectorIndex(dimension=4096, deletion_threshold=1.0)
    old = np.zeros(4096, dtype=np.float32)
    old[0] = 1.0
    new = np.zeros(4096, dtype=np.float32)
    new[1] = 1.0

    index.add("same-id", old)
    index.add("same-id", new)

    assert index.size == 1
    assert index.total_size == 2
    assert index.deleted_ids == {0}
    assert index.has_vector("same-id")
    assert index.get_vector("same-id") == pytest.approx(new)
    assert index.search(new, top_k=5) == [("same-id", pytest.approx(1.0))]


def test_vector_batch_rejects_duplicate_ids_before_mutation():
    index = VectorIndex(dimension=4096)
    embedding = np.zeros(4096, dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate node IDs"):
        index.add_batch([("duplicate", embedding), ("duplicate", embedding)])
    assert index.size == index.total_size == 0

def test_vector_index_constructor_rejects_non_contract_dimension():
    with pytest.raises(ValueError, match="dimension=4096"):
        VectorIndex(dimension=3)


def test_typed_graph_paths_are_directional_symmetric_and_confidence_aware():
    graph = GraphIndex()
    graph.add_edge("cause", "effect", "led_to", weight=1.0, confidence=0.4, edge_id="causal")
    assert graph.compute_weighted_proximity_score("effect", ["cause"]) == pytest.approx(0.2)
    assert graph.compute_weighted_proximity_score("cause", ["effect"]) == 0.0

    graph.add_edge("left", "right", "same_session", weight=1.0, edge_id="symmetric")
    assert graph.compute_weighted_proximity_score("left", ["right"]) == pytest.approx(0.5)


def test_future_graph_edge_is_not_traversable():
    graph = GraphIndex()
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    graph.add_edge("a", "b", "supports", valid_from=future, edge_id="future")
    assert graph.compute_weighted_proximity_score("b", ["a"]) == 0.0


def test_graph_edge_validity_supports_historical_as_of_queries():
    graph = GraphIndex()
    graph.add_edge(
        "a",
        "b",
        "supports",
        valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2026-02-01T00:00:00+00:00",
        edge_id="bounded",
    )
    january = datetime(2026, 1, 15, tzinfo=timezone.utc)
    february = datetime(2026, 2, 15, tzinfo=timezone.utc)
    assert graph.compute_weighted_proximity_score("b", ["a"], as_of=january) > 0.0
    assert graph.compute_weighted_proximity_score("b", ["a"], as_of=february) == 0.0


def test_graph_rejects_malformed_declared_validity_boundary():
    graph = GraphIndex()
    graph.add_edge("a", "b", "supports", valid_from="not-a-time", edge_id="bad-time")
    assert graph.compute_weighted_proximity_score("b", ["a"]) == 0.0


def test_graph_rebuild_failure_keeps_previous_live_graph():
    graph = GraphIndex()
    graph.add_edge("old-a", "old-b", "supports", edge_id="old")

    with pytest.raises(ValueError, match="self-loops"):
        graph.rebuild_from_edges(
            [
                {
                    "id": "invalid",
                    "source_id": "same",
                    "target_id": "same",
                    "type": "supports",
                    "weight": 1.0,
                    "metadata": {},
                }
            ]
        )

    assert graph.edge_count == 1
    assert graph.get_edge_by_id("old") is not None


def test_remove_edge_by_id_preserves_parallel_relation():
    graph = GraphIndex()
    graph.add_edge("a", "b", "next_turn", edge_id="one")
    graph.add_edge("a", "b", "same_session", edge_id="two")
    assert graph.remove_edge_by_id("one") is True
    assert graph.edge_count == 1
    assert graph.get_edge_by_id("two") is not None


def test_explicit_weights_are_not_overridden_and_min_score_is_normalized(monkeypatch):
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", False)
    vector = _VectorEngine([_result("a", 1.0)])
    ranker = HybridRanker(vector, _GraphEngine(), _BM25(), query_routing_enabled=True)
    results, _, _ = ranker.search(
        "when was this",
        search_mode="vector_only",
        vector_weight=1.0,
        graph_weight=0.0,
        bm25_boost_weight=0.0,
        min_score=0.9,
        track_access=False,
    )
    assert len(results) == 1
    assert results[0]["combined_score"] == pytest.approx(1.0)


def test_entity_route_does_not_hide_raw_turns_when_fact_extraction_is_off():
    vector = _VectorEngine([
        _result("raw", 1.0, {"type": "turn"}),
        _result("fact", 0.9, {"type": "extracted_fact"}),
    ])
    ranker = HybridRanker(vector, _GraphEngine(), _BM25(), query_routing_enabled=True)
    results, _, _ = ranker.search("who is Alice", search_mode="vector_only", track_access=False)
    assert [result["node_id"] for result in results] == ["raw", "fact"]
    assert vector.filters == [None]


def test_rrf_is_monotone_weighted_and_rejects_invalid_rank_lists():
    scores = rrf_fuse(
        {
            "dense": [("a", 1.0), ("b", 0.5)],
            "sparse": [("b", 2.0), ("a", 1.0)],
        },
        k=60,
        signal_weights={"dense": 1.0, "sparse": 0.0},
    )
    assert scores["a"] > scores["b"]
    with pytest.raises(ValueError, match="duplicate ID"):
        rrf_fuse({"dense": [("a", 1.0), ("a", 0.5)]})
    with pytest.raises(ValueError, match="non-negative"):
        rrf_fuse({"dense": [("a", 1.0)]}, signal_weights={"dense": -1.0})


def test_ranker_rejects_ambiguous_or_invalid_numeric_controls():
    ranker = HybridRanker(
        _VectorEngine([_result("a", 1.0)]),
        _GraphEngine(),
        _BM25(),
        query_routing_enabled=False,
    )
    with pytest.raises(ValueError, match="rerank_pool"):
        ranker.search("query", rerank_pool=-1, track_access=False)
    with pytest.raises(ValueError, match="greater than or equal to top_k"):
        ranker.search("query", top_k=10, rerank_pool=5, track_access=False)
    with pytest.raises(ValueError, match="vector_weight"):
        ranker.search("query", vector_weight=float("nan"), track_access=False)


def test_request_model_enforces_rerank_pool_contract():
    with pytest.raises(ValueError, match="greater than or equal to top_k"):
        HybridSearchRequest(query_text="query", top_k=10, rerank_pool=5)

    request = HybridSearchRequest(query_text="query", top_k=10, rerank_pool=0)
    assert request.rerank_pool == 0


def test_graph_only_anchor_must_exist_inside_requested_scope():
    vector = _VectorEngine([_result("anchor", 1.0, {"conversation": "a"})])
    ranker = HybridRanker(
        vector, _GraphEngine(), _BM25(), query_routing_enabled=False
    )
    with pytest.raises(ValueError, match="does not exist"):
        ranker.search(
            "query", search_mode="graph_only", anchor_nodes=["missing"],
            track_access=False,
        )
    with pytest.raises(ValueError, match="outside the requested metadata scope"):
        ranker.search(
            "query", search_mode="graph_only", anchor_nodes=["anchor"],
            filter_metadata={"conversation": "b"}, track_access=False,
        )


@pytest.mark.parametrize(
    ("query", "expected"),
    [("latest update", "new"), ("previous update", "old")],
)
def test_temporal_order_intent_ranks_latest_and_previous(monkeypatch, query, expected):
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", False)
    vector = _VectorEngine([
        _result("old", 0.8, {"timestamp": "2026-01-01T00:00:00+00:00"}),
        _result("new", 0.8, {"timestamp": "2026-02-01T00:00:00+00:00"}),
    ])
    ranker = HybridRanker(vector, _GraphEngine(), _BM25(), query_routing_enabled=True)
    results, _, _ = ranker.search(query, search_mode="hybrid", track_access=False)
    assert results[0]["node_id"] == expected


def test_controlled_mode_bypasses_reranker_but_hybrid_reranks_when_pool_equals_top_k(monkeypatch):
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", False)
    candidates = [_result(str(index), 1.0 - index / 10) for index in range(5)]
    vector = _VectorEngine(candidates)
    reranker = _RecordingReranker()
    ranker = HybridRanker(vector, _GraphEngine(), _BM25(), reranker=reranker, query_routing_enabled=False)

    ranker.search("query", top_k=5, rerank_pool=5, search_mode="vector_only", track_access=False)
    assert reranker.calls == 0

    results, _, _ = ranker.search("query", top_k=5, rerank_pool=5, search_mode="hybrid", track_access=False)
    assert reranker.calls == 1
    assert reranker.pool_size == 5
    assert all(result["rerank_attempted"] and result["rerank_applied"] for result in results)


def test_rerank_pool_is_the_exact_candidate_cap(monkeypatch):
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", False)
    candidates = [_result(str(index), 1.0 - index / 10) for index in range(8)]
    reranker = _RecordingReranker()
    ranker = HybridRanker(
        _VectorEngine(candidates), _GraphEngine(), _BM25(),
        reranker=reranker, query_routing_enabled=False,
    )

    results, _, _ = ranker.search(
        "query", top_k=2, rerank_pool=3, search_mode="hybrid", track_access=False,
    )
    assert reranker.pool_size == 3
    assert len(results) == 2


@pytest.mark.parametrize(
    "search_mode", ["vector_only", "sparse_only", "vector_sparse", "graph_only"]
)
def test_controlled_modes_bypass_temporal_validity_and_salience(
    monkeypatch, search_mode,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("controlled mode invoked hybrid lifecycle scoring")

    monkeypatch.setattr(settings, "query_time_expansion_enabled", True)
    monkeypatch.setattr(settings, "salience_enabled", True)
    monkeypatch.setattr(hybrid_ranker_module, "extract_time_range", forbidden)
    monkeypatch.setattr(hybrid_ranker_module, "temporal_relevance", forbidden)
    monkeypatch.setattr(hybrid_ranker_module, "validity_relevance", forbidden)
    monkeypatch.setattr(hybrid_ranker_module, "compute_salience", forbidden)

    expired = _result(
        "expired", 1.0, {"timestamp": "2020-01-01T00:00:00+00:00"},
        valid_until="2020-01-02T00:00:00+00:00",
    )
    bm25 = _RankedBM25([("expired", 4.0)])
    graph = _GraphEngine()
    ranker = HybridRanker(
        _VectorEngine([expired]), graph, bm25,
        query_routing_enabled=True, temporal_decay_enabled=True,
    )
    kwargs = {"anchor_nodes": ["expired"]} if search_mode == "graph_only" else {}
    results, _, _ = ranker.search(
        "latest memory", search_mode=search_mode, track_access=False, **kwargs,
    )

    assert [result["node_id"] for result in results] == ["expired"]
    assert results[0]["time_score"] == 0.0
    assert results[0]["salience_score"] == 0.0
    if search_mode == "graph_only":
        assert graph.proximity_calls[0]["temporal_decay"] is False


def test_hybrid_retains_graph_temporal_decay():
    graph = _GraphEngine()
    ranker = HybridRanker(
        _VectorEngine([_result("a", 1.0)]), graph, _BM25(),
        query_routing_enabled=False, temporal_decay_enabled=True,
    )
    ranker.search("query", search_mode="hybrid", rerank_pool=0, track_access=False)
    assert graph.proximity_calls[0]["temporal_decay"] is True


def test_sparse_only_pool_and_order_are_strictly_raw_bm25(monkeypatch):
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", False)
    top = _result("raw-top", 0.0)
    lower = [
        {
            **_result(f"overlap-{index}", 0.0),
            "text": f"needle {index}",
        }
        for index in range(100)
    ]
    vector = _VectorEngine([top, *lower])
    hits = [("raw-top", 100.0)] + [
        (f"overlap-{index}", 1.0 - index / 1000)
        for index in range(100)
    ]
    ranker = HybridRanker(
        vector,
        _GraphEngine(),
        _RankedBM25(hits, forbid_tokenize=True),
        query_routing_enabled=False,
    )

    results, _, candidate_count = ranker.search(
        "needle", top_k=3, rerank_pool=0, search_mode="sparse_only",
        fusion_mode="mlp", track_access=False,
    )

    assert [result["node_id"] for result in results] == [
        "raw-top", "overlap-0", "overlap-1",
    ]
    assert results[0]["bm25_score"] == 100.0
    assert all(result["vector_score"] == 0.0 for result in results)
    assert all(result["fusion_mode"] == "raw_bm25" for result in results)
    assert candidate_count == 100


def test_controlled_mode_does_not_initialize_post_rerankers(monkeypatch):
    import engine.gnn_reranker as gnn_module
    import engine.lexical_reranker as lexical_module
    import storage.colbert_store as colbert_module

    calls = {"colbert": 0, "gnn": 0, "lexical": 0}

    def colbert_enabled():
        calls["colbert"] += 1
        return False

    def get_gnn_reranker():
        calls["gnn"] += 1
        return None

    def lexical_rerank(*_args, **_kwargs):
        calls["lexical"] += 1
        return []

    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", True)
    monkeypatch.setattr(colbert_module, "colbert_enabled", colbert_enabled)
    monkeypatch.setattr(gnn_module, "get_gnn_reranker", get_gnn_reranker)
    monkeypatch.setattr(
        lexical_module, "rerank_with_query_local_lexical_rrf", lexical_rerank,
    )
    reranker = _RecordingReranker()
    ranker = HybridRanker(
        _VectorEngine([_result("a", 1.0)]), _GraphEngine(), _BM25(),
        reranker=reranker, query_routing_enabled=False,
    )

    ranker.search(
        "query", top_k=1, rerank_pool=1, search_mode="vector_only",
        track_access=False,
    )
    assert calls == {"colbert": 0, "gnn": 0, "lexical": 0}
    assert reranker.calls == 0


def test_invalid_fact_is_removed_before_reranker(monkeypatch):
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", False)
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    candidates = [
        _result("expired", 1.0, {}, valid_until=past),
        _result("active", 0.5),
    ]
    # SQLite node fields live outside metadata.
    vector = _VectorEngine(candidates)
    vector.sqlite_store.nodes["expired"]["valid_until"] = past
    reranker = _RecordingReranker()
    ranker = HybridRanker(vector, _GraphEngine(), _BM25(), reranker=reranker, query_routing_enabled=False)
    results, _, _ = ranker.search("query", top_k=1, rerank_pool=2, track_access=False)
    assert [result["node_id"] for result in results] == ["active"]
    assert reranker.pool_size == 1
