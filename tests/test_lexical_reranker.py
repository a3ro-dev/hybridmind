import pytest

from engine.lexical_reranker import BoundedTokenSetCache, rerank_with_query_local_lexical_rrf


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def test_query_local_lexical_rrf_promotes_under_ranked_exact_match():
    candidates = [
        {"node_id": "base", "text": "general project discussion", "combined_score": 0.9},
        {"node_id": "target", "text": "atlas launch project", "combined_score": 0.1},
        {"node_id": "noise", "text": "weather report", "combined_score": 0.2},
    ]

    ranked = rerank_with_query_local_lexical_rrf(
        "atlas project",
        candidates,
        _tokenize,
        lexical_weight=0.75,
        pool_size=3,
    )

    assert ranked[0]["node_id"] == "target"
    assert ranked[0]["lexical_rerank_score"] > 0
    assert "pre_lexical_combined_score" in ranked[0]


def test_query_local_lexical_rrf_is_bounded_and_score_sorted():
    candidates = [
        {"node_id": str(index), "text": f"term {index}", "combined_score": 1.0 / (index + 1)}
        for index in range(10)
    ]

    ranked = rerank_with_query_local_lexical_rrf(
        "term",
        candidates,
        _tokenize,
        pool_size=4,
    )

    assert len(ranked) == 4
    assert [item["combined_score"] for item in ranked] == sorted(
        (item["combined_score"] for item in ranked), reverse=True
    )


def test_query_local_lexical_rrf_validates_weight():
    with pytest.raises(ValueError, match="lexical_weight"):
        rerank_with_query_local_lexical_rrf(
            "query",
            [{"text": "query", "combined_score": 1.0}],
            _tokenize,
            lexical_weight=1.1,
        )


def test_bounded_token_set_cache_is_reused_and_evicts():
    calls = []

    def tokenize(text):
        calls.append(text)
        return text.lower().split()

    cache = BoundedTokenSetCache(tokenize, max_entries=2)
    assert cache.get("one two") == frozenset({"one", "two"})
    assert cache.get("one two") == frozenset({"one", "two"})
    cache.get("three")
    cache.get("four")
    cache.get("one two")

    assert calls == ["one two", "three", "four", "one two"]
    assert cache.info() == {"entries": 2, "max_entries": 2, "hits": 1, "misses": 4}


def test_cached_lexical_ranking_is_identical_to_uncached():
    candidates = [
        {"node_id": str(index), "text": f"term {index}", "combined_score": 1.0 / (index + 1)}
        for index in range(10)
    ]
    uncached = rerank_with_query_local_lexical_rrf(
        "term", [dict(candidate) for candidate in candidates], _tokenize, pool_size=10
    )
    cache = BoundedTokenSetCache(_tokenize, max_entries=20)
    cached = rerank_with_query_local_lexical_rrf(
        "term",
        [dict(candidate) for candidate in candidates],
        _tokenize,
        pool_size=10,
        document_term_cache=cache,
    )
    assert [item["node_id"] for item in cached] == [item["node_id"] for item in uncached]
    assert [item["combined_score"] for item in cached] == [item["combined_score"] for item in uncached]


def test_hybrid_ranker_applies_lexical_stage_before_cross_encoder(monkeypatch):
    from config import settings
    from engine.hybrid_ranker import HybridRanker

    class Store:
        def get_node(self, _node_id):
            return None

    class VectorEngine:
        sqlite_store = Store()

        def search(self, **_kwargs):
            results = [
                {
                    "node_id": f"base-{index}",
                    "text": f"generic memory {index}",
                    "metadata": {},
                    "vector_score": 1.0 - index * 0.01,
                }
                for index in range(29)
            ]
            results.append(
                {
                    "node_id": "target",
                    "text": "atlas launch project",
                    "metadata": {},
                    "vector_score": 0.1,
                }
            )
            return results, 0.0, len(results)

    class GraphEngine:
        graph_index = None

        def compute_proximity_scores(self, *, node_ids, **_kwargs):
            return {node_id: 0.0 for node_id in node_ids}

    class BM25:
        def search(self, _query, *, top_k):
            assert top_k == 5000
            return []

        def tokenize(self, text):
            return text.lower().split()

    class RecordingReranker:
        def __init__(self):
            self.seen_ids = []

        def rerank(self, _query, candidates, top_k=None):
            assert top_k is None
            self.seen_ids = [candidate["node_id"] for candidate in candidates]
            return candidates

    monkeypatch.setattr("engine.query_decomposition.decompose_query", lambda _query: [])
    monkeypatch.setattr(settings, "local_lexical_rerank_enabled", True)
    monkeypatch.setattr(settings, "local_lexical_rerank_pool_size", 30)
    monkeypatch.setattr(settings, "local_lexical_rerank_weight", 1.0)

    reranker = RecordingReranker()
    ranker = HybridRanker(
        VectorEngine(),
        GraphEngine(),
        bm25_index=BM25(),
        disable_graph_expansion=True,
        reranker=reranker,
        query_routing_enabled=False,
    )
    results, _, _ = ranker.search(
        "atlas project",
        top_k=2,
        rerank_pool=5,
        fusion_mode="rrf",
    )

    assert reranker.seen_ids[0] == "target"
    assert results[0]["node_id"] == "target"
    assert len(results) == 2
