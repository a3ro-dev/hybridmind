"""Regression tests for query-cache state and synchronization semantics."""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import api.search as search_api
from engine.cache import QueryCache
from models.search import HybridSearchRequest


def test_pattern_invalidation_cannot_deadlock_and_clears_cache():
    cache = QueryCache(maxsize=4, ttl=60)
    cache.set("vector", {"query": "alpha"}, {"node": "a"})
    cache.set("hybrid", {"query": "beta"}, {"node": "b"})

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache.invalidate_pattern, "vector")
        try:
            future.result(timeout=1.0)
        except TimeoutError as exc:  # pragma: no cover - failure guard
            raise AssertionError("cache pattern invalidation deadlocked") from exc

    assert cache.stats["size"] == 0


def test_access_tracking_bypasses_hybrid_response_cache(monkeypatch):
    class CacheSpy:
        get_calls = 0
        set_calls = 0

        def get(self, *_args, **_kwargs):
            self.get_calls += 1
            raise AssertionError("stateful search consulted response cache")

        def set(self, *_args, **_kwargs):
            self.set_calls += 1
            raise AssertionError("stateful search populated response cache")

    class RankerSpy:
        track_access = None

        def search(self, **kwargs):
            self.track_access = kwargs["track_access"]
            return (
                [
                    {
                        "node_id": "node-1",
                        "text": "evidence",
                        "metadata": {},
                        "vector_score": 1.0,
                        "graph_score": 0.0,
                        "combined_score": 1.0,
                        "reasoning": "test",
                    }
                ],
                0.1,
                1,
                {
                    "schema_version": "hybridmind.search-execution/v1",
                    "cache_hit": False,
                    "stages": {},
                },
            )

    class StoreSpy:
        @staticmethod
        def get_corpus_generation():
            return 7

    cache = CacheSpy()
    ranker = RankerSpy()
    monkeypatch.setattr(search_api, "get_query_cache", lambda: cache)

    response = asyncio.run(
        search_api.hybrid_search(
            HybridSearchRequest(query_text="where", track_access=True),
            ranker,
            StoreSpy(),
        )
    )

    assert [result.node_id for result in response.results] == ["node-1"]
    assert ranker.track_access is True
    assert response.execution_trace["corpus_generation"] == 7
    assert cache.get_calls == 0
    assert cache.set_calls == 0
