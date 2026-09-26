import numpy as np
import pytest

from config import Settings, settings
from engine.reranker import CrossEncoderReranker, LLMReranker, get_reranker


class _Model:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def predict(self, pairs, batch_size=32):
        self.calls += 1
        return self.scores


def _candidates(count=2):
    return [
        {"node_id": str(index), "text": f"passage {index}", "combined_score": 1.0 - index / 10}
        for index in range(count)
    ]


def test_reranker_is_opt_in_and_startup_warmup_is_off_by_default():
    config = Settings()
    assert config.rerank_mode == "off"
    assert config.reranker_warmup_enabled is False


def test_invalid_reranker_mode_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "rerank_mode", "mystery")
    with pytest.raises(ValueError, match="off, cross, tei, llm"):
        get_reranker()


@pytest.mark.parametrize("scores", [[0.2], [0.2, float("nan")]])
def test_cross_encoder_rejects_wrong_count_and_non_finite_scores(scores):
    reranker = CrossEncoderReranker()
    reranker._model = _Model(scores)
    candidates = _candidates()
    result = reranker.rerank("query", candidates)
    assert [candidate["node_id"] for candidate in result] == ["0", "1"]
    assert all(candidate["rerank_applied"] is False for candidate in result)
    assert all(candidate["rerank_failure_type"] == "ValueError" for candidate in result)


def test_cross_encoder_enforces_pair_and_text_bounds_before_model_call(monkeypatch):
    monkeypatch.setattr(settings, "reranker_max_pairs", 1)
    reranker = CrossEncoderReranker()
    model = _Model(np.array([0.2, 0.1]))
    reranker._model = model
    result = reranker.rerank("query", _candidates())
    assert model.calls == 0
    assert all(candidate["rerank_failure_type"] == "ValueError" for candidate in result)


def test_llm_requires_complete_permutation_and_disables_fallback(monkeypatch):
    observed = {}

    def fake_completion(*_args, **kwargs):
        observed.update(kwargs)
        return "[1, 1]"

    monkeypatch.setattr("engine.llm_client.chat_completion", fake_completion)
    monkeypatch.setattr(settings, "allow_research_proxy", False)
    candidates = _candidates()
    result = LLMReranker().rerank("query", candidates)
    assert [candidate["node_id"] for candidate in result] == ["0", "1"]
    assert all(candidate["rerank_applied"] is False for candidate in result)
    assert all(candidate["rerank_failure_type"] == "ValueError" for candidate in result)
    assert observed["preferred"] == "zai"
    assert observed["allow_fallback"] is False


def test_llm_accepts_only_full_permutation(monkeypatch):
    monkeypatch.setattr("engine.llm_client.chat_completion", lambda *_a, **_k: "[1, 0]")
    result = LLMReranker().rerank("query", _candidates())
    assert [candidate["node_id"] for candidate in result] == ["1", "0"]
    assert all(candidate["rerank_applied"] is True for candidate in result)


def _tei(handler):
    import httpx
    from engine.reranker import TEIReranker

    return TEIReranker("http://127.0.0.1:8080", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_tei_reranker_sends_documented_request_and_maps_indices():
    import json as _json
    import httpx

    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = _json.loads(request.content)
        return httpx.Response(200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])

    ranked = _tei(handler).rerank("q", [{"text": "a", "combined_score": 1.0}, {"text": "b", "combined_score": 0.5}])
    assert seen["path"] == "/rerank"
    assert seen["body"] == {"query": "q", "texts": ["a", "b"], "raw_scores": False, "truncate": False}
    assert [c["text"] for c in ranked][0] in {"a", "b"}
    assert {c["text"]: c["rerank_score"] for c in ranked} == {"a": 0.1, "b": 0.9}
    assert all(c["rerank_applied"] for c in ranked)


@pytest.mark.parametrize("payload", [
    [{"index": 0, "score": 0.5}],
    [{"index": 0, "score": 0.5}, {"index": 0, "score": 0.4}],
    [{"index": 0, "score": 0.5}, {"index": 5, "score": 0.4}],
])
def test_tei_reranker_marks_incomplete_responses_as_failed(payload):
    import httpx

    ranked = _tei(lambda request: httpx.Response(200, json=payload)).rerank(
        "q", [{"text": "a", "combined_score": 1.0}, {"text": "b", "combined_score": 0.5}]
    )
    assert all(c["rerank_applied"] is False and c["rerank_failure_type"] for c in ranked)


def test_tei_reranker_url_binding(monkeypatch):
    from engine.reranker import _tei_reranker_endpoint

    monkeypatch.setattr(settings, "reranker_tei_url", "http://localhost:8081/")
    assert _tei_reranker_endpoint() == ("http://localhost:8081", "")
    monkeypatch.setattr(settings, "reranker_tei_url", "http://evil.example.com")
    monkeypatch.setattr(settings, "runpod_api_key", "")
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        _tei_reranker_endpoint()
    monkeypatch.setattr(settings, "runpod_api_key", "k")
    with pytest.raises(ValueError):
        _tei_reranker_endpoint()
