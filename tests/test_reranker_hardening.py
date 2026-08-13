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
    with pytest.raises(ValueError, match="off, cross, llm"):
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
