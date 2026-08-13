from config import settings
from engine import query_decomposition


def test_decomposition_uses_central_provider_once(monkeypatch):
    calls = []

    def fake_completion(messages, **kwargs):
        calls.append((messages, kwargs))
        return '["What company employs Alice?", "What city is that company based in?"]'

    monkeypatch.setattr(query_decomposition.llm_client, "chat_completion", fake_completion)
    result = query_decomposition.decompose_query(
        "What company employs Alice and what city is that company based in?",
        model="configured-model",
        enabled=True,
    )

    assert len(calls) == 1
    assert calls[0][1]["max_tokens"] == 300
    assert calls[0][1]["model"] == "configured-model"
    assert result == [
        "What company employs Alice?",
        "What city is that company based in?",
    ]


def test_decomposition_rejects_lost_temporal_constraint(monkeypatch):
    monkeypatch.setattr(
        query_decomposition.llm_client,
        "chat_completion",
        lambda *args, **kwargs: '["What job did Alice have?", "What company employed Alice?"]',
    )
    assert (
        query_decomposition.decompose_query(
            "What job did Alice have before 2025?", enabled=True
        )
        == []
    )


def test_decomposition_rejects_more_than_three_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(
        query_decomposition.llm_client,
        "chat_completion",
        lambda *args, **kwargs: '["a?", "b?", "c?", "d?"]',
    )
    assert query_decomposition.decompose_query("How are a and b connected?", enabled=True) == []


def test_enabled_none_defers_to_settings(monkeypatch):
    monkeypatch.setattr(settings, "query_decomposition_enabled", False)
    monkeypatch.setattr(
        query_decomposition.llm_client,
        "chat_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call")),
    )
    assert query_decomposition.decompose_query("question", enabled=None) == []
