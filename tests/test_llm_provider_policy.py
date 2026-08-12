"""Offline tests for hosted/self-hosted LLM provider boundaries."""

from config import settings
from engine import llm_client


def _disable_production_providers(monkeypatch):
    monkeypatch.setattr(settings, "zai_api_key", "")
    monkeypatch.setattr(settings, "runpod_api_key", "")
    monkeypatch.setattr(settings, "runpod_llm_endpoint_id", "")


def test_research_proxy_is_excluded_without_explicit_opt_in(monkeypatch):
    _disable_production_providers(monkeypatch)
    monkeypatch.setattr(settings, "research_proxy_api_key", "test-key")
    monkeypatch.setattr(settings, "allow_research_proxy", False)

    assert llm_client.provider_chain() == ()
    assert not llm_client.is_configured()


def test_auto_research_mode_does_not_spend_zai_budget(monkeypatch):
    monkeypatch.setattr(settings, "zai_api_key", "zai-key")
    monkeypatch.setattr(settings, "zai_base_url", "https://zai.invalid/v1")
    monkeypatch.setattr(settings, "runpod_api_key", "runpod-key")
    monkeypatch.setattr(settings, "runpod_llm_endpoint_id", "endpoint")
    monkeypatch.setattr(settings, "research_proxy_api_key", "research-key")
    monkeypatch.setattr(settings, "allow_research_proxy", True)

    assert llm_client.provider_chain() == ("runpod", "research_proxy")
    assert llm_client.provider_chain("zai") == ("zai", "research_proxy")


def test_research_proxy_opt_in_uses_only_configured_research_model(monkeypatch, caplog):
    _disable_production_providers(monkeypatch)
    monkeypatch.setattr(settings, "allow_research_proxy", True)
    monkeypatch.setattr(settings, "research_proxy_api_key", "research-key")
    monkeypatch.setattr(settings, "research_proxy_base_url", "https://research.invalid/v1")
    monkeypatch.setattr(settings, "research_proxy_model", "qwen/research-model")

    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(llm_client, "_openai_compatible_completion", fake_completion)

    content = llm_client.chat_completion(
        [{"role": "user", "content": "hello"}],
        preferred="zai",
        model="glm-4.6",
    )

    assert content == "ok"
    assert seen["provider"] == "research proxy"
    assert seen["model"] == "qwen/research-model"
    assert seen["disable_qwen_thinking"] is True
    assert "not a production fallback" in caplog.text


def test_zai_never_falls_back_to_runpod(monkeypatch):
    monkeypatch.setattr(settings, "zai_api_key", "")
    monkeypatch.setattr(settings, "runpod_api_key", "runpod-key")
    monkeypatch.setattr(settings, "runpod_llm_endpoint_id", "endpoint")
    monkeypatch.setattr(settings, "allow_research_proxy", False)

    assert llm_client.provider_chain("zai") == ()


def test_eval_research_opt_in_selects_proxy_without_paid_fallback(monkeypatch):
    import eval_common

    monkeypatch.setattr(settings, "allow_research_proxy", True)
    seen = {}

    def fake_chat(messages, **kwargs):
        seen.update(kwargs)
        return "answer"

    monkeypatch.setattr(eval_common.llm_client, "chat_completion", fake_chat)

    result = eval_common._call({"messages": [], "model": "unexpected-model"})

    assert result == "answer"
    assert seen["preferred"] == "research_proxy"
    assert seen["allow_fallback"] is False
    assert seen["model"] == settings.qa_model
