"""Live preflight cannot outlive the admitted wall/runtime budget."""

import time

from config import settings
from scripts import preflight


def test_expired_plan_budget_prevents_tei_network_call(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "runpod_tei_embedding_url", "https://example.api.runpod.ai")
    monkeypatch.setattr(settings, "runpod_api_key", "secret")
    monkeypatch.setattr(settings, "allow_custom_provider_urls", False)
    monkeypatch.setattr(preflight, "_preflight_deadline", time.monotonic() - 1.0)
    monkeypatch.setattr(preflight, "_provider_runtime_ceiling_seconds", 30.0)
    monkeypatch.setattr(preflight, "_usage_ceiling", {"embedding_calls": 1, "embedding_input_tokens": 8})
    monkeypatch.setattr(preflight, "_usage_actual", {})
    monkeypatch.setattr(preflight.httpx, "post", lambda *args, **kwargs: calls.append(1))

    ok, detail = preflight.runpod_tei()

    assert ok is False
    assert "budget exhausted" in detail
    assert calls == []


def test_tei_retry_cannot_exceed_declared_embedding_call_ceiling(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "runpod_tei_embedding_url", "https://example.api.runpod.ai")
    monkeypatch.setattr(settings, "runpod_api_key", "secret")
    monkeypatch.setattr(settings, "allow_custom_provider_urls", False)
    monkeypatch.setattr(preflight, "_preflight_deadline", time.monotonic() + 30.0)
    monkeypatch.setattr(preflight, "_provider_runtime_ceiling_seconds", 30.0)
    monkeypatch.setattr(preflight, "_usage_ceiling", {"embedding_calls": 1, "embedding_input_tokens": 8})
    monkeypatch.setattr(preflight, "_usage_actual", {})
    monkeypatch.setattr(preflight.time, "sleep", lambda *_: None)

    def timeout(*args, **kwargs):
        calls.append(1)
        raise preflight.httpx.TimeoutException("timeout")

    monkeypatch.setattr(preflight.httpx, "post", timeout)

    ok, detail = preflight.runpod_tei()

    assert ok is False
    assert "budget exhausted after 1 attempts" in detail
    assert calls == [1]
    assert preflight._usage_actual == {"embedding_calls": 1, "embedding_input_tokens": 8}
