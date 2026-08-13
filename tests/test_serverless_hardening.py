"""
Regression guards for RunPod serverless handling (embedding + LLM).

These lock in the invariants we hardened on 2026-07-05:
  1. The embedding engine NEVER silently falls back to a wrong-dimension model —
     an unreachable endpoint raises loudly instead of poisoning retrieval.
  2. A wrong-dimension response from the endpoint is rejected.
  3. Transient errors are retried; terminal errors are not.
  4. Cleanup (close) and best-effort cancel never raise.

None of these hit a live endpoint.
"""
import httpx
import numpy as np
import pytest

from engine import serverless_util as su
import engine.embedding as embedding_module
from config import settings
from engine.embedding import RemoteEmbeddingEngine, TEIEmbeddingEngine


def test_retry_transient_retries_then_raises():
    calls = {"n": 0}

    def always_timeout():
        calls["n"] += 1
        raise httpx.ReadTimeout("boom")

    with pytest.raises(httpx.ReadTimeout):
        su.retry_transient(always_timeout, attempts=3, base_delay=0.0)
    assert calls["n"] == 3  # all attempts used


def test_retry_transient_does_not_retry_terminal():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        raise httpx.HTTPStatusError("400", request=resp.request, response=resp)

    with pytest.raises(httpx.HTTPStatusError):
        su.retry_transient(bad_request, attempts=4, base_delay=0.0)
    assert calls["n"] == 1  # terminal → no retries


def test_is_transient_classification():
    assert su.is_transient(httpx.ConnectError("x"))
    assert su.is_transient(httpx.ReadTimeout("x"))
    r503 = httpx.Response(503, request=httpx.Request("GET", "http://x"))
    assert su.is_transient(httpx.HTTPStatusError("503", request=r503.request, response=r503))
    r400 = httpx.Response(400, request=httpx.Request("GET", "http://x"))
    assert not su.is_transient(httpx.HTTPStatusError("400", request=r400.request, response=r400))


def test_unreachable_endpoint_raises_not_fallback(monkeypatch):
    # Point at a dead port; confirm embed() RAISES rather than returning a
    # local-model vector of the wrong dimension.
    # Patch the symbol used by engine.embedding, not only its definition
    # module; the engine imports this callable at module import time.
    monkeypatch.setattr(
        embedding_module, "retry_transient", lambda fn, **k: _fast_retry(fn)
    )
    eng = TEIEmbeddingEngine(base_url="http://127.0.0.1:1", api_key="x", dimension=4096)
    with pytest.raises(RuntimeError):
        eng.embed("hello")
    eng.close()


def test_wrong_dimension_response_rejected(monkeypatch):
    eng = TEIEmbeddingEngine(base_url="http://x", api_key="x", dimension=4096)

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return [[0.1, 0.2, 0.3]]  # 3-dim, not 4096

    monkeypatch.setattr(eng._client, "post", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError):
        eng.embed("hello")
    eng.close()


def test_correct_dimension_response_ok(monkeypatch):
    eng = TEIEmbeddingEngine(base_url="http://x", api_key="x", dimension=4096)

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return [[1.0] + [0.0] * 4095]

    monkeypatch.setattr(eng._client, "post", lambda *a, **k: _Resp())
    vec = eng.embed("hello")
    assert vec.shape == (4096,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    eng.close()


def test_embedding_engine_constructor_rejects_non_contract_dimension():
    with pytest.raises(ValueError, match="dimension=4096"):
        TEIEmbeddingEngine(base_url="http://x", api_key="x", dimension=4)


@pytest.mark.parametrize(
    "payload",
    [
        [[1.0] + [0.0] * 4095],  # one row returned for a two-row request
        [[float("nan")] + [0.0] * 4095] * 2,
    ],
)
def test_provider_rejects_wrong_row_count_and_non_finite_values(monkeypatch, payload):
    eng = TEIEmbeddingEngine(base_url="http://x", api_key="x", dimension=4096)

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    monkeypatch.setattr(eng._client, "post", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError):
        eng.embed_batch(["one", "two"])
    eng.close()


def test_openai_compatible_provider_rejects_incomplete_batch(monkeypatch):
    eng = RemoteEmbeddingEngine(
        base_url="http://x", api_key="x", dimension=4096
    )

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"data": [{"index": 0, "embedding": [1.0] + [0.0] * 4095}]}

    monkeypatch.setattr(eng._client, "post", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="terminal provider response"):
        eng.embed_batch(["one", "two"])
    eng.close()


def test_research_embedding_proxy_requires_opt_in_and_own_key(monkeypatch):
    monkeypatch.setenv("HC_EMBEDDING_URL", "https://research.invalid/v1")
    monkeypatch.delenv("RUNPOD_EMBEDDING_URL", raising=False)
    monkeypatch.delenv("HC_API_KEY", raising=False)
    monkeypatch.setattr(embedding_module, "_embedding_engine", None)
    monkeypatch.setattr(settings, "allow_research_proxy", False)
    with pytest.raises(RuntimeError, match="ALLOW_RESEARCH_PROXY"):
        embedding_module._remote_embedding_credentials(
            "https://research.invalid/v1", ""
        )


def test_embedding_credentials_are_bound_to_provider_host(monkeypatch):
    monkeypatch.setattr(settings, "allow_custom_provider_urls", False)
    monkeypatch.setattr(settings, "runpod_api_key", "runpod-key")
    with pytest.raises(ValueError, match="untrusted host"):
        embedding_module._remote_embedding_credentials(
            "", "https://attacker.invalid/v1"
        )

    monkeypatch.setattr(settings, "allow_research_proxy", True)
    monkeypatch.setattr(settings, "research_proxy_api_key", "research-key")
    with pytest.raises(ValueError, match="untrusted host"):
        embedding_module._remote_embedding_credentials(
            "https://attacker.invalid/v1", ""
        )


def test_tei_policy_uses_only_runpod_credential(monkeypatch):
    monkeypatch.setattr(settings, "runpod_api_key", "runpod-key")
    monkeypatch.setattr(settings, "research_proxy_api_key", "research-key")
    monkeypatch.setattr(settings, "allow_custom_provider_urls", False)

    url, key = embedding_module._runpod_tei_credentials(
        "https://example.api.runpod.ai"
    )
    assert url == "https://example.api.runpod.ai"
    assert key == "runpod-key"

    monkeypatch.setattr(settings, "allow_research_proxy", True)
    monkeypatch.setattr(settings, "research_proxy_api_key", "")
    with pytest.raises(RuntimeError, match="requires HC_API_KEY"):
        embedding_module._remote_embedding_credentials(
            "https://research.invalid/v1", ""
        )


def test_close_and_health_never_raise():
    eng = TEIEmbeddingEngine(base_url="http://127.0.0.1:1", api_key="x", dimension=4096)
    assert eng.health() is False   # dead endpoint → False, no exception
    eng.close()
    eng.close()                    # idempotent


def _fast_retry(fn):
    # one attempt, surfaces the underlying error immediately
    return fn()
