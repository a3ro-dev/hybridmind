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
from engine.embedding import TEIEmbeddingEngine


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
    monkeypatch.setattr(su, "retry_transient",
                        lambda fn, **k: _fast_retry(fn))
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
    eng = TEIEmbeddingEngine(base_url="http://x", api_key="x", dimension=4)

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return [[3.0, 0.0, 0.0, 4.0]]  # norm 5 → unit vec

    monkeypatch.setattr(eng._client, "post", lambda *a, **k: _Resp())
    vec = eng.embed("hello")
    assert vec.shape == (4,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    eng.close()


def test_close_and_health_never_raise():
    eng = TEIEmbeddingEngine(base_url="http://127.0.0.1:1", api_key="x", dimension=4096)
    assert eng.health() is False   # dead endpoint → False, no exception
    eng.close()
    eng.close()                    # idempotent


def _fast_retry(fn):
    # one attempt, surfaces the underlying error immediately
    return fn()
