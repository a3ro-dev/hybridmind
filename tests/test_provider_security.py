"""Offline credential-routing tests for provider and administration clients."""

from __future__ import annotations

import httpx

from scripts import runpod_endpoint_admin
from sdk.memory import HybridMemory


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_runpod_admin_keeps_account_key_out_of_url(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    captured = {}

    def fake_patch(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response({"id": "endpoint-1", "name": "test", "workersMin": 0})

    monkeypatch.setattr(runpod_endpoint_admin.httpx, "patch", fake_patch)
    result = runpod_endpoint_admin.set_workers_min("endpoint-1", 0)

    assert result["workersMin"] == 0
    assert "test-key" not in captured["url"]
    assert "api_key" not in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"] == {"workersMin": 0}


def test_runpod_admin_rejects_endpoint_path_injection(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    try:
        runpod_endpoint_admin.set_workers_min("../other?api_key=leak", 0)
    except ValueError as exc:
        assert "unsupported characters" in str(exc)
    else:
        raise AssertionError("unsafe endpoint ID was accepted")


def test_sdk_forwards_api_key_in_header_not_url():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.headers.get("X-HybridMind-API-Key")
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    memory = HybridMemory(base_url="https://memory.invalid", client=client, api_key="sdk-secret")
    try:
        assert memory._get("/health") == {"status": "ok"}
    finally:
        memory.close()

    assert captured["key"] == "sdk-secret"
    assert "sdk-secret" not in captured["url"]
