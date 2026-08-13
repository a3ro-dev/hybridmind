"""Minimal RunPod endpoint administration over the current REST API.

Unlike the legacy GraphQL API, the REST API accepts the account key in an
Authorization header.  Keeping it out of the URL prevents disclosure through
shell history, proxy access logs, exception URLs, and monitoring systems.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.provider_policy import validate_runpod_endpoint_id


API_ROOT = "https://rest.runpod.io/v1"


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is required")
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def list_endpoints(*, timeout: float = 30.0) -> list[dict]:
    response = httpx.get(f"{API_ROOT}/endpoints", headers=_headers(), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("RunPod returned an invalid endpoint list")
    return payload


def set_workers_min(
    endpoint_id: str,
    workers_min: int,
    *,
    timeout: float = 30.0,
) -> dict:
    endpoint_id = validate_runpod_endpoint_id(endpoint_id)
    if workers_min < 0:
        raise ValueError("workers_min must be non-negative")
    response = httpx.patch(
        f"{API_ROOT}/endpoints/{endpoint_id}",
        headers=_headers(),
        json={"workersMin": workers_min},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("RunPod returned an invalid endpoint update")
    return payload


def set_all_workers_min(workers_min: int) -> list[tuple[str, str, int]]:
    results: list[tuple[str, str, int]] = []
    for endpoint in list_endpoints():
        endpoint_id = validate_runpod_endpoint_id(str(endpoint.get("id", "")))
        result = set_workers_min(endpoint_id, workers_min)
        name = str(result.get("name") or endpoint.get("name") or endpoint_id)
        actual = int(result.get("workersMin", workers_min))
        results.append((endpoint_id, name, actual))
    return results


def configured_endpoint_ids() -> list[str]:
    values: Iterable[str] = (
        os.environ.get("RUNPOD_TEI_ENDPOINT_ID", ""),
        os.environ.get("RUNPOD_LLM_ENDPOINT_ID", ""),
    )
    unique: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in unique:
            unique.append(validate_runpod_endpoint_id(value))
    return unique
