"""Fail-closed preflight for live evaluation providers.

Preflight itself can consume provider resources: a TEI warmup embeds one text,
and hosted LLM checks generate a tiny completion.  A validated live plan is
therefore mandatory and must include those calls in its declared usage.

Examples:
  python scripts/preflight.py --plan live-plan.json --validate-only
  python scripts/preflight.py --plan live-plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

import httpx

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"'))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from engine.provider_policy import validate_provider_url, validate_runpod_endpoint_id
from engine.resource_accounting import ResourceAccountingError, load_and_validate_live_plan


_preflight_deadline: float | None = None
_provider_runtime_ceiling_seconds: float | None = None
_usage_ceiling: dict[str, int] = {}
_usage_actual: dict[str, int] = {}


def _remaining_seconds(default: float) -> float:
    """Bound every live request by the plan's total wall clock."""
    if _preflight_deadline is None:
        return max(0.0, default)
    return max(0.0, min(default, _preflight_deadline - time.monotonic()))


def _reserve_usage(**amounts: int) -> bool:
    """Atomically reserve declared live-plan units before an external call."""
    for key, amount in amounts.items():
        if not isinstance(amount, int) or amount < 0:
            raise ValueError("preflight usage reservations must be non-negative integers")
        ceiling = int(_usage_ceiling.get(key, 0))
        if int(_usage_actual.get(key, 0)) + amount > ceiling:
            return False
    for key, amount in amounts.items():
        _usage_actual[key] = int(_usage_actual.get(key, 0)) + amount
    return True


def _safe_http_detail(response: httpx.Response) -> str:
    """Return status only; response bodies can echo credentials or request data."""
    return f"HTTP {response.status_code}"


def check(name: str, fn: Callable[[], tuple[bool, str]]) -> bool:
    try:
        ok, detail = fn()
        mark = "OK " if ok else "DOWN"
        print(f"[{mark}] {name}: {detail}")
        return ok
    except Exception as exc:
        print(f"[DOWN] {name}: {type(exc).__name__}")
        return False


def zai() -> tuple[bool, str]:
    key = settings.zai_api_key
    if not key:
        return False, "Z.AI credentials are not configured"
    base = validate_provider_url(
        settings.zai_base_url,
        "zai",
        allow_custom=settings.allow_custom_provider_urls,
    )
    timeout = _remaining_seconds(45.0)
    if timeout <= 0:
        return False, "live-plan wall-time budget exhausted"
    if not _reserve_usage(
        llm_calls=1,
        reader_input_tokens=16,
        reader_output_tokens=4,
    ):
        return False, "live-plan LLM/token budget exhausted before request"
    response = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": settings.qa_model,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_tokens": 4,
            "temperature": 0,
        },
        timeout=timeout,
    )
    return response.status_code == 200, _safe_http_detail(response)


def research_proxy() -> tuple[bool, str]:
    if not settings.allow_research_proxy:
        return False, "research proxy is not explicitly enabled"
    if not settings.research_proxy_api_key:
        return False, "research proxy credentials are not configured"
    base = validate_provider_url(
        settings.research_proxy_base_url,
        "research_proxy",
        allow_custom=settings.allow_custom_provider_urls,
    )
    payload = {
        "model": settings.research_proxy_model,
        "messages": [{"role": "user", "content": "Reply OK."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    if "qwen" in settings.research_proxy_model.lower():
        payload["reasoning_effort"] = "none"
    timeout = _remaining_seconds(45.0)
    if timeout <= 0:
        return False, "live-plan wall-time budget exhausted"
    if not _reserve_usage(
        llm_calls=1,
        reader_input_tokens=16,
        reader_output_tokens=4,
    ):
        return False, "live-plan LLM/token budget exhausted before request"
    response = httpx.post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.research_proxy_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    return response.status_code == 200, _safe_http_detail(response)


def runpod_llm() -> tuple[bool, str]:
    endpoint_id = settings.runpod_llm_endpoint_id
    key = settings.runpod_api_key
    if not endpoint_id or not key:
        return False, "RunPod LLM credentials/endpoint are not configured"
    endpoint_id = validate_runpod_endpoint_id(endpoint_id)
    timeout = _remaining_seconds(15.0)
    if timeout <= 0:
        return False, "live-plan wall-time budget exhausted"
    response = httpx.get(
        f"https://api.runpod.ai/v2/{endpoint_id}/health",
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    try:
        ready = int(response.json().get("workers", {}).get("ready", 0))
    except (ValueError, TypeError, AttributeError):
        ready = 0
    return response.status_code == 200 and ready > 0, f"{_safe_http_detail(response)}, {ready} workers ready"


def runpod_tei() -> tuple[bool, str]:
    base = settings.runpod_tei_embedding_url
    expected_dimension = 4096
    if settings.embedding_dimension != expected_dimension:
        return False, "embedding dimension is not the required 4096"
    if not base:
        return False, "RunPod TEI endpoint is not configured"
    key = settings.runpod_api_key
    if not key:
        return False, "RunPod credentials are not configured"
    base = validate_provider_url(
        base,
        "runpod",
        allow_custom=settings.allow_custom_provider_urls,
    )
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    configured_budget = float(os.getenv("HYBRIDMIND_PREFLIGHT_WARM_SECONDS", "180"))
    plan_runtime_budget = (
        configured_budget
        if _provider_runtime_ceiling_seconds is None
        else _provider_runtime_ceiling_seconds
    )
    admitted_budget = min(configured_budget, plan_runtime_budget)
    # Leave one second for client teardown and the final receipt so the
    # provider check itself cannot consume the entire admitted wall budget.
    budget = _remaining_seconds(max(0.0, admitted_budget - 1.0))
    if budget <= 0:
        return False, "live-plan provider-runtime/wall-time budget exhausted"
    deadline = time.monotonic() + max(0.0, budget)
    attempts = 0
    while True:
        if not _reserve_usage(embedding_calls=1, embedding_input_tokens=8):
            return False, f"live-plan embedding-call/token budget exhausted after {attempts} attempts"
        attempts += 1
        try:
            request_timeout = min(60.0, deadline - time.monotonic(), _remaining_seconds(60.0))
            if request_timeout <= 0:
                return False, f"live-plan budget exhausted after {attempts - 1} attempts"
            response = httpx.post(
                f"{base}/embed",
                headers=headers,
                json={"inputs": "preflight"},
                timeout=request_timeout,
            )
            if response.status_code != 200:
                if response.status_code in (408, 429, 500, 502, 503, 504) and time.monotonic() < deadline:
                    time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
                    continue
                return False, _safe_http_detail(response)
            body = response.json()
            dimension = len(body[0]) if isinstance(body, list) and body else None
            detail = f"HTTP 200, dim={dimension} (expected {expected_dimension})"
            if attempts > 1:
                detail += f", warm after {attempts} attempts"
            return dimension == expected_dimension, detail
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            if time.monotonic() >= deadline:
                return False, f"{type(exc).__name__} after {attempts} attempts"
            time.sleep(min(3.0, max(0.0, deadline - time.monotonic())))


PROVIDER_CHECKS: dict[str, tuple[str, Callable[[], tuple[bool, str]]]] = {
    "tei": ("RunPod TEI embedding (strict 4096-dim)", runpod_tei),
    "runpod_llm": ("RunPod LLM", runpod_llm),
    "zai": ("Z.AI canonical answering/judge", zai),
    "research_proxy": ("research-only proxy (explicit opt-in)", research_proxy),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate report, resource, usage, and spend gates without provider calls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global _preflight_deadline, _provider_runtime_ceiling_seconds
    global _usage_ceiling, _usage_actual
    try:
        args = parse_args(argv)
        plan, gate = load_and_validate_live_plan(args.plan.resolve())
    except (ResourceAccountingError, OSError, ValueError) as exc:
        print(f"FAIL: live plan rejected before provider checks: {exc}", file=sys.stderr)
        return 2

    print("=== HybridMind live-eval preflight ===")
    print(
        "[OK ] plan gate: "
        f"offline_report_sha256={gate.report_sha256}, "
        f"projected_cost_usd={gate.projected_cost_usd:.8f}, "
        f"available_memory_bytes={gate.available_memory_bytes}, "
        f"free_disk_bytes={gate.free_disk_bytes}"
    )
    if args.validate_only:
        print("VALIDATE ONLY: zero provider checks performed.")
        return 0

    _preflight_deadline = time.monotonic() + float(plan["max_wall_seconds"])
    _provider_runtime_ceiling_seconds = float(
        plan["usage_ceiling"]["provider_runtime_seconds"]
    )
    _usage_ceiling = {
        key: int(value) for key, value in plan["usage_ceiling"].items()
    }
    _usage_actual = {key: 0 for key in _usage_ceiling}
    results = []
    provider_checks_started = time.monotonic()
    for provider in plan["providers"]:
        name, function = PROVIDER_CHECKS[provider]
        results.append(check(name, function))
    provider_wall_seconds = time.monotonic() - provider_checks_started
    _usage_actual["provider_runtime_seconds"] = math.ceil(provider_wall_seconds)
    if provider_wall_seconds > _provider_runtime_ceiling_seconds:
        print(
            "[DOWN] live-plan provider-runtime ceiling exceeded: "
            f"{provider_wall_seconds:.3f}s > {_provider_runtime_ceiling_seconds:.3f}s"
        )
        results.append(False)
    print(f"PROVIDER CHECK WALL SECONDS: {provider_wall_seconds:.3f}")
    print(f"USAGE ACTUAL: {json.dumps(_usage_actual, sort_keys=True)}")
    if all(results):
        print("All plan-selected dependencies passed preflight.")
        return 0
    print("FAIL: one or more plan-selected dependencies are down. Do not run evaluation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
