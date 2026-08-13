"""Central policy and transport for HybridMind chat-completion providers.

Production hosted inference is Z.AI (GLM-4.6). A configured RunPod vLLM
endpoint is an allowed self-hosted backend. The Hack Club-compatible research
proxy is intentionally absent from every provider chain unless
``settings.allow_research_proxy`` is true.

Callers choose a preferred provider, but they do not handle credentials,
URLs, retries, or provider-specific model names themselves. This keeps the
research proxy from becoming an accidental production fallback.
"""
from __future__ import annotations

import atexit
import logging
import random
import threading
import time
from typing import Literal, Optional, Sequence

import httpx

from config import settings
from engine import runpod_llm
from engine.provider_policy import validate_provider_url, validate_runpod_endpoint_id

logger = logging.getLogger(__name__)

Provider = Literal["auto", "zai", "runpod", "research_proxy"]

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.5
_BACKOFF_CAP = 20.0

_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                    timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
                )
    return _client


def _research_proxy_configured() -> bool:
    return bool(
        settings.allow_research_proxy
        and settings.research_proxy_api_key.strip()
        and settings.research_proxy_base_url.strip()
    )


def _provider_configured(provider: str) -> bool:
    if provider == "zai":
        configured = bool(settings.zai_api_key.strip() and settings.zai_base_url.strip())
        if configured:
            validate_provider_url(
                settings.zai_base_url,
                "zai",
                allow_custom=settings.allow_custom_provider_urls,
            )
        return configured
    if provider == "runpod":
        configured = runpod_llm.is_configured()
        if configured:
            validate_runpod_endpoint_id(settings.runpod_llm_endpoint_id)
        return configured
    if provider == "research_proxy":
        configured = _research_proxy_configured()
        if configured:
            validate_provider_url(
                settings.research_proxy_base_url,
                "research_proxy",
                allow_custom=settings.allow_custom_provider_urls,
            )
        return configured
    return False


def provider_chain(
    preferred: Provider = "auto",
    *,
    allow_fallback: bool = True,
) -> tuple[str, ...]:
    """Return the configured provider order allowed by the current policy.

    ``zai`` is special: it is the canonical hosted QA/judge backend, so its
    only possible fallback is the explicitly enabled research proxy. It never
    silently changes a benchmark to a different self-hosted model.
    """
    if preferred not in {"auto", "zai", "runpod", "research_proxy"}:
        raise ValueError(f"Unsupported LLM provider: {preferred!r}")

    if preferred == "research_proxy":
        candidates: list[str] = ["research_proxy"]
    elif preferred == "zai":
        candidates = ["zai"]
        if allow_fallback:
            candidates.append("research_proxy")
    else:
        if preferred == "auto":
            # Enabling research mode selects the free hosted backend instead
            # of silently spending Z.AI budget. RunPod remains first because
            # it is self-hosted and explicitly configured.
            candidates = ["runpod"]
            candidates.append("research_proxy" if settings.allow_research_proxy else "zai")
        else:
            candidates = ["runpod"]
        if preferred == "runpod" and allow_fallback:
            candidates.append("research_proxy" if settings.allow_research_proxy else "zai")

    return tuple(p for p in candidates if _provider_configured(p))


def is_configured(
    preferred: Provider = "auto",
    *,
    allow_fallback: bool = True,
) -> bool:
    """Return whether at least one policy-allowed provider is configured."""
    return bool(provider_chain(preferred, allow_fallback=allow_fallback))


def chat_completion(
    messages: Sequence[dict],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    model: Optional[str] = None,
    response_format: Optional[dict] = None,
    preferred: Provider = "auto",
    allow_fallback: bool = True,
    enable_thinking: bool = False,
) -> Optional[str]:
    """Return completion text from the first successful policy-allowed backend.

    ``model`` is the Z.AI model override. RunPod and research-proxy models are
    always sourced from their dedicated settings, preventing a GLM model name
    from being sent accidentally to a Qwen deployment (or vice versa).
    """
    chain = provider_chain(preferred, allow_fallback=allow_fallback)
    if not chain:
        logger.warning("No policy-allowed LLM provider is configured (preferred=%s)", preferred)
        return None

    for index, provider in enumerate(chain):
        if index:
            logger.warning("LLM provider %s failed; trying configured %s backend", chain[index - 1], provider)

        if provider == "runpod":
            content = runpod_llm.chat_completion(
                list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                model=settings.runpod_llm_model,
                response_format=response_format,
                enable_thinking=enable_thinking,
            )
        elif provider == "zai":
            content = _openai_compatible_completion(
                provider="Z.AI",
                base_url=validate_provider_url(
                    settings.zai_base_url,
                    "zai",
                    allow_custom=settings.allow_custom_provider_urls,
                ),
                api_key=settings.zai_api_key,
                model=model or settings.qa_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                disable_qwen_thinking=False,
            )
        else:
            logger.warning(
                "Using explicitly enabled research proxy; this backend is not a production fallback"
            )
            content = _openai_compatible_completion(
                provider="research proxy",
                base_url=validate_provider_url(
                    settings.research_proxy_base_url,
                    "research_proxy",
                    allow_custom=settings.allow_custom_provider_urls,
                ),
                api_key=settings.research_proxy_api_key,
                model=settings.research_proxy_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                disable_qwen_thinking=True,
            )

        if content:
            return content
    return None


def _openai_compatible_completion(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[dict],
    max_tokens: int,
    temperature: float,
    response_format: Optional[dict],
    disable_qwen_thinking: bool,
) -> Optional[str]:
    payload: dict = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    if disable_qwen_thinking and "qwen" in model.lower():
        payload["reasoning_effort"] = "none"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = _get_client().post(endpoint, headers=headers, json=payload)
            if response.status_code in _RETRYABLE_STATUS:
                raise httpx.HTTPStatusError(
                    f"retryable {provider} response",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return content if isinstance(content, str) and content.strip() else None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # Some OpenAI-compatible servers do not implement JSON schema.
            if status == 400 and "response_format" in payload:
                payload.pop("response_format", None)
                continue
            if status in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, provider, f"HTTP {status}")
                continue
            logger.error("%s completion failed: HTTP %s", provider, status)
            return None
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.PoolTimeout,
        ) as exc:
            if attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, provider, type(exc).__name__)
                continue
            logger.error(
                "%s completion failed after %s attempts type=%s",
                provider,
                _MAX_ATTEMPTS,
                type(exc).__name__,
            )
            return None
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(
                "%s returned an invalid completion payload type=%s",
                provider,
                type(exc).__name__,
            )
            return None
    return None


def _sleep_backoff(attempt: int, provider: str, reason: str) -> None:
    delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2**attempt) * (0.5 + random.random()))
    logger.warning(
        "%s retry %s/%s in %.1fs (%s)", provider, attempt + 1, _MAX_ATTEMPTS, delay, reason
    )
    time.sleep(delay)


def close() -> None:
    """Release the shared HTTP connection pool."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


atexit.register(close)
