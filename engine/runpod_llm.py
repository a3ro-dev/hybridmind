"""
Self-hosted RunPod vLLM chat-completion client.

Unlike Hack Club's proxy (synchronous OpenAI-compatible /chat/completions),
RunPod Serverless is job-queue based: POST /run submits a job and returns
immediately with a job id; the actual result is fetched by polling
/status/{id} until it reaches COMPLETED or FAILED.

Configuration is loaded exclusively through ``config.settings``.

Important: Qwen3.5 defaults to an extended "thinking" mode that burns output
tokens on a reasoning trace and leaves `content` null/truncated unless
disabled. This client always sets chat_template_kwargs.enable_thinking=False
unless the caller explicitly opts in.

Used through ``engine.llm_client`` so fallback policy remains centralized.
"""
from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import List, Optional

import httpx

from config import settings
from engine.serverless_util import retry_transient

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
# Budget spans queue wait + cold start (worker waking, model load) + generation.
# Serverless scale-from-zero on a 9B can eat 60-90s before the job even starts.
_DEFAULT_TIMEOUT_S = 240.0

_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    headers={
                        "Authorization": f"Bearer {settings.runpod_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
                )
    return _client


def is_configured() -> bool:
    return bool(settings.runpod_api_key.strip() and settings.runpod_llm_endpoint_id.strip())


def _base_url() -> str:
    endpoint_id = settings.runpod_llm_endpoint_id.strip()
    return f"https://api.runpod.ai/v2/{endpoint_id}" if endpoint_id else ""


def chat_completion(
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
    model: Optional[str] = None,
    response_format: Optional[dict] = None,
    enable_thinking: bool = False,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> Optional[str]:
    """
    Run a chat completion on the self-hosted RunPod vLLM endpoint.

    Returns the response content string, or None on failure, timeout, or if
    RunPod isn't configured (caller should fall back to another backend).
    """
    if not is_configured():
        return None

    openai_input: dict = {
        "model": model or settings.runpod_llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if response_format:
        openai_input["response_format"] = response_format

    client = _get_client()

    def _submit() -> str:
        submit = client.post(
            f"{_base_url()}/run",
            json={"input": {"openai_route": "/v1/chat/completions", "openai_input": openai_input}},
        )
        submit.raise_for_status()
        jid = submit.json().get("id")
        if not jid:
            raise RuntimeError(f"no job id in submit response: {submit.text[:300]}")
        return jid

    try:
        # Retry the submit itself: a scale-from-zero endpoint can 429/503 the
        # /run call while workers spin up, which is transient, not terminal.
        job_id = retry_transient(_submit, label="runpod_llm /run")
    except Exception as exc:
        logger.error(
            "runpod_llm: submit failed after retries type=%s",
            type(exc).__name__,
        )
        return None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status_resp = client.get(f"{_base_url()}/status/{job_id}")
            status_resp.raise_for_status()
            data = status_resp.json()
        except Exception as e:
            # Transient poll error (network blip, brief 5xx) — keep polling until
            # the deadline rather than abandoning an in-flight job.
            logger.warning(
                "runpod_llm: status poll error; will retry type=%s", type(e).__name__
            )
            time.sleep(_POLL_INTERVAL_S)
            continue

        status = data.get("status")
        if status == "COMPLETED":
            content = _extract_content(data, job_id)
            if content is None:
                logger.error(f"runpod_llm: job {job_id} COMPLETED but no usable content")
            return content
        elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            # Terminal server-side states — no cleanup needed, the job is over.
            logger.error("runpod_llm: job ended status=%s", status)
            return None
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            time.sleep(_POLL_INTERVAL_S)
        else:
            logger.warning(f"runpod_llm: job {job_id} unknown status '{status}', polling on")
            time.sleep(_POLL_INTERVAL_S)

    # We hit our own deadline while the job is still running server-side. Cancel
    # it so it stops occupying (and billing) a serverless worker for a result
    # nobody is waiting for anymore.
    logger.error(f"runpod_llm: timed out waiting for job {job_id} after {timeout_s}s; cancelling")
    cancel(job_id)
    return None


def _extract_content(data: dict, job_id: str) -> Optional[str]:
    """Pull the message content out of a COMPLETED job payload, defensively."""
    try:
        output = data["output"]
        choice = output[0] if isinstance(output, list) else output
        content = choice["choices"][0]["message"]["content"]
        return content if content else None
    except Exception as e:
        logger.error(
            "runpod_llm: unexpected completed payload type=%s", type(e).__name__
        )
        return None


def cancel(job_id: str) -> bool:
    """
    Best-effort cancel of a running/queued serverless job. Safe to call on an
    already-finished job (RunPod no-ops). Returns True if the cancel request
    was accepted. Never raises — cleanup must not mask the original failure.
    """
    if not is_configured() or not job_id:
        return False
    try:
        r = _get_client().post(f"{_base_url()}/cancel/{job_id}", timeout=15)
        ok = r.status_code < 400
        if not ok:
            logger.warning(f"runpod_llm: cancel {job_id} returned HTTP {r.status_code}")
        return ok
    except Exception as exc:
        logger.warning(
            "runpod_llm: cancel failed job=%s type=%s",
            job_id,
            type(exc).__name__,
        )
        return False


def health() -> Optional[dict]:
    """
    Return the endpoint's worker/job health dict, or None if unreachable.
    Note: 'ready' worker counts can be stale — treat as a hint, not a guarantee
    (a real embed/chat call is the only true readiness signal).
    """
    if not is_configured():
        return None
    try:
        r = _get_client().get(f"{_base_url()}/health", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning(
            "runpod_llm: health check failed type=%s", type(exc).__name__
        )
        return None


def close() -> None:
    """Release the shared HTTP client (connection pool). Idempotent."""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None


atexit.register(close)
