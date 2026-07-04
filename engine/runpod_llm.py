"""
Self-hosted RunPod vLLM chat-completion client.

Unlike Hack Club's proxy (synchronous OpenAI-compatible /chat/completions),
RunPod Serverless is job-queue based: POST /run submits a job and returns
immediately with a job id; the actual result is fetched by polling
/status/{id} until it reaches COMPLETED or FAILED.

Env vars:
  RUNPOD_API_KEY          — RunPod account API key (Bearer token)
  RUNPOD_LLM_ENDPOINT_ID  — Serverless endpoint id (e.g. "e4vphzghmbvt7j")
  RUNPOD_LLM_MODEL        — model id as registered by vLLM (default: qwen/qwen3.5-9b)

Important: Qwen3.5 defaults to an extended "thinking" mode that burns output
tokens on a reasoning trace and leaves `content` null/truncated unless
disabled. This client always sets chat_template_kwargs.enable_thinking=False
unless the caller explicitly opts in.

Used by engine/fact_extractor.py and engine/consolidation.py as the primary
LLM backend when configured; both fall back to the Hack Club proxy path
when RUNPOD_LLM_ENDPOINT_ID is unset, so existing deployments keep working.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_API_KEY = os.getenv("RUNPOD_API_KEY", "")
_ENDPOINT_ID = os.getenv("RUNPOD_LLM_ENDPOINT_ID", "")
_MODEL = os.getenv("RUNPOD_LLM_MODEL", "qwen/qwen3.5-9b")
_BASE_URL = f"https://api.runpod.ai/v2/{_ENDPOINT_ID}" if _ENDPOINT_ID else ""

_POLL_INTERVAL_S = 2.0
_DEFAULT_TIMEOUT_S = 120.0

_client: Optional[httpx.Client] = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers={"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        )
    return _client


def is_configured() -> bool:
    return bool(_API_KEY and _ENDPOINT_ID)


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
        "model": model or _MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if response_format:
        openai_input["response_format"] = response_format

    client = _get_client()
    try:
        submit = client.post(
            f"{_BASE_URL}/run",
            json={"input": {"openai_route": "/v1/chat/completions", "openai_input": openai_input}},
        )
        submit.raise_for_status()
        job_id = submit.json().get("id")
        if not job_id:
            logger.error(f"runpod_llm: no job id in submit response: {submit.text[:300]}")
            return None
    except Exception as e:
        logger.error(f"runpod_llm: submit failed: {e}")
        return None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status_resp = client.get(f"{_BASE_URL}/status/{job_id}")
            status_resp.raise_for_status()
            data = status_resp.json()
        except Exception as e:
            logger.warning(f"runpod_llm: status poll error: {e}")
            time.sleep(_POLL_INTERVAL_S)
            continue

        status = data.get("status")
        if status == "COMPLETED":
            try:
                output = data["output"]
                choice = output[0] if isinstance(output, list) else output
                return choice["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"runpod_llm: unexpected COMPLETED payload shape ({e}): {str(data)[:300]}")
                return None
        elif status == "FAILED":
            logger.error(f"runpod_llm: job {job_id} failed: {data.get('error')}")
            return None
        time.sleep(_POLL_INTERVAL_S)

    logger.error(f"runpod_llm: timed out waiting for job {job_id} after {timeout_s}s")
    return None
