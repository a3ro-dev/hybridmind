"""
Shared retry/backoff helpers for RunPod serverless backends.

Both of our self-hosted endpoints scale to zero and pay a cold-start tax on
the first request after idle (an 8B model loading into VRAM = tens of seconds),
and both can transiently 429/503 when all workers are busy. Neither of those is
a real failure — it is a *state* the client must wait through. These helpers
centralize that policy so the embedding and LLM clients stay in sync.

Transient (retry): connection errors, read/pool timeouts, HTTP 408/429/5xx.
Terminal (raise):  4xx other than 408/429 (bad request, auth) — retrying can't help.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP statuses worth retrying: request timeout, rate limit, and all 5xx.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def is_transient(exc: Exception) -> bool:
    """True if `exc` looks like a cold start / saturation blip, not a hard error."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError,
                        httpx.ReadError, httpx.RemoteProtocolError, httpx.PoolTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


def retry_transient(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    label: str = "serverless call",
) -> T:
    """
    Call `fn` up to `attempts` times, backing off exponentially on transient
    errors. Re-raises the last exception if all attempts fail, and re-raises
    terminal (non-transient) errors immediately without burning retries.

    Backoff is deterministic (no jitter) so behavior is reproducible in evals.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classify, then re-raise or retry
            last_exc = exc
            if not is_transient(exc) or i == attempts - 1:
                raise
            delay = min(base_delay * (2 ** i), max_delay)
            logger.warning(
                "%s: transient error (%s), retry %d/%d in %.1fs",
                label, type(exc).__name__, i + 1, attempts - 1, delay,
            )
            time.sleep(delay)
    # Unreachable (loop either returns or raises), but keeps type-checkers happy.
    assert last_exc is not None
    raise last_exc
