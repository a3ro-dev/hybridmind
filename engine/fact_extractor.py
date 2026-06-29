"""
Ingest-time fact extraction via HackClub proxy.
Called once per session during ingest. Never at query time.

Uses a shared pooled httpx client + bounded retry with exponential backoff and
jitter, so transient HC proxy drops ("socket connection was closed
unexpectedly", connection resets, 429/5xx) don't abort a benchmark run.
"""
import json
import logging
import os
import random
import time
from dotenv import load_dotenv

import httpx

load_dotenv()

logger = logging.getLogger(__name__)

_HC_API_KEY = os.getenv("HC_API_KEY", "")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1").rstrip("/")
_MODEL = "openai/gpt-4o"

# Retry configuration
_MAX_ATTEMPTS = 4
_BACKOFF_BASE = 1.5  # seconds
_BACKOFF_CAP = 30.0  # seconds

# Shared pooled client — a single connection pool reduces the socket churn that
# contributes to the HC proxy dropping connections under sustained load.
_CLIENT: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
        )
    return _CLIENT

_SYSTEM_PROMPT = """\
You are a precise fact extraction engine. Given a conversation, extract EVERY discrete, answerable fact.

Return a JSON array where each element is:
{"fact": "<one clear, self-contained sentence>", "entities": ["name1", "name2"], "date": "<YYYY-MM-DD or empty>"}

RULES:
1. Extract ALL facts: names, ages, relationships, dates, locations, jobs, hobbies, preferences, plans, events, health info, family details, education, achievements, emotions, numeric values
2. Each fact MUST be self-contained — readable without any context
3. Resolve relative dates (e.g. "last Friday" → actual date if known)
4. Include minor details — they may be queried later
5. Extract 5-20 facts per conversation session
6. Return ONLY a valid JSON array, no markdown, no explanation, no code fences

Example output:
[{"fact": "Alice works as a software engineer at Google", "entities": ["Alice", "Google"], "date": ""}, {"fact": "Alice moved to San Francisco in 2023", "entities": ["Alice", "San Francisco"], "date": "2023-01-01"}]
"""


def extract_facts_from_session(turns: list[dict]) -> list[dict]:
    """
    Extract discrete facts from a list of conversation turns.

    Controlled by FACT_EXTRACTION_ENABLED env var (default: "false").
    When disabled, returns [] immediately. When enabled, uses Claude Haiku
    via HackClub proxy to extract structured facts.

    Args:
        turns: List of dicts with keys: speaker (str), text (str), date (str)

    Returns:
        List of dicts: {fact: str, entities: list[str], date: str}
        Returns [] on any failure — caller must handle gracefully.
    """
    # Quick check: fact extraction must be explicitly enabled
    if os.getenv("FACT_EXTRACTION_ENABLED", "").lower() not in ("1", "true", "yes"):
        return []

    if not turns:
        return []

    if not _HC_API_KEY:
        logger.warning("HC_API_KEY not set — fact extraction disabled")
        return []

    # Build the conversation text with date and speaker context
    lines = []
    for t in turns:
        date = t.get("date", "").strip()
        speaker = t.get("speaker", "").strip()
        text = t.get("text", "").strip()
        if not text:
            continue
        prefix = f"[{date}] " if date else ""
        lines.append(f"{prefix}{speaker}: {text}")

    conversation = "\n".join(lines)[:16000]  # ~12k tokens max (increased from 8000)

    if not conversation.strip():
        return []

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": conversation},
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {_HC_API_KEY}",
        "Content-Type": "application/json",
    }
    client = _get_client()

    logger.info(
        f"fact_extractor: calling {_MODEL} at {_BASE_URL} for {len(turns)} turns "
        f"({len(conversation)} chars)"
    )

    content = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.post(
                f"{_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            # Retry on transient server-side / rate-limit statuses.
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            break  # success

        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, f"HTTP {status}")
                continue
            # Non-retryable (400/401/422/etc.) or out of attempts.
            body = e.response.text[:500] if e.response is not None else ""
            logger.error(f"fact_extractor HTTP error: {status} {body}")
            return []
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.PoolTimeout,
        ) as e:
            if attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, f"{type(e).__name__}: {e}")
                continue
            logger.error(
                f"fact_extractor transient error to {_BASE_URL} after "
                f"{_MAX_ATTEMPTS} attempts ({type(e).__name__}): {e}"
            )
            return []
        except Exception as e:
            logger.error(f"fact_extractor unexpected error ({type(e).__name__}): {e}")
            return []

    if content is None:
        return []

    try:
        logger.debug(f"fact_extractor: raw response ({len(content)} chars): {content[:500]}")

        # Extract JSON array robustly
        start = content.find("[")
        end = content.rfind("]") + 1
        if start < 0 or end <= start:
            logger.warning(f"fact_extractor: no JSON array found in response. Content: {content[:300]}")
            return []

        facts = json.loads(content[start:end])
        if not isinstance(facts, list):
            logger.warning(f"fact_extractor: response is not a list: {type(facts)}")
            return []

        # Validate and clean
        cleaned = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            fact_text = str(item.get("fact", "")).strip()
            if not fact_text:
                continue
            cleaned.append({
                "fact": fact_text,
                "entities": item.get("entities", []) if isinstance(item.get("entities"), list) else [],
                "date": str(item.get("date", "")).strip(),
            })

        logger.info(f"fact_extractor: extracted {len(cleaned)} facts from {len(turns)} turns")
        return cleaned

    except json.JSONDecodeError as e:
        logger.error(f"fact_extractor JSON parse error: {e}. Raw content: {content[:500]}")
    except Exception as e:
        logger.error(f"fact_extractor unexpected error ({type(e).__name__}): {e}")

    return []


def _sleep_backoff(attempt: int, reason: str) -> None:
    """Exponential backoff with jitter between retries."""
    delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt) * (0.5 + random.random()))
    logger.warning(
        f"fact_extractor: retry {attempt + 1}/{_MAX_ATTEMPTS} after {delay:.1f}s ({reason})"
    )
    time.sleep(delay)
