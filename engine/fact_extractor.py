"""
Ingest-time fact extraction.
Called once per session during ingest. Never at query time.

Primary backend: self-hosted RunPod vLLM (engine/runpod_llm.py), when
RUNPOD_LLM_ENDPOINT_ID is configured. Falls back to the Hack Club proxy
otherwise, so existing deployments without RunPod keep working unchanged.

Uses a shared pooled httpx client + bounded retry with exponential backoff and
jitter, so transient HC proxy drops ("socket connection was closed
unexpectedly", connection resets, 429/5xx) don't abort a benchmark run.
"""
import json
import logging
import os
import random
import time
from typing import Optional
from dotenv import load_dotenv

import httpx

from engine import runpod_llm

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

    if not runpod_llm.is_configured() and not _HC_API_KEY:
        logger.warning("Neither RunPod nor HC_API_KEY configured — fact extraction disabled")
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

    _FACT_JSON_SCHEMA = {
        "name": "facts",
        "schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string"},
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "date": {"type": "string"},
                        },
                        "required": ["fact", "entities", "date"],
                    },
                }
            },
            "required": ["facts"],
        },
        "strict": True,
    }

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": conversation},
    ]
    response_format = {"type": "json_schema", "json_schema": _FACT_JSON_SCHEMA}

    logger.info(
        f"fact_extractor: extracting facts for {len(turns)} turns ({len(conversation)} chars)"
    )
    content = _call_llm(messages, max_tokens=1536, response_format=response_format)
    if content is None:
        return []

    cleaned = _parse_facts_content(content)

    # Retry if we got 0 facts — rephrase to elicit at least a few
    if not cleaned and turns:
        logger.info("fact_extractor: 0 facts extracted, retrying with rephrased prompt")
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "You extract facts from conversations. "
                    "Return a JSON array of at least 3 short facts. "
                    'Format: [{"fact":"...","entities":[],"date":""}]. '
                    "Return ONLY the JSON array."
                ),
            },
            {"role": "user", "content": conversation[:8000]},
        ]
        retry_content = _call_llm(retry_messages, max_tokens=1024)
        if retry_content:
            cleaned.extend(_parse_facts_content(retry_content))

    logger.info(f"fact_extractor: extracted {len(cleaned)} facts from {len(turns)} turns")
    return cleaned


def _call_llm(
    messages: list[dict],
    max_tokens: int,
    response_format: Optional[dict] = None,
) -> Optional[str]:
    """
    Call the configured LLM backend: self-hosted RunPod vLLM first (if
    configured), falling back to the Hack Club proxy. Returns raw content
    string, or None if both backends fail/are unavailable.
    """
    if runpod_llm.is_configured():
        content = runpod_llm.chat_completion(
            messages, max_tokens=max_tokens, temperature=0.0, response_format=response_format,
        )
        if content is not None:
            return content
        logger.warning("fact_extractor: RunPod LLM unavailable/failed, falling back to HC proxy")

    if not _HC_API_KEY:
        logger.warning("fact_extractor: HC_API_KEY not set and RunPod unavailable — cannot extract facts")
        return None

    payload = {
        "model": _MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if response_format:
        payload["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {_HC_API_KEY}",
        "Content-Type": "application/json",
    }
    client = _get_client()

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.post(f"{_BASE_URL}/chat/completions", headers=headers, json=payload)
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, f"HTTP {status}")
                continue
            body = e.response.text[:500] if e.response is not None else ""
            logger.error(f"fact_extractor HTTP error: {status} {body}")
            return None
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
            return None
        except Exception as e:
            logger.error(f"fact_extractor unexpected error ({type(e).__name__}): {e}")
            return None
    return None


def _parse_facts_content(content: str) -> list[dict]:
    """Parse+clean one LLM response into a validated facts list. Returns [] on any failure."""
    if not content:
        return []
    try:
        logger.debug(f"fact_extractor: raw response ({len(content)} chars): {content[:500]}")

        # Parse: try json_schema envelope first, then raw array fallback
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "facts" in parsed:
            raw_facts = parsed["facts"]
        elif isinstance(parsed, list):
            raw_facts = parsed
        else:
            # Try extracting a JSON array from free-text response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                raw_facts = json.loads(content[start:end])
            else:
                logger.warning(f"fact_extractor: unparseable response: {content[:300]}")
                return []

        if not isinstance(raw_facts, list):
            logger.warning(f"fact_extractor: facts is not a list: {type(raw_facts)}")
            return []

        cleaned = []
        for item in raw_facts:
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
