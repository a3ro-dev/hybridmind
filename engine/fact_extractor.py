"""
Ingest-time fact extraction using Claude Haiku via HackClub proxy.
Called once per session during ingest. Never at query time.
"""
import json
import logging
import os
from dotenv import load_dotenv

import httpx

load_dotenv()

logger = logging.getLogger(__name__)

_HC_API_KEY = os.getenv("HC_API_KEY", "")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1").rstrip("/")
_MODEL = "~anthropic/claude-haiku-latest"

_SYSTEM_PROMPT = """\
You are a fact extraction engine for conversational memory.

Given a conversation, extract every discrete, answerable fact as a JSON array.
Each element must be:
  {"fact": "<one sentence fact>", "entities": ["name1", "name2"], "date": "<YYYY-MM-DD or empty string>"}

Rules:
- Include: names, relationships, dates, locations, activities, career decisions, identity facts, numeric facts (ages, durations)
- Exclude: generic chitchat, greetings, vague sentiments
- Each fact must be self-contained and answerable without context
- If a date is mentioned or implied (e.g. "last Friday" relative to a known date), resolve it and include it
- Return ONLY the JSON array, no explanation
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

    conversation = "\n".join(lines)[:8000]  # ~6k tokens max

    if not conversation.strip():
        return []

    try:
        response = httpx.post(
            f"{_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {_HC_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": conversation},
                ],
                "max_tokens": 2048,
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Extract JSON array robustly
        start = content.find("[")
        end = content.rfind("]") + 1
        if start < 0 or end <= start:
            logger.warning("fact_extractor: no JSON array found in response")
            return []

        facts = json.loads(content[start:end])
        if not isinstance(facts, list):
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

    except httpx.HTTPStatusError as e:
        logger.warning(f"fact_extractor HTTP error: {e.response.status_code} {e.response.text[:200]}")
    except json.JSONDecodeError as e:
        logger.warning(f"fact_extractor JSON parse error: {e}")
    except Exception as e:
        logger.warning(f"fact_extractor unexpected error: {e}")

    return []
