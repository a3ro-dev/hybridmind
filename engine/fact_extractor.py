"""
Ingest-time fact extraction.
Called once per session during ingest. Never at query time.

Provider policy, credentials, pooling, and retries are centralized in
``engine.llm_client``. The research proxy is unavailable unless its explicit
configuration opt-in is enabled.
"""
import json
import logging
from typing import Optional

from config import settings
from engine import llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise fact extraction engine. Given a conversation, extract EVERY discrete, answerable fact.

Return a JSON array where each element is:
{"fact": "<one clear, self-contained sentence>", "entities": ["name1", "name2"], "date": "<ISO-8601 date or empty>", "memory_kind": "world|experience|observation|opinion", "confidence": 0.0, "caused_by": ["<exact fact sentence from this output>"]}

RULES:
1. Extract ALL facts: names, ages, relationships, dates, locations, jobs, hobbies, preferences, plans, events, health info, family details, education, achievements, emotions, numeric values
2. Each fact MUST be self-contained — readable without any context
3. Resolve relative dates (e.g. "last Friday" → actual date if known)
4. Include minor details — they may be queried later
5. Extract 5-20 facts per conversation session
6. Tag external facts as world, first-person events as experience, direct perceptions as observation, and preferences/beliefs as opinion
7. Confidence measures extraction certainty, not importance
8. Add caused_by only for explicit cause/effect statements and reference exact fact strings from this output
9. Return ONLY a valid JSON array, no markdown, no explanation, no code fences

Example output:
[{"fact": "Alice works as a software engineer at Google", "entities": ["Alice", "Google"], "date": "", "memory_kind": "world", "confidence": 0.95, "caused_by": []}, {"fact": "Alice moved to San Francisco in 2023", "entities": ["Alice", "San Francisco"], "date": "2023-01-01", "memory_kind": "experience", "confidence": 0.9, "caused_by": []}]
"""


def extract_facts_from_session(turns: list[dict]) -> list[dict]:
    """
    Extract discrete facts from a list of conversation turns.

    Controlled by ``settings.fact_extraction_enabled`` (default false).

    Args:
        turns: List of dicts with keys: speaker (str), text (str), date (str)

    Returns:
        List of dicts: {fact: str, entities: list[str], date: str}
        Returns [] on any failure — caller must handle gracefully.
    """
    # Quick check: fact extraction must be explicitly enabled
    if not settings.fact_extraction_enabled:
        return []

    if not turns:
        return []

    if not llm_client.is_configured():
        logger.warning("No policy-allowed LLM is configured — fact extraction disabled")
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
                            "memory_kind": {"type": "string", "enum": ["world", "experience", "observation", "opinion"]},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "caused_by": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["fact", "entities", "date", "memory_kind", "confidence", "caused_by"],
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
                    'Format: [{"fact":"...","entities":[],"date":"","memory_kind":"world","confidence":0.9}]. '
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
    """Call the centralized provider policy and return raw content, if available."""
    return llm_client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        model=settings.fact_model,
        response_format=response_format,
    )


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
                "memory_kind": str(item.get("memory_kind", "world")).strip().lower(),
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 1.0)))),
                "caused_by": item.get("caused_by", []) if isinstance(item.get("caused_by"), list) else [],
            })
        return cleaned

    except json.JSONDecodeError as e:
        logger.error(f"fact_extractor JSON parse error: {e}. Raw content: {content[:500]}")
    except Exception as e:
        logger.error(f"fact_extractor unexpected error ({type(e).__name__}): {e}")
    return []
