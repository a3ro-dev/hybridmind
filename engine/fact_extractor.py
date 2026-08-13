"""
Ingest-time fact extraction.
Called once per session during ingest. Never at query time.

Provider policy, credentials, pooling, and retries are centralized in
``engine.llm_client``. The research proxy is unavailable unless its explicit
configuration opt-in is enabled.
"""
import json
import logging
from datetime import datetime
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
5. Return an empty facts array when the input contains no durable, answerable facts
6. Tag external facts as world, first-person events as experience, direct perceptions as observation, and preferences/beliefs as opinion
7. Confidence measures extraction certainty, not importance
8. Add caused_by only for explicit cause/effect statements and reference exact fact strings from this output
9. Return ONLY a valid JSON array, no markdown, no explanation, no code fences

Example output:
[{"fact": "Alice works as a software engineer at Google", "entities": ["Alice", "Google"], "date": "", "memory_kind": "world", "confidence": 0.95, "caused_by": []}, {"fact": "Alice moved to San Francisco in 2023", "entities": ["Alice", "San Francisco"], "date": "2023-01-01", "memory_kind": "experience", "confidence": 0.9, "caused_by": []}]
"""


class FactExtractionError(RuntimeError):
    """Fact extraction was requested but could not be completed faithfully."""


def extract_facts_from_session(turns: list[dict]) -> list[dict]:
    """
    Extract discrete facts from a list of conversation turns.

    Controlled by ``settings.fact_extraction_enabled`` (default false).

    Args:
        turns: List of dicts with keys: speaker (str), text (str), date (str)

    Returns:
        List of dicts: {fact: str, entities: list[str], date: str}
        Returns [] only for empty/factless input. Provider and parse failures
        raise ``FactExtractionError`` so callers cannot report false success.
    """
    # Quick check: fact extraction must be explicitly enabled
    if not settings.fact_extraction_enabled:
        raise FactExtractionError("fact extraction is disabled by configuration")

    if not turns:
        return []

    if not llm_client.is_configured():
        raise FactExtractionError("no policy-allowed fact extraction provider is configured")

    # Build the conversation text with date and speaker context
    lines: list[str] = []
    for t in turns:
        date = t.get("date", "").strip()
        speaker = t.get("speaker", "").strip()
        text = t.get("text", "").strip()
        if not text:
            continue
        prefix = f"[{date}] " if date else ""
        lines.append(f"{prefix}{speaker}: {text}")

    if not lines:
        return []

    max_chars = int(getattr(settings, "fact_extraction_max_chars_per_request", 12_000))
    max_requests = int(getattr(settings, "fact_extraction_max_requests_per_session", 8))
    if max_chars < 1 or max_requests < 1:
        raise FactExtractionError("fact extraction request ceilings must be positive")

    conversations: list[str] = []
    current: list[str] = []
    current_chars = 0
    for line in lines:
        if len(line) > max_chars:
            raise FactExtractionError(
                f"one conversation turn exceeds the {max_chars}-character request ceiling"
            )
        projected = current_chars + len(line) + (1 if current else 0)
        if current and projected > max_chars:
            conversations.append("\n".join(current))
            current = []
            current_chars = 0
        current.append(line)
        current_chars += len(line) + (1 if len(current) > 1 else 0)
    if current:
        conversations.append("\n".join(current))
    if len(conversations) > max_requests:
        raise FactExtractionError(
            f"session requires {len(conversations)} extraction requests; ceiling is {max_requests}"
        )

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

    response_format = {"type": "json_schema", "json_schema": _FACT_JSON_SCHEMA}

    logger.info(
        "fact_extractor: extracting facts for %d turns in %d bounded request(s)",
        len(turns),
        len(conversations),
    )
    cleaned: list[dict] = []
    for conversation in conversations:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": conversation},
        ]
        content = _call_llm(messages, max_tokens=1536, response_format=response_format)
        if content is None:
            raise FactExtractionError("fact extraction provider returned no usable response")
        cleaned.extend(_parse_facts_content(content))

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
    """Parse and validate a response; raise on malformed provider output."""
    if not content:
        raise FactExtractionError("empty fact extraction response")
    try:
        logger.debug("fact_extractor: received response chars=%d", len(content))

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
                raise FactExtractionError("fact extraction response contains no JSON facts array")

        if not isinstance(raw_facts, list):
            raise FactExtractionError("fact extraction response field 'facts' is not an array")

        cleaned = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_text = str(item.get("fact", "")).strip()
            if not fact_text:
                continue
            memory_kind = str(item.get("memory_kind", "world")).strip().lower()
            if memory_kind not in {"world", "experience", "observation", "opinion"}:
                raise FactExtractionError("fact extraction returned an invalid memory_kind")
            entities = item.get("entities", [])
            caused_by = item.get("caused_by", [])
            if not isinstance(entities, list) or not all(isinstance(value, str) for value in entities):
                raise FactExtractionError("fact extraction returned invalid entities")
            if not isinstance(caused_by, list) or not all(isinstance(value, str) for value in caused_by):
                raise FactExtractionError("fact extraction returned invalid caused_by")
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError) as exc:
                raise FactExtractionError("fact extraction returned invalid confidence") from exc
            if not 0.0 <= confidence <= 1.0:
                raise FactExtractionError("fact extraction confidence is outside [0, 1]")
            date_value = str(item.get("date", "")).strip()
            if date_value:
                try:
                    # Validate the complete provider value.  The permissive temporal
                    # query parser is intentionally not used here because extracted
                    # dates are persisted as event/valid time.
                    datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise FactExtractionError(
                        "fact extraction returned an invalid ISO-8601 date"
                    ) from exc
            cleaned.append({
                "fact": fact_text,
                "entities": entities,
                "date": date_value,
                "memory_kind": memory_kind,
                "confidence": confidence,
                "caused_by": caused_by,
            })
        return cleaned

    except json.JSONDecodeError as e:
        logger.error("fact_extractor JSON parse error type=%s", type(e).__name__)
        raise FactExtractionError("fact extraction returned malformed JSON") from e
    except FactExtractionError:
        raise
    except Exception as e:
        logger.error("fact_extractor validation error type=%s", type(e).__name__)
        raise FactExtractionError("fact extraction response validation failed") from e
