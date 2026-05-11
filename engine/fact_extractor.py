"""
Session-level fact extraction using claude-haiku-latest.
Called ONCE per LoCoMo session at ingest time. Never at query time.
"""

import logging
from typing import Optional

from engine.llm import LLMEngine

logger = logging.getLogger(__name__)


def extract_facts_from_session(
    session_turns: list[dict],
    llm: Optional[LLMEngine] = None
) -> list[dict]:
    """
    Extract structured facts from a list of conversation turns.

    Args:
        session_turns: List of {"speaker": str, "text": str, "date": str} dicts.
                       The "date" field is optional.
        llm: Optional pre-created LLMEngine instance (creates one if None).

    Returns:
        List of fact dicts: {"fact": str, "entities": list[str], "date": str}
    """
    if not session_turns:
        return []

    if llm is None:
        try:
            llm = LLMEngine()
        except Exception as e:
            logger.error(f"Cannot create LLMEngine for fact extraction: {e}")
            return []

    # Determine the earliest date across all turns
    dates = [t.get("date", "") for t in session_turns if t.get("date")]
    earliest_date = dates[0] if dates else ""

    # Concatenate all turns into a single string for the LLM
    lines = []
    for turn in session_turns:
        date_prefix = f"[{turn.get('date', '')}] " if turn.get("date") else ""
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "").strip()
        if text:
            lines.append(f"{date_prefix}{speaker}: {text}")
    full_text = "\n".join(lines)

    if not full_text.strip():
        return []

    # Call LLMEngine.extract_metadata — returns ExtractedData with:
    #   .key_facts: list[str]
    #   .entities: list[dict]  [{name, type, description}]
    #   .relationships: list[dict]  [{source, target, relationship}]
    try:
        extracted = llm.extract_metadata(full_text)
    except Exception as e:
        logger.error(f"extract_metadata failed: {e}")
        return []

    facts: list[dict] = []

    # Extract entity names for quick lookup
    entity_names = [e.get("name", "") for e in (extracted.entities or []) if e.get("name")]

    # One fact per key_fact string
    for fact_text in (extracted.key_facts or []):
        fact_text = fact_text.strip()
        if not fact_text:
            continue
        # Find which entities are mentioned in this fact
        mentioned = [name for name in entity_names if name.lower() in fact_text.lower()]
        facts.append({
            "fact": fact_text,
            "entities": mentioned,
            "date": earliest_date,
        })

    # One fact per relationship triple
    for rel in (extracted.relationships or []):
        subject = rel.get("source", "").strip()
        predicate = rel.get("relationship", "").strip()
        obj = rel.get("target", "").strip()
        if not (subject and predicate and obj):
            continue
        fact_text = f"{subject} {predicate} {obj}"
        facts.append({
            "fact": fact_text,
            "entities": [subject, obj],
            "date": earliest_date,
        })

    logger.debug(
        f"extract_facts_from_session: {len(session_turns)} turns → "
        f"{len(facts)} facts ({len(extracted.key_facts or [])} key_facts, "
        f"{len(extracted.relationships or [])} relationships)"
    )
    return facts
