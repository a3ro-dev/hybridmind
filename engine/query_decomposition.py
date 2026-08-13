"""
Multi-hop query decomposition (Phase 6.2.2, docs/PHASE_6_REALISTIC.md section 4).

A single dense query embedding for a two-hop question geometrically lands
between the two evidence clusters and hits neither — this is a property of
mean-pooled query embeddings, not a tuning problem. Decomposing a
multihop-routed query into sub-questions via the RunPod LLM and retrieving
per sub-question lets each hop's retrieval anchor on its own cluster.

Off by default (see `enabled` param / settings.query_decomposition_enabled).
Both the live hybrid ranker and evaluation retrieval path opt in explicitly.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from config import settings
from engine import llm_client

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT_VERSION = "decompose_v1"

_DECOMPOSE_SYSTEM_PROMPT = (
    "Break the following multi-hop question into 2-3 simpler sub-questions "
    "that, answered in sequence, would let someone answer the original "
    "question. Each sub-question MUST only mention entities/concepts already "
    "present in the original question -- do not invent new named entities. "
    'Return ONLY a JSON array of strings, e.g. ["...", "..."].'
)

_TEMPORAL_CONSTRAINTS = re.compile(
    r"\b(before|after|during|between|latest|last|previous|first|when|in\s+\d{4})\b",
    re.IGNORECASE,
)


def _entity_tokens(text: str) -> set:
    """Coarse proxy for 'named entity': capitalized words, as a stand-in for full NER."""
    return {t.lower() for t in re.findall(r"\b[A-Z][a-zA-Z']*\b", text)}


def decompose_query(
    query_text: str,
    model: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> List[str]:
    """
    Return sub-questions for `query_text`, or [] when decomposition is
    disabled, unavailable, fails, or degenerates to <=1 sub-question (the
    fall-through guard from docs/PHASE_6_REALISTIC.md section 4).

    enabled=None defers to settings.query_decomposition_enabled (default
    False); pass enabled=True/False explicitly to override per call site
    (the eval harness does this since it wants the feature on regardless of
    the server-wide default).
    """
    if enabled is None:
        enabled = settings.query_decomposition_enabled
    if not enabled:
        return []
    query_text = query_text.strip()
    if not query_text or len(query_text) > 2_000:
        logger.warning("query_decomposition refused empty/oversized query")
        return []

    content = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
        ],
        max_tokens=300,
        temperature=0.0,
        model=model or settings.query_decomposition_model,
    )
    if not content:
        return []

    start, end = content.find("["), content.rfind("]") + 1
    if start < 0 or end <= start:
        return []
    try:
        sub_questions = json.loads(content[start:end])
    except json.JSONDecodeError:
        return []
    if not isinstance(sub_questions, list) or not all(isinstance(s, str) for s in sub_questions):
        return []

    # Protocol permits exactly 2-3 bounded sub-questions. More creates an
    # unbounded retrieval/token multiplier and is rejected, not truncated.
    if not 2 <= len(sub_questions) <= 3:
        return []

    # Guard: reject sub-questions that introduce entities absent from the original query.
    original_entities = _entity_tokens(query_text)
    accepted = []
    seen = set()
    for sq in sub_questions:
        sq = " ".join(sq.split())
        normalized = sq.casefold()
        if not sq or len(sq) > 500 or normalized == query_text.casefold() or normalized in seen:
            continue
        novel = _entity_tokens(sq) - original_entities
        if novel:
            logger.debug(f"query_decomposition: rejecting sub-question with novel entities {novel}: {sq!r}")
            continue
        accepted.append(sq)
        seen.add(normalized)

    # Do not let decomposition erase explicit temporal constraints from the
    # original question. At least one accepted sub-question must retain each
    # lexical constraint marker.
    constraints = {match.group(0).casefold() for match in _TEMPORAL_CONSTRAINTS.finditer(query_text)}
    accepted_text = " ".join(accepted).casefold()
    if any(constraint not in accepted_text for constraint in constraints):
        logger.debug("query_decomposition rejected: temporal constraint was lost")
        return []

    return accepted if len(accepted) > 1 else []
