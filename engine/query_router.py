"""
Query router for HybridMind LoCoMo evaluation.
Routes questions to the appropriate search strategy using regex patterns.
Zero LLM calls — pure regex-based classification.
"""

import re

TEMPORAL_PATTERNS = re.compile(
    r'\b(when|how long|before|after|during|date|year|month|first time|last time|'
    r'how many (?:days|months|years)|\d{4})\b',
    re.IGNORECASE
)

MULTIHOP_PATTERNS = re.compile(
    r'\b(relationship between|how.*connected|both|common between|link between)\b',
    re.IGNORECASE
)

ENTITY_PATTERNS = re.compile(
    r'\b(who is|what is|where is|who was|what was|tell me about)\b',
    re.IGNORECASE
)


def route_query(query_text: str) -> dict:
    """
    Classify a query and return the appropriate search strategy.

    Args:
        query_text: The question/query string.

    Returns:
        Dict with keys:
          - "type": one of "temporal", "multihop", "entity", "default"
          - "metadata_filter": reserved for a future, evidence-backed filter
    """
    if TEMPORAL_PATTERNS.search(query_text):
        return {"type": "temporal", "metadata_filter": None}
    if MULTIHOP_PATTERNS.search(query_text):
        return {"type": "multihop", "metadata_filter": None}
    if ENTITY_PATTERNS.search(query_text):
        # Entity wording does not imply that raw turns are irrelevant.  Fact
        # extraction is optional and off by default, so forcing an
        # ``extracted_fact`` filter here can make an otherwise answerable query
        # return an empty pool.  Callers may still provide an explicit filter.
        return {"type": "entity", "metadata_filter": None}
    return {"type": "default", "metadata_filter": None}
