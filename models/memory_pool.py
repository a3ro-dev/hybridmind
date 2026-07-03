"""
Memory pool classification for HybridMind.

Classifies memory nodes into typed pools for lifecycle management.
Pure rule-based — no LLM calls.
"""
from __future__ import annotations

import re
from enum import Enum


class MemoryPool(str, Enum):
    """Logical memory pool for lifecycle routing."""
    RAW = "raw"           # Unprocessed conversation turns
    EVENTS = "events"     # Time-anchored events / appointments / facts with dates
    NOTES = "notes"       # Opinions, reflections, preferences
    SUMMARY = "summary"   # Consolidated / summarized facts


_TEMPORAL_RE = re.compile(
    r'\b(\d{4}[-/]\d{1,2}|january|february|march|april|may|june|july|august|'
    r'september|october|november|december|monday|tuesday|wednesday|thursday|'
    r'friday|saturday|sunday|yesterday|today|tomorrow|last week|next week|'
    r'last month|last year|in \d{4}|on \d|at \d|\d+ (?:days?|weeks?|months?|years?) ago)\b',
    re.I
)
_OPINION_RE = re.compile(
    r"\b(i think|i believe|i feel|i prefer|i like|i love|i hate|in my opinion|"
    r"personally|honestly|to me|it seems|i find|i noticed|i realized|i remember "
    r"thinking|prefers?|likes?|loves?|hates?)\b",
    re.I
)
_SUMMARY_RE = re.compile(
    r"\b(in general|overall|in summary|to summarize|across|on average|typically|"
    r"usually|often|generally speaking|as a whole|in total|combined)\b",
    re.I
)


def classify_memory_type(text: str) -> MemoryPool:
    """
    Classify a memory text into a MemoryPool.

    Priority: SUMMARY > EVENTS > NOTES > RAW
    """
    if not text or not text.strip():
        return MemoryPool.RAW

    t = text.strip()
    if _SUMMARY_RE.search(t):
        return MemoryPool.SUMMARY
    if _TEMPORAL_RE.search(t):
        return MemoryPool.EVENTS
    if _OPINION_RE.search(t):
        return MemoryPool.NOTES
    return MemoryPool.RAW
