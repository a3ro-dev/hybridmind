"""Scope-local retrieval corpus shared by the dense, sparse and graph channels.

A ``ScopedCorpus`` is a derived, rebuildable view over the authoritative
SQLite rows (or, in offline benchmark harnesses, over dataset records) for one
retrieval scope: a conversation, a LongMemEval haystack, or one container.
Every channel ranks the same ``TurnRecord`` list, so channel outputs share one
identity space (``node_id``) and one stable evidence identity (``evidence_id``).

Chronological order is part of the corpus contract: sequence edges, neighbour
windows and evidence packing all read ``turns`` in their sorted order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Metadata keys that may carry a stable, citable evidence ID, in priority order.
EVIDENCE_ID_KEYS = ("evidence_id", "dia_id", "source_id")
SESSION_KEYS = ("session_id", "sessionId", "locomo_session")


@dataclass(frozen=True)
class TurnRecord:
    """One retrievable unit (normally one conversation turn)."""

    node_id: str
    evidence_id: str
    text: str
    session_key: str = ""
    # Chronological sort key inside the scope. Tuples compare element-wise, so
    # callers use e.g. (session_datetime_iso, session_number, turn_index).
    order: Tuple[Any, ...] = ()
    speaker: str = ""
    timestamp: Optional[str] = None
    # Text indexed by the sparse channel and embedded by the dense channel.
    # ``None`` means "same as text". Index text may carry derived expansions
    # (speaker, resolved dates, captions) without changing the rendered value.
    index_text: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        return self.text if self.index_text is None else self.index_text


class ScopedCorpus:
    """Chronologically ordered, identity-checked turn list for one scope."""

    def __init__(
        self,
        turns: Iterable[TurnRecord],
        scope_key: str = "",
        generation: Optional[int] = None,
        as_of: Optional[str] = None,
    ):
        ordered = sorted(turns, key=lambda t: (t.order, t.node_id))
        seen: Dict[str, int] = {}
        for index, turn in enumerate(ordered):
            if not turn.node_id:
                raise ValueError("turn node_id must be non-empty")
            if turn.node_id in seen:
                raise ValueError(f"duplicate node_id in scope: {turn.node_id!r}")
            seen[turn.node_id] = index
        self.turns: List[TurnRecord] = ordered
        self.index_of: Dict[str, int] = seen
        self.scope_key = scope_key
        self.generation = generation
        self.as_of = as_of

    def __len__(self) -> int:
        return len(self.turns)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.index_of

    def turn(self, node_id: str) -> TurnRecord:
        return self.turns[self.index_of[node_id]]

    def same_session_neighbors(self, node_id: str, window: int) -> List[str]:
        """Node IDs within ``window`` positions in the same session, in order."""
        i = self.index_of[node_id]
        session = self.turns[i].session_key
        return [
            self.turns[j].node_id
            for j in range(max(0, i - window), min(len(self.turns), i + window + 1))
            if self.turns[j].session_key == session
        ]

    def rank_key(self, node_id: str) -> int:
        """Deterministic tie-break: earlier turns first."""
        return self.index_of[node_id]

    def sort_scores(self, scores: Mapping[str, float]) -> List[Tuple[str, float]]:
        """Sort ``{node_id: score}`` descending with the corpus tie-break."""
        return sorted(scores.items(), key=lambda item: (-item[1], self.index_of[item[0]]))


def evidence_id_for(node_id: str, metadata: Mapping[str, Any]) -> str:
    for key in EVIDENCE_ID_KEYS:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return node_id


def session_key_for(metadata: Mapping[str, Any]) -> str:
    for key in SESSION_KEYS:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def turn_from_node(node: Mapping[str, Any]) -> TurnRecord:
    """Build a ``TurnRecord`` from a ``SQLiteStore`` node dict."""
    metadata = dict(node.get("metadata") or {})
    turn_index = metadata.get("turn_index")
    session_number = metadata.get("session_number", metadata.get("locomo_session"))
    timestamp = node.get("event_time") or metadata.get("timestamp") or metadata.get("date")
    parsed = parse_turn_time(timestamp)
    order: Sequence[Any] = (
        # Parsed instants sort in time; unparseable or missing times sort
        # after them, never lexicographically among them.
        0 if parsed is not None else 1,
        parsed or "",
        _as_int(session_number),
        _as_int(turn_index),
        str(node.get("created_at") or ""),
    )
    sparse = metadata.get("sparse_text")
    return TurnRecord(
        node_id=str(node["id"]),
        evidence_id=evidence_id_for(str(node["id"]), metadata),
        text=str(node.get("text") or ""),
        session_key=session_key_for(metadata),
        order=tuple(order),
        speaker=str(metadata.get("speaker") or metadata.get("role") or ""),
        timestamp=str(timestamp) if timestamp else None,
        index_text=str(sparse) if sparse else None,
        metadata=metadata,
    )


_TIME_FORMATS = (
    "%I:%M %p on %d %B, %Y",  # LoCoMo: "1:56 pm on 8 May, 2023"
    "%Y/%m/%d (%a) %H:%M",    # LongMemEval: "2023/05/20 (Sat) 02:21"
)


def parse_turn_time(value: Any) -> Optional[str]:
    """ISO-8601 UTC for ISO, LoCoMo and LongMemEval timestamps; else None."""
    if value in (None, ""):
        return None
    from engine.temporal import parse_datetime

    parsed = parse_datetime(value)
    if parsed is None:
        text = " ".join(str(value).split())
        for fmt in _TIME_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    return parsed.astimezone(timezone.utc).isoformat() if parsed is not None else None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
