"""
Memory consolidation and importance scoring for HybridMind.

Provides:
- llm_summarize(): multi-fact → single summary string via centralized LLM policy
- consolidate_sessions(): batch consolidation of old sessions into summary nodes
- importance_score(): composite score for memory lifecycle pruning

Design:
- All operations are idempotent (double-run safe).
- Pruning is soft-delete only (sets deleted_at, never removes rows).
- Production hosted inference is Z.AI; RunPod vLLM is supported as a
  self-hosted backend. The research proxy requires explicit opt-in.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import settings
from engine import llm_client

logger = logging.getLogger(__name__)

def _call_llm(messages: list, max_tokens: int = 512, model: Optional[str] = None) -> Optional[str]:
    """Call the centralized provider policy."""
    return llm_client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        model=model or settings.consolidation_model,
    )


def llm_summarize(facts: List[str], model: Optional[str] = None) -> str:
    """
    Summarize a list of fact strings into a single concise paragraph.

    Every non-empty fact is included in a bounded hierarchical request. If the
    provider is unavailable or any stage fails, return an empty string rather
    than silently storing a truncated concatenation as a successful summary.

    The result is a lossy derived retrieval aid. Callers must preserve and link
    the exact source facts.
    """
    if not facts:
        return ""
    clean_facts = [str(fact).strip() for fact in facts if str(fact).strip()]
    if not clean_facts:
        return ""

    # Bound per-request context and total provider calls. Refuse oversized work
    # rather than silently dropping facts or creating an unbounded token bill.
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for fact in clean_facts:
        if current and (len(current) >= 20 or current_chars + len(fact) > 12_000):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(fact)
        current_chars += len(fact)
    if current:
        batches.append(current)
    if len(batches) > 15:
        logger.error(
            "consolidation refused: %d facts require %d first-stage calls (limit=15)",
            len(clean_facts),
            len(batches),
        )
        return ""

    def summarize_batch(items: list[str], label: str) -> Optional[str]:
        numbered = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(items))
        messages = [
            {
                "role": "system",
                "content": (
                    "Create a concise derived retrieval summary of every supplied item. "
                    "Preserve named entities, dates, validity changes, disagreements, and "
                    "explicit causal links. Do not resolve contradictions or invent facts. "
                    "Output only the summary."
                ),
            },
            {"role": "user", "content": f"{label}:\n{numbered}\n\nDerived summary:"},
        ]
        result = _call_llm(messages, max_tokens=768, model=model)
        return result.strip() if result and result.strip() else None

    summaries: list[str] = []
    for batch in batches:
        summary = summarize_batch(batch, "Source facts")
        if summary is None:
            logger.error("consolidation failed: provider returned no usable first-stage summary")
            return ""
        summaries.append(summary)
    if len(summaries) == 1:
        return summaries[0]
    final_summary = summarize_batch(summaries, "Partial derived summaries")
    if final_summary is None:
        logger.error("consolidation failed: provider returned no usable final summary")
        return ""
    return final_summary


def _consolidate_sessions_unlocked(
    db_manager,
    min_facts: int = 5,
    max_age_hours: int = 24,
    model: Optional[str] = None,
    archive_sources: bool = False,
) -> Dict[str, Any]:
    """
    Group extracted_fact nodes by (container, session), then create a derived
    summary while retaining every exact source fact and provenance edge.

    Idempotent: the summary ID is derived from the ordered source fingerprint.
    Returns statistics dict with sessions_processed, summaries_created.
    """
    store = db_manager.sqlite_store
    if archive_sources:
        raise ValueError(
            "archive_sources is disabled: a lossy summary cannot replace exact source facts"
        )
    cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()

    # Fetch all extracted_fact nodes
    try:
        with store._cursor() as cursor:
            cursor.execute("""
                SELECT id, text, metadata, created_at FROM nodes
                WHERE json_extract(metadata, '$.type') = 'extracted_fact'
                  AND deleted_at IS NULL
                  AND created_at < ?
                ORDER BY json_extract(metadata, '$.session_id'), created_at
            """, (cutoff,))
            rows = cursor.fetchall()
    except Exception as exc:
        logger.error(
            "consolidation: failed to query nodes type=%s", type(exc).__name__
        )
        return {
            "sessions_processed": 0,
            "summaries_created": 0,
            "error_type": type(exc).__name__,
        }

    # A session name is not globally unique. Container scope is part of the key
    # so summaries cannot leak facts between corpora or evaluation runs.
    by_session: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    skipped_missing_session = 0
    for row in rows:
        meta = {}
        try:
            meta = json.loads(row["metadata"])
        except Exception:
            pass
        sid = meta.get("session_id") or meta.get("sessionId")
        if not sid:
            skipped_missing_session += 1
            continue
        container = meta.get("container_tag") or meta.get("containerTag") or "__default__"
        by_session.setdefault((str(container), str(sid)), []).append(
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": meta,
                "created_at": str(row["created_at"]),
            }
        )

    sessions_processed = 0
    summaries_created = 0
    sources_archived = 0

    failures: list[dict[str, str]] = []

    for (container, sid), nodes in by_session.items():
        if len(nodes) < min_facts:
            continue

        canonical_sources = [
            {
                "id": node["id"],
                "text": node["text"],
                "metadata": node["metadata"],
                "created_at": node["created_at"],
            }
            for node in nodes
        ]
        source_fingerprint = hashlib.sha256(
            json.dumps(
                canonical_sources,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        summary_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"hybridmind:session-summary:{container}:{sid}:{source_fingerprint}",
            )
        )
        existing = store.get_node(summary_id)
        if existing:
            logger.debug("consolidation: exact source generation already summarized: %s", summary_id)
            sessions_processed += 1
            continue

        # Summarize
        facts = [n["text"] for n in nodes]
        summary_text = llm_summarize(facts, model=model)
        if not summary_text:
            continue

        # Create summary node
        try:
            from engine.embedding import validate_embedding_4096
            embedding = validate_embedding_4096(
                db_manager.embedding_engine.embed(summary_text),
                label="session summary embedding",
            )

            summary_metadata = {
                "type": "session_summary",
                "memory_pool": "summary",
                "summary_session_id": sid,
                "container_tag": None if container == "__default__" else container,
                "source_count": len(nodes),
                "source_ids": [node["id"] for node in nodes],
                "source_fingerprint_sha256": source_fingerprint,
                "lossy_derived_summary": True,
                "summary_model": model or settings.consolidation_model,
                "summarized_at": datetime.utcnow().isoformat(),
            }
            with store.transaction():
                store.create_node(
                    node_id=summary_id,
                    text=summary_text,
                    metadata=summary_metadata,
                    embedding=embedding,
                    raw_embedding=embedding,
                )
                db_manager.vector_index.add(summary_id, embedding)
                db_manager.graph_index.add_node(summary_id)
                db_manager.bm25_index.add(summary_id, summary_text)

                # Each source is retained as an independently retrievable entity;
                # the summary is a PROV-style derivation, never a replacement.
                for source_node in nodes:
                    eid = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{summary_id}:source:{source_node['id']}",
                        )
                    )
                    store.create_edge(
                        edge_id=eid,
                        source_id=summary_id,
                        target_id=source_node["id"],
                        edge_type="derived_from",
                        weight=0.9,
                        metadata={
                            "activity": "session_consolidation",
                            "source_fingerprint_sha256": source_fingerprint,
                            "lossy": True,
                        },
                    )
                    db_manager.graph_index.add_edge(
                        edge_id=eid,
                        source_id=summary_id,
                        target_id=source_node["id"],
                        edge_type="derived_from",
                        weight=0.9,
                        activity="session_consolidation",
                        source_fingerprint_sha256=source_fingerprint,
                        lossy=True,
                    )

            summaries_created += 1
            logger.info(f"consolidation: session {sid!r} → summary node {summary_id} ({len(nodes)} facts)")
        except Exception as exc:
            # SQL rolled back. Remove any already-projected summary state so a
            # failed derivation is never visible in only a subset of indexes.
            for projection, operation in (
                ("bm25", lambda: db_manager.bm25_index.remove(summary_id)),
                ("vector", lambda: db_manager.vector_index.remove(summary_id)),
                ("graph", lambda: db_manager.graph_index.remove_node(summary_id)),
            ):
                try:
                    operation()
                except Exception as cleanup_exc:
                    logger.critical(
                        "consolidation cleanup failed projection=%s type=%s",
                        projection,
                        type(cleanup_exc).__name__,
                    )
            logger.error(
                "consolidation: failed to create summary for session=%r type=%s",
                sid,
                type(exc).__name__,
            )
            failures.append(
                {
                    "container": container,
                    "session_id": sid,
                    "error_type": type(exc).__name__,
                }
            )

        sessions_processed += 1

    return {
        "sessions_processed": sessions_processed,
        "summaries_created": summaries_created,
        "sources_archived": sources_archived,
        "sessions_total": len(by_session),
        "skipped_missing_session": skipped_missing_session,
        "failures": failures,
    }


def consolidate_sessions(
    db_manager,
    min_facts: int = 5,
    max_age_hours: int = 24,
    model: Optional[str] = None,
    archive_sources: bool = False,
) -> Dict[str, Any]:
    """Run consolidation under the shared SQL/projection mutation boundary."""
    mutation = getattr(db_manager, "mutation", None)
    if mutation is None:
        # Lightweight test doubles and standalone compatibility managers may
        # not implement coordination. Real DatabaseManager instances always do.
        return _consolidate_sessions_unlocked(
            db_manager,
            min_facts=min_facts,
            max_age_hours=max_age_hours,
            model=model,
            archive_sources=archive_sources,
        )
    with mutation():
        return _consolidate_sessions_unlocked(
            db_manager,
            min_facts=min_facts,
            max_age_hours=max_age_hours,
            model=model,
            archive_sources=archive_sources,
        )


def importance_score(
    node_id: str,
    db_manager,
    *,
    max_graph_degree: Optional[float] = None,
) -> float:
    """
    Compute a composite importance score [0, 1] for a node.

    Combines:
    - Recency: exponential decay on created_at (half-life ~30 days)
    - Degree centrality: normalized edge count in graph
    - Access frequency: metadata.access_count if tracked (default 1)
    """
    try:
        node = db_manager.sqlite_store.get_node(node_id)
        if node is None:
            return 0.0

        from config import settings

        # Recency score uses event time when available.
        created_at_raw = node.get("event_time") or node.get("created_at")
        if created_at_raw:
            try:
                if isinstance(created_at_raw, str):
                    created_dt = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
                else:
                    created_dt = created_at_raw
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                else:
                    created_dt = created_dt.astimezone(timezone.utc)
                age_days = max(
                    0.0,
                    (datetime.now(timezone.utc) - created_dt).total_seconds() / 86400,
                )
                recency = math.exp(
                    -math.log(2) * age_days / settings.salience_recency_half_life_days
                )
            except Exception:
                recency = 0.5
        else:
            recency = 0.5

        # Degree centrality
        try:
            graph = db_manager.graph_index.graph
            degree = graph.degree(node_id) if graph.has_node(node_id) else 0
            max_degree = (
                float(max_graph_degree)
                if max_graph_degree is not None
                else max(dict(graph.degree()).values(), default=1)
            )
            centrality = degree / max(max_degree, 1)
        except Exception:
            centrality = 0.0

        # Access frequency
        access_count = int(node.get("access_count") or 0)
        access_score = min(1.0, math.log1p(access_count) / math.log1p(20))  # saturates at 20 hits

        # Weighted composite shares the retrieval salience configuration.
        weights = (
            settings.salience_recency_weight,
            settings.salience_centrality_weight,
            settings.salience_frequency_weight,
        )
        score = (
            weights[0] * recency + weights[1] * centrality + weights[2] * access_score
        ) / max(sum(weights), 1e-9)
        return round(max(0.0, min(1.0, score)), 4)

    except Exception as exc:
        logger.error(
            "importance_score: failed node=%s type=%s",
            node_id,
            type(exc).__name__,
        )
        return 0.0


def check_contradiction(
    new_fact_text: str,
    existing_nodes: List[Dict[str, Any]],
    embedding_engine,
    threshold: float = 0.85,
) -> Optional[str]:
    """
    Check if new_fact_text supersedes any existing node.

    Returns the node_id of an existing node only when a simple subject/slot
    match has a different value. Semantic similarity is not evidence of
    contradiction and is deliberately not used for this decision.

    This is intentionally cheap and conservative. It is not full natural
    language inference.
    """
    if not existing_nodes or not new_fact_text.strip():
        return None
    try:
        new_slot = _extract_fact_slot(new_fact_text)
        if new_slot:
            for node in existing_nodes:
                existing_slot = _extract_fact_slot(str(node.get("text", "")))
                if (
                    existing_slot
                    and new_slot[:2] == existing_slot[:2]
                    and new_slot[2] != existing_slot[2]
                ):
                    logger.debug(f"check_contradiction: slot update -> conflict with {node.get('id')}")
                    return node.get("id")

        return None
    except Exception as exc:
        logger.debug(
            "check_contradiction: failed type=%s", type(exc).__name__
        )
        return None


_SLOT_PATTERNS = [
    re.compile(
        r"^(?:the\s+)?(?P<subject>[a-z0-9_\-\s']+?)\s+"
        r"(?P<slot>address|email|phone(?:\s+number)?|location)\s+"
        r"(?:is|=|:)\s+(?P<value>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:the\s+)?(?P<subject>[a-z0-9_\-\s']+?)\s+"
        r"(?:prefers|likes|uses|favorite\s+(?P<slot2>[a-z0-9_\-\s]+)\s+is)\s+"
        r"(?P<value>.+)$",
        re.IGNORECASE,
    ),
]


def _normalize_slot_part(value: str) -> str:
    value = value.lower().strip().rstrip(".")
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    value = value.replace("'s", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _extract_fact_slot(text: str) -> Optional[tuple[str, str, str]]:
    """Extract a simple subject/slot/value triple from update-like facts."""
    text = text.strip()
    for pattern in _SLOT_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        subject = _normalize_slot_part(match.group("subject"))
        slot = match.groupdict().get("slot") or match.groupdict().get("slot2") or "preference"
        slot = _normalize_slot_part(slot)
        value = _normalize_slot_part(match.group("value"))
        if subject and slot and value:
            return subject, slot, value
    return None
