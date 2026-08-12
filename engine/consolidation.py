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
import logging
import math
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import settings
from engine import llm_client

logger = logging.getLogger(__name__)

def _call_llm(messages: list, max_tokens: int = 512, model: Optional[str] = None) -> Optional[str]:
    """Call the centralized provider policy."""
    return llm_client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=0.3,
        model=settings.consolidation_model,
    )


def llm_summarize(facts: List[str], model: Optional[str] = None) -> str:
    """
    Summarize a list of fact strings into a single concise paragraph.

    Returns a deterministic joined summary if no LLM is available.
    Falls back to joining with semicolons so callers always get something usable.
    """
    if not facts:
        return ""
    numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts[:50]))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a memory consolidation system. Merge the following facts "
                "into one concise, information-dense summary paragraph. Preserve all "
                "key entities, dates, and relationships. Output only the summary, no "
                "preamble or explanation."
            ),
        },
        {"role": "user", "content": f"Facts to consolidate:\n{numbered}\n\nSummary:"},
    ]
    result = _call_llm(messages, max_tokens=512, model=model)
    if result and result.strip():
        return result.strip()
    # Fallback: join with semicolons (no LLM available)
    return "; ".join(facts[:20])


def consolidate_sessions(
    db_manager,
    min_facts: int = 5,
    max_age_hours: int = 24,
    model: Optional[str] = None,
    archive_sources: bool = False,
) -> Dict[str, Any]:
    """
    Group extracted_fact nodes by session, summarize old sessions.

    Idempotent: skips sessions that already have a summary node.
    Returns statistics dict with sessions_processed, summaries_created.
    """
    store = db_manager.sqlite_store
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
    except Exception as e:
        logger.error(f"consolidation: failed to query nodes: {e}")
        return {"sessions_processed": 0, "summaries_created": 0, "error": str(e)}

    # Group by session_id
    by_session: Dict[str, List[Dict]] = {}
    for row in rows:
        meta = {}
        try:
            meta = json.loads(row["metadata"])
        except Exception:
            pass
        sid = meta.get("session_id") or meta.get("sessionId", "_unknown")
        by_session.setdefault(sid, []).append(
            {"id": row["id"], "text": row["text"], "session_id": sid}
        )

    sessions_processed = 0
    summaries_created = 0
    sources_archived = 0

    for sid, nodes in by_session.items():
        if len(nodes) < min_facts:
            continue

        # Check if already summarized for this session
        try:
            with store._cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM nodes
                    WHERE json_extract(metadata, '$.type') = 'session_summary'
                      AND json_extract(metadata, '$.summary_session_id') = ?
                      AND deleted_at IS NULL
                    LIMIT 1
                """, (sid,))
                existing = cursor.fetchone()
            if existing:
                logger.debug(f"consolidation: session {sid!r} already summarized — skipping")
                sessions_processed += 1
                continue
        except Exception as e:
            logger.warning(f"consolidation: existence check failed for session {sid!r}: {e}")

        # Summarize
        facts = [n["text"] for n in nodes]
        summary_text = llm_summarize(facts, model=model)
        if not summary_text:
            continue

        # Create summary node
        try:
            from engine.embedding import validate_embedding_4096
            summary_id = str(uuid.uuid4())
            embedding = validate_embedding_4096(
                db_manager.embedding_engine.embed(summary_text),
                label="session summary embedding",
            )

            store.create_node(
                node_id=summary_id,
                text=summary_text,
                metadata={
                    "type": "session_summary",
                    "memory_pool": "summary",
                    "summary_session_id": sid,
                    "source_count": len(nodes),
                    "summarized_at": datetime.utcnow().isoformat(),
                },
                embedding=embedding,
                raw_embedding=embedding,
            )
            db_manager.vector_index.add(summary_id, embedding)
            db_manager.graph_index.add_node(summary_id)
            db_manager.bm25_index.add(summary_id, summary_text)

            # Link summary → source facts via belongs_to edges
            for source_node in nodes:
                eid = str(uuid.uuid4())
                try:
                    store.create_edge(
                        edge_id=eid,
                        source_id=summary_id,
                        target_id=source_node["id"],
                        edge_type="belongs_to",
                        weight=0.9,
                    )
                    db_manager.graph_index.add_edge(
                        edge_id=eid,
                        source_id=summary_id,
                        target_id=source_node["id"],
                        edge_type="belongs_to",
                        weight=0.9,
                    )
                except Exception as e:
                    logger.debug(f"consolidation: edge failed {summary_id}→{source_node['id']}: {e}")

            if archive_sources:
                source_ids = [source["id"] for source in nodes]
                sources_archived += store.archive_nodes(source_ids, summary_id)
                for source_id in source_ids:
                    db_manager.vector_index.remove(source_id)
                    db_manager.graph_index.remove_node(source_id)

            summaries_created += 1
            logger.info(f"consolidation: session {sid!r} → summary node {summary_id} ({len(nodes)} facts)")
        except Exception as e:
            logger.error(f"consolidation: failed to create summary node for session {sid!r}: {e}")

        sessions_processed += 1

    return {
        "sessions_processed": sessions_processed,
        "summaries_created": summaries_created,
        "sources_archived": sources_archived,
        "sessions_total": len(by_session),
    }


def importance_score(node_id: str, db_manager) -> float:
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
                    created_dt = datetime.fromisoformat(created_at_raw)
                else:
                    created_dt = created_at_raw
                age_days = (datetime.utcnow() - created_dt).total_seconds() / 86400
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
            max_degree = max(dict(graph.degree()).values(), default=1)
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

    except Exception as e:
        logger.error(f"importance_score: failed for {node_id}: {e}")
        return 0.0


def check_contradiction(
    new_fact_text: str,
    existing_nodes: List[Dict[str, Any]],
    embedding_engine,
    threshold: float = 0.85,
) -> Optional[str]:
    """
    Check if new_fact_text supersedes any existing node.

    Returns the node_id of an existing node when a simple slot-value update is
    detected, or the highest-similarity node if cosine similarity >= threshold.

    This is intentionally cheap and conservative. It is not full natural
    language inference.
    """
    if not existing_nodes or not new_fact_text.strip():
        return None
    try:
        import numpy as np
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

        new_emb = np.asarray(embedding_engine.embed(new_fact_text), dtype=np.float32)
        new_norm = new_emb / (np.linalg.norm(new_emb) + 1e-8)

        best_id = None
        best_sim = -1.0
        for node in existing_nodes:
            existing_emb = node.get("embedding")
            if existing_emb is None:
                continue
            e = np.asarray(existing_emb, dtype=np.float32)
            e_norm = e / (np.linalg.norm(e) + 1e-8)
            sim = float(np.dot(new_norm, e_norm))
            if sim > best_sim:
                best_sim = sim
                best_id = node.get("id")

        if best_sim >= threshold:
            logger.debug(f"check_contradiction: similarity {best_sim:.3f} >= {threshold} → conflict with {best_id}")
            return best_id
        return None
    except Exception as e:
        logger.debug(f"check_contradiction: failed: {e}")
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
