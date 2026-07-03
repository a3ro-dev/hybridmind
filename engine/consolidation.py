"""
Memory consolidation and importance scoring for HybridMind.

Provides:
- llm_summarize(): multi-fact → single summary string via HC LLM proxy
- consolidate_sessions(): batch consolidation of old sessions into summary nodes
- importance_score(): composite score for memory lifecycle pruning

Design:
- All operations are idempotent (double-run safe).
- Pruning is soft-delete only (sets deleted_at, never removes rows).
- Uses the same HC proxy + retry pattern as fact_extractor.py.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_HC_API_KEY = os.getenv("HC_API_KEY", "")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1").rstrip("/")
_DEFAULT_MODEL = os.getenv("HYBRIDMIND_CONSOLIDATION_MODEL", "openai/gpt-4o-mini")

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.5
_BACKOFF_CAP = 20.0

_CLIENT: Optional[httpx.Client] = None


def _get_client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0),
        )
    return _CLIENT


def _sleep_backoff(attempt: int, reason: str) -> None:
    delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt) * (0.5 + random.random()))
    logger.warning(f"consolidation: retry {attempt+1}/{_MAX_ATTEMPTS} in {delay:.1f}s ({reason})")
    time.sleep(delay)


def _call_llm(messages: list, max_tokens: int = 512, model: Optional[str] = None) -> Optional[str]:
    """Single LLM call with retry/backoff. Returns content or None."""
    if not _HC_API_KEY:
        return None
    model = model or _DEFAULT_MODEL
    headers = {"Authorization": f"Bearer {_HC_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
    client = _get_client()
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = client.post(f"{_BASE_URL}/chat/completions", headers=headers, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < _MAX_ATTEMPTS - 1:
                    _sleep_backoff(attempt, f"HTTP {resp.status_code}")
                    continue
                return None
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, type(e).__name__)
                continue
            logger.error(f"consolidation: LLM call failed after {_MAX_ATTEMPTS} attempts: {e}")
            return None
        except Exception as e:
            logger.error(f"consolidation: unexpected LLM error: {e}")
            return None
    return None


def llm_summarize(facts: List[str], model: Optional[str] = None) -> str:
    """
    Summarize a list of fact strings into a single concise paragraph.

    Returns empty string if LLM is unavailable (no HC_API_KEY or network fail).
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
            import numpy as np
            summary_id = str(uuid.uuid4())
            embedding = db_manager.embedding_engine.embed(summary_text)
            embedding = np.asarray(embedding, dtype=np.float32)

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

            summaries_created += 1
            logger.info(f"consolidation: session {sid!r} → summary node {summary_id} ({len(nodes)} facts)")
        except Exception as e:
            logger.error(f"consolidation: failed to create summary node for session {sid!r}: {e}")

        sessions_processed += 1

    return {
        "sessions_processed": sessions_processed,
        "summaries_created": summaries_created,
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

        # Recency score (half-life 30 days)
        created_at_raw = node.get("created_at")
        if created_at_raw:
            try:
                if isinstance(created_at_raw, str):
                    created_dt = datetime.fromisoformat(created_at_raw)
                else:
                    created_dt = created_at_raw
                age_days = (datetime.utcnow() - created_dt).total_seconds() / 86400
                recency = math.exp(-math.log(2) * age_days / 30)
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
        meta = node.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        access_count = int(meta.get("access_count", 1))
        access_score = min(1.0, math.log1p(access_count) / math.log1p(20))  # saturates at 20 hits

        # Weighted composite
        score = 0.4 * recency + 0.4 * centrality + 0.2 * access_score
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
    Check if new_fact_text semantically contradicts any existing node.

    Returns the node_id of the highest-similarity conflicting node if
    cosine similarity >= threshold, else None.

    Uses dense embedding similarity only — no LLM call. Cheap and fast.
    """
    if not existing_nodes or not new_fact_text.strip():
        return None
    try:
        import numpy as np
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
