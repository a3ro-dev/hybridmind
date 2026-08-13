"""
Community detection for HybridMind graph.

Uses networkx's Louvain algorithm (networkx >= 3.0, already in requirements)
to identify clusters of densely-connected nodes, then optionally creates a
community summary node for each cluster via the same LLM used for consolidation.

Design:
- Idempotent: checks for existing community_summary node before creating.
- Community IDs are deterministic per run but may change between runs as
  new nodes/edges are added. The summary node stores the member list.
- Admin-endpoint-only: never runs automatically at ingest.
"""
from __future__ import annotations

import json
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def detect_communities(graph_index) -> Dict[str, int]:
    """
    Run Louvain community detection on the graph.

    Returns {node_id: community_id} where community_id is an integer.
    Returns empty dict if graph has fewer than 2 connected nodes. This is a
    structural clustering heuristic, not causal or temporal reasoning.
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        logger.error("community_detector: networkx not available")
        return {}

    try:
        graph = graph_index.graph
        if graph.number_of_nodes() < 2:
            logger.info("community_detector: graph too small for community detection")
            return {}

        # Louvain works on undirected graphs
        if graph.is_directed():
            G = graph.to_undirected()
        else:
            G = graph

        # Remove isolated nodes for cleaner communities
        G = G.copy()
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

        if G.number_of_nodes() < 2:
            return {}

        communities = louvain_communities(G, seed=42)
        node_to_community = {}
        for cid, members in enumerate(communities):
            for node_id in members:
                node_to_community[node_id] = cid

        logger.info(
            f"community_detector: {len(communities)} communities detected "
            f"across {G.number_of_nodes()} nodes"
        )
        return node_to_community
    except Exception as exc:
        logger.error(
            "community_detector: detection failed type=%s", type(exc).__name__
        )
        return {}


def run_community_detection(db_manager) -> Dict[str, Any]:
    """
    Detect communities, summarize each, create community_summary nodes.

    Idempotent: existing summaries for the same community_id are preserved
    (detection always starts fresh so community IDs may shift; summaries
    created in this run use a fresh UUID-based community_id to avoid confusion).

    Returns summary stats.
    """
    from engine.consolidation import llm_summarize

    node_to_community = detect_communities(db_manager.graph_index)
    if not node_to_community:
        return {"communities_found": 0, "summaries_created": 0}

    # Group nodes by community
    community_nodes: Dict[int, List[str]] = {}
    for node_id, cid in node_to_community.items():
        community_nodes.setdefault(cid, []).append(node_id)

    store = db_manager.sqlite_store
    communities_created = 0

    for cid, member_ids in community_nodes.items():
        if len(member_ids) < 3:
            continue  # skip tiny communities

        # Preserve every live member. llm_summarize enforces its own explicit
        # request/call ceilings and refuses instead of silently truncating.
        member_texts: List[str] = []
        live_member_ids: List[str] = []
        for nid in sorted(member_ids):
            try:
                node = store.get_node(nid)
                if node and node.get("text"):
                    member_texts.append(node["text"])
                    live_member_ids.append(nid)
            except Exception as exc:
                logger.warning(
                    "community_detector: member read failed type=%s", type(exc).__name__
                )

        if not member_texts:
            continue

        summary_text = llm_summarize(member_texts)
        if not summary_text:
            logger.error("community_detector: summary failed for community %s", cid)
            continue

        # Create community summary node
        try:
            from engine.embedding import validate_embedding_4096
            member_fingerprint = hashlib.sha256(
                json.dumps(live_member_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            summary_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"hybridmind:community-summary:{member_fingerprint}",
                )
            )
            if store.get_node(summary_id):
                continue
            embedding = validate_embedding_4096(
                db_manager.embedding_engine.embed(summary_text),
                label="community summary embedding",
            )

            store.create_node(
                node_id=summary_id,
                text=summary_text,
                metadata={
                    "type": "community_summary",
                    "community_id": cid,
                    "member_count": len(live_member_ids),
                    "member_ids": live_member_ids,
                    "member_fingerprint_sha256": member_fingerprint,
                    "lossy_derived_summary": True,
                    "detected_at": datetime.utcnow().isoformat(),
                },
                embedding=embedding,
                raw_embedding=embedding,
            )
            db_manager.vector_index.add(summary_id, embedding)
            db_manager.graph_index.add_node(summary_id)
            db_manager.bm25_index.add(summary_id, summary_text)

            # Link community summary → member nodes via belongs_to
            for member_id in live_member_ids:
                eid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{summary_id}:member:{member_id}"))
                try:
                    store.create_edge(
                        edge_id=eid,
                        source_id=summary_id,
                        target_id=member_id,
                        edge_type="belongs_to",
                        weight=0.5,
                    )
                    db_manager.graph_index.add_edge(
                        edge_id=eid,
                        source_id=summary_id,
                        target_id=member_id,
                        edge_type="belongs_to",
                        weight=0.5,
                    )
                except Exception as exc:
                    logger.warning(
                        "community_detector: provenance edge failed type=%s",
                        type(exc).__name__,
                    )

            communities_created += 1
            logger.info(f"community_detector: community {cid} → summary node {summary_id} ({len(member_ids)} members)")
        except Exception as e:
            logger.error(
                "community_detector: failed to create summary for community %s type=%s",
                cid,
                type(e).__name__,
            )

    return {
        "communities_found": len(community_nodes),
        "communities_summarized": sum(1 for m in community_nodes.values() if len(m) >= 3),
        "summaries_created": communities_created,
    }
