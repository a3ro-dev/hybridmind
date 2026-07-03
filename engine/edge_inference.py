"""
Automatic edge inference for HybridMind.

Runs at ingest time (node creation / bulk import / session-facts) when
HYBRIDMIND_AUTO_EDGES_ENABLED=true.

Two mechanisms (both off by default):
1. Cosine-threshold edges — vector search for top-N similar nodes,
   add a `similar_to` edge when cosine score >= threshold.
   Threshold: HYBRIDMIND_AUTO_EDGE_COSINE_THRESHOLD (default 0.75)
   Max edges per node: HYBRIDMIND_AUTO_EDGE_MAX_PER_NODE (default 5)

2. Entity co-occurrence edges — link nodes that share named entities.
   Sources (in priority order):
     a) Entities already extracted by fact_extractor and stored in
        node.metadata["entities"].
     b) spaCy NER (en_core_web_sm) on raw node text, when spaCy is installed
        and HYBRIDMIND_AUTO_EDGE_ENTITY_ENABLED=true.

Both mechanisms are designed to be called immediately after a node is fully
inserted (text, embedding, and graph node all registered).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _settings():
    from config import settings
    return settings


# --------------------------------------------------------------------- #
# Cosine-threshold edges
# --------------------------------------------------------------------- #

def infer_cosine_edges(
    node_id: str,
    embedding,
    vector_index,
    sqlite_store,
    graph_index,
    threshold: Optional[float] = None,
    max_edges: Optional[int] = None,
) -> int:
    """
    Find the top-N most similar existing nodes and create `similar_to` edges.

    Returns the number of edges created.
    """
    cfg = _settings()
    threshold = threshold if threshold is not None else cfg.auto_edge_cosine_threshold
    max_edges = max_edges if max_edges is not None else cfg.auto_edge_max_per_node

    try:
        import numpy as np
        emb = np.asarray(embedding, dtype=np.float32)
        candidates: List[Tuple[str, float]] = vector_index.search(emb, top_k=max_edges + 5)
    except Exception as e:
        logger.debug(f"edge_inference: vector search failed for {node_id}: {e}")
        return 0

    created = 0
    for cand_id, score in candidates:
        if cand_id == node_id:
            continue
        if float(score) < threshold:
            continue
        if created >= max_edges:
            break
        try:
            edge_id = str(uuid.uuid4())
            weight = round(min(1.0, float(score)), 4)
            sqlite_store.create_edge(edge_id, node_id, cand_id, "similar_to", weight)
            graph_index.add_edge(
                source_id=node_id,
                target_id=cand_id,
                edge_type="similar_to",
                weight=weight,
                edge_id=edge_id,
            )
            created += 1
        except Exception as e:
            logger.debug(f"edge_inference: failed to create similar_to edge {node_id}→{cand_id}: {e}")

    if created:
        logger.debug(f"edge_inference: {created} cosine edges created for {node_id} (τ={threshold})")
    return created


# --------------------------------------------------------------------- #
# Entity co-occurrence edges
# --------------------------------------------------------------------- #

def _extract_entities_spacy(text: str) -> List[str]:
    """Extract named entities using spaCy (optional dep)."""
    try:
        import spacy
        nlp = _get_spacy_model()
        doc = nlp(text)
        return list({ent.text.lower() for ent in doc.ents if len(ent.text) > 1})
    except Exception:
        return []


_spacy_nlp = None


def _get_spacy_model():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model not found. Install with: "
                "python -m spacy download en_core_web_sm"
            )
    return _spacy_nlp


def infer_entity_edges(
    node_id: str,
    node_metadata: Dict[str, Any],
    node_text: str,
    sqlite_store,
    graph_index,
    entity_index: Optional[Dict[str, List[str]]] = None,
) -> int:
    """
    Create `co_occurs` edges between nodes that share named entities.

    entity_index (optional): in-memory {entity_str → [node_id, ...]} for the
    current batch.  When absent, tries sqlite_store.search_nodes_by_entity().

    Returns the number of edges created.
    """
    cfg = _settings()

    # Collect entities: prefer pre-extracted from metadata, then spaCy
    entities: List[str] = []
    meta_entities = node_metadata.get("entities", [])
    if isinstance(meta_entities, list):
        entities = [str(e).lower() for e in meta_entities if e]

    if not entities and cfg.auto_edge_entity_enabled:
        entities = _extract_entities_spacy(node_text)

    if not entities:
        return 0

    created = 0
    seen_pairs = set()

    for entity in entities:
        # Find other nodes mentioning the same entity
        related_ids: List[str] = []
        if entity_index is not None:
            related_ids = [nid for nid in entity_index.get(entity, []) if nid != node_id]
        else:
            try:
                related_ids = sqlite_store.search_nodes_by_entity(entity, exclude_id=node_id)
            except Exception:
                continue  # method may not exist in older versions

        for rel_id in related_ids[:cfg.auto_edge_max_per_node]:
            pair = tuple(sorted([node_id, rel_id]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            try:
                edge_id = str(uuid.uuid4())
                sqlite_store.create_edge(edge_id, node_id, rel_id, "co_occurs", 0.6)
                graph_index.add_edge(
                    source_id=node_id,
                    target_id=rel_id,
                    edge_type="co_occurs",
                    weight=0.6,
                    edge_id=edge_id,
                )
                created += 1
            except Exception as e:
                logger.debug(f"edge_inference: co_occurs edge failed {node_id}→{rel_id}: {e}")

    if created:
        logger.debug(f"edge_inference: {created} entity edges created for {node_id}")
    return created


# --------------------------------------------------------------------- #
# Unified ingest-time hook
# --------------------------------------------------------------------- #

def run_auto_edge_inference(
    node_id: str,
    embedding,
    node_metadata: Dict[str, Any],
    node_text: str,
    vector_index,
    sqlite_store,
    graph_index,
    entity_index: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, int]:
    """
    Run all enabled auto-edge mechanisms for a newly ingested node.
    Returns {mechanism: edges_created}.
    """
    cfg = _settings()
    if not cfg.auto_edges_enabled:
        return {}

    result = {}

    cosine_n = infer_cosine_edges(
        node_id=node_id,
        embedding=embedding,
        vector_index=vector_index,
        sqlite_store=sqlite_store,
        graph_index=graph_index,
    )
    result["cosine"] = cosine_n

    entity_n = infer_entity_edges(
        node_id=node_id,
        node_metadata=node_metadata,
        node_text=node_text,
        sqlite_store=sqlite_store,
        graph_index=graph_index,
        entity_index=entity_index,
    )
    result["entity"] = entity_n

    return result
