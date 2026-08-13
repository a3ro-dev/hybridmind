"""
Node CRUD API endpoints for HybridMind.
"""

import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from models.node import (
    NodeCreate,
    NodeUpdate,
    NodeResponse,
    NodeDeleteResponse,
    EdgeSummary
)
from api.dependencies import (
    get_sqlite_store,
    get_vector_index,
    get_bm25_index,
    get_graph_index,
    get_embedding_engine,
    get_visual_store,
    coordinate_mutation,
)
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from typing import Any as _AnyType
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine, validate_embedding_4096
from engine.cache import invalidate_cache
from engine.edge_inference import run_auto_edge_inference

# ── Text cleaning for embedding quality ──────────────────────────────────────

_METADATA_PREFIX_RE = re.compile(
    r'^(\[DATE:[^\]]*\]\s*)?(\[SPEAKER:[^\]]*\]\s*)?'
)


def _strip_metadata_prefixes(text: str) -> str:
    """
    Strip noisy [DATE: ...] and [SPEAKER: ...] prefixes before embedding.
    The raw text (with prefixes) is preserved in storage for BM25 and display;
    only the embedding vector is computed from the cleaned text for better
    semantic signal.
    """
    return _METADATA_PREFIX_RE.sub('', text, count=1).strip()


def _sentence_chunks(text: str) -> List[str]:
    """Return useful sub-sentence retrieval units without duplicating the parent."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) > 5
    ]
    return sentences if len(sentences) > 1 else []


def _embed_sentence_chunks(
    sentences: List[str], embedding_engine: EmbeddingEngine
) -> List[np.ndarray]:
    """Embed all derived chunks before any persistent mutation can occur."""
    if not sentences:
        return []
    values = np.asarray(
        embedding_engine.embed_batch(
            [_strip_metadata_prefixes(sentence) for sentence in sentences]
        ),
        dtype=np.float32,
    )
    if values.shape != (len(sentences), 4096):
        raise ValueError(
            f"sentence embedding batch returned {values.shape}; "
            f"expected ({len(sentences)}, 4096)"
        )
    return [
        validate_embedding_4096(value, label=f"sentence embedding {index}")
        for index, value in enumerate(values)
    ]


def _rebuild_primary_indexes(
    sqlite_store: SQLiteStore,
    vector_index: VectorIndex,
    bm25_index: _AnyType,
    graph_index: GraphIndex,
) -> None:
    """Restore every primary derived index from authoritative live SQL rows."""
    nodes = sqlite_store.list_nodes(limit=1_000_000)
    live_ids = {node["id"] for node in nodes}
    vector_index.rebuild_from_embeddings(sqlite_store.get_all_node_embeddings())
    edges = [
        edge
        for edge in sqlite_store.get_all_edges()
        if edge["source_id"] in live_ids and edge["target_id"] in live_ids
    ]
    graph_index.rebuild_from_edges(edges)
    for node in nodes:
        if not graph_index.has_node(node["id"]):
            graph_index.add_node(
                node["id"],
                event_time=node.get("event_time"),
                memory_kind=node.get("memory_kind"),
                confidence=node.get("confidence", 1.0),
            )
    if hasattr(bm25_index, "rebuild_from_nodes"):
        bm25_index.rebuild_from_nodes(nodes)
    else:
        bm25_index.clear()
        bm25_index.add_batch([(node["id"], node["text"]) for node in nodes])


router = APIRouter(prefix="/nodes", tags=["Nodes"])


@router.post("", response_model=NodeResponse, status_code=201)
async def create_node(
    node: NodeCreate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    bm25_index: _AnyType = Depends(get_bm25_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    mutation_guard: None = Depends(coordinate_mutation),
) -> NodeResponse:
    """
    Create a new node with text and optional embedding.
    
    If no embedding is provided, one will be generated automatically
    using the configured exact-width remote embedding backend.
    """
    if len(node.text) > 50000:
        raise HTTPException(status_code=422, detail="Text exceeds maximum length of 50,000 characters")

    # The optional late-interaction store is a separate persistence engine. It
    # is rejected before any provider work or mutation until this endpoint can
    # include it in the same compensation protocol as the primary indexes.
    from storage.colbert_store import colbert_enabled
    if colbert_enabled():
        raise HTTPException(
            status_code=409,
            detail="ColBERT ingestion is not transactionally supported by this endpoint",
        )

    # Generate node ID
    node_id = str(uuid.uuid4())
    metadata = dict(node.metadata or {})
    entities = node.entities or metadata.get("entities", [])
    event_time = (
        node.event_time
        or metadata.get("event_time")
        or metadata.get("date")
        or metadata.get("timestamp")
    )
    valid_from = node.valid_from or metadata.get("valid_from")
    valid_until = node.valid_until or metadata.get("valid_until")
    try:
        event_time, valid_from, valid_until = SQLiteStore.normalize_temporal_fields(
            event_time=event_time,
            valid_from=valid_from,
            valid_until=valid_until,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    memory_kind = node.memory_kind or metadata.get("memory_kind")
    if entities:
        metadata["entities"] = entities
    if event_time:
        metadata["event_time"] = event_time
        if metadata.get("date"):
            metadata["date"] = event_time
        if metadata.get("timestamp"):
            metadata["timestamp"] = event_time
    if valid_from:
        metadata["valid_from"] = valid_from
    if valid_until:
        metadata["valid_until"] = valid_until
    if memory_kind:
        metadata["memory_kind"] = memory_kind
    metadata["confidence"] = node.confidence
    
    # Generate or use provided embedding
    from config import settings
    # Strip noisy metadata prefixes before embedding for cleaner semantic signal
    embed_text = _strip_metadata_prefixes(node.text)

    if node.embedding:
        raw_embedding = validate_embedding_4096(node.embedding, label="provided node embedding")
        embedding = raw_embedding
    else:
        raw_embedding = validate_embedding_4096(
            embedding_engine.embed(embed_text), label="generated node embedding"
        )
        embedding = raw_embedding
        
        if getattr(settings, "use_graph_conditioned_embeddings", False):
            # Query vector index for top-5 semantically similar existing nodes
            results = vector_index.search(raw_embedding, top_k=5)
            if results:
                neighbor_embeddings = []
                for sim_node_id, score in results:
                    n = sqlite_store.get_node(sim_node_id)
                    if n:
                        n_emb = n.get("raw_embedding")
                        if n_emb is None:
                            n_emb = n.get("embedding")
                        if n_emb is not None:
                            neighbor_embeddings.append(n_emb)
                            
                if neighbor_embeddings:
                    embedding = embedding_engine.embed_with_graph_context(
                        node.text,
                        neighbor_embeddings,
                        alpha=0.7
                    )
                    embedding = validate_embedding_4096(
                        embedding, label="graph-conditioned node embedding"
                    )
    
    sentences = _sentence_chunks(node.text)
    try:
        child_embeddings = _embed_sentence_chunks(sentences, embedding_engine)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Exact 4096-dimensional sentence embedding failed; node was not stored",
        ) from exc

    # Mutate authoritative SQL and all primary indexes as one compensatable
    # unit. SQL rolls back on any failure; indexes are then rebuilt from it.
    try:
        with sqlite_store.transaction():
            result = sqlite_store.create_node(
                node_id=node_id,
                text=node.text,
                metadata=metadata,
                embedding=embedding,
                raw_embedding=raw_embedding,
                event_time=event_time,
                valid_from=valid_from,
                valid_until=valid_until,
                memory_kind=memory_kind,
                confidence=node.confidence,
            )
            graph_index.add_node(
                node_id,
                event_time=event_time,
                memory_kind=memory_kind,
                confidence=node.confidence,
            )

            session_id = metadata.get("sessionId") or metadata.get("session_id")
            if session_id:
                container_tag = metadata.get("containerTag") or metadata.get("container_tag")
                turn_index = metadata.get("turn_index")
                try:
                    turn_index = int(turn_index) if turn_index is not None else None
                except (TypeError, ValueError) as exc:
                    raise ValueError("turn_index must be an integer") from exc
                prev_node = sqlite_store.get_latest_node_by_session(
                    session_id,
                    exclude_node_id=node_id,
                    container_tag=container_tag,
                    before_turn_index=turn_index,
                )
                if prev_node and prev_node["id"] != node_id:
                    for edge_type, weight in (("next_turn", 1.0), ("same_session", 0.5)):
                        edge_id = str(uuid.uuid4())
                        sqlite_store.create_edge(
                            edge_id, prev_node["id"], node_id, edge_type, weight
                        )
                        graph_index.add_edge(
                            source_id=prev_node["id"],
                            target_id=node_id,
                            edge_type=edge_type,
                            weight=weight,
                            edge_id=edge_id,
                        )

            for i, (sentence, child_embedding) in enumerate(
                zip(sentences, child_embeddings)
            ):
                child_id = f"{node_id}_{i}"
                child_metadata = metadata.copy()
                child_metadata.update(
                    {"parent_id": node_id, "is_sentence_chunk": True, "chunk_index": i}
                )
                sqlite_store.create_node(
                    node_id=child_id,
                    text=sentence,
                    metadata=child_metadata,
                    embedding=child_embedding,
                    raw_embedding=child_embedding,
                    event_time=event_time,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    memory_kind=memory_kind,
                    confidence=node.confidence,
                )
                graph_index.add_node(
                    child_id,
                    event_time=event_time,
                    memory_kind=memory_kind,
                    confidence=node.confidence,
                )
                edge_id = str(uuid.uuid4())
                sqlite_store.create_edge(
                    edge_id, child_id, node_id, "belongs_to", 1.0
                )
                graph_index.add_edge(
                    source_id=child_id,
                    target_id=node_id,
                    edge_type="belongs_to",
                    weight=1.0,
                    edge_id=edge_id,
                )
                vector_index.add(child_id, child_embedding)
                bm25_index.add(child_id, sentence)

            vector_index.add(node_id, embedding)
            bm25_index.add(node_id, node.text)
            run_auto_edge_inference(
                node_id=node_id,
                embedding=embedding,
                node_metadata=metadata,
                node_text=node.text,
                vector_index=vector_index,
                sqlite_store=sqlite_store,
                graph_index=graph_index,
                event_time=event_time,
            )
    except Exception as exc:
        try:
            _rebuild_primary_indexes(
                sqlite_store, vector_index, bm25_index, graph_index
            )
        except Exception as rebuild_exc:
            raise HTTPException(
                status_code=500,
                detail="Node creation failed and index recovery also failed",
            ) from rebuild_exc
        raise HTTPException(
            status_code=500,
            detail="Node creation failed; no authoritative row was committed",
        ) from exc

    # Invalidate search cache
    invalidate_cache()
    
    return NodeResponse(
        id=result["id"],
        text=result["text"],
        metadata=result["metadata"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
        modality=result["metadata"].get("modality", "text"),
        event_time=result.get("event_time"),
        valid_from=result.get("valid_from"),
        valid_until=result.get("valid_until"),
        memory_kind=result.get("memory_kind"),
        confidence=result.get("confidence", 1.0),
        edges=[]
    )


@router.get("/{node_id}", response_model=NodeResponse)
async def get_node(
    node_id: str,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> NodeResponse:
    """
    Retrieve a node by ID with its relationships.
    """
    node = sqlite_store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Get connected edges
    edges_data = sqlite_store.get_node_edges(node_id)
    edges = []
    for edge in edges_data:
        # Determine target node
        if edge["source_id"] == node_id:
            target_id = edge["target_id"]
            direction = "outgoing"
        else:
            target_id = edge["source_id"]
            direction = "incoming"
        
        edges.append(EdgeSummary(
            edge_id=edge["id"],
            target_id=target_id,
            type=edge["type"],
            weight=edge["weight"],
            direction=direction
        ))
    
    return NodeResponse(
        id=node["id"],
        text=node["text"],
        metadata=node["metadata"],
        created_at=node["created_at"],
        updated_at=node["updated_at"],
        modality=node["metadata"].get("modality", "text"),
        event_time=node.get("event_time"),
        valid_from=node.get("valid_from"),
        valid_until=node.get("valid_until"),
        memory_kind=node.get("memory_kind"),
        confidence=node.get("confidence", 1.0),
        access_count=node.get("access_count", 0),
        last_accessed_at=node.get("last_accessed_at"),
        archived_at=node.get("archived_at"),
        archived_by=node.get("archived_by"),
        edges=edges
    )


@router.put("/{node_id}", response_model=NodeResponse)
async def update_node(
    node_id: str,
    update: NodeUpdate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    bm25_index: _AnyType = Depends(get_bm25_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    mutation_guard: None = Depends(coordinate_mutation),
) -> NodeResponse:
    """
    Update node content and optionally regenerate embedding.
    """
    if update.text is not None and len(update.text) > 50000:
        raise HTTPException(status_code=422, detail="Text exceeds maximum length of 50,000 characters")

    # Check if node exists
    existing = sqlite_store.get_node(node_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Prepare update values
    new_text = update.text if update.text is not None else existing["text"]
    new_metadata = dict(update.metadata if update.metadata is not None else existing["metadata"])
    if update.entities is not None:
        new_metadata["entities"] = update.entities
    for key, value in (
        ("event_time", update.event_time),
        ("valid_from", update.valid_from),
        ("valid_until", update.valid_until),
        ("memory_kind", update.memory_kind),
        ("confidence", update.confidence),
    ):
        if value is not None:
            new_metadata[key] = value

    # Structured values may arrive either through the first-class request
    # fields or through metadata. Keep SQLite's normalized columns and entity
    # table synchronized with the representation returned by the API.
    effective_entities = (
        update.entities
        if update.entities is not None
        else (new_metadata.get("entities") if update.metadata is not None else None)
    )
    effective_event_time = update.event_time
    if effective_event_time is None and update.metadata is not None:
        effective_event_time = (
            new_metadata.get("event_time")
            or new_metadata.get("date")
            or new_metadata.get("timestamp")
        )
    effective_valid_from = update.valid_from
    if effective_valid_from is None and update.metadata is not None:
        effective_valid_from = new_metadata.get("valid_from")
    effective_valid_until = update.valid_until
    if effective_valid_until is None and update.metadata is not None:
        effective_valid_until = new_metadata.get("valid_until")
    event_time_supplied = effective_event_time is not None
    valid_from_supplied = effective_valid_from is not None
    valid_until_supplied = effective_valid_until is not None
    try:
        normalized_event, normalized_from, normalized_until = (
            SQLiteStore.normalize_temporal_fields(
                event_time=(
                    effective_event_time
                    if effective_event_time is not None
                    else existing.get("event_time")
                ),
                valid_from=(
                    effective_valid_from
                    if effective_valid_from is not None
                    else existing.get("valid_from")
                ),
                valid_until=(
                    effective_valid_until
                    if effective_valid_until is not None
                    else existing.get("valid_until")
                ),
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if event_time_supplied:
        effective_event_time = normalized_event or existing.get("event_time")
        new_metadata["event_time"] = effective_event_time
        if new_metadata.get("date"):
            new_metadata["date"] = effective_event_time
        if new_metadata.get("timestamp"):
            new_metadata["timestamp"] = effective_event_time
    if valid_from_supplied:
        effective_valid_from = normalized_from or existing.get("valid_from")
        new_metadata["valid_from"] = effective_valid_from
    if valid_until_supplied:
        effective_valid_until = normalized_until or existing.get("valid_until")
        new_metadata["valid_until"] = effective_valid_until
    effective_memory_kind = update.memory_kind
    if effective_memory_kind is None and update.metadata is not None:
        effective_memory_kind = new_metadata.get("memory_kind")
    effective_confidence = update.confidence
    if effective_confidence is None and update.metadata is not None:
        metadata_confidence = new_metadata.get("confidence")
        if metadata_confidence is not None:
            effective_confidence = float(metadata_confidence)
    
    text_changed = update.text is not None and update.text != existing["text"]

    # Text and its embedding are one consistency unit. A caller cannot opt into
    # persisting changed text under a stale vector.
    new_embedding = existing["embedding"]
    new_raw_embedding = existing.get("raw_embedding")

    if text_changed or update.regenerate_embedding:
        new_raw_embedding = validate_embedding_4096(
            embedding_engine.embed(new_text), label="updated node embedding"
        )
        new_embedding = new_raw_embedding
        
        from config import settings
        if getattr(settings, "use_graph_conditioned_embeddings", False):
            results = vector_index.search(new_raw_embedding, top_k=6)
            # exclude self
            results = [(n_id, s) for n_id, s in results if n_id != node_id][:5]
            if results:
                neighbor_embeddings = []
                for sim_node_id, score in results:
                    n = sqlite_store.get_node(sim_node_id)
                    if n:
                        n_emb = n.get("raw_embedding")
                        if n_emb is None:
                            n_emb = n.get("embedding")
                        if n_emb is not None:
                            neighbor_embeddings.append(n_emb)
                            
                if neighbor_embeddings:
                    new_embedding = embedding_engine.embed_with_graph_context(
                        new_text,
                        neighbor_embeddings,
                        alpha=0.7
                    )
                    new_embedding = validate_embedding_4096(
                        new_embedding, label="updated graph-conditioned embedding"
                    )

    new_sentences = _sentence_chunks(new_text) if text_changed else []
    try:
        new_child_embeddings = (
            _embed_sentence_chunks(new_sentences, embedding_engine) if text_changed else []
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Exact 4096-dimensional sentence embedding failed; node was not updated",
        ) from exc

    try:
        with sqlite_store.transaction():
            result = sqlite_store.update_node(
                node_id=node_id,
                text=new_text,
                metadata=new_metadata,
                embedding=new_embedding,
                raw_embedding=new_raw_embedding,
                event_time=effective_event_time,
                valid_from=effective_valid_from,
                valid_until=effective_valid_until,
                memory_kind=effective_memory_kind,
                confidence=effective_confidence,
            )
            if effective_entities is not None:
                sqlite_store.upsert_node_entities(node_id, effective_entities)

            if new_embedding is not None:
                vector_index.add(node_id, new_embedding)
            if text_changed:
                bm25_index.add(node_id, new_text)
            graph_index.add_node(
                node_id,
                event_time=result.get("event_time"),
                memory_kind=result.get("memory_kind"),
                confidence=result.get("confidence", 1.0),
            )

            if text_changed:
                old_child_ids, _ = sqlite_store.delete_sentence_children(node_id)
                for child_id in old_child_ids:
                    vector_index.remove(child_id)
                    bm25_index.remove(child_id)
                    graph_index.remove_node(child_id)

                for index, (sentence, child_embedding) in enumerate(
                    zip(new_sentences, new_child_embeddings)
                ):
                    child_id = f"{node_id}_{index}"
                    child_metadata = dict(new_metadata)
                    child_metadata.update(
                        {
                            "parent_id": node_id,
                            "is_sentence_chunk": True,
                            "chunk_index": index,
                        }
                    )
                    sqlite_store.create_node(
                        node_id=child_id,
                        text=sentence,
                        metadata=child_metadata,
                        embedding=child_embedding,
                        raw_embedding=child_embedding,
                        event_time=result.get("event_time"),
                        valid_from=result.get("valid_from"),
                        valid_until=result.get("valid_until"),
                        memory_kind=result.get("memory_kind"),
                        confidence=result.get("confidence", 1.0),
                    )
                    graph_index.add_node(
                        child_id,
                        event_time=result.get("event_time"),
                        memory_kind=result.get("memory_kind"),
                        confidence=result.get("confidence", 1.0),
                    )
                    edge_id = str(uuid.uuid4())
                    sqlite_store.create_edge(
                        edge_id, child_id, node_id, "belongs_to", 1.0
                    )
                    graph_index.add_edge(
                        source_id=child_id,
                        target_id=node_id,
                        edge_type="belongs_to",
                        weight=1.0,
                        edge_id=edge_id,
                    )
                    vector_index.add(child_id, child_embedding)
                    bm25_index.add(child_id, sentence)
            elif any(
                value is not None
                for value in (
                    update.metadata,
                    update.entities,
                    update.event_time,
                    update.valid_from,
                    update.valid_until,
                    update.memory_kind,
                    update.confidence,
                )
            ):
                for index, child in enumerate(
                    sqlite_store.get_sentence_children(node_id)
                ):
                    child_metadata = dict(new_metadata)
                    child_metadata.update(
                        {
                            "parent_id": node_id,
                            "is_sentence_chunk": True,
                            "chunk_index": index,
                        }
                    )
                    sqlite_store.update_node(
                        child["id"],
                        metadata=child_metadata,
                        embedding=child["embedding"],
                        raw_embedding=child["raw_embedding"],
                        event_time=result.get("event_time"),
                        valid_from=result.get("valid_from"),
                        valid_until=result.get("valid_until"),
                        memory_kind=result.get("memory_kind"),
                        confidence=result.get("confidence", 1.0),
                    )
                    graph_index.add_node(
                        child["id"],
                        event_time=result.get("event_time"),
                        memory_kind=result.get("memory_kind"),
                        confidence=result.get("confidence", 1.0),
                    )
    except Exception as exc:
        try:
            _rebuild_primary_indexes(
                sqlite_store, vector_index, bm25_index, graph_index
            )
        except Exception as rebuild_exc:
            raise HTTPException(
                status_code=500,
                detail="Node update failed and index recovery also failed",
            ) from rebuild_exc
        raise HTTPException(
            status_code=500,
            detail="Node update failed; authoritative state was rolled back",
        ) from exc

    # Invalidate search cache
    invalidate_cache()
    
    # Get edges for response
    edges_data = sqlite_store.get_node_edges(node_id)
    edges = []
    for edge in edges_data:
        if edge["source_id"] == node_id:
            target_id = edge["target_id"]
            direction = "outgoing"
        else:
            target_id = edge["source_id"]
            direction = "incoming"
        
        edges.append(EdgeSummary(
            edge_id=edge["id"],
            target_id=target_id,
            type=edge["type"],
            weight=edge["weight"],
            direction=direction
        ))
    
    return NodeResponse(
        id=result["id"],
        text=result["text"],
        metadata=result["metadata"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
        modality=result["metadata"].get("modality", "text"),
        event_time=result.get("event_time"),
        valid_from=result.get("valid_from"),
        valid_until=result.get("valid_until"),
        memory_kind=result.get("memory_kind"),
        confidence=result.get("confidence", 1.0),
        edges=edges
    )


@router.delete("/{node_id}", response_model=NodeDeleteResponse)
async def delete_node(
    node_id: str,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    bm25_index: _AnyType = Depends(get_bm25_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    mutation_guard: None = Depends(coordinate_mutation),
) -> NodeDeleteResponse:
    """
    Delete a node and all its associated edges.
    """
    # Check if node exists
    existing = sqlite_store.get_node(node_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    try:
        with sqlite_store.transaction():
            # DELETE /nodes is the SDK's `forget` operation and therefore an
            # erasure boundary. Lifecycle pruning uses soft_delete_node instead.
            deleted_ids, edges_removed = sqlite_store.erase_node_family(node_id)
            deleted = node_id in deleted_ids
            for deleted_id in deleted_ids:
                vector_index.remove(deleted_id)
                bm25_index.remove(deleted_id)
                graph_index.remove_node(deleted_id)
    except Exception as exc:
        try:
            _rebuild_primary_indexes(
                sqlite_store, vector_index, bm25_index, graph_index
            )
        except Exception as rebuild_exc:
            raise HTTPException(
                status_code=500,
                detail="Node deletion failed and index recovery also failed",
            ) from rebuild_exc
        raise HTTPException(
            status_code=500,
            detail="Node deletion failed; authoritative state was rolled back",
        ) from exc
    
    # Invalidate search cache
    invalidate_cache()
    
    return NodeDeleteResponse(
        deleted=deleted,
        node_id=node_id,
        edges_removed=edges_removed
    )


@router.get("", response_model=List[NodeResponse])
async def list_nodes(
    skip: int = Query(default=0, ge=0, description="Number of nodes to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum nodes to return"),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> List[NodeResponse]:
    """
    List all nodes with pagination.
    """
    nodes = sqlite_store.list_nodes(skip=skip, limit=limit)
    
    results = []
    for node in nodes:
        # Get edges for each node
        edges_data = sqlite_store.get_node_edges(node["id"])
        edges = []
        for edge in edges_data:
            if edge["source_id"] == node["id"]:
                target_id = edge["target_id"]
                direction = "outgoing"
            else:
                target_id = edge["source_id"]
                direction = "incoming"
            
            edges.append(EdgeSummary(
                edge_id=edge["id"],
                target_id=target_id,
                type=edge["type"],
                weight=edge["weight"],
                direction=direction
            ))
        
        results.append(NodeResponse(
            id=node["id"],
            text=node["text"],
            metadata=node["metadata"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
            modality=node["metadata"].get("modality", "text"),
            event_time=node.get("event_time"),
            valid_from=node.get("valid_from"),
            valid_until=node.get("valid_until"),
            memory_kind=node.get("memory_kind"),
            confidence=node.get("confidence", 1.0),
            access_count=node.get("access_count", 0),
            last_accessed_at=node.get("last_accessed_at"),
            archived_at=node.get("archived_at"),
            archived_by=node.get("archived_by"),
            edges=edges
        ))
    
    return results


from pydantic import BaseModel
from storage.colbert_store import ColbertStore

class ImageNodeCreate(BaseModel):
    image_b64: str
    caption: str
    metadata: Optional[Dict[str, _AnyType]] = None


@router.post("/image", response_model=NodeResponse, status_code=201)
async def create_image_node(
    node: ImageNodeCreate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    bm25_index: _AnyType = Depends(get_bm25_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    visual_store: ColbertStore = Depends(get_visual_store),
    mutation_guard: None = Depends(coordinate_mutation),
) -> NodeResponse:
    """
    Ingest an image by computing its patch vectors via a remote ColQwen2.5 server
    and storing them in VisualColbertStore. Surfaced via image caption text embedding.
    """
    from engine.image_embedding import get_image_embedding_engine
    from config import settings

    if not node.caption.strip():
        raise HTTPException(status_code=422, detail="Image caption must not be empty")
    if len(node.caption) > settings.image_ingest_max_caption_chars:
        raise HTTPException(status_code=422, detail="Image caption exceeds the configured limit")
    if len(node.image_b64) > settings.image_ingest_max_base64_chars:
        raise HTTPException(status_code=413, detail="Encoded image exceeds the configured limit")

    img_engine = get_image_embedding_engine()
    if img_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Remote image embedding service is not configured",
        )

    # Complete and validate all provider work before mutating any store.
    patch_vectors = img_engine.embed_image(node.image_b64)
    if patch_vectors is None:
        raise HTTPException(
            status_code=503,
            detail="Remote image embedding failed",
        )
    patch_vectors = np.asarray(patch_vectors, dtype=np.float32)
    if (
        patch_vectors.ndim != 2
        or patch_vectors.shape[0] < 1
        or patch_vectors.shape[1] < 1
        or patch_vectors.shape[0] > settings.image_ingest_max_patch_vectors
        or patch_vectors.shape[1] > settings.image_ingest_max_patch_dimension
        or patch_vectors.nbytes > settings.image_ingest_max_patch_bytes
        or not np.all(np.isfinite(patch_vectors))
    ):
        raise HTTPException(
            status_code=502,
            detail="Remote image embedding returned an invalid or excessive matrix",
        )

    node_id = str(uuid.uuid4())
    caption_raw_emb = validate_embedding_4096(
        embedding_engine.embed(node.caption), label="image caption embedding"
    )
    caption_emb = caption_raw_emb

    if getattr(settings, "use_graph_conditioned_embeddings", False):
        results = vector_index.search(caption_raw_emb, top_k=5)
        if results:
            neighbor_embeddings = []
            for sim_node_id, score in results:
                n = sqlite_store.get_node(sim_node_id)
                if n:
                    n_emb = n.get("raw_embedding")
                    if n_emb is None:
                        n_emb = n.get("embedding")
                    if n_emb is not None:
                        neighbor_embeddings.append(n_emb)
            if neighbor_embeddings:
                caption_emb = embedding_engine.embed_with_graph_context(
                    node.caption,
                    neighbor_embeddings,
                    alpha=0.7
                )
                caption_emb = validate_embedding_4096(
                    caption_emb, label="graph-conditioned image caption embedding"
                )

    metadata = dict(node.metadata or {})
    metadata["modality"] = "image"
    visual_written = False
    try:
        with sqlite_store.transaction():
            result = sqlite_store.create_node(
                node_id=node_id,
                text=node.caption,
                metadata=metadata,
                embedding=caption_emb,
                raw_embedding=caption_raw_emb,
            )
            vector_index.add(node_id, caption_emb)
            bm25_index.add(node_id, node.caption)
            graph_index.add_node(node_id, modality="image")
            run_auto_edge_inference(
                node_id=node_id,
                embedding=caption_emb,
                node_metadata=metadata,
                node_text=node.caption,
                vector_index=vector_index,
                sqlite_store=sqlite_store,
                graph_index=graph_index,
            )
            visual_store.add(node_id, patch_vectors)
            visual_written = True
    except Exception as exc:
        cleanup_failed = False
        if visual_written or visual_store.has(node_id):
            try:
                visual_store.remove(node_id)
            except Exception:
                cleanup_failed = True
        try:
            _rebuild_primary_indexes(
                sqlite_store, vector_index, bm25_index, graph_index
            )
        except Exception as rebuild_exc:
            raise HTTPException(
                status_code=500,
                detail="Image ingestion failed and primary index recovery also failed",
            ) from rebuild_exc
        if cleanup_failed:
            raise HTTPException(
                status_code=500,
                detail="Image ingestion failed and visual cleanup did not complete",
            ) from exc
        raise HTTPException(
            status_code=500,
            detail="Image ingestion failed; no authoritative row was committed",
        ) from exc

    # Invalidate search cache
    invalidate_cache()

    return NodeResponse(
        id=node_id,
        text=node.caption,
        metadata=metadata,
        created_at=result["created_at"],
        updated_at=result["updated_at"],
        modality="image",
        edges=[]
    )

