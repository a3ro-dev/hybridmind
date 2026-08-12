"""
Node CRUD API endpoints for HybridMind.
"""

import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional
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
    get_visual_store
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


router = APIRouter(prefix="/nodes", tags=["Nodes"])


@router.post("", response_model=NodeResponse, status_code=201)
async def create_node(
    node: NodeCreate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    bm25_index: _AnyType = Depends(get_bm25_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine)
) -> NodeResponse:
    """
    Create a new node with text and optional embedding.
    
    If no embedding is provided, one will be generated automatically
    using the configured embedding model (all-MiniLM-L6-v2 by default).
    """
    if len(node.text) > 50000:
        raise HTTPException(status_code=422, detail="Text exceeds maximum length of 50,000 characters")

    # Generate node ID
    node_id = str(uuid.uuid4())
    metadata = dict(node.metadata or {})
    entities = node.entities or metadata.get("entities", [])
    event_time = node.event_time or metadata.get("event_time") or metadata.get("date")
    valid_from = node.valid_from or metadata.get("valid_from")
    valid_until = node.valid_until or metadata.get("valid_until")
    memory_kind = node.memory_kind or metadata.get("memory_kind")
    if entities:
        metadata["entities"] = entities
    if event_time:
        metadata["event_time"] = event_time
    if valid_from:
        metadata["valid_from"] = valid_from
    if valid_until:
        metadata["valid_until"] = valid_until
    if memory_kind:
        metadata["memory_kind"] = memory_kind
    metadata["confidence"] = node.confidence
    
    # Generate or use provided embedding
    import numpy as np
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
    
    # Store in SQLite
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
    
    # Add to graph
    graph_index.add_node(node_id, event_time=event_time, memory_kind=memory_kind)
    
    # Structural Edges (Priority 2)
    session_id = metadata.get("sessionId") or metadata.get("session_id")
    if session_id:
        prev_node = sqlite_store.get_latest_node_by_session(
            session_id, exclude_node_id=node_id
        )
        if prev_node and prev_node["id"] != node_id:
            # Temporal 'next_turn' edge
            t_edge_id = str(uuid.uuid4())
            sqlite_store.create_edge(t_edge_id, prev_node["id"], node_id, "next_turn", 1.0)
            graph_index.add_edge(
                source_id=prev_node["id"],
                target_id=node_id,
                edge_type="next_turn",
                weight=1.0,
                edge_id=t_edge_id,
            )
            
            # Session 'same_session' edge
            s_edge_id = str(uuid.uuid4())
            sqlite_store.create_edge(s_edge_id, prev_node["id"], node_id, "same_session", 0.5)
            graph_index.add_edge(
                source_id=prev_node["id"],
                target_id=node_id,
                edge_type="same_session",
                weight=0.5,
                edge_id=s_edge_id,
            )
    
    # Chunking / SGMem Approach (Priority 3)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', node.text) if len(s.strip()) > 5]
    if not sentences:
        sentences = [node.text]
        
    for i, sentence in enumerate(sentences):
        child_id = f"{node_id}_{i}"
        # Also clean sentence chunks for embedding
        child_embedding = validate_embedding_4096(
            embedding_engine.embed(_strip_metadata_prefixes(sentence)),
            label="sentence embedding",
        )
        
        child_metadata = metadata.copy()
        child_metadata.update({"parent_id": node_id, "is_sentence_chunk": True})
        
        # Link child sentence to parent map
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
        graph_index.add_node(child_id)
        
        # Edge to parent
        c_edge_id = str(uuid.uuid4())
        sqlite_store.create_edge(c_edge_id, child_id, node_id, "belongs_to", 1.0)
        graph_index.add_edge(
            source_id=child_id,
            target_id=node_id,
            edge_type="belongs_to",
            weight=1.0,
            edge_id=c_edge_id,
        )
        
        # Index children
        vector_index.add(child_id, child_embedding)
        bm25_index.add(child_id, sentence)
    
    # Also index the parent for general macro searches
    vector_index.add(node_id, embedding)
    bm25_index.add(node_id, node.text)

    # Auto-edge inference (config-gated: HYBRIDMIND_AUTO_EDGES_ENABLED=true)
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

    # ColBERT per-token vectors (opt-in: HYBRIDMIND_COLBERT_ENABLED=true)
    from storage.colbert_store import maybe_store_colbert, colbert_enabled
    if colbert_enabled():
        from api.dependencies import get_colbert_store
        cs = get_colbert_store()
        if cs is not None:
            maybe_store_colbert(node_id, node.text, embedding_engine, cs)

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
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine)
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
        effective_event_time = new_metadata.get("event_time") or new_metadata.get("date")
    effective_valid_from = update.valid_from
    if effective_valid_from is None and update.metadata is not None:
        effective_valid_from = new_metadata.get("valid_from")
    effective_valid_until = update.valid_until
    if effective_valid_until is None and update.metadata is not None:
        effective_valid_until = new_metadata.get("valid_until")
    effective_memory_kind = update.memory_kind
    if effective_memory_kind is None and update.metadata is not None:
        effective_memory_kind = new_metadata.get("memory_kind")
    effective_confidence = update.confidence
    if effective_confidence is None and update.metadata is not None:
        metadata_confidence = new_metadata.get("confidence")
        if metadata_confidence is not None:
            effective_confidence = float(metadata_confidence)
    
    # Regenerate embedding if requested and text changed
    new_embedding = existing["embedding"]
    new_raw_embedding = existing.get("raw_embedding")
    
    if update.regenerate_embedding and update.text is not None:
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
    
    # Update in SQLite
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
    
    # Update vector index if embedding changed
    if new_embedding is not None:
        vector_index.add(node_id, new_embedding)  # add() handles replacement
    
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
    graph_index: GraphIndex = Depends(get_graph_index)
) -> NodeDeleteResponse:
    """
    Delete a node and all its associated edges.
    """
    # Check if node exists
    existing = sqlite_store.get_node(node_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Soft delete from SQLite (and hard delete its edges)
    deleted, edges_removed = sqlite_store.delete_node(node_id)
    
    # Do NOT remove from FAISS yet (handled by compaction)
    
    # Remove from graph index
    graph_index.remove_node(node_id)
    
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
    visual_store: ColbertStore = Depends(get_visual_store)
) -> NodeResponse:
    """
    Ingest an image by computing its patch vectors via a remote ColQwen2.5 server
    and storing them in VisualColbertStore. Surfaced via image caption text embedding.
    """
    from engine.image_embedding import get_image_embedding_engine
    from config import settings

    img_engine = get_image_embedding_engine()
    if img_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Remote image embedding service is not configured. Set HYBRIDMIND_IMAGE_EMBEDDING_URL in .env."
        )

    # 1. Embed image patches via remote ColQwen2.5
    patch_vectors = img_engine.embed_image(node.image_b64)
    if patch_vectors is None or len(patch_vectors) == 0:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate patch vectors from remote image embedding service."
        )
    import numpy as np
    patch_vectors = np.asarray(patch_vectors, dtype=np.float32)

    # 2. Generate node ID
    node_id = str(uuid.uuid4())

    # 3. Save patch vectors to visual store
    visual_store.add(node_id, patch_vectors)

    # 4. Embed the caption text
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
                    n_emb = n.get("raw_embedding") or n.get("embedding")
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

    # 5. Build node metadata
    metadata = node.metadata or {}
    metadata["modality"] = "image"

    # 6. Store in SQLite
    result = sqlite_store.create_node(
        node_id=node_id,
        text=node.caption,
        metadata=metadata,
        embedding=caption_emb,
        raw_embedding=caption_raw_emb
    )

    # 7. Add to vector index and graph index
    vector_index.add(node_id, caption_emb)
    bm25_index.add(node_id, node.caption)
    graph_index.add_node(node_id)

    # 8. Auto-edge inference
    run_auto_edge_inference(
        node_id=node_id,
        embedding=caption_emb,
        node_metadata=metadata,
        node_text=node.caption,
        vector_index=vector_index,
        sqlite_store=sqlite_store,
        graph_index=graph_index,
    )

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

