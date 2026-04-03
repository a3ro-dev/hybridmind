"""
Node CRUD API endpoints for HybridMind.
"""

import uuid
from datetime import datetime
from typing import List, Optional
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
    get_embedding_engine
)
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.bm25_index import BM25Index
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine
from engine.cache import invalidate_cache

router = APIRouter(prefix="/nodes", tags=["Nodes"])


@router.post("", response_model=NodeResponse, status_code=201)
async def create_node(
    node: NodeCreate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    bm25_index: BM25Index = Depends(get_bm25_index),
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
    
    # Generate or use provided embedding
    import numpy as np
    from config import settings
    if node.embedding:
        raw_embedding = np.array(node.embedding, dtype=np.float32)
        embedding = raw_embedding
    else:
        raw_embedding = embedding_engine.embed(node.text)
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
    
    # Store in SQLite
    result = sqlite_store.create_node(
        node_id=node_id,
        text=node.text,
        metadata=node.metadata or {},
        embedding=embedding,
        raw_embedding=raw_embedding
    )
    
    # Add to graph
    graph_index.add_node(node_id)
    
    # Structural Edges (Priority 2)
    session_id = (node.metadata or {}).get("sessionId")
    if session_id:
        prev_node = sqlite_store.get_latest_node_by_session(session_id)
        if prev_node and prev_node["id"] != node_id:
            # Temporal 'next_turn' edge
            t_edge_id = str(uuid.uuid4())
            sqlite_store.create_edge(t_edge_id, prev_node["id"], node_id, "next_turn", 1.0)
            graph_index.add_edge(t_edge_id, prev_node["id"], node_id, "next_turn", 1.0)
            
            # Session 'same_session' edge
            s_edge_id = str(uuid.uuid4())
            sqlite_store.create_edge(s_edge_id, prev_node["id"], node_id, "same_session", 0.5)
            graph_index.add_edge(s_edge_id, prev_node["id"], node_id, "same_session", 0.5)
    
    # Chunking / SGMem Approach (Priority 3)
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', node.text) if len(s.strip()) > 5]
    if not sentences:
        sentences = [node.text]
        
    for i, sentence in enumerate(sentences):
        child_id = f"{node_id}_{i}"
        child_embedding = embedding_engine.embed(sentence)
        
        child_metadata = (node.metadata or {}).copy()
        child_metadata.update({"parent_id": node_id, "is_sentence_chunk": True})
        
        # Link child sentence to parent map
        sqlite_store.create_node(
            node_id=child_id,
            text=sentence,
            metadata=child_metadata,
            embedding=child_embedding,
            raw_embedding=child_embedding
        )
        graph_index.add_node(child_id)
        
        # Edge to parent
        c_edge_id = str(uuid.uuid4())
        sqlite_store.create_edge(c_edge_id, child_id, node_id, "belongs_to", 1.0)
        graph_index.add_edge(c_edge_id, child_id, node_id, "belongs_to", 1.0)
        
        # Index children
        vector_index.add(child_id, child_embedding)
        bm25_index.add(child_id, sentence)
    
    # Also index the parent for general macro searches
    vector_index.add(node_id, embedding)
    bm25_index.add(node_id, node.text)
    
    # Invalidate search cache
    invalidate_cache()
    
    return NodeResponse(
        id=result["id"],
        text=result["text"],
        metadata=result["metadata"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
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
    new_metadata = update.metadata if update.metadata is not None else existing["metadata"]
    
    # Regenerate embedding if requested and text changed
    new_embedding = existing["embedding"]
    new_raw_embedding = existing.get("raw_embedding")
    
    if update.regenerate_embedding and update.text is not None:
        new_raw_embedding = embedding_engine.embed(new_text)
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
    
    # Update in SQLite
    result = sqlite_store.update_node(
        node_id=node_id,
        text=new_text,
        metadata=new_metadata,
        embedding=new_embedding,
        raw_embedding=new_raw_embedding
    )
    
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
            edges=edges
        ))
    
    return results

