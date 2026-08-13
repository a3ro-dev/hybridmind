"""
Edge CRUD API endpoints for HybridMind.
"""

import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from models.edge import (
    EdgeCreate,
    EdgeUpdate,
    EdgeResponse,
    EdgeDeleteResponse,
    EdgeType
)
from api.dependencies import (
    get_sqlite_store,
    get_graph_index,
    coordinate_mutation,
)
from storage.sqlite_store import SQLiteStore
from storage.graph_index import GraphIndex
from engine.cache import invalidate_cache

router = APIRouter(prefix="/edges", tags=["Edges"])


def _rebuild_graph_from_sql(
    sqlite_store: SQLiteStore, graph_index: GraphIndex
) -> None:
    """Replace the graph projection from active authoritative SQL state."""
    nodes = sqlite_store.list_nodes(limit=1_000_000)
    live_ids = {node["id"] for node in nodes}
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


def _recover_graph_or_raise(
    sqlite_store: SQLiteStore, graph_index: GraphIndex, operation: str
) -> None:
    try:
        _rebuild_graph_from_sql(sqlite_store, graph_index)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Edge {operation} failed and graph recovery also failed",
        ) from exc


@router.post("", response_model=EdgeResponse, status_code=201)
async def create_edge(
    edge: EdgeCreate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    graph_index: GraphIndex = Depends(get_graph_index),
    mutation_guard: None = Depends(coordinate_mutation),
) -> EdgeResponse:
    """
    Create a relationship between two nodes.
    """
    # BUG-6: Prevent self-loop edges
    if edge.source_id == edge.target_id:
        raise HTTPException(
            status_code=422,
            detail="Self-loop edges are not allowed: source_id and target_id must differ."
        )

    # Validate source node exists
    source_node = sqlite_store.get_node(edge.source_id)
    if source_node is None:
        raise HTTPException(
            status_code=404,
            detail=f"Source node {edge.source_id} not found"
        )
    
    # Validate target node exists
    target_node = sqlite_store.get_node(edge.target_id)
    if target_node is None:
        raise HTTPException(
            status_code=404,
            detail=f"Target node {edge.target_id} not found"
        )
    
    # Generate edge ID
    edge_id = str(uuid.uuid4())
    
    try:
        with sqlite_store.transaction():
            result = sqlite_store.create_edge(
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.type,
                weight=edge.weight,
                metadata=edge.metadata or {},
                valid_from=edge.valid_from.isoformat() if edge.valid_from else None,
                valid_until=edge.valid_until.isoformat() if edge.valid_until else None,
                superseded_by=edge.superseded_by,
                confidence=edge.confidence,
            )
            graph_attrs = dict(edge.metadata or {})
            graph_attrs.update({
                "created_at": result.get("created_at"),
                "valid_from": result.get("valid_from"),
                "valid_until": result.get("valid_until"),
                "superseded_by": result.get("superseded_by"),
                "confidence": result.get("confidence", 1.0),
            })
            graph_index.add_edge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.type,
                weight=edge.weight,
                edge_id=edge_id,
                **graph_attrs,
            )
    except Exception as exc:
        _recover_graph_or_raise(sqlite_store, graph_index, "creation")
        raise HTTPException(
            status_code=500,
            detail="Edge creation failed; authoritative state was rolled back",
        ) from exc

    # Invalidate search cache
    invalidate_cache()

    return EdgeResponse(
        id=result["id"],
        source_id=result["source_id"],
        target_id=result["target_id"],
        type=result["type"],
        weight=result["weight"],
        metadata=result["metadata"],
        created_at=result["created_at"],
        valid_from=result.get("valid_from"),
        valid_until=result.get("valid_until"),
        superseded_by=result.get("superseded_by"),
        confidence=result.get("confidence", 1.0),
    )


@router.get("/types", response_model=List[str])
async def list_edge_types() -> List[str]:
    """
    Get list of all supported edge relationship types in the agent taxonomy.
    """
    return [e.value for e in EdgeType]


@router.get("/{edge_id}", response_model=EdgeResponse)
async def get_edge(
    edge_id: str,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> EdgeResponse:
    """
    Retrieve an edge by ID.
    """
    edge = sqlite_store.get_edge(edge_id)
    if edge is None:
        raise HTTPException(status_code=404, detail=f"Edge {edge_id} not found")
    
    return EdgeResponse(
        id=edge["id"],
        source_id=edge["source_id"],
        target_id=edge["target_id"],
        type=edge["type"],
        weight=edge["weight"],
        metadata=edge["metadata"],
        created_at=edge["created_at"],
        valid_from=edge.get("valid_from"),
        valid_until=edge.get("valid_until"),
        superseded_by=edge.get("superseded_by"),
        confidence=edge.get("confidence", 1.0),
    )


@router.put("/{edge_id}", response_model=EdgeResponse)
async def update_edge(
    edge_id: str,
    update: EdgeUpdate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    graph_index: GraphIndex = Depends(get_graph_index),
    mutation_guard: None = Depends(coordinate_mutation),
) -> EdgeResponse:
    """
    Update edge type, weight, or metadata.
    """
    # Check if edge exists
    existing = sqlite_store.get_edge(edge_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Edge {edge_id} not found")
    
    try:
        with sqlite_store.transaction():
            result = sqlite_store.update_edge(
                edge_id=edge_id,
                edge_type=update.type,
                weight=update.weight,
                metadata=update.metadata,
                valid_until=update.valid_until.isoformat() if update.valid_until else None,
                superseded_by=update.superseded_by,
                confidence=update.confidence,
            )
            # Update only this typed relation; parallel edges between the same
            # nodes represent independent facts and must remain live.
            graph_index.remove_edge_by_id(edge_id)
            graph_attrs = dict(result.get("metadata", {}))
            graph_attrs.update({
                "created_at": result.get("created_at"),
                "valid_from": result.get("valid_from"),
                "valid_until": result.get("valid_until"),
                "superseded_by": result.get("superseded_by"),
                "confidence": result.get("confidence", 1.0),
            })
            graph_index.add_edge(
                source_id=result["source_id"],
                target_id=result["target_id"],
                edge_type=result["type"],
                weight=result["weight"],
                edge_id=result["id"],
                **graph_attrs,
            )
    except Exception as exc:
        _recover_graph_or_raise(sqlite_store, graph_index, "update")
        raise HTTPException(
            status_code=500,
            detail="Edge update failed; authoritative state was rolled back",
        ) from exc

    # Invalidate search cache
    invalidate_cache()

    return EdgeResponse(
        id=result["id"],
        source_id=result["source_id"],
        target_id=result["target_id"],
        type=result["type"],
        weight=result["weight"],
        metadata=result["metadata"],
        created_at=result["created_at"],
        valid_from=result.get("valid_from"),
        valid_until=result.get("valid_until"),
        superseded_by=result.get("superseded_by"),
        confidence=result.get("confidence", 1.0),
    )


@router.delete("/{edge_id}", response_model=EdgeDeleteResponse)
async def delete_edge(
    edge_id: str,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    graph_index: GraphIndex = Depends(get_graph_index),
    mutation_guard: None = Depends(coordinate_mutation),
) -> EdgeDeleteResponse:
    """
    Delete an edge.
    """
    # Check if edge exists
    existing = sqlite_store.get_edge(edge_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Edge {edge_id} not found")
    
    try:
        with sqlite_store.transaction():
            deleted = sqlite_store.delete_edge(edge_id)
            graph_index.remove_edge_by_id(edge_id)
    except Exception as exc:
        _recover_graph_or_raise(sqlite_store, graph_index, "deletion")
        raise HTTPException(
            status_code=500,
            detail="Edge deletion failed; authoritative state was rolled back",
        ) from exc
    
    # Invalidate search cache
    invalidate_cache()
    
    return EdgeDeleteResponse(
        deleted=deleted,
        edge_id=edge_id
    )


@router.get("", response_model=List[EdgeResponse])
async def list_edges(
    source_id: Optional[str] = Query(default=None, description="Filter by source node"),
    target_id: Optional[str] = Query(default=None, description="Filter by target node"),
    edge_type: Optional[str] = Query(default=None, description="Filter by edge type"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> List[EdgeResponse]:
    """
    List edges with optional filtering.
    """
    # Get all edges and filter
    all_edges = sqlite_store.get_all_edges()
    
    filtered = []
    for edge in all_edges:
        if source_id and edge["source_id"] != source_id:
            continue
        if target_id and edge["target_id"] != target_id:
            continue
        if edge_type and edge["type"] != edge_type:
            continue
        filtered.append(edge)
    
    # Apply pagination
    paginated = filtered[skip:skip + limit]
    
    return [
        EdgeResponse(
            id=edge["id"],
            source_id=edge["source_id"],
            target_id=edge["target_id"],
            type=edge["type"],
            weight=edge["weight"],
            metadata=edge["metadata"],
            created_at=edge["created_at"],
            valid_from=edge.get("valid_from"),
            valid_until=edge.get("valid_until"),
            superseded_by=edge.get("superseded_by"),
            confidence=edge.get("confidence", 1.0),
        )
        for edge in paginated
    ]


@router.get("/node/{node_id}", response_model=List[EdgeResponse])
async def get_node_edges(
    node_id: str,
    direction: str = Query(default="both", description="'outgoing', 'incoming', or 'both'"),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> List[EdgeResponse]:
    """
    Get all edges connected to a specific node.
    """
    # Validate node exists
    node = sqlite_store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Validate direction
    if direction not in ("outgoing", "incoming", "both"):
        raise HTTPException(
            status_code=400,
            detail="direction must be 'outgoing', 'incoming', or 'both'"
        )
    
    edges = sqlite_store.get_node_edges(node_id, direction=direction)
    
    return [
        EdgeResponse(
            id=edge["id"],
            source_id=edge["source_id"],
            target_id=edge["target_id"],
            type=edge["type"],
            weight=edge["weight"],
            metadata=edge["metadata"],
            created_at=edge["created_at"],
            valid_from=edge.get("valid_from"),
            valid_until=edge.get("valid_until"),
            superseded_by=edge.get("superseded_by"),
            confidence=edge.get("confidence", 1.0),
        )
        for edge in edges
    ]

