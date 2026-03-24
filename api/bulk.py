"""
Bulk operations API endpoints for HybridMind.
Fast batch import for nodes and edges.
Includes LLM-powered unstructured data processing.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import (
    get_sqlite_store,
    get_vector_index,
    get_graph_index,
    get_embedding_engine,
    get_db_manager,
)
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine
from engine.cache import invalidate_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bulk", tags=["Bulk Operations"])

# Default API key for OSM API
DEFAULT_OSM_API_KEY = "osm_dkrYgl3rXWDEbmVbDmdn6jKenn8m0KXm5XiMHHBk"


# ==================== Request/Response Models ====================

class BulkNodeCreate(BaseModel):
    """Single node in bulk create request."""
    id: Optional[str] = Field(default=None, description="Optional custom ID")
    text: str = Field(..., min_length=1, max_length=50000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulkNodesRequest(BaseModel):
    """Bulk node creation request."""
    nodes: List[BulkNodeCreate] = Field(..., min_length=1, max_length=1000)
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings for nodes"
    )


class BulkEdgeCreate(BaseModel):
    """Single edge in bulk create request."""
    id: Optional[str] = Field(default=None, description="Optional custom ID")
    source_id: str
    target_id: str
    type: str = Field(..., min_length=1)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulkEdgesRequest(BaseModel):
    """Bulk edge creation request."""
    edges: List[BulkEdgeCreate] = Field(..., min_length=1, max_length=5000)
    skip_validation: bool = Field(
        default=False,
        description="Skip node existence validation for faster import"
    )


class BulkResult(BaseModel):
    """Result of bulk operation."""
    success: bool
    created: int
    failed: int
    errors: List[str]
    elapsed_ms: float


class BulkImportRequest(BaseModel):
    """Combined bulk import of nodes and edges."""
    nodes: List[BulkNodeCreate] = Field(default_factory=list)
    edges: List[BulkEdgeCreate] = Field(default_factory=list)
    generate_embeddings: bool = Field(default=True)


class BulkImportResult(BaseModel):
    """Result of combined bulk import."""
    nodes: BulkResult
    edges: BulkResult
    total_elapsed_ms: float


class UnstructuredDataRequest(BaseModel):
    """Request for processing unstructured data via LLM."""
    text: str = Field(..., min_length=10, max_length=100000, description="Raw unstructured text to process")
    api_key: Optional[str] = Field(default=None, description="Optional OSM API key")
    model: str = Field(default="qwen3.5-397b-a17b", description="LLM model to use")


class UnstructuredDataResult(BaseModel):
    """Result of unstructured data processing."""
    success: bool
    summary: str
    nodes_created: int
    edges_created: int
    nodes_failed: int
    edges_failed: int
    extracted_entities: List[Dict[str, Any]]
    errors: List[str]
    elapsed_ms: float


# ==================== Endpoints ====================

@router.post("/nodes", response_model=BulkResult)
async def bulk_create_nodes(
    request: BulkNodesRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
):
    """
    Bulk create nodes with optional embedding generation.
    
    Features:
    - Batch embedding generation for efficiency
    - Batch vector index insertion
    - Automatic ID generation if not provided
    
    Maximum 1000 nodes per request.
    """
    start_time = time.perf_counter()
    created = 0
    failed = 0
    errors = []
    
    nodes_to_create = []
    
    # Prepare nodes
    for node in request.nodes:
        node_id = node.id or f"node_{uuid.uuid4().hex[:12]}"
        
        # Check if ID already exists
        if sqlite_store.get_node(node_id):
            errors.append(f"Node {node_id} already exists")
            failed += 1
            continue
        
        nodes_to_create.append({
            "id": node_id,
            "text": node.text,
            "metadata": node.metadata
        })
    
    if not nodes_to_create:
        return BulkResult(
            success=False,
            created=0,
            failed=failed,
            errors=errors,
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 2)
        )
    
    # Generate embeddings in batch
    embeddings = None
    if request.generate_embeddings:
        texts = [n["text"] for n in nodes_to_create]
        try:
            embeddings = embedding_engine.embed_batch(texts, show_progress=False)
        except Exception as e:
            errors.append(f"Embedding generation failed: {str(e)}")
            embeddings = None
    
    # Create nodes in database
    vector_batch = []
    
    for i, node_data in enumerate(nodes_to_create):
        try:
            embedding = embeddings[i] if embeddings is not None else None
            
            # Create in SQLite
            sqlite_store.create_node(
                node_id=node_data["id"],
                text=node_data["text"],
                metadata=node_data["metadata"],
                embedding=embedding
            )
            
            # Add to graph index
            graph_index.add_node(node_data["id"])
            
            # Prepare vector batch
            if embedding is not None:
                vector_batch.append((node_data["id"], embedding))
            
            created += 1
            
        except Exception as e:
            errors.append(f"Failed to create node {node_data['id']}: {str(e)}")
            failed += 1
    
    # Batch add to vector index
    if vector_batch:
        try:
            vector_index.add_batch(vector_batch)
        except Exception as e:
            errors.append(f"Vector index batch add failed: {str(e)}")
    
    # Invalidate cache
    invalidate_cache()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Bulk created {created} nodes in {elapsed:.0f}ms")
    
    return BulkResult(
        success=failed == 0,
        created=created,
        failed=failed,
        errors=errors[:10],  # Limit error messages
        elapsed_ms=round(elapsed, 2)
    )


@router.post("/edges", response_model=BulkResult)
async def bulk_create_edges(
    request: BulkEdgesRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    graph_index: GraphIndex = Depends(get_graph_index),
):
    """
    Bulk create edges between existing nodes.
    
    Features:
    - Optional node existence validation (disable for faster import)
    - Automatic ID generation if not provided
    
    Maximum 5000 edges per request.
    """
    start_time = time.perf_counter()
    created = 0
    failed = 0
    errors = []
    
    # Validate nodes exist (unless skip_validation)
    if not request.skip_validation:
        node_ids = set()
        for edge in request.edges:
            node_ids.add(edge.source_id)
            node_ids.add(edge.target_id)
        
        # Check which nodes exist
        missing = []
        for node_id in node_ids:
            if not sqlite_store.get_node(node_id):
                missing.append(node_id)
        
        if missing:
            return BulkResult(
                success=False,
                created=0,
                failed=len(request.edges),
                errors=[f"Missing nodes: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"],
                elapsed_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )
    
    # Create edges
    for edge in request.edges:
        edge_id = edge.id or f"edge_{uuid.uuid4().hex[:12]}"
        
        try:
            # Create in SQLite
            sqlite_store.create_edge(
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.type,
                weight=edge.weight,
                metadata=edge.metadata
            )
            
            # Add to graph index
            graph_index.add_edge(
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.type,
                weight=edge.weight
            )
            
            created += 1
            
        except Exception as e:
            errors.append(f"Failed to create edge {edge_id}: {str(e)}")
            failed += 1
    
    # Invalidate cache
    invalidate_cache()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Bulk created {created} edges in {elapsed:.0f}ms")
    
    return BulkResult(
        success=failed == 0,
        created=created,
        failed=failed,
        errors=errors[:10],
        elapsed_ms=round(elapsed, 2)
    )


@router.post("/import", response_model=BulkImportResult)
async def bulk_import(
    request: BulkImportRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
):
    """
    Combined bulk import of nodes and edges.
    
    Imports nodes first, then edges. Useful for loading
    complete knowledge graphs in a single request.
    """
    total_start = time.perf_counter()
    
    # Import nodes
    node_result = await bulk_create_nodes(
        BulkNodesRequest(
            nodes=request.nodes,
            generate_embeddings=request.generate_embeddings
        ),
        sqlite_store=sqlite_store,
        vector_index=vector_index,
        graph_index=graph_index,
        embedding_engine=embedding_engine,
    ) if request.nodes else BulkResult(
        success=True, created=0, failed=0, errors=[], elapsed_ms=0
    )
    
    # Import edges (skip validation since we just created the nodes)
    edge_result = await bulk_create_edges(
        BulkEdgesRequest(
            edges=request.edges,
            skip_validation=False  # Still validate in case of partial node failures
        ),
        sqlite_store=sqlite_store,
        graph_index=graph_index,
    ) if request.edges else BulkResult(
        success=True, created=0, failed=0, errors=[], elapsed_ms=0
    )
    
    total_elapsed = (time.perf_counter() - total_start) * 1000
    
    logger.info(
        f"Bulk import complete: {node_result.created} nodes, "
        f"{edge_result.created} edges in {total_elapsed:.0f}ms"
    )
    
    return BulkImportResult(
        nodes=node_result,
        edges=edge_result,
        total_elapsed_ms=round(total_elapsed, 2)
    )


@router.post("/unstructured", response_model=UnstructuredDataResult)
async def process_unstructured_data(
    request: UnstructuredDataRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
):
    """
    Process unstructured text using LLM and extract knowledge graph.
    
    Uses OSM API with Qwen3.5 397B A17B to:
    - Extract entities, concepts, and facts from raw text
    - Create structured nodes with rich metadata
    - Identify and create relationships between nodes
    
    Perfect for ingesting:
    - Wikipedia articles
    - Research papers
    - Documentation
    - Any large text content
    """
    start_time = time.perf_counter()
    errors = []
    
    # Get API key
    api_key = request.api_key or os.getenv("OSM_API_KEY") or DEFAULT_OSM_API_KEY
    
    try:
        from engine.llm import LLMEngine
        llm = LLMEngine(api_key=api_key, model=request.model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LLM engine: {str(e)}"
        )
    
    # Process unstructured data with LLM
    try:
        extracted = llm.process_unstructured(request.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM processing failed: {str(e)}"
        )
    
    summary = extracted.get("summary", "")
    raw_nodes = extracted.get("nodes", [])
    raw_edges = extracted.get("edges", [])
    
    # Create nodes
    nodes_created = 0
    nodes_failed = 0
    node_id_map = {}  # Map index to actual node ID
    
    # Prepare texts for batch embedding
    texts = [n.get("text", "")[:2000] for n in raw_nodes if n.get("text")]
    
    # Generate embeddings in batch
    embeddings = None
    if texts:
        try:
            embeddings = embedding_engine.embed_batch(texts, show_progress=False)
        except Exception as e:
            errors.append(f"Embedding generation failed: {str(e)}")
    
    # Create nodes
    for i, node_data in enumerate(raw_nodes):
        node_text = node_data.get("text", "")
        if not node_text or len(node_text) < 10:
            nodes_failed += 1
            continue
        
        node_id = f"node_{uuid.uuid4().hex[:12]}"
        node_id_map[i] = node_id
        
        metadata = node_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["source"] = "llm_extraction"
        metadata["summary_context"] = summary[:200]
        
        try:
            embedding = embeddings[i] if embeddings is not None and i < len(embeddings) else None
            
            sqlite_store.create_node(
                node_id=node_id,
                text=node_text[:5000],
                metadata=metadata,
                embedding=embedding
            )
            
            graph_index.add_node(node_id)
            
            if embedding is not None:
                vector_index.add(node_id, embedding)
            
            nodes_created += 1
            
        except Exception as e:
            errors.append(f"Failed to create node {i}: {str(e)}")
            nodes_failed += 1
    
    # Create edges
    edges_created = 0
    edges_failed = 0
    
    for edge_data in raw_edges:
        source_idx = edge_data.get("source_index")
        target_idx = edge_data.get("target_index")
        
        if source_idx is None or target_idx is None:
            edges_failed += 1
            continue
        
        source_id = node_id_map.get(source_idx)
        target_id = node_id_map.get(target_idx)
        
        if not source_id or not target_id:
            edges_failed += 1
            continue
        
        edge_id = f"edge_{uuid.uuid4().hex[:12]}"
        edge_type = edge_data.get("type", "relates_to")
        weight = edge_data.get("weight", 0.5)
        
        try:
            sqlite_store.create_edge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=float(weight) if isinstance(weight, (int, float)) else 0.5,
                metadata={"reasoning": edge_data.get("reasoning", "")}
            )
            
            graph_index.add_edge(
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=float(weight) if isinstance(weight, (int, float)) else 0.5
            )
            
            edges_created += 1
            
        except Exception as e:
            errors.append(f"Failed to create edge: {str(e)}")
            edges_failed += 1
    
    # Invalidate cache
    invalidate_cache()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    
    # Collect entity info for response
    extracted_entities = []
    for i, node_data in enumerate(raw_nodes[:20]):
        if i in node_id_map:
            extracted_entities.append({
                "node_id": node_id_map[i],
                "text_preview": node_data.get("text", "")[:100],
                "metadata": node_data.get("metadata", {})
            })
    
    logger.info(
        f"Unstructured import: {nodes_created} nodes, {edges_created} edges in {elapsed:.0f}ms"
    )
    
    return UnstructuredDataResult(
        success=nodes_failed == 0 and edges_failed == 0,
        summary=summary,
        nodes_created=nodes_created,
        edges_created=edges_created,
        nodes_failed=nodes_failed,
        edges_failed=edges_failed,
        extracted_entities=extracted_entities,
        errors=errors[:10],
        elapsed_ms=round(elapsed, 2)
    )


@router.delete("/clear", response_model=dict)
async def clear_all_data():
    """
    Clear all data from the database.
    
    **WARNING**: This permanently deletes all nodes, edges, and indexes.
    Use with caution.
    """
    start_time = time.perf_counter()
    
    try:
        db_manager = get_db_manager()
        
        # Get counts before clearing
        stats = db_manager.get_stats()
        nodes_count = stats["total_nodes"]
        edges_count = stats["total_edges"]
        
        # Clear vector index
        db_manager.vector_index.clear()
        
        # Clear graph index
        db_manager.graph_index.clear()
        
        # Clear SQLite (need to delete all records)
        with db_manager.sqlite_store._cursor() as cursor:
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM nodes")
        
        # Invalidate cache
        invalidate_cache()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "deleted_nodes": nodes_count,
            "deleted_edges": edges_count,
            "elapsed_ms": round(elapsed, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")

