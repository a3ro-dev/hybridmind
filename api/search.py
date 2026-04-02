"""
Search API endpoints for HybridMind.
Vector, Graph, and Hybrid search operations.

Features:
- Vector similarity search (semantic)
- Graph traversal search (relational)
- Hybrid search with CRS algorithm
- Query result caching for performance
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from models.search import (
    VectorSearchRequest,
    GraphSearchRequest,
    HybridSearchRequest,
    SearchResult,
    SearchResponse,
    StatsResponse
)
from api.dependencies import (
    get_vector_engine,
    get_graph_engine,
    get_hybrid_ranker,
    get_db_manager,
    get_sqlite_store
)
from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine
from engine.hybrid_ranker import HybridRanker
from engine.cache import get_query_cache
from storage.sqlite_store import SQLiteStore

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/vector", response_model=SearchResponse)
async def vector_search(
    request: VectorSearchRequest,
    vector_engine: VectorSearchEngine = Depends(get_vector_engine)
) -> SearchResponse:
    """
    Pure vector similarity search using cosine similarity.
    
    Returns nodes ranked by semantic similarity to the query text.
    Uses the configured embedding model (all-MiniLM-L6-v2 by default).
    
    Results are cached for 5 minutes for faster repeated queries.
    """
    # Check cache first
    cache = get_query_cache()
    cache_params = {
        "query_text": request.query_text,
        "top_k": request.top_k,
        "min_score": request.min_score,
        "filter_metadata": request.filter_metadata
    }
    
    cached = cache.get("vector", cache_params)
    if cached:
        return SearchResponse(**cached)
    
    # Execute search
    results, query_time_ms, total_candidates = vector_engine.search(
        query_text=request.query_text,
        top_k=request.top_k,
        min_score=request.min_score,
        filter_metadata=request.filter_metadata
    )
    
    search_results = [
        SearchResult(
            node_id=r["node_id"],
            text=r["text"],
            metadata=r["metadata"],
            vector_score=r["vector_score"],
            reasoning=r["reasoning"]
        )
        for r in results
    ]
    
    response = SearchResponse(
        results=search_results,
        query_time_ms=query_time_ms,
        total_candidates=total_candidates,
        search_type="vector"
    )
    
    # Cache the result
    cache.set("vector", cache_params, response.model_dump())
    
    return response


@router.get("/graph", response_model=SearchResponse)
async def graph_search(
    start_id: str = Query(..., description="Starting node ID"),
    depth: int = Query(default=2, ge=1, le=5, description="Maximum traversal depth"),
    edge_types: Optional[List[str]] = Query(default=None, description="Filter by edge types"),
    direction: str = Query(default="both", description="'outgoing', 'incoming', or 'both'"),
    graph_engine: GraphSearchEngine = Depends(get_graph_engine),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> SearchResponse:
    """
    Graph traversal search from a starting node.
    
    Returns nodes reachable within the specified depth,
    ranked by graph proximity (closer nodes first).
    """
    # Validate start node exists
    start_node = sqlite_store.get_node(start_id)
    if start_node is None:
        raise HTTPException(status_code=404, detail=f"Start node {start_id} not found")
    
    results, query_time_ms, total_candidates = graph_engine.traverse(
        start_id=start_id,
        depth=depth,
        edge_types=edge_types,
        direction=direction
    )
    
    search_results = [
        SearchResult(
            node_id=r["node_id"],
            text=r["text"],
            metadata=r["metadata"],
            graph_score=r["graph_score"],
            depth=r["depth"],
            path=r["path"],
            reasoning=r["reasoning"]
        )
        for r in results
    ]
    
    return SearchResponse(
        results=search_results,
        query_time_ms=query_time_ms,
        total_candidates=total_candidates,
        search_type="graph"
    )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    hybrid_ranker: HybridRanker = Depends(get_hybrid_ranker)
) -> SearchResponse:
    """
    Hybrid vector + graph search using the CRS algorithm.
    
    Combines semantic similarity (vector) with graph proximity
    using configurable weights:
    
    **CRS = α * vector_score + β * graph_score**
    
    Where α = vector_weight and β = graph_weight.
    
    If anchor_nodes are provided, graph scores are computed relative
    to those nodes. Otherwise, the top vector results are used as anchors.
    
    Results are cached for 5 minutes for faster repeated queries.
    """
    # Check cache first
    cache = get_query_cache()
    cache_params = {
        "query_text": request.query_text,
        "top_k": request.top_k,
        "vector_weight": request.vector_weight,
        "graph_weight": request.graph_weight,
        "anchor_nodes": request.anchor_nodes,
        "max_depth": request.max_depth,
        "min_score": request.min_score,
        "filter_metadata": request.filter_metadata,
        "deduplicate": request.deduplicate
    }
    
    cached = cache.get("hybrid", cache_params)
    if cached:
        # Return cached result with cache indicator
        cached_response = SearchResponse(**cached)
        return cached_response
    
    # Execute search
    results, query_time_ms, total_candidates = hybrid_ranker.search(
        query_text=request.query_text,
        top_k=request.top_k,
        vector_weight=request.vector_weight,
        graph_weight=request.graph_weight,
        anchor_nodes=request.anchor_nodes,
        max_depth=request.max_depth,
        edge_type_weights=request.edge_type_weights,
        min_score=request.min_score,
        filter_metadata=request.filter_metadata,
        deduplicate=request.deduplicate
    )
    
    search_results = [
        SearchResult(
            node_id=r["node_id"],
            text=r["text"],
            metadata=r["metadata"],
            vector_score=r["vector_score"],
            graph_score=r["graph_score"],
            combined_score=r["combined_score"],
            reasoning=r["reasoning"]
        )
        for r in results
    ]
    
    response = SearchResponse(
        results=search_results,
        query_time_ms=query_time_ms,
        total_candidates=total_candidates,
        search_type="hybrid"
    )
    
    # Cache the result
    cache.set("hybrid", cache_params, response.model_dump())
    
    return response


@router.post("/compare", response_model=dict)
async def compare_search_modes(
    request: HybridSearchRequest,
    hybrid_ranker: HybridRanker = Depends(get_hybrid_ranker)
) -> dict:
    """
    Compare results across vector-only, graph-only, and hybrid search.
    
    Useful for demonstrating the advantages of hybrid search by
    showing how it combines the best of both approaches.
    """
    comparison = hybrid_ranker.compare_search_modes(
        query_text=request.query_text,
        top_k=request.top_k,
        vector_weight=request.vector_weight,
        graph_weight=request.graph_weight,
        anchor_nodes=request.anchor_nodes
    )
    
    return {
        "query_text": request.query_text,
        "vector_only": {
            "results": [
                {
                    "node_id": r["node_id"],
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "score": r.get("vector_score", 0)
                }
                for r in comparison["vector_only"]["results"]
            ],
            "query_time_ms": comparison["vector_only"]["query_time_ms"]
        },
        "graph_only": {
            "results": [
                {
                    "node_id": r["node_id"],
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "score": r.get("graph_score", 0),
                    "depth": r.get("depth", 0)
                }
                for r in comparison["graph_only"]["results"]
            ],
            "query_time_ms": comparison["graph_only"]["query_time_ms"]
        },
        "hybrid": {
            "results": [
                {
                    "node_id": r["node_id"],
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "vector_score": r.get("vector_score", 0),
                    "graph_score": r.get("graph_score", 0),
                    "combined_score": r.get("combined_score", 0)
                }
                for r in comparison["hybrid"]["results"]
            ],
            "query_time_ms": comparison["hybrid"]["query_time_ms"]
        },
        "analysis": comparison["analysis"]
    }


@router.get("/path/{source_id}/{target_id}")
async def find_path(
    source_id: str,
    target_id: str,
    graph_engine: GraphSearchEngine = Depends(get_graph_engine),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> dict:
    """
    Find the shortest path between two nodes.
    """
    # Validate nodes exist
    source = sqlite_store.get_node(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Source node {source_id} not found")
    
    target = sqlite_store.get_node(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Target node {target_id} not found")
    
    path_result = graph_engine.find_path(source_id, target_id)
    
    if path_result is None:
        return {
            "source_id": source_id,
            "target_id": target_id,
            "path_exists": False,
            "message": "No path exists between these nodes"
        }
    
    return {
        "source_id": source_id,
        "target_id": target_id,
        "path_exists": True,
        **path_result
    }


# Utility endpoint - moved here for consistency
@router.get("/stats", response_model=StatsResponse, tags=["Utility"])
async def get_stats() -> StatsResponse:
    """
    Get database statistics including node/edge counts and index sizes.
    """
    db_manager = get_db_manager()
    stats = db_manager.get_stats()
    
    total_edges = stats["total_edges"]
    total_nodes = stats["total_nodes"]
    avg_edges = total_edges / total_nodes if total_nodes > 0 else 0.0
    
    return StatsResponse(
        total_nodes=stats["total_nodes"],
        total_edges=stats["total_edges"],
        edge_types=stats["edge_types"],
        avg_edges_per_node=round(avg_edges, 2),
        vector_index_size=stats["vector_index_size"],
        database_size_bytes=stats["database_size_bytes"]
    )

