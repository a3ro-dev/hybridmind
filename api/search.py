"""
Search API endpoints for HybridMind.
Vector, Graph, and Hybrid search operations.

Features:
- Vector similarity search (semantic)
- Graph traversal search (relational)
- Dense/sparse/graph fusion with optional reranking
- Query result caching for performance
"""

from datetime import datetime, timezone
import hashlib
import json
from typing import List, Literal, Optional
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
from config import settings

router = APIRouter(prefix="/search", tags=["Search"])


def _config_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
    ).hexdigest()


def _require_aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=422,
            detail="as_of must include an explicit timezone offset",
        )
    return value.astimezone(timezone.utc)


@router.post("/vector", response_model=SearchResponse)
async def vector_search(
    request: VectorSearchRequest,
    vector_engine: VectorSearchEngine = Depends(get_vector_engine),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
) -> SearchResponse:
    """
    Pure vector similarity search using cosine similarity.

    Returns nodes ranked by semantic similarity to the query text.
    Uses the configured exact 4096-dimensional remote embedding model.

    Stateless results use the configured bounded TTL cache.
    """
    # Check cache first
    cache = get_query_cache()
    corpus_generation = sqlite_store.get_corpus_generation()
    vector_index_stats = vector_engine.vector_index.get_stats()
    cache_params = {
        "query_text": request.query_text,
        "top_k": request.top_k,
        "min_score": request.min_score,
        "filter_metadata": request.filter_metadata,
        "corpus_generation": corpus_generation,
        "hnsw_ef_search": vector_index_stats.get("hnsw_ef_search"),
        "hnsw_ef_construction": vector_index_stats.get("hnsw_ef_construction"),
    }

    cached = cache.get("vector", cache_params)
    if cached:
        response = SearchResponse(**cached)
        if response.execution_trace is not None:
            response.execution_trace["cache_hit"] = True
        return response

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
        search_type="vector",
        execution_trace={
            "schema_version": "hybridmind.search-execution/v1",
            "corpus_generation": corpus_generation,
            "search_mode": "vector_only",
            "resolved_config_sha256": _config_sha256(cache_params),
            "resolved_controls": {
                "search_mode": "vector_only",
                "top_k": request.top_k,
                "hnsw_ef_search": vector_index_stats.get("hnsw_ef_search"),
                "hnsw_ef_construction": vector_index_stats.get(
                    "hnsw_ef_construction"
                ),
            },
            "cache_hit": False,
            "stages": {
                "dense": {
                    "requested": True,
                    "executed": True,
                    "candidates": total_candidates,
                    "identity": type(vector_engine.vector_index).__name__,
                    "hnsw_ef_search": vector_index_stats.get("hnsw_ef_search"),
                    "hnsw_ef_construction": vector_index_stats.get(
                        "hnsw_ef_construction"
                    ),
                },
                "sparse": {"requested": False, "executed": False, "candidates": 0},
                "graph": {"requested": False, "executed": False, "candidates": 0},
            },
        },
    )

    # Cache the result
    cache.set("vector", cache_params, response.model_dump())

    return response


@router.get("/graph", response_model=SearchResponse)
async def graph_search(
    start_id: str = Query(..., description="Starting node ID"),
    depth: int = Query(default=2, ge=1, le=5, description="Maximum traversal depth"),
    edge_types: Optional[List[str]] = Query(default=None, description="Filter by edge types"),
    direction: Literal["outgoing", "incoming", "both", "typed"] = Query(
        default="both", description="'outgoing', 'incoming', 'both', or 'typed'"
    ),
    as_of: Optional[datetime] = Query(
        default=None, description="Timezone-aware point-in-time validity boundary"
    ),
    graph_engine: GraphSearchEngine = Depends(get_graph_engine),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> SearchResponse:
    """
    Graph traversal search from a starting node.

    Returns nodes reachable within the specified depth,
    ranked by graph proximity (closer nodes first).
    """
    as_of = _require_aware(as_of)
    # Validate start node exists
    start_node = sqlite_store.get_node(start_id)
    if start_node is None:
        raise HTTPException(status_code=404, detail=f"Start node {start_id} not found")

    results, query_time_ms, total_candidates = graph_engine.traverse(
        start_id=start_id,
        depth=depth,
        edge_types=edge_types,
        direction=direction,
        as_of=as_of,
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
        search_type="graph",
        execution_trace={
            "schema_version": "hybridmind.search-execution/v1",
            "corpus_generation": sqlite_store.get_corpus_generation(),
            "search_mode": "graph_only",
            "as_of": as_of.isoformat() if as_of is not None else None,
            "resolved_config_sha256": _config_sha256({
                "start_id": start_id,
                "depth": depth,
                "edge_types": edge_types,
                "direction": direction,
                "as_of": as_of,
            }),
            "cache_hit": False,
            "stages": {
                "dense": {"requested": False, "executed": False, "candidates": 0},
                "sparse": {"requested": False, "executed": False, "candidates": 0},
                "graph": {
                    "requested": True,
                    "executed": True,
                    "candidates": total_candidates,
                },
            },
        },
    )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    hybrid_ranker: HybridRanker = Depends(get_hybrid_ranker),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
) -> SearchResponse:
    """
    Hybrid dense, sparse, and graph retrieval with optional reranking.

    Stateless requests may use the bounded query cache. Requests that track
    node access always execute the ranker so access counters remain correct.
    """
    # Check cache first
    cache = get_query_cache()
    corpus_generation = sqlite_store.get_corpus_generation()
    cache_params = {
        "query_text": request.query_text,
        "top_k": request.top_k,
        "vector_weight": request.vector_weight,
        "graph_weight": request.graph_weight,
        "anchor_nodes": request.anchor_nodes,
        "max_depth": request.max_depth,
        "min_score": request.min_score,
        "filter_metadata": request.filter_metadata,
        "deduplicate": request.deduplicate,
        "rerank_pool": request.rerank_pool,
        "bm25_boost_weight": request.bm25_boost_weight,
        "overlap_threshold": request.overlap_threshold,
        "fusion_mode": request.fusion_mode,
        "include_images": request.include_images,
        "search_mode": request.search_mode,
        "route_weights": request.route_weights,
        "track_access": request.track_access,
        "as_of": request.as_of.isoformat() if request.as_of is not None else None,
        "corpus_generation": corpus_generation,
    }

    should_track_access = (
        settings.access_tracking_enabled
        if request.track_access is None
        else request.track_access
    )
    if not should_track_access:
        cached = cache.get("hybrid", cache_params)
        if cached:
            cached_response = SearchResponse(**cached)
            if cached_response.execution_trace is not None:
                cached_response.execution_trace["cache_hit"] = True
            return cached_response

    # Execute search
    results, query_time_ms, total_candidates, execution_trace = hybrid_ranker.search(
        query_text=request.query_text,
        top_k=request.top_k,
        vector_weight=request.vector_weight,
        graph_weight=request.graph_weight,
        anchor_nodes=request.anchor_nodes,
        max_depth=request.max_depth,
        min_score=request.min_score,
        filter_metadata=request.filter_metadata,
        deduplicate=request.deduplicate,
        rerank_pool=request.rerank_pool,
        bm25_boost_weight=request.bm25_boost_weight,
        overlap_threshold=request.overlap_threshold,
        fusion_mode=request.fusion_mode,
        include_images=request.include_images,
        search_mode=request.search_mode,
        route_weights=request.route_weights,
        track_access=request.track_access,
        as_of=request.as_of,
        return_trace=True,
    )
    generation_after = sqlite_store.get_corpus_generation()
    if generation_after != corpus_generation:
        raise HTTPException(
            status_code=409,
            detail="corpus changed during search; retry against a stable generation",
        )
    execution_trace["corpus_generation"] = corpus_generation

    search_results = [
        SearchResult(
            node_id=r["node_id"],
            text=r["text"],
            metadata=r["metadata"],
            vector_score=r["vector_score"],
            graph_score=r["graph_score"],
            graph_gate=r.get("graph_gate"),
            effective_graph_score=r.get("effective_graph_score"),
            combined_score=r["combined_score"],
            rerank_score=r.get("rerank_score"),
            rerank_attempted=r.get("rerank_attempted"),
            rerank_applied=r.get("rerank_applied"),
            rerank_failure_type=r.get("rerank_failure_type"),
            bm25_score=r.get("bm25_score"),
            time_score=r.get("time_score"),
            salience_score=r.get("salience_score"),
            reasoning=r["reasoning"]
        )
        for r in results
    ]

    response = SearchResponse(
        results=search_results,
        query_time_ms=query_time_ms,
        total_candidates=total_candidates,
        search_type=request.search_mode,
        execution_trace=execution_trace,
    )

    # Stateful searches must not be cached: a cache hit would skip the
    # ranker's access-count update and return salience based on stale state.
    if not should_track_access:
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
