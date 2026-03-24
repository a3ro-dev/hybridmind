"""
HybridMind FastAPI Application Entry Point.

Vector + Graph Native Database for AI Retrieval.

Production-ready with:
- Eager model initialization (eliminates cold start)
- Query caching for repeated requests
- Rate limiting for protection
- Comprehensive health endpoints
"""

import logging
import time
import psutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from api.nodes import router as nodes_router
from api.edges import router as edges_router
from api.search import router as search_router
from api.bulk import router as bulk_router
# Comparison router has been removed.
from api.dependencies import get_db_manager
from engine.cache import get_query_cache
from middleware.rate_limit import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Track startup time for metrics
_startup_time: Optional[float] = None
_model_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown.
    
    Performs eager initialization to eliminate cold start:
    1. Initialize embedding model (the slow part ~3-4s)
    2. Run warmup embedding to fully load model weights
    3. Load persisted indexes from disk
    4. Validate all components are ready
    """
    global _startup_time, _model_loaded
    
    startup_start = time.perf_counter()
    logger.info("Warming up HybridMind...")
    
    # Step 1: Get database manager (triggers all component initialization)
    logger.info("  Initializing storage components...")
    db_manager = get_db_manager()
    
    # Step 2: Force embedding model load with warmup query
    logger.info("  Loading embedding model...")
    warmup_start = time.perf_counter()
    
    embedding_engine = db_manager.embedding_engine
    if embedding_engine.model is not None:
        # Run warmup embedding to ensure model is fully loaded
        _ = embedding_engine.embed("warmup query for model initialization")
        _model_loaded = True
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        logger.info(f"  Embedding model loaded in {warmup_time:.0f}ms")
    else:
        logger.warning("  Embedding model not available, using mock embeddings")
    
    # Step 3: Initialize query cache
    logger.info("  Initializing query cache...")
    cache = get_query_cache(
        maxsize=settings.cache_size,
        ttl=300  # 5 minute TTL
    )
    
    # Step 4: Log stats
    stats = db_manager.get_stats()
    total_startup = (time.perf_counter() - startup_start) * 1000
    _startup_time = time.time()
    
    logger.info(f"HybridMind ready in {total_startup:.0f}ms")
    logger.info(f"   Loaded: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    logger.info(f"   Vector index: {stats['vector_index_size']} embeddings")
    logger.info(f"   Graph index: {stats['graph_node_count']} nodes, {stats['graph_edge_count']} edges")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HybridMind...")
    db_manager.save_indexes()
    db_manager.close()
    logger.info("HybridMind shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="HybridMind",
    description="""
## Vector + Graph Native Database for AI Retrieval

HybridMind is a high-performance hybrid database that combines vector embeddings 
with graph-based relationships for superior AI retrieval.

### Key Features

- **Vector Search**: Semantic similarity using cosine distance with FAISS
- **Graph Search**: Relationship traversal using NetworkX
- **Hybrid Search**: Contextual Relevance Score (CRS) combining both approaches
- **Query Caching**: Fast repeated queries with TTL-based cache
- **Rate Limiting**: Protection against abuse

### Hybrid Scoring Algorithm (CRS)

```
CRS = α × vector_score + β × graph_score
```

Where:
- `α` = vector_weight (default 0.6)
- `β` = graph_weight (default 0.4)
- `vector_score` = cosine similarity (0-1)
- `graph_score` = inverse shortest path distance (0-1)

### Use Case

Research paper knowledge graph with semantic search and citation relationships.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware (100 requests/minute default)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=100,
    enabled=not settings.debug  # Disable in debug mode
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(nodes_router)
app.include_router(edges_router)
app.include_router(search_router)
app.include_router(bulk_router)
# comparison_router removed


# ==================== Health & Utility Endpoints ====================

# Response models for health endpoints
class HealthResponse(BaseModel):
    """Comprehensive health check response."""
    status: str
    timestamp: float
    uptime_seconds: float
    components: dict
    metrics: dict


class ReadinessResponse(BaseModel):
    """Kubernetes readiness probe response."""
    status: str
    model_loaded: bool
    nodes_loaded: int
    edges_loaded: int


class LivenessResponse(BaseModel):
    """Kubernetes liveness probe response."""
    status: str


@app.get("/", tags=["Utility"])
async def root():
    """Welcome endpoint with API overview."""
    return {
        "name": "HybridMind",
        "version": "1.0.0",
        "description": "Vector + Graph Native Database for AI Retrieval",
        "docs": "/docs",
        "endpoints": {
            "nodes": "/nodes",
            "edges": "/edges",
            "search": {
                "vector": "/search/vector",
                "graph": "/search/graph",
                "hybrid": "/search/hybrid",
                "compare": "/search/compare"
            },
            "bulk": {
                "nodes": "/bulk/nodes",
                "edges": "/bulk/edges",
                "import": "/bulk/import"
            },
            # Comparison endpoints disabled
            "health": {
                "full": "/health",
                "ready": "/ready",
                "live": "/live"
            },
            "stats": "/search/stats",
            "cache": "/cache/stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns detailed status of all components:
    - Embedding model status and latency
    - Vector index status and size
    - Graph index status and size
    - Database connectivity
    - System metrics (CPU, memory, disk)
    """
    components = {}
    
    try:
        db_manager = get_db_manager()
        
        # Check embedding model
        try:
            start = time.perf_counter()
            _ = db_manager.embedding_engine.embed("health check")
            latency = (time.perf_counter() - start) * 1000
            components["embedding"] = {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "model": settings.embedding_model
            }
        except Exception as e:
            components["embedding"] = {"status": "unhealthy", "error": str(e)}
        
        # Check vector index
        components["vector_index"] = {
            "status": "healthy",
            "size": db_manager.vector_index.size,
            "dimension": db_manager.vector_index.dimension
        }
        
        # Check graph index
        components["graph_index"] = {
            "status": "healthy",
            "nodes": db_manager.graph_index.node_count,
            "edges": db_manager.graph_index.edge_count
        }
        
        # Check database
        try:
            node_count = db_manager.sqlite_store.count_nodes()
            components["database"] = {
                "status": "healthy",
                "nodes": node_count,
                "size_bytes": db_manager.sqlite_store.get_database_size()
            }
        except Exception as e:
            components["database"] = {"status": "unhealthy", "error": str(e)}
        
        # Check cache
        cache = get_query_cache()
        components["cache"] = {
            "status": "healthy",
            **cache.stats
        }
        
    except Exception as e:
        components["system"] = {"status": "unhealthy", "error": str(e)}
    
    # System metrics
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_mb": round(psutil.virtual_memory().available / (1024 * 1024)),
    }
    
    # Try to get disk usage (may fail on some systems)
    try:
        disk = psutil.disk_usage('/')
        metrics["disk_percent"] = disk.percent
    except:
        pass
    
    # Calculate uptime
    uptime = time.time() - _startup_time if _startup_time else 0
    
    # Determine overall status
    unhealthy_components = [
        name for name, info in components.items()
        if isinstance(info, dict) and info.get("status") == "unhealthy"
    ]
    
    if not unhealthy_components:
        status = "healthy"
    elif len(unhealthy_components) < len(components):
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        timestamp=time.time(),
        uptime_seconds=round(uptime, 1),
        components=components,
        metrics=metrics
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """
    Kubernetes readiness probe.
    
    Returns ready status only when:
    - Database manager is initialized
    - Embedding model is loaded
    - Data is loaded from disk
    """
    try:
        db_manager = get_db_manager()
        stats = db_manager.get_stats()
        
        return ReadinessResponse(
            status="ready" if _model_loaded else "not_ready",
            model_loaded=_model_loaded,
            nodes_loaded=stats["total_nodes"],
            edges_loaded=stats["total_edges"]
        )
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model_loaded": False, "nodes_loaded": 0, "edges_loaded": 0}
        )


@app.get("/live", response_model=LivenessResponse, tags=["Health"])
async def liveness_check():
    """
    Kubernetes liveness probe.
    
    Simple check that the application is running.
    Always returns success if the server is responding.
    """
    return LivenessResponse(status="alive")


@app.get("/cache/stats", tags=["Utility"])
async def cache_stats():
    """Get query cache statistics."""
    cache = get_query_cache()
    return cache.stats


@app.post("/cache/clear", tags=["Utility"])
async def clear_cache():
    """Clear the query cache."""
    cache = get_query_cache()
    cache.invalidate_all()
    return {"status": "success", "message": "Cache cleared"}


@app.post("/snapshot", tags=["Utility"])
async def create_snapshot():
    """Create a persistence snapshot of indexes."""
    try:
        db_manager = get_db_manager()
        db_manager.save_indexes()
        return {
            "status": "success",
            "message": "Indexes saved to disk"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.get("/database", tags=["Utility"])
async def get_database_info():
    """
    Get information about the .mind database file.
    
    HybridMind uses the `.mind` extension as its native database format.
    A .mind file is a directory containing:
    - store.db: SQLite database
    - vectors.faiss: FAISS vector index
    - graph.nx: NetworkX graph
    - manifest.json: Metadata and stats
    """
    try:
        db_manager = get_db_manager()
        return db_manager.mind_file.get_info()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/database/export", tags=["Utility"])
async def export_database(compress: bool = True):
    """
    Export the .mind database to a portable archive.
    
    Creates a .mind.zip file that can be shared and imported elsewhere.
    """
    try:
        db_manager = get_db_manager()
        output = f"data/export_{int(time.time())}"
        result = db_manager.mind_file.export(output, compress=compress)
        
        if result:
            return {
                "status": "success",
                "path": result,
                "message": "Database exported successfully"
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Export failed"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hybridmind.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
