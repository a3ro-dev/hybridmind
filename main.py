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
from api.comparison import router as comparison_router
from api.dependencies import get_db_manager
from engine.cache import get_query_cache


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Track startup time for metrics
_startup_time: Optional[float] = None
_model_loaded: bool = False


def verify_integrity(mind_path: str) -> str:
    from storage.mindfile import MindFile
    import hashlib
    from pathlib import Path
    import zipfile
    import shutil
    
    mind = MindFile(mind_path)
    if not mind.exists:
        return "New database"
        
    manifest = mind.read_manifest()
    if not manifest or "checksums" not in manifest:
        return "No checksums to verify"
        
    corrupted = False
    paths = mind.get_paths()
    for comp, expected_hash in manifest["checksums"].items():
        comp_path = Path(mind_path) / comp
        if comp == "vectors" and not comp_path.exists() and Path(str(comp_path) + ".faiss").exists():
            comp_path = Path(str(comp_path) + ".faiss")
            
        if not comp_path.exists() and comp != "vectors.map":
            logger.warning(f"  Component {comp} missing")
            corrupted = True
            break
            
        if comp_path.exists():
            sha_hash = hashlib.sha256()
            with open(comp_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha_hash.update(chunk)
            if sha_hash.hexdigest() != expected_hash:
                logger.warning(f"  Checksum mismatch for {comp}")
                corrupted = True
                break
                
    if not corrupted:
        return "PASSED"
        
    logger.warning("  Database corrupted! Attempting to restore from backup...")
    backup_dir = Path("data/backups")
    if backup_dir.exists():
        backups = sorted(backup_dir.glob("snapshot_*.mind.zip"))
        if backups:
            latest = backups[-1]
            logger.info(f"  Restoring from {latest}")
            shutil.rmtree(mind_path, ignore_errors=True)
            with zipfile.ZipFile(latest, 'r') as zip_ref:
                zip_ref.extractall(Path(mind_path).parent)
            return "PASSED (Restored from backup)"
            
    logger.error("  No valid backup found. Starting with empty indexes.")
    Path(paths["vector_index"]).unlink(missing_ok=True)
    Path(paths["vector_index"] + ".faiss").unlink(missing_ok=True)
    Path(paths["graph"]).unlink(missing_ok=True)
    return "FAILED"

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
    
    # Step 0: Integrity Check
    integrity_status = verify_integrity(settings.mind_file_path)
    
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

    # Step 2.5: BUG-1 — Verify FAISS index is in sync with SQLite
    sqlite_count = db_manager.sqlite_store.count_nodes()
    faiss_count = db_manager.vector_index.size
    if abs(sqlite_count - faiss_count) > 0:
        logger.warning(
            f"Index mismatch detected: SQLite={sqlite_count}, FAISS={faiss_count}. "
            "Rebuilding indexes from SQLite..."
        )
        import numpy as np
        all_embeddings = db_manager.sqlite_store.get_all_node_embeddings()
        db_manager.vector_index.clear()
        db_manager.graph_index.clear()
        for node_id, embedding in all_embeddings:
            db_manager.vector_index.add(node_id, embedding)
            db_manager.graph_index.add_node(node_id)
        all_edges = db_manager.sqlite_store.get_all_edges()
        for edge in all_edges:
            db_manager.graph_index.add_edge(
                edge["source_id"], edge["target_id"],
                edge["type"], edge.get("weight", 1.0)
            )
        logger.info(
            f"Index rebuild complete: {db_manager.vector_index.size} vectors, "
            f"{db_manager.graph_index.node_count} graph nodes"
        )
    else:
        logger.info(f"Index sync verified: SQLite={sqlite_count}, FAISS={faiss_count}")

    # Step 3: Initialize query cache
    logger.info("  Initializing query cache...")
    cache = get_query_cache(
        maxsize=settings.cache_size,
        ttl=300  # 5 minute TTL
    )
    
    # Step 4: Log stats summary
    stats = db_manager.get_stats()
    total_startup = (time.perf_counter() - startup_start) * 1000
    _startup_time = time.time()
    
    manifest = db_manager.mind_file.read_manifest() or {}
    version = manifest.get('snapshot_version', 0)
    timestamp = manifest.get('modified', 'Unknown')
    soft_deleted = db_manager.sqlite_store.get_deleted_nodes_count()
    
    graph_embeddings_enabled = getattr(settings, "USE_GRAPH_CONDITIONED_EMBEDDINGS", False)
    
    print(f"\nHybridMind starting up")
    print(f"- Nodes: {stats['total_nodes']} ({soft_deleted} pending compaction)")
    print(f"- Edges: {stats['total_edges']}")
    print(f"- FAISS index: {stats['vector_index_size']} vectors")
    print(f"- Graph nodes: {stats['graph_node_count']}")
    print(f"- Snapshot manifest: v{version} @ {timestamp}")
    print(f"- Checksum verification: {integrity_status}")
    print(f"- Graph-conditioned embeddings: {'ENABLED' if graph_embeddings_enabled else 'DISABLED'}\n")
    
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

# Rate limiting removed — single-user local deployment does not require throttling.


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
app.include_router(comparison_router)


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


@app.post("/admin/compact", tags=["Admin"])
async def compact_database():
    """
    Compact the database by rebuilding FAISS index and hard-deleting soft-deleted nodes.
    """
    try:
        db_manager = get_db_manager()
        
        # 1. Rebuild FAISS index from scratch using only non-deleted nodes
        embeddings = db_manager.sqlite_store.get_all_node_embeddings()
        db_manager.vector_index.rebuild_from_embeddings(embeddings)
        
        # 2. Save indexes atomically
        stats = db_manager.get_stats()
        db_manager.mind_file.create_snapshot(
            sqlite_conn=db_manager.sqlite_store._get_connection(),
            vector_index=db_manager.vector_index,
            graph_index=db_manager.graph_index,
            nodes_count=stats["total_nodes"],
            edges_count=stats["total_edges"]
        )
        
        # 3. Hard-delete soft-deleted rows from SQLite
        deleted_count = db_manager.sqlite_store.hard_delete_soft_deleted_nodes()
        db_manager.vector_index.deleted_ids.clear()
        
        return {
            "status": "success", 
            "message": "Database compacted successfully",
            "compacted_nodes": deleted_count
        }
    except Exception as e:
        logger.error(f"Compaction failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hybridmind.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
