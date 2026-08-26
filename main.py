"""
HybridMind FastAPI Application Entry Point.

Local dense, sparse, and graph retrieval service for controlled experiments.

The service has bounded caching, authentication/rate controls, and health
endpoints. Production suitability and retrieval quality are deployment- and
benchmark-dependent; this module does not claim either by itself.
"""

import asyncio
import hashlib
import json
import logging
import time
import os
import psutil
import secrets
import ipaddress
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from config import settings
from api.nodes import router as nodes_router
from api.edges import router as edges_router
from api.search import router as search_router
from api.bulk import router as bulk_router
from api.comparison import router as comparison_router
from api.dependencies import coordinate_mutation, get_db_manager
from engine.cache import get_query_cache
from engine.device import gpu_info as _gpu_info


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Track startup time for metrics
_startup_time: Optional[float] = None
_model_loaded: bool = False


def _validate_api_security_configuration() -> None:
    """Refuse a network bind that would expose an unauthenticated API."""
    host = settings.host.strip().lower()
    try:
        local_bind = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local_bind = host == "localhost"
    if (
        not local_bind
        and not settings.api_key.strip()
        and not settings.allow_unauthenticated_private_networks
    ):
        raise RuntimeError("A non-loopback HYBRIDMIND_HOST requires HYBRIDMIND_API_KEY")


def verify_integrity(mind_path: str) -> str:
    from storage.mindfile import MindFile
    from pathlib import Path

    mind = MindFile(mind_path)
    if not mind.exists:
        return "New database"

    try:
        MindFile._validate_sqlite(mind.sqlite_path)
        return "PASSED (SQLite source of truth)"
    except Exception:
        logger.error("Live SQLite integrity check failed; searching verified backups")

    backup_dir = Path(settings.backup_dir)
    for backup in reversed(sorted(backup_dir.glob("snapshot_*.mind.zip"))):
        try:
            MindFile.validate_archive(str(backup))
            if mind.restore_from_archive(str(backup)):
                return "PASSED (restored from verified backup)"
        except Exception:
            logger.warning("Rejected invalid snapshot backup: %s", backup.name)
    # Never delete or replace live data when no verified backup exists.
    return "FAILED (live data preserved)"


async def _memory_compression_worker(db_manager) -> None:
    """Periodically create lossy derived summaries when explicitly enabled."""
    interval = max(60, settings.memory_compression_interval_seconds)
    while True:
        await asyncio.sleep(interval)
        try:
            from engine.consolidation import consolidate_sessions

            result = await asyncio.to_thread(
                consolidate_sessions,
                db_manager,
                min_facts=settings.memory_compression_min_facts,
                max_age_hours=settings.memory_compression_max_age_hours,
                model=settings.consolidation_model,
                archive_sources=settings.memory_compression_archive_sources,
            )
            logger.info(f"memory compression cycle: {result}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Memory compression cycle failed type=%s", type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown.

    Initializes and validates local storage. A remote embedding warm-up is
    opt-in because it can wake billable infrastructure; use the budget-gated
    preflight before enabling it for an evaluation deployment.
    """
    global _startup_time, _model_loaded

    _validate_api_security_configuration()
    startup_start = time.perf_counter()
    logger.info("Warming up HybridMind...")

    # Step 0: Integrity Check
    integrity_status = verify_integrity(settings.mind_file_path)

    # Step 1: Get database manager (triggers all component initialization)
    logger.info("  Initializing storage components...")
    db_manager = get_db_manager()

    # Step 2: Resolve the embedding backend. Network warm-up is opt-in.
    logger.info("  Resolving embedding backend...")
    warmup_start = time.perf_counter()

    embedding_engine = db_manager.embedding_engine
    if embedding_engine.model is not None:
        _model_loaded = True
        warmup_vec = None
        if settings.startup_embedding_warmup_enabled:
            try:
                if hasattr(embedding_engine, "warmup"):
                    warmup_vec = embedding_engine.warmup(
                        timeout_s=settings.startup_embedding_warmup_seconds
                    )
                else:
                    warmup_vec = embedding_engine.embed(
                        "warmup query for model initialization"
                    )
                warmup_time = (time.perf_counter() - warmup_start) * 1000
                logger.info("  Embedding endpoint warmed in %.0fms", warmup_time)
            except Exception as exc:
                logger.error(
                    "Embedding endpoint not ready at startup type=%s; the first "
                    "real call will fail closed.",
                    type(exc).__name__,
                )
        else:
            logger.info(
                "  Remote embedding warm-up skipped (run budget-gated preflight first)"
            )

        # Step 2.1: Hard-fail on embedding/FAISS dimension mismatch.
        # A silent mismatch corrupts every similarity score in the index — never allow it.
        # Only checkable when warmup actually returned a vector.
        if warmup_vec is not None and (actual_dim := int(warmup_vec.shape[-1])) != (
            index_dim := db_manager.vector_index.dimension
        ):
            raise RuntimeError(
                f"Embedding/FAISS dimension mismatch: the resolved embedding backend "
                f"({type(embedding_engine).__name__}) outputs {actual_dim}-dim vectors, "
                f"but the FAISS index at {settings.mind_file_path} was built with "
                f"{index_dim} dims. This corrupts every similarity score. Fix by either "
                f"(a) setting RUNPOD_TEI_EMBEDDING_URL to a TEI endpoint serving a "
                f"{index_dim}-dim model, or (b) re-indexing the existing corpus for the "
                f"current embedder with `python scripts/reindex_embeddings.py`."
            )
    else:
        raise RuntimeError(
            "The embedding backend did not initialize. HybridMind requires an exact "
            "4096-dimensional backend and has no mock/local fallback."
        )

    # Step 2.5: BUG-1 — Verify FAISS index is in sync with SQLite
    sqlite_count = db_manager.sqlite_store.count_retrievable_nodes()
    faiss_count = db_manager.vector_index.size
    if abs(sqlite_count - faiss_count) > 0:
        logger.warning(
            f"Index mismatch detected: SQLite={sqlite_count}, FAISS={faiss_count}. "
            "Rebuilding indexes from SQLite..."
        )
        db_manager._rebuild_indexes()
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
        ttl=300,  # 5 minute TTL
    )

    # Step 4: Log stats summary
    stats = db_manager.get_stats()
    total_startup = (time.perf_counter() - startup_start) * 1000
    _startup_time = time.time()

    manifest = db_manager.mind_file.read_manifest() or {}
    version = manifest.get("snapshot_version", 0)
    timestamp = manifest.get("modified", "Unknown")
    soft_deleted = db_manager.sqlite_store.get_deleted_nodes_count()

    graph_embeddings_enabled = getattr(
        settings, "use_graph_conditioned_embeddings", False
    )

    print(f"\nHybridMind starting up")
    print(f"- Nodes: {stats['total_nodes']} ({soft_deleted} pending compaction)")
    print(f"- Edges: {stats['total_edges']}")
    print(f"- FAISS index: {stats['vector_index_size']} vectors")
    print(f"- Graph nodes: {stats['graph_node_count']}")
    print(f"- Snapshot manifest: v{version} @ {timestamp}")
    print(f"- Checksum verification: {integrity_status}")
    print(
        f"- Graph-conditioned embeddings: {'ENABLED' if graph_embeddings_enabled else 'DISABLED'}"
    )

    from engine.llm_client import provider_chain

    providers = provider_chain()
    print(
        "- Fact Extractor LLM: "
        + (
            " -> ".join(providers)
            if providers
            else "DISABLED (no policy-allowed provider)"
        )
    )

    print("\n")

    compression_task = None
    if settings.memory_compression_enabled:
        compression_task = asyncio.create_task(_memory_compression_worker(db_manager))
        logger.info(
            "Derived-summary consolidation enabled: interval=%ss archive_sources=%s",
            settings.memory_compression_interval_seconds,
            settings.memory_compression_archive_sources,
        )

    yield

    # Shutdown
    logger.info("Shutting down HybridMind...")
    if compression_task is not None:
        compression_task.cancel()
        try:
            await compression_task
        except asyncio.CancelledError:
            pass
    db_manager.save_indexes()
    db_manager.close(save=False)
    logger.info("HybridMind shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="HybridMind",
    description="""
## Local dense, sparse, and graph retrieval service

HybridMind exposes controlled retrieval paths over SQLite-backed memory records.
Quality and latency must be established by an evidence-ID benchmark for the
specific corpus and deployment.

### Key Features

- **Vector Search**: Semantic similarity using cosine distance with FAISS
- **Graph Search**: Relationship traversal using NetworkX
- **Hybrid Search**: weighted reciprocal-rank fusion with optional reranking
- **Query Caching**: Fast repeated queries with TTL-based cache
- **Rate Limiting**: Protection against abuse

### Default fusion

```
RRF(d) = Σ weight(signal) / (k + rank(signal, d))
```

Request controls can isolate vector, sparse, graph, or combined paths. A
cross-encoder may rerank the bounded fusion pool when configured; responses
expose whether it was attempted and applied.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.cors_allowed_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-HybridMind-API-Key"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        host.strip() for host in settings.trusted_hosts.split(",") if host.strip()
    ],
    www_redirect=False,
)

_request_windows = defaultdict(deque)
_rate_lock = asyncio.Lock()
_public_paths = {"/live"}
_expensive_prefixes = (
    "/ingest/",
    "/bulk/",
    "/snapshot",
    "/database/export",
    "/admin/",
    "/health",
)


def _is_local_client(request: Request) -> bool:
    host = request.client.host if request.client else ""
    if host == "testclient":
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or (
            settings.allow_unauthenticated_private_networks and address.is_private
        )
    except ValueError:
        return False


@app.middleware("http")
async def authenticate_and_limit(request: Request, call_next):
    """Require a constant-time API-key check off loopback and cap costly calls."""
    if request.method != "OPTIONS" and request.url.path not in _public_paths:
        configured = settings.api_key
        supplied = request.headers.get("X-HybridMind-API-Key", "")
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            supplied = authorization[7:]
        local_bypass = settings.allow_unauthenticated_localhost and _is_local_client(
            request
        )
        if configured:
            if not supplied or not secrets.compare_digest(supplied, configured):
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        elif not local_bypass:
            return JSONResponse(
                status_code=503,
                content={"detail": "API authentication is not configured"},
            )

        expensive = request.url.path.startswith(_expensive_prefixes)
        limit = (
            settings.expensive_rate_limit_per_minute
            if expensive
            else settings.request_rate_limit_per_minute
        )
        # Starlette's synthetic test client is not a network principal and is
        # shared across the entire test process; do not let one test module
        # exhaust another module's window.
        if limit > 0 and (not request.client or request.client.host != "testclient"):
            key = (
                (request.client.host if request.client else "unknown"),
                "expensive" if expensive else "standard",
            )
            now = time.monotonic()
            async with _rate_lock:
                window = _request_windows[key]
                while window and now - window[0] >= 60:
                    window.popleft()
                if len(window) >= limit:
                    return JSONResponse(
                        status_code=429, content={"detail": "Rate limit exceeded"}
                    )
                window.append(now)
    return await call_next(request)


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
    logger.error(
        "Unexpected request failure path=%s type=%s",
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": "An unexpected error occurred",
        },
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
        "description": "Local dense, sparse, and graph retrieval service",
        "docs": "/docs",
        "endpoints": {
            "nodes": "/nodes",
            "edges": "/edges",
            "search": {
                "vector": "/search/vector",
                "graph": "/search/graph",
                "hybrid": "/search/hybrid",
                "compare": "/search/compare",
            },
            "bulk": {
                "nodes": "/bulk/nodes",
                "edges": "/bulk/edges",
                "import": "/bulk/import",
            },
            "health": {"full": "/health", "ready": "/ready", "live": "/live"},
            "stats": "/search/stats",
            "cache": "/cache/stats",
        },
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

        # Do not make a billable/remote provider call from a probe by default.
        components["embedding"] = {
            "status": "configured",
            "model": settings.embedding_model,
            "remote_probe_enabled": settings.health_remote_checks,
        }
        if settings.health_remote_checks:
            try:
                start = time.perf_counter()
                db_manager.embedding_engine.embed("health check")
                components["embedding"].update(
                    {
                        "status": "healthy",
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                )
            except Exception:
                components["embedding"] = {
                    "status": "unhealthy",
                    "error": "Remote embedding probe failed",
                }

        # Check vector index
        components["vector_index"] = {
            "status": "healthy",
            "size": db_manager.vector_index.size,
            "dimension": db_manager.vector_index.dimension,
        }

        # Check graph index
        components["graph_index"] = {
            "status": "healthy",
            "nodes": db_manager.graph_index.node_count,
            "edges": db_manager.graph_index.edge_count,
        }

        # Check database
        try:
            node_count = db_manager.sqlite_store.count_nodes()
            components["database"] = {
                "status": "healthy",
                "nodes": node_count,
                "size_bytes": db_manager.sqlite_store.get_database_size(),
            }
        except Exception:
            components["database"] = {
                "status": "unhealthy",
                "error": "Database check failed",
            }

        # Check cache
        cache = get_query_cache()
        components["cache"] = {"status": "healthy", **cache.stats}

        # GPU / device info
        components["gpu"] = _gpu_info()

    except Exception:
        components["system"] = {
            "status": "unhealthy",
            "error": "Component initialization failed",
        }

    # System metrics
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_mb": round(psutil.virtual_memory().available / (1024 * 1024)),
    }

    # Try to get disk usage (may fail on some systems)
    try:
        disk = psutil.disk_usage("/")
        metrics["disk_percent"] = disk.percent
    except:
        pass

    # Calculate uptime
    uptime = time.time() - _startup_time if _startup_time else 0

    # Determine overall status
    unhealthy_components = [
        name
        for name, info in components.items()
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
        metrics=metrics,
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

        return {
            "status": "online",
            "model_loaded": db_manager.embedding_engine.model is not None,
            "nodes_loaded": stats.get("total_nodes", 0),
            "edges_loaded": stats.get("total_edges", 0),
            "graph_nodes": stats.get("graph_node_count", 0),
            "vector_nodes": stats.get("vector_index_size", 0),
            "settings": {
                "graph_conditioned_embeddings": getattr(
                    settings, "use_graph_conditioned_embeddings", False
                ),
                "dimensions": getattr(settings, "embedding_dimension", 4096),
            },
        }
    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model_loaded": False,
                "nodes_loaded": 0,
                "edges_loaded": 0,
            },
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
        snapshot = db_manager.save_indexes()
        return {
            "status": "success",
            "message": "Verified snapshot created",
            "snapshot": snapshot.name,
        }
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Snapshot creation failed"},
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
    except Exception:
        return JSONResponse(
            status_code=500, content={"error": "Database information unavailable"}
        )


@app.post("/database/export", tags=["Utility"])
async def export_database(compress: bool = True):
    """
    Export the .mind database to a portable archive.

    Creates a .mind.zip file that can be shared and imported elsewhere.
    """
    try:
        db_manager = get_db_manager()
        if not compress:
            return JSONResponse(
                status_code=400,
                content={"error": "Only verified compressed exports are supported"},
            )
        result = db_manager.save_indexes()
        return {
            "status": "success",
            "snapshot": result.name,
            "message": "Verified database snapshot exported successfully",
        }
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Export failed"})


@app.post("/admin/compact", tags=["Admin"])
async def compact_database():
    """
    Compact the database by rebuilding FAISS index and hard-deleting soft-deleted nodes.
    """
    try:
        db_manager = get_db_manager()

        # Preserve one verified, internally consistent recovery point before the
        # irreversible history purge. save_indexes owns the process mutation
        # guard, so run it before taking the async guard below.
        await asyncio.to_thread(db_manager.save_indexes)

        async with db_manager.mutation_async():
            try:
                with db_manager.sqlite_store.transaction():
                    deleted_count = (
                        db_manager.sqlite_store.hard_delete_soft_deleted_nodes()
                    )
                    db_manager._rebuild_indexes()
            except Exception:
                # SQL has rolled back. Replace every derived index from that
                # authoritative pre-compaction state before reporting failure.
                db_manager._rebuild_indexes()
                raise

        from engine.cache import invalidate_cache

        invalidate_cache()
        return {
            "status": "success",
            "message": "Database compacted successfully",
            "compacted_nodes": deleted_count,
        }
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Database compaction failed"},
        )


@app.post("/admin/clear", tags=["Admin"])
async def clear_database():
    """Clear all data from the database."""
    db_manager = None
    try:
        db_manager = get_db_manager()
        async with db_manager.mutation_async():
            try:
                with db_manager.sqlite_store.transaction():
                    with db_manager.sqlite_store._cursor() as cursor:
                        cursor.execute("DELETE FROM edges")
                        # Clear/forget is an erasure boundary. Immutable history
                        # must not retain text after current nodes are deleted.
                        cursor.execute("DELETE FROM node_versions")
                        cursor.execute("DELETE FROM nodes")
                    db_manager.vector_index.clear()
                    db_manager.graph_index.clear()
                    db_manager.bm25_index.clear()
                    if getattr(db_manager, "visual_store", None) is not None:
                        db_manager.visual_store.clear()
                    if getattr(db_manager, "colbert_store", None) is not None:
                        db_manager.colbert_store.clear()
            except Exception:
                db_manager._rebuild_indexes()
                raise

        from engine.cache import invalidate_cache

        invalidate_cache()
        clear_fact_cache()
        return {"status": "success", "message": "Database cleared"}
    except Exception as exc:
        logger.error("Database clear failed type=%s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"error": "Database clear failed"})


# ==================== Admin: Memory Lifecycle ====================


class ConsolidateRequest(BaseModel):
    min_facts: int = settings.memory_compression_min_facts
    max_age_hours: int = settings.memory_compression_max_age_hours
    model: Optional[str] = None
    archive_sources: bool = settings.memory_compression_archive_sources


@app.post("/admin/consolidate", tags=["Admin"])
async def consolidate_memory(request: ConsolidateRequest = ConsolidateRequest()):
    """
    Consolidate old session memories into summary nodes.

    Groups extracted_fact nodes by session_id, summarizes sessions with
    >= min_facts facts that are older than max_age_hours. Idempotent.
    """
    try:
        db_manager = get_db_manager()
        from engine.consolidation import consolidate_sessions

        result = await asyncio.to_thread(
            consolidate_sessions,
            db_manager,
            min_facts=request.min_facts,
            max_age_hours=request.max_age_hours,
            model=request.model,
            archive_sources=request.archive_sources,
        )
        status = "partial" if result.get("failures") else "success"
        return {"status": status, **result}
    except ValueError:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": "Unsafe consolidation request rejected",
            },
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Consolidation failed"},
        )


class PruneRequest(BaseModel):
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)


@app.post("/admin/prune-low-importance", tags=["Admin"])
async def prune_low_importance(request: PruneRequest = PruneRequest()):
    """
    Soft-delete low-importance memory nodes.

    Computes importance_score() for every node (recency + centrality +
    access frequency). Nodes with score < threshold are soft-deleted.
    Runs compaction automatically after pruning.
    """
    db_manager = None
    try:
        db_manager = get_db_manager()
        from engine.consolidation import importance_score

        async with db_manager.mutation_async():
            # Score one stable graph/SQL view, then apply the complete mutation
            # as a transaction. A single projection failure aborts the request.
            with db_manager.sqlite_store._cursor() as cursor:
                cursor.execute("SELECT id FROM nodes WHERE deleted_at IS NULL")
                all_ids = [row["id"] for row in cursor.fetchall()]

            graph = db_manager.graph_index.graph
            max_graph_degree = max(
                (float(degree) for _, degree in graph.degree()),
                default=1.0,
            )

            to_prune = [
                node_id
                for node_id in all_ids
                if importance_score(
                    node_id,
                    db_manager,
                    max_graph_degree=max_graph_degree,
                )
                < request.threshold
            ]
            try:
                with db_manager.sqlite_store.transaction():
                    for node_id in to_prune:
                        if not db_manager.sqlite_store.soft_delete_node(node_id):
                            raise RuntimeError("node disappeared during pruning")
                        db_manager.vector_index.remove(node_id)
                        db_manager.graph_index.remove_node(node_id)
                        db_manager.bm25_index.remove(node_id)
            except Exception:
                db_manager._rebuild_indexes()
                raise

        from engine.cache import invalidate_cache

        invalidate_cache()
        pruned = len(to_prune)

        logger.info(
            f"prune-low-importance: pruned {pruned}/{len(all_ids)} nodes (threshold={request.threshold})"
        )
        return {
            "status": "success",
            "threshold": request.threshold,
            "nodes_evaluated": len(all_ids),
            "nodes_pruned": pruned,
        }
    except Exception as exc:
        logger.error("Pruning failed type=%s", type(exc).__name__)
        return JSONResponse(
            status_code=500, content={"status": "error", "message": "Pruning failed"}
        )


@app.post("/admin/detect-communities", tags=["Admin"])
async def detect_communities(
    mutation_guard: None = Depends(coordinate_mutation),
):
    """
    Run Louvain community detection and create community summary nodes.

    Detects clusters in the memory graph and creates a summary node for
    each community with >= 3 members. Idempotent per run.
    """
    try:
        db_manager = get_db_manager()
        from engine.community_detector import run_community_detection

        result = await asyncio.to_thread(run_community_detection, db_manager)
        return {"status": "success", **result}
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Community detection failed"},
        )


# ==================== Ingest Helpers ====================


class SessionTurn(BaseModel):
    """A single conversation turn for fact extraction."""

    speaker: str = ""
    text: str
    date: str = ""


class SessionFactsRequest(BaseModel):
    """Request body for /ingest/session-facts."""

    session_id: str
    turns: List[SessionTurn]
    container_tag: Optional[str] = None


class SessionFactsResponse(BaseModel):
    """Response from /ingest/session-facts."""

    session_id: str
    facts_extracted: int
    node_ids: List[str]


# ---- Per-session fact-extraction cache (single-flight) -------------------------
# The memorybench harness re-ingests the SAME conversation sessions once per
# question, so without memoization each session's (slow, ~10-20s) gpt-4o fact
# extraction runs ~25x per conversation and overloads any configured provider.
# We memoize the extracted facts list (NOT node ids, since
# nodes must still be created per container_tag) keyed by the session content
# hash, with single-flight so concurrent ingests of the same session share one
# LLM call.
from collections import OrderedDict

_fact_cache: "OrderedDict[str, list]" = OrderedDict()
_fact_cache_tasks: Dict[str, asyncio.Task] = {}
_fact_cache_meta_lock = asyncio.Lock()
_fact_cache_generation = 0


def _canonical_fact_text(value: str) -> str:
    """Normalize only for identity; preserve the extracted text for storage."""
    return " ".join(value.casefold().split())


def _stable_fact_node_id(
    *, session_id: str, container_tag: Optional[str], fact_text: str
) -> str:
    """Return a retry-stable ID scoped to one session and one container."""
    import uuid

    identity = json.dumps(
        [container_tag or "", session_id, _canonical_fact_text(fact_text)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"hybridmind:extracted-fact:v1:{identity}")
    )


def _find_existing_fact_id(
    sqlite_store,
    *,
    stable_id: str,
    session_id: str,
    container_tag: Optional[str],
    fact_text: str,
) -> Optional[str]:
    """Find a current stable or legacy fact with the same scoped identity."""
    existing = sqlite_store.get_node(stable_id)
    if existing and not existing.get("deleted_at") and not existing.get("archived_at"):
        existing_metadata = existing.get("metadata") or {}
        existing_container = (
            existing_metadata.get("container_tag")
            or existing_metadata.get("containerTag")
            or None
        )
        if (
            existing_metadata.get("type") == "extracted_fact"
            and (
                existing_metadata.get("session_id")
                or existing_metadata.get("sessionId")
            )
            == session_id
            and existing_container == (container_tag or None)
            and _canonical_fact_text(str(existing.get("text", "")))
            == _canonical_fact_text(fact_text)
        ):
            return stable_id
        raise RuntimeError("stable extracted-fact ID is occupied by different content")

    with sqlite_store._cursor() as cursor:
        cursor.execute(
            """
            SELECT id, text, metadata
            FROM nodes
            WHERE deleted_at IS NULL AND archived_at IS NULL
              AND json_extract(metadata, '$.type') = 'extracted_fact'
              AND (
                    json_extract(metadata, '$.session_id') = ?
                    OR json_extract(metadata, '$.sessionId') = ?
              )
            ORDER BY id
            """,
            (session_id, session_id),
        )
        rows = cursor.fetchall()

    wanted_text = _canonical_fact_text(fact_text)
    wanted_container = container_tag or None
    for row in rows:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        candidate_container = (
            metadata.get("container_tag") or metadata.get("containerTag") or None
        )
        if candidate_container != wanted_container:
            continue
        if _canonical_fact_text(str(row["text"])) == wanted_text:
            return str(row["id"])
    return None


def _fact_conflict_plan(*, node_id: str, new_valid_from: str, prior: dict) -> dict:
    """Plan a conservative bi-temporal relation for one slot conflict."""
    from engine.temporal import parse_datetime

    prior_id = str(prior["id"])
    prior_from = (
        prior.get("valid_from") or prior.get("event_time") or prior.get("created_at")
    )
    new_dt = parse_datetime(new_valid_from)
    prior_dt = parse_datetime(prior_from)
    if new_dt is not None and prior_dt is not None and new_dt < prior_dt:
        return {
            "edge_type": "supersedes",
            "source_id": prior_id,
            "target_id": node_id,
            "valid_from": prior_from,
            "close_id": node_id,
            "closed_by": prior_id,
        }
    if new_dt is not None and prior_dt is not None and new_dt == prior_dt:
        return {
            "edge_type": "contradicts",
            "source_id": node_id,
            "target_id": prior_id,
            "valid_from": new_valid_from,
            "close_id": None,
            "closed_by": None,
        }
    return {
        "edge_type": "supersedes",
        "source_id": node_id,
        "target_id": prior_id,
        "valid_from": new_valid_from,
        "close_id": prior_id,
        "closed_by": node_id,
    }


def _fact_cache_key(turns_dicts: List[dict]) -> str:
    """Stable hash of the conversation turns (speaker/text/date)."""
    h = hashlib.sha256()
    for t in turns_dicts:
        h.update(
            (
                str(t.get("speaker", ""))
                + "\x1f"
                + str(t.get("text", ""))
                + "\x1f"
                + str(t.get("date", ""))
                + "\x1e"
            ).encode("utf-8")
        )
    return h.hexdigest()


async def _get_or_extract_facts(session_id: str, turns_dicts: List[dict]) -> list:
    """Return extracted facts for these turns with bounded LRU single-flight.

    Valid empty results are cached. Provider/malformed-output failures raise and
    are never cached, so they cannot be confused with a factless session.
    """
    from engine.fact_extractor import extract_facts_from_session

    key = _fact_cache_key(turns_dicts)

    global _fact_cache_generation
    async with _fact_cache_meta_lock:
        if key in _fact_cache:
            cached = _fact_cache[key]
            _fact_cache.move_to_end(key)
            logger.info("Fact cache hit facts=%d", len(cached))
            return cached
        task = _fact_cache_tasks.get(key)
        generation = _fact_cache_generation
        if task is None:
            task = asyncio.create_task(
                asyncio.to_thread(extract_facts_from_session, turns_dicts)
            )
            _fact_cache_tasks[key] = task

    try:
        facts = await asyncio.shield(task)
    except Exception as exc:
        logger.error("Fact extraction failed type=%s", type(exc).__name__)
        raise
    finally:
        async with _fact_cache_meta_lock:
            if task.done() and _fact_cache_tasks.get(key) is task:
                _fact_cache_tasks.pop(key, None)

    async with _fact_cache_meta_lock:
        if generation == _fact_cache_generation:
            _fact_cache[key] = facts
            _fact_cache.move_to_end(key)
            cache_limit = max(1, int(settings.fact_extraction_cache_max_entries))
            while len(_fact_cache) > cache_limit:
                _fact_cache.popitem(last=False)
    return facts


def clear_fact_cache() -> None:
    """Drop all memoized fact extractions (called on /admin/clear)."""
    global _fact_cache_generation
    _fact_cache_generation += 1
    _fact_cache.clear()


@app.post("/ingest/session-facts", response_model=SessionFactsResponse, tags=["Ingest"])
async def ingest_session_facts(
    request: SessionFactsRequest,
    mutation_guard: None = Depends(coordinate_mutation),
):
    """
    Extract facts from a conversation session using the configured policy-allowed model and store them
    as 'extracted_fact' nodes in HybridMind.

    Called ONCE per LoCoMo session at ingest time. Never at query time.

    Args:
        request.session_id: Unique session identifier
        request.turns: List of {speaker, text, date} dicts
        request.container_tag: Optional container/run tag for the nodes

    Returns:
        List of created node IDs for the extracted facts.
    """
    if not settings.fact_extraction_enabled:
        raise HTTPException(
            status_code=409, detail="Fact extraction is disabled by configuration"
        )
    from engine import llm_client as _llm_client

    if not _llm_client.is_configured():
        raise HTTPException(
            status_code=503,
            detail="No policy-allowed fact extraction provider is configured",
        )
    from storage.colbert_store import colbert_enabled

    if colbert_enabled():
        raise HTTPException(
            status_code=409,
            detail=(
                "ColBERT fact ingestion is disabled until it can participate in "
                "the same transaction and recovery protocol"
            ),
        )

    import uuid as _uuid
    import numpy as _np
    from engine.cache import invalidate_cache

    # Convert pydantic models to plain dicts for the extractor
    turns_dicts = [
        {"speaker": t.speaker, "text": t.text, "date": t.date} for t in request.turns
    ]

    # Run LLM-based fact extraction (ingest-time only), memoized per unique
    # session content with single-flight so the same session is never extracted
    # more than once regardless of how many questions re-ingest it.
    try:
        facts = await _get_or_extract_facts(request.session_id, turns_dicts)
    except Exception as exc:
        logger.error("Session fact extraction unavailable type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Fact extraction failed; no empty-result fallback was used.",
        ) from exc

    if not facts:
        return SessionFactsResponse(
            session_id=request.session_id, facts_extracted=0, node_ids=[]
        )

    db_manager = get_db_manager()
    node_ids: List[str] = []
    fact_id_by_text: Dict[str, str] = {}
    causal_links: List[tuple[str, str]] = []

    # A provider may repeat an item. Identity is exact normalized text within
    # the session/container; retaining both would make retries non-idempotent.
    valid_facts = []
    seen_fact_texts = set()
    for fact in facts:
        fact_text = str(fact.get("fact", "")).strip()
        canonical = _canonical_fact_text(fact_text)
        if not canonical or canonical in seen_fact_texts:
            continue
        seen_fact_texts.add(canonical)
        valid_facts.append(fact)
    if not valid_facts:
        return SessionFactsResponse(
            session_id=request.session_id, facts_extracted=0, node_ids=[]
        )

    fact_texts = [str(f["fact"]).strip() for f in valid_facts]
    fact_records = []
    new_records = []
    for fact, fact_text in zip(valid_facts, fact_texts):
        stable_id = _stable_fact_node_id(
            session_id=request.session_id,
            container_tag=request.container_tag,
            fact_text=fact_text,
        )
        existing_id = _find_existing_fact_id(
            db_manager.sqlite_store,
            stable_id=stable_id,
            session_id=request.session_id,
            container_tag=request.container_tag,
            fact_text=fact_text,
        )
        record = {
            "fact": fact,
            "text": fact_text,
            "id": existing_id or stable_id,
            "existing": existing_id is not None,
        }
        fact_records.append(record)
        if existing_id is None:
            new_records.append(record)

    new_fact_texts = [record["text"] for record in new_records]
    try:
        embeddings = (
            db_manager.embedding_engine.embed_batch(new_fact_texts)
            if new_fact_texts
            else _np.empty((0, 4096), dtype=_np.float32)
        )
        if embeddings.ndim != 2 or embeddings.shape != (len(new_fact_texts), 4096):
            raise ValueError(
                f"fact embedding batch has shape {embeddings.shape}; expected "
                f"({len(new_fact_texts)}, 4096)"
            )
        if not _np.all(_np.isfinite(embeddings)):
            raise ValueError("fact embedding batch contains non-finite values")
    except Exception as exc:
        logger.error(
            "Fact embedding failed; refusing fallback type=%s", type(exc).__name__
        )
        raise HTTPException(
            status_code=503,
            detail="Exact 4096-dimensional fact embedding failed; no fallback was used.",
        ) from exc

    # Find all raw turn nodes for this session
    session_turns = []
    try:
        with db_manager.sqlite_store._cursor() as cursor:
            cursor.execute(
                """
                SELECT id FROM nodes
                WHERE (
                    json_extract(metadata, '$.sessionId') = ?
                    OR json_extract(metadata, '$.session_id') = ?
                )
                AND (
                    (
                        ? IS NULL
                        AND COALESCE(
                            json_extract(metadata, '$.containerTag'), ''
                        ) = ''
                        AND COALESCE(
                            json_extract(metadata, '$.container_tag'), ''
                        ) = ''
                    )
                    OR (
                        ? IS NOT NULL
                        AND (
                            json_extract(metadata, '$.containerTag') = ?
                            OR json_extract(metadata, '$.container_tag') = ?
                        )
                    )
                )
                """,
                (
                    request.session_id,
                    request.session_id,
                    request.container_tag,
                    request.container_tag,
                    request.container_tag,
                    request.container_tag,
                ),
            )
            session_turns = [row[0] for row in cursor.fetchall()]
    except Exception as exc:
        logger.warning(
            "Failed to query session turns for edges type=%s", type(exc).__name__
        )
    # Build a complete mutation plan before touching authoritative or derived
    # state. Contradiction lookup is advisory; persistence itself is fail-closed.
    from datetime import datetime, timezone
    from engine.consolidation import check_contradiction
    from engine.edge_inference import run_auto_edge_inference
    from models.memory_pool import classify_memory_type

    asserted_at = datetime.now(timezone.utc).isoformat()
    prepared_facts = []
    for record, embedding in zip(new_records, embeddings):
        fact = record["fact"]
        fact_text = record["text"]
        node_id = record["id"]
        pool = classify_memory_type(fact_text)
        memory_kind = str(fact.get("memory_kind", "world")).lower()
        if memory_kind not in {"world", "experience", "observation", "opinion"}:
            memory_kind = "world"
        confidence = float(fact.get("confidence", 1.0))
        if not _np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise HTTPException(
                status_code=422,
                detail="Extracted fact confidence must be finite and within [0, 1]",
            )
        metadata = {
            "type": "extracted_fact",
            "session_id": request.session_id,
            "entities": fact.get("entities", []),
            "date": fact.get("date", ""),
            "event_time": fact.get("date", "") or None,
            "asserted_at": asserted_at,
            "memory_pool": pool.value,
            "memory_kind": memory_kind,
            "confidence": confidence,
        }
        effective_from = metadata["event_time"] or metadata["asserted_at"]
        if request.container_tag:
            metadata["container_tag"] = request.container_tag
            metadata["containerTag"] = request.container_tag

        contradicted_id = None
        try:
            candidates = db_manager.vector_index.search(
                _np.asarray(embedding, dtype=_np.float32), top_k=5
            )
            existing_nodes = []
            for cand_id, _score in candidates:
                node_data = db_manager.sqlite_store.get_node(cand_id)
                if not node_data or node_data.get("deleted_at"):
                    continue
                candidate_metadata = node_data.get("metadata") or {}
                if not isinstance(candidate_metadata, dict):
                    candidate_metadata = {}
                candidate_container = candidate_metadata.get(
                    "containerTag"
                ) or candidate_metadata.get("container_tag")
                if (candidate_container or None) != (request.container_tag or None):
                    continue
                from engine.temporal import parse_datetime

                candidate_until = parse_datetime(node_data.get("valid_until"))
                effective_dt = parse_datetime(effective_from)
                if (
                    candidate_until is not None
                    and effective_dt is not None
                    and candidate_until <= effective_dt
                ):
                    continue
                if candidate_metadata.get("type") == "extracted_fact":
                    existing_nodes.append(
                        {
                            "id": cand_id,
                            "text": node_data["text"],
                            "metadata": candidate_metadata,
                        }
                    )
            contradicted_id = check_contradiction(
                fact_text,
                existing_nodes,
                db_manager.embedding_engine,
                threshold=getattr(settings, "fact_contradiction_threshold", 0.85),
            )
        except Exception as exc:
            logger.warning(
                "Fact contradiction check unavailable type=%s", type(exc).__name__
            )

        prepared_facts.append(
            {
                "fact": fact,
                "text": fact_text,
                "embedding": embedding,
                "id": node_id,
                "metadata": metadata,
                "contradicted_id": contradicted_id,
            }
        )
        fact_id_by_text[_canonical_fact_text(fact_text)] = node_id
        for cause_text in fact.get("caused_by", []):
            if isinstance(cause_text, str) and cause_text.strip():
                causal_links.append(
                    (_canonical_fact_text(cause_text), _canonical_fact_text(fact_text))
                )

    for record in fact_records:
        node_ids.append(record["id"])
        fact_id_by_text[_canonical_fact_text(record["text"])] = record["id"]

    try:
        with db_manager.sqlite_store.transaction():
            for prepared in prepared_facts:
                node_id = prepared["id"]
                fact_text = prepared["text"]
                embedding = prepared["embedding"]
                metadata = prepared["metadata"]
                db_manager.sqlite_store.create_node(
                    node_id=node_id,
                    text=fact_text,
                    metadata=metadata,
                    embedding=embedding,
                    raw_embedding=embedding,
                    event_time=metadata["event_time"],
                    valid_from=metadata["event_time"] or metadata["asserted_at"],
                    memory_kind=metadata["memory_kind"],
                    confidence=metadata["confidence"],
                )
                db_manager.vector_index.add(node_id, embedding)
                db_manager.bm25_index.add(node_id, fact_text)
                db_manager.graph_index.add_node(
                    node_id,
                    event_time=metadata["event_time"],
                    memory_kind=metadata["memory_kind"],
                    confidence=metadata["confidence"],
                )

                contradicted_id = prepared["contradicted_id"]
                if contradicted_id:
                    prior = db_manager.sqlite_store.get_node(contradicted_id)
                    new_from = metadata["event_time"] or metadata["asserted_at"]
                    if prior is None:
                        raise RuntimeError(
                            "contradicted fact disappeared during ingestion"
                        )
                    plan = _fact_conflict_plan(
                        node_id=node_id,
                        new_valid_from=new_from,
                        prior=prior,
                    )
                    close_id = plan["close_id"]
                    if close_id is not None:
                        closing_node = db_manager.sqlite_store.get_node(close_id)
                        if closing_node is None:
                            raise RuntimeError(
                                "fact to close disappeared during ingestion"
                            )
                        closing_metadata = dict(closing_node.get("metadata") or {})
                        closing_metadata["superseded_by"] = plan["closed_by"]
                        db_manager.sqlite_store.update_node(
                            close_id,
                            metadata=closing_metadata,
                            valid_until=plan["valid_from"],
                        )

                    relation_type = plan["edge_type"]
                    relation_source = plan["source_id"]
                    relation_target = plan["target_id"]
                    relation_time = plan["valid_from"]
                    edge_id = str(
                        _uuid.uuid5(
                            _uuid.NAMESPACE_URL,
                            f"hybridmind:{relation_type}:{relation_source}:{relation_target}",
                        )
                    )
                    db_manager.sqlite_store.create_edge(
                        edge_id=edge_id,
                        source_id=relation_source,
                        target_id=relation_target,
                        edge_type=relation_type,
                        weight=1.0,
                        valid_from=relation_time,
                        confidence=1.0,
                    )
                    db_manager.graph_index.add_edge(
                        edge_id=edge_id,
                        source_id=relation_source,
                        target_id=relation_target,
                        edge_type=relation_type,
                        weight=1.0,
                        valid_from=relation_time,
                        confidence=1.0,
                    )

                for turn_id in session_turns:
                    edge_id = str(
                        _uuid.uuid5(
                            _uuid.NAMESPACE_URL,
                            f"hybridmind:belongs_to:{node_id}:{turn_id}",
                        )
                    )
                    db_manager.sqlite_store.create_edge(
                        edge_id=edge_id,
                        source_id=node_id,
                        target_id=turn_id,
                        edge_type="belongs_to",
                        weight=1.0,
                    )
                    db_manager.graph_index.add_edge(
                        edge_id=edge_id,
                        source_id=node_id,
                        target_id=turn_id,
                        edge_type="belongs_to",
                        weight=1.0,
                    )

                run_auto_edge_inference(
                    node_id=node_id,
                    embedding=embedding,
                    node_metadata=metadata,
                    node_text=fact_text,
                    vector_index=db_manager.vector_index,
                    sqlite_store=db_manager.sqlite_store,
                    graph_index=db_manager.graph_index,
                    event_time=metadata["event_time"],
                )

            if settings.causal_edges_enabled:
                for cause_text, effect_text in causal_links:
                    cause_id = fact_id_by_text.get(cause_text)
                    effect_id = fact_id_by_text.get(effect_text)
                    if not cause_id or not effect_id or cause_id == effect_id:
                        continue
                    edge_id = str(
                        _uuid.uuid5(
                            _uuid.NAMESPACE_URL,
                            f"hybridmind:led_to:{cause_id}:{effect_id}",
                        )
                    )
                    edge_metadata = {"inferred_by": "structured_fact_extractor"}
                    db_manager.sqlite_store.create_edge(
                        edge_id=edge_id,
                        source_id=cause_id,
                        target_id=effect_id,
                        edge_type="led_to",
                        weight=0.9,
                        metadata=edge_metadata,
                        confidence=0.9,
                    )
                    db_manager.graph_index.add_edge(
                        edge_id=edge_id,
                        source_id=cause_id,
                        target_id=effect_id,
                        edge_type="led_to",
                        weight=0.9,
                        confidence=0.9,
                        **edge_metadata,
                    )
    except Exception as exc:
        logger.error("Fact persistence failed type=%s", type(exc).__name__)
        try:
            db_manager._rebuild_indexes()
        except Exception as rebuild_exc:
            logger.critical(
                "Fact persistence recovery failed type=%s",
                type(rebuild_exc).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail="Fact persistence failed and index recovery did not complete",
            ) from rebuild_exc
        raise HTTPException(
            status_code=500,
            detail="Fact persistence failed; no authoritative rows were committed",
        ) from exc

    invalidate_cache()

    logger.info(
        f"session-facts: session={request.session_id}, "
        f"turns={len(turns_dicts)}, facts={len(node_ids)}, "
        f"new_nodes={len(prepared_facts)}"
    )

    return SessionFactsResponse(
        session_id=request.session_id,
        facts_extracted=len(node_ids),
        node_ids=node_ids,
    )


def start():
    import uvicorn

    _validate_api_security_configuration()
    if settings.debug:
        uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
    else:
        uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    start()
