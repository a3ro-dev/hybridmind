"""
FastAPI dependencies for HybridMind.
Provides singleton instances of storage and engine components.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import AsyncGenerator, Generator, Iterator
from pathlib import Path

from config import settings
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from storage.mindfile import MindFile
from storage.bm25_index import sparse_document_text
from storage.colbert_store import ColbertStore, colbert_enabled
from engine.device import resolve_device
from engine.embedding import EmbeddingEngine, get_embedding_engine as get_embedding_backend
from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine
from engine.hybrid_ranker import HybridRanker
from engine.reranker import get_reranker

logger = logging.getLogger(__name__)


class ProcessMutationCoordinator:
    """Serialize authoritative SQL and derived-index mutations in this process.

    A ``threading.RLock`` cannot protect async request tasks: two tasks run on
    the same event-loop thread and would therefore appear re-entrant while the
    first task is suspended.  This bridge uses a non-reentrant process lock and
    acquires it asynchronously without blocking the event loop.  Callers must
    take the guard only at the outermost operation boundary.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def sync(self) -> Iterator[None]:
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    @asynccontextmanager
    async def async_(self) -> AsyncGenerator[None, None]:
        # A non-blocking attempt plus an async backoff is cancellation-safe: a
        # cancelled waiter can never acquire the lock in an abandoned worker.
        while not self._lock.acquire(blocking=False):
            await asyncio.sleep(0.001)
        try:
            yield
        finally:
            self._lock.release()


class DatabaseManager:
    """
    Singleton manager for all database components.
    Ensures components are initialized once and reused.
    """
    
    _instance = None
    _initialized = False
    _snapshot_lock = threading.Lock()
    _mutation_coordinator = ProcessMutationCoordinator()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info("Initializing HybridMind database components...")
        
        # Initialize .mind file (HybridMind's native format)
        self.mind_file = MindFile(settings.mind_file_path)
        if not self.mind_file.exists:
            logger.info(f"Creating new .mind database: {settings.mind_file_path}")
            self.mind_file.initialize(metadata={
                "description": "HybridMind Vector + Graph Database",
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension
            })
        
        # Get paths from .mind file
        paths = self.mind_file.get_paths()
        
        # Initialize storage components using .mind paths
        self.sqlite_store = SQLiteStore(paths["sqlite"])
        self.vector_index = VectorIndex(
            dimension=settings.embedding_dimension,
            hnsw_ef_search=settings.hnsw_ef_search,
            hnsw_ef_construction=settings.hnsw_ef_construction,
            # SQLite is the trusted source of truth.  Persisted index pickles
            # are deliberately not loaded; derived indexes are rebuilt below.
            index_path=None
        )
        
        from storage.bm25_index import create_sparse_index
        self.bm25_index = create_sparse_index(
            backend=settings.sparse_retrieval_backend,
            index_path=None,
        )
        self.graph_index = GraphIndex(index_path=None)
        
        # ColBERT per-token vectors (opt-in, off by default)
        if colbert_enabled():
            self.colbert_store = ColbertStore(root_dir=paths["root"])
            logger.info(f"ColBERT store initialized at {paths['root']}/colbert/")
        else:
            self.colbert_store = None
        
        # Visual memory store
        self.visual_store = ColbertStore(root_dir=str(Path(paths["root"]) / "visual"))
        logger.info(f"Visual memory store initialized at {paths['root']}/visual/")
        
        # Initialize embedding engine with resolved device
        _device = resolve_device(settings.device)
        self.embedding_engine = get_embedding_backend(
            model_name=settings.embedding_model,
            device=_device,
        )
        
        # Sanity-check: configured embedding dim must match vector index dim.
        # Mismatch = different model was used to build the stored index.
        # Run: python scripts/reindex_embeddings.py  to rebuild.
        _emb_dim = settings.embedding_dimension
        _idx_dim = self.vector_index.dimension
        if _emb_dim != _idx_dim:
            raise RuntimeError(
                f"Embedding model dimension ({_emb_dim}) does not match the stored "
                f"vector index dimension ({_idx_dim}). The model was changed without "
                f"re-indexing. Run:  python scripts/reindex_embeddings.py  to rebuild."
            )

        # Initialize search engines
        self.vector_engine = VectorSearchEngine(
            vector_index=self.vector_index,
            sqlite_store=self.sqlite_store,
            embedding_engine=self.embedding_engine
        )
        
        self.graph_engine = GraphSearchEngine(
            graph_index=self.graph_index,
            sqlite_store=self.sqlite_store
        )
        
        # Initialize the optional reranker without downloading/loading a large
        # model unless deployment explicitly requests startup warmup.
        reranker = get_reranker()
        if settings.reranker_warmup_enabled and getattr(reranker, "enabled", False):
            try:
                reranker.warmup()
            except Exception as exc:
                logger.warning(
                    "Reranker warmup failed; disabling it for this process (%s)",
                    type(exc).__name__,
                )
                reranker = None

        # Initialize hybrid ranker
        self.hybrid_ranker = HybridRanker(
            vector_engine=self.vector_engine,
            graph_engine=self.graph_engine,
            bm25_index=self.bm25_index,
            reranker=reranker,
        )

        # Rebuild indexes from SQLite on startup
        self._rebuild_indexes()
        
        self._initialized = True
        logger.info("HybridMind database components initialized successfully")
    
    def _rebuild_indexes(self):
        """Fully replace every derived index from authoritative live SQLite rows."""
        try:
            nodes = self.sqlite_store.list_nodes(limit=1000000)
            live_ids = {node["id"] for node in nodes}
            embeddings = self.sqlite_store.get_all_node_embeddings(
                include_archived=False
            )
            self.vector_index.rebuild_from_embeddings(embeddings)
            logger.info("Vector index rebuilt with %d nodes", len(embeddings))

            edges = [
                edge
                for edge in self.sqlite_store.get_all_edges()
                if edge["source_id"] in live_ids and edge["target_id"] in live_ids
            ]
            self.graph_index.rebuild_from_edges(edges)
            logger.info("Graph index rebuilt with %d edges", len(edges))

            bm25_batch = []
            for node in nodes:
                if not self.graph_index.has_node(node["id"]):
                    self.graph_index.add_node(
                        node["id"],
                        event_time=node.get("event_time"),
                        memory_kind=node.get("memory_kind"),
                        confidence=node.get("confidence", 1.0),
                    )
                bm25_batch.append((
                    node["id"],
                    sparse_document_text(node["text"], node.get("metadata")),
                ))
            
            self.bm25_index.clear()
            self.bm25_index.add_batch(bm25_batch)
            logger.info(f"BM25 index rebuilt with {len(bm25_batch)} documents")

        except Exception as exc:
            raise RuntimeError(
                "Index rebuild failed. HybridMind will not start with a partial or "
                "dimension-inconsistent 4096-dimensional index "
                f"({type(exc).__name__})"
            ) from exc
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        return {
            "total_nodes": self.sqlite_store.count_nodes(),
            "total_edges": self.sqlite_store.count_edges(),
            "edge_types": self.sqlite_store.get_edge_type_counts(),
            "vector_index_size": self.vector_index.size,
            "graph_node_count": self.graph_index.node_count,
            "graph_edge_count": self.graph_index.edge_count,
            "database_size_bytes": self.sqlite_store.get_database_size(),
            "bm25_index_size": self.bm25_index.size,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension
        }

    def mutation(self):
        """Return the synchronous process-wide mutation guard."""
        return self._mutation_coordinator.sync()

    def mutation_async(self):
        """Return the asynchronous process-wide mutation guard."""
        return self._mutation_coordinator.async_()
    
    def save_indexes(self):
        """Create and return a verified snapshot; failures propagate to callers."""
        with self.mutation():
            with self._snapshot_lock:
                stats = self.get_stats()
                snapshot = self.mind_file.create_snapshot(
                    sqlite_conn=self.sqlite_store._get_connection(),
                    vector_index=self.vector_index,
                    graph_index=self.graph_index,
                    nodes_count=stats["total_nodes"],
                    edges_count=stats["total_edges"],
                    backup_dir=settings.backup_dir,
                )

                backup_dir = Path(settings.backup_dir)
                backups = sorted(backup_dir.glob("snapshot_*.mind.zip"))
                retention = max(1, settings.snapshot_retention)
                for old_backup in backups[:-retention]:
                    old_backup.unlink(missing_ok=True)
        logger.info("Verified snapshot created: %s", snapshot.name)
        return snapshot
    
    def close(self, save: bool = True):
        """Close all connections."""
        try:
            if save:
                self.save_indexes()
        finally:
            self.sqlite_store.close()
            logger.info("Database connections closed")


# Singleton instance
_db_manager: DatabaseManager = None


def get_db_manager() -> DatabaseManager:
    """Get the database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def coordinate_mutation() -> AsyncGenerator[None, None]:
    """FastAPI dependency holding the process mutation guard for one request."""
    async with get_db_manager().mutation_async():
        yield


# Dependency injection functions for FastAPI
def get_sqlite_store() -> SQLiteStore:
    """FastAPI dependency for SQLite store."""
    return get_db_manager().sqlite_store


def get_bm25_index():
    """FastAPI dependency for sparse index (BM25Index / BM25SBackend / SpladeBackend)."""
    return get_db_manager().bm25_index

def get_vector_index() -> VectorIndex:
    """FastAPI dependency for vector index."""
    return get_db_manager().vector_index


def get_graph_index() -> GraphIndex:
    """FastAPI dependency for graph index."""
    return get_db_manager().graph_index


def get_embedding_engine() -> EmbeddingEngine:
    """FastAPI dependency for embedding engine."""
    return get_db_manager().embedding_engine


def get_vector_engine() -> VectorSearchEngine:
    """FastAPI dependency for vector search engine."""
    return get_db_manager().vector_engine


def get_graph_engine() -> GraphSearchEngine:
    """FastAPI dependency for graph search engine."""
    return get_db_manager().graph_engine


def get_hybrid_ranker() -> HybridRanker:
    """FastAPI dependency for hybrid ranker."""
    return get_db_manager().hybrid_ranker


def get_colbert_store():
    """FastAPI dependency for ColBERT vector store (None when disabled)."""
    return get_db_manager().colbert_store if hasattr(get_db_manager(), 'colbert_store') else None


def get_visual_store() -> ColbertStore:
    """FastAPI dependency for Visual memory store."""
    return get_db_manager().visual_store

