"""
FastAPI dependencies for HybridMind.
Provides singleton instances of storage and engine components.
"""

import logging
from functools import lru_cache
from typing import Generator
from pathlib import Path

from config import settings
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from storage.mindfile import MindFile
from engine.embedding import EmbeddingEngine
from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine
from engine.hybrid_ranker import HybridRanker

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Singleton manager for all database components.
    Ensures components are initialized once and reused.
    """
    
    _instance = None
    _initialized = False
    
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
            index_path=paths["vector_index"]
        )
        
        from storage.bm25_index import BM25Index
        self.bm25_index = BM25Index(
            index_path=str(Path(paths["root"]) / "bm25.pkl")
        )
        self.graph_index = GraphIndex(index_path=paths["graph"])
        
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine(
            model_name=settings.embedding_model
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
        
        # Initialize hybrid ranker
        self.hybrid_ranker = HybridRanker(
            vector_engine=self.vector_engine,
            graph_engine=self.graph_engine,
            bm25_index=self.bm25_index
        )
        
        # Rebuild indexes from SQLite on startup
        self._rebuild_indexes()
        
        self._initialized = True
        logger.info("HybridMind database components initialized successfully")
    
    def _rebuild_indexes(self):
        """Rebuild vector and graph indexes from SQLite."""
        try:
            # Rebuild vector index
            embeddings = self.sqlite_store.get_all_node_embeddings()
            if embeddings:
                self.vector_index.rebuild_from_embeddings(embeddings)
                logger.info(f"Vector index rebuilt with {len(embeddings)} nodes")
            
            # Rebuild graph index
            edges = self.sqlite_store.get_all_edges()
            if edges:
                self.graph_index.rebuild_from_edges(edges)
                logger.info(f"Graph index rebuilt with {len(edges)} edges")
            
            # Add orphan nodes to graph
            nodes = self.sqlite_store.list_nodes(limit=10000)
            bm25_batch = []
            for node in nodes:
                if not self.graph_index.has_node(node["id"]):
                    self.graph_index.add_node(node["id"])
                bm25_batch.append((node["id"], node["text"]))
            
            self.bm25_index.clear()
            self.bm25_index.add_batch(bm25_batch)
            logger.info(f"BM25 index rebuilt with {len(bm25_batch)} documents")
                    
        except Exception as e:
            logger.error(f"Error rebuilding indexes: {e}")
    
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
    
    def save_indexes(self):
        """Save indexes to disk and update .mind manifest."""
        try:
            stats = self.get_stats()
            self.bm25_index.save()
            self.mind_file.create_snapshot(
                sqlite_conn=self.sqlite_store._get_connection(),
                vector_index=self.vector_index,
                graph_index=self.graph_index,
                nodes_count=stats["total_nodes"],
                edges_count=stats["total_edges"]
            )
            
            # Backup rotation
            import os
            from pathlib import Path
            from datetime import datetime
            
            backup_dir = Path("data/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = str(backup_dir / f"snapshot_{timestamp}")
            self.mind_file.export(export_path, compress=True)
            
            # Keep last 3
            backups = sorted(backup_dir.glob("snapshot_*.mind.zip"))
            if len(backups) > 3:
                for old_backup in backups[:-3]:
                    old_backup.unlink(missing_ok=True)
                    
            logger.info("Indexes saved to disk and backed up")
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")
    
    def close(self):
        """Close all connections."""
        self.save_indexes()
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


# Dependency injection functions for FastAPI
def get_sqlite_store() -> SQLiteStore:
    """FastAPI dependency for SQLite store."""
    return get_db_manager().sqlite_store


from storage.bm25_index import BM25Index

def get_bm25_index() -> BM25Index:
    """FastAPI dependency for BM25 index."""
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

