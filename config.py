"""
Configuration management for HybridMind.
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix="HYBRIDMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application
    app_name: str = "HybridMind"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database (.mind format)
    # The .mind extension is HybridMind's native database format
    # It bundles SQLite + FAISS + NetworkX into a single directory
    mind_file_path: str = "data/hybridmind.mind"
    
    # Legacy paths (for backward compatibility)
    database_path: str = "data/hybridmind.mind/store.db"
    vector_index_path: str = "data/hybridmind.mind/vectors"
    graph_index_path: str = "data/hybridmind.mind/graph.nx"
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    use_graph_conditioned_embeddings: bool = True
    embedding_timeout_seconds: int = 30
    
    # Search Defaults
    default_top_k: int = 10
    default_vector_weight: float = 0.6
    default_graph_weight: float = 0.4
    max_traversal_depth: int = 5
    
    # Performance
    batch_size: int = 32
    cache_size: int = 1000
    
    # API
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Neo4j (for comparison)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # ChromaDB (for comparison)
    chromadb_path: str = "data/chromadb"
    chromadb_collection: str = "arxiv_papers"
    
    def get_data_dir(self) -> Path:
        """Get the data directory, creating it if necessary."""
        data_dir = Path(self.database_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_database_url(self) -> str:
        """Get the SQLite database URL."""
        return f"sqlite+aiosqlite:///{self.database_path}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()

