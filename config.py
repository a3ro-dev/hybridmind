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
    
    # Device (auto = cuda > mps > cpu)
    device: str = "auto"
    use_faiss_gpu: bool = False  # opt-in; Linux/Docker only (no Windows wheel)

    # Embedding: self-hosted RunPod TEI (Qwen3-Embedding-8B, native 4096-dim)
    # is now the primary backend when RUNPOD_TEI_EMBEDDING_URL is set (see
    # engine/embedding.py). embedding_model/embedding_dimension below are the
    # *local fallback* config — bge-m3 is always 1024-dim regardless of this
    # setting, so if the TEI endpoint is down and local fallback kicks in,
    # vectors will mismatch a 4096-dim FAISS index. Re-run
    # scripts/reindex_embeddings.py after changing either backend's dimension.
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 4096
    use_graph_conditioned_embeddings: bool = True
    embedding_timeout_seconds: int = 30
    embedding_batch_size: int = 32  # per-batch size for model.encode()

    # Search Defaults (tuned for LoCoMo-style factoid queries)
    default_top_k: int = 10
    default_vector_weight: float = 0.5
    default_graph_weight: float = 0.15
    max_traversal_depth: int = 5

    # Fusion (rrf | linear)
    fusion_mode: str = "rrf"
    fusion_rrf_k: int = 60  # RRF constant; higher = smoother rank penalty

    # Fusion MLP (Phase 2 post-training) — set to checkpoint path to enable
    fusion_model_path: Optional[str] = None

    # Reranker model (used by CrossEncoderReranker)
    # mxbai-rerank-large-v2: Apache 2.0, ~84% Hit@1 vs 77% bge-reranker-v2-m3, 8x faster
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v2"

    # Sparse retrieval backend: "bm25" (pure Python, no deps) | "bm25s" (100x faster, needs bm25s+PyStemmer) | "splade" (needs fastembed)
    sparse_retrieval_backend: str = "bm25s"

    # Query routing: classify query type → apply per-type vector/graph/bm25 weights
    query_routing_enabled: bool = True

    # Temporal decay: weight graph edges by recency (exp decay on created_at)
    temporal_decay_enabled: bool = False   # set True to activate; tune half_life_days per use case
    temporal_decay_half_life_days: float = 30.0  # 7 for conversation, 90 for domain knowledge

    # Auto-edge inference (Phase 3)
    auto_edges_enabled: bool = False
    # 0.70/10 (lowered from 0.75/5): ablation showed too-sparse auto-edges yield
    # zero graph signal on graph-dependent queries — a slightly looser threshold
    # and higher per-node cap keeps precision reasonable while actually producing
    # enough edges for graph traversal/proximity scoring to matter.
    auto_edge_cosine_threshold: float = 0.70
    auto_edge_max_per_node: int = 10
    auto_edge_entity_enabled: bool = False  # requires spaCy or fact entities

    # Opt-in research modules (Phase 3)
    colbert_enabled: bool = False   # ColBERT MaxSim re-rank; needs bge-m3 colbert vecs
    gnn_enabled: bool = False       # GNN reranker; needs torch-geometric
    gnn_model_path: Optional[str] = None  # path to trained GNN checkpoint (.pt)
    
    # Memory lifecycle & Visual memory
    fact_contradiction_threshold: float = 0.85
    image_embedding_url: Optional[str] = None

    # Performance
    batch_size: int = 32            # legacy alias; use embedding_batch_size for new code
    cache_size: int = 1000
    
    # API
    host: str = "0.0.0.0"
    port: int = 8000

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

