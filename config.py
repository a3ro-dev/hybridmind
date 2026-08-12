"""
Configuration management for HybridMind.
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
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
    # is the primary and ONLY backend when HYBRIDMIND_EMBEDDING_DIMENSION=4096.
    # There is no local or lower-dimensional fallback. The server refuses to
    # start without a remote backend capable of exact 4096-dimensional output.
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
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
    rerank_mode: str = Field(
        default="cross",
        validation_alias=AliasChoices("RERANK_MODE", "HYBRIDMIND_RERANK_MODE"),
    )
    rerank_rrf_weight: float = 0.70
    rerank_cross_encoder_weight: float = 0.30

    # Sparse retrieval backend: "bm25" (pure Python, no deps) | "bm25s" (100x faster, needs bm25s+PyStemmer) | "splade" (needs fastembed)
    sparse_retrieval_backend: str = "bm25s"

    # Query routing: classify query type → apply per-type vector/graph/bm25 weights
    query_routing_enabled: bool = True
    query_time_expansion_enabled: bool = True
    query_decomposition_model: Optional[str] = None

    # Query-type routing weights. Keeping these in Settings makes experiment
    # configurations explicit and hashable instead of hiding them in ranker code.
    route_temporal_vector_weight: float = 0.30
    route_temporal_graph_weight: float = 0.35
    route_temporal_sparse_weight: float = 0.15
    route_temporal_time_weight: float = 0.20
    route_multihop_vector_weight: float = 0.20
    route_multihop_graph_weight: float = 0.70
    route_multihop_sparse_weight: float = 0.10
    route_entity_vector_weight: float = 0.35
    route_entity_graph_weight: float = 0.25
    route_entity_sparse_weight: float = 0.40
    route_default_vector_weight: float = 0.50
    route_default_graph_weight: float = 0.15
    route_default_sparse_weight: float = 0.35

    # Multi-hop query decomposition (Phase 6.2.2). The live hybrid ranker and
    # evaluation harness both use this flag; it remains opt-in because it adds
    # a provider call and changes the candidate-generation protocol.
    query_decomposition_enabled: bool = False
    rerank_pool_size: int = 25  # Phase 6.2.4 candidate pool size knob

    # Lightweight query-local lexical reranking. The 500-candidate LoCoMo
    # checkpoint replay improved exact-source Recall@10 at ~61ms mean cost.
    local_lexical_rerank_enabled: bool = True
    local_lexical_rerank_weight: float = 0.5
    local_lexical_rerank_pool_size: int = 500
    local_lexical_term_cache_size: int = 20_000

    # Temporal decay: weight graph edges by recency (exp decay on created_at)
    temporal_decay_enabled: bool = False   # set True to activate; tune half_life_days per use case
    temporal_decay_half_life_days: float = 30.0  # 7 for conversation, 90 for domain knowledge
    temporal_edges_enabled: bool = False
    temporal_edge_window_days: float = 30.0
    temporal_edge_half_life_days: float = 7.0
    temporal_edge_max_per_node: int = 5

    # Access/salience scoring is opt-in so benchmark runs remain bit-identical
    # unless the experiment deliberately enables stateful retrieval.
    access_tracking_enabled: bool = False
    salience_enabled: bool = False
    salience_weight: float = 0.10
    salience_recency_half_life_days: float = 30.0
    salience_access_half_life_days: float = 14.0
    salience_recency_weight: float = 0.45
    salience_frequency_weight: float = 0.35
    salience_centrality_weight: float = 0.20

    # Auto-edge inference (Phase 3)
    auto_edges_enabled: bool = False
    # 0.70/10 (lowered from 0.75/5): ablation showed too-sparse auto-edges yield
    # zero graph signal on graph-dependent queries — a slightly looser threshold
    # and higher per-node cap keeps precision reasonable while actually producing
    # enough edges for graph traversal/proximity scoring to matter.
    auto_edge_cosine_threshold: float = 0.70
    auto_edge_max_per_node: int = 10
    auto_edge_entity_enabled: bool = False  # requires spaCy or fact entities
    causal_edges_enabled: bool = False

    # Opt-in research modules (Phase 3)
    colbert_enabled: bool = False   # ColBERT MaxSim re-rank; needs bge-m3 colbert vecs
    gnn_enabled: bool = False       # GNN reranker; needs torch-geometric
    gnn_model_path: Optional[str] = None  # path to trained GNN checkpoint (.pt)
    
    # Memory lifecycle & Visual memory
    fact_contradiction_threshold: float = 0.85
    image_embedding_url: Optional[str] = None

    # Structured ingestion and observer/reflector lifecycle. Z.AI is the
    # production hosted provider; RunPod is self-hosted; Hack Club is available
    # only under the explicit research-proxy gate below.
    fact_extraction_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "FACT_EXTRACTION_ENABLED", "HYBRIDMIND_FACT_EXTRACTION_ENABLED"
        ),
    )
    fact_model: str = "glm-4.6"
    consolidation_model: str = "glm-4.6"
    memory_compression_enabled: bool = False
    memory_compression_archive_sources: bool = False
    memory_compression_min_facts: int = 5
    memory_compression_max_age_hours: int = 24
    memory_compression_interval_seconds: int = 3600

    # Provider configuration is centralized here. AliasChoices preserves the
    # canonical unprefixed deployment variables documented by the project.
    zai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ZAI_API_KEY", "HYBRIDMIND_ZAI_API_KEY"),
    )
    zai_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        validation_alias=AliasChoices("ZAI_BASE_URL", "HYBRIDMIND_ZAI_BASE_URL"),
    )
    qa_model: str = "glm-4.6"
    runpod_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RUNPOD_API_KEY", "HYBRIDMIND_RUNPOD_API_KEY"),
    )
    runpod_llm_endpoint_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            "RUNPOD_LLM_ENDPOINT_ID", "HYBRIDMIND_RUNPOD_LLM_ENDPOINT_ID"
        ),
    )
    runpod_llm_model: str = Field(
        default="qwen/qwen3.5-9b",
        validation_alias=AliasChoices("RUNPOD_LLM_MODEL", "HYBRIDMIND_RUNPOD_LLM_MODEL"),
    )
    allow_research_proxy: bool = False
    research_proxy_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "HC_API_KEY", "HACKCLUB_API_KEY", "HYBRIDMIND_RESEARCH_PROXY_API_KEY"
        ),
    )
    research_proxy_base_url: str = "https://ai.hackclub.com/proxy/v1"
    research_proxy_model: str = "qwen/qwen3.5-9b"

    # Performance
    batch_size: int = 32            # legacy alias; use embedding_batch_size for new code
    cache_size: int = 1000
    
    # API
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("embedding_dimension")
    @classmethod
    def require_4096_dimensions(cls, value: int) -> int:
        if value != 4096:
            raise ValueError(
                "HybridMind requires embedding_dimension=4096; no lower-dimensional "
                "or local fallback is supported."
            )
        return value

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

