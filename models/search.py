"""Search-related Pydantic models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

from config import settings


class VectorSearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filter_metadata: Optional[Dict[str, Any]] = None


class GraphSearchRequest(BaseModel):
    start_id: str
    depth: int = Field(default=2, ge=1, le=5)
    edge_types: Optional[List[str]] = None
    direction: str = "both"


class HybridSearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)
    vector_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    graph_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    anchor_nodes: Optional[List[str]] = None
    max_depth: int = Field(default=2, ge=1, le=5)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filter_metadata: Optional[Dict[str, Any]] = None
    deduplicate: bool = Field(default=True, description="Deduplicate results with identical text")
    rerank_pool: int = Field(default_factory=lambda: settings.rerank_pool_size, ge=0, le=100, description="Maximum candidate pool fed to the cross-encoder reranker; 0 disables that stage. A positive value must be at least top_k.")
    bm25_boost_weight: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sparse RRF weight; when omitted, query routing may select it")
    overlap_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="BM25 overlap fraction below which graph score is ramped down")
    fusion_mode: Optional[str] = Field(
        default=None,
        pattern="^(rrf|linear|mlp)$",
        description="Override config fusion_mode: rrf, linear, or mlp",
    )
    include_images: bool = Field(default=False, description="Include relevant visual memory images ranked by ColQwen2.5 MaxSim")
    search_mode: str = Field(
        default="hybrid",
        pattern="^(hybrid|vector_only|sparse_only|vector_sparse|graph_only)$",
        description="Controlled retrieval mode for production queries and ablations",
    )
    route_weights: bool = Field(default=True, description="Apply query-type routing weights")
    track_access: Optional[bool] = Field(default=None, description="Override config access tracking for this request")

    @model_validator(mode="after")
    def validate_cross_field_contracts(self):
        if self.search_mode == "graph_only" and not self.anchor_nodes:
            raise ValueError("graph_only search requires at least one anchor node")
        if 0 < self.rerank_pool < self.top_k:
            raise ValueError("positive rerank_pool must be greater than or equal to top_k")
        return self


class SearchResult(BaseModel):
    node_id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    graph_gate: Optional[float] = None
    effective_graph_score: Optional[float] = None
    combined_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_attempted: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_failure_type: Optional[str] = None
    bm25_score: Optional[float] = None
    time_score: Optional[float] = None
    salience_score: Optional[float] = None
    depth: Optional[int] = None
    path: Optional[List[str]] = None
    reasoning: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float
    total_candidates: int
    search_type: str


class StatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    edge_types: Dict[str, int]
    avg_edges_per_node: float
    vector_index_size: int
    database_size_bytes: int
