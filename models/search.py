"""Search-related Pydantic models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


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
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    graph_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    anchor_nodes: Optional[List[str]] = None
    max_depth: int = Field(default=2, ge=1, le=5)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filter_metadata: Optional[Dict[str, Any]] = None
    deduplicate: bool = Field(default=True, description="Deduplicate results with identical text")
    rerank_pool: int = Field(default=25, ge=1, le=100, description="Candidate pool size fed to the reranker before slicing top_k; set <= top_k to disable reranking")
    bm25_boost_weight: float = Field(default=0.35, ge=0.0, le=2.0, description="BM25 keyword overlap boost applied on top of vector score")
    overlap_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="BM25 overlap fraction below which graph score is ramped down")
    fusion_mode: Optional[str] = Field(default=None, description="Override config fusion_mode: 'rrf' or 'linear'")
    include_images: bool = Field(default=False, description="Include relevant visual memory images ranked by ColQwen2.5 MaxSim")


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
