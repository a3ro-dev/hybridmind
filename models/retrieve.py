"""Request/response models for tri-signal retrieval (``POST /retrieve``)."""

import math
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Channel = Literal["dense", "sparse", "graph"]
ScopeValue = Union[str, int, float, bool]


class EvidenceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["turn", "window", "propagate", "session"] = "turn"
    window: int = Field(default=1, ge=0, le=10)
    propagation_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    budget_tokens: Optional[int] = Field(default=None, ge=1, le=200_000)
    order: Literal["chronological", "rank"] = "chronological"


class RetrieveRequest(BaseModel):
    """Omitted fields take the server defaults in ``config.py`` (trisignal_*)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=20_000)
    scope: Dict[str, ScopeValue] = Field(
        default_factory=dict,
        description="Exact-match metadata filters that define the retrieval scope "
        "(e.g. {'containerTag': 'user-42'} or {'session_id': 's1'}).",
    )
    channels: Optional[List[Channel]] = None
    top_k: int = Field(default=10, ge=1, le=200)
    channel_k: Optional[int] = Field(default=None, ge=1, le=1000)
    fusion: Optional[Literal["rrf", "dbsf", "zscore", "minmax_linear"]] = None
    rrf_k: Optional[int] = Field(default=None, ge=0, le=1000)
    weights: Optional[Dict[Channel, float]] = None
    dense_mode: Optional[Literal["exact", "hnsw"]] = None
    ann_audit: bool = False
    graph_method: Optional[Literal["emg", "ppr"]] = None
    graph_extractor: Optional[str] = Field(default=None, max_length=200)
    query_keys: Optional[List[str]] = Field(
        default=None,
        description="Caller-supplied graph anchor keys; default derives them from the query text.",
    )
    ppr_passage_seed_channel: Optional[Literal["dense", "sparse"]] = None
    rerank_pool: int = Field(default=0, ge=0, le=100)
    evidence: Optional[EvidenceOptions] = None
    as_of: Optional[datetime] = None

    @field_validator("weights")
    @classmethod
    def _finite_weights(cls, value):
        if value is None:
            return value
        for name, weight in value.items():
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"weight for {name!r} must be finite and non-negative")
        return value

    @field_validator("scope")
    @classmethod
    def _scope_keys(cls, value):
        for key in value:
            if not key or not key.replace("_", "").isalnum() or not key.isascii():
                raise ValueError(f"unsupported scope key {key!r}")
        return value

    @field_validator("channels")
    @classmethod
    def _unique_channels(cls, value):
        if value is not None and (not value or len(set(value)) != len(value)):
            raise ValueError("channels must be a non-empty list without repeats")
        return value

    @field_validator("as_of")
    @classmethod
    def _aware(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("as_of must include an explicit timezone offset")
        return value

    @model_validator(mode="after")
    def _rerank_pool(self):
        if 0 < self.rerank_pool < self.top_k:
            raise ValueError("a positive rerank_pool must be >= top_k (0 disables reranking)")
        return self


class RetrievedHit(BaseModel):
    node_id: str
    evidence_id: str
    text: str
    metadata: Dict[str, Any]
    rank: int
    score: float
    sources: List[Channel]
    channel_ranks: Dict[str, Optional[int]]
    channel_scores: Dict[str, Optional[float]]
    rerank_score: Optional[float] = None


class EvidenceItemModel(BaseModel):
    node_id: str
    evidence_id: str
    role: Literal["hit", "context"]
    rank: Optional[int] = None
    score: Optional[float] = None
    expanded_from: Optional[str] = None
    tokens: int
    text: str


class EvidencePackModel(BaseModel):
    strategy: str
    budget_tokens: Optional[int]
    packed_tokens: int
    dropped_hits: int
    items: List[EvidenceItemModel]


class RetrieveResponse(BaseModel):
    hits: List[RetrievedHit]
    evidence: Optional[EvidencePackModel] = None
    trace: Dict[str, Any]
