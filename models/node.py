"""Node-related Pydantic models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NodeCreate(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    modality: str = "text"
    entities: List[str] = Field(default_factory=list)
    event_time: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    memory_kind: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class NodeUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    regenerate_embedding: bool = False
    entities: Optional[List[str]] = None
    event_time: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    memory_kind: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EdgeSummary(BaseModel):
    edge_id: str
    target_id: str
    type: str
    weight: float
    direction: str


class NodeResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    modality: str = "text"
    event_time: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    memory_kind: Optional[str] = None
    confidence: float = 1.0
    access_count: int = 0
    last_accessed_at: Optional[str] = None
    archived_at: Optional[str] = None
    archived_by: Optional[str] = None
    edges: List[EdgeSummary] = Field(default_factory=list)


class NodeDeleteResponse(BaseModel):
    deleted: bool
    node_id: str
    edges_removed: int
