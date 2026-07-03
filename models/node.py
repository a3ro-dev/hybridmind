"""Node-related Pydantic models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NodeCreate(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    modality: str = "text"


class NodeUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    regenerate_embedding: bool = False


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
    edges: List[EdgeSummary] = Field(default_factory=list)


class NodeDeleteResponse(BaseModel):
    deleted: bool
    node_id: str
    edges_removed: int
