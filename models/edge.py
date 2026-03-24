"""Edge-related Pydantic models."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


from enum import Enum

class EdgeType(str, Enum):
    led_to = "led_to"
    contradicts = "contradicts"
    supports = "supports"
    caused_by = "caused_by"
    retrieved_during = "retrieved_during"
    refined_by = "refined_by"
    depends_on = "depends_on"
    analogous_to = "analogous_to"
    invalidated_by = "invalidated_by"
    derived_from = "derived_from"

class EdgeCreate(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class EdgeUpdate(BaseModel):
    type: Optional[EdgeType] = None
    weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class EdgeResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: str
    weight: float
    metadata: Dict[str, Any]
    created_at: datetime


class EdgeDeleteResponse(BaseModel):
    deleted: bool
    edge_id: str
