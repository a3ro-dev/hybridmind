"""Edge-related Pydantic models."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


from enum import Enum

class EdgeType(str, Enum):
    # Original semantic edge types
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

    # Auto-inferred structural/proximity edges (Phase 3)
    similar_to = "similar_to"          # cosine-threshold at ingest
    co_occurs = "co_occurs"            # entity co-occurrence
    mentions = "mentions"              # entity → mention node
    next_turn = "next_turn"            # sequential session turns
    same_session = "same_session"      # nodes in same session
    belongs_to = "belongs_to"          # sentence chunk → parent


# Typed walk-weight map used by compute_weighted_proximity_score().
# Weights determine how much each edge type contributes to graph proximity.
EDGE_TYPE_WALK_WEIGHTS: dict = {
    # Strong causal / logical links
    EdgeType.caused_by: 1.0,
    EdgeType.led_to: 1.0,
    EdgeType.depends_on: 0.9,
    EdgeType.contradicts: 0.9,
    EdgeType.supports: 0.9,
    EdgeType.derived_from: 0.9,
    EdgeType.invalidated_by: 0.8,
    EdgeType.refined_by: 0.8,

    # Structural / proximity
    EdgeType.similar_to: 0.7,
    EdgeType.analogous_to: 0.7,
    EdgeType.co_occurs: 0.6,
    EdgeType.mentions: 0.5,
    EdgeType.retrieved_during: 0.4,

    # Session-graph edges (useful for multi-turn recall)
    EdgeType.next_turn: 0.6,
    EdgeType.same_session: 0.5,
    EdgeType.belongs_to: 0.3,
}

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
