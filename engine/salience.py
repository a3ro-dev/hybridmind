"""ACT-R-inspired, bounded memory salience scoring."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from engine.temporal import parse_datetime


def _decay(value, now: datetime, half_life_days: float, default: float = 0.0) -> float:
    parsed = parse_datetime(value)
    if parsed is None:
        return default
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / max(half_life_days, 1e-6))


def compute_salience(node: dict, graph_index, settings, now=None) -> float:
    """Return a deterministic [0, 1] recency/frequency/centrality score."""
    now = parse_datetime(now) or datetime.now(timezone.utc)
    recency = _decay(
        node.get("event_time") or node.get("created_at"),
        now,
        settings.salience_recency_half_life_days,
        default=0.0,
    )
    access_count = max(0, int(node.get("access_count") or 0))
    frequency = min(1.0, math.log1p(access_count) / math.log1p(20.0))
    access_recency = _decay(
        node.get("last_accessed_at"),
        now,
        settings.salience_access_half_life_days,
        default=0.0,
    )
    frequency *= 0.5 + 0.5 * access_recency

    graph = graph_index.graph
    degree = graph.degree(node["id"]) if graph.has_node(node["id"]) else 0
    max_degree = max((value for _, value in graph.degree()), default=1)
    centrality = min(1.0, float(degree) / max(float(max_degree), 1.0))

    weights = (
        settings.salience_recency_weight,
        settings.salience_frequency_weight,
        settings.salience_centrality_weight,
    )
    weight_sum = max(sum(weights), 1e-9)
    score = (weights[0] * recency + weights[1] * frequency + weights[2] * centrality) / weight_sum
    return max(0.0, min(1.0, score))
