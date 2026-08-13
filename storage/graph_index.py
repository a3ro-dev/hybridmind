"""
NetworkX-based graph index for HybridMind.
Handles graph storage, traversal, and proximity scoring.
"""

import math
import pickle
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx


# Relations that are semantically symmetric even though they are persisted in
# a directed container. Causal, dependency, ownership, and sequence relations
# deliberately do not appear here.
_SYMMETRIC_EDGE_TYPES = {
    "analogous_to",
    "co_occurs",
    "contradicts",
    "same_session",
    "similar_to",
    "temporally_near",
}


def _parse_graph_time(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


class GraphIndex:
    """
    NetworkX-based directed graph for relationship storage and traversal.
    Supports BFS/DFS traversal, shortest path computation, and proximity scoring.
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize graph index.
        
        Args:
            index_path: Path for graph persistence
        """
        self.index_path = Path(index_path) if index_path else None
        # Multiple typed relations may connect the same pair (for example a
        # next_turn edge and a same_session edge). DiGraph silently overwrote
        # the first relation; MultiDiGraph preserves the SQLite edge model.
        self.graph = nx.MultiDiGraph()
        self._edge_locations: Dict[str, Tuple[str, str, Any]] = {}
        
        # Load from disk if exists
        if self.index_path and self.index_path.exists():
            self.load()
    
    @property
    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return self.graph.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return self.graph.number_of_edges()
    
    # ==================== Node Operations ====================
    
    def add_node(self, node_id: str, **attrs):
        """Add a node to the graph."""
        self.graph.add_node(node_id, **attrs)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self.graph:
            return False
        for source, target, key, data in list(
            self.graph.in_edges(node_id, keys=True, data=True)
        ) + list(self.graph.out_edges(node_id, keys=True, data=True)):
            edge_id = data.get("edge_id")
            if edge_id is not None:
                self._edge_locations.pop(str(edge_id), None)
        self.graph.remove_node(node_id)
        return True
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self.graph
    
    def get_node_attrs(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes."""
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])
    
    # ==================== Edge Operations ====================
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        edge_id: Optional[str] = None,
        **attrs
    ):
        """
        Add a directed edge to the graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Relationship type
            weight: Edge weight (0.0 to 1.0)
            edge_id: Optional edge identifier
            **attrs: Additional edge attributes
        """
        if source_id == target_id:
            raise ValueError("Graph self-loops are not supported")
        if not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0:
            raise ValueError("Graph edge weight must be finite and in [0, 1]")
        confidence = attrs.get("confidence", 1.0)
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("Graph edge confidence must be finite and in [0, 1]")
        if edge_id is not None and str(edge_id) in self._edge_locations:
            raise ValueError(f"Graph edge ID already exists: {edge_id}")

        # Ensure nodes exist
        if source_id not in self.graph:
            self.graph.add_node(source_id)
        if target_id not in self.graph:
            self.graph.add_node(target_id)
        
        key = edge_id or f"{edge_type}:{self.graph.number_of_edges(source_id, target_id)}"
        self.graph.add_edge(
            source_id,
            target_id,
            key=key,
            type=edge_type,
            weight=weight,
            edge_id=edge_id,
            **attrs
        )
        if edge_id is not None:
            self._edge_locations[str(edge_id)] = (source_id, target_id, key)
    
    def remove_edge(self, source_id: str, target_id: str) -> bool:
        """Remove every typed edge between two nodes."""
        if not self.graph.has_edge(source_id, target_id):
            return False
        for key, data in list(self.graph[source_id][target_id].items()):
            edge_id = data.get("edge_id")
            if edge_id is not None:
                self._edge_locations.pop(str(edge_id), None)
            self.graph.remove_edge(source_id, target_id, key=key)
        return True
    
    def remove_edge_by_id(self, edge_id: str) -> bool:
        """Remove an edge by its ID."""
        location = self._edge_locations.pop(str(edge_id), None)
        if location is None:
            return False
        source, target, key = location
        if not self.graph.has_edge(source, target, key=key):
            raise RuntimeError("Graph edge locator is inconsistent with graph state")
        self.graph.remove_edge(source, target, key=key)
        return True
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get edge data between two nodes."""
        records = self._edge_records(source_id, target_id)
        if not records:
            return None
        return max(records, key=lambda data: float(data.get("weight", 1.0)))
    
    def get_edge_by_id(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get edge data by edge ID."""
        location = self._edge_locations.get(str(edge_id))
        if location is None:
            return None
        source, target, key = location
        data = self.graph.get_edge_data(source, target, key)
        if data is None:
            raise RuntimeError("Graph edge locator is inconsistent with graph state")
        return {"source_id": source, "target_id": target, **data}
    
    def get_node_edges(
        self,
        node_id: str,
        direction: str = "both",
        edge_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all edges connected to a node.
        
        Args:
            node_id: Node ID
            direction: 'outgoing', 'incoming', or 'both'
            edge_types: Filter by edge types
            
        Returns:
            List of edge data dictionaries
        """
        if node_id not in self.graph:
            return []
        
        edges = []
        
        # Outgoing edges
        if direction in ("outgoing", "both"):
            for _, target, _key, data in self.graph.out_edges(node_id, keys=True, data=True):
                if edge_types is None or data.get("type") in edge_types:
                    edges.append({
                        "source_id": node_id,
                        "target_id": target,
                        "direction": "outgoing",
                        **data
                    })
        
        # Incoming edges
        if direction in ("incoming", "both"):
            for source, _, _key, data in self.graph.in_edges(node_id, keys=True, data=True):
                if edge_types is None or data.get("type") in edge_types:
                    edges.append({
                        "source_id": source,
                        "target_id": node_id,
                        "direction": "incoming",
                        **data
                    })
        
        return edges
    
    # ==================== Traversal Operations ====================
    
    def traverse_bfs(
        self,
        start_id: str,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: Optional[List[str]] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Tuple[str, int, List[str]]]:
        """
        BFS traversal from a starting node.
        
        Args:
            start_id: Starting node ID
            max_depth: Maximum traversal depth
            direction: 'outgoing', 'incoming', or 'both'
            edge_types: Filter by edge types
            
        Returns:
            List of (node_id, depth, path) tuples
        """
        if start_id not in self.graph:
            return []
        
        reference_time = _parse_graph_time(as_of) or datetime.now(timezone.utc)
        visited: Dict[str, Tuple[int, List[str]]] = {start_id: (0, [start_id])}
        queue = deque([(start_id, 0, [start_id])])
        results = []
        
        while queue:
            node, depth, path = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get neighbors based on direction. ``typed`` follows every edge
            # forward but only traverses the reverse of symmetric relations.
            neighbors = set()

            if direction in ("outgoing", "both", "typed"):
                for _, target, _key, data in self.graph.out_edges(node, keys=True, data=True):
                    if (
                        (edge_types is None or data.get("type") in edge_types)
                        and self._edge_is_active(data, reference_time)
                    ):
                        neighbors.add(target)

            if direction in ("incoming", "both"):
                for source, _, _key, data in self.graph.in_edges(node, keys=True, data=True):
                    if (
                        (edge_types is None or data.get("type") in edge_types)
                        and self._edge_is_active(data, reference_time)
                    ):
                        neighbors.add(source)
            elif direction == "typed":
                for source, _, _key, data in self.graph.in_edges(node, keys=True, data=True):
                    if (
                        data.get("type") in _SYMMETRIC_EDGE_TYPES
                        and (edge_types is None or data.get("type") in edge_types)
                        and self._edge_is_active(data, reference_time)
                    ):
                        neighbors.add(source)

            for neighbor in sorted(neighbors):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    visited[neighbor] = (depth + 1, new_path)
                    queue.append((neighbor, depth + 1, new_path))
                    results.append((neighbor, depth + 1, new_path))
        
        return results
    
    def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
        weighted: bool = True
    ) -> Optional[Tuple[List[str], float]]:
        """
        Find shortest path between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            weighted: Use edge weights (inverse for shortest path)
            
        Returns:
            (path, total_weight) or None if no path exists
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None
        
        try:
            if weighted:
                # Use inverse weight for "shortest" weighted path
                path = nx.shortest_path(
                    self.graph,
                    source_id,
                    target_id,
                    weight=lambda u, v, d: min(
                        1.0 / max(float(attrs.get("weight", 1.0)), 0.01)
                        for attrs in d.values()
                    ),
                )
                # Calculate actual path weight
                total_weight = sum(
                    max(
                        float(edge.get("weight", 1.0))
                        for edge in self._edge_records(path[i], path[i + 1])
                    )
                    for i in range(len(path) - 1)
                )
            else:
                path = nx.shortest_path(self.graph, source_id, target_id)
                total_weight = len(path) - 1
            
            return (path, total_weight)
        except nx.NetworkXNoPath:
            return None
    
    def get_shortest_path_length(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[int]:
        """Get shortest path length (number of hops)."""
        try:
            return nx.shortest_path_length(self.graph, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    # ==================== Scoring Operations ====================

    def _edge_records(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        if not self.graph.has_edge(source_id, target_id):
            return []
        return [dict(attrs) for attrs in self.graph[source_id][target_id].values()]

    @staticmethod
    def _edge_is_active(edge: Dict[str, Any], reference_time: datetime) -> bool:
        """Evaluate half-open edge validity at a transaction-independent time."""
        confidence = float(edge.get("confidence", 1.0))
        if not math.isfinite(confidence) or confidence <= 0.0:
            return False
        valid_from_raw = edge.get("valid_from")
        valid_until_raw = edge.get("valid_until")
        valid_from = _parse_graph_time(valid_from_raw)
        valid_until = _parse_graph_time(valid_until_raw)
        # A malformed declared boundary is corruption, not an unbounded range.
        if valid_from_raw not in (None, "") and valid_from is None:
            return False
        if valid_until_raw not in (None, "") and valid_until is None:
            return False
        if valid_from is not None and reference_time < valid_from:
            return False
        if valid_until is not None and reference_time >= valid_until:
            return False
        return True
    
    def compute_proximity_score(
        self,
        node_id: str,
        reference_nodes: List[str],
        max_depth: int = 3,
        as_of: Optional[datetime] = None,
    ) -> float:
        """
        Compute graph proximity score for a node relative to reference nodes.
        
        Score formula: 1 / (1 + min_distance)
        
        Args:
            node_id: Target node ID
            reference_nodes: List of anchor/reference node IDs
            max_depth: Maximum path length to consider
            
        Returns:
            Proximity score between 0.0 and 1.0
        """
        return self._best_path_proximity(
            node_id,
            reference_nodes,
            max_depth,
            direction="typed",
            as_of=as_of,
        )
    
    def compute_weighted_proximity_score(
        self,
        node_id: str,
        reference_nodes: List[str],
        max_depth: int = 3,
        edge_type_weights: Optional[Dict[str, float]] = None,
        direction: str = "typed",
        as_of: Optional[datetime] = None,
    ) -> float:
        """
        Compute weighted proximity score considering edge types and weights.
        
        Args:
            node_id: Target node ID
            reference_nodes: List of anchor/reference node IDs
            max_depth: Maximum path length
            edge_type_weights: Bonus weights for specific edge types
            
        Returns:
            Weighted proximity score
        """
        return self._best_path_proximity(
            node_id,
            reference_nodes,
            max_depth,
            edge_type_weights=edge_type_weights,
            direction=direction,
            as_of=as_of,
        )

    def _best_path_proximity(
        self,
        node_id: str,
        reference_nodes: List[str],
        max_depth: int,
        edge_type_weights: Optional[Dict[str, float]] = None,
        temporal_decay: bool = False,
        half_life_days: float = 30.0,
        skip_stale: bool = True,
        direction: str = "typed",
        as_of: Optional[datetime] = None,
    ) -> float:
        """Maximum confidence-aware strength over directionally valid paths."""
        if not reference_nodes or node_id not in self.graph:
            return 0.0
        reference_time = _parse_graph_time(as_of) or datetime.now(tz=timezone.utc)
        best_score = 0.0

        for reference in reference_nodes:
            if reference not in self.graph:
                continue
            if reference == node_id:
                return 1.0

            queue = deque([(reference, 0, 1.0)])
            best_state: Dict[Tuple[str, int], float] = {(reference, 0): 1.0}
            while queue:
                current, depth, strength = queue.popleft()
                if depth >= max_depth:
                    continue

                transitions: Dict[str, List[Dict[str, Any]]] = {}
                if direction in {"outgoing", "both", "typed"}:
                    for neighbor in self.graph.successors(current):
                        transitions.setdefault(neighbor, []).extend(
                            self._edge_records(current, neighbor)
                        )
                if direction in {"incoming", "both"}:
                    for neighbor in self.graph.predecessors(current):
                        transitions.setdefault(neighbor, []).extend(
                            self._edge_records(neighbor, current)
                        )
                elif direction == "typed":
                    for neighbor in self.graph.predecessors(current):
                        transitions.setdefault(neighbor, []).extend(
                            edge
                            for edge in self._edge_records(neighbor, current)
                            if edge.get("type") in _SYMMETRIC_EDGE_TYPES
                        )

                for neighbor in sorted(transitions):
                    options = transitions[neighbor]
                    best_edge = 0.0
                    for edge in options:
                        if skip_stale and not self._edge_is_active(edge, reference_time):
                            continue

                        edge_strength = max(0.0, min(1.0, float(edge.get("weight", 1.0))))
                        confidence = max(0.0, min(1.0, float(edge.get("confidence", 1.0))))
                        edge_strength *= confidence
                        if edge_type_weights:
                            edge_type = edge.get("type", "")
                            configured = edge_type_weights.get(edge_type, edge_type_weights.get(str(edge_type), 1.0))
                            edge_strength *= max(0.0, float(configured))
                        if temporal_decay:
                            edge_strength *= self._temporal_decay(
                                edge.get("valid_from") or edge.get("created_at"),
                                half_life_days,
                                as_of=reference_time,
                            )
                        best_edge = max(best_edge, edge_strength)

                    if best_edge <= 0.0:
                        continue
                    next_depth = depth + 1
                    next_strength = strength * best_edge
                    state = (neighbor, next_depth)
                    if next_strength <= best_state.get(state, -1.0):
                        continue
                    best_state[state] = next_strength
                    if neighbor == node_id:
                        best_score = max(best_score, next_strength / (1.0 + next_depth))
                    queue.append((neighbor, next_depth, next_strength))

        return min(1.0, best_score)
    
    # ==================== Temporal Scoring ====================

    @staticmethod
    def _temporal_decay(
        created_at_str: Optional[str],
        half_life_days: float = 30.0,
        *,
        as_of: Optional[datetime] = None,
    ) -> float:
        """
        Exponential decay weight based on edge age.

        w(t) = exp(-ln(2) * Δdays / half_life_days)

        Returns 1.0 if created_at is missing (no decay applied).
        half_life_days=7  → conversation memory (halves weekly)
        half_life_days=30 → default
        half_life_days=90 → domain knowledge (halves quarterly)
        """
        if not created_at_str:
            return 1.0
        try:
            if isinstance(created_at_str, datetime):
                edge_time = created_at_str
            else:
                # Parse ISO string (SQLite stores as "YYYY-MM-DD HH:MM:SS.ffffff")
                s = str(created_at_str).replace(" ", "T")
                if s.endswith("+00:00") or "Z" in s:
                    edge_time = datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=timezone.utc)
                else:
                    edge_time = datetime.fromisoformat(s)
                    if edge_time.tzinfo is None:
                        edge_time = edge_time.replace(tzinfo=timezone.utc)
            reference_time = _parse_graph_time(as_of) or datetime.now(tz=timezone.utc)
            delta_days = max(0.0, (reference_time - edge_time).total_seconds() / 86400.0)
            return math.exp(-math.log(2) * delta_days / half_life_days)
        except Exception:
            return 1.0

    def compute_temporal_proximity_score(
        self,
        node_id: str,
        reference_nodes: List[str],
        max_depth: int = 3,
        half_life_days: float = 30.0,
        skip_stale: bool = True,
        edge_type_weights: Optional[Dict[str, float]] = None,
        direction: str = "typed",
        as_of: Optional[datetime] = None,
    ) -> float:
        """
        Proximity score with temporal edge decay.

        G_temporal(q, n) = (1 / (1 + d_min)) * min(decay(e) for e in path)

        Stale edges (valid_until set and in the past) are skipped when
        skip_stale=True, making the effective graph reflect current knowledge.

        Args:
            node_id: Target node to score.
            reference_nodes: Anchor nodes (from vector search or explicit).
            max_depth: BFS depth limit.
            half_life_days: Decay half-life; tune per use-case.
            skip_stale: If True, skip edges where valid_until < now.

        Returns:
            Score in [0, 1]; 0 means no valid path found.
        """
        return self._best_path_proximity(
            node_id,
            reference_nodes,
            max_depth,
            edge_type_weights=edge_type_weights,
            temporal_decay=True,
            half_life_days=half_life_days,
            skip_stale=skip_stale,
            direction=direction,
            as_of=as_of,
        )

    # ==================== Persistence ====================

    def save(self, path: Optional[str] = None):
        """Save graph to disk."""
        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise ValueError("No path specified for saving")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load(self, path: Optional[str] = None):
        """Load graph from disk."""
        load_path = Path(path) if path else self.index_path
        if load_path is None or not load_path.exists():
            return
        
        with open(load_path, 'rb') as f:
            loaded = pickle.load(f)
            self.graph = (
                loaded
                if isinstance(loaded, nx.MultiDiGraph)
                else nx.MultiDiGraph(loaded)
            )
        self._rebuild_edge_locations()
    
    def rebuild_from_edges(self, edges: List[Dict[str, Any]]):
        """
        Rebuild graph from list of edge dictionaries.
        Used when loading from SQLite.
        """
        # Build and validate a complete replacement away from the live graph.
        # A malformed row or duplicate edge ID must leave the serving index
        # unchanged rather than exposing a half-rebuilt graph.
        replacement = GraphIndex(index_path=None)
        for edge in edges:
            # Propagate created_at + temporal fields so temporal_decay can use them
            extra = {}
            for field in ("created_at", "valid_from", "valid_until", "superseded_by", "confidence"):
                val = edge.get(field)
                if val is not None:
                    extra[field] = str(val) if isinstance(val, datetime) else val
            attrs = {**edge.get("metadata", {}), **extra}
            replacement.add_edge(
                source_id=edge["source_id"],
                target_id=edge["target_id"],
                edge_type=edge["type"],
                weight=edge.get("weight", 1.0),
                edge_id=edge.get("id"),
                **attrs,
            )
        self.graph = replacement.graph
        self._edge_locations = replacement._edge_locations
    
    def clear(self):
        """Clear all nodes and edges."""
        self.graph = nx.MultiDiGraph()
        self._edge_locations = {}

    def _rebuild_edge_locations(self) -> None:
        locations: Dict[str, Tuple[str, str, Any]] = {}
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            edge_id = data.get("edge_id")
            if edge_id is None:
                continue
            normalized = str(edge_id)
            if normalized in locations:
                raise ValueError(f"Graph contains duplicate edge ID: {normalized}")
            locations[normalized] = (source, target, key)
        self._edge_locations = locations
    
    # ==================== Analytics ====================
    
    def get_edge_type_counts(self) -> Dict[str, int]:
        """Get counts by edge type."""
        counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("type", "unknown")
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts
    
    def get_node_degree(self, node_id: str) -> Tuple[int, int]:
        """Get (in_degree, out_degree) for a node."""
        if node_id not in self.graph:
            return (0, 0)
        return (
            self.graph.in_degree(node_id),
            self.graph.out_degree(node_id)
        )
    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both"
    ) -> Set[str]:
        """Get immediate neighbors of a node."""
        if node_id not in self.graph:
            return set()
        
        neighbors = set()
        
        if direction in ("outgoing", "both"):
            neighbors.update(self.graph.successors(node_id))
        
        if direction in ("incoming", "both"):
            neighbors.update(self.graph.predecessors(node_id))
        
        return neighbors

