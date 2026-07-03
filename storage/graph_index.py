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
        self.graph = nx.DiGraph()
        
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
        # Ensure nodes exist
        if source_id not in self.graph:
            self.graph.add_node(source_id)
        if target_id not in self.graph:
            self.graph.add_node(target_id)
        
        self.graph.add_edge(
            source_id,
            target_id,
            type=edge_type,
            weight=weight,
            edge_id=edge_id,
            **attrs
        )
    
    def remove_edge(self, source_id: str, target_id: str) -> bool:
        """Remove an edge between two nodes."""
        if not self.graph.has_edge(source_id, target_id):
            return False
        self.graph.remove_edge(source_id, target_id)
        return True
    
    def remove_edge_by_id(self, edge_id: str) -> bool:
        """Remove an edge by its ID."""
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                self.graph.remove_edge(u, v)
                return True
        return False
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get edge data between two nodes."""
        if not self.graph.has_edge(source_id, target_id):
            return None
        return dict(self.graph.edges[source_id, target_id])
    
    def get_edge_by_id(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get edge data by edge ID."""
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return {
                    "source_id": u,
                    "target_id": v,
                    **data
                }
        return None
    
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
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if edge_types is None or data.get("type") in edge_types:
                    edges.append({
                        "source_id": node_id,
                        "target_id": target,
                        "direction": "outgoing",
                        **data
                    })
        
        # Incoming edges
        if direction in ("incoming", "both"):
            for source, _, data in self.graph.in_edges(node_id, data=True):
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
        edge_types: Optional[List[str]] = None
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
        
        visited: Dict[str, Tuple[int, List[str]]] = {start_id: (0, [start_id])}
        queue = deque([(start_id, 0, [start_id])])
        results = []
        
        while queue:
            node, depth, path = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get neighbors based on direction
            neighbors = set()
            
            if direction in ("outgoing", "both"):
                for _, target, data in self.graph.out_edges(node, data=True):
                    if edge_types is None or data.get("type") in edge_types:
                        neighbors.add(target)
            
            if direction in ("incoming", "both"):
                for source, _, data in self.graph.in_edges(node, data=True):
                    if edge_types is None or data.get("type") in edge_types:
                        neighbors.add(source)
            
            for neighbor in neighbors:
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
                    weight=lambda u, v, d: 1.0 / max(d.get("weight", 1.0), 0.01)
                )
                # Calculate actual path weight
                total_weight = sum(
                    self.graph.edges[path[i], path[i+1]].get("weight", 1.0)
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
    
    def compute_proximity_score(
        self,
        node_id: str,
        reference_nodes: List[str],
        max_depth: int = 3
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
        if not reference_nodes or node_id not in self.graph:
            return 0.0
        
        min_distance = float('inf')
        
        for ref_node in reference_nodes:
            if ref_node not in self.graph:
                continue
            
            if ref_node == node_id:
                return 1.0  # Same node
            
            # Try forward path
            try:
                dist = nx.shortest_path_length(self.graph, ref_node, node_id)
                if dist <= max_depth:
                    min_distance = min(min_distance, dist)
            except nx.NetworkXNoPath:
                pass
            
            # Try reverse path (undirected proximity)
            try:
                dist = nx.shortest_path_length(self.graph, node_id, ref_node)
                if dist <= max_depth:
                    min_distance = min(min_distance, dist)
            except nx.NetworkXNoPath:
                pass
        
        if min_distance == float('inf'):
            return 0.0
        
        return 1.0 / (1.0 + min_distance)
    
    def compute_weighted_proximity_score(
        self,
        node_id: str,
        reference_nodes: List[str],
        max_depth: int = 3,
        edge_type_weights: Optional[Dict[str, float]] = None
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
        base_score = self.compute_proximity_score(node_id, reference_nodes, max_depth)
        
        if base_score == 0.0 or not edge_type_weights:
            return base_score
        
        # Add bonus for valuable edge types along paths
        bonus = 0.0
        for ref_node in reference_nodes:
            path_result = self.get_shortest_path(ref_node, node_id)
            if path_result:
                path, _ = path_result
                for i in range(len(path) - 1):
                    edge_data = self.get_edge(path[i], path[i+1])
                    if edge_data:
                        edge_type = edge_data.get("type", "")
                        if edge_type in edge_type_weights:
                            bonus += edge_type_weights[edge_type] * 0.1
        
        return min(1.0, base_score + bonus)
    
    # ==================== Temporal Scoring ====================

    @staticmethod
    def _temporal_decay(created_at_str: Optional[str], half_life_days: float = 30.0) -> float:
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
            now = datetime.now(tz=timezone.utc)
            delta_days = (now - edge_time).total_seconds() / 86400.0
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
        if not reference_nodes or node_id not in self.graph:
            return 0.0

        now = datetime.now(tz=timezone.utc)
        min_weighted_dist = float("inf")

        for ref_node in reference_nodes:
            if ref_node not in self.graph:
                continue
            if ref_node == node_id:
                return 1.0

            # Try both directions (bidirectional proximity like base method)
            for src, tgt in [(ref_node, node_id), (node_id, ref_node)]:
                try:
                    path = nx.shortest_path(self.graph, src, tgt)
                    if len(path) - 1 > max_depth:
                        continue

                    # Compute minimum decay along path edges
                    path_decay = 1.0
                    valid_path = True
                    for i in range(len(path) - 1):
                        edge_data = self.graph.edges.get((path[i], path[i + 1]), {})

                        # Skip stale edges (superseded facts)
                        if skip_stale:
                            vu = edge_data.get("valid_until")
                            if vu:
                                try:
                                    if isinstance(vu, str):
                                        vu_dt = datetime.fromisoformat(vu.replace(" ", "T"))
                                        if vu_dt.tzinfo is None:
                                            vu_dt = vu_dt.replace(tzinfo=timezone.utc)
                                    else:
                                        vu_dt = vu
                                    if vu_dt < now:
                                        valid_path = False
                                        break
                                except Exception:
                                    pass

                        decay = self._temporal_decay(
                            edge_data.get("created_at"), half_life_days
                        )
                        path_decay = min(path_decay, decay)

                    if not valid_path:
                        continue

                    # Effective distance: hop count discounted by decay
                    # (lower decay → path is "longer" in temporal sense)
                    eff_dist = (len(path) - 1) / max(path_decay, 1e-6)
                    min_weighted_dist = min(min_weighted_dist, eff_dist)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        if min_weighted_dist == float("inf"):
            return 0.0
        return 1.0 / (1.0 + min_weighted_dist)

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
            self.graph = pickle.load(f)
    
    def rebuild_from_edges(self, edges: List[Dict[str, Any]]):
        """
        Rebuild graph from list of edge dictionaries.
        Used when loading from SQLite.
        """
        self.graph = nx.DiGraph()

        for edge in edges:
            # Propagate created_at + temporal fields so temporal_decay can use them
            extra = {}
            for field in ("created_at", "valid_from", "valid_until", "superseded_by", "confidence"):
                val = edge.get(field)
                if val is not None:
                    extra[field] = str(val) if isinstance(val, datetime) else val
            self.add_edge(
                source_id=edge["source_id"],
                target_id=edge["target_id"],
                edge_type=edge["type"],
                weight=edge.get("weight", 1.0),
                edge_id=edge.get("id"),
                **extra,
                **edge.get("metadata", {})
            )
    
    def clear(self):
        """Clear all nodes and edges."""
        self.graph = nx.DiGraph()
    
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

