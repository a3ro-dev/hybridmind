"""
Graph search engine for HybridMind.
Handles graph traversal and proximity-based search.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from storage.graph_index import GraphIndex
from storage.sqlite_store import SQLiteStore


class GraphSearchEngine:
    """
    Graph search engine for relationship-based traversal and scoring.
    """
    
    def __init__(
        self,
        graph_index: GraphIndex,
        sqlite_store: SQLiteStore
    ):
        """
        Initialize graph search engine.
        
        Args:
            graph_index: NetworkX graph index
            sqlite_store: SQLite storage for node metadata
        """
        self.graph_index = graph_index
        self.sqlite_store = sqlite_store
    
    def traverse(
        self,
        start_id: str,
        depth: int = 2,
        edge_types: Optional[List[str]] = None,
        direction: str = "both"
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        """
        Perform graph traversal from a starting node.
        
        Args:
            start_id: Starting node ID
            depth: Maximum traversal depth
            edge_types: Filter by edge types
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            Tuple of (results, query_time_ms, total_candidates)
        """
        start_time = time.perf_counter()
        
        # Check if start node exists
        if not self.graph_index.has_node(start_id):
            # Try to get from SQLite
            start_node = self.sqlite_store.get_node(start_id)
            if start_node is None:
                return [], 0.0, 0
        
        # Perform BFS traversal
        traversal_results = self.graph_index.traverse_bfs(
            start_id=start_id,
            max_depth=depth,
            direction=direction,
            edge_types=edge_types
        )
        
        # Fetch node details and build results
        results = []
        for node_id, dist, path in traversal_results:
            node = self.sqlite_store.get_node(node_id)
            if node is None:
                continue
            
            # Calculate graph score based on distance
            graph_score = 1.0 / (1.0 + dist)
            
            results.append({
                "node_id": node_id,
                "text": node["text"],
                "metadata": node["metadata"],
                "graph_score": round(graph_score, 4),
                "depth": dist,
                "path": path,
                "reasoning": f"Reachable in {dist} hop(s) from start node"
            })
        
        # Sort by graph score (closer nodes first)
        results.sort(key=lambda x: (-x["graph_score"], x["depth"]))
        
        query_time_ms = (time.perf_counter() - start_time) * 1000
        
        return results, round(query_time_ms, 2), len(traversal_results)
    
    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get immediate neighbors of a node with edge details.
        
        Args:
            node_id: Node ID
            edge_types: Filter by edge types
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of neighbor info with edge details
        """
        edges = self.graph_index.get_node_edges(
            node_id,
            direction=direction,
            edge_types=edge_types
        )
        
        neighbors = []
        seen_nodes: Set[str] = set()
        
        for edge in edges:
            # Determine neighbor node
            if edge["direction"] == "outgoing":
                neighbor_id = edge["target_id"]
            else:
                neighbor_id = edge["source_id"]
            
            if neighbor_id in seen_nodes:
                continue
            seen_nodes.add(neighbor_id)
            
            # Fetch neighbor details
            neighbor_node = self.sqlite_store.get_node(neighbor_id)
            if neighbor_node:
                neighbors.append({
                    "node_id": neighbor_id,
                    "text": neighbor_node["text"],
                    "metadata": neighbor_node["metadata"],
                    "edge_type": edge["type"],
                    "edge_weight": edge["weight"],
                    "direction": edge["direction"]
                })
        
        return neighbors
    
    def compute_proximity_scores(
        self,
        node_ids: List[str],
        reference_nodes: List[str],
        max_depth: int = 3,
        edge_type_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute graph proximity scores for multiple nodes.
        
        Args:
            node_ids: List of nodes to score
            reference_nodes: Anchor nodes for proximity
            max_depth: Maximum path length
            
        Returns:
            Dict mapping node_id to proximity score
        """
        scores = {}
        
        for node_id in node_ids:
            if edge_type_weights:
                score = self.graph_index.compute_weighted_proximity_score(
                    node_id,
                    reference_nodes,
                    max_depth,
                    edge_type_weights
                )
            else:
                score = self.graph_index.compute_proximity_score(
                    node_id,
                    reference_nodes,
                    max_depth
                )
            scores[node_id] = round(score, 4)
        
        return scores
    
    def find_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            Path info or None if no path exists
        """
        result = self.graph_index.get_shortest_path(source_id, target_id)
        
        if result is None:
            return None
        
        path, total_weight = result
        
        # Build path with node details
        path_nodes = []
        for node_id in path:
            node = self.sqlite_store.get_node(node_id)
            if node:
                path_nodes.append({
                    "node_id": node_id,
                    "text": node["text"][:100] + "..." if len(node["text"]) > 100 else node["text"],
                    "metadata": node["metadata"]
                })
        
        # Get edges along path
        path_edges = []
        for i in range(len(path) - 1):
            edge = self.graph_index.get_edge(path[i], path[i+1])
            if edge:
                path_edges.append({
                    "from": path[i],
                    "to": path[i+1],
                    "type": edge.get("type", "unknown"),
                    "weight": edge.get("weight", 1.0)
                })
        
        return {
            "path": path,
            "length": len(path) - 1,
            "total_weight": round(total_weight, 4),
            "nodes": path_nodes,
            "edges": path_edges
        }
    
    def get_connected_component(
        self,
        node_id: str,
        max_size: int = 100
    ) -> List[str]:
        """
        Get all nodes in the same connected component.
        
        Args:
            node_id: Starting node
            max_size: Maximum nodes to return
            
        Returns:
            List of node IDs in the component
        """
        # Use BFS with large depth
        traversal = self.graph_index.traverse_bfs(
            start_id=node_id,
            max_depth=10,
            direction="both"
        )
        
        component = [node_id] + [nid for nid, _, _ in traversal]
        return component[:max_size]
    
    def add_to_index(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        edge_id: Optional[str] = None
    ):
        """Add edge to graph index."""
        self.graph_index.add_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            edge_id=edge_id
        )
    
    def remove_node_from_index(self, node_id: str):
        """Remove node from graph index."""
        self.graph_index.remove_node(node_id)
    
    def remove_edge_from_index(self, edge_id: str):
        """Remove edge from graph index by ID."""
        self.graph_index.remove_edge_by_id(edge_id)
    
    def rebuild_index(self):
        """Rebuild graph index from SQLite store."""
        edges = self.sqlite_store.get_all_edges()
        self.graph_index.rebuild_from_edges(edges)
        
        # Also add nodes without edges
        nodes = self.sqlite_store.list_nodes(limit=10000)
        for node in nodes:
            if not self.graph_index.has_node(node["id"]):
                self.graph_index.add_node(node["id"])

