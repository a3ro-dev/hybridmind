"""
Hybrid ranker for HybridMind.
Implements the Contextual Relevance Score (CRS) algorithm.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine


class HybridRanker:
    """
    Hybrid search ranker combining vector similarity and graph proximity.
    
    Implements the Contextual Relevance Score (CRS) algorithm:
    CRS = α * vector_score + β * graph_score + γ * relationship_bonus
    
    Where:
    - α (vector_weight): Weight for semantic similarity
    - β (graph_weight): Weight for graph proximity
    - γ: Additive bonus for specific edge types
    """
    
    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        graph_engine: GraphSearchEngine
    ):
        """
        Initialize hybrid ranker.
        
        Args:
            vector_engine: Vector search engine
            graph_engine: Graph search engine
        """
        self.vector_engine = vector_engine
        self.graph_engine = graph_engine
    
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        anchor_nodes: Optional[List[str]] = None,
        max_depth: int = 2,
        edge_type_weights: Optional[Dict[str, float]] = None,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        """
        Perform hybrid vector + graph search.
        
        Args:
            query_text: Search query text
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (α)
            graph_weight: Weight for graph proximity (β)
            anchor_nodes: Optional anchor nodes for graph scoring
            max_depth: Maximum graph traversal depth
            edge_type_weights: Bonus weights for edge types
            min_score: Minimum combined score threshold
            filter_metadata: Optional metadata filters (e.g., container tag)
            deduplicate: If True, remove results with identical text content
            
        Returns:
            Tuple of (results, query_time_ms, total_candidates)
        """
        start_time = time.perf_counter()
        
        # Normalize weights
        total_weight = vector_weight + graph_weight
        if total_weight > 0:
            alpha = vector_weight / total_weight
            beta = graph_weight / total_weight
        else:
            alpha, beta = 0.5, 0.5
        
        # Step 1: Vector search - get candidates
        # Request extra candidates for re-ranking + dedup headroom
        vector_k = top_k * 5 if deduplicate else top_k * 3
        vector_results, _, _ = self.vector_engine.search(
            query_text=query_text,
            top_k=vector_k,
            min_score=0.0,  # We'll filter later
            filter_metadata=filter_metadata
        )
        
        if not vector_results:
            query_time_ms = (time.perf_counter() - start_time) * 1000
            return [], round(query_time_ms, 2), 0
        
        # Step 2: Determine reference nodes for graph scoring
        if anchor_nodes:
            reference_nodes = anchor_nodes
        else:
            # Use top vector results as references
            reference_nodes = [r["node_id"] for r in vector_results[:3]]
        
        # Step 3: Compute graph scores for all candidates
        candidate_ids = [r["node_id"] for r in vector_results]
        graph_scores = self.graph_engine.compute_proximity_scores(
            node_ids=candidate_ids,
            reference_nodes=reference_nodes,
            max_depth=max_depth,
            edge_type_weights=edge_type_weights
        )
        
        # Step 4: Compute hybrid scores and build results
        hybrid_results = []
        
        for result in vector_results:
            node_id = result["node_id"]
            vector_score = result["vector_score"]
            graph_score = graph_scores.get(node_id, 0.0)
            
            # Compute CRS
            combined_score = self._compute_crs(
                vector_score=vector_score,
                graph_score=graph_score,
                alpha=alpha,
                beta=beta,
                edge_type_weights=edge_type_weights,
                node_id=node_id
            )
            
            if combined_score < min_score:
                continue
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                vector_score=vector_score,
                graph_score=graph_score,
                combined_score=combined_score,
                alpha=alpha,
                beta=beta,
                reference_nodes=reference_nodes,
                node_id=node_id
            )
            
            hybrid_results.append({
                "node_id": node_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "vector_score": round(vector_score, 4),
                "graph_score": round(graph_score, 4),
                "combined_score": round(combined_score, 4),
                "reasoning": reasoning
            })
        
        # Step 5: Sort by combined score
        hybrid_results.sort(key=lambda x: -x["combined_score"])
        
        # Step 6: Deduplicate by text content (keep highest-scoring copy)
        if deduplicate:
            seen_texts: Set[str] = set()
            deduped = []
            for result in hybrid_results:
                # Normalize text for comparison (strip whitespace)
                text_key = result["text"].strip()
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    deduped.append(result)
            hybrid_results = deduped
        
        # Take top-k
        hybrid_results = hybrid_results[:top_k]
        
        query_time_ms = (time.perf_counter() - start_time) * 1000
        
        return hybrid_results, round(query_time_ms, 2), len(vector_results)
    
    def _compute_crs(
        self,
        vector_score: float,
        graph_score: float,
        alpha: float,
        beta: float,
        edge_type_weights: Optional[Dict[str, float]] = None,
        node_id: Optional[str] = None
    ) -> float:
        """
        Compute Contextual Relevance Score.
        
        CRS = α * V + β * G + γ * R
        
        Args:
            vector_score: Vector similarity score (0-1)
            graph_score: Graph proximity score (0-1)
            alpha: Vector weight
            beta: Graph weight
            edge_type_weights: Edge type bonus weights
            node_id: Node ID for relationship bonus
            
        Returns:
            Combined CRS score (0-1)
        """
        # Base score
        base_score = (alpha * vector_score) + (beta * graph_score)
        
        # Relationship bonus (optional, additive)
        rel_bonus = 0.0
        if edge_type_weights and node_id:
            # Get edges connected to this node
            edges = self.graph_engine.graph_index.get_node_edges(node_id, direction="both")
            for edge in edges:
                edge_type = edge.get("type", "")
                if edge_type in edge_type_weights:
                    # Small additive bonus for valuable relationships
                    rel_bonus += edge_type_weights[edge_type] * 0.05
        
        # Cap at 1.0
        return min(1.0, base_score + rel_bonus)
    
    def _generate_reasoning(
        self,
        vector_score: float,
        graph_score: float,
        combined_score: float,
        alpha: float,
        beta: float,
        reference_nodes: List[str],
        node_id: str
    ) -> str:
        """Generate human-readable explanation of the score."""
        parts = []
        
        # Vector contribution
        if vector_score >= 0.8:
            parts.append(f"High semantic similarity ({vector_score:.0%})")
        elif vector_score >= 0.5:
            parts.append(f"Moderate semantic similarity ({vector_score:.0%})")
        else:
            parts.append(f"Low semantic similarity ({vector_score:.0%})")
        
        # Graph contribution
        if graph_score > 0:
            if graph_score >= 0.5:
                parts.append(f"strongly connected in graph ({graph_score:.0%})")
            else:
                parts.append(f"graph connection found ({graph_score:.0%})")
        else:
            parts.append("no direct graph connection")
        
        # Build final reasoning
        if len(parts) >= 2:
            return f"{parts[0]}, {parts[1]}"
        return parts[0] if parts else "Combined score calculation"
    
    def compare_search_modes(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        anchor_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare results across vector-only, graph-only, and hybrid modes.
        Useful for demonstrating hybrid advantages.
        
        Args:
            query_text: Search query
            top_k: Number of results per mode
            vector_weight: Weight for vector in hybrid
            graph_weight: Weight for graph in hybrid
            anchor_nodes: Anchor nodes for graph search
            
        Returns:
            Comparison results with all three modes
        """
        # Vector-only search
        vector_results, vector_time, vector_candidates = self.vector_engine.search(
            query_text=query_text,
            top_k=top_k
        )
        
        # Graph-only search (requires anchor)
        graph_results = []
        graph_time = 0.0
        graph_candidates = 0
        
        if anchor_nodes:
            for anchor in anchor_nodes:
                results, time_ms, candidates = self.graph_engine.traverse(
                    start_id=anchor,
                    depth=2
                )
                graph_results.extend(results)
                graph_time += time_ms
                graph_candidates += candidates
            
            # Deduplicate
            seen: Set[str] = set()
            unique_graph = []
            for r in graph_results:
                if r["node_id"] not in seen:
                    seen.add(r["node_id"])
                    unique_graph.append(r)
            graph_results = unique_graph[:top_k]
        
        # Hybrid search
        hybrid_results, hybrid_time, hybrid_candidates = self.search(
            query_text=query_text,
            top_k=top_k,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            anchor_nodes=anchor_nodes
        )
        
        # Analyze overlap and unique finds
        vector_ids = {r["node_id"] for r in vector_results}
        graph_ids = {r["node_id"] for r in graph_results}
        hybrid_ids = {r["node_id"] for r in hybrid_results}
        
        return {
            "vector_only": {
                "results": vector_results,
                "query_time_ms": vector_time,
                "total_candidates": vector_candidates
            },
            "graph_only": {
                "results": graph_results,
                "query_time_ms": graph_time,
                "total_candidates": graph_candidates
            },
            "hybrid": {
                "results": hybrid_results,
                "query_time_ms": hybrid_time,
                "total_candidates": hybrid_candidates
            },
            "analysis": {
                "vector_unique": len(vector_ids - hybrid_ids),
                "graph_unique": len(graph_ids - hybrid_ids),
                "hybrid_unique": len(hybrid_ids - vector_ids - graph_ids),
                "overlap_all": len(vector_ids & graph_ids & hybrid_ids),
                "hybrid_combines_best": len(hybrid_ids & (vector_ids | graph_ids))
            }
        }

