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
        graph_engine: GraphSearchEngine,
        bm25_index: Optional[Any] = None
    ):
        """
        Initialize hybrid ranker.
        
        Args:
            vector_engine: Vector search engine
            graph_engine: Graph search engine
            bm25_index: Optional BM25 index for exact matching
        """
        self.vector_engine = vector_engine
        self.graph_engine = graph_engine
        self.bm25_index = bm25_index
    
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
        deduplicate: bool = True,
        search_mode: str = "hybrid"
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        start_time = time.perf_counter()
        
        vector_k = top_k * 5 if deduplicate else top_k * 3
        # Expand candidates to accommodate SGMem sentence chunks
        vector_k = max(40, vector_k * 2)
        
        # Step 1: Run parallel Vector and BM25 search
        vector_results, _, _ = self.vector_engine.search(
            query_text=query_text,
            top_k=vector_k,
            min_score=0.0,
            filter_metadata=filter_metadata
        )
        
        bm25_results = []
        if self.bm25_index:
            bm25_hits = self.bm25_index.search(query_text, top_k=50000 if filter_metadata else vector_k)
            for n_id, score in bm25_hits:
                node = self.vector_engine.sqlite_store.get_node(n_id)
                if node:
                    if filter_metadata and not self.vector_engine._matches_filter(node["metadata"], filter_metadata):
                        continue
                    bm25_results.append({
                        "node_id": n_id,
                        "text": node["text"],
                        "metadata": node["metadata"],
                        "bm25_score": score
                    })
                    if len(bm25_results) >= vector_k:
                        break

        # Step 2: Apply Reciprocal Rank Fusion (RRF)
        # Using a lower k (e.g., 20) instead of 60 makes the top ranks stand out much more
        k_rrf = 20
        scores = {}
        node_data = {}
        
        for rank, res in enumerate(vector_results):
            nid = res["node_id"]
            node_data[nid] = res
            scores[nid] = scores.get(nid, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        for rank, res in enumerate(bm25_results):
            nid = res["node_id"]
            if nid not in node_data:
                node_data[nid] = res
                node_data[nid]["vector_score"] = 0.0  # Fallback for contract
            scores[nid] = scores.get(nid, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        # Step 3: SGMem Chunk Rollup to Parent
        rolled_up_scores = {}
        rolled_up_nodes = {}
        
        for nid, score in scores.items():
            meta = node_data[nid].get("metadata", {})
            if meta.get("is_sentence_chunk") and meta.get("parent_id"):
                parent_id = meta["parent_id"]
                rolled_up_scores[parent_id] = rolled_up_scores.get(parent_id, 0.0) + score
                if parent_id not in rolled_up_nodes:
                    p_node = self.vector_engine.sqlite_store.get_node(parent_id)
                    if p_node:
                        p_data = {
                            "node_id": parent_id,
                            "text": p_node["text"],
                            "metadata": p_node["metadata"],
                            "vector_score": node_data[nid].get("vector_score", 0.0)
                        }
                        rolled_up_nodes[parent_id] = p_data
            else:
                rolled_up_scores[nid] = rolled_up_scores.get(nid, 0.0) + score
                rolled_up_nodes[nid] = node_data[nid]
                
        # Get top rolled up nodes
        sorted_rrf = sorted(rolled_up_scores.items(), key=lambda x: -x[1])
        candidate_ids = [nid for nid, _ in sorted_rrf[:vector_k] if nid in rolled_up_nodes]
        

        
        if not candidate_ids:
            return [], round((time.perf_counter() - start_time) * 1000, 2), 0
            
        # Step 4: Compute Graph Scores
        if anchor_nodes:
            reference_nodes = anchor_nodes
        else:
            reference_nodes = candidate_ids[:3]
            
        graph_scores = self.graph_engine.compute_proximity_scores(
            node_ids=candidate_ids,
            reference_nodes=reference_nodes,
            max_depth=max_depth,
            edge_type_weights={"next_turn": 1.0, "same_session": 0.5, "belongs_to": 0.1, **(edge_type_weights or {})}
        )
        
        graph_ranks = {nid: rank for rank, (nid, score) in enumerate(sorted(graph_scores.items(), key=lambda item: -item[1])) if score > 0}
        
        # Compose Final Output
        hybrid_results = []
        for nid in candidate_ids:
            base_rrf = rolled_up_scores[nid]
            g_rank = graph_ranks.get(nid, -1)
            g_score = graph_scores.get(nid, 0.0)
            
            total_score = base_rrf
            if g_rank >= 0:
                total_score += (1.0 / (k_rrf + g_rank + 1))
                
            hybrid_results.append({
                "node_id": nid,
                "text": rolled_up_nodes[nid]["text"],
                "metadata": rolled_up_nodes[nid]["metadata"],
                "vector_score": rolled_up_nodes[nid].get("vector_score", 0.0),
                "graph_score": g_score,
                "combined_score": total_score,
                "reasoning": f"RRF Hybrid Score: {total_score:.4f}"
            })
            
        if deduplicate:
            seen_texts: Set[str] = set()
            deduped = []
            for result in hybrid_results:
                text_key = result["text"].strip()
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    deduped.append(result)
            hybrid_results = deduped
            
        # Step 5: Exact Match Cross-Encoder Re-Ranking Boost
        def bm25_overlap(query: str, text: str) -> float:
            q_terms = set(self.bm25_index.tokenize(query))
            t_terms = set(self.bm25_index.tokenize(text))
            if not q_terms: return 0.0
            overlap = sum(1 for qt in q_terms if qt in t_terms)
            return overlap / len(q_terms)
            
        for r in hybrid_results[:top_k * 2]:
            boost = bm25_overlap(query_text, r["text"])
            r["combined_score"] += boost * 0.25  # Major boost for exact keyword matches
            
        hybrid_results.sort(key=lambda x: -x["combined_score"])
        
        hybrid_results = [r for r in hybrid_results if r["combined_score"] >= min_score][:top_k]
        
        query_time_ms = (time.perf_counter() - start_time) * 1000
        return hybrid_results, round(query_time_ms, 2), len(candidate_ids)
    
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

