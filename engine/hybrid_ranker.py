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
    Score = α * V(q, n) + β * G(A, n)

    Where:
    - α (vector_weight): Weight for semantic similarity
    - β (graph_weight): Weight for graph proximity
    """

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        graph_engine: GraphSearchEngine,
        bm25_index: Optional[Any] = None,
        disable_graph_expansion: bool = False
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
        self.disable_graph_expansion = disable_graph_expansion

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        anchor_nodes: Optional[List[str]] = None,
        max_depth: int = 2,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
        search_mode: str = "hybrid",
        bm25_boost_weight: float = 0.25
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        start_time = time.perf_counter()

        # We need candidate generation. We will pull top_k * 5 vector results and bm25 results.
        vector_k = top_k * 5 if deduplicate else top_k * 3
        # Expand candidates to accommodate SGMem sentence chunks
        vector_k = max(40, vector_k * 2)

        # We also need a larger candidate pool if graph helps recall
        candidate_k = max(100, vector_k)

        # Step 1: Run Vector and BM25 search
        vector_results, _, _ = self.vector_engine.search(
            query_text=query_text,
            top_k=candidate_k,
            min_score=0.0,
            filter_metadata=filter_metadata
        )

        bm25_results = []
        if self.bm25_index:
            bm25_hits = self.bm25_index.search(query_text, top_k=50000 if filter_metadata else candidate_k)
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
                    if len(bm25_results) >= candidate_k:
                        break

        # Step 2: Combine vector and BM25 into a baseline V score.
        # Vector score is cosine similarity (0 to 1). We should boost it if BM25 matches.
        def bm25_overlap(query: str, text: str) -> float:
            if not self.bm25_index:
                return 0.0
            q_terms = set(self.bm25_index.tokenize(query))
            t_terms = set(self.bm25_index.tokenize(text))
            if not q_terms: return 0.0
            overlap = sum(1 for qt in q_terms if qt in t_terms)
            return overlap / len(q_terms)

        scores = {}
        node_data = {}

        # Give vector results their base cosine score + bm25 overlap boost
        for res in vector_results:
            nid = res["node_id"]
            node_data[nid] = res
            boost = bm25_overlap(query_text, res["text"]) * bm25_boost_weight
            scores[nid] = res.get("vector_score", 0.0) + boost

        # Give bm25 results their bm25 overlap boost if they didn't have a vector score
        for res in bm25_results:
            nid = res["node_id"]
            if nid not in node_data:
                node_data[nid] = res
                overlap = bm25_overlap(query_text, res["text"])
                boost = overlap * bm25_boost_weight

                # FIX: BM25-only hits lack a vector_score, meaning their max score is 0.25.
                # Since weak semantic vector hits average 0.4-0.6, BM25-only exact matches
                # were mathematically incapable of ever reaching the top 10.
                # We assign a synthetic base score proportional to keyword overlap to fix this.
                synthetic_base = min(0.65, overlap) * (1.0 if bm25_boost_weight > 0 else 0.0)
                node_data[nid]["vector_score"] = synthetic_base
                scores[nid] = synthetic_base + boost

        # Step 3: SGMem Chunk Rollup to Parent
        rolled_up_scores = {}
        rolled_up_nodes = {}

        for nid, score in scores.items():
            meta = node_data[nid].get("metadata", {})
            if meta.get("is_sentence_chunk") and meta.get("parent_id"):
                parent_id = meta["parent_id"]
                rolled_up_scores[parent_id] = max(rolled_up_scores.get(parent_id, 0.0), score)
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
                rolled_up_scores[nid] = score
                rolled_up_nodes[nid] = node_data[nid]

        # We cap normalized score to 1.0 just in case
        for nid in rolled_up_scores:
            rolled_up_scores[nid] = min(rolled_up_scores[nid], 1.0)

        sorted_rrf = sorted(rolled_up_scores.items(), key=lambda x: -x[1])
        candidate_ids = [nid for nid, _ in sorted_rrf[:candidate_k] if nid in rolled_up_nodes]

        if not candidate_ids:
            return [], round((time.perf_counter() - start_time) * 1000, 2), 0

        # Optional graph-aware candidate expansion path:
        # Before we compute graph scores, add graph neighbors of anchor nodes to candidate pool
        # This allows graph structure to affect recall
        if anchor_nodes and not self.disable_graph_expansion:
            reference_nodes = anchor_nodes
        else:
            reference_nodes = candidate_ids[:3]

        # Expand candidates
        expanded_candidates = set(candidate_ids)
        if not self.disable_graph_expansion:
            for ref in reference_nodes:
            # Traversal
                try:
                    # get nodes from graph within max_depth
                    neighbors, _, _ = self.graph_engine.traverse(start_id=ref, depth=max_depth)
                    for n in neighbors:
                        expanded_candidates.add(n["node_id"])
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Graph traversal failed for {ref}: {e}")

        # Add the expanded candidates to our node_data and rolled_up_scores if missing
        expanded_candidates_list = list(expanded_candidates)
        for nid in expanded_candidates_list:
            if nid not in rolled_up_nodes:
                p_node = self.vector_engine.sqlite_store.get_node(nid)
                if p_node:
                    if filter_metadata and not self.vector_engine._matches_filter(p_node["metadata"], filter_metadata):
                        continue
                    p_data = {
                        "node_id": nid,
                        "text": p_node["text"],
                        "metadata": p_node["metadata"],
                        "vector_score": 0.0 # pure graph candidates have 0 initial vector score
                    }
                    rolled_up_nodes[nid] = p_data
                    rolled_up_scores[nid] = 0.0

        # Update candidate_ids to include expanded pool
        candidate_ids = [nid for nid in expanded_candidates_list if nid in rolled_up_nodes]

        # Step 4: Compute Graph Scores
        graph_scores = self.graph_engine.compute_proximity_scores(
            node_ids=candidate_ids,
            reference_nodes=reference_nodes,
            max_depth=max_depth,
        )

        # Step 5: Late Fusion Scoring
        # Score(q,n) = α·V(q,n) + β·G(A,n)

        hybrid_results = []
        for nid in candidate_ids:
            # Use the normalized RRF score (which includes vector+BM25) as V
            v_score = rolled_up_scores[nid]
            g_score = graph_scores.get(nid, 0.0)

            combined_score = (vector_weight * v_score) + (graph_weight * g_score)

            hybrid_results.append({
                "node_id": nid,
                "text": rolled_up_nodes[nid]["text"],
                "metadata": rolled_up_nodes[nid]["metadata"],
                "vector_score": v_score,
                "graph_score": g_score,
                "combined_score": combined_score,
                "reasoning": f"Score = {vector_weight}*{v_score:.4f} + {graph_weight}*{g_score:.4f}"
            })

        if deduplicate:
            seen_texts: Set[str] = set()
            deduped = []
            # We sort by combined score first so we keep the highest scoring version of duplicate texts
            hybrid_results.sort(key=lambda x: -x["combined_score"])
            for result in hybrid_results:
                text_key = result["text"].strip()
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    deduped.append(result)
            hybrid_results = deduped

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

        if anchor_nodes and not self.disable_graph_expansion:
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
