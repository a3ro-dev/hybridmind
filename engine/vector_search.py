"""
Vector search engine for HybridMind.
Handles semantic similarity search using FAISS.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from storage.vector_index import VectorIndex
from storage.sqlite_store import SQLiteStore
from engine.embedding import EmbeddingEngine


class VectorSearchEngine:
    """
    Vector search engine combining embedding generation with FAISS indexing.
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        sqlite_store: SQLiteStore,
        embedding_engine: EmbeddingEngine
    ):
        """
        Initialize vector search engine.
        
        Args:
            vector_index: FAISS vector index
            sqlite_store: SQLite storage for metadata
            embedding_engine: Embedding generation engine
        """
        self.vector_index = vector_index
        self.sqlite_store = sqlite_store
        self.embedding_engine = embedding_engine
    
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], float, int]:
        """
        Perform vector similarity search.
        
        Args:
            query_text: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            filter_metadata: Optional metadata filters
            
        Returns:
            Tuple of (results, query_time_ms, total_candidates)
        """
        start_time = time.perf_counter()
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query_text)
        
        # Search vector index (get extra candidates for filtering + dedup headroom)
        search_k = top_k * 5 if filter_metadata else top_k * 3
        candidates = self.vector_index.search(
            query_embedding,
            top_k=search_k,
            min_score=min_score
        )
        
        # Fetch node details, apply filters, and deduplicate by text
        results = []
        seen_texts: set = set()
        for node_id, score in candidates:
            node = self.sqlite_store.get_node(node_id)
            if node is None:
                continue
            
            # Apply metadata filter
            if filter_metadata and not self._matches_filter(node["metadata"], filter_metadata):
                continue
            
            # Deduplicate: skip if we already have a result with identical text
            text_key = node["text"].strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            
            results.append({
                "node_id": node_id,
                "text": node["text"],
                "metadata": node["metadata"],
                "vector_score": round(score, 4),
                "reasoning": f"Semantic similarity: {score:.2%}"
            })
            
            if len(results) >= top_k:
                break
        
        query_time_ms = (time.perf_counter() - start_time) * 1000
        
        return results, round(query_time_ms, 2), len(candidates)
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search by pre-computed embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            min_score: Minimum score
            
        Returns:
            List of (node_id, score) tuples
        """
        return self.vector_index.search(query_embedding, top_k, min_score)
    
    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter_criteria: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Supports:
        - Exact match: {"field": "value"}
        - List contains: {"tags": "machine learning"}
        - Comparison: {"year": {"$gte": 2020}}
        """
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            meta_value = metadata[key]
            
            # Handle comparison operators
            if isinstance(value, dict):
                if not self._apply_comparison(meta_value, value):
                    return False
            # Handle list containment
            elif isinstance(meta_value, list):
                if value not in meta_value:
                    return False
            # Exact match
            elif meta_value != value:
                return False
        
        return True
    
    def _apply_comparison(
        self,
        value: Any,
        operators: Dict[str, Any]
    ) -> bool:
        """Apply comparison operators."""
        for op, target in operators.items():
            if op == "$gt" and not (value > target):
                return False
            elif op == "$gte" and not (value >= target):
                return False
            elif op == "$lt" and not (value < target):
                return False
            elif op == "$lte" and not (value <= target):
                return False
            elif op == "$ne" and not (value != target):
                return False
            elif op == "$in" and value not in target:
                return False
            elif op == "$nin" and value in target:
                return False
        return True
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.embedding_engine.embed(text)
    
    def add_to_index(self, node_id: str, embedding: np.ndarray):
        """Add embedding to vector index."""
        self.vector_index.add(node_id, embedding)
    
    def remove_from_index(self, node_id: str):
        """Remove embedding from vector index."""
        self.vector_index.remove(node_id)
    
    def rebuild_index(self):
        """Rebuild vector index from SQLite store."""
        embeddings = self.sqlite_store.get_all_node_embeddings()
        self.vector_index.rebuild_from_embeddings(embeddings)

