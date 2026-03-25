"""
FAISS-based vector index for HybridMind.
Handles vector storage, similarity search, and persistence.

Optimized with:
- Soft delete support (avoids full rebuild on removal)
- Automatic compaction when deletion threshold reached
- Scalable index types for different dataset sizes
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorIndex:
    """
    FAISS-based vector index for similarity search.
    
    Features:
    - Soft delete support for efficient removal
    - Automatic compaction when deletions exceed threshold
    - IndexFlatIP for cosine similarity with normalized vectors
    - Falls back to NumPy if FAISS is not available
    """
    
    def __init__(
        self,
        dimension: int = 384,
        index_path: Optional[str] = None,
        deletion_threshold: float = 0.2
    ):
        """
        Initialize vector index.
        
        Args:
            dimension: Embedding dimension (384 for MiniLM)
            index_path: Path for index persistence
            deletion_threshold: Trigger compaction when this fraction is deleted (0.2 = 20%)
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.deletion_threshold = deletion_threshold
        
        # Mapping between FAISS indices and node IDs
        self.id_map: Dict[int, str] = {}  # FAISS idx -> node_id
        self.reverse_map: Dict[str, int] = {}  # node_id -> FAISS idx
        
        # Soft delete tracking
        self.deleted_ids: Set[str] = set()
        
        # Initialize index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
            self._use_faiss = True
            logger.info(f"FAISS vector index initialized: dimension={dimension}")
        else:
            # Fallback to NumPy-based search
            self._vectors: List[np.ndarray] = []
            self._use_faiss = False
            logger.warning("FAISS not available, using NumPy fallback")
        
        # Load from disk if exists
        if self.index_path and self.index_path.exists():
            self.load()
    
    @property
    def size(self) -> int:
        """Get number of vectors in index (excluding soft-deleted)."""
        if self._use_faiss:
            return self.index.ntotal - len(self.deleted_ids)
        return len(self._vectors) - len(self.deleted_ids)
    
    @property
    def total_size(self) -> int:
        """Get total vectors including soft-deleted."""
        if self._use_faiss:
            return self.index.ntotal
        return len(self._vectors)
    
    @property
    def deletion_ratio(self) -> float:
        """Get ratio of deleted to total vectors."""
        total = self.total_size
        if total == 0:
            return 0.0
        return len(self.deleted_ids) / total
    
    def add(self, node_id: str, embedding: np.ndarray):
        """
        Add a vector to the index.
        
        Args:
            node_id: Unique node identifier
            embedding: Vector embedding (will be normalized)
        """
        # Normalize for cosine similarity
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized = embedding / norm
        else:
            normalized = embedding
        
        # Remove old entry if exists (uses soft delete)
        if node_id in self.reverse_map:
            self.remove(node_id)
        
        # Add to index
        idx = self.total_size
        
        if self._use_faiss:
            self.index.add(normalized.reshape(1, -1))
        else:
            self._vectors.append(normalized)
        
        # Update mappings
        self.id_map[idx] = node_id
        self.reverse_map[node_id] = idx
    
    def add_batch(self, nodes: List[Tuple[str, np.ndarray]]):
        """
        Add multiple vectors in batch (more efficient).
        
        Args:
            nodes: List of (node_id, embedding) tuples
        """
        if not nodes:
            return
        
        # Remove existing entries first
        for node_id, _ in nodes:
            if node_id in self.reverse_map:
                self.remove(node_id)
        
        # Prepare normalized vectors
        vectors = []
        for node_id, embedding in nodes:
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized = embedding / norm
            else:
                normalized = embedding
            vectors.append(normalized)
        
        # Batch add
        start_idx = self.total_size
        
        if self._use_faiss:
            vectors_array = np.vstack(vectors).astype(np.float32)
            self.index.add(vectors_array)
        else:
            self._vectors.extend(vectors)
        
        # Update mappings
        for i, (node_id, _) in enumerate(nodes):
            idx = start_idx + i
            self.id_map[idx] = node_id
            self.reverse_map[node_id] = idx
        
        logger.debug(f"Batch added {len(nodes)} vectors to index")
    
    def remove(self, node_id: str) -> bool:
        """
        Soft delete a vector from the index.
        Marks as deleted without rebuilding index.
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if node_id not in self.reverse_map:
            return False
        
        # Soft delete - just mark as deleted
        self.deleted_ids.add(node_id)
        
        # Check if compaction needed
        if self.deletion_ratio > self.deletion_threshold:
            logger.info(
                f"Deletion threshold exceeded ({self.deletion_ratio:.1%}), "
                f"triggering compaction"
            )
            self._compact()
        
        return True
    
    def _compact(self):
        """
        Rebuild index excluding soft-deleted vectors.
        Called automatically when deletion threshold exceeded.
        """
        if not self.deleted_ids:
            return
        
        logger.info(f"Compacting vector index: removing {len(self.deleted_ids)} deleted entries")
        start_count = self.total_size
        
        # Collect all non-deleted vectors
        remaining = []
        remaining_ids = []
        
        for idx, node_id in sorted(self.id_map.items()):
            if node_id not in self.deleted_ids:
                if self._use_faiss:
                    # Reconstruct vector from FAISS index
                    vec = np.zeros(self.dimension, dtype=np.float32)
                    self.index.reconstruct(idx, vec)
                else:
                    vec = self._vectors[idx]
                remaining.append(vec)
                remaining_ids.append(node_id)
        
        # Rebuild index
        self._rebuild(remaining, remaining_ids)
        
        # Clear deleted set
        self.deleted_ids.clear()
        
        logger.info(
            f"Compaction complete: {start_count} -> {self.total_size} vectors"
        )
    
    def _rebuild(self, vectors: List[np.ndarray], node_ids: List[str]):
        """Rebuild index with new vectors."""
        # Clear current index
        if self._use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self._vectors = []
        
        self.id_map = {}
        self.reverse_map = {}
        
        # Add all vectors
        if vectors:
            if self._use_faiss:
                vectors_array = np.vstack(vectors).astype(np.float32)
                self.index.add(vectors_array)
            else:
                self._vectors = vectors
            
            for idx, nid in enumerate(node_ids):
                self.id_map[idx] = nid
                self.reverse_map[nid] = idx
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        Automatically filters out soft-deleted entries.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of (node_id, score) tuples sorted by score descending
        """
        if self.size == 0:
            return []
        
        # Normalize query
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            normalized_query = query / norm
        else:
            normalized_query = query
        
        # Request extra results to account for deleted items
        fetch_k = min(top_k + len(self.deleted_ids), self.total_size)
        
        if self._use_faiss:
            # FAISS search
            scores, indices = self.index.search(
                normalized_query.reshape(1, -1),
                fetch_k
            )
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:  # FAISS returns -1 for empty slots
                    continue
                
                node_id = self.id_map.get(idx)
                if node_id and node_id not in self.deleted_ids:
                    sim_score = float(score)
                    if sim_score >= min_score:
                        results.append((node_id, sim_score))
                        
                        if len(results) >= top_k:
                            break
        else:
            # NumPy fallback
            if len(self._vectors) == 0:
                return []
            
            vectors_matrix = np.vstack(self._vectors)
            # Cosine similarity via dot product (vectors are normalized)
            similarities = np.dot(vectors_matrix, normalized_query)
            
            # Get top indices
            top_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in top_indices:
                idx = int(idx)
                node_id = self.id_map.get(idx)
                if node_id and node_id not in self.deleted_ids:
                    sim_score = float(similarities[idx])
                    if sim_score >= min_score:
                        results.append((node_id, sim_score))
                        
                        if len(results) >= top_k:
                            break
        
        return results
    
    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        """Get vector by node ID."""
        if node_id not in self.reverse_map or node_id in self.deleted_ids:
            return None
        
        idx = self.reverse_map[node_id]
        
        if self._use_faiss:
            vec = np.zeros(self.dimension, dtype=np.float32)
            self.index.reconstruct(idx, vec)
            return vec
        else:
            return self._vectors[idx].copy()
    
    def has_vector(self, node_id: str) -> bool:
        """Check if node has a vector in the index."""
        return node_id in self.reverse_map and node_id not in self.deleted_ids
    
    def save(self, path: Optional[str] = None):
        """Save index to disk."""
        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise ValueError("No path specified for saving")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "dimension": self.dimension,
            "id_map": self.id_map,
            "reverse_map": self.reverse_map,
            "deleted_ids": self.deleted_ids,
            "use_faiss": self._use_faiss,
            "deletion_threshold": self.deletion_threshold,
        }
        
        if self._use_faiss:
            # Save FAISS index separately
            faiss_path = save_path.with_suffix('.faiss')
            faiss.write_index(self.index, str(faiss_path))
        else:
            data["vectors"] = self._vectors
        
        # Save metadata
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Vector index saved: {self.size} vectors to {save_path}")
    
    def load(self, path: Optional[str] = None):
        """Load index from disk."""
        load_path = Path(path) if path else self.index_path
        if load_path is None or not load_path.exists():
            return
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data["dimension"]
        self.id_map = data["id_map"]
        self.reverse_map = data["reverse_map"]
        self.deleted_ids = data.get("deleted_ids", set())
        self.deletion_threshold = data.get("deletion_threshold", 0.2)
        
        if data.get("use_faiss", False) and FAISS_AVAILABLE:
            faiss_path = load_path.with_suffix('.faiss')
            if faiss_path.exists():
                self.index = faiss.read_index(str(faiss_path))
                self._use_faiss = True
        elif "vectors" in data:
            self._vectors = data["vectors"]
            self._use_faiss = False
        
        logger.info(
            f"Vector index loaded: {self.size} vectors "
            f"({len(self.deleted_ids)} soft-deleted)"
        )
    
    def rebuild_from_embeddings(self, embeddings: List[Tuple[str, np.ndarray]]):
        """
        Rebuild entire index from list of (node_id, embedding) tuples.
        Used when loading from SQLite.
        
        Always performs a FULL REPLACEMENT — never appends to existing index.
        """
        # BUG-2 fix: explicitly reset all state before rebuilding
        if self._use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self._vectors = []
        self.id_map = {}
        self.reverse_map = {}
        self.deleted_ids.clear()

        vectors = []
        ids = []

        for node_id, embedding in embeddings:
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized = embedding / norm
            else:
                normalized = embedding
            vectors.append(normalized)
            ids.append(node_id)

        if vectors:
            if self._use_faiss:
                vectors_array = np.vstack(vectors).astype(np.float32)
                self.index.add(vectors_array)
            else:
                self._vectors = vectors

            for idx, nid in enumerate(ids):
                self.id_map[idx] = nid
                self.reverse_map[nid] = idx

        logger.info(f"Vector index rebuilt with {len(embeddings)} embeddings")
    
    def clear(self):
        """Clear all vectors from index."""
        if self._use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self._vectors = []
        self.id_map = {}
        self.reverse_map = {}
        self.deleted_ids = set()
    
    def force_compact(self):
        """Force compaction regardless of threshold."""
        self._compact()
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.total_size,
            "active_vectors": self.size,
            "deleted_vectors": len(self.deleted_ids),
            "deletion_ratio": round(self.deletion_ratio, 4),
            "deletion_threshold": self.deletion_threshold,
            "dimension": self.dimension,
            "using_faiss": self._use_faiss
        }
