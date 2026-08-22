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
        dimension: int = 4096,
        index_path: Optional[str] = None,
        deletion_threshold: float = 0.2,
        hnsw_ef_search: int = 64,
        hnsw_ef_construction: int = 40,
    ):
        """
        Initialize vector index.
        
        Args:
            dimension: Embedding dimension (fixed to 4096 in HybridMind)
            index_path: Path for index persistence
            deletion_threshold: Trigger compaction when this fraction is deleted (0.2 = 20%)
        """
        if dimension != 4096:
            raise ValueError("HybridMind VectorIndex requires dimension=4096")
        if not 1 <= hnsw_ef_search <= 4096:
            raise ValueError("hnsw_ef_search must be between 1 and 4096")
        if not 1 <= hnsw_ef_construction <= 4096:
            raise ValueError("hnsw_ef_construction must be between 1 and 4096")
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.deletion_threshold = deletion_threshold
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        
        # Mapping between FAISS indices and node IDs
        self.id_map: Dict[int, str] = {}  # FAISS idx -> node_id
        self.reverse_map: Dict[str, int] = {}  # node_id -> FAISS idx
        
        # HNSW cannot physically delete rows. Tombstones therefore identify
        # immutable vector *rows*, not logical node IDs. A node may be updated
        # and reinserted under the same ID; ID-level tombstones would hide the
        # replacement together with the stale generation.
        self.deleted_ids: Set[int] = set()
        
        # Raw vector cache for compaction (HNSW does not support reconstruct())
        self._raw_vectors: Dict[int, np.ndarray] = {}
        
        # Initialize index
        if FAISS_AVAILABLE:
            self.index = self._new_faiss_index()
            self._use_faiss = True
            logger.info(
                "FAISS HNSW vector index initialized: dimension=%s efSearch=%s "
                "efConstruction=%s",
                dimension,
                self.hnsw_ef_search,
                self.hnsw_ef_construction,
            )
        else:
            # Fallback to NumPy-based search
            self._vectors: List[np.ndarray] = []
            self._use_faiss = False
            logger.warning("FAISS not available, using NumPy fallback")
        
        # Load from disk if exists
        if self.index_path and self.index_path.exists():
            self.load()

    def _new_faiss_index(self):
        index = faiss.IndexHNSWFlat(
            self.dimension, 32, faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efSearch = self.hnsw_ef_search
        index.hnsw.efConstruction = self.hnsw_ef_construction
        return index
    
    @property
    def size(self) -> int:
        """Get number of vectors in index (excluding soft-deleted)."""
        return len(self.reverse_map)
    
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
        embedding = self._validate_embedding(node_id, embedding)
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
            # Cache raw vector for compaction (HNSW doesn't support reconstruct)
            self._raw_vectors[idx] = normalized.copy()
        else:
            self._vectors.append(normalized)
        
        # Update mappings
        self.id_map[idx] = node_id
        self.reverse_map[node_id] = idx

    def _validate_embedding(self, node_id: str, embedding: np.ndarray) -> np.ndarray:
        """Validate the exact vector contract before mutating index state."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1 or embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding for node {node_id!r} has shape {embedding.shape}; "
                f"HybridMind requires exactly ({self.dimension},). No dimension "
                "coercion or fallback is permitted."
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError(f"Embedding for node {node_id!r} contains non-finite values")
        return embedding
    
    def add_batch(self, nodes: List[Tuple[str, np.ndarray]]):
        """
        Add multiple vectors in batch (more efficient).
        
        Args:
            nodes: List of (node_id, embedding) tuples
        """
        if not nodes:
            return

        node_ids = [node_id for node_id, _ in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Vector batch contains duplicate node IDs")

        # Validate the whole batch before mutating the live index.
        validated = [
            (node_id, self._validate_embedding(node_id, embedding))
            for node_id, embedding in nodes
        ]

        # Remove existing entries first
        for node_id, _ in validated:
            if node_id in self.reverse_map:
                self.remove(node_id)
        
        # Prepare normalized vectors
        vectors = []
        for node_id, embedding in validated:
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
            # Cache raw vectors for compaction
            for i, v in enumerate(vectors):
                self._raw_vectors[start_idx + i] = v.copy()
        else:
            self._vectors.extend(vectors)
        
        # Update mappings
        for i, (node_id, _) in enumerate(validated):
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
        old_index = self.reverse_map.pop(node_id, None)
        if old_index is None:
            return False

        self.deleted_ids.add(old_index)

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
            if idx not in self.deleted_ids:
                if self._use_faiss:
                    # Use raw vector cache — HNSW does not support reconstruct()
                    vec = self._raw_vectors.get(idx)
                    if vec is None:
                        raise RuntimeError(
                            "vector compaction cannot reconstruct active FAISS row "
                            f"idx={idx}; refusing to drop derived-index data"
                        )
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
        """Build a complete replacement before swapping live vector state."""
        if len(vectors) != len(node_ids):
            raise ValueError("Vector rebuild row count does not match node ID count")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Vector rebuild contains duplicate node IDs")
        validated = [
            self._validate_embedding(node_id, vector)
            for node_id, vector in zip(node_ids, vectors)
        ]
        new_id_map = {index: node_id for index, node_id in enumerate(node_ids)}
        new_reverse_map = {node_id: index for index, node_id in enumerate(node_ids)}
        if self._use_faiss:
            new_index = self._new_faiss_index()
            if validated:
                new_index.add(np.vstack(validated).astype(np.float32))
            new_raw_vectors = {
                index: vector.copy() for index, vector in enumerate(validated)
            }
        else:
            new_vectors = [vector.copy() for vector in validated]

        if self._use_faiss:
            self.index = new_index
            self._raw_vectors = new_raw_vectors
        else:
            self._vectors = new_vectors
        self.id_map = new_id_map
        self.reverse_map = new_reverse_map
    
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
        query = self._validate_embedding("query", query_embedding)
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
                if node_id and idx not in self.deleted_ids:
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
                if node_id and idx not in self.deleted_ids:
                    sim_score = float(similarities[idx])
                    if sim_score >= min_score:
                        results.append((node_id, sim_score))
                        
                        if len(results) >= top_k:
                            break
        
        return results
    
    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        """Get vector by node ID."""
        if node_id not in self.reverse_map:
            return None
        
        idx = self.reverse_map[node_id]
        
        if self._use_faiss:
            # Use raw vector cache — HNSW does not support reconstruct()
            vec = self._raw_vectors.get(idx)
            if vec is not None:
                return vec.copy()
            return None
        else:
            return self._vectors[idx].copy()
    
    def has_vector(self, node_id: str) -> bool:
        """Check if node has a vector in the index."""
        return node_id in self.reverse_map
    
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
            "deleted_indices": self.deleted_ids,
            "use_faiss": self._use_faiss,
            "deletion_threshold": self.deletion_threshold,
            "raw_vectors": self._raw_vectors,
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
        
        persisted_dimension = data["dimension"]
        if persisted_dimension != 4096:
            raise ValueError(
                f"Persisted vector index dimension is {persisted_dimension}; expected 4096"
            )
        self.dimension = persisted_dimension
        self.id_map = {int(index): node_id for index, node_id in data["id_map"].items()}
        self.reverse_map = {
            str(node_id): int(index) for node_id, index in data["reverse_map"].items()
        }
        if "deleted_indices" in data:
            self.deleted_ids = {int(index) for index in data["deleted_indices"]}
        else:
            # Backward compatibility for snapshots whose tombstones were node
            # IDs. Convert each ID to the row that was current when saved.
            legacy_deleted_ids = {str(node_id) for node_id in data.get("deleted_ids", set())}
            self.deleted_ids = {
                index
                for index, node_id in self.id_map.items()
                if str(node_id) in legacy_deleted_ids
            }
            for node_id in legacy_deleted_ids:
                self.reverse_map.pop(node_id, None)
        self.deletion_threshold = data.get("deletion_threshold", 0.2)
        raw_vectors = data.get("raw_vectors", {})

        if data.get("use_faiss", False) and FAISS_AVAILABLE:
            faiss_path = load_path.with_suffix('.faiss')
            if faiss_path.exists():
                loaded_index = faiss.read_index(str(faiss_path))
                if loaded_index.d != self.dimension:
                    raise ValueError(
                        f"Persisted FAISS index dimension is {loaded_index.d}; "
                        f"expected {self.dimension}"
                    )
                self.index = loaded_index
                if not hasattr(self.index, "hnsw"):
                    raise ValueError("Persisted FAISS index is not an HNSW index")
                self.index.hnsw.efSearch = self.hnsw_ef_search
                self.index.hnsw.efConstruction = self.hnsw_ef_construction
                self._use_faiss = True
        elif "vectors" in data:
            self._vectors = [
                self._validate_embedding(self.id_map.get(index, str(index)), vector)
                for index, vector in enumerate(data["vectors"])
            ]
            self._use_faiss = False

        total_size = self.total_size
        expected_indices = set(range(total_size))
        if set(self.id_map) != expected_indices:
            raise ValueError("Persisted vector ID map does not match index rows")
        if not self.deleted_ids.issubset(expected_indices):
            raise ValueError("Persisted vector tombstones reference nonexistent rows")
        active_rows = expected_indices - self.deleted_ids
        expected_reverse = {self.id_map[index]: index for index in active_rows}
        if self.reverse_map != expected_reverse:
            raise ValueError("Persisted vector reverse map is inconsistent")
        validated_raw_vectors = {}
        for index, vector in raw_vectors.items():
            if int(index) not in expected_indices:
                raise ValueError("Persisted raw-vector cache references a nonexistent row")
            validated_raw_vectors[int(index)] = self._validate_embedding(
                self.id_map[int(index)], vector
            )
        self._raw_vectors = validated_raw_vectors
        
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
        # Validate everything before resetting the live index. A corrupt row
        # must not leave a partially rebuilt index behind.
        node_ids = [node_id for node_id, _ in embeddings]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Vector rebuild contains duplicate node IDs")
        validated = [
            (node_id, self._validate_embedding(node_id, embedding))
            for node_id, embedding in embeddings
        ]

        vectors = []
        ids = []

        for node_id, embedding in validated:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized = embedding / norm
            else:
                normalized = embedding
            vectors.append(normalized)
            ids.append(node_id)

        self._rebuild(vectors, ids)
        self.deleted_ids.clear()

        logger.info(f"Vector index rebuilt with {len(embeddings)} embeddings")
    
    def clear(self):
        """Clear all vectors from index."""
        if self._use_faiss:
            self.index = self._new_faiss_index()
            self._raw_vectors = {}
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
            "hnsw_ef_search": self.hnsw_ef_search if self._use_faiss else None,
            "hnsw_ef_construction": (
                self.hnsw_ef_construction if self._use_faiss else None
            ),
            "deletion_threshold": self.deletion_threshold,
            "dimension": self.dimension,
            "using_faiss": self._use_faiss
        }
