"""
Embedding pipeline for HybridMind.
Generates vector embeddings using sentence-transformers locally.
384-dimensional MiniLM-L6-v2 embeddings for reliable, fast, offline operation.
"""

import logging
import os
import time
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Default embedding model and dimension (sentence-transformers)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIMENSION = 384


class EmbeddingEngine:
    """
    Embedding generation using local sentence-transformers.
    Returns 384-dimensional vectors.
    No API dependencies — works fully offline.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize embedding engine with local sentence-transformers.

        Args:
            model_name: Sentence-transformers model name
            device: Device for inference ('cpu', 'cuda', None for auto)
            cache_folder: Cache folder for model downloads
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        self._cache_folder = cache_folder

    def _ensure_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        kwargs = {}
        if self._device:
            kwargs["device"] = self._device
        logger.info(f"Loading local embedding model: {self.model_name}...")
        t0 = time.perf_counter()
        self._model = SentenceTransformer(self.model_name, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Local embedding model loaded: {self.model_name} "
            f"(dim={dim}, device={self._model.device}, {elapsed:.0f}ms)"
        )

    @property
    def dimension(self) -> int:
        """Get embedding dimension (384 for all-MiniLM-L6-v2)."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return _DEFAULT_DIMENSION

    @property
    def is_available(self) -> bool:
        """Always available (fully local)."""
        self._ensure_model()
        return self._model is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector in-place."""
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            return (vec / norm).astype(np.float32)
        return vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            normalize: Whether to L2-normalize (default True)

        Returns:
            Embedding vector of shape (384,)
        """
        self._ensure_model()
        vec = self._model.encode(text, normalize_embeddings=False)
        if normalize:
            vec = self._normalize(vec)
        return vec.astype(np.float32)

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding (default 64)
            show_progress: Show progress bar

        Returns:
            Array of embedding vectors (num_texts × 384)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=False,
        )
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    def embed_with_graph_context(
        self,
        text: str,
        neighbor_embeddings: List[np.ndarray],
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Generate a graph-conditioned embedding: alpha*V + (1-alpha)*G_mean.
        """
        own_embedding = self.embed(text, normalize=False)
        if getattr(self, "disable_neighborhood_averaging", False) or not neighbor_embeddings:
            return self._normalize(own_embedding)

        neighbor_mean = np.mean(neighbor_embeddings, axis=0)
        final = alpha * own_embedding + (1.0 - alpha) * neighbor_mean

        final_normed = self._normalize(final)

        # Debug: report conditioning effect
        own_normed = self._normalize(own_embedding.copy())
        cosine_diff = 1.0 - float(np.dot(own_normed, final_normed))
        logger.debug(
            f"Graph conditioning: {len(neighbor_embeddings)} neighbors, "
            f"alpha={alpha}, cosine_diff={cosine_diff:.4f}"
        )
        return final_normed

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def compute_similarity_batch(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and multiple embeddings."""
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            return np.zeros(len(embeddings))

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        embeddings_normalized = embeddings / norms

        return np.dot(embeddings_normalized, query_normalized)

    # Legacy compatibility: expose a .model attribute that some callers check
    @property
    def model(self):
        """Legacy compatibility — returns self."""
        self._ensure_model()
        return self


# Singleton instance for shared use
_embedding_engine: Optional[EmbeddingEngine] = None


def get_embedding_engine(
    model_name: str = _DEFAULT_MODEL
) -> EmbeddingEngine:
    """Get or create embedding engine singleton."""
    global _embedding_engine

    if _embedding_engine is None or _embedding_engine.model_name != model_name:
        _embedding_engine = EmbeddingEngine(model_name=model_name)

    return _embedding_engine
