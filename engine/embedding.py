"""
Embedding pipeline for HybridMind.
Generates vector embeddings using qwen/qwen3-embedding-8b via HackClub
OpenAI-compatible API (d=4096).

Falls back to mock embeddings if API is unavailable.
"""

import logging
import os
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import openai for API-based embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning(
        "openai package not available. "
        "Using mock embeddings. Install with: pip install openai"
    )

# Default embedding model and dimension
_DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
_DEFAULT_DIMENSION = 4096
_DEFAULT_BASE_URL = "https://ai.hackclub.com"


class EmbeddingEngine:
    """
    Embedding generation using qwen3-embedding-8b via HackClub OpenAI-compatible API.
    Returns 4096-dimensional vectors.
    Falls back to mock embeddings if API is unavailable.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,  # kept for API compatibility; ignored
        cache_folder: Optional[str] = None  # kept for API compatibility; ignored
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: Embedding model name (default: qwen/qwen3-embedding-8b)
            device: Ignored (kept for backward compat)
            cache_folder: Ignored (kept for backward compat)
        """
        self.model_name = model_name
        self._dimension: int = _DEFAULT_DIMENSION

        # Resolve API credentials
        api_key = (
            os.getenv("HACKCLUB_API_KEY")
            or os.getenv("HC_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        base_url = os.getenv("OPENAI_BASE_URL") or _DEFAULT_BASE_URL

        self._client: Optional["OpenAI"] = None
        if OPENAI_AVAILABLE and api_key:
            try:
                self._client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    f"EmbeddingEngine initialized: model={model_name}, "
                    f"base_url={base_url}, dimension={self._dimension}"
                )
            except Exception as e:
                logger.error(f"Failed to create OpenAI client: {e}")
                self._client = None
        else:
            if not OPENAI_AVAILABLE:
                logger.warning("openai package not installed — using mock embeddings")
            else:
                logger.warning("No API key found — using mock embeddings")

    @property
    def dimension(self) -> int:
        """Get embedding dimension (always 4096 for qwen3-embedding-8b)."""
        return self._dimension

    @property
    def is_available(self) -> bool:
        """Check if embedding client is available."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, texts: List[str]) -> List[np.ndarray]:
        """Call the embedding API for a list of texts."""
        if self._client is None:
            return [self._mock_embed(t) for t in texts]

        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            # Sort by index to preserve order (API may reorder)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}. Falling back to mock.")
            return [self._mock_embed(t) for t in texts]

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector in-place."""
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            return (vec / norm).astype(np.float32)
        return vec

    def _mock_embed(self, text: str) -> np.ndarray:
        """
        Generate a deterministic mock embedding from text hash.
        Provides stable embeddings for testing without an API key.
        """
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16) % (2**32))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        return self._normalize(embedding)

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
            Embedding vector of shape (4096,)
        """
        import concurrent.futures
        from config import settings
        timeout = getattr(settings, "embedding_timeout_seconds", 60)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._do_embed, text, normalize)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                try:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=503, detail="Embedding model unavailable")
                except ImportError:
                    raise TimeoutError("Embedding API timed out")

    def _do_embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """Actual embedding logic (no timeout wrapper)."""
        vectors = self._call_api([text])
        vec = vectors[0]
        if normalize:
            vec = self._normalize(vec)
        return vec

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            normalize: Whether to L2-normalize embeddings
            batch_size: API batch size
            show_progress: Show progress bar (currently ignored)

        Returns:
            Array of embedding vectors (num_texts × 4096)
        """
        if not texts:
            return np.array([]).reshape(0, self._dimension)

        all_vectors: List[np.ndarray] = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs = self._call_api(batch)
            if normalize:
                vecs = [self._normalize(v) for v in vecs]
            all_vectors.extend(vecs)

        return np.vstack(all_vectors).astype(np.float32)

    def embed_with_graph_context(
        self,
        text: str,
        neighbor_embeddings: List[np.ndarray],
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Generate a graph-conditioned embedding: alpha*V + (1-alpha)*G_mean.
        """
        own_embedding = self._do_embed(text, normalize=False)
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
        """Legacy compatibility — returns self if client is available."""
        return self if self._client is not None else None


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
