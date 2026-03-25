"""
Embedding pipeline for HybridMind.
Generates vector embeddings using sentence-transformers.
"""

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. "
        "Using mock embeddings. Install with: pip install sentence-transformers"
    )


class EmbeddingEngine:
    """
    Embedding generation using sentence-transformers.
    Falls back to mock embeddings if library not available.
    Auto-detects GPU for faster inference.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize embedding engine.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run model on ('cpu', 'cuda', 'auto'). If None, auto-detects GPU.
            cache_folder: Folder to cache model files
        """
        self.model_name = model_name
        self._model = None
        self._cache_folder = cache_folder
        self._dimension: Optional[int] = None
        
        # Auto-detect GPU if device not specified
        if device is None or device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} - using CUDA")
                else:
                    self._device = "cpu"
                    logger.info("No GPU detected - using CPU")
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device
        
        # Default dimensions for known models
        self._known_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L6-v2": 384,
            "paraphrase-mpnet-base-v2": 768,
        }
    
    @property
    def model(self) -> Optional["SentenceTransformer"]:
        """Lazy load the model."""
        if self._model is None and TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self._device,
                    cache_folder=self._cache_folder
                )
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._model = None
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is not None:
            return self._dimension
        
        # Try to get from loaded model
        if self.model is not None:
            return self._dimension
        
        # Fall back to known dimensions
        return self._known_dimensions.get(self.model_name, 384)
    
    @property
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self.model is not None
    
    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text with a timeout.
        """
        import concurrent.futures
        from config import settings
        from fastapi import HTTPException
        
        timeout = getattr(settings, "embedding_timeout_seconds", 30)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._do_embed, text, normalize)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise HTTPException(status_code=503, detail="Embedding model unavailable")

    def _do_embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Actual embedding generation logic.
        """
        if self.model is not None:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embedding.astype(np.float32)
        else:
            # Mock embedding for testing without model
            return self._mock_embed(text, normalize)
            
    def embed_with_graph_context(
        self, 
        text: str, 
        neighbor_embeddings: List[np.ndarray], 
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Generate a graph-conditioned embedding using CRS logic: alpha * V + beta * G.
        """
        own_embedding = self._do_embed(text, normalize=False)
        if not neighbor_embeddings:
            norm = np.linalg.norm(own_embedding)
            if norm > 0:
                return own_embedding / norm
            return own_embedding
            
        neighbor_mean = np.mean(neighbor_embeddings, axis=0)
        final = alpha * own_embedding + (1.0 - alpha) * neighbor_mean

        norm = np.linalg.norm(final)
        if norm > 0:
            final_normed = (final / norm).astype(np.float32)
        else:
            final_normed = final.astype(np.float32)

        # Debug: report conditioning effect
        own_norm_val = np.linalg.norm(own_embedding)
        if own_norm_val > 0:
            own_normed = own_embedding / own_norm_val
            cosine_diff = 1.0 - float(np.dot(own_normed, final_normed))
        else:
            cosine_diff = 0.0
        logger.debug(
            f"Graph conditioning: {len(neighbor_embeddings)} neighbors, "
            f"alpha={alpha}, cosine_diff={cosine_diff:.4f}"
        )

        return final_normed
    
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
            batch_size: Processing batch size
            show_progress: Show progress bar
            
        Returns:
            Array of embedding vectors (num_texts x dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        if self.model is not None:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress
            )
            return embeddings.astype(np.float32)
        else:
            # Mock embeddings for testing
            return np.vstack([self._mock_embed(t, normalize) for t in texts])
    
    def _mock_embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate mock embedding based on text hash.
        Provides deterministic embeddings for testing.
        
        Args:
            text: Input text
            normalize: Whether to normalize
            
        Returns:
            Mock embedding vector
        """
        # Use hash for deterministic results
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Generate pseudo-random vector from hash
        np.random.seed(int(text_hash[:8], 16) % (2**32))
        embedding = np.random.randn(self.dimension).astype(np.float32)
        
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Normalize if needed
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
        """
        Compute cosine similarity between query and multiple embeddings.
        
        Args:
            query_embedding: Query vector (dimension,)
            embeddings: Matrix of embeddings (n x dimension)
            
        Returns:
            Array of similarity scores (n,)
        """
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            return np.zeros(len(embeddings))
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        embeddings_normalized = embeddings / norms
        
        # Compute dot products
        similarities = np.dot(embeddings_normalized, query_normalized)
        
        return similarities


# Singleton instance for shared use
_embedding_engine: Optional[EmbeddingEngine] = None


def get_embedding_engine(
    model_name: str = "all-MiniLM-L6-v2"
) -> EmbeddingEngine:
    """Get or create embedding engine singleton."""
    global _embedding_engine
    
    if _embedding_engine is None or _embedding_engine.model_name != model_name:
        _embedding_engine = EmbeddingEngine(model_name=model_name)
    
    return _embedding_engine

