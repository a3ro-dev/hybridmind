"""
Embedding pipeline for HybridMind.

Supports two backends, selected by model name:
- bge-m3  (BAAI/bge-m3*)  : FlagEmbedding BGEM3FlagModel, 1024-dim, returns
  dense + sparse (lexical weights) + colbert (per-token) vectors natively.
- SentenceTransformer (*): all other models, including the legacy
  all-mpnet-base-v2 (768-dim) for CPU-only deploys.

Default: BAAI/bge-m3 (1024-dim).  To keep the old model:
  HYBRIDMIND_EMBEDDING_MODEL=all-mpnet-base-v2
  HYBRIDMIND_EMBEDDING_DIMENSION=768
"""

import logging
import time
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-m3"
_DEFAULT_DIMENSION = 1024

_BGE_M3_PREFIX = "BAAI/bge-m3"


def _is_bge_m3(model_name: str) -> bool:
    return model_name.startswith(_BGE_M3_PREFIX)


class EmbeddingEngine:
    """
    Embedding generation for HybridMind.

    When model_name starts with "BAAI/bge-m3", uses FlagEmbedding BGEM3FlagModel
    (dense 1024-dim + native sparse/colbert).  All other names use SentenceTransformer.

    Both backends are fully local — no API dependencies.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        self.model_name = model_name
        self._model = None
        self._device = device
        self._cache_folder = cache_folder
        self._use_bge_m3 = _is_bge_m3(model_name)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        t0 = time.perf_counter()
        if self._use_bge_m3:
            self._load_bge_m3()
        else:
            self._load_sentence_transformer()
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Embedding model loaded: {self.model_name} "
            f"(dim={self.dimension}, device={self._device or 'auto'}, {elapsed:.0f}ms)"
        )

    def _load_bge_m3(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
            device = self._device or "cpu"
            use_fp16 = device != "cpu"
            logger.info(f"Loading BGEM3FlagModel: {self.model_name} (device={device}, fp16={use_fp16})")
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=use_fp16,
                device=device,
            )
            self._bge_native = True  # native tri-vector support
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed — loading bge-m3 via sentence-transformers "
                "(dense only, no native sparse/colbert). "
                "For full support: pip install FlagEmbedding>=1.2.10"
            )
            from sentence_transformers import SentenceTransformer
            kwargs: dict = {}
            if self._device:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._use_bge_m3 = False  # use SentenceTransformer encode path
            self._bge_native = False

    def _load_sentence_transformer(self):
        from sentence_transformers import SentenceTransformer
        kwargs: dict = {}
        if self._device:
            kwargs["device"] = self._device
        logger.info(f"Loading SentenceTransformer: {self.model_name}...")
        self._model = SentenceTransformer(self.model_name, **kwargs)

    # ------------------------------------------------------------------
    # Dimension
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        if self._model is not None:
            if self._use_bge_m3:
                return 1024
            return self._model.get_sentence_embedding_dimension()
        return _DEFAULT_DIMENSION if self._use_bge_m3 else 768

    @property
    def is_available(self) -> bool:
        self._ensure_model()
        return self._model is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            return (vec / norm).astype(np.float32)
        return vec

    # ------------------------------------------------------------------
    # Public API — dense embeddings (same interface regardless of backend)
    # ------------------------------------------------------------------

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate a dense embedding for a single text."""
        self._ensure_model()
        if self._use_bge_m3:
            out = self._model.encode(
                [text],
                batch_size=1,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        else:
            vec = self._model.encode(text, normalize_embeddings=False)
        if normalize:
            vec = self._normalize(vec)
        return vec.astype(np.float32)

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate dense embeddings for a list of texts."""
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        self._ensure_model()

        if self._use_bge_m3:
            out = self._model.encode(
                texts,
                batch_size=batch_size,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                show_progress_bar=show_progress,
            )
            embeddings = np.asarray(out["dense_vecs"], dtype=np.float32)
        else:
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

    def embed_hybrid(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_colbert: bool = False,
    ) -> Dict:
        """
        bge-m3 native hybrid output: dense + sparse + (optional) colbert.

        For SentenceTransformer models, only dense is returned (sparse=None,
        colbert=None).

        Returns dict with keys:
        - "dense"   : np.ndarray (N, 1024)
        - "sparse"  : list of dicts {token_id: weight} or None
        - "colbert" : list of np.ndarray (seq_len, 1024) or None
        """
        if not texts:
            return {"dense": np.array([]).reshape(0, self.dimension), "sparse": None, "colbert": None}

        self._ensure_model()

        if self._use_bge_m3:
            out = self._model.encode(
                texts,
                batch_size=batch_size,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=return_colbert,
            )
            dense = np.asarray(out["dense_vecs"], dtype=np.float32)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            dense = dense / norms
            return {
                "dense": dense,
                "sparse": out.get("lexical_weights"),
                "colbert": out.get("colbert_vecs") if return_colbert else None,
            }

        # SentenceTransformer fallback: dense only
        embeddings = self.embed_batch(texts, normalize=True, batch_size=batch_size)
        return {"dense": embeddings, "sparse": None, "colbert": None}

    # ------------------------------------------------------------------
    # Graph-conditioned embedding (neighbourhood averaging)
    # ------------------------------------------------------------------

    def embed_with_graph_context(
        self,
        text: str,
        neighbor_embeddings: List[np.ndarray],
        alpha: float = 0.7,
    ) -> np.ndarray:
        """Graph-conditioned embedding: alpha*own + (1-alpha)*mean(neighbours)."""
        own_embedding = self.embed(text, normalize=False)
        if getattr(self, "disable_neighborhood_averaging", False) or not neighbor_embeddings:
            return self._normalize(own_embedding)

        neighbor_mean = np.mean(neighbor_embeddings, axis=0)
        final = alpha * own_embedding + (1.0 - alpha) * neighbor_mean
        final_normed = self._normalize(final)

        own_normed = self._normalize(own_embedding.copy())
        cosine_diff = 1.0 - float(np.dot(own_normed, final_normed))
        logger.debug(
            f"Graph conditioning: {len(neighbor_embeddings)} neighbours, "
            f"alpha={alpha}, cosine_diff={cosine_diff:.4f}"
        )
        return final_normed

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def compute_similarity_batch(
        self, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            return np.zeros(len(embeddings))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        embeddings_normalized = embeddings / norms
        return np.dot(embeddings_normalized, query_normalized)

    # Legacy compatibility
    @property
    def model(self):
        self._ensure_model()
        return self


# Singleton instance for shared use
_embedding_engine: Optional[EmbeddingEngine] = None


def get_embedding_engine(model_name: str = _DEFAULT_MODEL) -> EmbeddingEngine:
    global _embedding_engine
    if _embedding_engine is None or _embedding_engine.model_name != model_name:
        _embedding_engine = EmbeddingEngine(model_name=model_name)
    return _embedding_engine
