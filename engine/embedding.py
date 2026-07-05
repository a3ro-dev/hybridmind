"""
Embedding pipeline for HybridMind.

Backends (selected automatically, in priority order):
1. TEIEmbeddingEngine: when RUNPOD_TEI_EMBEDDING_URL is set. Calls a
   self-hosted HuggingFace TEI /embed endpoint (e.g. our own RunPod
   Qwen3-Embedding-8B deployment). Native output dim (4096 for that model,
   no MRL truncation). Falls back to local bge-m3 on HTTP errors.
2. RemoteEmbeddingEngine: when HC_EMBEDDING_URL or RUNPOD_EMBEDDING_URL is
   set. Calls a remote OpenAI-compatible /v1/embeddings endpoint (e.g. Hack
   Club AI). Falls back to local bge-m3 on HTTP errors.
3. EmbeddingEngine (local):
  - bge-m3  (BAAI/bge-m3*)  : FlagEmbedding BGEM3FlagModel, 1024-dim,
    dense + sparse (lexical weights) + colbert (per-token) natively.
  - SentenceTransformer (*): all other models, e.g. all-mpnet-base-v2 (768-dim).

Env vars:
  RUNPOD_TEI_EMBEDDING_URL — self-hosted RunPod TEI base URL (raw HF TEI
                             protocol, e.g. https://<id>.api.runpod.ai)
  RUNPOD_API_KEY           — RunPod account API key (Bearer token)
  HC_EMBEDDING_URL      — Hack Club AI base URL (https://ai.hackclub.com/proxy/v1)
  HC_API_KEY            — Hack Club AI API key
  RUNPOD_EMBEDDING_URL  — fallback OpenAI-compatible RunPod base URL (legacy)
  HYBRIDMIND_EMBEDDING_DIMENSION — output dimension; default 4096 (native
                                   Qwen3-Embedding-8B dim via TEI). Set to
                                   1024 if using the MRL-truncated OpenAI-
                                   compatible path instead.
"""

import logging
import os
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


class RemoteEmbeddingEngine:
    """
    Remote embedding backend via any OpenAI-compatible /v1/embeddings endpoint.

    Configured with:
      HC_EMBEDDING_URL   — base URL (Hack Club AI: https://ai.hackclub.com/proxy/v1)
      HC_API_KEY         — bearer token
      HYBRIDMIND_EMBEDDING_DIMENSION — MRL output dim (default: 1024)

    Falls back to the local EmbeddingEngine on any HTTP error.
    """

    def __init__(self, base_url: str, api_key: str, dimension: int = 1024, model: str = "qwen/qwen3-embedding-8b"):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._dimension = dimension
        self.model_name = model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
        )
        self._fallback: Optional[EmbeddingEngine] = None
        logger.info(f"RemoteEmbeddingEngine: {self.base_url} model={model} dim={dimension}")

    def _get_fallback(self) -> EmbeddingEngine:
        if self._fallback is None:
            logger.warning("RunPod embedding: falling back to local bge-m3")
            self._fallback = EmbeddingEngine(model_name=_DEFAULT_MODEL)
        return self._fallback

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_available(self) -> bool:
        return True

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0 else vec

    def _call_api(self, texts: List[str]) -> Optional[np.ndarray]:
        payload: dict = {"model": self.model_name, "input": texts, "dimensions": self._dimension}
        try:
            resp = self._client.post(f"{self.base_url}/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            return vecs / norms
        except Exception as e:
            logger.error(f"RunPod embedding API error: {e}")
            return None

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        result = self._call_api([text])
        if result is not None:
            return result[0]
        return self._get_fallback().embed(text, normalize=normalize)

    def embed_batch(self, texts: List[str], normalize: bool = True, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            result = self._call_api(chunk)
            if result is not None:
                all_vecs.append(result)
            else:
                fb = self._get_fallback().embed_batch(chunk, normalize=normalize, batch_size=batch_size)
                all_vecs.append(fb)
        return np.vstack(all_vecs).astype(np.float32)

    def embed_hybrid(self, texts: List[str], batch_size: int = 32, return_colbert: bool = False) -> Dict:
        # RunPod endpoint returns dense only; sparse/colbert not available remotely
        dense = self.embed_batch(texts, normalize=True, batch_size=batch_size)
        return {"dense": dense, "sparse": None, "colbert": None}

    def embed_with_graph_context(self, text: str, neighbor_embeddings: List[np.ndarray], alpha: float = 0.7) -> np.ndarray:
        own = self.embed(text, normalize=False)
        if not neighbor_embeddings:
            return self._normalize(own)
        neighbor_mean = np.mean(neighbor_embeddings, axis=0)
        return self._normalize(alpha * own + (1.0 - alpha) * neighbor_mean)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(embedding1), np.linalg.norm(embedding2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (n1 * n2))

    def compute_similarity_batch(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        qn = np.linalg.norm(query_embedding)
        if qn == 0:
            return np.zeros(len(embeddings))
        q = query_embedding / qn
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return np.dot(embeddings / norms, q)

    @property
    def model(self):
        return self


class TEIEmbeddingEngine:
    """
    Remote embedding backend for a self-hosted HuggingFace TEI (text-embeddings-
    inference) deployment, e.g. Qwen3-Embedding-8B on our own RunPod pod.

    TEI's raw /embed protocol differs from OpenAI's /v1/embeddings:
      - Request:  {"inputs": "<str>"} or {"inputs": ["<str>", ...]}
        (no "model"/"dimensions" fields — the deployed model's native output
        dimension is always returned, no MRL truncation.)
      - Response: a plain List[List[float]], order-preserved, no wrapper
        object and no per-item "index" to sort by (unlike OpenAI's schema).

    Configured with:
      RUNPOD_TEI_EMBEDDING_URL — base URL (e.g. https://<id>.api.runpod.ai)
      RUNPOD_API_KEY           — bearer token
      HYBRIDMIND_EMBEDDING_DIMENSION — informational only (TEI doesn't take
                                         a dimension param); default 4096.

    Falls back to the local EmbeddingEngine on any HTTP error.
    """

    def __init__(self, base_url: str, api_key: str, dimension: int = 4096):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._dimension = dimension
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
        )
        self._fallback: Optional[EmbeddingEngine] = None
        logger.info(f"TEIEmbeddingEngine: {self.base_url} dim={dimension}")

    def _get_fallback(self) -> EmbeddingEngine:
        if self._fallback is None:
            logger.warning("TEI embedding: falling back to local bge-m3")
            self._fallback = EmbeddingEngine(model_name=_DEFAULT_MODEL)
        return self._fallback

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_available(self) -> bool:
        return True

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0 else vec

    def _call_api(self, texts: List[str]) -> Optional[np.ndarray]:
        try:
            resp = self._client.post(f"{self.base_url}/embed", json={"inputs": texts})
            resp.raise_for_status()
            vecs = np.array(resp.json(), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            return vecs / norms
        except Exception as e:
            logger.error(f"TEI embedding API error: {e}")
            return None

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        result = self._call_api([text])
        if result is not None:
            return result[0]
        return self._get_fallback().embed(text, normalize=normalize)

    def embed_batch(self, texts: List[str], normalize: bool = True, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            result = self._call_api(chunk)
            if result is not None:
                all_vecs.append(result)
            else:
                fb = self._get_fallback().embed_batch(chunk, normalize=normalize, batch_size=batch_size)
                all_vecs.append(fb)
        return np.vstack(all_vecs).astype(np.float32)

    def embed_hybrid(self, texts: List[str], batch_size: int = 32, return_colbert: bool = False) -> Dict:
        # TEI returns dense only; sparse/colbert not available remotely
        dense = self.embed_batch(texts, normalize=True, batch_size=batch_size)
        return {"dense": dense, "sparse": None, "colbert": None}

    def embed_with_graph_context(self, text: str, neighbor_embeddings: List[np.ndarray], alpha: float = 0.7) -> np.ndarray:
        own = self.embed(text, normalize=False)
        if not neighbor_embeddings:
            return self._normalize(own)
        neighbor_mean = np.mean(neighbor_embeddings, axis=0)
        return self._normalize(alpha * own + (1.0 - alpha) * neighbor_mean)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(embedding1), np.linalg.norm(embedding2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (n1 * n2))

    def compute_similarity_batch(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        qn = np.linalg.norm(query_embedding)
        if qn == 0:
            return np.zeros(len(embeddings))
        q = query_embedding / qn
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return np.dot(embeddings / norms, q)

    @property
    def model(self):
        return self


# Singleton instance for shared use
_embedding_engine = None


def get_embedding_engine(model_name: str = _DEFAULT_MODEL, device: Optional[str] = None):
    """
    Return the process-wide embedding engine singleton.

    Priority:
    1. TEIEmbeddingEngine if RUNPOD_TEI_EMBEDDING_URL is set (self-hosted TEI)
    2. RemoteEmbeddingEngine if HC_EMBEDDING_URL or RUNPOD_EMBEDDING_URL is set
    3. Local EmbeddingEngine (bge-m3 or specified model_name)
    """
    global _embedding_engine

    tei_url = os.getenv("RUNPOD_TEI_EMBEDDING_URL", "").strip()
    if tei_url:
        if not isinstance(_embedding_engine, TEIEmbeddingEngine):
            api_key = os.getenv("RUNPOD_API_KEY", "")
            dim = int(os.getenv("HYBRIDMIND_EMBEDDING_DIMENSION", "4096"))
            _embedding_engine = TEIEmbeddingEngine(base_url=tei_url, api_key=api_key, dimension=dim)
        return _embedding_engine

    remote_url = (
        os.getenv("HC_EMBEDDING_URL", "").strip()
        or os.getenv("RUNPOD_EMBEDDING_URL", "").strip()
    )
    if remote_url:
        if not isinstance(_embedding_engine, RemoteEmbeddingEngine):
            api_key = os.getenv("HC_API_KEY") or os.getenv("RUNPOD_API_KEY", "")
            dim = int(os.getenv("HYBRIDMIND_EMBEDDING_DIMENSION", "1024"))
            remote_model = os.getenv("HYBRIDMIND_REMOTE_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
            _embedding_engine = RemoteEmbeddingEngine(base_url=remote_url, api_key=api_key, dimension=dim, model=remote_model)
        return _embedding_engine

    if _embedding_engine is None or (
        isinstance(_embedding_engine, EmbeddingEngine) and _embedding_engine.model_name != model_name
    ):
        _embedding_engine = EmbeddingEngine(model_name=model_name, device=device)
    return _embedding_engine
