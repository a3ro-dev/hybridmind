"""
Remote image embedding client for HybridMind visual memory.

Connects to a separately-deployed ColQwen2.5 embedding server
(RunPod Serverless or local FastAPI). Returns per-patch ColBERT-style
vectors for image nodes and query vectors for MaxSim scoring.

Configure via:
  HYBRIDMIND_IMAGE_EMBEDDING_URL=https://api.runpod.ai/v2/{endpoint_id}/runsync
  HYBRIDMIND_IMAGE_RUNPOD_KEY=<your_runpod_key>  (optional, for RunPod auth)

If HYBRIDMIND_IMAGE_EMBEDDING_URL is not set, get_image_embedding_engine()
returns None and all image paths degrade gracefully.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_IMAGE_EMB_URL = os.getenv("HYBRIDMIND_IMAGE_EMBEDDING_URL", "")
_IMAGE_RUNPOD_KEY = os.getenv("HYBRIDMIND_IMAGE_RUNPOD_KEY", "")

_INSTANCE: Optional["RemoteImageEmbeddingEngine"] = None


class RemoteImageEmbeddingEngine:
    """
    HTTP client for a remote ColQwen2.5 embedding service.

    The remote server exposes:
      POST /embed_image  {"image_b64": "..."} → {"patch_vectors": [[...], ...]}
      POST /embed_query  {"query": "..."}     → {"query_vectors": [[...], ...]}
      GET  /health                            → {"status": "ok"}

    For RunPod Serverless the URL format is:
      https://api.runpod.ai/v2/{endpoint_id}/runsync
    and the payload wraps the above in {"input": {...}}.
    """

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._is_runpod = "runpod.ai" in base_url
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
        )

    def _post(self, path: str, payload: dict) -> dict:
        """POST to the server, handling RunPod's wrapping format."""
        if self._is_runpod:
            # RunPod Serverless: POST to base_url with {"input": payload}
            resp = self._client.post(self.base_url, json={"input": payload})
        else:
            resp = self._client.post(f"{self.base_url}{path}", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # RunPod wraps output in {"output": ...}
        return data.get("output", data)

    def embed_image(self, image_b64: str) -> Optional[np.ndarray]:
        """
        Compute ColQwen2.5 patch vectors for an image.

        Returns ndarray of shape (num_patches, dim) or None on failure.
        """
        try:
            result = self._post("/embed_image", {"image_b64": image_b64})
            patch_vectors = result.get("patch_vectors")
            if patch_vectors is None:
                logger.warning("image_embedding: server returned no patch_vectors")
                return None
            return np.asarray(patch_vectors, dtype=np.float32)
        except Exception as e:
            logger.error(f"image_embedding: embed_image failed: {e}")
            return None

    def embed_query(self, query_text: str) -> Optional[np.ndarray]:
        """
        Compute ColQwen2.5 query patch vectors for a text query.

        Returns ndarray of shape (num_query_patches, dim) or None on failure.
        """
        try:
            result = self._post("/embed_query", {"query": query_text})
            query_vectors = result.get("query_vectors")
            if query_vectors is None:
                logger.warning("image_embedding: server returned no query_vectors")
                return None
            return np.asarray(query_vectors, dtype=np.float32)
        except Exception as e:
            logger.error(f"image_embedding: embed_query failed: {e}")
            return None

    def health(self) -> bool:
        """Ping the server health endpoint. Returns True if alive."""
        try:
            if self._is_runpod:
                resp = self._client.get(self.base_url.replace("/runsync", "/health"))
            else:
                resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False


def get_image_embedding_engine() -> Optional[RemoteImageEmbeddingEngine]:
    """
    Singleton factory. Returns None if HYBRIDMIND_IMAGE_EMBEDDING_URL is not set.
    """
    global _INSTANCE
    if not _IMAGE_EMB_URL:
        return None
    if _INSTANCE is None:
        _INSTANCE = RemoteImageEmbeddingEngine(
            base_url=_IMAGE_EMB_URL,
            api_key=_IMAGE_RUNPOD_KEY,
        )
        logger.info(f"image_embedding: engine initialized → {_IMAGE_EMB_URL}")
    return _INSTANCE
