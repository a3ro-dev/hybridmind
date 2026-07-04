"""
Reranking stage for HybridMind.

Re-scores the top hybrid candidates with a stronger relevance model before the
final top-k slice. Two backends, selected by the RERANK_MODE env var:

- "cross" (default): local sentence-transformers CrossEncoder
  (cross-encoder/ms-marco-MiniLM-L-6-v2). Runs fully offline, no HC proxy load,
  ~50-250ms for ~25 candidates. Auto-uses CUDA if available.
- "llm": listwise reranking via a fast HackClub model. Adds proxy
  latency/dependency; kept as an A/B fallback.
- "off": no reranking (passthrough).

All rerankers fail open: any error returns the candidates unchanged.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from engine.device import resolve_device as _resolve_device


def _default_cross_model() -> str:
    """Single source of truth for the cross-encoder model: config.settings.reranker_model."""
    from config import settings
    return settings.reranker_model


_LLM_MODEL = os.getenv("RERANK_LLM_MODEL", "~google/gemini-flash-latest")


def _rerank_mode() -> str:
    return os.getenv("RERANK_MODE", "cross").strip().lower()


class CrossEncoderReranker:
    """Local cross-encoder reranker (query, passage) -> relevance score."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or _default_cross_model()
        self._model = None
        self._device = device

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        device = self._device if self._device is not None else _resolve_device("auto")
        logger.info(f"Loading cross-encoder reranker: {self.model_name} (device={device})...")
        self._model = CrossEncoder(self.model_name, device=device)
        logger.info("Cross-encoder reranker loaded")

    def warmup(self):
        """Pre-load the model and run one tiny prediction (called at startup)."""
        try:
            self._ensure_model()
            self._model.predict([("warmup query", "warmup passage")])
        except Exception as e:
            logger.warning(f"Cross-encoder warmup failed: {e}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        try:
            self._ensure_model()
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = self._model.predict(pairs, batch_size=32)

            # Normalize reranker scores to [0, 1]
            s_arr = np.array([float(s) for s in scores])
            s_min, s_max = s_arr.min(), s_arr.max()
            if s_max > s_min:
                s_norm = (s_arr - s_min) / (s_max - s_min)
            else:
                s_norm = np.ones_like(s_arr) * 0.5

            # Normalize combined_scores to [0, 1] for fair blending
            cs_arr = np.array([c.get("combined_score", 0.0) for c in candidates])
            cs_min, cs_max = cs_arr.min(), cs_arr.max()
            if cs_max > cs_min:
                cs_norm = (cs_arr - cs_min) / (cs_max - cs_min)
            else:
                cs_norm = np.ones_like(cs_arr) * 0.5

            for c, s, sn, csn in zip(candidates, s_arr, s_norm, cs_norm):
                c["rerank_score"] = float(s)
                c["combined_score"] = 0.7 * float(csn) + 0.3 * float(sn)

            ranked = sorted(candidates, key=lambda c: -c.get("combined_score", 0.0))
            return ranked[:top_k] if top_k else ranked
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed, returning unchanged: {e}")
            return candidates[:top_k] if top_k else candidates


class LLMReranker:
    """Listwise reranker via a fast HackClub model (A/B fallback)."""

    def __init__(self, model: str = _LLM_MODEL):
        self.model = model
        self._api_key = os.getenv("HC_API_KEY", "")
        self._base_url = os.getenv(
            "OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1"
        ).rstrip("/")

    def warmup(self):
        return  # nothing to preload

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates or not self._api_key:
            return candidates[:top_k] if top_k else candidates
        try:
            import httpx

            listing = "\n".join(
                f"[{i}] {c.get('text', '')[:300]}" for i, c in enumerate(candidates)
            )
            sys_prompt = (
                "You are a search reranker. Given a query and a numbered list of "
                "passages, return ONLY a JSON array of passage indices ordered from "
                "most to least relevant to the query. Include every index exactly once. "
                "Example: [3,0,5,1,2,4]"
            )
            user = f"Query: {query}\n\nPassages:\n{listing}"
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            start, end = content.find("["), content.rfind("]") + 1
            order = json.loads(content[start:end]) if start >= 0 and end > start else []

            # Validate: keep only valid, unique indices; append any the LLM dropped.
            seen = set()
            ranked: List[Dict[str, Any]] = []
            for idx in order:
                if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                    seen.add(idx)
                    c = candidates[idx]
                    c["rerank_score"] = float(len(candidates) - len(ranked))
                    ranked.append(c)
            for i, c in enumerate(candidates):
                if i not in seen:
                    ranked.append(c)
            return ranked[:top_k] if top_k else ranked
        except Exception as e:
            logger.warning(f"LLM rerank failed, returning unchanged: {e}")
            return candidates[:top_k] if top_k else candidates


class _NoOpReranker:
    def warmup(self):
        return

    def rerank(self, query, candidates, top_k=None):
        return candidates[:top_k] if top_k else candidates


_reranker = None


def get_reranker():
    """Get the process-wide reranker singleton based on RERANK_MODE."""
    global _reranker
    mode = _rerank_mode()
    # Rebuild if mode changed (supports A/B toggling between server restarts).
    if _reranker is None or getattr(_reranker, "_mode", None) != mode:
        if mode == "off":
            r = _NoOpReranker()
        elif mode == "llm":
            r = LLMReranker()
        else:
            # settings.reranker_model (HYBRIDMIND_RERANKER_MODEL) is the single source of truth.
            r = CrossEncoderReranker(model_name=_default_cross_model())
        r._mode = mode
        _reranker = r
    return _reranker
