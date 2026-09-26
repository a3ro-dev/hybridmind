"""
Reranking stage for HybridMind.

Re-scores the top hybrid candidates with a stronger relevance model before the
final top-k slice. Two backends, selected by the RERANK_MODE env var:

- "cross" (opt-in): local sentence-transformers CrossEncoder using
  ``settings.reranker_model``. Runs fully offline, with no hosted LLM load,
  ~50-250ms for ~25 candidates. Auto-uses CUDA if available.
- "tei": a self-hosted HuggingFace TEI ``/rerank`` endpoint serving a
  sequence-classification cross-encoder (default ``BAAI/bge-reranker-v2-m3``).
  RunPod URLs are bound to ``RUNPOD_API_KEY``; loopback URLs are keyless.
- "llm": listwise reranking through the centralized LLM policy. The research
  proxy remains unavailable unless explicitly enabled.
- "off": no reranking (passthrough).

Failures preserve the original order but are marked explicitly on every result.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from engine.device import resolve_device as _resolve_device


def _default_cross_model() -> str:
    """Single source of truth for the cross-encoder model: config.settings.reranker_model."""
    from config import settings
    return settings.reranker_model


def _rerank_mode() -> str:
    from config import settings
    mode = settings.rerank_mode.strip().lower()
    if mode not in {"off", "cross", "tei", "llm"}:
        raise ValueError("rerank_mode must be one of: off, cross, tei, llm")
    return mode


def _validate_inputs(query: str, candidates: List[Dict[str, Any]]) -> None:
    from config import settings

    if not isinstance(query, str) or not query.strip():
        raise ValueError("reranker query must be a non-empty string")
    if len(query) > settings.reranker_max_query_chars:
        raise ValueError("reranker query exceeds configured character limit")
    if len(candidates) > settings.reranker_max_pairs:
        raise ValueError("reranker candidate pool exceeds configured pair limit")
    for candidate in candidates:
        text = candidate.get("text", "")
        if not isinstance(text, str):
            raise ValueError("reranker candidate text must be a string")
        if len(text) > settings.reranker_max_text_chars:
            raise ValueError("reranker candidate text exceeds configured character limit")


def _mark_failure(
    candidates: List[Dict[str, Any]],
    failure_type: str,
    top_k: Optional[int],
) -> List[Dict[str, Any]]:
    for candidate in candidates:
        candidate["rerank_applied"] = False
        candidate["rerank_failure_type"] = failure_type
    return candidates[:top_k] if top_k else candidates


def _apply_scores(
    candidates: List[Dict[str, Any]],
    scores: Any,
    top_k: Optional[int],
) -> List[Dict[str, Any]]:
    """Validate model scores and blend them with the fusion score.

    ``rerank_score`` keeps the raw model score; ``combined_score`` becomes the
    configured convex blend of min-max normalized fusion and model scores.
    """
    # Normalize reranker scores to [0, 1]
    s_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s_arr.shape != (len(candidates),):
        raise ValueError(
            f"cross-encoder returned {s_arr.shape[0]} scores for "
            f"{len(candidates)} candidates"
        )
    if not np.all(np.isfinite(s_arr)):
        raise ValueError("cross-encoder returned non-finite scores")
    s_min, s_max = s_arr.min(), s_arr.max()
    if s_max > s_min:
        s_norm = (s_arr - s_min) / (s_max - s_min)
    else:
        s_norm = np.ones_like(s_arr) * 0.5

    # Normalize combined_scores to [0, 1] for fair blending
    cs_arr = np.array([c.get("combined_score", 0.0) for c in candidates])
    if not np.all(np.isfinite(cs_arr)):
        raise ValueError("candidate fusion scores contain non-finite values")
    cs_min, cs_max = cs_arr.min(), cs_arr.max()
    if cs_max > cs_min:
        cs_norm = (cs_arr - cs_min) / (cs_max - cs_min)
    else:
        cs_norm = np.ones_like(cs_arr) * 0.5

    from config import settings

    blend_total = settings.rerank_rrf_weight + settings.rerank_cross_encoder_weight
    if (
        settings.rerank_rrf_weight < 0
        or settings.rerank_cross_encoder_weight < 0
        or blend_total <= 0
    ):
        raise ValueError("reranker blend weights must be non-negative with a positive sum")
    rrf_weight = settings.rerank_rrf_weight / blend_total
    cross_weight = settings.rerank_cross_encoder_weight / blend_total

    for c, s, sn, csn in zip(candidates, s_arr, s_norm, cs_norm):
        c["rerank_score"] = float(s)
        c["rerank_applied"] = True
        c["combined_score"] = (
            rrf_weight * float(csn)
            + cross_weight * float(sn)
        )

    ranked = sorted(candidates, key=lambda c: -c.get("combined_score", 0.0))
    return ranked[:top_k] if top_k else ranked


class CrossEncoderReranker:
    """Local cross-encoder reranker (query, passage) -> relevance score."""

    enabled = True

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
        self._ensure_model()
        self._model.predict([("warmup query", "warmup passage")])

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        try:
            _validate_inputs(query, candidates)
            self._ensure_model()
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = self._model.predict(pairs, batch_size=32)

            return _apply_scores(candidates, scores, top_k)
        except Exception as e:
            logger.warning("Cross-encoder rerank failed (%s)", type(e).__name__)
            return _mark_failure(candidates, type(e).__name__, top_k)


class TEIReranker:
    """Cross-encoder served by HuggingFace TEI ``POST /rerank`` (API 1.9.4).

    Request ``{query, texts, raw_scores: false, truncate: false}``; the response
    is a list of ``{index, score}``. Every candidate index must come back
    exactly once with a finite score, otherwise the pool is marked failed.
    """

    enabled = True

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60.0, client: Any = None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model_name = _default_cross_model()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

    def warmup(self):
        self._score("warmup query", ["warmup passage"])

    def _score(self, query: str, texts: List[str]) -> np.ndarray:
        from engine.serverless_util import retry_transient

        def _once(chunk: List[str]) -> np.ndarray:
            resp = self._client.post(
                f"{self.base_url}/rerank",
                json={"query": query, "texts": chunk, "raw_scores": False, "truncate": False},
            )
            resp.raise_for_status()
            scores = np.full(len(chunk), np.nan)
            for item in resp.json():
                index = int(item["index"])
                if not 0 <= index < len(chunk) or not np.isnan(scores[index]):
                    raise ValueError("TEI /rerank returned an out-of-range or duplicate index")
                scores[index] = float(item["score"])
            if np.isnan(scores).any():
                raise ValueError("TEI /rerank did not score every candidate")
            return scores

        # Pairs are scored independently, so chunking at TEI's default
        # max-client-batch-size (32) gives the same scores as one request.
        return np.concatenate([
            retry_transient(lambda c=texts[i:i + 32]: _once(c), label="TEI /rerank")
            for i in range(0, len(texts), 32)
        ])

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        try:
            _validate_inputs(query, candidates)
            scores = self._score(query, [c.get("text", "") for c in candidates])
            return _apply_scores(candidates, scores, top_k)
        except Exception as e:
            logger.warning("TEI rerank failed (%s)", type(e).__name__)
            return _mark_failure(candidates, type(e).__name__, top_k)


def _tei_reranker_endpoint() -> tuple[str, str]:
    """Bind the configured TEI rerank URL to its credential (or none if loopback)."""
    from config import settings
    from engine.provider_policy import validate_loopback_url, validate_provider_url

    url = settings.reranker_tei_url.strip()
    if not url:
        raise RuntimeError("rerank_mode=tei requires HYBRIDMIND_RERANKER_TEI_URL")
    try:
        return validate_loopback_url(url, "TEI reranker"), ""
    except ValueError:
        pass
    api_key = settings.runpod_api_key.strip()
    if not api_key:
        raise RuntimeError("a non-loopback TEI reranker URL requires RUNPOD_API_KEY")
    return validate_provider_url(url, "runpod", allow_custom=settings.allow_custom_provider_urls), api_key


class LLMReranker:
    """Listwise reranker through the centralized provider policy."""

    enabled = True

    def __init__(self, model: Optional[str] = None):
        self.model = model

    def warmup(self):
        return  # nothing to preload

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates[:top_k] if top_k else candidates
        try:
            _validate_inputs(query, candidates)
            from config import settings
            from engine import llm_client

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
            content = llm_client.chat_completion(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user},
                ],
                model=settings.qa_model,
                max_tokens=512,
                temperature=0.0,
                preferred="research_proxy" if settings.allow_research_proxy else "zai",
                allow_fallback=False,
            )
            if not content:
                raise RuntimeError("LLM reranker returned no content")
            start, end = content.find("["), content.rfind("]") + 1
            if start < 0 or end <= start:
                raise ValueError("LLM reranker response did not contain an array")
            order = json.loads(content[start:end])
            expected = list(range(len(candidates)))
            if (
                not isinstance(order, list)
                or any(type(index) is not int for index in order)
                or len(order) != len(expected)
                or sorted(order) != expected
            ):
                raise ValueError("LLM reranker response is not a complete permutation")

            ranked: List[Dict[str, Any]] = []
            for idx in order:
                candidate = candidates[idx]
                candidate["rerank_score"] = float(len(candidates) - len(ranked))
                candidate["rerank_applied"] = True
                ranked.append(candidate)
            return ranked[:top_k] if top_k else ranked
        except Exception as e:
            logger.warning("LLM rerank failed (%s)", type(e).__name__)
            return _mark_failure(candidates, type(e).__name__, top_k)


class _NoOpReranker:
    enabled = False

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
        elif mode == "tei":
            url, api_key = _tei_reranker_endpoint()
            r = TEIReranker(url, api_key=api_key)
        else:
            # settings.reranker_model (HYBRIDMIND_RERANKER_MODEL) is the single source of truth.
            r = CrossEncoderReranker(model_name=_default_cross_model())
        r._mode = mode
        _reranker = r
    return _reranker
