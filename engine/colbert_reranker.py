"""
ColBERT MaxSim late-interaction re-rank stage (opt-in, off by default).

Enable with: HYBRIDMIND_COLBERT_ENABLED=true
Requires: bge-m3 (BAAI/bge-m3) for native colbert output.

At query time: encode the query with bge-m3 to get per-token colbert vectors,
then re-rank candidates using MaxSim (sum of max cosine per query token).

Storage: per-node colbert vectors are stored in <mind>/colbert/*.npz
by the ingest pipeline when colbert is enabled.

Heavy storage (~100-200KB/node). Off by default.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _colbert_enabled() -> bool:
    try:
        from config import settings
        return bool(settings.colbert_enabled)
    except Exception:
        return False


def _get_colbert_store():
    try:
        from api.dependencies import get_colbert_store
        return get_colbert_store()
    except Exception:
        return None


def colbert_maxsim_rerank(
    query_text: str,
    candidates: List[Dict[str, Any]],
    embedding_engine,
    top_k: Optional[int] = None,
    alpha: float = 0.3,  # weight of ColBERT score in final blend
) -> List[Dict[str, Any]]:
    """
    Re-rank candidates using ColBERT MaxSim late interaction.

    1. Encode the query with bge-m3 → per-token colbert vectors.
    2. Load stored colbert vectors for each candidate from the colbert store.
    3. Compute MaxSim(query, candidate) for each.
    4. Blend with the existing combined_score: alpha * maxsim + (1-alpha) * combined.
    """
    store = _get_colbert_store()
    if store is None:
        return candidates

    try:
        # Encode query as colbert tokens
        hybrid = embedding_engine.embed_hybrid([query_text], return_colbert=True)
        query_colberts = hybrid.get("colbert")
        if not query_colberts or query_colberts[0] is None:
            return candidates
        q_colbert = np.asarray(query_colberts[0], dtype=np.float32)
        if len(q_colbert) == 0:
            return candidates

        # Normalize query tokens
        q_norm = q_colbert / (np.linalg.norm(q_colbert, axis=1, keepdims=True) + 1e-8)

        from storage.colbert_store import compute_maxsim

        for cand in candidates:
            nid = cand.get("node_id", "")
            c_colbert = store.get(nid)
            if c_colbert is not None and len(c_colbert) > 0:
                c_norm = c_colbert / (np.linalg.norm(c_colbert, axis=1, keepdims=True) + 1e-8)
                sim = q_norm @ c_norm.T
                maxsim = float(np.mean(np.max(sim, axis=1)))
                cand["colbert_score"] = maxsim
                # Blend with existing combined_score
                combined = cand.get("combined_score", 0.0)
                cand["combined_score"] = alpha * maxsim + (1.0 - alpha) * combined
                cand["reasoning"] = (
                    cand.get("reasoning", "") +
                    f" + ColBERT(maxsim={maxsim:.4f}, α={alpha})"
                )
            else:
                cand["colbert_score"] = 0.0

        candidates.sort(key=lambda x: -x.get("combined_score", 0.0))
        return candidates[:top_k] if top_k else candidates

    except Exception as exc:
        logger.warning(
            "ColBERT MaxSim rerank failed; returning unchanged type=%s",
            type(exc).__name__,
        )
        return candidates[:top_k] if top_k else candidates
