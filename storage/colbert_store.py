"""
ColBERT per-token vector storage for MaxSim late interaction.

ColBERT vectors are per-token matrices (seq_len, dim) — too large and
variable to store efficiently in SQLite. This module stores them as
individual .npz files in <mind_root>/colbert/.

Enable: HYBRIDMIND_COLBERT_ENABLED=true
Requires: bge-m3 (BAAI/bge-m3) — the only model with native colbert output.

Storage is opt-in and heavy (~100-200KB per node). Off by default.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ColbertStore:
    """File-based storage for ColBERT per-token vectors."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir) / "colbert"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, node_id: str) -> Path:
        safe_id = node_id.replace("/", "_").replace("\\", "_")
        return self.root_dir / f"{safe_id}.npz"

    def add(self, node_id: str, colbert: np.ndarray) -> None:
        """Store colbert vectors for a node. colbert shape: (seq_len, dim)."""
        np.savez_compressed(self._path(node_id), colbert=colbert.astype(np.float32))

    def add_batch(self, items: List[tuple[str, np.ndarray]]) -> None:
        for node_id, colbert in items:
            self.add(node_id, colbert)

    def get(self, node_id: str) -> Optional[np.ndarray]:
        path = self._path(node_id)
        if not path.exists():
            return None
        try:
            return np.load(path)["colbert"]
        except Exception as e:
            logger.debug(f"ColbertStore: failed to load {node_id}: {e}")
            return None

    def has(self, node_id: str) -> bool:
        return self._path(node_id).exists()

    def remove(self, node_id: str) -> bool:
        path = self._path(node_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        for f in self.root_dir.glob("*.npz"):
            f.unlink()
        logger.info("ColbertStore: cleared all vectors")

    def size(self) -> int:
        return len(list(self.root_dir.glob("*.npz")))

    def total_bytes(self) -> int:
        return sum(
            f.stat().st_size for f in self.root_dir.glob("*.npz")
        )


def colbert_enabled() -> bool:
    try:
        from config import settings
        return bool(settings.colbert_enabled)
    except Exception:
        return False


def compute_maxsim(
    query_colbert: np.ndarray,     # (Q, dim)
    candidate_colbert: np.ndarray,  # (C, dim)
) -> float:
    """
    ColBERT MaxSim score: for each query token, find the max cosine similarity
    with any candidate token, then average.

    Both inputs must be L2-normalized.
    """
    q = query_colbert.astype(np.float32)
    c = candidate_colbert.astype(np.float32)

    # Normalize
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    c_norm = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)

    # Pairwise cosine: (Q, C)
    sim = q_norm @ c_norm.T
    # MaxSim: for each query token, max over candidates
    max_per_query = np.max(sim, axis=1)
    return float(np.mean(max_per_query))


def compute_maxsim_batch(
    query_colbert: np.ndarray,
    candidates: List[np.ndarray],
) -> np.ndarray:
    """Compute MaxSim scores for a query against multiple candidates."""
    scores = np.zeros(len(candidates), dtype=np.float32)
    for i, cand in enumerate(candidates):
        if cand is not None and len(cand) > 0:
            scores[i] = compute_maxsim(query_colbert, cand)
    return scores


def maybe_store_colbert(
    node_id: str,
    text: str,
    embedding_engine,
    colbert_store,
) -> bool:
    """
    Compute and store ColBERT per-token vectors for a node.

    Returns True if stored, False if colbert is disabled or failed.
    """
    if colbert_store is None or not colbert_enabled():
        return False
    try:
        hybrid = embedding_engine.embed_hybrid([text], return_colbert=True)
        colbert_list = hybrid.get("colbert")
        if colbert_list and len(colbert_list) > 0 and colbert_list[0] is not None:
            colbert_store.add(node_id, colbert_list[0])
            return True
    except Exception as e:
        logger.debug(f"ColBERT storage failed for {node_id}: {e}")
    return False
