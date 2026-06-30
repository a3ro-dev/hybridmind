"""
Score fusion strategies for HybridMind hybrid retrieval.

Strategies:
- rrf    : Reciprocal Rank Fusion — rank-based, no weight tuning needed.
- linear : Original linear weighted sum (kept for A/B comparison and back-compat).
- mlp    : FusionScorer MLP head — ships with a heuristic init that mimics RRF;
           loads a trained checkpoint when HYBRIDMIND_FUSION_MODEL is set.

Usage:
    from engine.fusion import get_fusion_fn, fuse
    fuse_fn = get_fusion_fn()               # reads config
    score = fuse_fn(signals)                # signals: dict of name → (score, rank)
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------- #

def rrf_fuse(
    rank_lists: Dict[str, List[Tuple[str, float]]],
    k: int = 60,
    signal_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion over multiple per-signal rank lists.

    Args:
        rank_lists     : {signal_name: [(node_id, score), ...]} sorted descending by score.
        k              : RRF smoothing constant (higher = flatter penalty curve).
        signal_weights : Optional per-signal multiplier (e.g. {"dense": 0.3, "graph": 0.9}).
                         When absent, all signals contribute equally.

    Returns:
        {node_id: rrf_score}  — higher is better; not bounded to [0,1].
    """
    scores: Dict[str, float] = {}
    for signal_name, items in rank_lists.items():
        if not items:
            continue
        weight = signal_weights.get(signal_name, 1.0) if signal_weights else 1.0
        for rank, (node_id, _score) in enumerate(items, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + weight * (1.0 / (k + rank))
    return scores


# --------------------------------------------------------------------- #
# Feature-based MLP fusion head
# --------------------------------------------------------------------- #

_FEATURE_NAMES = [
    "dense_score",    # cosine / vector score
    "bm25_score",     # BM25 keyword overlap boost
    "graph_score",    # graph proximity score
    "dense_rank",     # rank in dense list (normalised 0-1)
    "bm25_rank",      # rank in bm25 list
    "graph_rank",     # rank in graph list
    # query type one-hot (added dynamically when query_type is provided)
    # "qt_factoid", "qt_temporal", "qt_multihop", "qt_entity"
]

_QUERY_TYPES = ["factoid", "temporal", "multihop", "entity"]
_FULL_DIM = len(_FEATURE_NAMES) + len(_QUERY_TYPES)  # 10


class FusionScorer:
    """
    Lightweight MLP (2-layer, ~200 params) that predicts relevance from
    per-candidate fusion features.

    Ships with a heuristic weight init that approximates RRF/linear so it
    works correctly without training.  When HYBRIDMIND_FUSION_MODEL points
    at a .npz checkpoint, the trained weights are loaded instead.

    Training:  scripts/train_fusion_mlp.py  (RunPod)
    """

    HIDDEN_DIM = 8

    def __init__(self, checkpoint_path: Optional[str] = None):
        self._W1: Optional[np.ndarray] = None  # (HIDDEN_DIM, _FULL_DIM)
        self._b1: Optional[np.ndarray] = None  # (HIDDEN_DIM,)
        self._W2: Optional[np.ndarray] = None  # (1, HIDDEN_DIM)
        self._b2: Optional[np.ndarray] = None  # (1,)
        self._loaded = False
        if checkpoint_path:
            self._load(checkpoint_path)
        else:
            self._init_heuristic()

    def _init_heuristic(self):
        """Weights that approximate RRF without any training."""
        # Input dim: [dense_score, bm25_score, graph_score, dense_rank, bm25_rank, graph_rank, qt_*]
        # Hidden: relu(x @ W1.T + b1); output: sigmoid(h @ W2.T + b2)
        rng = np.random.default_rng(42)
        scale = 0.1
        W1 = rng.normal(scale=scale, size=(self.HIDDEN_DIM, _FULL_DIM)).astype(np.float32)
        # Bias hidden units to activate on positive features (scores)
        b1 = np.full(self.HIDDEN_DIM, 0.1, dtype=np.float32)
        # Uniform output — fall back to average of hidden units
        W2 = np.ones((1, self.HIDDEN_DIM), dtype=np.float32) / self.HIDDEN_DIM
        b2 = np.zeros(1, dtype=np.float32)
        self._W1, self._b1, self._W2, self._b2 = W1, b1, W2, b2
        self._loaded = True

    def _load(self, path: str):
        try:
            data = np.load(path)
            self._W1 = data["W1"].astype(np.float32)
            self._b1 = data["b1"].astype(np.float32)
            self._W2 = data["W2"].astype(np.float32)
            self._b2 = data["b2"].astype(np.float32)
            self._loaded = True
            logger.info(f"FusionScorer: loaded checkpoint from {path}")
        except Exception as e:
            logger.warning(f"FusionScorer: failed to load checkpoint {path}: {e}. Using heuristic init.")
            self._init_heuristic()

    def score(self, feature_vector: np.ndarray) -> float:
        """Forward pass: feature_vector shape (_FULL_DIM,) → scalar."""
        x = feature_vector.reshape(-1).astype(np.float32)
        h = np.maximum(0, self._W1 @ x + self._b1)  # relu
        logit = (self._W2 @ h + self._b2)[0]
        return float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid

    def score_batch(self, features: np.ndarray) -> np.ndarray:
        """features shape (N, _FULL_DIM) → (N,)."""
        H = np.maximum(0, features @ self._W1.T + self._b1)  # (N, HIDDEN)
        logits = H @ self._W2.T + self._b2  # (N, 1)
        return (1.0 / (1.0 + np.exp(-logits))).reshape(-1)

    def save(self, path: str):
        np.savez(
            path,
            W1=self._W1, b1=self._b1,
            W2=self._W2, b2=self._b2,
        )
        logger.info(f"FusionScorer: saved checkpoint to {path}")


def _build_feature_vector(
    dense_score: float,
    bm25_score: float,
    graph_score: float,
    dense_rank: int,
    bm25_rank: int,
    graph_rank: int,
    total_candidates: int,
    query_type: str = "factoid",
) -> np.ndarray:
    qt_vec = np.zeros(len(_QUERY_TYPES), dtype=np.float32)
    if query_type in _QUERY_TYPES:
        qt_vec[_QUERY_TYPES.index(query_type)] = 1.0

    n = max(total_candidates, 1)
    base = np.array([
        dense_score,
        bm25_score,
        graph_score,
        1.0 - (dense_rank - 1) / n,  # normalised rank: 1 = top, 0 = bottom
        1.0 - (bm25_rank - 1) / n,
        1.0 - (graph_rank - 1) / n,
    ], dtype=np.float32)
    return np.concatenate([base, qt_vec])


# --------------------------------------------------------------------- #
# Public entry-point
# --------------------------------------------------------------------- #

_fusion_scorer: Optional[FusionScorer] = None


def get_fusion_mode() -> str:
    try:
        from config import settings
        return settings.fusion_mode
    except Exception:
        return "rrf"


def get_rrf_k() -> int:
    try:
        from config import settings
        return settings.fusion_rrf_k
    except Exception:
        return 60


def get_fusion_scorer() -> FusionScorer:
    global _fusion_scorer
    if _fusion_scorer is None:
        try:
            from config import settings
            ckpt = settings.fusion_model_path
        except Exception:
            ckpt = None
        _fusion_scorer = FusionScorer(checkpoint_path=ckpt)
    return _fusion_scorer
