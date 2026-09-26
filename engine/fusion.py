"""
Score fusion strategies for HybridMind hybrid retrieval.

Strategies:
- rrf    : Reciprocal Rank Fusion — rank-based, no weight tuning needed.
- linear : Original linear weighted sum (kept for A/B comparison and back-compat).
- mlp    : FusionScorer MLP head — ships with a heuristic init that mimics RRF;
           loads a trained checkpoint when HYBRIDMIND_FUSION_MODEL is set.
- dbsf / zscore / weighted_linear (+ minmax_normalize): pure score-based
           fusion ports (Qdrant DBSF, opsem z-score, EMG query-local min-max);
           opt-in only, see DEVIATIONS.

Usage:
    from engine.fusion import get_fusion_fn, fuse
    fuse_fn = get_fusion_fn()               # reads config
    score = fuse_fn(signals)                # signals: dict of name → (score, rank)
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Mapping, Optional, Tuple

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
    if not isinstance(k, int) or k < 0:
        raise ValueError("RRF k must be a non-negative integer")
    scores: Dict[str, float] = {}
    for signal_name, items in rank_lists.items():
        if not items:
            continue
        weight = signal_weights.get(signal_name, 1.0) if signal_weights else 1.0
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"RRF weight for {signal_name!r} must be finite and non-negative")
        seen: set[str] = set()
        for rank, (node_id, _score) in enumerate(items, start=1):
            if node_id in seen:
                raise ValueError(f"RRF rank list {signal_name!r} contains duplicate ID {node_id!r}")
            seen.add(node_id)
            scores[node_id] = scores.get(node_id, 0.0) + weight * (1.0 / (k + rank))
    return scores


# --------------------------------------------------------------------- #
# Score-based fusion (opt-in; RRF k=60 stays the default)
# --------------------------------------------------------------------- #
#
# dbsf_fuse is a port of Qdrant's distribution-based score fusion:
#   https://github.com/qdrant/qdrant @ 6ab21cac18ebb6f4ae29102c7f8f5cc11affd5de
#   lib/segment/src/common/score_fusion.rs (score_fusion, distr_norm, norm,
#   welfords_mean_variance). SPDX-License-Identifier: Apache-2.0.
#   Copyright 2026 Qdrant Solutions GmbH.
#
# zscore_fuse follows opsem's z-score convex fusion:
#   https://github.com/Chrislysen/opsem @ 68186a45882dd85ea66fcc70ee38ba28f6de9a90
#   tune11_multiencoder.py (_z and the alpha*z(bm25) + (1-alpha)*z(dense) sum).
#   SPDX-License-Identifier: MIT. Copyright (c) 2026 Christian Lysenstøen.
#
# minmax_normalize / weighted_linear_fuse follow EMG (entity memory graph):
#   https://github.com/Sun668/em_graph_memory @ f020e855be06ac9f33ec888945ff6b305d81cb07
#   code/em_graph/recall/retrieval.py (normalize_semantic_scores mode
#   "query_local_minmax_v1"; entity_weight*E + semantic_weight*S, 0.30/0.70).
#   SPDX-License-Identifier: MIT. Copyright (c) 2026 Sun668.

DEVIATIONS: List[str] = [
    "dbsf_fuse: float64 arithmetic instead of Qdrant's f32, so normalized values can differ in the last f32 ulp.",
    "dbsf_fuse: lists are keyed by channel name and weighted by name (absent weight = 1.0); Qdrant weights are "
    "positional with the same 1.0 default.",
    "dbsf_fuse: Welford runs over each list in (score desc, node_id) order, the order of a sorted Qdrant result list.",
    "dbsf_fuse/zscore_fuse/weighted_linear_fuse: return {node_id: score}; the final ordering and tie-break "
    "belong to ScopedCorpus.sort_scores (chronological index), not to Qdrant's ScoredPoint ordering.",
    "zscore_fuse: opsem z-scores a complete per-conversation score vector. Here each channel's missing "
    "candidates are first filled with that channel's minimum observed raw score, then z is computed over the "
    "union of candidates. With complete score maps (every scope turn scored) this is exactly opsem.",
    "zscore_fuse: an empty channel map is a constant vector and contributes z=0 to every candidate.",
    "zscore_fuse: weights are applied as given ({channel: w}, absent = 1.0); opsem hard-codes alpha and 1-alpha. "
    "Pass weights summing to 1 for the convex form.",
    "minmax_normalize: every value must be finite (EMG only checks the min and max).",
    "All score-based fusers reject non-finite scores and non-finite or negative weights (upstream does not validate).",
]

# EMG's default fusion weights mapped onto HybridMind channel names
# (EMG entity channel -> "graph", EMG embedding channel -> "dense").
EMG_LINEAR_WEIGHTS: Dict[str, float] = {"graph": 0.30, "dense": 0.70}

_Z_STD_FLOOR = 1e-9  # opsem: z = 0 when the population std is below this


def _validated_weights(weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, value in (weights or {}).items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"fusion weight for {name!r} must be finite and non-negative")
        out[name] = value
    return out


def _validated_scores(name: str, scores: Mapping[str, float]) -> Dict[str, float]:
    out = {str(node_id): float(value) for node_id, value in scores.items()}
    for node_id, value in out.items():
        if not math.isfinite(value):
            raise ValueError(f"score for {node_id!r} in {name!r} must be finite")
    return out


def _weighted_sum(
    normalized: Mapping[str, Mapping[str, float]], weights: Mapping[str, float]
) -> Dict[str, float]:
    """Sum weighted per-channel scores in sorted channel/node order (stable float sums)."""
    fused: Dict[str, float] = {}
    for name in sorted(normalized):
        weight = weights.get(name, 1.0)
        for node_id in sorted(normalized[name]):
            fused[node_id] = fused.get(node_id, 0.0) + weight * normalized[name][node_id]
    return dict(sorted(fused.items()))


def _distr_norm(scores: Mapping[str, float]) -> Dict[str, float]:
    """Qdrant ``distr_norm``: (s - (mu - 3 sigma)) / (6 sigma), sample sigma, unclipped."""
    if len(scores) < 2:
        return {node_id: 0.5 for node_id in scores}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    # Welford's one-pass mean / sample variance, as in welfords_mean_variance.
    mean = 0.0
    aggregate = 0.0
    for k, (_node_id, value) in enumerate(ordered, start=1):
        old_delta = value - mean
        mean += old_delta / k
        aggregate += old_delta * (value - mean)
    std_dev = math.sqrt(aggregate / (len(ordered) - 1))
    low = mean - 3.0 * std_dev
    high = mean + 3.0 * std_dev
    if low == high:  # Qdrant norm(): "Protect against division by zero"
        return {node_id: 0.5 for node_id in scores}
    return {node_id: (value - low) / (high - low) for node_id, value in scores.items()}


def dbsf_fuse(
    score_maps: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Distribution-based score fusion (Qdrant DBSF).

    Each channel's returned scores are normalized on that list's own mean and
    sample standard deviation to ``(s - (mu - 3 sigma)) / (6 sigma)`` without
    clipping; a single-element list maps to 0.5 and a zero-variance list maps
    every member to 0.5. Normalized scores are multiplied by the channel weight
    and summed; a node absent from a channel gets 0 from it.

    Returns ``{node_id: fused_score}`` (higher is better).
    """
    w = _validated_weights(weights)
    normalized = {name: _distr_norm(_validated_scores(name, scores)) for name, scores in score_maps.items()}
    return _weighted_sum(normalized, w)


def _zscore(scores: Mapping[str, float]) -> Dict[str, float]:
    """opsem ``_z``: population std; all zeros when std < 1e-9."""
    ids = list(scores)
    values = np.asarray([scores[i] for i in ids], dtype=float)
    std = values.std() if values.size else 0.0
    if std < _Z_STD_FLOOR:
        return {i: 0.0 for i in ids}
    z = (values - values.mean()) / std
    return {i: float(v) for i, v in zip(ids, z)}


def zscore_fuse(
    score_maps: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Weighted z-score fusion (opsem), convex when the weights sum to 1.

    Every channel is z-normalized over the union of candidates across all
    channels. A candidate a channel did not score takes that channel's minimum
    observed raw score (so a channel's non-matches sit at its floor rather than
    at an arbitrary 0); an empty channel contributes 0. Pass complete score
    maps (every scope turn) to reproduce opsem exactly.
    """
    w = _validated_weights(weights)
    clean = {name: _validated_scores(name, scores) for name, scores in score_maps.items()}
    union = sorted({node_id for scores in clean.values() for node_id in scores})
    normalized: Dict[str, Dict[str, float]] = {}
    for name, scores in clean.items():
        if not scores:
            normalized[name] = {node_id: 0.0 for node_id in union}
            continue
        floor = min(scores.values())
        normalized[name] = _zscore({node_id: scores.get(node_id, floor) for node_id in union})
    return _weighted_sum(normalized, w)


def minmax_normalize(scores: Mapping[str, float]) -> Dict[str, float]:
    """EMG ``query_local_minmax_v1``: (s - min) / (max - min); span <= 0 -> all 0.0."""
    clean = _validated_scores("minmax", scores)
    if not clean:
        return {}
    low = min(clean.values())
    span = max(clean.values()) - low
    if span <= 0.0:
        return {node_id: 0.0 for node_id in clean}
    return {node_id: (value - low) / span for node_id, value in clean.items()}


def weighted_linear_fuse(
    score_maps: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """``sum_c w_c * s_c(node)`` over pre-normalized scores; a missing score is 0.

    EMG uses ``0.30 * entity + 0.70 * embedding`` (``EMG_LINEAR_WEIGHTS``)
    after min-max normalizing the embedding scores.
    """
    w = _validated_weights(weights)
    clean = {name: _validated_scores(name, scores) for name, scores in score_maps.items()}
    return _weighted_sum(clean, w)


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
        except Exception as exc:
            logger.warning(
                "FusionScorer: checkpoint load failed type=%s; using heuristic init",
                type(exc).__name__,
            )
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
    if query_type == "default":
        query_type = "factoid"
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
