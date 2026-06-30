"""
Train the FusionScorer MLP head for HybridMind hybrid retrieval.

This is a learning-to-rank script intended to run on RunPod (GPU) after you have
accumulated ground-truth labels from LoCoMo / LongMemEval / MuSiQue eval runs.

Approach: pairwise logistic loss (Bradley-Terry / RankNet variant).
For each query, positive passages (ground-truth) are preferred over negatives.
Feature vector: [dense_score, bm25_score, graph_score, dense_rank, bm25_rank,
                 graph_rank, query_type_one_hot×4]   →  dim = 10

Usage:
    python scripts/train_fusion_mlp.py \
        --data-path benchmarks/fusion_train_data.jsonl \
        --output models/fusion_scorer.npz \
        --epochs 100 \
        --lr 0.01

Data format (JSONL, one JSON object per line):
    {
      "query_id": "locomo_0001",
      "query_type": "factoid",
      "candidates": [
        {"node_id": "...", "dense_score": 0.82, "bm25_score": 0.4,
         "graph_score": 0.1, "is_relevant": true},
        ...
      ]
    }
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_fusion_mlp")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_features(cand: dict, dense_rank: int, bm25_rank: int, graph_rank: int,
                   total: int, query_type: str) -> np.ndarray:
    from engine.fusion import _build_feature_vector
    return _build_feature_vector(
        dense_score=float(cand.get("dense_score", 0)),
        bm25_score=float(cand.get("bm25_score", 0)),
        graph_score=float(cand.get("graph_score", 0)),
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        graph_rank=graph_rank,
        total_candidates=total,
        query_type=query_type,
    )


def train(data_path: str, output_path: str, epochs: int = 100, lr: float = 0.01,
          hidden: int = 8, seed: int = 42):
    from engine.fusion import FusionScorer, _FULL_DIM

    logger.info(f"Loading training data from {data_path}")
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} queries")

    # Build training pairs (positive, negative feature vectors)
    pairs = []
    for rec in records:
        qt = rec.get("query_type", "factoid")
        cands = rec.get("candidates", [])
        if not cands:
            continue

        total = len(cands)
        # Sort by dense_score to get ranks
        dense_sorted = sorted(range(total), key=lambda i: -cands[i].get("dense_score", 0))
        bm25_sorted = sorted(range(total), key=lambda i: -cands[i].get("bm25_score", 0))
        graph_sorted = sorted(range(total), key=lambda i: -cands[i].get("graph_score", 0))
        dense_rank = {i: r + 1 for r, i in enumerate(dense_sorted)}
        bm25_rank = {i: r + 1 for r, i in enumerate(bm25_sorted)}
        graph_rank = {i: r + 1 for r, i in enumerate(graph_sorted)}

        positives = [i for i, c in enumerate(cands) if c.get("is_relevant")]
        negatives = [i for i, c in enumerate(cands) if not c.get("is_relevant")]

        for pi in positives:
            for ni in negatives:
                fp = build_features(cands[pi], dense_rank[pi], bm25_rank[pi], graph_rank[pi], total, qt)
                fn = build_features(cands[ni], dense_rank[ni], bm25_rank[ni], graph_rank[ni], total, qt)
                pairs.append((fp, fn))

    logger.info(f"Built {len(pairs)} training pairs")
    if not pairs:
        logger.error("No training pairs found. Check data format.")
        sys.exit(1)

    # Initialize weights
    rng = np.random.default_rng(seed)
    W1 = rng.normal(scale=0.1, size=(hidden, _FULL_DIM)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.normal(scale=0.1, size=(1, hidden)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)

    for epoch in range(epochs):
        total_loss = 0.0
        rng.shuffle(pairs)
        for fp, fn in pairs:
            # Forward pass
            h_pos = np.maximum(0, W1 @ fp + b1)
            h_neg = np.maximum(0, W1 @ fn + b1)
            s_pos = (W2 @ h_pos + b2)[0]
            s_neg = (W2 @ h_neg + b2)[0]
            diff = s_pos - s_neg
            prob = sigmoid(np.array([diff]))[0]
            loss = -np.log(prob + 1e-9)
            total_loss += float(loss)

            # Backward pass (RankNet gradient)
            d_diff = prob - 1.0
            # Gradient w.r.t. W2, b2
            dW2 = d_diff * h_pos.reshape(1, -1) - d_diff * h_neg.reshape(1, -1)
            db2 = np.array([d_diff])
            # Gradient through hidden layer
            dh_pos = d_diff * W2.T.reshape(-1)
            dh_neg = -d_diff * W2.T.reshape(-1)
            dh_pos *= (h_pos > 0).astype(np.float32)
            dh_neg *= (h_neg > 0).astype(np.float32)
            dW1 = dh_pos.reshape(-1, 1) @ fp.reshape(1, -1) + dh_neg.reshape(-1, 1) @ fn.reshape(1, -1)
            db1 = dh_pos + dh_neg
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(pairs):.4f}")

    scorer = FusionScorer()
    scorer._W1, scorer._b1, scorer._W2, scorer._b2 = W1, b1, W2, b2
    scorer._loaded = True
    scorer.save(output_path)
    logger.info(f"Saved FusionScorer checkpoint to {output_path}")
    logger.info(
        "Set HYBRIDMIND_FUSION_MODEL=<path> and HYBRIDMIND_FUSION_MODE=mlp to use it."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="benchmarks/fusion_train_data.jsonl")
    parser.add_argument("--output", default="models/fusion_scorer.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train(args.data_path, args.output, args.epochs, args.lr, args.hidden, args.seed)
