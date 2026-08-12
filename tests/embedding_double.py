"""Deterministic 4096-dimensional embedding test double.

This is not a runtime fallback. Test fixtures inject it before the application
is imported so unit tests remain offline while exercising the production index
dimension invariant.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


class Deterministic4096EmbeddingEngine:
    dimension = 4096
    model_name = "test/deterministic-4096"

    @property
    def model(self):
        return self

    def warmup(self, timeout_s=None):
        return self.embed("warmup")

    def embed(self, text: str) -> np.ndarray:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        features = normalized.split()
        compact = normalized.replace(" ", "_")
        features.extend(compact[i : i + 4] for i in range(max(0, len(compact) - 3)))

        vector = np.zeros(self.dimension, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_batch(self, texts):
        return np.stack([self.embed(text) for text in texts]).astype(np.float32)

    def embed_with_graph_context(self, text, neighbor_embeddings, alpha=0.7):
        base = self.embed(text)
        if not neighbor_embeddings:
            return base
        neighborhood = np.mean(np.asarray(neighbor_embeddings, dtype=np.float32), axis=0)
        combined = alpha * base + (1.0 - alpha) * neighborhood
        norm = float(np.linalg.norm(combined))
        return combined / norm if norm > 0 else combined

    def compute_similarity(self, embedding1, embedding2):
        a = np.asarray(embedding1, dtype=np.float32)
        b = np.asarray(embedding2, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
