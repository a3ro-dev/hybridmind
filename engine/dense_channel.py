"""Scope-local dense channel: exact cosine ranking plus an HNSW recall audit.

Exact search is the default dense channel for a ``ScopedCorpus``: scopes are
small (hundreds to low thousands of turns), so one matmul is cheap and there is
no ANN recall loss to explain away. ``HNSWScope`` and ``ann_audit`` exist to
measure what HNSW would lose at a given (M, efConstruction, efSearch, k).

These are pure functions over explicit vectors. They are dimension-agnostic so
offline harnesses can feed cached reference vectors (e.g. 1536-d EMG artifacts);
runtime callers must pass ``expected_dim=4096``.

Port attribution: the dense scoring (L2-normalised dot products over a memory
matrix) follows EMG, https://github.com/Sun668/em_graph_memory, commit
f020e855be06ac9f33ec888945ff6b305d81cb07, code/em_graph/recall/embedding_index.py
(``_l2_normalize``, ``MemoryEmbeddingIndex.scores``). SPDX: MIT,
Copyright (c) 2026 Sun668.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from engine.corpus import ScopedCorpus

DEVIATIONS = [
    "Zero-norm vectors raise; EMG clamps the norm at 1e-12 and keeps a zero row.",
    "Non-finite vectors or similarities raise; EMG maps NaN/inf similarities to 0.0.",
    "A missing turn vector raises; EMG silently scores only ids present in its index.",
    "The query vector is L2-normalised here; EMG relies on its embedding protocol "
    "to deliver a normalised query.",
    "Ties break by corpus chronological index; EMG returns an unordered dict.",
]

# FAISS IndexHNSWFlat sizes its candidate heap as max(efSearch, k) but stops the
# level-0 beam once efSearch processed distances lie below the next candidate
# (faiss/impl/HNSW.cpp, search_from_candidates). efSearch, not k, bounds
# exploration, so the tail of a top-k with k > efSearch degrades.
K_EXCEEDS_EF_CAVEAT = (
    "FAISS HNSW does not widen the beam to k: exploration stops after efSearch "
    "below-threshold distances even though the result heap holds max(efSearch, k). "
    "Recall@k for k > efSearch is therefore bounded by efSearch (storage survey probe, "
    "synthetic d=64 N=3000 M=32: recall@300 = 0.925 at efSearch=64 vs 1.0 at 300). "
    "Audit at the production candidate k (hybrid_ranker candidate_k >= 100, "
    "vector_search asks top_k*3 >= 300) plus tombstone fetch_k, not only k=10."
)


def _unit(vector: Any, dim: int, label: str) -> np.ndarray:
    """Validate one vector (1-D, width ``dim``, finite, non-zero) and L2-normalise it."""
    vec = np.asarray(vector, dtype=np.float32)
    if vec.ndim != 1 or vec.shape[0] != dim:
        raise ValueError(f"{label} has shape {vec.shape}; expected ({dim},)")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vec.astype(np.float64)))
    if norm == 0.0:
        raise ValueError(f"{label} has zero norm")
    return (vec / norm).astype(np.float32)


def _dot_rows(matrix: np.ndarray, unit_query: np.ndarray) -> np.ndarray:
    """Row-wise dot products. einsum, not ``@``: BLAS gemv can round identical
    rows differently (blocked vs remainder kernels), which would let rounding,
    not chronology, break ties between duplicate turns."""
    return np.einsum("ij,j->i", matrix, unit_query)


@dataclass(frozen=True, eq=False)
class DenseMatrix:
    """L2-normalised float32 rows aligned with ``corpus.turns`` (row i == turn i)."""

    corpus: ScopedCorpus
    matrix: np.ndarray
    dim: int

    @classmethod
    def from_vectors(
        cls,
        corpus: ScopedCorpus,
        vectors: Mapping[str, np.ndarray],
        expected_dim: Optional[int],
    ) -> "DenseMatrix":
        """Fail closed on a missing, non-finite, wrong-width or zero-norm vector.

        ``expected_dim=None`` infers the width from the first turn and then
        requires every row to match it. Extra ids in ``vectors`` are ignored.
        """
        missing = [turn.node_id for turn in corpus.turns if turn.node_id not in vectors]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(corpus)} turns have no vector "
                f"(first: {missing[:3]}); refusing a partial dense matrix"
            )
        if expected_dim is None:
            expected_dim = (
                int(np.asarray(vectors[corpus.turns[0].node_id]).shape[-1]) if len(corpus) else 0
            )
        rows = [_unit(vectors[t.node_id], expected_dim, f"vector for {t.node_id!r}") for t in corpus.turns]
        matrix = np.vstack(rows) if rows else np.zeros((0, expected_dim), dtype=np.float32)
        matrix.setflags(write=False)
        return cls(corpus=corpus, matrix=matrix, dim=int(expected_dim))

    @property
    def node_ids(self) -> List[str]:
        return [turn.node_id for turn in self.corpus.turns]


def _cosines(query_vec: np.ndarray, dm: DenseMatrix) -> np.ndarray:
    sims = _dot_rows(dm.matrix, _unit(query_vec, dm.dim, "query vector"))
    if not np.all(np.isfinite(sims)):
        raise ValueError("dense similarities contain non-finite values")
    return sims


def exact_scores(query_vec: np.ndarray, dm: DenseMatrix) -> Dict[str, float]:
    """Cosine similarity of the query against every turn: ``{node_id: cosine}``."""
    return {turn.node_id: float(s) for turn, s in zip(dm.corpus.turns, _cosines(query_vec, dm))}


def _ranked_rows(sims: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    """Row indices by score desc, then row (== chronological index) asc."""
    order = np.lexsort((np.arange(len(sims)), -sims))
    return order if top_k is None else order[:top_k]


def exact_rank(query_vec: np.ndarray, dm: DenseMatrix, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """Exact dense ranking over the whole scope (all turns when ``top_k`` is None).

    Every turn carries a measured cosine, so the full ranking is returned;
    ordering matches ``ScopedCorpus.sort_scores``.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    sims = _cosines(query_vec, dm)
    turns = dm.corpus.turns
    return [(turns[i].node_id, float(sims[i])) for i in _ranked_rows(sims, top_k)]


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("HNSW mode requires faiss; refusing to substitute exact search") from exc
    return faiss


class HNSWScope:
    """FAISS IndexHNSWFlat (inner product) over one ``DenseMatrix``."""

    def __init__(self, dm: DenseMatrix, m: int = 32, ef_construction: int = 40, ef_search: int = 64):
        if m < 2 or ef_construction < 1 or ef_search < 1:
            raise ValueError("HNSW requires m >= 2, ef_construction >= 1, ef_search >= 1")
        faiss = _require_faiss()
        self.dm = dm
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.index = faiss.IndexHNSWFlat(dm.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        start = time.perf_counter()
        if len(dm.matrix):
            # Multithreaded insertion makes the HNSW graph (and so the served
            # ranking) depend on scheduling; build single-threaded.
            threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            try:
                self.index.add(np.ascontiguousarray(dm.matrix))
            finally:
                faiss.omp_set_num_threads(threads)
        self.build_seconds = time.perf_counter() - start
        if self.index.ntotal != len(dm.matrix):
            raise RuntimeError(f"HNSW inserted {self.index.ntotal} rows, expected {len(dm.matrix)}")
        self.ef_search = int(ef_search)

    @property
    def ef_search(self) -> int:
        return int(self.index.hnsw.efSearch)

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        if value < 1:
            raise ValueError("ef_search must be positive")
        self.index.hnsw.efSearch = int(value)

    def search_rows(self, unit_query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Raw ANN (rows, scores) for an already-normalised query, re-sorted deterministically."""
        k = min(int(k), len(self.dm.matrix))
        if k < 1:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        scores, rows = self.index.search(unit_query.reshape(1, -1), k)
        keep = rows[0] >= 0
        rows, scores = rows[0][keep], scores[0][keep]
        order = np.lexsort((rows, -scores))
        return rows[order], scores[order]

    def rank(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        rows, scores = self.search_rows(_unit(query_vec, self.dm.dim, "query vector"), top_k)
        turns = self.dm.corpus.turns
        return [(turns[int(r)].node_id, float(s)) for r, s in zip(rows, scores)]

    def params(self) -> Dict[str, Any]:
        return {"index_type": "IndexHNSWFlat", "metric": "inner_product", "m": self.m,
                "ef_construction": self.ef_construction, "ef_search": self.ef_search}


def ann_recall(exact_ids: Sequence[Any], ann_ids: Sequence[Any], k: int) -> float:
    """len(exact top-k & ANN top-k) / min(k, len(exact_ids))."""
    if k < 1:
        raise ValueError("k must be at least 1")
    denominator = min(k, len(exact_ids))
    if denominator == 0:
        raise ValueError("recall is undefined for an empty exact result")
    return len(set(exact_ids[:k]) & set(ann_ids[:k])) / denominator


def _latency(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}


def ann_audit(
    dm: DenseMatrix,
    queries: np.ndarray,
    ks: Sequence[int],
    m: int = 32,
    ef_construction: int = 40,
    ef_search_grid: Sequence[int] = (64,),
    exclude_rows: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Recall@k of HNSW against exact search, per efSearch, plus per-query latency.

    ``exclude_rows[i]`` (leave-one-out) drops that row from query i's exact and
    ANN lists; ANN is then searched with k+1 so the self hit does not cost a slot.
    """
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim != 2 or len(queries) == 0:
        raise ValueError("queries must be a non-empty 2-D array")
    if exclude_rows is not None and len(exclude_rows) != len(queries):
        raise ValueError("exclude_rows must align with queries")
    ks = sorted({int(k) for k in ks})
    if not ks or ks[0] < 1 or not ef_search_grid:
        raise ValueError("ks and ef_search_grid must be non-empty and positive")
    n = len(dm.matrix)
    available = n - (1 if exclude_rows is not None else 0)
    if available < 1:
        raise ValueError("scope is too small to audit")
    k_max = min(ks[-1], available)
    extra = 1 if exclude_rows is not None else 0
    units = np.vstack([_unit(q, dm.dim, f"query {i}") for i, q in enumerate(queries)])

    exact_lists: List[List[int]] = []
    exact_ms: List[float] = []
    for i, q in enumerate(units):
        start = time.perf_counter()
        sims = _dot_rows(dm.matrix, q)
        if exclude_rows is not None:
            sims = sims.copy()
            sims[int(exclude_rows[i])] = -np.inf
        rows = _ranked_rows(sims, k_max)
        exact_ms.append((time.perf_counter() - start) * 1000.0)
        exact_lists.append([int(r) for r in rows])

    scope = HNSWScope(dm, m=m, ef_construction=ef_construction, ef_search=int(ef_search_grid[0]))
    by_ef = []
    for ef in ef_search_grid:
        scope.ef_search = int(ef)
        ann_ms: List[float] = []
        per_k: Dict[int, List[float]] = {k: [] for k in ks}
        for i, q in enumerate(units):
            start = time.perf_counter()
            rows, _scores = scope.search_rows(q, k_max + extra)
            ann_ms.append((time.perf_counter() - start) * 1000.0)
            ann = [int(r) for r in rows if exclude_rows is None or int(r) != int(exclude_rows[i])]
            for k in ks:
                per_k[k].append(ann_recall(exact_lists[i], ann, min(k, available)))
        by_ef.append({
            "ef_search": int(ef),
            "recall_at_k": {str(k): float(np.mean(v)) for k, v in per_k.items()},
            "min_recall_at_k": {str(k): float(np.min(v)) for k, v in per_k.items()},
            "k_exceeds_ef_search": {str(k): bool(min(k, available) > ef) for k in ks},
            "latency_ms": _latency(ann_ms),
        })

    return {
        "n_vectors": n,
        "dim": dm.dim,
        "n_queries": len(queries),
        "leave_one_out": exclude_rows is not None,
        "ks": ks,
        "effective_k": {str(k): min(k, available) for k in ks},
        "hnsw": {"index_type": "IndexHNSWFlat", "metric": "inner_product", "m": scope.m,
                 "ef_construction": scope.ef_construction,
                 "ef_search_grid": [int(ef) for ef in ef_search_grid]},
        "build_seconds": scope.build_seconds,
        "exact_latency_ms": _latency(exact_ms),
        "by_ef_search": by_ef,
        "caveat": K_EXCEEDS_EF_CAVEAT,
    }
