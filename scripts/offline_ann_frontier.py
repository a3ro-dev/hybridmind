"""Bounded, provider-free ANN frontier for native 4096-dimensional vectors.

This is a measurement harness, not a production index replacement.  The CLI is
intentionally fixed to the repository's exact 4096-dimensional float32 contract.
The ``run_frontier`` dimension escape hatch is marked ``mechanics_test_only``
and exists solely for tiny unit tests; it is never exposed by the CLI.

Backends are explicit and fail closed:

* ``flat_ip``: exact FAISS IndexFlatIP oracle;
* ``hnsw_flat``: FAISS IndexHNSWFlat;
* ``hnsw_sq8``: FAISS IndexHNSWSQ with per-dimension 8-bit scalar quantization.

No embeddings, models, hosted services, or provider calls are used.  Results
separate measured observations from analytic byte projections and are written
atomically as a versioned JSON artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PRODUCTION_DIMENSION = 4096
DEFAULT_SIZES = (256, 1024)
DEFAULT_SEEDS = (0, 1)
DEFAULT_BUILD_ORDERS = ("natural", "reverse", "permuted")
DEFAULT_QUERY_COUNT = 64
DEFAULT_TOP_K = 10
HNSW_M = 32
HNSW_EF_SEARCH = 64
HNSW_EF_CONSTRUCTION = 80
FAISS_THREADS = 1
HARD_MAX_VECTORS = 4096
HARD_MAX_QUERIES = 1024
MEMORY_CAP_BYTES = 512 * 1024 * 1024


class ResourceGateError(RuntimeError):
    """Raised when a declared workload exceeds a host safety bound."""


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend cannot be constructed exactly."""


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parents[1], check=True,
            capture_output=True, text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("faiss-cpu", "numpy", "psutil"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _require_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise BackendUnavailableError(
            "FAISS is required; refusing to substitute another ANN backend"
        ) from exc
    return faiss


def available_memory_budget_bytes() -> int:
    """Return min(512 MiB, half currently available host RAM)."""
    try:
        import psutil  # type: ignore

        available = int(psutil.virtual_memory().available)
    except Exception as exc:  # pragma: no cover - psutil is a required dependency
        raise ResourceGateError("psutil is required for the host memory safety gate") from exc
    return min(MEMORY_CAP_BYTES, max(1, available // 2))


def estimated_working_bytes(vector_count: int, dimension: int) -> int:
    """Conservative vector working-set estimate used before allocation.

    The factor of three covers the canonical vectors, an oracle/search scratch
    copy, and backend build/search scratch.  It is deliberately a safety gate,
    not a claim about actual resident memory.
    """
    if vector_count < 1 or dimension < 1:
        raise ValueError("vector_count and dimension must be positive")
    return vector_count * dimension * 4 * 3 + vector_count * 8


def _validate_workload(vector_count: int, dimension: int, query_count: int) -> None:
    if vector_count > HARD_MAX_VECTORS:
        raise ResourceGateError(
            f"vector count {vector_count} exceeds hard cap {HARD_MAX_VECTORS}"
        )
    if query_count > HARD_MAX_QUERIES:
        raise ResourceGateError(
            f"query count {query_count} exceeds hard cap {HARD_MAX_QUERIES}"
        )
    estimate = estimated_working_bytes(vector_count, dimension)
    budget = available_memory_budget_bytes()
    if estimate > budget:
        raise ResourceGateError(
            f"estimated working memory {estimate} exceeds host budget {budget}"
        )


def _finite_normalized_vectors(count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((count, dimension), dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values /= np.where(norms > 0, norms, 1.0)
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.shape != (count, dimension) or not np.all(np.isfinite(values)):
        raise ValueError("deterministic vector generation produced invalid values")
    return values


def _build_order(name: str, count: int, seed: int) -> np.ndarray:
    if name == "natural":
        return np.arange(count, dtype=np.int64)
    if name == "reverse":
        return np.arange(count - 1, -1, -1, dtype=np.int64)
    if name == "permuted":
        return np.random.default_rng(seed + 0x9E3779B9).permutation(count).astype(
            np.int64
        )
    raise ValueError(f"unknown build order: {name!r}")


def _set_and_attest_hnsw_controls(
    base: Any, ef_search: int, ef_construction: int,
) -> dict[str, int]:
    hnsw = getattr(base, "hnsw", None)
    if hnsw is None:
        raise BackendUnavailableError(
            "requested HNSW backend does not expose executable HNSW controls"
        )
    hnsw.efSearch = ef_search
    hnsw.efConstruction = ef_construction
    executed = {
        "ef_search": int(hnsw.efSearch),
        "ef_construction": int(hnsw.efConstruction),
    }
    if executed != {
        "ef_search": ef_search,
        "ef_construction": ef_construction,
    }:
        raise RuntimeError(
            f"HNSW control attestation mismatch: requested efSearch={ef_search}, "
            f"efConstruction={ef_construction}, executed={executed}"
        )
    return executed


def _read_hnsw_controls(index: Any) -> dict[str, int] | None:
    faiss = _require_faiss()
    base = getattr(index, "index", index)
    try:
        base = faiss.downcast_index(base)
    except Exception:
        pass
    hnsw = getattr(base, "hnsw", None)
    if hnsw is None:
        return None
    return {
        "ef_search": int(hnsw.efSearch),
        "ef_construction": int(hnsw.efConstruction),
    }


def build_index(
    backend: str,
    vectors: np.ndarray,
    build_order: np.ndarray,
    *,
    dimension: int,
    ef_search: int = HNSW_EF_SEARCH,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
) -> Any:
    """Build one explicitly requested FAISS backend; never substitute."""
    try:
        faiss = _require_faiss()
    except BackendUnavailableError:
        raise
    except Exception as exc:
        raise BackendUnavailableError(
            "FAISS is required; refusing to substitute another ANN backend"
        ) from exc
    if vectors.shape[1] != dimension:
        raise ValueError("vector dimension does not match declared dimension")
    if ef_search < 1 or ef_construction < 1:
        raise ValueError("ef_search and ef_construction must be positive")
    if backend == "flat_ip":
        base = faiss.IndexFlatIP(dimension)
    elif backend == "hnsw_flat":
        if not hasattr(faiss, "IndexHNSWFlat"):
            raise BackendUnavailableError("FAISS IndexHNSWFlat is unavailable")
        base = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    elif backend == "hnsw_sq8":
        if not hasattr(faiss, "IndexHNSWSQ") or not hasattr(faiss, "ScalarQuantizer"):
            raise BackendUnavailableError("FAISS scalar-quantized HNSW is unavailable")
        qtype = getattr(faiss.ScalarQuantizer, "QT_8bit", None)
        if qtype is None:
            raise BackendUnavailableError("FAISS QT_8bit scalar quantizer is unavailable")
        try:
            base = faiss.IndexHNSWSQ(
                dimension, qtype, HNSW_M, faiss.METRIC_INNER_PRODUCT
            )
        except Exception as exc:
            raise BackendUnavailableError(
                "FAISS IndexHNSWSQ(QT_8bit) could not be constructed"
            ) from exc
    else:
        raise ValueError(f"unknown backend {backend!r}")

    requested_controls = None
    if backend != "flat_ip":
        requested_controls = _set_and_attest_hnsw_controls(
            base, ef_search, ef_construction,
        )

    index = faiss.IndexIDMap2(base)
    ordered_vectors = np.ascontiguousarray(vectors[build_order], dtype=np.float32)
    ordered_ids = np.ascontiguousarray(build_order, dtype=np.int64)
    if not index.is_trained:
        try:
            index.train(ordered_vectors)
        except Exception as exc:
            raise BackendUnavailableError(
                f"requested backend {backend!r} failed training"
            ) from exc
    index.add_with_ids(ordered_vectors, ordered_ids)
    if index.ntotal != len(vectors):
        raise RuntimeError(f"{backend} inserted {index.ntotal} rows, expected {len(vectors)}")
    executed_controls = _read_hnsw_controls(index)
    if executed_controls != requested_controls:
        raise RuntimeError(
            f"HNSW controls changed during wrapping/build: requested={requested_controls}, "
            f"executed={executed_controls}"
        )
    return index


class _RSSSampler:
    def __init__(self) -> None:
        import psutil  # type: ignore

        self._process = psutil.Process(os.getpid())
        self.start = int(self._process.memory_info().rss)
        self.peak = self.start
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, int(self._process.memory_info().rss))
            except Exception:
                pass
            self._stop.wait(0.005)

    def __enter__(self) -> "_RSSSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.peak = max(self.peak, int(self._process.memory_info().rss))
        except Exception:
            pass


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency sample cannot be empty")
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in (50, 95, 99)}


def _rank_loss(oracle: Sequence[int], approximate: Sequence[int], k: int) -> float:
    oracle_ranks = {int(value): rank for rank, value in enumerate(oracle[:k])}
    approximate_ranks = {int(value): rank for rank, value in enumerate(approximate[:k])}
    union = set(oracle_ranks) | set(approximate_ranks)
    if not union:
        return 0.0
    missing_rank = k
    return float(
        np.mean(
            [
                abs(
                    oracle_ranks.get(item, missing_rank)
                    - approximate_ranks.get(item, missing_rank)
                )
                for item in union
            ]
        )
    )


def _id_digest(rows: Iterable[Sequence[int]]) -> str:
    canonical = json.dumps([[int(value) for value in row] for row in rows], separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _delete_probe(index: Any) -> dict[str, Any]:
    """Probe native deletion without changing the benchmark's measured index."""
    faiss = _require_faiss()
    try:
        clone = faiss.clone_index(index)
    except Exception:
        try:
            serialized = faiss.serialize_index(index)
            clone = faiss.deserialize_index(serialized)
        except Exception as exc:
            return {
                "supported": False,
                "removed": 0,
                "error_type": type(exc).__name__,
                "error": "index clone/deserialize failed",
            }
    try:
        removed = int(clone.remove_ids(np.asarray([0], dtype=np.int64)))
        return {"supported": True, "removed": removed, "remaining": int(clone.ntotal)}
    except Exception as exc:
        return {
            "supported": False,
            "removed": 0,
            "error_type": type(exc).__name__,
            "error": "native delete is unsupported for this index",
        }


def _measure_backend(
    backend: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    oracle_ids: list[list[int]],
    oracle_scores: list[list[float]],
    build_order: np.ndarray,
    *,
    dimension: int,
    ef_search: int,
    ef_construction: int,
) -> dict[str, Any]:
    gc.collect()
    with _RSSSampler() as rss:
        build_start = time.perf_counter()
        index = build_index(
            backend, vectors, build_order, dimension=dimension,
            ef_search=ef_search,
            ef_construction=ef_construction,
        )
        build_seconds = time.perf_counter() - build_start
        serialized_bytes = len(_require_faiss().serialize_index(index))

        cold_start = time.perf_counter()
        index.search(np.ascontiguousarray(queries[:1]), DEFAULT_TOP_K)
        cold_search_ms = (time.perf_counter() - cold_start) * 1000.0
        index.search(np.ascontiguousarray(queries[: min(8, len(queries))]), DEFAULT_TOP_K)

        result_ids: list[list[int]] = []
        result_scores: list[list[float]] = []
        latencies_ms: list[float] = []
        for query in queries:
            start = time.perf_counter()
            distances, ids = index.search(np.ascontiguousarray(query[None, :]), DEFAULT_TOP_K)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            result_ids.append([int(value) for value in ids[0] if int(value) >= 0])
            result_scores.append([float(value) for value in distances[0] if np.isfinite(value)])

        replay_ids: list[list[int]] = []
        for query in queries:
            _distances, ids = index.search(np.ascontiguousarray(query[None, :]), DEFAULT_TOP_K)
            replay_ids.append([int(value) for value in ids[0] if int(value) >= 0])

        delete_probe = _delete_probe(index)
        peak_rss = rss.peak
        rss_start = rss.start

    k = min(DEFAULT_TOP_K, len(vectors))
    recall1 = float(
        np.mean(
            [bool(approx and oracle and approx[0] == oracle[0]) for approx, oracle in zip(result_ids, oracle_ids)]
        )
    )
    recall10 = float(
        np.mean(
            [len(set(approx[:k]) & set(oracle[:k])) / k for approx, oracle in zip(result_ids, oracle_ids)]
        )
    )
    rank_losses = [_rank_loss(oracle, approx, k) for oracle, approx in zip(oracle_ids, result_ids)]
    score_gaps: list[float] = []
    for oracle, approx in zip(oracle_ids, result_ids):
        oracle_scores_by_id = {item: score for item, score in zip(oracle, oracle_scores[len(score_gaps)])}
        candidate_scores = vectors @ queries[len(score_gaps)]
        gap = []
        for rank, item in enumerate(oracle[:k]):
            approx_item = approx[rank] if rank < len(approx) else None
            approx_score = float(candidate_scores[approx_item]) if approx_item is not None else 0.0
            gap.append(max(0.0, float(oracle_scores_by_id.get(item, 0.0)) - approx_score))
        score_gaps.append(float(np.mean(gap)) if gap else 0.0)

    return {
        "evidence_class": "measured_offline",
        "backend": backend,
        "hnsw_ef_search": ef_search if backend != "flat_ip" else None,
        "hnsw_ef_construction": (
            ef_construction if backend != "flat_ip" else None
        ),
        "executed_hnsw_controls": _read_hnsw_controls(index),
        "vector_count": int(len(vectors)),
        "query_count": int(len(queries)),
        "build_seconds": float(build_seconds),
        "build_latency_ms": _percentiles([build_seconds * 1000.0]),
        "cold_search_ms": float(cold_search_ms),
        "search_latency_ms": _percentiles(latencies_ms),
        "serialized_index_bytes": int(serialized_bytes),
        "rss_start_bytes": int(rss_start),
        "rss_peak_bytes": int(peak_rss),
        "rss_increase_bytes": int(max(0, peak_rss - rss_start)),
        "recall_at_1": recall1,
        "recall_at_10": recall10,
        "mean_rank_loss_at_10": float(np.mean(rank_losses)),
        "mean_oracle_score_gap_at_10": float(np.mean(score_gaps)),
        "oracle_neighbor_ids": oracle_ids,
        "retrieved_neighbor_ids": result_ids,
        "deterministic_replay": {
            "search_ids_equal": result_ids == replay_ids,
            "result_id_sha256": _id_digest(result_ids),
        },
        "mutation_delete": delete_probe,
    }


def _analytic_projection(vector_count: int, dimension: int) -> dict[str, int]:
    raw = vector_count * dimension * 4
    links = vector_count * (HNSW_M * 2 * 4)
    return {
        "evidence_class": "analytic_projection",
        "raw_float32_vector_bytes": int(raw),
        "hnsw_flat_formula_bytes": int(raw + links),
        "hnsw_sq8_code_plus_links_formula_bytes": int(vector_count * dimension + links),
    }


def run_frontier(
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    build_orders: Sequence[str] = DEFAULT_BUILD_ORDERS,
    backends: Sequence[str] = ("flat_ip", "hnsw_flat", "hnsw_sq8"),
    query_count: int = DEFAULT_QUERY_COUNT,
    ef_search_values: Sequence[int] = (HNSW_EF_SEARCH,),
    ef_construction_values: Sequence[int] = (HNSW_EF_CONSTRUCTION,),
    dimension: int = PRODUCTION_DIMENSION,
    mechanics_test_only: bool = False,
) -> dict[str, Any]:
    """Run the bounded ANN frontier.

    ``dimension != 4096`` is accepted only with ``mechanics_test_only=True``;
    this is a mechanics-only unit-test double and never a production workload.
    """
    started = time.perf_counter()
    if dimension != PRODUCTION_DIMENSION and not mechanics_test_only:
        raise ValueError("production ANN frontier requires dimension=4096")
    if (
        not sizes or not seeds or not build_orders or not backends
        or not ef_search_values or not ef_construction_values
    ):
        raise ValueError(
            "sizes, seeds, build_orders, backends, ef_search_values, and "
            "ef_construction_values must be non-empty"
        )
    if query_count < 1:
        raise ValueError("query_count must be positive")
    for backend in backends:
        if backend not in {"flat_ip", "hnsw_flat", "hnsw_sq8"}:
            raise ValueError(f"unknown requested backend {backend!r}")
    normalized_ef_search = [int(value) for value in ef_search_values]
    if any(value < 1 or value > 4096 for value in normalized_ef_search):
        raise ValueError("ef_search values must be in [1, 4096]")
    normalized_ef_construction = [int(value) for value in ef_construction_values]
    if any(value < 1 or value > 4096 for value in normalized_ef_construction):
        raise ValueError("ef_construction values must be in [1, 4096]")

    faiss = _require_faiss()
    if not hasattr(faiss, "omp_set_num_threads"):
        raise BackendUnavailableError("FAISS thread control is unavailable")
    faiss.omp_set_num_threads(FAISS_THREADS)

    normalized_sizes = [int(size) for size in sizes]
    normalized_seeds = [int(seed) for seed in seeds]
    for size in normalized_sizes:
        _validate_workload(size, dimension, query_count)

    results: list[dict[str, Any]] = []
    for size in normalized_sizes:
        for seed in normalized_seeds:
            vectors = _finite_normalized_vectors(size, dimension, seed)
            queries = _finite_normalized_vectors(query_count, dimension, seed ^ 0xA5A5A5A5)
            for order_name in build_orders:
                order = _build_order(order_name, size, seed)
                oracle = build_index("flat_ip", vectors, order, dimension=dimension)
                oracle_distances, oracle_indices = oracle.search(queries, DEFAULT_TOP_K)
                oracle_ids = [[int(value) for value in row if int(value) >= 0] for row in oracle_indices]
                oracle_scores = [[float(value) for value in row if np.isfinite(value)] for row in oracle_distances]
                for backend in backends:
                    backend_ef_values = (
                        [(HNSW_EF_SEARCH, HNSW_EF_CONSTRUCTION)]
                        if backend == "flat_ip"
                        else [
                            (ef_search, ef_construction)
                            for ef_construction in normalized_ef_construction
                            for ef_search in normalized_ef_search
                        ]
                    )
                    for ef_search, ef_construction in backend_ef_values:
                        measured = _measure_backend(
                            backend,
                            vectors,
                            queries,
                            oracle_ids,
                            oracle_scores,
                            order,
                            dimension=dimension,
                            ef_search=ef_search,
                            ef_construction=ef_construction,
                        )
                        measured.update({"seed": seed, "build_order": order_name})
                        results.append(measured)
                del oracle
            del vectors, queries
            gc.collect()

    source_path = Path(__file__).resolve()
    status = _git_output("status", "--porcelain=v1", "-z") or ""
    return {
        "schema_version": "hybridmind.offline-ann-frontier/v3",
        "classification": "measured_offline_ann_frontier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0,
        "network_calls": 0,
        "execution": {
            "provider_calls": 0,
            "external_network_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "dimension": int(dimension),
        "mechanics_test_only": bool(mechanics_test_only),
        "claim_boundary": (
            "synthetic ANN mechanics and resource frontier only; independent "
            "random queries do not measure semantic or exact source-evidence quality"
        ),
        "provenance": {
            "source": {"path": str(source_path), "sha256": _sha256(source_path)},
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree": {
                "dirty": bool(status),
                "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            },
            "dependencies": _dependency_versions(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "synthetic_generator": {
            "distribution": "independent standard normal",
            "normalization": "row-wise L2",
            "dtype": "float32",
            "finite_validation": True,
            "query_relation_to_corpus": "independent random draws",
        },
        "declared_workload": {
            "sizes": normalized_sizes,
            "seeds": normalized_seeds,
            "build_orders": list(build_orders),
            "backends": list(backends),
            "query_count": int(query_count),
            "ef_search_values": normalized_ef_search,
            "ef_construction_values": normalized_ef_construction,
            "top_k": DEFAULT_TOP_K,
            "hnsw_m": HNSW_M,
            "hnsw_ef_search": HNSW_EF_SEARCH,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "faiss_threads": FAISS_THREADS,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "safety_gate": {
            "hard_max_vectors": HARD_MAX_VECTORS,
            "memory_budget_bytes": available_memory_budget_bytes(),
            "estimated_working_bytes_by_size": {
                str(size): estimated_working_bytes(size, dimension) for size in normalized_sizes
            },
        },
        "projections": {
            str(size): _analytic_projection(size, dimension) for size in normalized_sizes
        },
        "results": results,
        "interpretation_limits": [
            "Synthetic independent vectors test ANN approximation, build order, and resources only.",
            "FlatIP is the exact neighbor oracle for each generated corpus and query set.",
            "Same-index search replay does not prove deterministic index construction across processes.",
            "Delete probing is not a sustained freshness or mutation-throughput benchmark.",
            "No provider, embedding model, reranker, reader, or external network call executed.",
        ],
    }


def write_json_atomic(report: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_csv(value: str, cast: Any) -> list[Any]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value cannot be empty")
    try:
        return [cast(part) for part in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated value: {value!r}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=lambda value: _parse_csv(value, int), default=list(DEFAULT_SIZES))
    parser.add_argument("--seeds", type=lambda value: _parse_csv(value, int), default=list(DEFAULT_SEEDS))
    parser.add_argument("--build-orders", type=lambda value: _parse_csv(value, str), default=list(DEFAULT_BUILD_ORDERS))
    parser.add_argument("--backends", type=lambda value: _parse_csv(value, str), default=["flat_ip", "hnsw_flat", "hnsw_sq8"])
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument(
        "--ef-search-values",
        type=lambda value: _parse_csv(value, int),
        default=[HNSW_EF_SEARCH],
    )
    parser.add_argument(
        "--ef-construction-values",
        type=lambda value: _parse_csv(value, int),
        default=[HNSW_EF_CONSTRUCTION],
    )
    args = parser.parse_args(argv)
    report = run_frontier(
        sizes=args.sizes,
        seeds=args.seeds,
        build_orders=args.build_orders,
        backends=args.backends,
        query_count=args.query_count,
        ef_search_values=args.ef_search_values,
        ef_construction_values=args.ef_construction_values,
    )
    write_json_atomic(report, args.output)
    print(json.dumps({"output": str(args.output), "results": len(report["results"]), "provider_calls": 0}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
