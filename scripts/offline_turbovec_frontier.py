"""Bounded, provider-free TurboVec/TurboQuant frontier at native 4096-d.

This is an opt-in research harness for the third-party MIT ``turbovec``
implementation. It compares 2/4-bit TurboQuant and TQ+ calibration with an
exact FAISS inner-product oracle on deterministic normalized vectors. Synthetic
neighbors measure ANN mechanics and storage only, never semantic evidence
quality. Artifacts are create-once, atomic receipts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIMENSION = 4096
HARD_MAX_VECTORS = 4096
HARD_MAX_QUERIES = 256
MEMORY_CAP_BYTES = 512 * 1024 * 1024
SCHEMA = "hybridmind.offline-turbovec-frontier/v1"


class BackendUnavailableError(RuntimeError):
    pass


class ResourceGateError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_backends() -> tuple[Any, Any]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise BackendUnavailableError("FAISS exact oracle is required") from exc
    try:
        import turbovec  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise BackendUnavailableError(
            "turbovec is required; no quantizer substitute is permitted"
        ) from exc
    if _version("turbovec") != "1.0.0":
        raise BackendUnavailableError(
            "this attested harness requires exactly turbovec==1.0.0"
        )
    return faiss, turbovec


def _available_memory_budget() -> int:
    try:
        import psutil  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ResourceGateError("psutil is required for the memory gate") from exc
    return min(MEMORY_CAP_BYTES, max(1, int(psutil.virtual_memory().available) // 2))


def estimated_working_bytes(vector_count: int, dimension: int) -> int:
    if vector_count < 1 or dimension < 1:
        raise ValueError("vector_count and dimension must be positive")
    # source vectors + queries/oracle scratch + quantizer build/search scratch
    return vector_count * dimension * 4 * 3 + vector_count * 16


def _validate_workload(vector_count: int, query_count: int, dimension: int) -> dict[str, int]:
    if vector_count < 10 or vector_count > HARD_MAX_VECTORS:
        raise ResourceGateError(
            f"vector_count must be in [10, {HARD_MAX_VECTORS}]"
        )
    if query_count < 1 or query_count > HARD_MAX_QUERIES:
        raise ResourceGateError(
            f"query_count must be in [1, {HARD_MAX_QUERIES}]"
        )
    estimate = estimated_working_bytes(vector_count, dimension)
    budget = _available_memory_budget()
    if estimate > budget:
        raise ResourceGateError(
            f"estimated working memory {estimate} exceeds host budget {budget}"
        )
    return {"estimated_working_bytes": estimate, "budget_bytes": budget}


def _normalized_vectors(count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((count, dimension), dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values /= np.where(norms > 0, norms, 1.0)
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.shape != (count, dimension) or not np.all(np.isfinite(values)):
        raise ValueError("invalid deterministic vector fixture")
    return values


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency observations must be non-empty, finite, and non-negative")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "n": len(values),
    }


def _recall(retrieved: np.ndarray, oracle: np.ndarray, k: int) -> float:
    if retrieved.shape[0] != oracle.shape[0]:
        raise ValueError("retrieved/oracle query counts differ")
    width = min(k, retrieved.shape[1], oracle.shape[1])
    if width < 1:
        raise ValueError("recall width must be positive")
    return float(
        mean(
            len(set(map(int, row[:width])) & set(map(int, gold[:width]))) / width
            for row, gold in zip(retrieved, oracle)
        )
    )


def _package_binary_provenance(module: Any) -> dict[str, str]:
    package_root = Path(module.__file__).resolve().parent
    binaries = sorted(
        path for path in package_root.iterdir() if path.suffix.lower() in {".pyd", ".so", ".dll"}
    )
    if not binaries:
        raise BackendUnavailableError("turbovec native extension could not be attested")
    return {str(path): _sha256(path) for path in binaries}


def _oracle(faiss: Any, vectors: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    observations: list[float] = []
    distances = indices = None
    for _ in range(3):
        started = time.perf_counter_ns()
        distances, indices = index.search(queries, k)
        observations.append((time.perf_counter_ns() - started) / 1_000_000)
    assert distances is not None and indices is not None
    return distances, indices, _percentiles(observations)


def _measure_variant(
    turbovec: Any,
    vectors: np.ndarray,
    queries: np.ndarray,
    oracle_scores: np.ndarray,
    oracle_ids: np.ndarray,
    *,
    bit_width: int,
    calibrated: bool,
    seed: int,
    repetitions: int,
    top_k: int,
) -> dict[str, Any]:
    import psutil  # type: ignore

    process = psutil.Process()
    rss_start = int(process.memory_info().rss)
    index = turbovec.IdMapIndex(dim=vectors.shape[1], bit_width=bit_width)
    calibration_ms = 0.0
    calibration_rows = 0
    if calibrated:
        calibration_rows = min(1024, len(vectors))
        permutation = np.random.default_rng(seed ^ 0xC411B).permutation(len(vectors))
        sample = np.ascontiguousarray(vectors[permutation[:calibration_rows]])
        started = time.perf_counter_ns()
        index.calibrate(sample)
        calibration_ms = (time.perf_counter_ns() - started) / 1_000_000
    ids = np.arange(len(vectors), dtype=np.uint64)
    started = time.perf_counter_ns()
    index.add_with_ids(vectors, ids)
    build_ms = (time.perf_counter_ns() - started) / 1_000_000
    index.prepare()
    rss_peak = int(process.memory_info().rss)

    latency: list[float] = []
    scores = result_ids = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        scores, result_ids = index.search(queries, top_k)
        latency.append((time.perf_counter_ns() - started) / 1_000_000)
    assert scores is not None and result_ids is not None
    scores = np.asarray(scores, dtype=np.float32)
    result_ids = np.asarray(result_ids, dtype=np.uint64)
    if result_ids.shape != oracle_ids.shape or scores.shape != oracle_scores.shape:
        raise RuntimeError("TurboVec result shape does not match the exact oracle")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("TurboVec emitted non-finite scores")

    replay_scores, replay_ids = index.search(queries, top_k)
    payload = index.to_bytes()
    restored = turbovec.IdMapIndex.from_bytes(payload)
    restored_scores, restored_ids = restored.search(queries, top_k)

    # Stable-ID deletion semantics are required for a mutable derived index.
    removed_id = int(result_ids[0, 0])
    before = len(index)
    removed = bool(index.remove(removed_id))
    after = len(index)
    _, post_delete_ids = index.search(queries[:1], top_k)
    delete_ok = removed and after == before - 1 and removed_id not in set(map(int, post_delete_ids[0]))

    top1 = float(np.mean(result_ids[:, 0] == oracle_ids[:, 0]))
    recall10 = _recall(result_ids, oracle_ids, top_k)
    score_gap = float(np.mean(np.abs(oracle_scores - scores)))
    return {
        "backend": "turbovec.IdMapIndex",
        "bit_width": bit_width,
        "calibrated": calibrated,
        "calibration_state": restored.calibration_state,
        "calibration_rows": calibration_rows,
        "calibration_ms": calibration_ms,
        "build_ms": build_ms,
        "warm_batch_search_ms": _percentiles(latency),
        "warm_per_query_amortized_ms": {
            key: value / len(queries) if key != "n" else value
            for key, value in _percentiles(latency).items()
        },
        "recall_at_1": top1,
        "recall_at_10": recall10,
        "mean_absolute_oracle_score_gap_at_10": score_gap,
        "serialized_index_bytes": len(payload),
        "raw_float32_bytes": int(vectors.nbytes),
        "compression_ratio_vs_raw_float32": float(vectors.nbytes / len(payload)),
        "rss_increase_bytes": max(0, rss_peak - rss_start),
        "deterministic_replay": {
            "ids_equal": bool(np.array_equal(result_ids, replay_ids)),
            "scores_equal": bool(np.array_equal(scores, replay_scores)),
        },
        "persistence_roundtrip": {
            "ids_equal": bool(np.array_equal(result_ids, restored_ids)),
            "scores_equal": bool(np.array_equal(scores, restored_scores)),
        },
        "stable_id_delete": {
            "removed_id": removed_id,
            "remove_returned_true": removed,
            "size_before": before,
            "size_after": after,
            "deleted_id_absent_from_probe": delete_ok,
        },
        "retrieved_id_sha256": hashlib.sha256(result_ids.tobytes()).hexdigest(),
    }


def run_frontier(
    *,
    vector_count: int = 4096,
    query_count: int = 64,
    seeds: Sequence[int] = (0,),
    bit_widths: Sequence[int] = (2, 4),
    calibration_modes: Sequence[bool] = (False, True),
    repetitions: int = 3,
    top_k: int = 10,
    dimension: int = PRODUCTION_DIMENSION,
    mechanics_test_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if dimension != PRODUCTION_DIMENSION and not mechanics_test_only:
        raise ValueError("production TurboVec frontier requires dimension=4096")
    if not seeds or not bit_widths or not calibration_modes:
        raise ValueError("seeds, bit_widths, and calibration_modes must be non-empty")
    if any(bit not in {2, 4} for bit in bit_widths):
        raise ValueError("TurboVec frontier supports only 2-bit and 4-bit conditions")
    if repetitions < 2:
        raise ValueError("at least two timed repetitions are required")
    if top_k != 10:
        raise ValueError("this schema requires top_k=10")
    safety = _validate_workload(vector_count, query_count, dimension)
    faiss, turbovec = _require_backends()
    results: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for seed in map(int, seeds):
        vectors = _normalized_vectors(vector_count, dimension, seed)
        queries = _normalized_vectors(query_count, dimension, seed ^ 0xA5A5A5A5)
        oracle_scores, oracle_ids, oracle_latency = _oracle(
            faiss, vectors, queries, top_k
        )
        oracle_rows.append(
            {
                "seed": seed,
                "backend": "faiss.IndexFlatIP",
                "recall_at_1": 1.0,
                "recall_at_10": 1.0,
                "warm_batch_search_ms": oracle_latency,
                "raw_float32_bytes": int(vectors.nbytes),
            }
        )
        for bit_width in bit_widths:
            for calibrated in calibration_modes:
                row = _measure_variant(
                    turbovec,
                    vectors,
                    queries,
                    oracle_scores,
                    oracle_ids,
                    bit_width=int(bit_width),
                    calibrated=bool(calibrated),
                    seed=seed,
                    repetitions=repetitions,
                    top_k=top_k,
                )
                row["seed"] = seed
                results.append(row)
                gc.collect()
        del vectors, queries
        gc.collect()

    source = Path(__file__).resolve()
    status = _git_output("status", "--porcelain=v1", "-z") or ""
    return {
        "schema_version": SCHEMA,
        "classification": "measured_offline_turbovec_synthetic_frontier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0,
        "external_network_calls_during_experiment": 0,
        "execution": {
            "provider_calls": 0,
            "external_network_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "claim_boundary": [
            "Synthetic normalized vectors measure ANN mechanics, not semantic or exact-evidence retrieval.",
            "The FAISS and TurboVec batch timings are local single-host observations, not vendor-neutral throughput claims.",
            "No production backend is replaced without real native-4096 embedding and exact-evidence confirmation.",
        ],
        "workload": {
            "dimension": dimension,
            "vector_count": vector_count,
            "query_count": query_count,
            "top_k": top_k,
            "seeds": list(map(int, seeds)),
            "bit_widths": list(map(int, bit_widths)),
            "calibration_modes": list(map(bool, calibration_modes)),
            "repetitions": repetitions,
            "mechanics_test_only": mechanics_test_only,
        },
        "safety_gate": safety,
        "oracle": oracle_rows,
        "results": results,
        "provenance": {
            "source": {"path": str(source), "sha256": _sha256(source)},
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "dependency_versions": {
                "turbovec": _version("turbovec"),
                "faiss-cpu": _version("faiss-cpu"),
                "numpy": _version("numpy"),
                "psutil": _version("psutil"),
            },
            "turbovec_native_binary_sha256": _package_binary_provenance(turbovec),
            "turbovec_license": "MIT",
            "turbovec_project": "https://github.com/RyanCodrai/turbovec",
            "turbovec_release": "1.0.0",
        },
    }


def write_json_atomic_exclusive(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vector-count", type=int, default=4096)
    parser.add_argument("--query-count", type=int, default=64)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--bit-widths", default="2,4")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    report = run_frontier(
        vector_count=args.vector_count,
        query_count=args.query_count,
        seeds=[int(item) for item in args.seeds.split(",") if item.strip()],
        bit_widths=[int(item) for item in args.bit_widths.split(",") if item.strip()],
        repetitions=args.repetitions,
    )
    write_json_atomic_exclusive(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "results": len(report["results"]),
                "provider_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
