#!/usr/bin/env python3
"""Measure a bounded synthetic HybridMind index without external calls.

The output intentionally separates observations from analytic capacity and
token-cost scenarios.  It does not measure embedding inference, reader quality,
provider latency, or model KV-cache memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.resource_accounting import (
    DEFAULT_HNSW_M,
    EMBEDDING_DIMENSION,
    MEASURED_CLASSIFICATION,
    OFFLINE_REPORT_SCHEMA,
    capacity_projection,
    latency_summary,
    prompt_reduction,
    recursive_file_sizes,
    stable_result_hash,
    tokenomics_projection,
    validate_offline_report,
)
from storage.bm25_index import create_sparse_index
from storage.graph_index import GraphIndex
from storage.vector_index import VectorIndex


class PeakRssSampler:
    """Sample process RSS; the report calls this observed rather than exact."""

    def __init__(self, interval_seconds: float = 0.005):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.append(int(self.process.memory_info().rss))
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(int(self.process.memory_info().rss))

    def stop(self) -> None:
        self.samples.append(int(self.process.memory_info().rss))
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.samples.append(int(self.process.memory_info().rss))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vectors", type=_positive_int, default=256)
    parser.add_argument("--queries", type=_positive_int, default=32)
    parser.add_argument("--top-k", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--synthetic-source-tokens-per-chunk", type=_positive_int, default=64)
    parser.add_argument("--capacity-source-tokens", type=_positive_int, nargs="+", default=[10_000_000, 40_000_000, 100_000_000])
    parser.add_argument("--capacity-mean-source-tokens-per-chunk", type=_positive_int, default=256)
    parser.add_argument("--pq-code-bytes", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-vectors", type=_positive_int, default=4096)
    parser.add_argument("--max-estimated-working-bytes", type=_positive_int, default=512 * 1024 * 1024)
    parser.add_argument("--baseline-prompt-source-tokens", type=_positive_int)
    parser.add_argument("--retrieved-unique-source-tokens", type=_positive_int)

    for name in (
        "planned_queries",
        "embedding_calls",
        "embedding_input_tokens",
        "reranker_calls",
        "reranker_pairs",
        "reranker_input_tokens",
        "llm_calls",
        "reader_input_tokens",
        "reader_output_tokens",
        "provider_runtime_seconds",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=_nonnegative_int, default=0)
    parser.add_argument("--pricing-mode", choices=("priced", "unpriced"), default="unpriced")
    for name in (
        "embedding_usd_per_million_input_tokens",
        "reranker_usd_per_million_input_tokens",
        "reader_usd_per_million_input_tokens",
        "reader_usd_per_million_output_tokens",
        "fixed_usd_per_embedding_call",
        "fixed_usd_per_reranker_call",
        "fixed_usd_per_llm_call",
        "provider_runtime_usd_per_second",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=_nonnegative_float)
    return parser.parse_args(argv)


def _synthetic_text(node_number: int, token_count: int) -> str:
    words = [f"unique{node_number}", f"topic{node_number % 17}"]
    while len(words) < token_count:
        words.append(f"term{(node_number * 31 + len(words)) % 257}")
    return " ".join(words[:token_count])


def _component_query(
    vector_index: VectorIndex,
    sparse_index: Any,
    graph_index: GraphIndex,
    query_vector: np.ndarray,
    query_text: str,
    anchor_id: str,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    started = time.perf_counter_ns()
    vector_started = time.perf_counter_ns()
    vector_results = vector_index.search(query_vector, top_k=top_k, min_score=-1.0)
    vector_ms = (time.perf_counter_ns() - vector_started) / 1_000_000

    sparse_started = time.perf_counter_ns()
    sparse_results = sparse_index.search(query_text, top_k=top_k)
    sparse_ms = (time.perf_counter_ns() - sparse_started) / 1_000_000

    graph_started = time.perf_counter_ns()
    graph_results = graph_index.traverse_bfs(anchor_id, max_depth=2, direction="typed")
    graph_ms = (time.perf_counter_ns() - graph_started) / 1_000_000
    total_ms = (time.perf_counter_ns() - started) / 1_000_000
    result = {
        "vector": [(node_id, round(score, 8)) for node_id, score in vector_results],
        "sparse": [(node_id, round(score, 8)) for node_id, score in sparse_results],
        "graph": [(node_id, depth, path) for node_id, depth, path in graph_results],
    }
    return result, {
        "vector": vector_ms,
        "sparse": sparse_ms,
        "graph": graph_ms,
        "component_sequence": total_ms,
    }


def _run_query_pass(
    vector_index: VectorIndex,
    sparse_index: Any,
    graph_index: GraphIndex,
    query_vectors: list[np.ndarray],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    results: list[dict[str, Any]] = []
    samples: dict[str, list[float]] = {
        "vector": [],
        "sparse": [],
        "graph": [],
        "component_sequence": [],
    }
    for query_number, query_vector in enumerate(query_vectors):
        result, timings = _component_query(
            vector_index,
            sparse_index,
            graph_index,
            query_vector,
            f"unique{query_number}",
            f"node-{query_number:08d}",
            top_k,
        )
        results.append(result)
        for component, elapsed_ms in timings.items():
            samples[component].append(elapsed_ms)
    return results, samples


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(encoded)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.vectors > args.max_vectors:
        raise ValueError("--vectors exceeds the explicit --max-vectors safety bound")
    if args.queries > args.vectors:
        raise ValueError("--queries cannot exceed --vectors")
    estimated_vector_working_bytes = args.vectors * EMBEDDING_DIMENSION * 4 * 3
    if estimated_vector_working_bytes > args.max_estimated_working_bytes:
        raise ValueError("estimated vector working set exceeds the configured safety bound")
    if estimated_vector_working_bytes > psutil.virtual_memory().available // 2:
        raise ValueError("estimated vector working set exceeds half of currently available memory")

    sampler = PeakRssSampler()
    sampler.start()
    baseline_rss = sampler.samples[0]
    input_digest = hashlib.sha256()
    query_vectors: list[np.ndarray] = []
    texts: list[tuple[str, str]] = []
    vector_index: VectorIndex | None = None
    sparse_index: Any = None
    graph_index: GraphIndex | None = None

    try:
        rng = np.random.default_rng(args.seed)
        generation_started = time.perf_counter()
        vector_index = VectorIndex(dimension=EMBEDDING_DIMENSION)
        vector_add_seconds = 0.0
        for start in range(0, args.vectors, args.batch_size):
            count = min(args.batch_size, args.vectors - start)
            batch_array = rng.standard_normal(
                (count, EMBEDDING_DIMENSION), dtype=np.float32
            )
            batch: list[tuple[str, np.ndarray]] = []
            for offset in range(count):
                number = start + offset
                node_id = f"node-{number:08d}"
                vector = batch_array[offset]
                input_digest.update(vector.tobytes(order="C"))
                if number < args.queries:
                    query_vectors.append(vector.copy())
                batch.append((node_id, vector))
                text = _synthetic_text(number, args.synthetic_source_tokens_per_chunk)
                input_digest.update(text.encode("utf-8"))
                texts.append((node_id, text))
            add_started = time.perf_counter()
            vector_index.add_batch(batch)
            vector_add_seconds += time.perf_counter() - add_started
        generation_and_vector_add_seconds = time.perf_counter() - generation_started

        with tempfile.TemporaryDirectory(prefix="hybridmind-offline-resource-") as directory:
            serialization_root = Path(directory)
            sparse_index = create_sparse_index(
                "bm25s", index_path=str(serialization_root / "sparse")
            )
            sparse_started = time.perf_counter()
            sparse_index.add_batch(texts)
            # BM25S is lazily materialized, so include materialization in build.
            sparse_index.search("unique0", top_k=min(args.top_k, args.vectors))
            sparse_build_seconds = time.perf_counter() - sparse_started

            graph_index = GraphIndex(index_path=str(serialization_root / "graph.nx"))
            graph_started = time.perf_counter()
            for number, (node_id, _text) in enumerate(texts):
                graph_index.add_node(node_id, synthetic=True)
                if number:
                    graph_index.add_edge(
                        f"node-{number - 1:08d}",
                        node_id,
                        edge_type="next_turn",
                        edge_id=f"edge-{number:08d}",
                    )
            graph_build_seconds = time.perf_counter() - graph_started

            _component_query(
                vector_index,
                sparse_index,
                graph_index,
                query_vectors[0],
                "unique0",
                "node-00000000",
                args.top_k,
            )
            pass_one_results, _pass_one_timings = _run_query_pass(
                vector_index, sparse_index, graph_index, query_vectors, args.top_k
            )
            pass_two_results, pass_two_timings = _run_query_pass(
                vector_index, sparse_index, graph_index, query_vectors, args.top_k
            )

            serialization_started = time.perf_counter()
            vector_index.save(str(serialization_root / "vectors.meta"))
            sparse_index.save()
            graph_index.save()
            serialization_seconds = time.perf_counter() - serialization_started
            file_sizes = recursive_file_sizes(serialization_root)

        sampler.stop()
    except BaseException:
        sampler.stop()
        raise

    assert vector_index is not None and graph_index is not None
    component_sizes = {
        "vector": sum(size for name, size in file_sizes.items() if name.startswith("vectors")),
        "sparse": sum(size for name, size in file_sizes.items() if name.startswith("sparse")),
        "graph": sum(size for name, size in file_sizes.items() if name.startswith("graph")),
    }
    result_hash_one = stable_result_hash(pass_one_results)
    result_hash_two = stable_result_hash(pass_two_results)
    vector_self_hits = sum(
        bool(result["vector"]) and result["vector"][0][0] == f"node-{number:08d}"
        for number, result in enumerate(pass_two_results)
    )
    sparse_self_hits = sum(
        bool(result["sparse"]) and result["sparse"][0][0] == f"node-{number:08d}"
        for number, result in enumerate(pass_two_results)
    )
    total_index_build_seconds = vector_add_seconds + sparse_build_seconds + graph_build_seconds
    measured = {
        "classification": MEASURED_CLASSIFICATION,
        "measurement_scope": "synthetic vector, BM25S, and graph index components; sequential component query, not HTTP end-to-end",
        "embedding_inference_excluded": True,
        "reader_and_reranker_inference_excluded": True,
        "vector_count": args.vectors,
        "query_count": args.queries,
        "top_k": args.top_k,
        "dimension": EMBEDDING_DIMENSION,
        "vector_backend": "faiss_hnsw" if vector_index._use_faiss else "numpy_fallback",
        "sparse_backend": type(sparse_index).__name__,
        "graph_backend": type(graph_index.graph).__name__,
        "synthetic_input_sha256": input_digest.hexdigest(),
        "synthetic_generation_and_vector_add_seconds": generation_and_vector_add_seconds,
        "vector_index_add_seconds": vector_add_seconds,
        "vector_index_add_vectors_per_second": args.vectors / vector_add_seconds,
        "sparse_add_and_materialize_seconds": sparse_build_seconds,
        "sparse_add_and_materialize_documents_per_second": args.vectors / sparse_build_seconds,
        "graph_build_seconds": graph_build_seconds,
        "graph_build_nodes_per_second": args.vectors / graph_build_seconds,
        "total_index_build_seconds": total_index_build_seconds,
        "total_component_build_items_per_second": args.vectors / total_index_build_seconds,
        "serialization_seconds": serialization_seconds,
        "serialized_files_bytes": file_sizes,
        "serialized_component_bytes": component_sizes,
        "serialized_total_bytes": sum(component_sizes.values()),
        "baseline_process_rss_bytes": baseline_rss,
        "rss_samples_bytes": sampler.samples,
        "observed_peak_rss_bytes": max(sampler.samples),
        "observed_rss_increase_bytes": max(0, max(sampler.samples) - baseline_rss),
        "rss_sampling_interval_seconds": sampler.interval_seconds,
        "component_latency_samples_ms": pass_two_timings,
        "component_latency": {
            name: latency_summary(samples) for name, samples in pass_two_timings.items()
        },
        "component_sequence_latency_samples_ms": pass_two_timings["component_sequence"],
        "component_sequence_latency": latency_summary(
            pass_two_timings["component_sequence"]
        ),
        "deterministic_replay_result_sha256_first": result_hash_one,
        "deterministic_replay_result_sha256_second": result_hash_two,
        "deterministic_replay_equal": result_hash_one == result_hash_two,
        "vector_self_hit_rate_at_1": vector_self_hits / args.queries,
        "sparse_self_hit_rate_at_1": sparse_self_hits / args.queries,
    }

    indexed_source_tokens = args.vectors * args.synthetic_source_tokens_per_chunk
    baseline_tokens = args.baseline_prompt_source_tokens or indexed_source_tokens
    retrieved_tokens = args.retrieved_unique_source_tokens or (
        min(args.top_k, args.vectors) * args.synthetic_source_tokens_per_chunk
    )
    usage = {
        "queries": args.planned_queries,
        "embedding_calls": args.embedding_calls,
        "embedding_input_tokens": args.embedding_input_tokens,
        "reranker_calls": args.reranker_calls,
        "reranker_pairs": args.reranker_pairs,
        "reranker_input_tokens": args.reranker_input_tokens,
        "llm_calls": args.llm_calls,
        "reader_input_tokens": args.reader_input_tokens,
        "reader_output_tokens": args.reader_output_tokens,
        "provider_runtime_seconds": args.provider_runtime_seconds,
    }
    rates = {
        "pricing_mode": args.pricing_mode,
        "embedding_usd_per_million_input_tokens": args.embedding_usd_per_million_input_tokens,
        "reranker_usd_per_million_input_tokens": args.reranker_usd_per_million_input_tokens,
        "reader_usd_per_million_input_tokens": args.reader_usd_per_million_input_tokens,
        "reader_usd_per_million_output_tokens": args.reader_usd_per_million_output_tokens,
        "fixed_usd_per_embedding_call": args.fixed_usd_per_embedding_call,
        "fixed_usd_per_reranker_call": args.fixed_usd_per_reranker_call,
        "fixed_usd_per_llm_call": args.fixed_usd_per_llm_call,
        "provider_runtime_usd_per_second": args.provider_runtime_usd_per_second,
    }
    total_host_memory = int(psutil.virtual_memory().total)
    capacity_projections = [
        capacity_projection(
            target,
            args.capacity_mean_source_tokens_per_chunk,
            hnsw_m=DEFAULT_HNSW_M,
            pq_code_bytes=args.pq_code_bytes,
        )
        for target in args.capacity_source_tokens
    ]
    host_capacity_assessments = []
    for projection in capacity_projections:
        lower_bound = projection["total_bytes"][
            "hybridmind_current_vector_component_lower_bound"
        ]
        host_capacity_assessments.append(
            {
                "classification": "scenario_projection",
                "source_tokens": projection["source_tokens"],
                "host_total_memory_bytes": total_host_memory,
                "current_vector_component_lower_bound_bytes": lower_bound,
                "current_vector_component_lower_bound_fraction_of_host_memory": (
                    lower_bound / total_host_memory
                ),
                "passes_vector_lower_bound_half_ram_gate": lower_bound <= total_host_memory / 2,
                "feasibility_status": "not_established_due_excluded_components",
            }
        )
    report = {
        "schema_version": OFFLINE_REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "mode": "offline_synthetic",
            "external_network_calls": 0,
            "embedding_inference_performed": False,
            "actual_external_cost_usd": 0.0,
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_bytes": total_host_memory,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": networkx.__version__,
        },
        "configuration": {
            "seed": args.seed,
            "vectors": args.vectors,
            "queries": args.queries,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "synthetic_source_tokens_per_chunk": args.synthetic_source_tokens_per_chunk,
            "capacity_mean_source_tokens_per_chunk": args.capacity_mean_source_tokens_per_chunk,
            "max_vectors": args.max_vectors,
            "max_estimated_working_bytes": args.max_estimated_working_bytes,
        },
        "measured": measured,
        "capacity_projections": capacity_projections,
        "host_capacity_assessments": host_capacity_assessments,
        "prompt_token_scenario": prompt_reduction(
            indexed_source_tokens=indexed_source_tokens,
            baseline_prompt_source_tokens_per_query=baseline_tokens,
            retrieved_unique_source_tokens_per_query=retrieved_tokens,
        ),
        "tokenomics_scenario": tokenomics_projection(usage, rates),
        "interpretation_limits": [
            "Synthetic self-hit checks are integrity checks, not benchmark quality evidence.",
            "Sequential component latency is not API, network, embedding, reranker, or reader latency.",
            "Capacity values are arithmetic projections, not measured 10M/40M/100M feasibility.",
            "Prompt-source reduction is not transformer KV-cache reduction.",
            "Projected cost is based only on caller-supplied rates and incurs no charge.",
        ],
    }
    validate_offline_report(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
        _write_json_atomic(args.output.resolve(), report)
    except (ValueError, OSError) as exc:
        print(f"offline resource benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "external_network_calls": 0,
                "measured_component_p95_ms": report["measured"]["component_sequence_latency"]["p95_ms"],
                "measured_serialized_total_bytes": report["measured"]["serialized_total_bytes"],
                "capacity_values_are_projections": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
