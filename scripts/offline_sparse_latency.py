"""Repeated, randomized, provider-free LoCoMo sparse latency benchmark.

This benchmark measures only BM25S representation/index/query costs.  It does
not read gold answers or evidence annotations; quality belongs to the separate
exact-evidence experiment in ``offline_locomo_sparse_experiments.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_locomo_sparse_baseline import DEFAULT_DATASET, _sha256
from scripts.offline_locomo_sparse_experiments import _keys, _split_ids, _turn_records
from storage.bm25_index import BM25SBackend


SCHEMA = "hybridmind.offline-locomo-sparse-latency/v1"
CONDITIONS = ("raw", "speaker_prefix")
DEFAULT_SEED = 20260822
DEFAULT_MAX_CONVERSATIONS = 5
DEFAULT_MAX_QUERIES = 32
DEFAULT_BLOCKS = 2
DEFAULT_REPETITIONS = 3
DEFAULT_WARMUPS = 1
DEFAULT_COLD_BUILDS = 3
DEFAULT_BOOTSTRAP = 1000
DEFAULT_TOP_K = 25
MAX_CONVERSATIONS = 10
MAX_QUERIES = 128
MAX_BLOCKS = 8
MAX_REPETITIONS = 8
MAX_WARMUPS = 8
MAX_COLD_BUILDS = 8
MAX_BOOTSTRAP = 5000


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("bm25s", "numpy", "PyStemmer"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _validate_bounds(
    *, max_conversations: int, max_queries: int, blocks: int,
    repetitions: int, warmups: int, cold_builds: int, bootstrap_samples: int,
) -> None:
    bounds = {
        "max_conversations": (max_conversations, 1, MAX_CONVERSATIONS),
        "max_queries": (max_queries, 1, MAX_QUERIES),
        "blocks": (blocks, 1, MAX_BLOCKS),
        "repetitions": (repetitions, 1, MAX_REPETITIONS),
        "warmups": (warmups, 0, MAX_WARMUPS),
        "cold_builds": (cold_builds, 1, MAX_COLD_BUILDS),
        "bootstrap_samples": (bootstrap_samples, 100, MAX_BOOTSTRAP),
    }
    for name, (value, lower, upper) in bounds.items():
        if not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def select_conversations(
    data: list[dict], *, seed: int, max_conversations: int,
) -> list[str]:
    """Select whole conversations by a stable hash, without reading QA gold."""
    if max_conversations < 1:
        raise ValueError("max_conversations must be positive")
    sample_ids = [str(item.get("sample_id") or "").strip() for item in data]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("LoCoMo sample IDs must be unique and non-empty")
    ranked = sorted(
        sample_ids,
        key=lambda value: hashlib.sha256(
            f"conversation:{seed}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[: min(max_conversations, len(ranked))]


def build_timing_inputs(
    data: list[dict], *, sample_ids: Iterable[str], seed: int,
    max_queries: int,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """Return index records and question-only timing inputs.

    Deliberately no ``evidence``, ``answer``, or gold helper is accessed here.
    """
    allowed = set(sample_ids)
    records_by_sample: dict[str, list[dict[str, str]]] = {}
    questions: list[dict[str, str]] = []
    for item in data:
        sample_id = str(item.get("sample_id") or "").strip()
        if sample_id not in allowed:
            continue
        records = _turn_records(item)
        records_by_sample[sample_id] = records
        for qa_index, qa in enumerate(item.get("qa") or []):
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            question_id = hashlib.sha256(
                f"question:{seed}:{sample_id}:{qa_index}:{question}".encode("utf-8")
            ).hexdigest()
            questions.append({
                "question_id": question_id,
                "sample_id": sample_id,
                "question": question,
                "qa_index": str(qa_index),
            })
    questions.sort(key=lambda row: (row["question_id"], row["sample_id"]))
    if len(questions) > max_queries:
        questions = questions[:max_queries]
    if not questions:
        raise ValueError("timing selection has no non-empty questions")
    return records_by_sample, questions


def make_interleaved_schedule(
    question_ids: Iterable[str], *, conditions: tuple[str, ...] = CONDITIONS,
    blocks: int, repetitions: int, seed: int,
) -> list[dict[str, Any]]:
    """Create deterministic paired schedules with randomized condition order."""
    ids = sorted(set(question_ids))
    if not ids or not conditions:
        raise ValueError("schedule requires questions and conditions")
    if len(conditions) != len(set(conditions)):
        raise ValueError("conditions must be unique")
    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    for block in range(blocks):
        for repetition in range(repetitions):
            query_order = list(ids)
            rng.shuffle(query_order)
            for query_id in query_order:
                condition_order = list(conditions)
                rng.shuffle(condition_order)
                for condition in condition_order:
                    schedule.append({
                        "block": block,
                        "repetition": repetition,
                        "question_id": query_id,
                        "condition": condition,
                    })
    return schedule


def _condition_index(
    records_by_sample: dict[str, list[dict[str, str]]], condition: str,
) -> BM25SBackend:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    rows: list[tuple[str, str]] = []
    for sample_id in sorted(records_by_sample):
        for record in records_by_sample[sample_id]:
            for key_type, key_text in _keys(record, condition):
                rows.append((f"{record['source_id']}|{key_type}", key_text))
    if not rows:
        raise ValueError("timing index has no records")
    index = BM25SBackend()
    index.add_batch(rows)
    return index


def _timed_search(index: BM25SBackend, question: str, top_k: int) -> tuple[float, float]:
    wall_start = time.perf_counter_ns()
    process_start = time.process_time_ns()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stderr(sink):
            index.search(question, top_k=top_k)
    process_ns = time.process_time_ns() - process_start
    wall_ns = time.perf_counter_ns() - wall_start
    if wall_ns < 0 or process_ns < 0:
        raise RuntimeError("monotonic timer moved backwards")
    return wall_ns / 1_000_000.0, process_ns / 1_000_000.0


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile * 100.0))


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, value_key: str, seed: int,
    samples: int, statistic: Callable[[list[float]], float],
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        clusters.setdefault(str(row["sample_id"]), []).append(float(row[value_key]))
    values = [value for cluster in clusters.values() for value in cluster]
    if not values:
        return {"estimate": None, "ci95_low": None, "ci95_high": None,
                "n": 0, "clusters": 0, "bootstrap_samples": samples}
    if len(clusters) < 2:
        return {
            "estimate": float(statistic(values)),
            "ci95_low": None,
            "ci95_high": None,
            "n": len(values),
            "clusters": len(clusters),
            "bootstrap_samples": 0,
            "warning": "cluster interval requires at least two independent clusters",
        }
    rng = np.random.default_rng(seed)
    cluster_values = list(clusters.values())
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(cluster_values), size=len(cluster_values))
        resampled = [value for selected_cluster in selected for value in cluster_values[selected_cluster]]
        estimates[index] = statistic(resampled)
    estimates.sort()
    return {
        "estimate": float(statistic(values)),
        "ci95_low": float(estimates[math.floor(0.025 * (samples - 1))]),
        "ci95_high": float(estimates[math.ceil(0.975 * (samples - 1))]),
        "n": len(values),
        "clusters": len(cluster_values),
        "bootstrap_samples": samples,
    }


def summarize_latency(
    rows: list[dict[str, Any]], *, value_key: str, seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    values = [float(row[value_key]) for row in rows]
    statistics: dict[str, tuple[float, Callable[[list[float]], float]]] = {
        "mean": (0.0, lambda values: float(np.mean(values))),
        "p50": (0.50, lambda values: _percentile(values, 0.50) or 0.0),
        "p95": (0.95, lambda values: _percentile(values, 0.95) or 0.0),
        "p99": (0.99, lambda values: _percentile(values, 0.99) or 0.0),
    }
    result: dict[str, Any] = {"n": len(values)}
    for offset, (name, (_, statistic)) in enumerate(statistics.items()):
        result[name] = statistic(values) if values else None
        result[f"{name}_cluster_bootstrap_ci95"] = _cluster_bootstrap(
            rows, value_key=value_key, seed=seed + offset,
            samples=bootstrap_samples, statistic=statistic,
        )
    return result


def paired_deltas(
    rows: list[dict[str, Any]], *, baseline: str = "raw",
    candidate: str = "speaker_prefix", value_key: str = "wall_ms",
    seed: int, bootstrap_samples: int,
) -> dict[str, Any]:
    keyed = {
        (row["block"], row["repetition"], row["question_id"], row["condition"]): row
        for row in rows
    }
    pairs: list[dict[str, Any]] = []
    for block, repetition, question_id in sorted({
        (row["block"], row["repetition"], row["question_id"]) for row in rows
    }):
        left = keyed.get((block, repetition, question_id, baseline))
        right = keyed.get((block, repetition, question_id, candidate))
        if left is None or right is None:
            raise ValueError("paired schedule is incomplete")
        pairs.append({
            "block": block,
            "repetition": repetition,
            "question_id": question_id,
            "sample_id": left["sample_id"],
            "delta_ms": float(right[value_key]) - float(left[value_key]),
        })
    return {
        "candidate_minus_baseline": f"{candidate}-{baseline}",
        "value": value_key,
        "rows": pairs,
        "summary": summarize_latency(
            pairs, value_key="delta_ms", seed=seed,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def _timer_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("perf_counter", "process_time"):
        info = time.get_clock_info(name)
        metadata[name] = {
            "implementation": info.implementation,
            "monotonic": info.monotonic,
            "adjustable": info.adjustable,
            "resolution_seconds": info.resolution,
            "resolution_ms": info.resolution * 1000.0,
        }
    return metadata


def _empirical_timer_diagnostics(
    rows: list[dict[str, Any]], *, value_key: str,
) -> dict[str, Any]:
    values = [abs(float(row[value_key])) for row in rows]
    positives = sorted(value for value in values if value > 0.0)
    unique = sorted(set(values))
    return {
        "observations": len(values),
        "zero_fraction": (
            sum(value == 0.0 for value in values) / len(values) if values else None
        ),
        "smallest_positive_ms": positives[0] if positives else None,
        "unique_value_count": len(unique),
        "decision_eligible": bool(
            values
            and positives
            and sum(value == 0.0 for value in values) / len(values) < 0.50
        ),
    }


def run(
    dataset: Path, *, seed: int = DEFAULT_SEED,
    max_conversations: int = DEFAULT_MAX_CONVERSATIONS,
    max_queries: int = DEFAULT_MAX_QUERIES,
    blocks: int = DEFAULT_BLOCKS, repetitions: int = DEFAULT_REPETITIONS,
    warmups: int = DEFAULT_WARMUPS, cold_builds: int = DEFAULT_COLD_BUILDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP, top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    _validate_bounds(
        max_conversations=max_conversations, max_queries=max_queries,
        blocks=blocks, repetitions=repetitions, warmups=warmups,
        cold_builds=cold_builds, bootstrap_samples=bootstrap_samples,
    )
    if top_k < 1 or top_k > 1000:
        raise ValueError("top_k must be in [1, 1000]")
    started_wall = time.perf_counter()
    started_process = time.process_time()
    raw_data = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list) or len(raw_data) < 1:
        raise ValueError("LoCoMo dataset must contain at least one conversation")
    selected_ids = select_conversations(
        raw_data, seed=seed, max_conversations=max_conversations,
    )
    records_by_sample, questions = build_timing_inputs(
        raw_data, sample_ids=selected_ids, seed=seed, max_queries=max_queries,
    )
    question_by_id = {row["question_id"]: row for row in questions}
    schedule = make_interleaved_schedule(
        question_by_id, blocks=blocks, repetitions=repetitions, seed=seed,
    )
    condition_order_rng = random.Random(seed + 1)
    cold_build_rows: list[dict[str, Any]] = []
    for repetition in range(cold_builds):
        conditions = list(CONDITIONS)
        condition_order_rng.shuffle(conditions)
        for condition in conditions:
            wall_start = time.perf_counter_ns()
            process_start = time.process_time_ns()
            _condition_index(records_by_sample, condition)
            cold_build_rows.append({
                "repetition": repetition,
                "condition": condition,
                "sample_id": "__selected_conversations__",
                "wall_ms": (time.perf_counter_ns() - wall_start) / 1_000_000.0,
                "process_ms": (time.process_time_ns() - process_start) / 1_000_000.0,
            })
    indexes = {condition: _condition_index(records_by_sample, condition) for condition in CONDITIONS}
    for condition in CONDITIONS:
        warmup_questions = questions[:]
        for _ in range(warmups):
            for question in warmup_questions:
                _timed_search(indexes[condition], question["question"], top_k)
    query_rows: list[dict[str, Any]] = []
    for entry in schedule:
        question = question_by_id[entry["question_id"]]
        wall_ms, process_ms = _timed_search(
            indexes[entry["condition"]], question["question"], top_k,
        )
        query_rows.append({
            **entry,
            "sample_id": question["sample_id"],
            "wall_ms": wall_ms,
            "process_ms": process_ms,
        })
    by_condition: dict[str, Any] = {}
    for offset, condition in enumerate(CONDITIONS):
        selected = [row for row in query_rows if row["condition"] == condition]
        builds = [row for row in cold_build_rows if row["condition"] == condition]
        by_condition[condition] = {
            "warm_query_wall_ms": summarize_latency(
                selected, value_key="wall_ms", seed=seed + offset * 100,
                bootstrap_samples=bootstrap_samples,
            ),
            "warm_query_process_ms": summarize_latency(
                selected, value_key="process_ms", seed=seed + offset * 100 + 10,
                bootstrap_samples=bootstrap_samples,
            ),
            "cold_build_wall_ms": summarize_latency(
                builds, value_key="wall_ms", seed=seed + offset * 100 + 20,
                bootstrap_samples=bootstrap_samples,
            ),
            "cold_build_process_ms": summarize_latency(
                builds, value_key="process_ms", seed=seed + offset * 100 + 30,
                bootstrap_samples=bootstrap_samples,
            ),
        }
    status = _git_output("status", "--porcelain=v1", "-z") or ""
    source_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA,
        "experiment_id": "locomo-sparse-latency-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "measured_offline_repeated_randomized_paired_latency_experiment",
        "provider_calls": 0,
        "execution": {
            "provider_calls": 0,
            "external_network_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "quality_evaluation_performed": False,
            "elapsed_wall_seconds": time.perf_counter() - started_wall,
            "elapsed_process_seconds": time.process_time() - started_process,
        },
        "provenance": {
            "dataset": {"path": str(dataset.resolve()), "sha256": _sha256(dataset)},
            "source": {"path": str(source_path), "sha256": _sha256(source_path)},
            "quality_experiment_source": str((PROJECT_ROOT / "scripts" / "offline_locomo_sparse_experiments.py").resolve()),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree": {"dirty": bool(status), "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest()},
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "dependency_versions": _dependency_versions(),
            "seed": seed,
            "timer_metadata": _timer_metadata(),
        },
        "selection": {
            "conversation_ids": selected_ids,
            "question_count": len(questions),
            "question_ids": [question["question_id"] for question in questions],
            "selection_is_gold_free": True,
            "timing_inputs_fields": ["question_id", "sample_id", "question", "qa_index"],
        },
        "parameters": {
            "conditions": list(CONDITIONS), "blocks": blocks, "repetitions": repetitions,
            "warmups": warmups, "cold_builds": cold_builds, "top_k": top_k,
            "bootstrap_samples": bootstrap_samples,
        },
        "schedule": {
            "rows": schedule,
            "paired_per_block_repetition_query": True,
            "condition_order_randomized": True,
            "schedule_sha256": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode("utf-8")).hexdigest(),
        },
        "cold_build_rows": cold_build_rows,
        "query_rows": query_rows,
        "by_condition": by_condition,
        "paired_deltas": {
            "wall_ms": paired_deltas(query_rows, value_key="wall_ms", seed=seed + 1000, bootstrap_samples=bootstrap_samples),
            "process_ms": paired_deltas(query_rows, value_key="process_ms", seed=seed + 2000, bootstrap_samples=bootstrap_samples),
        },
        "empirical_timer_diagnostics": {
            "wall_ms": _empirical_timer_diagnostics(
                query_rows, value_key="wall_ms",
            ),
            "process_ms": _empirical_timer_diagnostics(
                query_rows, value_key="process_ms",
            ),
        },
        "interpretation_limits": [
            "Timing inputs contain questions only; no answers or gold evidence are read.",
            "Quality comes from the separate exact-evidence experiment and is not measured here.",
            "Warm-query timing excludes index construction; cold-build timing excludes query execution.",
            "Process time is CPU time for this process and omits waiting outside the process; wall time is the serving-cost proxy.",
            "A timer with at least 50% zero-valued observations is not decision-eligible for this workload.",
            "This benchmark measures BM25S raw versus speaker_prefix only, not end-to-end retrieval or answer quality.",
        ],
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-conversations", type=int, default=DEFAULT_MAX_CONVERSATIONS)
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--cold-builds", type=int, default=DEFAULT_COLD_BUILDS)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)
    output = args.output or PROJECT_ROOT / "experiments" / "results" / f"offline-locomo-sparse-latency-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    result = run(
        args.dataset.resolve(), seed=args.seed, max_conversations=args.max_conversations,
        max_queries=args.max_queries, blocks=args.blocks, repetitions=args.repetitions,
        warmups=args.warmups, cold_builds=args.cold_builds,
        bootstrap_samples=args.bootstrap_samples, top_k=args.top_k,
    )
    _atomic_write(output.resolve(), result)
    print(json.dumps({"output": str(output.resolve()), "provider_calls": 0, "question_count": result["selection"]["question_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
