"""Deterministic exact-evidence graph ablation with adversarial invariants.

This is intentionally a narrow relationship-retrieval test, not a claim of
general benchmark superiority. Explicit anchors are query-derived identifiers
and never selected from gold answers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.graph_search import GraphSearchEngine
from engine.hybrid_ranker import HybridRanker
from engine.vector_search import VectorSearchEngine
from storage.bm25_index import BM25Index
from storage.graph_index import GraphIndex
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from tests.embedding_double import Deterministic4096EmbeddingEngine


SCHEMA = "hybridmind.offline-graph-ablation/v1"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _bootstrap_delta(
    baseline: list[float], graph: list[float], *, seed: int, samples: int = 4000
) -> dict:
    if len(baseline) != len(graph) or not baseline:
        raise ValueError("paired non-empty observations are required")
    rng = np.random.default_rng(seed)
    differences = np.asarray(graph) - np.asarray(baseline)
    indexes = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indexes].mean(axis=1)
    result = {
        "delta": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
    }
    return result


def _add_node(
    store: SQLiteStore,
    vectors: VectorIndex,
    sparse: BM25Index,
    graph: GraphIndex,
    embedder: Deterministic4096EmbeddingEngine,
    *,
    node_id: str,
    text: str,
    metadata: dict,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> None:
    embedding = embedder.embed(text)
    store.create_node(
        node_id=node_id,
        text=text,
        metadata=metadata,
        embedding=embedding,
        raw_embedding=embedding,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    vectors.add(node_id, embedding)
    sparse.add(node_id, text)
    graph.add_node(node_id, valid_from=valid_from, valid_until=valid_until)


def _add_edge(
    store: SQLiteStore,
    graph: GraphIndex,
    *,
    edge_id: str,
    source: str,
    target: str,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> None:
    store.create_edge(
        edge_id,
        source,
        target,
        "supports",
        1.0,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    graph.add_edge(
        source,
        target,
        "supports",
        weight=1.0,
        edge_id=edge_id,
        valid_from=valid_from,
        valid_until=valid_until,
    )


def _ids(results: list[dict], k: int) -> set[str]:
    return {
        str((result.get("metadata") or {}).get("evidence_id"))
        for result in results[:k]
        if (result.get("metadata") or {}).get("evidence_id")
    }


def _assert_trace(trace: dict, mode: str) -> None:
    expected = {
        "vector_sparse": {"dense", "sparse"},
        "graph_only": {"graph"},
    }[mode]
    if trace.get("schema_version") != "hybridmind.search-execution/v1":
        raise RuntimeError("missing search execution trace")
    for stage in ("dense", "sparse", "graph"):
        evidence = trace["stages"][stage]
        if evidence.get("requested") is not (stage in expected):
            raise RuntimeError(f"invalid requested flag for {stage}")
        if evidence.get("executed") is not (stage in expected):
            raise RuntimeError(f"controlled stage did not execute: {stage}")


def run_ablation(
    work_dir: Path,
    *,
    cases: int = 40,
    distractors_per_case: int = 8,
    seed: int = 20260814,
) -> dict:
    if cases < 2 or distractors_per_case < 3:
        raise ValueError("at least two cases and three distractors are required")
    work_dir.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(work_dir / "graph-ablation.db")
    vectors = VectorIndex(dimension=4096)
    sparse = BM25Index()
    graph = GraphIndex()
    embedder = Deterministic4096EmbeddingEngine()

    case_rows = []
    for case in range(cases):
        scope = f"relationship-case-{case:03d}"
        anchor_id = f"anchor-{case:03d}"
        gold_id = f"gold-{case:03d}"
        evidence_id = f"synthetic:{scope}:gold"
        query = f"trace relation marker {case:03d} nexus"
        _add_node(
            store, vectors, sparse, graph, embedder,
            node_id=anchor_id,
            text=f"{query} explicit source anchor",
            metadata={
                "benchmark_sample_id": scope,
                "evidence_id": f"synthetic:{scope}:anchor",
                "anchor_key": f"marker-{case:03d}",
            },
        )
        _add_node(
            store, vectors, sparse, graph, embedder,
            node_id=gold_id,
            text=f"latent payload {case:03d} cobalt archive",
            metadata={"benchmark_sample_id": scope, "evidence_id": evidence_id},
        )
        _add_edge(
            store, graph,
            edge_id=f"edge-{case:03d}", source=anchor_id, target=gold_id,
        )
        for distractor in range(distractors_per_case):
            _add_node(
                store, vectors, sparse, graph, embedder,
                node_id=f"distractor-{case:03d}-{distractor:02d}",
                text=(
                    f"{query} lexical distractor {distractor:02d} "
                    "unrelated catalogue"
                ),
                metadata={
                    "benchmark_sample_id": scope,
                    "evidence_id": f"synthetic:{scope}:distractor:{distractor}",
                },
            )
        case_rows.append((scope, anchor_id, evidence_id, query))

    # A connected target in the wrong scope must never leak through expansion.
    _add_node(
        store, vectors, sparse, graph, embedder,
        node_id="cross-scope-target",
        text="malicious connected target",
        metadata={
            "benchmark_sample_id": "different-scope",
            "evidence_id": "synthetic:different-scope:malicious",
        },
    )
    _add_edge(
        store, graph,
        edge_id="cross-scope-edge", source=case_rows[0][1], target="cross-scope-target",
    )

    ranker = HybridRanker(
        VectorSearchEngine(vectors, store, embedder),
        GraphSearchEngine(graph, store),
        sparse,
        query_routing_enabled=False,
        temporal_decay_enabled=False,
    )
    baseline_hits: list[float] = []
    graph_hits: list[float] = []
    baseline_latency: list[float] = []
    graph_latency: list[float] = []
    rows = []
    for scope, anchor_id, evidence_id, query in case_rows:
        baseline, baseline_ms, _, baseline_trace = ranker.search(
            query,
            top_k=2,
            rerank_pool=0,
            search_mode="vector_sparse",
            vector_weight=0.5,
            graph_weight=0.0,
            bm25_boost_weight=0.5,
            route_weights=False,
            track_access=False,
            filter_metadata={"benchmark_sample_id": scope},
            return_trace=True,
        )
        graph_results, graph_ms, _, graph_trace = ranker.search(
            query,
            top_k=2,
            rerank_pool=0,
            search_mode="graph_only",
            vector_weight=0.0,
            graph_weight=1.0,
            bm25_boost_weight=0.0,
            route_weights=False,
            track_access=False,
            anchor_nodes=[anchor_id],
            filter_metadata={"benchmark_sample_id": scope},
            return_trace=True,
        )
        _assert_trace(baseline_trace, "vector_sparse")
        _assert_trace(graph_trace, "graph_only")
        baseline_hit = float(evidence_id in _ids(baseline, 2))
        graph_hit = float(evidence_id in _ids(graph_results, 2))
        baseline_hits.append(baseline_hit)
        graph_hits.append(graph_hit)
        baseline_latency.append(baseline_ms)
        graph_latency.append(graph_ms)
        rows.append({
            "scope": scope,
            "anchor_id": anchor_id,
            "gold_evidence_id": evidence_id,
            "baseline_hit_at_2": bool(baseline_hit),
            "graph_hit_at_2": bool(graph_hit),
            "baseline_retrieved_evidence_ids": sorted(_ids(baseline, 2)),
            "graph_retrieved_evidence_ids": sorted(_ids(graph_results, 2)),
            "baseline_trace": baseline_trace,
            "graph_trace": graph_trace,
        })

    scoped, _, _, _ = ranker.search(
        case_rows[0][3], top_k=10, rerank_pool=0,
        search_mode="graph_only", anchor_nodes=[case_rows[0][1]],
        vector_weight=0.0, graph_weight=1.0, bm25_boost_weight=0.0,
        route_weights=False, track_access=False,
        filter_metadata={"benchmark_sample_id": case_rows[0][0]}, return_trace=True,
    )
    scope_leakage = "cross-scope-target" in {row["node_id"] for row in scoped}

    temporal_scope = "temporal-adversarial"
    _add_node(
        store, vectors, sparse, graph, embedder,
        node_id="temporal-anchor", text="temporal explicit anchor",
        metadata={"benchmark_sample_id": temporal_scope, "evidence_id": "temporal:anchor"},
    )
    _add_node(
        store, vectors, sparse, graph, embedder,
        node_id="temporal-gold", text="bounded temporal payload",
        metadata={"benchmark_sample_id": temporal_scope, "evidence_id": "temporal:gold"},
        valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2026-02-01T00:00:00+00:00",
    )
    _add_edge(
        store, graph, edge_id="temporal-edge", source="temporal-anchor",
        target="temporal-gold", valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2026-02-01T00:00:00+00:00",
    )
    common_temporal = dict(
        query_text="temporal explicit anchor", top_k=3, rerank_pool=0,
        search_mode="graph_only", anchor_nodes=["temporal-anchor"],
        vector_weight=0.0, graph_weight=1.0, bm25_boost_weight=0.0,
        route_weights=False, track_access=False,
        filter_metadata={"benchmark_sample_id": temporal_scope},
    )
    january, _, _ = ranker.search(
        **common_temporal, as_of=datetime(2026, 1, 15, tzinfo=timezone.utc)
    )
    february, _, _ = ranker.search(
        **common_temporal, as_of=datetime(2026, 2, 15, tzinfo=timezone.utc)
    )
    temporal_pass = (
        "temporal:gold" in _ids(january, 3)
        and "temporal:gold" not in _ids(february, 3)
    )

    provenance_scope = "provenance-adversarial"
    _add_node(
        store, vectors, sparse, graph, embedder,
        node_id="provenance-anchor", text="provenance explicit anchor",
        metadata={"benchmark_sample_id": provenance_scope, "evidence_id": "provenance:anchor"},
    )
    for suffix in ("one", "two"):
        _add_node(
            store, vectors, sparse, graph, embedder,
            node_id=f"provenance-{suffix}", text="identical citable graph evidence",
            metadata={
                "benchmark_sample_id": provenance_scope,
                "evidence_id": f"provenance:{suffix}",
            },
        )
        _add_edge(
            store, graph, edge_id=f"provenance-edge-{suffix}",
            source="provenance-anchor", target=f"provenance-{suffix}",
        )
    provenance_results, _, _ = ranker.search(
        "provenance explicit anchor", top_k=3, rerank_pool=0,
        search_mode="graph_only", anchor_nodes=["provenance-anchor"],
        vector_weight=0.0, graph_weight=1.0, bm25_boost_weight=0.0,
        route_weights=False, track_access=False,
        filter_metadata={"benchmark_sample_id": provenance_scope},
    )
    provenance_ids = _ids(provenance_results, 3)
    provenance_pass = {"provenance:one", "provenance:two"}.issubset(provenance_ids)

    paired = _bootstrap_delta(baseline_hits, graph_hits, seed=seed)
    success = bool(
        paired["ci95_low"] > 0.0
        and not scope_leakage
        and temporal_pass
        and provenance_pass
    )
    result = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "provider_calls": 0,
        "seed": seed,
        "design": {
            "cases": cases,
            "distractors_per_case": distractors_per_case,
            "split": "deterministic synthetic relationship invariant",
            "metric_basis": "exact_evidence_id",
            "anchor_policy": "explicit query-derived anchor_key; gold-independent",
            "claim_boundary": "synthetic relationship retrieval only",
        },
        "conditions": {
            "vector_sparse": {
                "hit_at_2": statistics.fmean(baseline_hits),
                "latency_ms_mean": statistics.fmean(baseline_latency),
            },
            "graph_only": {
                "hit_at_2": statistics.fmean(graph_hits),
                "latency_ms_mean": statistics.fmean(graph_latency),
            },
        },
        "paired_effect": paired,
        "adversarial": {
            "cross_scope_leakage": scope_leakage,
            "historical_as_of_half_open_validity_pass": temporal_pass,
            "identical_text_distinct_provenance_pass": provenance_pass,
        },
        "hypothesis_h8_success": success,
        "rows": rows,
    }
    store.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--distractors-per-case", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/results/offline-graph-ablation-20260814.json"),
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="hybridmind_graph_ablation_") as temp:
        result = run_ablation(
            Path(temp), cases=args.cases,
            distractors_per_case=args.distractors_per_case,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
