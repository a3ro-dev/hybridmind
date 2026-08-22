"""Mechanics-only tests for the offline ANN frontier.

These tests deliberately use tiny dimensions and vector counts.  The production
CLI is fixed to the exact 4096-dimensional contract; the test-only escape hatch
is explicitly labelled ``mechanics_test_only`` by the implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import offline_ann_frontier as frontier


def test_production_dimension_is_enforced_outside_test_escape_hatch() -> None:
    with pytest.raises(ValueError, match="dimension=4096"):
        frontier.run_frontier(
            sizes=[8],
            seeds=[0],
            build_orders=["natural"],
            backends=["flat_ip"],
            query_count=2,
            dimension=8,
        )


def test_tiny_mechanics_frontier_has_exact_oracle_and_zero_calls() -> None:
    report = frontier.run_frontier(
        sizes=[16],
        seeds=[0, 1],
        build_orders=["natural", "reverse"],
        backends=["flat_ip", "hnsw_flat", "hnsw_sq8"],
        query_count=4,
        dimension=8,
        mechanics_test_only=True,
    )
    assert report["schema_version"] == "hybridmind.offline-ann-frontier/v3"
    assert report["provider_calls"] == 0
    assert report["network_calls"] == 0
    assert report["execution"]["external_network_calls"] == 0
    assert report["mechanics_test_only"] is True
    assert report["provenance"]["source"]["sha256"]
    assert report["declared_workload"]["faiss_threads"] == 1
    assert report["declared_workload"]["ef_search_values"] == [64]
    assert report["declared_workload"]["ef_construction_values"] == [80]
    assert len(report["results"]) == 2 * 2 * 3
    for row in report["results"]:
        assert row["evidence_class"] == "measured_offline"
        assert 0.0 <= row["recall_at_1"] <= 1.0
        assert 0.0 <= row["recall_at_10"] <= 1.0
        assert row["serialized_index_bytes"] > 0
        assert row["cold_search_ms"] >= 0.0
        assert row["deterministic_replay"]["search_ids_equal"] is True
        assert "mutation_delete" in row
        if row["backend"] == "flat_ip":
            assert row["executed_hnsw_controls"] is None
        else:
            assert row["executed_hnsw_controls"] == {
                "ef_search": row["hnsw_ef_search"],
                "ef_construction": frontier.HNSW_EF_CONSTRUCTION,
            }
    flat_rows = [row for row in report["results"] if row["backend"] == "flat_ip"]
    assert all(row["recall_at_1"] == 1.0 for row in flat_rows)
    assert all(row["recall_at_10"] == 1.0 for row in flat_rows)


def test_hnsw_search_budget_is_an_explicit_experimental_axis() -> None:
    report = frontier.run_frontier(
        sizes=[16],
        seeds=[0],
        build_orders=["natural"],
        backends=["flat_ip", "hnsw_flat"],
        query_count=2,
        ef_search_values=[8, 16],
        dimension=8,
        mechanics_test_only=True,
    )
    flat = [row for row in report["results"] if row["backend"] == "flat_ip"]
    hnsw = [row for row in report["results"] if row["backend"] == "hnsw_flat"]
    assert len(flat) == 1
    assert flat[0]["hnsw_ef_search"] is None
    assert {row["hnsw_ef_search"] for row in hnsw} == {8, 16}


def test_hnsw_construction_budget_is_an_explicit_experimental_axis() -> None:
    report = frontier.run_frontier(
        sizes=[16],
        seeds=[0],
        build_orders=["natural"],
        backends=["hnsw_flat"],
        query_count=2,
        ef_search_values=[8],
        ef_construction_values=[16, 32],
        dimension=8,
        mechanics_test_only=True,
    )
    assert {row["hnsw_ef_construction"] for row in report["results"]} == {16, 32}
    assert {
        row["executed_hnsw_controls"]["ef_construction"]
        for row in report["results"]
    } == {16, 32}


def test_requested_backend_does_not_silently_substitute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontier, "_require_faiss", lambda: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(frontier.BackendUnavailableError, match="FAISS is required"):
        frontier.build_index(
            "hnsw_flat",
            np.ones((4, 8), dtype=np.float32),
            np.arange(4, dtype=np.int64),
            dimension=8,
        )


def test_atomic_json_write_replaces_target_and_leaves_no_temp(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "frontier.json"
    frontier.write_json_atomic({"schema_version": "test", "provider_calls": 0}, output)
    assert json.loads(output.read_text(encoding="utf-8"))["provider_calls"] == 0
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_memory_gate_refuses_over_hard_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frontier, "available_memory_budget_bytes", lambda: 512 * 1024 * 1024)
    with pytest.raises(frontier.ResourceGateError, match="hard cap"):
        frontier.run_frontier(
            sizes=[frontier.HARD_MAX_VECTORS + 1],
            seeds=[0],
            build_orders=["natural"],
            backends=["flat_ip"],
            query_count=1,
            dimension=8,
            mechanics_test_only=True,
        )
