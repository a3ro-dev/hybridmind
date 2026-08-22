"""Behavioral tests for the optional TurboVec/TurboQuant research harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import offline_turbovec_frontier as frontier


def test_small_mechanics_frontier_is_attested_and_gold_free() -> None:
    report = frontier.run_frontier(
        vector_count=64,
        query_count=8,
        seeds=[3],
        bit_widths=[4],
        calibration_modes=[False, True],
        repetitions=2,
        dimension=32,
        mechanics_test_only=True,
    )

    assert report["schema_version"] == frontier.SCHEMA
    assert report["provider_calls"] == 0
    assert report["execution"]["external_network_calls"] == 0
    assert report["workload"]["mechanics_test_only"] is True
    assert len(report["results"]) == 2
    assert {row["calibrated"] for row in report["results"]} == {False, True}
    for row in report["results"]:
        assert 0.0 <= row["recall_at_1"] <= 1.0
        assert 0.0 <= row["recall_at_10"] <= 1.0
        assert row["serialized_index_bytes"] > 0
        assert row["deterministic_replay"] == {
            "ids_equal": True,
            "scores_equal": True,
        }
        assert row["persistence_roundtrip"] == {
            "ids_equal": True,
            "scores_equal": True,
        }
        assert row["stable_id_delete"]["deleted_id_absent_from_probe"] is True


def test_production_dimension_cannot_be_silently_reduced() -> None:
    with pytest.raises(ValueError, match="dimension=4096"):
        frontier.run_frontier(
            vector_count=64,
            query_count=8,
            seeds=[0],
            bit_widths=[4],
            calibration_modes=[False],
            repetitions=2,
            dimension=32,
        )


def test_atomic_writer_never_overwrites_receipt(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    frontier.write_json_atomic_exclusive(output, {"value": 1})

    with pytest.raises(FileExistsError):
        frontier.write_json_atomic_exclusive(output, {"value": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}


def test_resource_gate_rejects_unbounded_vector_count() -> None:
    with pytest.raises(frontier.ResourceGateError, match="vector_count"):
        frontier.run_frontier(
            vector_count=frontier.HARD_MAX_VECTORS + 1,
            query_count=8,
            seeds=[0],
            bit_widths=[4],
            calibration_modes=[False],
            repetitions=2,
        )
