"""Behavioral tests for the strictly offline sparse field-routing experiment."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import offline_sparse_field_routing as routing


def _conversation(sample_id: str, first: str, second: str) -> dict:
    return {
        "sample_id": sample_id,
        "conversation": {
            "session_1": [
                {"dia_id": "D1:1", "speaker": first, "text": f"{first} likes blue bicycles."},
                {"dia_id": "D1:2", "speaker": second, "text": f"{second} keeps a red notebook."},
                {"dia_id": "D1:3", "speaker": first, "text": "The meeting is next Tuesday."},
            ],
            "session_1_date_time": "2024-01-01 10:00:00",
        },
        "qa": [
            {
                "question": f"What does {first} like?",
                "answer": "blue bicycles",
                "evidence": ["D1:1"],
                "category": 1,
            },
            {
                "question": "When is the meeting?",
                "answer": "next Tuesday",
                "evidence": ["D1:3"],
                "category": 2,
            },
            {
                "question": f"What does {second} keep?",
                "answer": "a red notebook",
                "evidence": ["D1:2"],
                "category": 3,
            },
        ],
    }


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "locomo-mini.json"
    path.write_text(
        json.dumps([
            _conversation("conv-a", "Alice", "Bob"),
            _conversation("conv-b", "Carol", "Dan"),
        ]),
        encoding="utf-8",
    )
    return path


def test_router_is_normalized_boundary_match_and_has_no_gold_input():
    assert routing.normalize_query("  ALICE's plan? ") == "alice s plan"
    assert routing.route_representation("What did ALICE say?", ["Alice", "Bob"]) == "speaker_prefix"
    assert routing.route_representation("What happened at the anniversary?", ["Ann"]) == "raw"
    assert routing.route_representation("Who spoke?", ["Alice", "Bob"]) == "raw"
    assert routing.mentioned_speaker_names("Did bob, then Alice, agree?", ["Alice", "Bob"]) == (
        "alice",
        "bob",
    )
    parameters = inspect.signature(routing.route_representation).parameters
    assert set(parameters) == {"query", "speaker_names"}
    assert not any("gold" in name or "evidence" in name for name in parameters)


def test_rrf_fusion_is_source_id_exact_and_deterministic():
    ranked, scores = routing.rrf_fuse(
        ["locomo:a:D1:1", "locomo:a:D1:2"],
        ["locomo:a:D1:2", "locomo:a:D1:3"],
    )
    assert ranked[0] == "locomo:a:D1:2"
    assert set(ranked) == {"locomo:a:D1:1", "locomo:a:D1:2", "locomo:a:D1:3"}
    assert scores[0] > scores[1]


def test_routing_input_fingerprint_ignores_answer_and_evidence(tmp_path: Path):
    dataset = _dataset(tmp_path)
    original = json.loads(dataset.read_text(encoding="utf-8"))
    changed = deepcopy(original)
    changed[0]["qa"][0]["answer"] = "a deliberately different answer"
    changed[0]["qa"][0]["evidence"] = ["D1:3"]
    assert routing._retrieval_input_fingerprint(
        original, ["conv-a", "conv-b"]
    ) == routing._retrieval_input_fingerprint(
        changed, ["conv-a", "conv-b"]
    )


def test_run_reports_gold_independent_routing_provenance_and_zero_calls(tmp_path: Path):
    dataset = _dataset(tmp_path)
    result = routing.run(dataset, split_seed="test-split")

    assert result["classification"] == "post_hoc_exploratory_offline_sparse_field_routing"
    assert result["execution"]["strictly_offline"] is True
    assert result["execution"]["external_network_calls"] == 0
    assert result["execution"]["provider_calls"] == 0
    assert result["experimental_status"]["selection_or_promotion_performed"] is False
    assert result["experimental_status"]["requires_unseen_confirmatory_dataset_or_split"] is True

    development_ids = set(result["development"]["sample_ids"])
    held_out_ids = set(result["held_out"]["sample_ids"])
    assert development_ids
    assert held_out_ids
    assert development_ids.isdisjoint(held_out_ids)

    held_out = result["held_out"]
    assert held_out["routing"]["answers_or_gold_used_for_routing"] is False
    assert held_out["routing"]["speaker_mentioned_count"] > 0
    assert held_out["routing"]["speaker_not_mentioned_count"] > 0
    assert set(held_out["conditions"]) == set(routing.CONDITIONS)
    assert held_out["conditions"]["routed"]["selection_eligible"] is False
    assert held_out["conditions"]["rrf_multi_field"]["classification"] == "exploratory_multi_field_rrf"
    assert held_out["conditions"]["rrf_multi_field"]["footprint"]["requires_two_indexes"] is True
    assert isinstance(held_out["failure_rows"], list)
    assert (
        held_out["conditions"]["rrf_multi_field"]["footprint"]["index_tokens_regex_proxy"]
        > held_out["conditions"]["raw"]["footprint"]["index_tokens_regex_proxy"]
    )

    rows = held_out["question_rows"]
    assert rows
    for row in rows:
        expected = "speaker_prefix" if row["speaker_mentioned"] else "raw"
        assert row["routed_field"] == expected
        assert set(row["gold_evidence_ids"])
        for condition in routing.CONDITIONS:
            assert all(
                source_id.startswith("locomo:")
                for source_id in row["conditions"][condition]["ranked_evidence_ids_at_25"]
            )
    assert "true" in held_out["question_level_transitions"]["routed"]["by_speaker_mentioned"]
    assert "false" in held_out["question_level_transitions"]["routed"]["by_speaker_mentioned"]

    provenance = result["provenance"]
    assert len(provenance["dataset"]["sha256"]) == 64
    assert len(provenance["source"]["sha256"]) == 64
    assert len(provenance["config_sha256"]) == 64
    assert len(provenance["manifest_sha256"]) == 64
    assert all(
        len(value) == 64
        for value in provenance["retrieval_input_sha256"].values()
    )


def test_atomic_json_write_is_create_once_and_atomic(tmp_path: Path):
    output = tmp_path / "result.json"
    routing.atomic_write_json(output, {"schema": routing.SCHEMA, "value": 1})
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 1
    assert not list(tmp_path.glob(".result.json.*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        routing.atomic_write_json(output, {"schema": routing.SCHEMA, "value": 2})
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 1


def test_strict_evidence_validation_fails_closed(tmp_path: Path):
    data = [_conversation("conv-a", "Alice", "Bob"), _conversation("conv-b", "Carol", "Dan")]
    data[0]["qa"][0]["evidence"] = ["not-an-evidence-id"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(routing.ValidationError, match="invalid evidence annotation"):
        routing.run(path, strict_evidence=True)


def test_non_strict_failures_are_question_level_ledger_rows(tmp_path: Path):
    data = [_conversation("conv-a", "Alice", "Bob"), _conversation("conv-b", "Carol", "Dan")]
    data[0]["qa"][0]["evidence"] = ["not-an-evidence-id"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = routing.run(path, strict_evidence=False)
    rows = result["development"]["failure_rows"] + result["held_out"]["failure_rows"]
    assert len(rows) == 1
    assert rows[0]["classification"] == "failed_invalid_annotation"
    assert rows[0]["details"] == ["not-an-evidence-id"]


def test_cli_strict_failure_writes_failed_receipt(tmp_path: Path):
    data = [_conversation("conv-a", "Alice", "Bob"), _conversation("conv-b", "Carol", "Dan")]
    data[0]["qa"][0]["evidence"] = ["not-an-evidence-id"]
    dataset = tmp_path / "invalid.json"
    output = tmp_path / "failed.json"
    dataset.write_text(json.dumps(data), encoding="utf-8")

    assert routing.main([
        "--dataset", str(dataset),
        "--output", str(output),
        "--strict-evidence",
    ]) == 2
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == routing.FAILED_SCHEMA
    assert receipt["status"] == "failed"
    assert receipt["execution"]["provider_calls"] == 0
    assert "invalid evidence annotation" in receipt["reason"]
