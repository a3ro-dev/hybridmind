"""Behavioral tests for fail-closed sparse seed aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import aggregate_sparse_seed_results as aggregator


def _fixture_artifact(
    root: Path,
    name: str,
    *,
    split_seed: str = "seed-a",
    source: Path | None = None,
    dataset: Path | None = None,
    schema: str = aggregator.INPUT_SCHEMA,
    external_network_calls: int = 0,
    provider_calls: int | None = None,
    winner: str = "speaker_prefix",
    recall_delta: float = 0.02,
    mrr_delta: float = 0.01,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = source or root / "producer.py"
    dataset = dataset or root / "dataset.json"
    if not source.exists():
        source.write_text("producer-v2\n", encoding="utf-8")
    if not dataset.exists():
        dataset.write_text("[\"dataset\"]\n", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    criteria = {
        "quality_selection_criteria": [
            "maximize development exact-evidence recall@10",
            "then maximize development MRR",
        ],
        "representation_success_criteria": {
            "recall_at_10_absolute_gain_min": 0.01,
            "index_token_increase_max": 0.15,
        },
        "adaptive_success_criteria": {
            "recall_noninferiority_margin": -0.005,
            "source_token_reduction_min": 0.20,
        },
    }
    document = {
        "schema_version": schema,
        "classification": "measured_offline_paired_conversation_split_experiment",
        "execution": {
            "external_network_calls": external_network_calls,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "actual_external_cost_usd": 0.0,
        },
        "provenance": {
            "dataset": {"path": str(dataset), "sha256": dataset_hash},
            "source": {"path": str(source), "sha256": source_hash},
            "git_commit": "a" * 40,
            "python": "3.13.5",
            "platform": "test-platform",
            "logical_cpu_count": 4,
            "dependency_versions": {
                "bm25s": "0.3.9",
                "numpy": "2.4.3",
                "PyStemmer": "3.1.0",
            },
        },
        "information_boundary": {
            "split_seed": split_seed,
            "held_out": ["conversation-a", "conversation-b"],
        },
        "representation_experiment": {
            "selection": {
                "winner": winner,
                "quality_selection_criteria": criteria["quality_selection_criteria"],
            },
            "held_out": {
                "paired_delta_winner_minus_raw": {
                    "exact_evidence_recall_at_10": {
                        "mean": recall_delta,
                        "ci95_low": recall_delta - 0.01,
                        "ci95_high": recall_delta + 0.01,
                        "n": 5,
                    },
                    "mrr": {
                        "mean": mrr_delta,
                        "ci95_low": mrr_delta - 0.01,
                        "ci95_high": mrr_delta + 0.01,
                        "n": 5,
                    },
                },
                "quality_success": recall_delta >= 0.01,
                "predeclared_success": recall_delta >= 0.01,
                "success_criteria": criteria["representation_success_criteria"],
            },
        },
        "adaptive_k_experiment": {
            "predeclared_success": True,
            "success_criteria": criteria["adaptive_success_criteria"],
        },
    }
    if provider_calls is not None:
        document["provider_calls"] = provider_calls
    output = root / name
    output.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return output


def test_valid_v2_artifacts_aggregate_with_dispersion_and_claim_boundary(tmp_path: Path) -> None:
    first = _fixture_artifact(tmp_path, "a.json", split_seed="seed-a", recall_delta=0.02)
    second = _fixture_artifact(tmp_path, "b.json", split_seed="seed-b", recall_delta=-0.01, mrr_delta=-0.02)
    report = aggregator.aggregate([first, second])
    assert report["schema_version"] == aggregator.SCHEMA
    assert report["provider_calls"] == 0
    assert report["artifact_count"] == 2
    assert report["held_out_delta_summary"]["exact_evidence_recall_at_10"]["mean"] == pytest.approx(0.005)
    assert report["counts"]["positive_recall_at_10_delta"] == 1
    assert report["held_out_overlap"]["any_held_out_overlap"] is True
    assert any("not answer accuracy" in text for text in report["claim_boundary"])
    assert all(len(item["sha256"]) == 64 for item in report["artifacts"])


def test_rejects_dataset_hash_mismatch(tmp_path: Path) -> None:
    first = _fixture_artifact(tmp_path / "one", "a.json", split_seed="seed-a")
    second_dataset = tmp_path / "two" / "dataset.json"
    second_dataset.parent.mkdir(parents=True)
    second_dataset.write_text("different-dataset\n", encoding="utf-8")
    second = _fixture_artifact(
        tmp_path / "two", "b.json", split_seed="seed-b", dataset=second_dataset,
    )
    with pytest.raises(aggregator.AggregationValidationError, match="dataset SHA-256"):
        aggregator.aggregate([first, second])


def test_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    first_source = tmp_path / "one" / "producer.py"
    first = _fixture_artifact(tmp_path / "one", "a.json", split_seed="seed-a", source=first_source)
    second_source = tmp_path / "two" / "producer.py"
    second_source.parent.mkdir(parents=True)
    second_source.write_text("different-producer\n", encoding="utf-8")
    second = _fixture_artifact(tmp_path / "two", "b.json", split_seed="seed-b", source=second_source)
    with pytest.raises(aggregator.AggregationValidationError, match="source SHA-256"):
        aggregator.aggregate([first, second])


def test_rejects_duplicate_split_seed(tmp_path: Path) -> None:
    first = _fixture_artifact(tmp_path, "a.json", split_seed="same")
    second = _fixture_artifact(tmp_path, "b.json", split_seed="same")
    with pytest.raises(aggregator.AggregationValidationError, match="split seeds"):
        aggregator.aggregate([first, second])


def test_rejects_provider_calls(tmp_path: Path) -> None:
    first = _fixture_artifact(tmp_path, "a.json", split_seed="seed-a", external_network_calls=1)
    second = _fixture_artifact(tmp_path, "b.json", split_seed="seed-b")
    with pytest.raises(aggregator.AggregationValidationError, match="external_network_calls"):
        aggregator.aggregate([first, second])


def test_rejects_generic_provider_call_counter(tmp_path: Path) -> None:
    first = _fixture_artifact(tmp_path, "a.json", split_seed="seed-a", provider_calls=1)
    second = _fixture_artifact(tmp_path, "b.json", split_seed="seed-b")
    with pytest.raises(aggregator.AggregationValidationError, match="provider_calls"):
        aggregator.aggregate([first, second])


def test_rejects_non_v2_input(tmp_path: Path) -> None:
    first = _fixture_artifact(tmp_path, "a.json", split_seed="seed-a", schema="hybridmind.offline-locomo-sparse-experiment/v1")
    second = _fixture_artifact(tmp_path, "b.json", split_seed="seed-b")
    with pytest.raises(aggregator.AggregationValidationError, match="schema-v2"):
        aggregator.aggregate([first, second])


def test_rejects_mixed_execution_environments(tmp_path: Path) -> None:
    first = _fixture_artifact(tmp_path, "a.json", split_seed="seed-a")
    second = _fixture_artifact(tmp_path, "b.json", split_seed="seed-b")
    document = json.loads(second.read_text(encoding="utf-8"))
    document["provenance"]["dependency_versions"]["bm25s"] = "different"
    second.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        aggregator.AggregationValidationError, match="execution environment"
    ):
        aggregator.aggregate([first, second])


def test_atomic_writer_never_overwrites_an_artifact(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    aggregator.write_json_atomic(output, {"value": 1})

    with pytest.raises(FileExistsError):
        aggregator.write_json_atomic(output, {"value": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
