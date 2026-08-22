"""Behavioral tests for the offline BGE-M3 mechanism harness.

All tests use a deterministic fake encoder; no model is loaded and no network
or provider call is possible.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import offline_locomo_bgem3_mechanisms as bgem3


def _conversation(sample_id: str, first: str, second: str) -> dict:
    return {
        "sample_id": sample_id,
        "conversation": {
            "session_1": [
                {"dia_id": "D1:1", "speaker": first, "text": f"{first} keeps the blue notebook."},
                {"dia_id": "D1:2", "speaker": second, "text": f"{second} keeps the red bicycle."},
                {"dia_id": "D1:3", "speaker": first, "text": "The meeting is next Tuesday."},
                {"dia_id": "D1:4", "speaker": second, "text": "The train leaves after lunch."},
            ],
            "session_1_date_time": "2024-01-01 10:00:00",
        },
        "qa": [
            {"question": f"What does {first} keep?", "answer": "blue notebook", "evidence": ["D1:1"], "category": 1},
            {"question": "When is the meeting?", "answer": "next Tuesday", "evidence": ["D1:3"], "category": 2},
        ],
    }


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([_conversation("conv-a", "Alice", "Bob"), _conversation("conv-b", "Carol", "Dan")]), encoding="utf-8")
    return path


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: list[str]) -> list[dict]:
        self.calls.append(tuple(texts))
        result = []
        for text in texts:
            tokens = [token.lower().strip("?.!,:") for token in text.split() if token.strip("?.!,:")]
            sparse = {token: 1.0 for token in tokens}
            vectors = np.asarray([[1.0, 0.0] if token in {"alice", "carol", "blue", "notebook"} else [0.0, 1.0] for token in tokens], dtype=np.float32)
            result.append({"sparse": sparse, "colbert": vectors})
        return result


def test_maxsim_and_sparse_math_are_explicit():
    query = {"sparse": {"a": 2.0}, "colbert": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    document = {"sparse": {"a": 3.0}, "colbert": np.asarray([[1.0, 0.0], [0.2, 0.8]], dtype=np.float32)}
    assert bgem3.learned_sparse_score(query, document) == pytest.approx(6.0)
    assert bgem3.maxsim_score(query, document) == pytest.approx(0.9)


def test_dense_dimension_is_rejected_and_never_a_production_signal():
    with pytest.raises(bgem3.ValidationError, match="requires native 4096"):
        bgem3._reject_dense_output(np.zeros((1, 1024), dtype=np.float32))


def test_evaluation_is_gold_blind_exact_id_and_equal_pool(tmp_path: Path):
    dataset = _dataset(tmp_path)
    encoder = _FakeEncoder()
    result = bgem3.evaluate(dataset, encoder=encoder)
    assert result["execution"]["provider_calls"] == 0
    assert result["execution"]["external_network_calls"] == 0
    assert result["experiment"]["dense_output_used"] is False
    assert result["experiment"]["dense_output_dimension_excluded"] == 1024
    assert result["experiment"]["production_embedding_contract_dimension"] == 4096
    assert result["split"]["disjoint"] is True
    assert result["question_rows"] and all(row["status"] == "ok" for row in result["question_rows"])
    for row in result["question_rows"]:
        assert row["gold_evidence_ids"] and row["question_id"].startswith("locomo:")
        for condition in bgem3.CONDITIONS:
            details = row["conditions"][condition]
            assert len(details["candidate_pool_evidence_ids"]) == 4
            assert len(details["pre"]["evidence_ids"]) == 4
            assert len(details["post"]["evidence_ids"]) == 4
            assert all(value.startswith("locomo:") for value in details["candidate_pool_evidence_ids"])
    # The same speaker-prefixed authoritative representation is used for BGE
    # sparse retrieval and MaxSim; canonical IDs never enter the encoder.
    assert all(not "locomo:" in text for batch in encoder.calls for text in batch)
    assert any("Alice: " in text or "Bob: " in text for batch in encoder.calls for text in batch)
    # One document batch and one query batch per conversation, rather than one
    # expensive encoder invocation per question.
    assert len(encoder.calls) == 4
    assert result["storage"]["learned_sparse_postings"] > 0
    assert result["storage"]["colbert_token_vector_bytes"] > 0
    assert "candidate_pool_oracle_recall" in result["summaries"]["held_out"]["bgem3_learned_sparse"]
    assert "paired_delta_recall_at_10" in result["summaries"]["held_out"]["bm25s_speaker_prefix"]
    assert "pre_recall_at_10" in result["comparisons"]["held_out"]["paired_deltas"]
    assert result["execution"]["local_model_encode_batches"] == 4


def test_rankings_are_invariant_to_gold_annotation_changes(tmp_path: Path):
    original = _dataset(tmp_path)
    changed = tmp_path / "changed.json"
    value = json.loads(original.read_text(encoding="utf-8"))
    value[0]["qa"][0]["answer"] = "unrelated"
    value[0]["qa"][0]["evidence"] = ["D1:2"]
    changed.write_text(json.dumps(value), encoding="utf-8")
    first = bgem3.evaluate(original, encoder=_FakeEncoder())
    second = bgem3.evaluate(changed, encoder=_FakeEncoder())
    for left, right in zip(first["question_rows"], second["question_rows"]):
        for condition in bgem3.CONDITIONS:
            assert left["conditions"][condition]["candidate_pool_evidence_ids"] == right["conditions"][condition]["candidate_pool_evidence_ids"]
            assert left["conditions"][condition]["post"]["evidence_ids"] == right["conditions"][condition]["post"]["evidence_ids"]


def test_failure_ledger_preserves_invalid_questions(tmp_path: Path):
    path = _dataset(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value[0]["qa"][0]["evidence"] = ["malformed"]
    path.write_text(json.dumps(value), encoding="utf-8")
    result = bgem3.evaluate(path, encoder=_FakeEncoder())
    assert len(result["failure_ledger"]) == 1
    assert result["failure_ledger"][0]["status"] == "failed"
    assert len(result["question_rows"]) == 4


def test_snapshot_resolution_fails_closed_for_ambiguous_or_incomplete_cache(tmp_path: Path, monkeypatch):
    root = tmp_path / "snapshots"
    (root / "one").mkdir(parents=True)
    (root / "two").mkdir(parents=True)
    monkeypatch.setattr(bgem3, "DEFAULT_MODEL_ROOT", root)
    with pytest.raises(bgem3.ValidationError, match="pass --model-path"):
        bgem3.resolve_model_snapshot()
    with pytest.raises(bgem3.ValidationError, match="incomplete"):
        bgem3.resolve_model_snapshot(root / "one")


def test_complete_snapshot_is_explicitly_accepted_and_manifestable(tmp_path: Path):
    snapshot = tmp_path / "complete"
    snapshot.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "sparse_linear.pt", "colbert_linear.pt", "model.safetensors"):
        (snapshot / name).write_bytes(name.encode())
    (snapshot / "README.md").write_text("license: mit\n", encoding="utf-8")
    resolved = bgem3.resolve_model_snapshot(snapshot)
    manifest = bgem3._snapshot_manifest(resolved)
    assert resolved == snapshot.resolve()
    assert manifest["manifest_sha256"]
    assert all(item["sha256"] for item in manifest["files"])


def test_offline_loader_is_local_and_dense_disabled():
    source = inspect.getsource(bgem3)
    assert "HF_HUB_OFFLINE" in source
    assert "return_dense=False" in source
    assert "return_sparse=True" in source
    assert "return_colbert_vecs=True" in source
    assert "str(self.model_path)" in source


def test_atomic_artifact_is_create_once(tmp_path: Path):
    path = tmp_path / "result.json"
    bgem3.atomic_write_json(path, {"schema": bgem3.SCHEMA, "value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bgem3.atomic_write_json(path, {"schema": bgem3.SCHEMA, "value": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 1
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_cluster_bootstrap_is_conversation_clustered():
    rows = [{"sample_id": "a", "value": 0.0}, {"sample_id": "a", "value": 0.0}, {"sample_id": "b", "value": 1.0}]
    report = bgem3._cluster_bootstrap(rows, lambda row: row["value"], seed=7, samples=100)
    assert report["mean"] == pytest.approx(1 / 3)
    assert report["n_clusters"] == 2
    assert report["bootstrap_unit"] == "whole conversation"
