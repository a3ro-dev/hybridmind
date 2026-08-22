"""Behavioral tests for the offline fixed-pool reranking ablation."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts import offline_locomo_rerank_ablation as rerank


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
            {
                "question": f"What does {first} keep?",
                "answer": "blue notebook",
                "evidence": ["D1:1"],
                "category": 1,
            },
            {
                "question": "When is the meeting?",
                "answer": "next Tuesday",
                "evidence": ["D1:3"],
                "category": 2,
            },
        ],
    }


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "locomo-mini.json"
    path.write_text(json.dumps([
        _conversation("conv-a", "Alice", "Bob"),
        _conversation("conv-b", "Carol", "Dan"),
    ]), encoding="utf-8")
    return path


class _FakeScorer:
    """Gold-blind scorer: it receives only query and candidate text."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        # Put the exact lexical target first where it is present.  This is a
        # deterministic stand-in and never receives evidence IDs or gold.
        return [1.0 if "blue notebook" in text.lower() else 0.0 for text in texts]


def test_rerank_is_gold_blind_and_reports_fixed_pool_metrics(tmp_path: Path):
    scorer = _FakeScorer()
    result = rerank.evaluate(_dataset(tmp_path), scorer=scorer, pool_size=10, top_k=10)

    assert result["classification"] == "exploratory_offline_fixed_pool_cross_encoder_ablation"
    assert result["experiment"]["selection_or_promotion_performed"] is False
    assert result["experiment"]["pool_size"] == 10
    assert result["experiment"]["final_top_k"] == 10
    assert result["experiment"]["reranker_input"].startswith("authoritative raw")
    assert result["split"]["disjoint"] is True
    assert result["execution"]["provider_calls"] == 0
    assert result["execution"]["external_network_calls"] == 0
    assert result["timing"]["rerank_calls"] == len(scorer.calls)
    assert scorer.calls
    assert all(len(texts) == 4 for _, texts in scorer.calls)
    assert all(not any("locomo:" in text for text in texts) for _, texts in scorer.calls)
    assert all(": " not in text for _, texts in scorer.calls for text in texts)
    for offset in range(0, len(scorer.calls), 2):
        assert sorted(scorer.calls[offset][1]) == sorted(scorer.calls[offset + 1][1])

    rows = result["question_rows"]
    assert rows and all(row["status"] == "ok" for row in rows)
    for row in rows:
        assert row["gold_evidence_ids"]
        for condition in rerank.CONDITIONS:
            details = row["conditions"][condition]
            assert len(details["candidate_evidence_ids"]) <= 10
            assert len(details["pre_ranked_evidence_ids_at_10"]) <= 10
            assert len(details["post_ranked_evidence_ids_at_10"]) <= 10
            assert all(value.startswith("locomo:") for value in details["candidate_evidence_ids"])
            assert 0.0 <= details["oracle_recall_at_pool"] <= 1.0
    assert "candidate_pool_oracle_recall" in result["summaries"]["held_out"]["raw"]
    assert "paired_delta_recall_at_10" in result["summaries"]["held_out"]["speaker_prefix"]


def test_rerank_sorting_is_deterministic_and_ties_keep_candidate_order():
    class TieScorer:
        def score(self, _query: str, texts: list[str]) -> list[float]:
            return [0.5] * len(texts)

    ids = ["locomo:c:D1:2", "locomo:c:D1:1", "locomo:c:D1:3"]
    ranked, scores, _ = rerank._rerank(
        TieScorer(), "query", ids,
        {evidence_id: "candidate" for evidence_id in ids},
    )
    assert ranked == ids
    assert scores == [0.5, 0.5, 0.5]


def test_candidate_pool_ceiling_and_top_k_validation():
    with pytest.raises(rerank.ValidationError, match="at least final top_k"):
        rerank._validate_pool(9, 10)
    with pytest.raises(rerank.ValidationError, match="fixed at 10"):
        rerank._validate_pool(25, 5)


def test_failure_ledger_preserves_invalid_question_without_skipping(tmp_path: Path):
    path = _dataset(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data[0]["qa"][0]["evidence"] = ["not-an-evidence-id"]
    path.write_text(json.dumps(data), encoding="utf-8")
    result = rerank.evaluate(path, scorer=_FakeScorer(), pool_size=10, top_k=10)
    failures = result["failure_ledger"]
    assert len(failures) == 1
    assert failures[0]["status"] == "failed"
    assert failures[0]["failure"]["classification"] == "failed_invalid_annotation"
    assert len(result["question_rows"]) == 4
    assert result["execution"]["provider_calls"] == 0


def test_atomic_artifact_is_create_once(tmp_path: Path):
    output = tmp_path / "result.json"
    rerank.atomic_write_json(output, {"schema": rerank.SCHEMA, "value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rerank.atomic_write_json(output, {"schema": rerank.SCHEMA, "value": 2})
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 1
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_local_loader_is_explicitly_local_only():
    source = inspect.getsource(rerank.LocalMiniLMScorer)
    assert "local_files_only=True" in source
    assert "from_pretrained" in source


def test_cluster_bootstrap_resamples_whole_conversations():
    rows = [
        {"sample_id": "a", "value": 0.0},
        {"sample_id": "a", "value": 0.0},
        {"sample_id": "b", "value": 1.0},
    ]
    result = rerank._cluster_bootstrap(
        rows, lambda row: float(row["value"]), seed=7, samples=100,
    )
    assert result["mean"] == pytest.approx(1 / 3)
    assert result["n"] == 3
    assert result["n_clusters"] == 2
    assert result["bootstrap_unit"] == "whole conversation"


def test_rankings_ignore_answer_and_gold_annotation(tmp_path: Path):
    original = _dataset(tmp_path)
    changed = tmp_path / "locomo-changed.json"
    data = json.loads(original.read_text(encoding="utf-8"))
    data[0]["qa"][0]["answer"] = "changed answer"
    data[0]["qa"][0]["evidence"] = ["D1:2"]
    changed.write_text(json.dumps(data), encoding="utf-8")

    first = rerank.evaluate(original, scorer=_FakeScorer(), pool_size=10, top_k=10)
    second = rerank.evaluate(changed, scorer=_FakeScorer(), pool_size=10, top_k=10)
    first_row = next(row for row in first["question_rows"] if row["sample_id"] == "conv-a")
    second_row = next(row for row in second["question_rows"] if row["sample_id"] == "conv-a")
    for condition in rerank.CONDITIONS:
        assert first_row["conditions"][condition]["candidate_evidence_ids"] == (
            second_row["conditions"][condition]["candidate_evidence_ids"]
        )
        assert first_row["conditions"][condition]["post_ranked_evidence_ids_at_10"] == (
            second_row["conditions"][condition]["post_ranked_evidence_ids_at_10"]
        )
