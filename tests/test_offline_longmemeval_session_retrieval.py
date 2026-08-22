"""Behavioral contracts for the offline LongMemEval-S session benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.offline_longmemeval_session_retrieval import (
    CONDITIONS,
    DatasetNotRetrievalEvaluableError,
    _atomic_write_json,
    aggregate_turn_scores,
    build_documents,
    evaluate,
    metrics_for_ranking,
    require_retrieval_challenge,
    validate_dataset,
)


def _example(*, answer: str = "private answer", question_id: str = "q-1") -> dict:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "Where is the red bike?",
        # Retrieval must not inspect this field.  It intentionally does not
        # appear in the retrieval-only copy returned by validate_dataset.
        "answer": answer,
        "question_date": "2023/01/03",
        "haystack_dates": [
            "2023/01/01 (Sun) 09:00",
            "2023/01/02 (Mon) 09:00",
            "2023/01/03 (Tue) 09:00",
        ],
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "The red bike is in the garage.", "has_answer": True},
                {"role": "assistant", "content": "The garage is behind the house."},
            ],
            [
                {"role": "user", "content": "The blue car is in the driveway.", "has_answer": False},
                {"role": "assistant", "content": "The driveway is on the north side."},
            ],
            [
                {"role": "user", "content": "The green tent is in storage.", "has_answer": False},
            ],
        ],
        "answer_session_ids": ["s1"],
    }


def _write_dataset(tmp_path: Path, data: list[dict] | None = None) -> Path:
    path = tmp_path / "longmemeval_s.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data or [_example()]), encoding="utf-8")
    return path


def test_gold_blind_documents_and_rankings_ignore_answer_text(tmp_path: Path) -> None:
    source = [_example(answer="answer-alpha")]
    changed = copy.deepcopy(source)
    changed[0]["answer"] = "a completely different secret answer"

    validated = validate_dataset(source)
    assert "answer" not in validated[0]
    docs = build_documents(validated[0], "whole_session")
    assert all("has_answer" not in document.text for document in docs)
    assert all("answer-alpha" not in document.text for document in docs)

    first_path = _write_dataset(tmp_path / "first", source)
    second_path = _write_dataset(tmp_path / "second", changed)
    first = evaluate(first_path, mechanics_test_only=True)
    second = evaluate(second_path, mechanics_test_only=True)
    first_rows = [
        (row["condition"], row["ranked_session_ids"])
        for row in first["question_level_rows"]
    ]
    second_rows = [
        (row["condition"], row["ranked_session_ids"])
        for row in second["question_level_rows"]
    ]
    assert first_rows == second_rows
    assert first["dataset"]["sha256"] != second["dataset"]["sha256"]
    assert first["execution"]["provider_calls"] == 0
    assert first["execution"]["network_calls"] == 0


def test_gold_marker_field_and_rendered_marker_never_enter_documents() -> None:
    item = _example()
    item["haystack_sessions"][0][0]["content"] = (
        "has_answer=true The red bike is in the garage."
    )
    validated = validate_dataset([item])
    texts = [document.text for document in build_documents(validated[0], "turn_max")]
    assert all("has_answer" not in text for text in texts)
    assert any("red bike" in text for text in texts)


def test_turn_aggregation_is_gold_independent_and_deterministic() -> None:
    ranked_turns = [
        ("s1::turn:0", 0.90),
        ("s2::turn:0", 0.80),
        ("s1::turn:1", 0.70),
    ]
    mapping = {turn_id: turn_id.split("::", 1)[0] for turn_id, _ in ranked_turns}
    assert aggregate_turn_scores(
        ranked_turns, mapping, method="max"
    ) == [("s1", 0.90), ("s2", 0.80)]
    rrf = aggregate_turn_scores(ranked_turns, mapping, method="rrf", rrf_k=60)
    assert [session_id for session_id, _ in rrf] == ["s1", "s2"]
    assert rrf[0][1] == pytest.approx(1 / 61 + 1 / 63)
    with pytest.raises(ValueError, match="absent from turn mapping"):
        aggregate_turn_scores([("unknown", 1.0)], mapping, method="max")
    with pytest.raises(ValueError, match="duplicate turn ID"):
        aggregate_turn_scores(
            [("s1::turn:0", 1.0), ("s1::turn:0", 0.5)], mapping, method="max"
        )


def test_support_session_metric_math_is_exact() -> None:
    metrics = metrics_for_ranking(
        ["s2", "s1", "s3"], ["s1", "s3"], top_k=10
    )
    assert metrics["support_session_recall_at_k"] == {"1": 0.0, "5": 1.0, "10": 1.0}
    assert metrics["hit_at_k"] == {"1": 0.0, "5": 1.0, "10": 1.0}
    assert metrics["all_gold_at_k"] == {"1": 0.0, "5": 1.0, "10": 1.0}
    assert metrics["first_gold_rank_at_10"] == 2
    assert metrics["mrr_at_10"] == pytest.approx(0.5)


def test_validation_fails_closed_on_malformed_corpus_gold_and_ids() -> None:
    duplicate_ids = [_example(), _example(question_id="q-1")]
    with pytest.raises(ValueError, match="question IDs must be unique"):
        validate_dataset(duplicate_ids)

    missing_gold = _example()
    missing_gold["answer_session_ids"] = ["missing"]
    with pytest.raises(ValueError, match="absent from its haystack"):
        validate_dataset([missing_gold])

    mismatched = _example()
    mismatched["haystack_sessions"] = mismatched["haystack_sessions"][:-1]
    with pytest.raises(ValueError, match="mismatched session IDs"):
        validate_dataset([mismatched])

    bad_turn = _example()
    bad_turn["haystack_sessions"][0][0]["role"] = "system"
    with pytest.raises(ValueError, match="invalid role"):
        validate_dataset([bad_turn])


def test_oracle_context_subset_is_rejected_as_not_retrieval_evaluable() -> None:
    item = _example()
    item["answer_session_ids"] = list(item["haystack_session_ids"])
    examples = validate_dataset([item])

    with pytest.raises(
        DatasetNotRetrievalEvaluableError,
        match="there are no distractors",
    ) as captured:
        require_retrieval_challenge(examples, top_k=10)

    assert captured.value.audit["total_non_gold_sessions"] == 0
    assert captured.value.audit["examples_with_more_than_top_k_sessions"] == 0


def test_report_has_provenance_footprints_strata_rows_and_no_winner(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    raw_bytes = dataset.read_bytes()
    report = evaluate(dataset, mechanics_test_only=True)

    assert dataset.read_bytes() == raw_bytes
    assert report["dataset"]["sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert report["comparison"] == {
        "label": "exploratory",
        "winner_selected": False,
        "selection_dataset": None,
        "conditions": list(CONDITIONS),
    }
    assert report["configuration"]["mechanics_test_only"] is True
    assert set(report["conditions"]) == set(CONDITIONS)
    assert len(report["question_level_rows"]) == len(CONDITIONS)
    assert len(report["transitions"]) == 2
    assert report["conditions"]["whole_session"]["by_question_type"][
        "single-session-user"
    ]["n_questions"] == 1
    for condition in CONDITIONS:
        footprint = report["conditions"][condition]["footprint"]
        assert footprint["document_count"] > 0
        assert footprint["document_token_count"] > 0
        assert footprint["estimated_index_input_bytes"] > 0
    assert report["provenance"]["dataset_sha256"] == report["dataset"]["sha256"]
    assert report["execution"] == {
        "offline": True,
        "provider_calls": 0,
        "network_calls": 0,
        "external_network_calls": 0,
        "embedding_calls": 0,
        "reranker_calls": 0,
        "reader_calls": 0,
        "actual_external_cost_usd": 0.0,
    }
    assert "exact-turn evidence recall" in report["claim_boundary"]["not_measured"]


def test_atomic_writer_is_create_once_and_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    _atomic_write_json(target, {"value": 1})
    original = target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        _atomic_write_json(target, {"value": 2})
    assert target.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".result.json.*.tmp"))
