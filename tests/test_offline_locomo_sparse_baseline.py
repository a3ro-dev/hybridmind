import json

import pytest

from scripts.offline_locomo_sparse_baseline import evaluate


def test_sparse_baseline_uses_exact_scoped_evidence_and_zero_network(tmp_path):
    dataset = tmp_path / "locomo.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conversation-a",
                    "conversation": {
                        "session_1": [
                            {"dia_id": "D1:1", "text": "Alice owns the red bicycle."},
                            {"dia_id": "D1:2", "text": "Bob grows mint."},
                        ]
                    },
                    "qa": [
                        {
                            "question": "Who owns the red bicycle?",
                            "answer": "Alice",
                            "evidence": ["D1:1"],
                            "category": 1,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate(dataset, seed=7)

    assert report["execution"]["external_network_calls"] == 0
    assert report["dataset"]["evidence_questions"] == 1
    assert report["metrics"]["at_k"]["1"]["exact_evidence_recall"]["mean"] == 1.0


def test_sparse_baseline_fails_on_unresolved_gold_evidence(tmp_path):
    dataset = tmp_path / "locomo.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conversation-a",
                    "conversation": {
                        "session_1": [{"dia_id": "D1:1", "text": "One turn."}]
                    },
                    "qa": [
                        {"question": "What?", "answer": "x", "evidence": ["D9:9"]}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved evidence IDs"):
        evaluate(dataset, strict_evidence=True)
