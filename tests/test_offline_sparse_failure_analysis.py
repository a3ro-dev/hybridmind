from __future__ import annotations

import copy
import json

from scripts.offline_sparse_failure_analysis import (
    _paired_failure_analysis,
    _question_id,
    _retrieval_input_fingerprint,
    run,
)


def _dataset() -> list[dict]:
    return [
        {
            "sample_id": f"sample-{index}",
            "conversation": {
                "session_1": [
                    {"dia_id": "D1:1", "speaker": "Avery", "text": "alpha shared"},
                    {"dia_id": "D1:2", "speaker": "Blair", "text": "beta distinct"},
                ],
                "session_1_date_time": "2024-01-01",
            },
            "qa": [
                {
                    "question": "What did Avery say about alpha?",
                    "answer": "alpha shared",
                    "category": 1,
                    "evidence": ["D1:1"],
                },
                {
                    "question": "What did Blair say?",
                    "answer": "beta distinct",
                    "category": 2,
                    "evidence": ["D1:2"],
                },
            ],
        }
        for index in range(2)
    ]


def _row(question_id: str, gold: list[str], ranked: list[str], recall: float, mrr: float) -> dict:
    return {
        "question_id": question_id,
        "gold": gold,
        "ranked_ids_at_25": ranked,
        "recall_at_k": {"10": recall},
        "any_hit_at_k": {"10": float(recall > 0)},
        "all_hit_at_k": {"10": float(recall == 1)},
        "reciprocal_rank": mrr,
    }


def _metadata(question_id: str, category: str, gold: list[str]) -> dict:
    records = {
        "g1": {"text": "alpha shared", "speaker": "Avery"},
        "g2": {"text": "beta distinct", "speaker": "Blair"},
    }
    return {
        question_id: {
            "question": "What did Avery say about alpha?",
            "category": category,
            "sample_id": "sample-0",
            "gold": gold,
            "records": records,
        }
    }


def test_gold_annotations_do_not_change_retrieval_input_fingerprint():
    first = _dataset()
    second = copy.deepcopy(first)
    second[0]["qa"][0]["evidence"] = [["D1:2"]]
    assert _retrieval_input_fingerprint(first, {"sample-0"}, "raw") == _retrieval_input_fingerprint(
        second, {"sample-0"}, "raw"
    )
    assert _retrieval_input_fingerprint(first, {"sample-0"}, "speaker_prefix") == _retrieval_input_fingerprint(
        second, {"sample-0"}, "speaker_prefix"
    )


def test_pairing_scope_and_failure_transitions_are_exact():
    raw = [
        _row("q1", ["g1"], ["g1", "d"], 1.0, 1.0),
        _row("q2", ["g2"], ["d", "g2"], 1.0, 0.5),
    ]
    speaker = [
        _row("q1", ["g1"], ["d", "g1"], 1.0, 0.5),
        _row("q2", ["g2"], ["g2", "d"], 1.0, 1.0),
    ]
    metadata = {
        **_metadata("q1", "single-hop", ["g1"]),
        **_metadata("q2", "temporal", ["g2"]),
    }
    result = _paired_failure_analysis(raw, speaker, metadata, representative_limit=1)
    assert result["n_questions"] == 2
    assert [row["status"] for row in result["paired_changes"]] == ["unchanged", "unchanged"]
    assert result["catastrophic_miss_transitions_at_10"] == {"raw_hit_speaker_hit_at_10": 2}
    assert result["evidence_rank_transitions_within_top_25"] == {
        "improved_rank": 1,
        "regressed_rank": 1,
    }
    assert result["category_summary"]["single-hop"]["n"] == 1
    assert len(result["representatives"]["unchanged"]) == 1


def test_run_is_offline_scoped_and_writes_versioned_report(tmp_path):
    dataset = tmp_path / "locomo.json"
    dataset.write_text(json.dumps(_dataset()), encoding="utf-8")
    output = tmp_path / "failure-analysis.json"
    report = run(dataset, all_conversations=True, representative_limit=0)
    assert report["schema_version"].endswith("/v1")
    assert report["provider_calls"] == 0
    assert report["scope"]["selection"] == "all_declared_conversations"
    assert report["scope"]["selected"] == ["sample-0", "sample-1"]
    assert report["analysis"]["n_questions"] == 4
    assert report["information_boundary"]["retrieval_inputs_exclude_answer_and_evidence"] is True
    assert report["analysis"]["representatives"] == {
        "improved": [],
        "regressed": [],
        "unchanged": [],
    }

    from scripts.offline_sparse_failure_analysis import _atomic_write
    _atomic_write(output, report)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == report["schema_version"]
    assert output.exists()


def test_pairing_rejects_missing_or_mismatched_questions():
    row = _row("q1", ["g1"], ["g1"], 1.0, 1.0)
    metadata = _metadata("q1", "single-hop", ["g1"])
    try:
        _paired_failure_analysis([row], [], metadata)
    except ValueError as error:
        assert "identical ordered pair" in str(error)
    else:
        raise AssertionError("incomplete paired rows must fail closed")

    bad = _row("q1", ["g2"], ["g2"], 1.0, 1.0)
    try:
        _paired_failure_analysis([row], [bad], metadata)
    except ValueError as error:
        assert "gold IDs disagree" in str(error)
    else:
        raise AssertionError("mismatched question IDs must fail closed")
