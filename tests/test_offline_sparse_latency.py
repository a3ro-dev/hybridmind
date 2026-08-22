from __future__ import annotations

import json

import pytest

from scripts.offline_sparse_latency import (
    build_timing_inputs,
    make_interleaved_schedule,
    paired_deltas,
    run,
    select_conversations,
)


def _dataset() -> list[dict]:
    return [
        {
            "sample_id": f"sample-{index}",
            "conversation": {
                "session_1": [
                    {"dia_id": f"D{index}A", "speaker": "A", "text": f"alpha {index}"},
                    {"dia_id": f"D{index}B", "speaker": "B", "text": f"beta {index}"},
                ],
                "session_1_date_time": "2024-01-01",
            },
            "qa": [
                {"question": f"What is alpha {index}?", "answer": "gold answer", "evidence": [["D0A"]]},
                {"question": f"What is beta {index}?", "answer": "gold answer", "evidence": [["D0B"]]},
            ],
        }
        for index in range(4)
    ]


def test_selection_and_timing_inputs_are_deterministic_and_gold_free() -> None:
    data = _dataset()
    first = select_conversations(data, seed=7, max_conversations=3)
    second = select_conversations(data, seed=7, max_conversations=3)
    assert first == second
    records, questions = build_timing_inputs(data, sample_ids=first, seed=7, max_queries=4)
    assert records
    assert questions == build_timing_inputs(data, sample_ids=first, seed=7, max_queries=4)[1]
    assert all(set(row) == {"question_id", "sample_id", "question", "qa_index"} for row in questions)
    assert all("gold" not in row and "evidence" not in row for row in questions)


def test_schedule_is_paired_interleaved_and_seeded() -> None:
    ids = ["q1", "q2", "q3"]
    schedule = make_interleaved_schedule(ids, blocks=2, repetitions=2, seed=11)
    assert len(schedule) == 2 * 2 * len(ids) * 2
    for block in range(2):
        for repetition in range(2):
            rows = [row for row in schedule if row["block"] == block and row["repetition"] == repetition]
            assert {row["question_id"] for row in rows} == set(ids)
            assert all(sum(row["question_id"] == question_id for row in rows) == 2 for question_id in ids)
    assert schedule != make_interleaved_schedule(ids, blocks=2, repetitions=2, seed=12)
    assert schedule == make_interleaved_schedule(ids, blocks=2, repetitions=2, seed=11)


def test_paired_delta_requires_complete_pairs() -> None:
    rows = [
        {"block": 0, "repetition": 0, "question_id": "q", "sample_id": "s", "condition": "raw", "wall_ms": 2.0},
        {"block": 0, "repetition": 0, "question_id": "q", "sample_id": "s", "condition": "speaker_prefix", "wall_ms": 3.5},
    ]
    result = paired_deltas(rows, seed=1, bootstrap_samples=100)
    assert result["rows"][0]["delta_ms"] == pytest.approx(1.5)
    assert result["summary"]["n"] == 1
    with pytest.raises(ValueError):
        paired_deltas(rows[:1], seed=1, bootstrap_samples=100)


def test_run_is_offline_and_quality_separate(tmp_path) -> None:
    dataset = tmp_path / "locomo.json"
    dataset.write_text(json.dumps(_dataset()), encoding="utf-8")
    result = run(
        dataset, seed=4, max_conversations=2, max_queries=4, blocks=1,
        repetitions=1, warmups=1, cold_builds=1, bootstrap_samples=100, top_k=5,
    )
    assert result["provider_calls"] == 0
    assert result["execution"]["quality_evaluation_performed"] is False
    assert result["selection"]["selection_is_gold_free"] is True
    assert set(result["by_condition"]) == {"raw", "speaker_prefix"}
    assert result["paired_deltas"]["wall_ms"]["summary"]["n"] == 4
    build_ci = result["by_condition"]["raw"]["cold_build_wall_ms"][
        "mean_cluster_bootstrap_ci95"
    ]
    assert build_ci["clusters"] == 1
    assert build_ci["ci95_low"] is None
    assert "warning" in build_ci
    assert set(result["empirical_timer_diagnostics"]) == {"wall_ms", "process_ms"}
