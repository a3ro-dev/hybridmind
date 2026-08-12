import hashlib
import json
import sqlite3
from pathlib import Path

from benchmarks.kv_reduction_eval import (
    RegexTokenCounter,
    evaluate_frontier,
    kv_bytes_per_token,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    question = "What project did Ada start?"
    question_id = hashlib.sha1(question.encode()).hexdigest()[:12]
    dataset = [
        {
            "sample_id": "conversation-1",
            "conversation": {
                "speaker_a": "Ada",
                "speaker_b": "Ben",
                "session_1_date_time": "1 January 2026",
                "session_1": [
                    {
                        "speaker": "Ada",
                        "dia_id": "D1:1",
                        "text": "I started the Atlas project.",
                    },
                    {
                        "speaker": "Ben",
                        "dia_id": "D1:2",
                        "text": "That sounds useful.",
                    },
                ],
            },
            "qa": [
                {
                    "question": question,
                    "answer": "Atlas",
                    "evidence": ["D1:1"],
                    "category": 1,
                }
            ],
        }
    ]
    dataset_path = tmp_path / "locomo.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    database_path = tmp_path / "store.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, text TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO nodes (id, text) VALUES (?, ?)",
            [
                ("relevant", "[DATE: 1 January 2026] [SPEAKER: Ada] I started the Atlas project."),
                ("irrelevant", "Ben discussed an unrelated weather forecast."),
            ],
        )

    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps(
            {
                "question_id": question_id,
                "gold_rank_post_rerank": 1,
                "retrieved_ids_at_k": {
                    "1": ["relevant"],
                    "2": ["relevant", "irrelevant"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ledger_path, dataset_path, database_path


def test_evaluate_frontier_joins_ledger_source_and_store(tmp_path: Path):
    ledger_path, dataset_path, database_path = _write_fixture(tmp_path)

    result = evaluate_frontier(
        ledger_path=ledger_path,
        dataset_path=dataset_path,
        database_path=database_path,
        token_counter=RegexTokenCounter(),
        k_values=(1, 2),
        hypothesis_k=1,
        min_context_reduction=0.0,
        min_answer_proxy_hit=1.0,
        bootstrap_resamples=100,
        absolute_kv_bytes_per_token=1024.0,
    )

    assert result["coverage"]["matched_records"] == 1
    assert result["coverage"]["retrieved_node_reference_resolution_rate"] == 1.0
    assert result["frontier"]["1"]["answer_overlap_proxy_hit_with_gold_evidence"]["mean"] == 1.0
    assert result["frontier"]["1"]["exact_source_recall"]["mean"] == 1.0
    assert result["frontier"]["1"]["estimated_kv_bytes"]["bytes_per_token"] == 1024.0
    assert result["hypothesis"]["passed"] is True


def test_missing_retrieved_node_invalidates_data_gate(tmp_path: Path):
    ledger_path, dataset_path, database_path = _write_fixture(tmp_path)
    record = json.loads(ledger_path.read_text(encoding="utf-8"))
    record["retrieved_ids_at_k"]["1"] = ["missing"]
    ledger_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = evaluate_frontier(
        ledger_path=ledger_path,
        dataset_path=dataset_path,
        database_path=database_path,
        token_counter=RegexTokenCounter(),
        k_values=(1,),
        hypothesis_k=1,
        min_context_reduction=0.0,
        min_answer_proxy_hit=0.0,
        min_node_resolution=1.0,
        bootstrap_resamples=20,
    )

    assert result["coverage"]["retrieved_node_reference_resolution_rate"] == 0.0
    assert result["hypothesis"]["data_valid"] is False
    assert result["hypothesis"]["passed"] is False


def test_memorybench_checkpoint_is_self_contained(tmp_path: Path):
    _, dataset_path, _ = _write_fixture(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "questions": {
                    "conversation-1-q0": {
                        "groundTruth": "Atlas",
                        "phases": {
                            "search": {
                                "status": "completed",
                                "durationMs": 12.5,
                                "results": [
                                    {
                                        "node_id": "chronological-first",
                                        "text": "An unrelated item persisted first.",
                                        "combined_score": 0.1,
                                    },
                                    {
                                        "node_id": "historical-node",
                                        "text": (
                                            "[DATE: 1 January 2026] [SPEAKER: Ada] "
                                            "I started the Atlas project."
                                        ),
                                        "combined_score": 0.9,
                                    }
                                ],
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_frontier(
        ledger_path=None,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        database_path=None,
        token_counter=RegexTokenCounter(),
        k_values=(1,),
        hypothesis_k=1,
        min_context_reduction=0.0,
        min_answer_proxy_hit=1.0,
        bootstrap_resamples=20,
    )

    assert result["coverage"]["retrieved_node_reference_resolution_rate"] == 1.0
    assert result["search_duration_ms"]["mean"] == 12.5
    assert result["frontier"]["1"]["exact_source_recall"]["mean"] == 1.0
    assert result["hypothesis"]["passed"] is True


def test_checkpoint_lexical_ranking_reports_paired_hypothesis(tmp_path: Path):
    _, dataset_path, _ = _write_fixture(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "questions": {
                    "conversation-1-q0": {
                        "question": "What Atlas project did Ada start?",
                        "groundTruth": "Atlas",
                        "phases": {
                            "search": {
                                "status": "completed",
                                "durationMs": 12.5,
                                "results": [
                                    {
                                        "node_id": "baseline-first",
                                        "text": "Ada discussed a project update.",
                                        "combined_score": 0.9,
                                    },
                                    {
                                        "node_id": "evidence",
                                        "text": "I started the Atlas project.",
                                        "combined_score": 0.1,
                                    },
                                ],
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_frontier(
        ledger_path=None,
        checkpoint_path=checkpoint_path,
        checkpoint_ranking="local-lexical-rrf",
        local_lexical_pool_size=2,
        local_lexical_weight=1.0,
        dataset_path=dataset_path,
        database_path=None,
        token_counter=RegexTokenCounter(),
        k_values=(1,),
        hypothesis_k=1,
        min_context_reduction=0.0,
        min_answer_proxy_hit=0.0,
        min_exact_source_recall_improvement=0.5,
        bootstrap_resamples=20,
    )

    ranking = result["ranking_hypothesis"]
    assert ranking["baseline_mean"] == 0.0
    assert ranking["variant_mean"] == 1.0
    assert ranking["improvement"]["mean"] == 1.0
    assert ranking["passed"] is True
    assert ranking["confirmatory"] is False
    assert result["offline_rerank_duration_ms"]["n"] == 1


def test_kv_bytes_per_token_formula():
    assert kv_bytes_per_token(layers=32, kv_heads=8, head_dim=128, element_bytes=2) == 131072
