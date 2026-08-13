import json
import subprocess
from pathlib import Path

import pytest

from eval_stats import metric_values
from scripts.ablation_matrix import (
    CLIENT_REQUEST_ATTESTED,
    MODE_BY_NAME,
    MODES,
    _redact_config,
    annotate_ledger,
    execute_request_attested_plan,
    resolve_mode,
    select_modes,
    write_plans,
)


def _native_ledger_row() -> dict:
    return {
        "question_id": "q-1",
        "question_type": "default",
        "gold_evidence_ids": ["node-1"],
        "retrieved_ids_at_k": {"1": ["node-1"]},
        "gold_in_pool_pre_rerank": True,
        "gold_rank_post_rerank": 1,
        "raw_llm_answer": "answer",
        "judged_correct": True,
        "judge_rationale": "correct",
        "prompt_version": "test",
        "config_hash": "native-hash",
        "seed": 42,
        "timestamp": 1.0,
    }


def test_matrix_has_required_distinct_modes():
    assert {mode.name for mode in MODES} == {
        "vector_only",
        "sparse_only",
        "vector_sparse",
        "graph_only",
        "hybrid",
        "temporal",
        "salience",
        "structured_facts",
        "compression",
    }


def test_resolved_mode_pins_4096_and_is_deterministic():
    first = resolve_mode(MODE_BY_NAME["temporal"], benchmark="locomo")
    second = resolve_mode(MODE_BY_NAME["temporal"], benchmark="locomo")

    assert first["plan_hash"] == second["plan_hash"]
    assert first["resolved_settings"]["embedding_dimension"] == 4096
    assert first["server_boot_environment"]["HYBRIDMIND_TEMPORAL_DECAY_ENABLED"] == "true"
    assert first["server_boot_environment"]["HYBRIDMIND_QUERY_TIME_EXPANSION_ENABLED"] == "true"
    assert first["server_boot_environment"]["HYBRIDMIND_SALIENCE_ENABLED"] == "false"
    assert first["request_parameters"] == {
        "top_k": 15,
        "vector_weight": 0.5,
        "graph_weight": 0.15,
        "bm25_boost_weight": 0.35,
        "rerank_pool": 0,
        "fusion_mode": "rrf",
        "search_mode": "hybrid",
        "route_weights": False,
        "track_access": False,
    }
    assert first["protocol"]["server_runtime_attested"] is False
    assert first["protocol"]["attestation_scope"] == "client_request_and_artifact_integrity_only"

    graph_only = resolve_mode(MODE_BY_NAME["graph_only"], benchmark="locomo")
    assert graph_only["request_parameters"]["search_mode"] == "graph_only"
    assert graph_only["evaluator_parameters"]["graph_anchor_strategy"] == "explicit"
    assert graph_only["protocol"]["result_eligibility"] == "plan_only_missing_external_attestation"


@pytest.mark.parametrize(
    "mode_name", ["graph_only", "hybrid", "temporal", "salience", "structured_facts", "compression"]
)
def test_stateful_or_anchor_dependent_modes_are_plan_only(mode_name, tmp_path: Path):
    source = tmp_path / "ledger.jsonl"
    source.write_text(json.dumps(_native_ledger_row()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plan-only"):
        annotate_ledger(
            source,
            tmp_path / "out.jsonl",
            resolve_mode(MODE_BY_NAME[mode_name], benchmark="locomo"),
            require_attestation=True,
        )


def test_all_provider_api_keys_are_redacted():
    redacted = _redact_config(
        {
            "zai_api_key": "zai-secret",
            "runpod_api_key": "runpod-secret",
            "research_proxy_api_key": "research-secret",
            "research_proxy_base_url": "https://example.invalid/v1",
        }
    )
    assert redacted == {
        "zai_api_key": "<redacted>",
        "runpod_api_key": "<redacted>",
        "research_proxy_api_key": "<redacted>",
        "research_proxy_base_url": "https://example.invalid/v1",
    }


def test_select_modes_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        select_modes("vector_only,not-a-mode")


def test_plans_are_written_stably(tmp_path: Path):
    plan = resolve_mode(MODE_BY_NAME["hybrid"], benchmark="locomo")
    paths = write_plans([plan], tmp_path)

    assert paths == [tmp_path / f"hybrid_{plan['plan_hash']}.plan.json"]
    persisted = json.loads(paths[0].read_text(encoding="utf-8"))
    assert persisted["protocol"]["result_status"] == "planned; no benchmark result recorded"


def test_annotated_ledger_keeps_native_metrics_and_adds_provenance(tmp_path: Path):
    source = tmp_path / "native.jsonl"
    source.write_text(json.dumps(_native_ledger_row()) + "\n", encoding="utf-8")
    destination = tmp_path / "annotated.jsonl"
    plan = resolve_mode(MODE_BY_NAME["hybrid"], benchmark="locomo")

    assert annotate_ledger(source, destination, plan) == 1
    row = json.loads(destination.read_text(encoding="utf-8"))
    assert row["gold_rank_post_rerank"] == 1
    assert row["source_config_hash"] == "native-hash"
    assert row["config_hash"] == plan["plan_hash"]
    assert row["ablation"]["mode"] == "hybrid"
    # eval_stats intentionally accepts additional provenance fields, so the
    # copied output remains a drop-in comparison ledger.
    assert metric_values([row], "hit1") == {"q-1": 1.0}


def test_annotate_ledger_rejects_incomplete_rows(tmp_path: Path):
    source = tmp_path / "incomplete.jsonl"
    source.write_text(json.dumps({"question_id": "q-1"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        annotate_ledger(
            source,
            tmp_path / "out.jsonl",
            resolve_mode(MODE_BY_NAME["vector_only"], benchmark="locomo"),
        )


def test_request_attested_executor_runs_retrieval_only_and_verifies_receipt(
    monkeypatch, tmp_path: Path
):
    import eval_ledger

    plan = resolve_mode(MODE_BY_NAME["vector_only"], benchmark="locomo")
    seen = {}
    monkeypatch.delenv("HYBRIDMIND_TEMPORAL_DECAY_ENABLED", raising=False)
    monkeypatch.setattr(eval_ledger, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(
        eval_ledger,
        "git_worktree_state",
        lambda: {"dirty": False, "status_sha256": "test-status"},
    )

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        run_dir = Path(kwargs["env"]["HYBRIDMIND_EVAL_RESULTS_DIR"])
        config = {
            "top_k": 15,
            "vector_weight": 1.0,
            "graph_weight": 0.0,
            "bm25_boost": 0.0,
            "rerank_pool": 0,
            "fusion_mode": "rrf",
            "search_mode": "vector_only",
            "route_weights": False,
            "track_access": False,
            "ablation": {
                "plan_hash": plan["plan_hash"],
                "mode": plan["mode"],
                "resolved_settings_sha256": plan["resolved_settings_sha256"],
            },
        }
        writer = eval_ledger.LedgerWriter("locomo", config, results_dir=run_dir)
        writer.write(
            question_id="q", question_type="default", gold_evidence_ids=["gold"],
            pool_metrics=eval_ledger.compute_pool_metrics(
                [{"node_id": "gold"}], lambda result: True
            ),
        )
        writer.finalize(status="completed", summary={"n": 1})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    destination = execute_request_attested_plan(
        plan, tmp_path, question_limit=1, timeout_seconds=10
    )
    row = json.loads(destination.read_text(encoding="utf-8"))
    assert row["ablation"]["verification_status"] == CLIENT_REQUEST_ATTESTED
    assert "--with-answers" not in seen["command"]
    assert "--decompose-multihop" not in seen["command"]
    assert seen["command"][seen["command"].index("--rerank-pool") + 1] == "0"
    assert seen["command"][seen["command"].index("--top-k") + 1] == "15"
    assert seen["command"][seen["command"].index("--fusion-mode") + 1] == "rrf"
    assert "HYBRIDMIND_TEMPORAL_DECAY_ENABLED" not in seen.get("env", {})
