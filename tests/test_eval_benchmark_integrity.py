"""Offline regression tests for benchmark integrity contracts."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_common
import eval_ledger
import eval_locomo_retrieval as locomo
import eval_longmemeval_retrieval as longmem
import eval_musique_retrieval as musique
from config import settings
from scripts.ablation_matrix import (
    CLIENT_REQUEST_ATTESTED,
    MODE_BY_NAME,
    annotate_ledger,
    resolve_mode,
)
from scripts.ingest_locomo import (
    _session_sort_key,
    evidence_exists,
    evidence_id,
    normalize_locomo_event_time,
    post_node,
)


class _Response:
    def __init__(self, results=None, error: Exception | None = None):
        self._results = results or []
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return {"results": self._results}


class _Client:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.calls = []

    def post(self, path, json=None, **kwargs):
        self.calls.append((path, json, kwargs))
        return _Response(self.results, self.error)


class _AsyncResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            error = RuntimeError("sensitive response body")
            error.response = type("Response", (), {"status_code": self.status_code})()
            raise error

    def json(self):
        return self._body


class _AsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, path, json=None):
        self.calls.append((path, json))
        return self.responses.pop(0)


def _use_temp_ledgers(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(eval_ledger, "RESULTS_DIR", tmp_path)


def test_locomo_loader_qualifies_evidence_and_question_ids(monkeypatch, tmp_path: Path):
    dataset = [{
        "sample_id": "conv-a",
        "qa": [{"question": "same?", "answer": "a", "evidence": ["D1:2"], "category": 1}],
    }, {
        "sample_id": "conv-b",
        "qa": [{"question": "same?", "answer": "a", "evidence": ["D1:2"], "category": 1}],
    }]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    monkeypatch.setattr(locomo, "LOCOMO_PATH", path)

    questions = locomo.load_questions(samples_per_category=0)

    assert questions[0]["evidence"] == ["locomo:conv-a:D1:2"]
    assert questions[1]["evidence"] == ["locomo:conv-b:D1:2"]
    assert questions[0]["question_id"] != questions[1]["question_id"]


def test_locomo_ingestion_ids_and_session_order_are_stable():
    assert evidence_id("conv-a", "D1:2") == "locomo:conv-a:D1:2"
    assert sorted(["session_10", "session_2", "session_1"], key=_session_sort_key) == [
        "session_1", "session_2", "session_10"
    ]


def test_locomo_ingestion_normalizes_only_declared_timestamp_formats():
    assert normalize_locomo_event_time("1:56 pm on 8 May, 2023") == (
        "2023-05-08T13:56:00+00:00"
    )
    assert normalize_locomo_event_time("2023-05-08T13:56:00+05:30") == (
        "2023-05-08T08:26:00+00:00"
    )
    assert normalize_locomo_event_time("") is None
    with pytest.raises(ValueError, match="LoCoMo session timestamp"):
        normalize_locomo_event_time("sometime last Friday")


def test_ingest_resume_verifies_evidence_through_remote_api():
    expected = "locomo:conv-a:D1:2"
    text = "[SPEAKER: Alex] exact stored turn"
    client = _AsyncClient([_AsyncResponse(200, {"results": [
        {"text": text, "metadata": {"evidence_id": expected}}
    ]})])

    assert asyncio.run(
        evidence_exists(client, expected_id=expected, expected_text=text)
    )
    assert client.calls[0][0] == "/search/vector"
    assert client.calls[0][1]["filter_metadata"] == {"evidence_id": expected}


def test_ingest_409_is_accepted_only_after_remote_metadata_verification():
    expected = "locomo:conv-a:D1:2"
    text = "[SPEAKER: Alex] exact stored turn"
    payload = {"text": text, "metadata": {"evidence_id": expected}}
    client = _AsyncClient([
        _AsyncResponse(200, {"results": []}),
        _AsyncResponse(409, {}),
        _AsyncResponse(200, {"results": [
            {"text": text, "metadata": {"evidence_id": expected}}
        ]}),
    ])

    assert asyncio.run(post_node(client, asyncio.Semaphore(1), payload))
    assert [path for path, _ in client.calls] == [
        "/search/vector", "/nodes", "/search/vector"
    ]


def test_api_key_header_uses_authoritative_settings_and_errors_are_sanitized(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "super-secret")
    assert eval_common.api_headers() == {
        "X-HybridMind-API-Key": "super-secret"
    }
    error = RuntimeError("https://secret.example/?token=super-secret")
    assert eval_common.sanitized_error(error) == "RuntimeError"


def test_paid_fallback_requires_prices_before_live_llm_call(monkeypatch):
    args = SimpleNamespace(
        input_cost_per_million_tokens=0.0,
        output_cost_per_million_tokens=0.0,
    )
    monkeypatch.setattr(settings, "allow_research_proxy", False)
    monkeypatch.setattr(
        eval_common.llm_client, "provider_chain", lambda: ("runpod", "zai")
    )
    with pytest.raises(SystemExit, match="requires explicit"):
        eval_common.enforce_priced_llm_budget(
            args, decomposition_requested=True
        )

    monkeypatch.setattr(
        eval_common.llm_client, "provider_chain", lambda: ("runpod",)
    )
    eval_common.enforce_priced_llm_budget(args, decomposition_requested=True)


@pytest.mark.parametrize("module", [locomo, longmem, musique])
def test_live_eval_defaults_are_dry_and_decomposition_is_opt_in(monkeypatch, module):
    monkeypatch.setattr(sys, "argv", [module.__file__])
    args = module.parse_args()
    assert args.execute is False
    assert args.decompose_multihop is False


def test_exact_evidence_is_distinct_from_answer_overlap():
    result = {
        "text": "The answer is Paris.",
        "metadata": {"benchmark_sample_id": "conv-a", "evidence_id": "locomo:conv-a:D9:9"},
    }
    assert locomo.is_relevant(result["text"], "Paris")
    assert not locomo.is_exact_evidence(result, {"locomo:conv-a:D1:2"}, "conv-a")


def test_locomo_request_is_scoped_uses_final_top_k_and_correct_graph_key(
    monkeypatch, tmp_path: Path
):
    _use_temp_ledgers(monkeypatch, tmp_path)
    result = {
        "node_id": "node-1",
        "text": "Paris",
        "metadata": {
            "benchmark_sample_id": "conv-a",
            "evidence_id": "locomo:conv-a:D1:2",
        },
        "combined_score": 1.0,
        "rerank_score": 0.8,
        "rerank_attempted": True,
        "rerank_applied": True,
    }
    client = _Client([result])
    question = {
        "question_id": "q1", "question": "Where?", "answer": "Paris",
        "category": "single-hop", "sample_id": "conv-a",
        "evidence": ["locomo:conv-a:D1:2"],
    }

    summary = locomo.run_eval(
        [question], client, top_k=10, rerank_pool=25, verbose=False,
        decompose_multihop=False, search_mode="graph_only",
        anchor_node_ids=["anchor"], fusion_mode="rrf",
    )

    payload = client.calls[0][1]
    assert payload["top_k"] == 10
    assert payload["rerank_pool"] == 25
    assert payload["fusion_mode"] == "rrf"
    assert payload["filter_metadata"] == {"benchmark_sample_id": "conv-a"}
    assert payload["anchor_nodes"] == ["anchor"]
    assert "anchor_node_ids" not in payload
    assert summary["hit_at_1"] == 1.0
    assert summary["metric_basis"] == "exact_evidence_id"


def test_locomo_retrieval_error_is_ledgered_and_fails_closed(monkeypatch, tmp_path: Path):
    _use_temp_ledgers(monkeypatch, tmp_path)
    client = _Client(error=RuntimeError("offline"))
    question = {
        "question_id": "q1", "question": "Where?", "answer": "Paris",
        "category": "single-hop", "sample_id": "conv-a",
        "evidence": ["locomo:conv-a:D1:2"],
    }

    with pytest.raises(locomo.EvaluationRunError, match="no score is valid"):
        locomo.run_eval(
            [question], client, top_k=10, rerank_pool=10, verbose=False,
            decompose_multihop=False,
        )

    ledgers = list(tmp_path.glob("*.jsonl"))
    assert len(ledgers) == 1
    row = json.loads(ledgers[0].read_text(encoding="utf-8"))
    assert row["status"] == "retrieval_error"
    assert row["error_type"] == "RuntimeError"
    assert row["judged_correct"] is None
    completions = list(tmp_path.glob("*.completion.json"))
    assert len(completions) == 1
    completion = json.loads(completions[0].read_text(encoding="utf-8"))
    assert completion["status"] == "failed"
    assert completion["summary"]["reason"] == "retrieval_error"


def test_ledger_is_unique_immutable_and_does_not_invent_pre_rerank(tmp_path: Path):
    metrics = eval_ledger.compute_pool_metrics(
        [{"node_id": "n1"}], lambda result: result["node_id"] == "n1"
    )
    assert metrics["gold_in_pool_pre_rerank"] is None
    assert metrics["gold_rank_pre_rerank"] is None
    assert metrics["gold_rank_post_rerank"] == 1

    first = eval_ledger.LedgerWriter("test", {"x": 1}, results_dir=tmp_path)
    second = eval_ledger.LedgerWriter("test", {"x": 1}, results_dir=tmp_path)
    assert first.path != second.path
    assert first.manifest_path.is_file()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"] == {"x": 1}
    assert manifest["run_id"] == first.run_id
    first.finalize(status="completed", summary={"n": 0})
    assert first.completion_path.is_file()
    with pytest.raises(RuntimeError, match="finalized"):
        first.write(
            question_id="late", question_type="default", gold_evidence_ids=[],
            pool_metrics=eval_ledger.empty_pool_metrics(),
        )


def test_ledger_rejects_duplicate_questions_and_false_completion_count(tmp_path: Path):
    writer = eval_ledger.LedgerWriter("test", {"x": 1}, results_dir=tmp_path)
    writer.write(
        question_id="q", question_type="default", gold_evidence_ids=[],
        pool_metrics=eval_ledger.empty_pool_metrics(),
    )
    with pytest.raises(ValueError, match="duplicate question_id"):
        writer.write(
            question_id="q", question_type="default", gold_evidence_ids=[],
            pool_metrics=eval_ledger.empty_pool_metrics(),
        )
    with pytest.raises(ValueError, match="claims n=2"):
        writer.finalize(status="completed", summary={"n": 2})


def test_zai_answer_model_override_is_forwarded(monkeypatch):
    monkeypatch.setattr(settings, "allow_research_proxy", False)
    seen = {}

    def fake_chat(messages, **kwargs):
        seen.update(kwargs)
        return "answer"

    monkeypatch.setattr(eval_common.llm_client, "chat_completion", fake_chat)
    assert eval_common._call({"messages": [], "model": "glm-override"}) == "answer"
    assert seen["model"] == "glm-override"


def test_answer_provider_unavailable_has_explicit_status(monkeypatch):
    monkeypatch.setattr(eval_common, "_is_llm_available", lambda model=None: False)
    result = eval_common.answer_question_with_status("q", ["context"], model="m")
    assert result.status == "provider_unavailable"
    assert result.answer == ""


def test_live_budget_fails_before_query_or_llm_overrun(monkeypatch):
    budget = eval_common.EvaluationBudget(
        max_queries=1,
        max_llm_calls=1,
        max_embedding_texts=1,
        max_input_tokens=100,
        max_output_tokens=10,
        max_wall_seconds=60,
        max_estimated_spend_usd=0.001,
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=1.0,
        allow_unpriced_embedding=True,
    )
    monkeypatch.setattr(settings, "allow_research_proxy", False)
    monkeypatch.setattr(eval_common.llm_client, "chat_completion", lambda *a, **k: "ok")
    with budget.activate():
        eval_common.record_retrieval_query()
        with pytest.raises(eval_common.EvaluationBudgetExceeded, match="query budget"):
            eval_common.record_retrieval_query()
        assert eval_common._call({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}) == "ok"
        with pytest.raises(eval_common.EvaluationBudgetExceeded, match="LLM-call budget"):
            eval_common._call({"messages": [{"role": "user", "content": "again"}], "max_tokens": 1})
    assert budget.usage()["queries"] == 1
    assert budget.usage()["llm_calls"] == 1
    assert budget.usage()["output_tokens_conservative"] == 10


def test_multihop_decomposition_attempt_consumes_llm_budget(monkeypatch):
    budget = eval_common.EvaluationBudget(
        max_queries=0, max_llm_calls=0, max_embedding_texts=0,
        max_input_tokens=0, max_output_tokens=0, max_wall_seconds=60,
        max_estimated_spend_usd=0,
    )
    monkeypatch.setattr(eval_common.llm_client, "is_configured", lambda *a, **k: True)
    with budget.activate(), pytest.raises(
        eval_common.EvaluationBudgetExceeded, match="LLM-call budget"
    ):
        eval_common.retrieve_with_decomposition(
            "combine these facts", "multihop", lambda query: [], decompose_enabled=True
        )


def test_unconfigured_decomposition_does_not_consume_llm_budget(monkeypatch):
    budget = eval_common.EvaluationBudget(
        max_queries=0, max_llm_calls=0, max_embedding_texts=0,
        max_input_tokens=0, max_output_tokens=0, max_wall_seconds=60,
        max_estimated_spend_usd=0,
    )
    monkeypatch.setattr(eval_common.llm_client, "is_configured", lambda *a, **k: False)
    monkeypatch.setattr(
        eval_common.llm_client, "chat_completion", lambda *a, **k: None
    )
    seen = []
    with budget.activate():
        assert eval_common.retrieve_with_decomposition(
            "combine these facts", "multihop", lambda query: seen.append(query) or [],
            decompose_enabled=True,
        ) == []
    assert seen == ["combine these facts"]
    assert budget.usage()["llm_calls"] == 0


def test_decomposition_reserves_actual_300_token_provider_cap(monkeypatch):
    budget = eval_common.EvaluationBudget(
        max_queries=0, max_llm_calls=1, max_embedding_texts=0,
        max_input_tokens=100, max_output_tokens=299, max_wall_seconds=60,
        max_estimated_spend_usd=1,
    )
    monkeypatch.setattr(eval_common.llm_client, "is_configured", lambda *a, **k: True)
    with budget.activate(), pytest.raises(
        eval_common.EvaluationBudgetExceeded, match="output-token budget"
    ):
        eval_common.retrieve_with_decomposition(
            "combine these facts", "multihop", lambda query: [], decompose_enabled=True
        )


def test_musique_answer_overlap_cannot_upgrade_exact_miss(monkeypatch, tmp_path: Path):
    _use_temp_ledgers(monkeypatch, tmp_path)
    client = _Client([{
        "node_id": "n1", "text": "Paris", "metadata": {"paragraph_id": "wrong"},
        "combined_score": 1.0, "rerank_score": 0.9,
        "rerank_attempted": True, "rerank_applied": True,
    }])
    question = {
        "question_id": "q", "question": "Where?", "answer": "Paris",
        "n_hops": 2, "supporting_ids": {"gold"}, "paragraphs": [],
    }
    summary = musique.run_eval(
        [question], client, top_k=10, rerank_pool=25,
        decompose_multihop=False,
    )
    assert summary["hit_at_1"] == 0.0
    assert summary["answer_overlap_proxy"]["hit_at_1"] == 1.0


def test_longmem_answer_overlap_cannot_upgrade_exact_miss(monkeypatch, tmp_path: Path):
    _use_temp_ledgers(monkeypatch, tmp_path)
    client = _Client([{
        "node_id": "n1", "text": "Paris", "metadata": {"session_id": "wrong"},
        "combined_score": 1.0, "rerank_score": 0.9,
        "rerank_attempted": True, "rerank_applied": True,
    }])
    question = {
        "question_id": "q", "question": "Where?", "answer": "Paris",
        "question_type": "single-session-user", "question_date": "",
        "evidence_ids": ["gold"],
    }
    summary = longmem.run_eval(
        [question], client, top_k=10, rerank_pool=25,
        decompose_multihop=False,
    )
    assert summary["hit_at_1"] == 0.0
    assert summary["answer_overlap_proxy"]["hit_at_1"] == 1.0


def test_multievidence_metrics_report_recall_and_all_hit_at_fixed_k(monkeypatch, tmp_path: Path):
    _use_temp_ledgers(monkeypatch, tmp_path)
    result = {
        "node_id": "n1", "text": "support", "metadata": {"session_id": "gold-1"},
        "combined_score": 1.0, "rerank_score": 0.9,
        "rerank_attempted": True, "rerank_applied": True,
    }
    long_summary = longmem.run_eval(
        [{
            "question_id": "long-q", "question": "Where?", "answer": "support",
            "question_type": "multi-session", "question_date": "",
            "evidence_ids": ["gold-1", "gold-2"],
        }],
        _Client([result]), top_k=10, rerank_pool=25, decompose_multihop=False,
    )
    assert long_summary["gold_evidence_recall_at_10"] == 0.5
    assert long_summary["all_gold_evidence_hit_at_10"] == 0.0

    musique_result = {
        **result,
        "metadata": {"paragraph_id": "gold-1"},
    }
    musique_summary = musique.run_eval(
        [{
            "question_id": "music-q", "question": "Where?", "answer": "support",
            "n_hops": 2, "supporting_ids": {"gold-1", "gold-2"}, "paragraphs": [],
        }],
        _Client([musique_result]), top_k=10, rerank_pool=25,
        decompose_multihop=False,
    )
    assert musique_summary["supporting_paragraph_recall_at_10"] == 0.5
    assert musique_summary["all_supporting_paragraphs_hit_at_10"] == 0.0


def test_strict_ablation_annotation_requires_matching_native_manifest(tmp_path: Path):
    plan = resolve_mode(MODE_BY_NAME["vector_only"], benchmark="locomo")
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
            "mode": "vector_only",
            "resolved_settings_sha256": plan["resolved_settings_sha256"],
        },
    }
    writer = eval_ledger.LedgerWriter("locomo", config, results_dir=tmp_path)
    writer.write(
        question_id="q", question_type="default", gold_evidence_ids=["gold"],
        pool_metrics=eval_ledger.compute_pool_metrics(
            [{"node_id": "gold"}], lambda result: True
        ),
    )
    writer.finalize(status="completed", summary={"n": 1})
    destination = tmp_path / "attested.jsonl"

    assert annotate_ledger(
        writer.path, destination, plan, require_attestation=True
    ) == 1
    row = json.loads(destination.read_text(encoding="utf-8"))
    assert row["ablation"]["verification_status"] == CLIENT_REQUEST_ATTESTED

    with writer.path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        annotate_ledger(
            writer.path, tmp_path / "tampered.jsonl", plan,
            require_attestation=True,
        )

    writer.manifest_path.unlink()
    with pytest.raises(ValueError, match="requires its immutable run manifest"):
        annotate_ledger(
            writer.path, tmp_path / "rejected.jsonl", plan, require_attestation=True
        )


def test_legacy_sample_scorer_is_quarantined_before_provider_access():
    from scripts import score_sample_20

    with pytest.raises(SystemExit, match="quarantined"):
        score_sample_20.main()


@pytest.mark.parametrize(
    ("runner", "question"),
    [
        (
            locomo.run_eval,
            {
                "question_id": "lq", "question": "Where?", "answer": "Paris",
                "category": "single-hop", "sample_id": "conv",
                "evidence": ["locomo:conv:d1"],
            },
        ),
        (
            longmem.run_eval,
            {
                "question_id": "mq", "question": "Where?", "answer": "Paris",
                "question_type": "single-session", "evidence_ids": ["s1"],
            },
        ),
        (
            musique.run_eval,
            {
                "question_id": "uq", "question": "Where?", "answer": "Paris",
                "n_hops": 2, "supporting_ids": {"p1"}, "paragraphs": [],
            },
        ),
    ],
)
def test_evaluators_reject_positive_rerank_pool_below_final_top_k(runner, question):
    client = _Client([])
    with pytest.raises(ValueError, match="greater than or equal to top_k"):
        runner([question], client, top_k=10, rerank_pool=9)
    assert client.calls == []


def test_locomo_same_set_sweep_is_quarantined_before_api_access():
    client = _Client(error=AssertionError("API must not be called"))
    with pytest.raises(SystemExit, match="same-set tuning"):
        locomo.sweep([], client)
    assert client.calls == []
