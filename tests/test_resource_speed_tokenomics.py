from __future__ import annotations

import copy
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.resource_accounting import (
    ANALYTIC_CLASSIFICATION,
    LIVE_PLAN_SCHEMA,
    ResourceAccountingError,
    capacity_projection,
    load_validated_offline_report,
    percentile,
    sha256_file,
    tokenomics_projection,
    validate_offline_report,
)
from scripts import offline_resource_frontier, preflight


def _offline_args(tmp_path: Path):
    return offline_resource_frontier.parse_args(
        [
            "--output",
            str(tmp_path / "offline.json"),
            "--vectors",
            "8",
            "--queries",
            "4",
            "--top-k",
            "3",
            "--batch-size",
            "4",
            "--capacity-source-tokens",
            "10000000",
            "40000000",
            "100000000",
        ]
    )


@pytest.fixture()
def offline_report(tmp_path: Path) -> tuple[Path, dict]:
    report = offline_resource_frontier.build_report(_offline_args(tmp_path))
    report_path = tmp_path / "offline.json"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return report_path, report


def _live_plan(report_path: Path, report: dict) -> dict:
    measured = report["measured"]
    return {
        "schema_version": LIVE_PLAN_SCHEMA,
        "offline_report_path": str(report_path),
        "offline_report_sha256": sha256_file(report_path),
        "providers": ["tei"],
        "preflight_usage_included": True,
        "planned_usage": {
            "queries": 1,
            "embedding_calls": 2,
            "embedding_input_tokens": 20,
            "reranker_calls": 0,
            "reranker_pairs": 0,
            "reranker_input_tokens": 0,
            "llm_calls": 0,
            "reader_input_tokens": 0,
            "reader_output_tokens": 0,
            "provider_runtime_seconds": 60,
        },
        "usage_ceiling": {
            "queries": 1,
            "embedding_calls": 2,
            "embedding_input_tokens": 20,
            "reranker_calls": 0,
            "reranker_pairs": 0,
            "reranker_input_tokens": 0,
            "llm_calls": 0,
            "reader_input_tokens": 0,
            "reader_output_tokens": 0,
            "provider_runtime_seconds": 60,
        },
        "rates": {
            "pricing_mode": "priced",
            "embedding_usd_per_million_input_tokens": 1.0,
            "reranker_usd_per_million_input_tokens": 0.0,
            "reader_usd_per_million_input_tokens": 0.0,
            "reader_usd_per_million_output_tokens": 0.0,
            "fixed_usd_per_embedding_call": 0.001,
            "fixed_usd_per_reranker_call": 0.0,
            "fixed_usd_per_llm_call": 0.0,
            "provider_runtime_usd_per_second": 0.0001,
        },
        "max_estimated_spend_usd": 0.01,
        "max_wall_seconds": 60,
        "resource_limits": {
            "max_report_age_hours": 24,
            "max_component_sequence_p95_ms": measured["component_sequence_latency"]["p95_ms"] + 1,
            "max_component_sequence_p99_ms": measured["component_sequence_latency"]["p99_ms"] + 1,
            "max_observed_peak_rss_bytes": measured["observed_peak_rss_bytes"] + 1,
            "max_serialized_total_bytes": measured["serialized_total_bytes"] + 1,
            "max_total_index_build_seconds": measured["total_index_build_seconds"] + 1,
            "min_available_memory_bytes": 0,
            "min_free_disk_bytes": 0,
        },
    }


def test_percentile_is_interpolated_and_rejects_invalid_samples():
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    with pytest.raises(ResourceAccountingError):
        percentile([], 0.5)
    with pytest.raises(ResourceAccountingError):
        percentile([float("nan")], 0.5)


def test_capacity_projection_distinguishes_projection_and_current_duplicate_cache():
    projection = capacity_projection(10_000_000, 256)
    assert projection["classification"] == ANALYTIC_CLASSIFICATION
    assert projection["vector_count"] == 39_063
    assert projection["bytes_per_vector"]["raw_float32_one_copy"] == 16_384
    assert projection["bytes_per_vector"]["faiss_hnsw_flat"] == 16_640
    assert projection["bytes_per_vector"]["hybridmind_current_vector_component_lower_bound"] == 33_024
    assert projection["bytes_per_vector"]["hnsw_links_plus_sq8_encoding"] == 4_352
    assert projection["bytes_per_vector"]["hnsw_links_plus_pq_encoding"] == 320
    assert projection["feasibility_status"] == "not_established_by_projection"


def test_unpriced_usage_never_reports_zero_as_a_known_cost():
    usage = {
        "queries": 1,
        "embedding_calls": 1,
        "embedding_input_tokens": 10,
        "reranker_calls": 0,
        "reranker_pairs": 0,
        "reranker_input_tokens": 0,
        "llm_calls": 0,
        "reader_input_tokens": 0,
        "reader_output_tokens": 0,
        "provider_runtime_seconds": 0,
    }
    result = tokenomics_projection(usage, {"pricing_mode": "unpriced"})
    assert result["pricing_complete"] is False
    assert result["projected_cost_usd"] is None
    assert result["actual_external_cost_usd"] == 0.0
    assert result["external_calls_performed"] == 0


def test_bounded_offline_report_is_valid_and_contains_raw_evidence(offline_report):
    report_path, report = offline_report
    validate_offline_report(report)
    loaded = load_validated_offline_report(report_path)
    assert loaded["execution"]["external_network_calls"] == 0
    assert loaded["measured"]["dimension"] == 4096
    assert len(loaded["measured"]["component_sequence_latency_samples_ms"]) == 4
    assert loaded["measured"]["serialized_total_bytes"] > 0
    assert all(
        item["classification"] == ANALYTIC_CLASSIFICATION
        for item in loaded["capacity_projections"]
    )
    assert loaded["prompt_token_scenario"]["model_kv_cache_reduction_claimed"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda report: report["execution"].update(external_network_calls=1), "zero external"),
        (lambda report: report["measured"].update(classification="measured_live"), "mislabeled"),
        (lambda report: report["measured"].update(serialized_total_bytes=1), "serialized total"),
        (
            lambda report: report["measured"].update(total_component_build_items_per_second=1),
            "throughput",
        ),
        (
            lambda report: report["measured"]["component_sequence_latency"].update(p95_ms=0.0),
            "raw samples",
        ),
        (
            lambda report: report["capacity_projections"][0].update(feasibility_status="validated"),
            "projection",
        ),
        (
            lambda report: report["host_capacity_assessments"][0].update(feasibility_status="validated"),
            "capacity assessment",
        ),
        (
            lambda report: report["prompt_token_scenario"].update(model_kv_cache_reduction_claimed=True),
            "prompt-token",
        ),
    ],
)
def test_offline_validator_rejects_fabricated_or_mislabeled_metrics(
    offline_report, mutator, message
):
    _path, valid = offline_report
    invalid = copy.deepcopy(valid)
    mutator(invalid)
    with pytest.raises(ResourceAccountingError, match=message):
        validate_offline_report(invalid)


def test_offline_workload_refuses_unbounded_or_memory_unsafe_requests(tmp_path: Path):
    args = _offline_args(tmp_path)
    args.vectors = args.max_vectors + 1
    with pytest.raises(ValueError, match="max-vectors"):
        offline_resource_frontier.build_report(args)


def test_preflight_missing_or_malformed_plan_makes_zero_provider_calls(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "PROVIDER_CHECKS",
        {"tei": ("TEI", lambda: (calls.append("tei") or True, "ok"))},
    )
    assert preflight.main(["--plan", str(tmp_path / "missing.json")]) == 2
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert preflight.main(["--plan", str(malformed)]) == 2
    assert calls == []


def test_preflight_over_budget_plan_makes_zero_provider_calls(
    offline_report, tmp_path, monkeypatch
):
    report_path, report = offline_report
    plan = _live_plan(report_path, report)
    plan["max_estimated_spend_usd"] = 0.0
    plan_path = tmp_path / "over-budget.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "PROVIDER_CHECKS",
        {"tei": ("TEI", lambda: (calls.append("tei") or True, "ok"))},
    )
    assert preflight.main(["--plan", str(plan_path)]) == 2
    assert calls == []


def test_preflight_validate_only_makes_zero_calls(offline_report, tmp_path, monkeypatch):
    report_path, report = offline_report
    plan_path = tmp_path / "valid.json"
    plan_path.write_text(json.dumps(_live_plan(report_path, report)), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "PROVIDER_CHECKS",
        {"tei": ("TEI", lambda: (calls.append("tei") or True, "ok"))},
    )
    assert preflight.main(["--plan", str(plan_path), "--validate-only"]) == 0
    assert calls == []


def test_preflight_calls_only_plan_selected_provider(offline_report, tmp_path, monkeypatch):
    report_path, report = offline_report
    plan_path = tmp_path / "valid.json"
    plan_path.write_text(json.dumps(_live_plan(report_path, report)), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "PROVIDER_CHECKS",
        {
            "tei": ("TEI", lambda: (calls.append("tei") or True, "ok")),
            "zai": ("Z.AI", lambda: (calls.append("zai") or True, "ok")),
        },
    )
    assert preflight.main(["--plan", str(plan_path)]) == 0
    assert calls == ["tei"]


def test_live_plan_rejects_report_from_different_host_before_checks(
    offline_report, tmp_path, monkeypatch
):
    report_path, report = offline_report
    report["host"]["node"] = platform.node() + "-different"
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    plan = _live_plan(report_path, report)
    plan_path = tmp_path / "other-host.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "PROVIDER_CHECKS",
        {"tei": ("TEI", lambda: (calls.append("tei") or True, "ok"))},
    )
    assert preflight.main(["--plan", str(plan_path)]) == 2
    assert calls == []


def test_live_plan_rejects_stale_report_before_checks(offline_report, tmp_path, monkeypatch):
    report_path, report = offline_report
    report["generated_at"] = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    plan = _live_plan(report_path, report)
    plan_path = tmp_path / "stale.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "PROVIDER_CHECKS",
        {"tei": ("TEI", lambda: (calls.append("tei") or True, "ok"))},
    )
    assert preflight.main(["--plan", str(plan_path)]) == 2
    assert calls == []
