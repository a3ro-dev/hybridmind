"""Resource, latency, and spend accounting for bounded offline experiments.

This module is deliberately independent of provider clients.  It validates
measurements and live-run plans but cannot initiate a network request.  The
separation makes it possible for preflight to fail closed before importing or
calling any provider-specific code.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import psutil


OFFLINE_REPORT_SCHEMA = "hybridmind-offline-resource-report/v1"
LIVE_PLAN_SCHEMA = "hybridmind-live-eval-plan/v1"
MEASURED_CLASSIFICATION = "measured_offline"
ANALYTIC_CLASSIFICATION = "analytic_projection"
SCENARIO_CLASSIFICATION = "scenario_projection"

EMBEDDING_DIMENSION = 4096
FLOAT32_BYTES = 4
DEFAULT_HNSW_M = 32


class ResourceAccountingError(ValueError):
    """A measurement artifact or live plan is invalid."""


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile over finite samples."""
    if not samples:
        raise ResourceAccountingError("percentile requires at least one sample")
    if not 0.0 <= quantile <= 1.0:
        raise ResourceAccountingError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in samples)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ResourceAccountingError("latency samples must be finite and non-negative")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize raw latency observations without discarding their count."""
    return {
        "sample_count": len(samples_ms),
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "p99_ms": percentile(samples_ms, 0.99),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def capacity_projection(
    source_tokens: int,
    mean_source_tokens_per_chunk: int,
    *,
    dimension: int = EMBEDDING_DIMENSION,
    hnsw_m: int = DEFAULT_HNSW_M,
    pq_code_bytes: int = 64,
) -> dict[str, Any]:
    """Compute transparent byte lower bounds; this is not a scale benchmark.

    FAISS documents ``4*d + M*2*4`` bytes/vector for IndexHNSWFlat.  HybridMind
    also retains one float32 copy in ``VectorIndex._raw_vectors`` for rebuilds,
    so the implementation-specific lower bound adds another ``4*d`` bytes.
    Python objects, IDs, allocator fragmentation, SQLite, BM25, graph storage,
    source text, and build-time scratch memory are intentionally excluded.
    """
    for name, value in (
        ("source_tokens", source_tokens),
        ("mean_source_tokens_per_chunk", mean_source_tokens_per_chunk),
        ("dimension", dimension),
        ("hnsw_m", hnsw_m),
        ("pq_code_bytes", pq_code_bytes),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ResourceAccountingError(f"{name} must be a positive integer")
    if dimension != EMBEDDING_DIMENSION:
        raise ResourceAccountingError("HybridMind capacity projections require 4096 dimensions")

    vector_count = math.ceil(source_tokens / mean_source_tokens_per_chunk)
    raw_float32_bytes_per_vector = dimension * FLOAT32_BYTES
    hnsw_link_bytes_per_vector = hnsw_m * 2 * 4
    faiss_hnsw_bytes_per_vector = raw_float32_bytes_per_vector + hnsw_link_bytes_per_vector
    hybridmind_vector_lower_bound_per_vector = (
        faiss_hnsw_bytes_per_vector + raw_float32_bytes_per_vector
    )
    return {
        "classification": ANALYTIC_CLASSIFICATION,
        "source_tokens": source_tokens,
        "mean_source_tokens_per_chunk": mean_source_tokens_per_chunk,
        "vector_count": vector_count,
        "assumptions": {
            "dimension": dimension,
            "hnsw_m": hnsw_m,
            "float32_bytes_per_component": FLOAT32_BYTES,
            "pq_code_bytes": pq_code_bytes,
            "faiss_hnsw_formula": "4*d + M*2*4 bytes/vector",
            "hybridmind_raw_rebuild_cache_copies": 1,
        },
        "bytes_per_vector": {
            "raw_float32_one_copy": raw_float32_bytes_per_vector,
            "hnsw_links_only": hnsw_link_bytes_per_vector,
            "faiss_hnsw_flat": faiss_hnsw_bytes_per_vector,
            "hybridmind_current_vector_component_lower_bound": hybridmind_vector_lower_bound_per_vector,
            "float16_encoding_only": dimension * 2,
            "sq8_encoding_only": dimension,
            "pq_encoding_only": pq_code_bytes,
            "hnsw_links_plus_float16_encoding": hnsw_link_bytes_per_vector + dimension * 2,
            "hnsw_links_plus_sq8_encoding": hnsw_link_bytes_per_vector + dimension,
            "hnsw_links_plus_pq_encoding": hnsw_link_bytes_per_vector + pq_code_bytes,
        },
        "total_bytes": {
            "raw_float32_one_copy": vector_count * raw_float32_bytes_per_vector,
            "hnsw_links_only": vector_count * hnsw_link_bytes_per_vector,
            "faiss_hnsw_flat": vector_count * faiss_hnsw_bytes_per_vector,
            "hybridmind_current_vector_component_lower_bound": (
                vector_count * hybridmind_vector_lower_bound_per_vector
            ),
            "float16_encoding_only": vector_count * dimension * 2,
            "sq8_encoding_only": vector_count * dimension,
            "pq_encoding_only": vector_count * pq_code_bytes,
            "hnsw_links_plus_float16_encoding": vector_count * (
                hnsw_link_bytes_per_vector + dimension * 2
            ),
            "hnsw_links_plus_sq8_encoding": vector_count * (
                hnsw_link_bytes_per_vector + dimension
            ),
            "hnsw_links_plus_pq_encoding": vector_count * (
                hnsw_link_bytes_per_vector + pq_code_bytes
            ),
        },
        "excluded_from_bounds": [
            "Python object and allocator overhead",
            "node IDs and ID maps",
            "SQLite rows and source text",
            "BM25 postings and corpus copies",
            "NetworkX nodes and edges",
            "quantizer/codebooks and inverted-list IDs",
            "build-time scratch memory",
        ],
        "feasibility_status": "not_established_by_projection",
    }


def tokenomics_projection(
    usage: Mapping[str, int],
    rates: Mapping[str, Any],
) -> dict[str, Any]:
    """Project cost from caller-supplied usage and prices without making calls."""
    required_usage = (
        "queries",
        "embedding_calls",
        "embedding_input_tokens",
        "reranker_calls",
        "reranker_pairs",
        "reranker_input_tokens",
        "llm_calls",
        "reader_input_tokens",
        "reader_output_tokens",
        "provider_runtime_seconds",
    )
    normalized_usage: dict[str, int] = {}
    for key in required_usage:
        value = usage.get(key)
        if not isinstance(value, int) or value < 0:
            raise ResourceAccountingError(f"usage.{key} must be a non-negative integer")
        normalized_usage[key] = value

    mode = rates.get("pricing_mode")
    if mode not in {"priced", "unpriced"}:
        raise ResourceAccountingError("rates.pricing_mode must be 'priced' or 'unpriced'")
    rate_keys = (
        "embedding_usd_per_million_input_tokens",
        "reranker_usd_per_million_input_tokens",
        "reader_usd_per_million_input_tokens",
        "reader_usd_per_million_output_tokens",
        "fixed_usd_per_embedding_call",
        "fixed_usd_per_reranker_call",
        "fixed_usd_per_llm_call",
        "provider_runtime_usd_per_second",
    )
    normalized_rates: dict[str, float | str] = {"pricing_mode": mode}
    for key in rate_keys:
        value = rates.get(key)
        if mode == "unpriced" and value is None:
            normalized_rates[key] = 0.0
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ResourceAccountingError(f"rates.{key} must be finite and non-negative")
        normalized_rates[key] = float(value)

    has_usage = any(normalized_usage[key] for key in required_usage if key != "queries")
    if mode == "unpriced":
        return {
            "classification": SCENARIO_CLASSIFICATION,
            "usage": normalized_usage,
            "rates": normalized_rates,
            "projected_cost_usd": None,
            "pricing_complete": not has_usage,
            "actual_external_cost_usd": 0.0,
            "external_calls_performed": 0,
        }

    cost = (
        normalized_usage["embedding_input_tokens"]
        * float(normalized_rates["embedding_usd_per_million_input_tokens"])
        + normalized_usage["reranker_input_tokens"]
        * float(normalized_rates["reranker_usd_per_million_input_tokens"])
        + normalized_usage["reader_input_tokens"]
        * float(normalized_rates["reader_usd_per_million_input_tokens"])
        + normalized_usage["reader_output_tokens"]
        * float(normalized_rates["reader_usd_per_million_output_tokens"])
    ) / 1_000_000
    cost += normalized_usage["embedding_calls"] * float(
        normalized_rates["fixed_usd_per_embedding_call"]
    )
    cost += normalized_usage["reranker_calls"] * float(
        normalized_rates["fixed_usd_per_reranker_call"]
    )
    cost += normalized_usage["llm_calls"] * float(
        normalized_rates["fixed_usd_per_llm_call"]
    )
    cost += normalized_usage["provider_runtime_seconds"] * float(
        normalized_rates["provider_runtime_usd_per_second"]
    )
    return {
        "classification": SCENARIO_CLASSIFICATION,
        "usage": normalized_usage,
        "rates": normalized_rates,
        "projected_cost_usd": cost,
        "pricing_complete": True,
        "actual_external_cost_usd": 0.0,
        "external_calls_performed": 0,
    }


def prompt_reduction(
    *,
    indexed_source_tokens: int,
    baseline_prompt_source_tokens_per_query: int,
    retrieved_unique_source_tokens_per_query: int,
) -> dict[str, Any]:
    """Compute prompt-source reduction for an explicitly defined scenario."""
    for name, value in (
        ("indexed_source_tokens", indexed_source_tokens),
        ("baseline_prompt_source_tokens_per_query", baseline_prompt_source_tokens_per_query),
        ("retrieved_unique_source_tokens_per_query", retrieved_unique_source_tokens_per_query),
    ):
        if not isinstance(value, int) or value < 0:
            raise ResourceAccountingError(f"{name} must be a non-negative integer")
    if baseline_prompt_source_tokens_per_query == 0:
        raise ResourceAccountingError("baseline prompt source tokens must be positive")
    if retrieved_unique_source_tokens_per_query > baseline_prompt_source_tokens_per_query:
        raise ResourceAccountingError("retrieved tokens cannot exceed the declared baseline")
    return {
        "classification": SCENARIO_CLASSIFICATION,
        "indexed_source_tokens": indexed_source_tokens,
        "baseline_prompt_source_tokens_per_query": baseline_prompt_source_tokens_per_query,
        "retrieved_unique_source_tokens_per_query": retrieved_unique_source_tokens_per_query,
        "source_token_reduction_fraction": (
            baseline_prompt_source_tokens_per_query - retrieved_unique_source_tokens_per_query
        ) / baseline_prompt_source_tokens_per_query,
        "retrieval_conditioned_effective_context_multiplier": (
            None
            if retrieved_unique_source_tokens_per_query == 0
            else indexed_source_tokens / retrieved_unique_source_tokens_per_query
        ),
        "model_kv_cache_reduction_claimed": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResourceAccountingError(f"{name} must be an object")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ResourceAccountingError(f"{name} must be finite and non-negative")
    return float(value)


def validate_offline_report(report: Mapping[str, Any]) -> None:
    """Recompute derived fields and reject mislabeled or internally inconsistent data."""
    if report.get("schema_version") != OFFLINE_REPORT_SCHEMA:
        raise ResourceAccountingError("unsupported offline report schema")
    execution = _require_mapping(report.get("execution"), "execution")
    if execution.get("mode") != "offline_synthetic":
        raise ResourceAccountingError("offline report execution.mode must be offline_synthetic")
    if execution.get("external_network_calls") != 0:
        raise ResourceAccountingError("offline report must record zero external network calls")
    if execution.get("embedding_inference_performed") is not False:
        raise ResourceAccountingError("offline report must not claim embedding inference")

    measured = _require_mapping(report.get("measured"), "measured")
    if measured.get("classification") != MEASURED_CLASSIFICATION:
        raise ResourceAccountingError("measured results are mislabeled")
    vector_count = measured.get("vector_count")
    if not isinstance(vector_count, int) or vector_count <= 0:
        raise ResourceAccountingError("measured.vector_count must be positive")
    add_seconds = _finite_nonnegative(measured.get("vector_index_add_seconds"), "vector_index_add_seconds")
    if add_seconds <= 0:
        raise ResourceAccountingError("vector index add time must be positive")
    throughput = _finite_nonnegative(
        measured.get("vector_index_add_vectors_per_second"),
        "vector_index_add_vectors_per_second",
    )
    expected_throughput = vector_count / add_seconds
    if not math.isclose(throughput, expected_throughput, rel_tol=1e-9, abs_tol=1e-9):
        raise ResourceAccountingError("vector ingestion throughput does not match count/time")
    for seconds_key, throughput_key in (
        ("sparse_add_and_materialize_seconds", "sparse_add_and_materialize_documents_per_second"),
        ("graph_build_seconds", "graph_build_nodes_per_second"),
        ("total_index_build_seconds", "total_component_build_items_per_second"),
    ):
        elapsed = _finite_nonnegative(measured.get(seconds_key), seconds_key)
        if elapsed <= 0:
            raise ResourceAccountingError(f"{seconds_key} must be positive")
        actual = _finite_nonnegative(measured.get(throughput_key), throughput_key)
        expected = vector_count / elapsed
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ResourceAccountingError(
                f"throughput {throughput_key} does not match count/time"
            )

    latency_samples = measured.get("component_sequence_latency_samples_ms")
    if not isinstance(latency_samples, list) or len(latency_samples) < 2:
        raise ResourceAccountingError("raw component latency samples are required")
    expected_latency = latency_summary(latency_samples)
    actual_latency = _require_mapping(
        measured.get("component_sequence_latency"), "component_sequence_latency"
    )
    for key, expected in expected_latency.items():
        actual = actual_latency.get(key)
        if isinstance(expected, int):
            if actual != expected:
                raise ResourceAccountingError(f"latency {key} does not match raw samples")
        elif not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ResourceAccountingError(f"latency {key} does not match raw samples")

    rss_samples = measured.get("rss_samples_bytes")
    if not isinstance(rss_samples, list) or not rss_samples:
        raise ResourceAccountingError("raw RSS samples are required")
    if any(not isinstance(item, int) or item < 0 for item in rss_samples):
        raise ResourceAccountingError("RSS samples must be non-negative integers")
    if measured.get("observed_peak_rss_bytes") != max(rss_samples):
        raise ResourceAccountingError("observed peak RSS does not match raw samples")

    component_sizes = measured.get("serialized_component_bytes")
    if not isinstance(component_sizes, Mapping) or not component_sizes:
        raise ResourceAccountingError("serialized component sizes are required")
    if any(not isinstance(value, int) or value < 0 for value in component_sizes.values()):
        raise ResourceAccountingError("serialized component sizes must be non-negative integers")
    if measured.get("serialized_total_bytes") != sum(component_sizes.values()):
        raise ResourceAccountingError("serialized total does not match component sizes")
    if measured.get("deterministic_replay_equal") is not True:
        raise ResourceAccountingError("offline query replay was not deterministic")

    projections = report.get("capacity_projections")
    if not isinstance(projections, list) or not projections:
        raise ResourceAccountingError("capacity projections are required")
    for position, projection in enumerate(projections):
        projection = _require_mapping(projection, f"capacity_projections[{position}]")
        assumptions = _require_mapping(projection.get("assumptions"), "projection.assumptions")
        expected = capacity_projection(
            projection.get("source_tokens"),
            projection.get("mean_source_tokens_per_chunk"),
            dimension=assumptions.get("dimension"),
            hnsw_m=assumptions.get("hnsw_m"),
            pq_code_bytes=assumptions.get("pq_code_bytes"),
        )
        if projection != expected:
            raise ResourceAccountingError("capacity projection does not match its assumptions")

    host = _require_mapping(report.get("host"), "host")
    host_total_memory = host.get("total_memory_bytes")
    if not isinstance(host_total_memory, int) or host_total_memory <= 0:
        raise ResourceAccountingError("host.total_memory_bytes must be positive")
    assessments = report.get("host_capacity_assessments")
    if not isinstance(assessments, list) or len(assessments) != len(projections):
        raise ResourceAccountingError("one host capacity assessment is required per projection")
    for projection, assessment in zip(projections, assessments):
        assessment = _require_mapping(assessment, "host_capacity_assessment")
        lower_bound = projection["total_bytes"][
            "hybridmind_current_vector_component_lower_bound"
        ]
        expected_assessment = {
            "classification": SCENARIO_CLASSIFICATION,
            "source_tokens": projection["source_tokens"],
            "host_total_memory_bytes": host_total_memory,
            "current_vector_component_lower_bound_bytes": lower_bound,
            "current_vector_component_lower_bound_fraction_of_host_memory": (
                lower_bound / host_total_memory
            ),
            "passes_vector_lower_bound_half_ram_gate": lower_bound <= host_total_memory / 2,
            "feasibility_status": "not_established_due_excluded_components",
        }
        if assessment != expected_assessment:
            raise ResourceAccountingError("host capacity assessment is internally inconsistent")

    prompt = _require_mapping(report.get("prompt_token_scenario"), "prompt_token_scenario")
    expected_prompt = prompt_reduction(
        indexed_source_tokens=prompt.get("indexed_source_tokens"),
        baseline_prompt_source_tokens_per_query=prompt.get(
            "baseline_prompt_source_tokens_per_query"
        ),
        retrieved_unique_source_tokens_per_query=prompt.get(
            "retrieved_unique_source_tokens_per_query"
        ),
    )
    if prompt != expected_prompt:
        raise ResourceAccountingError("prompt-token scenario is internally inconsistent")

    tokenomics = _require_mapping(report.get("tokenomics_scenario"), "tokenomics_scenario")
    expected_tokenomics = tokenomics_projection(
        _require_mapping(tokenomics.get("usage"), "tokenomics.usage"),
        _require_mapping(tokenomics.get("rates"), "tokenomics.rates"),
    )
    if tokenomics != expected_tokenomics:
        raise ResourceAccountingError("tokenomics scenario is internally inconsistent")


def load_validated_offline_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceAccountingError("offline report is unreadable or invalid JSON") from exc
    if not isinstance(report, dict):
        raise ResourceAccountingError("offline report root must be an object")
    validate_offline_report(report)
    return report


@dataclass(frozen=True)
class LiveGateResult:
    report_sha256: str
    projected_cost_usd: float
    available_memory_bytes: int
    free_disk_bytes: int
    checked_at: str


def validate_live_plan(plan: Mapping[str, Any], *, plan_path: Path) -> LiveGateResult:
    """Validate a live plan and current host resources before any remote call."""
    if plan.get("schema_version") != LIVE_PLAN_SCHEMA:
        raise ResourceAccountingError("unsupported live plan schema")
    report_ref = plan.get("offline_report_path")
    if not isinstance(report_ref, str) or not report_ref.strip():
        raise ResourceAccountingError("offline_report_path is required")
    report_path = Path(report_ref)
    if not report_path.is_absolute():
        report_path = (plan_path.parent / report_path).resolve()
    else:
        report_path = report_path.resolve()
    report = load_validated_offline_report(report_path)
    actual_hash = sha256_file(report_path)
    declared_hash = plan.get("offline_report_sha256")
    if declared_hash != actual_hash:
        raise ResourceAccountingError("offline report checksum mismatch")

    generated_at_text = report.get("generated_at")
    try:
        generated_at = datetime.fromisoformat(str(generated_at_text).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResourceAccountingError("offline report generated_at is invalid") from exc
    if generated_at.tzinfo is None:
        raise ResourceAccountingError("offline report generated_at must include a timezone")

    expected_host = _require_mapping(report.get("host"), "host").get("node")
    if expected_host != platform.node():
        raise ResourceAccountingError("offline report was produced on a different host")

    limits = _require_mapping(plan.get("resource_limits"), "resource_limits")
    maximum_age_hours = _finite_nonnegative(limits.get("max_report_age_hours"), "max_report_age_hours")
    age_hours = (datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours < 0 or age_hours > maximum_age_hours:
        raise ResourceAccountingError("offline report is stale or future-dated")

    measured = _require_mapping(report.get("measured"), "measured")
    latency = _require_mapping(measured.get("component_sequence_latency"), "component_sequence_latency")
    comparisons = (
        ("p95_ms", "max_component_sequence_p95_ms"),
        ("p99_ms", "max_component_sequence_p99_ms"),
        ("observed_peak_rss_bytes", "max_observed_peak_rss_bytes"),
        ("serialized_total_bytes", "max_serialized_total_bytes"),
        ("total_index_build_seconds", "max_total_index_build_seconds"),
    )
    for measured_key, limit_key in comparisons:
        observed = latency.get(measured_key) if measured_key.startswith("p") else measured.get(measured_key)
        ceiling = _finite_nonnegative(limits.get(limit_key), limit_key)
        if _finite_nonnegative(observed, measured_key) > ceiling:
            raise ResourceAccountingError(f"resource gate exceeded: {measured_key} > {limit_key}")

    min_available_memory = int(
        _finite_nonnegative(limits.get("min_available_memory_bytes"), "min_available_memory_bytes")
    )
    min_free_disk = int(_finite_nonnegative(limits.get("min_free_disk_bytes"), "min_free_disk_bytes"))
    available_memory = int(psutil.virtual_memory().available)
    free_disk = int(shutil.disk_usage(report_path.parent).free)
    if available_memory < min_available_memory:
        raise ResourceAccountingError("current available-memory gate failed")
    if free_disk < min_free_disk:
        raise ResourceAccountingError("current free-disk gate failed")

    planned_usage = _require_mapping(plan.get("planned_usage"), "planned_usage")
    usage_ceiling = _require_mapping(plan.get("usage_ceiling"), "usage_ceiling")
    usage_keys = (
        "queries",
        "embedding_calls",
        "embedding_input_tokens",
        "reranker_calls",
        "reranker_pairs",
        "reranker_input_tokens",
        "llm_calls",
        "reader_input_tokens",
        "reader_output_tokens",
        "provider_runtime_seconds",
    )
    for key in usage_keys:
        planned = planned_usage.get(key)
        ceiling = usage_ceiling.get(key)
        if not isinstance(planned, int) or planned < 0:
            raise ResourceAccountingError(f"planned_usage.{key} must be a non-negative integer")
        if not isinstance(ceiling, int) or ceiling < 0:
            raise ResourceAccountingError(f"usage_ceiling.{key} must be a non-negative integer")
        if planned > ceiling:
            raise ResourceAccountingError(f"planned usage exceeds ceiling: {key}")

    providers = plan.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ResourceAccountingError("providers must be a non-empty list")
    allowed_providers = {"tei", "runpod_llm", "zai", "research_proxy"}
    if any(provider not in allowed_providers for provider in providers):
        raise ResourceAccountingError("live plan contains an unsupported provider")
    if len(providers) != len(set(providers)):
        raise ResourceAccountingError("live plan providers must be unique")
    if "zai" in providers and "research_proxy" in providers:
        raise ResourceAccountingError("select either Z.AI or the research proxy, not both")
    if "tei" in providers and planned_usage.get("embedding_calls", 0) < 1:
        raise ResourceAccountingError("TEI preflight requires at least one planned embedding call")
    if ({"zai", "research_proxy"} & set(providers)) and planned_usage.get("llm_calls", 0) < 1:
        raise ResourceAccountingError("hosted LLM preflight requires at least one planned LLM call")

    rates = _require_mapping(plan.get("rates"), "rates")
    tokenomics = tokenomics_projection(planned_usage, rates)
    if tokenomics["pricing_complete"] is not True or tokenomics["projected_cost_usd"] is None:
        raise ResourceAccountingError("live provider usage must have complete explicit pricing")
    projected_cost = float(tokenomics["projected_cost_usd"])
    max_spend = _finite_nonnegative(plan.get("max_estimated_spend_usd"), "max_estimated_spend_usd")
    if projected_cost > max_spend:
        raise ResourceAccountingError("projected provider spend exceeds the live ceiling")

    max_wall_seconds = _finite_nonnegative(plan.get("max_wall_seconds"), "max_wall_seconds")
    if max_wall_seconds <= 0:
        raise ResourceAccountingError("max_wall_seconds must be positive")
    if plan.get("preflight_usage_included") is not True:
        raise ResourceAccountingError("plan must explicitly include preflight usage")

    return LiveGateResult(
        report_sha256=actual_hash,
        projected_cost_usd=projected_cost,
        available_memory_bytes=available_memory,
        free_disk_bytes=free_disk,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def load_and_validate_live_plan(path: Path) -> tuple[dict[str, Any], LiveGateResult]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceAccountingError("live plan is unreadable or invalid JSON") from exc
    if not isinstance(plan, dict):
        raise ResourceAccountingError("live plan root must be an object")
    result = validate_live_plan(plan, plan_path=path.resolve())
    return plan, result


def recursive_file_sizes(root: Path) -> dict[str, int]:
    """Return relative file sizes for a generated benchmark directory."""
    if not root.is_dir():
        raise ResourceAccountingError("serialization root is not a directory")
    sizes: dict[str, int] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        sizes[path.relative_to(root).as_posix()] = path.stat().st_size
    return sizes


def stable_result_hash(results: Iterable[Any]) -> str:
    encoded = json.dumps(list(results), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
