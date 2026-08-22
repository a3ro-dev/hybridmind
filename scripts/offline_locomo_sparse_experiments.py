"""Leakage-resistant, zero-provider LoCoMo sparse retrieval experiments.

The script selects a metadata-key representation using only a deterministic
conversation-level development split, evaluates the locked winner once on the
held-out conversations, and then calibrates one adaptive-k policy on development
rows before applying it once to held-out rows. Gold evidence never enters a
document, query, or runtime routing decision.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_locomo_sparse_baseline import (
    DEFAULT_DATASET,
    _TOKEN,
    _canonical_id,
    _gold_evidence,
    _sha256,
)
from storage.bm25_index import BM25SBackend


SCHEMA = "hybridmind.offline-locomo-sparse-experiment/v2"
SPLIT_SEED = "20260814"
BOOTSTRAP_SEED = 20260814
K_VALUES = (1, 5, 10, 25)
CATEGORY = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "world-knowledge",
    5: "adversarial",
}
SESSION_KEY = re.compile(r"^session_(\d+)$")
VARIANTS = (
    "raw",
    "speaker_prefix",
    "date_prefix",
    "speaker_date_prefix",
    "raw_speaker_multikey",
    "raw_speaker_date_multikey",
)


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _turn_records(item: dict) -> list[dict[str, str]]:
    sample_id = str(item.get("sample_id") or "").strip()
    conversation = item.get("conversation")
    if not sample_id or not isinstance(conversation, dict):
        raise ValueError("LoCoMo sample is missing its ID or conversation")
    session_keys = sorted(
        (key for key in conversation if SESSION_KEY.fullmatch(key)),
        key=lambda key: int(SESSION_KEY.fullmatch(key).group(1)),
    )
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for session_key in session_keys:
        session_date = str(
            conversation.get(f"{session_key}_date_time") or ""
        ).strip()
        messages = conversation.get(session_key)
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            source_id = _canonical_id(
                sample_id, message.get("dia_id") or f"{session_key}:{index}",
            )
            if source_id in seen:
                raise ValueError(f"duplicate evidence ID: {source_id}")
            seen.add(source_id)
            records.append({
                "source_id": source_id,
                "text": text,
                "speaker": str(message.get("speaker") or "").strip(),
                "date": session_date,
            })
    if not records:
        raise ValueError(f"LoCoMo sample {sample_id} has no turns")
    return records


def _keys(record: dict[str, str], variant: str) -> list[tuple[str, str]]:
    raw = record["text"]
    speaker = f"{record['speaker']}: {raw}" if record["speaker"] else raw
    dated = f"{record['date']}. {raw}" if record["date"] else raw
    speaker_dated = " ".join(
        part for part in (record["date"], record["speaker"], raw) if part
    )
    if variant == "raw":
        return [("raw", raw)]
    if variant == "speaker_prefix":
        return [("speaker", speaker)]
    if variant == "date_prefix":
        return [("date", dated)]
    if variant == "speaker_date_prefix":
        return [("speaker_date", speaker_dated)]
    if variant == "raw_speaker_multikey":
        return [("raw", raw), ("speaker", speaker)]
    if variant == "raw_speaker_date_multikey":
        return [("raw", raw), ("speaker", speaker), ("date", dated)]
    raise ValueError(f"unknown document variant: {variant}")


def _split_ids(data: list[dict], split_seed: str) -> dict[str, list[str]]:
    sample_ids = [str(item.get("sample_id") or "").strip() for item in data]
    if len(sample_ids) != len(set(sample_ids)) or any(not value for value in sample_ids):
        raise ValueError("LoCoMo sample IDs must be unique and non-empty")
    ranked = sorted(
        sample_ids,
        key=lambda value: hashlib.sha256(
            f"{split_seed}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    midpoint = len(ranked) // 2
    if midpoint == 0 or midpoint == len(ranked):
        raise ValueError("conversation split requires at least two samples")
    return {"development": ranked[:midpoint], "held_out": ranked[midpoint:]}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return float(ordered[round((len(ordered) - 1) * quantile)])


def _bootstrap_mean(
    values: list[float], *, seed: int, samples: int = 4000,
) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.sort(
        array[rng.integers(0, array.size, size=(samples, array.size))].mean(axis=1)
    )
    return {
        "mean": float(array.mean()),
        "ci95_low": float(estimates[math.floor(0.025 * (samples - 1))]),
        "ci95_high": float(estimates[math.ceil(0.975 * (samples - 1))]),
        "n": int(array.size),
        "bootstrap_samples": samples,
    }


def _aggregate(rows: list[dict], *, seed: int) -> dict:
    metrics: dict[str, Any] = {
        "n": len(rows),
        "mrr": _bootstrap_mean(
            [row["reciprocal_rank"] for row in rows], seed=seed + 1,
        ),
        "at_k": {},
        "query_latency_ms": {
            "mean": mean(row["latency_ms"] for row in rows),
            "p50": _percentile([row["latency_ms"] for row in rows], 0.50),
            "p95": _percentile([row["latency_ms"] for row in rows], 0.95),
            "p99": _percentile([row["latency_ms"] for row in rows], 0.99),
        },
    }
    for k in K_VALUES:
        metrics["at_k"][str(k)] = {
            "exact_evidence_recall": _bootstrap_mean(
                [row["recall_at_k"][str(k)] for row in rows],
                seed=seed + k * 10 + 1,
            ),
            "any_exact_evidence_hit": _bootstrap_mean(
                [row["any_hit_at_k"][str(k)] for row in rows],
                seed=seed + k * 10 + 2,
            ),
            "all_exact_evidence_hit": _bootstrap_mean(
                [row["all_hit_at_k"][str(k)] for row in rows],
                seed=seed + k * 10 + 3,
            ),
            "retrieved_source_tokens_regex_proxy": _bootstrap_mean(
                [row["tokens_at_k"][str(k)] for row in rows],
                seed=seed + k * 10 + 4,
            ),
        }
    categories: dict[str, Any] = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        categories[category] = {
            "n": len(selected),
            "exact_evidence_recall_at_10": mean(
                row["recall_at_k"]["10"] for row in selected
            ),
            "mrr": mean(row["reciprocal_rank"] for row in selected),
        }
    metrics["by_category"] = categories
    return metrics


def _evaluate_variant(
    data: list[dict], sample_ids: set[str], variant: str, *, seed: int,
) -> dict:
    rows: list[dict] = []
    index_tokens = 0
    source_tokens = 0
    source_turns = 0
    indexed_keys = 0
    exclusions = {"invalid_annotation": 0, "unresolved_evidence": 0, "no_evidence": 0}
    for item in data:
        sample_id = str(item.get("sample_id") or "").strip()
        if sample_id not in sample_ids:
            continue
        records = _turn_records(item)
        source_turns += len(records)
        record_by_id = {record["source_id"]: record for record in records}
        source_tokens += sum(len(_TOKEN.findall(record["text"])) for record in records)
        index_rows: list[tuple[str, str]] = []
        source_by_key: dict[str, str] = {}
        for record in records:
            for key_type, key_text in _keys(record, variant):
                key_id = f"{record['source_id']}|{key_type}"
                if key_id in source_by_key:
                    raise ValueError(f"duplicate retrieval key: {key_id}")
                source_by_key[key_id] = record["source_id"]
                index_rows.append((key_id, key_text))
                index_tokens += len(_TOKEN.findall(key_text))
        indexed_keys += len(index_rows)
        index = BM25SBackend()
        index.add_batch(index_rows)

        for qa_index, qa in enumerate(item.get("qa") or []):
            gold, invalid = _gold_evidence(sample_id, qa.get("evidence", []))
            if invalid:
                exclusions["invalid_annotation"] += 1
                continue
            if not gold:
                exclusions["no_evidence"] += 1
                continue
            if not gold.issubset(record_by_id):
                exclusions["unresolved_evidence"] += 1
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                raise ValueError("evidence-bearing question is empty")
            started = time.perf_counter()
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stderr(sink):
                    hits = index.search(question, top_k=len(index_rows))
            latency_ms = (time.perf_counter() - started) * 1000.0
            best_score: dict[str, float] = {}
            first_order: dict[str, int] = {}
            for order, (key_id, score) in enumerate(hits):
                source_id = source_by_key[key_id]
                numeric_score = float(score)
                first_order.setdefault(source_id, order)
                best_score[source_id] = max(
                    numeric_score, best_score.get(source_id, -math.inf),
                )
            ranked = sorted(
                best_score,
                key=lambda source_id: (-best_score[source_id], first_order[source_id]),
            )
            ranked_scores = [best_score[source_id] for source_id in ranked]
            first_rank = next(
                (position for position, source_id in enumerate(ranked, 1) if source_id in gold),
                None,
            )
            recall_at_k: dict[str, float] = {}
            any_hit_at_k: dict[str, float] = {}
            all_hit_at_k: dict[str, float] = {}
            tokens_at_k: dict[str, float] = {}
            for k in K_VALUES:
                selected = set(ranked[:k])
                covered = selected & gold
                recall_at_k[str(k)] = len(covered) / len(gold)
                any_hit_at_k[str(k)] = float(bool(covered))
                all_hit_at_k[str(k)] = float(covered == gold)
                tokens_at_k[str(k)] = float(sum(
                    len(_TOKEN.findall(record_by_id[source_id]["text"]))
                    for source_id in ranked[:k]
                ))
            rows.append({
                "question_id": hashlib.sha256(
                    f"{sample_id}\0{qa_index}\0{question}".encode("utf-8")
                ).hexdigest(),
                "sample_id": sample_id,
                "category": CATEGORY.get(qa.get("category"), "unknown"),
                "gold": sorted(gold),
                "ranked_ids_at_25": ranked[:25],
                "scores_at_25": ranked_scores[:25],
                "source_token_lengths_at_25": [
                    len(_TOKEN.findall(record_by_id[source_id]["text"]))
                    for source_id in ranked[:25]
                ],
                "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                "recall_at_k": recall_at_k,
                "any_hit_at_k": any_hit_at_k,
                "all_hit_at_k": all_hit_at_k,
                "tokens_at_k": tokens_at_k,
                "latency_ms": latency_ms,
            })
    if not rows:
        raise ValueError("split has no scorable questions")
    return {
        "variant": variant,
        "sample_ids": sorted(sample_ids),
        "source_turns": source_turns,
        "indexed_keys": indexed_keys,
        "source_tokens_regex_proxy": source_tokens,
        "index_tokens_regex_proxy": index_tokens,
        "index_token_ratio_vs_source": index_tokens / max(source_tokens, 1),
        "exclusions": exclusions,
        "metrics": _aggregate(rows, seed=seed),
        "_rows": rows,
    }


def _paired_delta(
    candidate_rows: list[dict], baseline_rows: list[dict], *, seed: int,
) -> dict:
    baseline = {row["question_id"]: row for row in baseline_rows}
    if set(baseline) != {row["question_id"] for row in candidate_rows}:
        raise ValueError("paired conditions do not contain identical questions")
    recall_deltas = [
        row["recall_at_k"]["10"]
        - baseline[row["question_id"]]["recall_at_k"]["10"]
        for row in candidate_rows
    ]
    mrr_deltas = [
        row["reciprocal_rank"] - baseline[row["question_id"]]["reciprocal_rank"]
        for row in candidate_rows
    ]
    return {
        "exact_evidence_recall_at_10": _bootstrap_mean(recall_deltas, seed=seed),
        "mrr": _bootstrap_mean(mrr_deltas, seed=seed + 1),
    }


def _select_variant(dev: dict[str, dict]) -> tuple[str, dict]:
    """Select a retrieval-quality candidate using reproducible constraints.

    The per-query timings collected by ``_evaluate_variant`` are useful
    observations, but they are sequential, single-process, sub-millisecond
    measurements.  They are not stable enough to decide which representation
    reaches held-out quality evaluation.  Resource acceptance belongs in a
    repeated, randomized benchmark; otherwise ambient system load changes the
    selected architecture while every ranked evidence ID remains identical.
    """
    raw = dev["raw"]
    raw_tokens = raw["index_tokens_regex_proxy"]
    raw_p95 = raw["metrics"]["query_latency_ms"]["p95"]
    eligible: list[str] = []
    rejected: dict[str, list[str]] = {}
    for name, result in dev.items():
        reasons = []
        if result["index_tokens_regex_proxy"] > raw_tokens * 1.15:
            reasons.append("index token footprint exceeds +15% development gate")
        raw_categories = raw["metrics"]["by_category"]
        for category, metrics in result["metrics"]["by_category"].items():
            delta = (
                metrics["exact_evidence_recall_at_10"]
                - raw_categories[category]["exact_evidence_recall_at_10"]
            )
            if delta < -0.02:
                reasons.append(f"{category} recall@10 regresses by more than 2 points")
        if reasons:
            rejected[name] = reasons
        else:
            eligible.append(name)
    if not eligible:
        raise RuntimeError("no representation survived predeclared development gates")
    winner = max(
        eligible,
        key=lambda name: (
            dev[name]["metrics"]["at_k"]["10"]["exact_evidence_recall"]["mean"],
            dev[name]["metrics"]["mrr"]["mean"],
            -dev[name]["index_tokens_regex_proxy"],
        ),
    )
    resource_observations = {}
    for name, result in dev.items():
        p95 = result["metrics"]["query_latency_ms"]["p95"]
        resource_observations[name] = {
            "query_p95_ms": p95,
            "query_p95_delta_ms_vs_raw": p95 - raw_p95,
            "query_p95_ratio_vs_raw": p95 / max(raw_p95, 1e-12),
            "index_token_ratio_vs_raw": (
                result["index_tokens_regex_proxy"] / max(raw_tokens, 1)
            ),
        }
    return winner, {
        "eligible": eligible,
        "rejected": rejected,
        "quality_selection_criteria": [
            "maximize development exact-evidence recall@10",
            "then maximize development MRR",
            "then minimize deterministic index token footprint",
        ],
        "resource_observations": resource_observations,
        "latency_decision": {
            "applied": False,
            "reason": (
                "sequential one-shot sub-millisecond wall-clock timings are "
                "exploratory and cannot select a representation reproducibly"
            ),
            "required_follow_up": (
                "repeated randomized paired latency benchmark with warm-up and "
                "an explicit absolute measurement-resolution margin"
            ),
        },
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("bm25s", "numpy", "PyStemmer"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _adaptive_feature(row: dict) -> float:
    scores = row["scores_at_25"]
    if not scores:
        return 1.0
    top = max(abs(scores[0]), 1e-12)
    fifth = scores[4] if len(scores) >= 5 else scores[-1]
    return float(fifth / top)


def _adaptive_metrics(rows: list[dict], threshold: float) -> dict:
    recalls = []
    tokens = []
    selected_k = []
    by_category: dict[str, list[float]] = {}
    for row in rows:
        k = 5 if _adaptive_feature(row) <= threshold else 10
        selected = set(row["ranked_ids_at_25"][:k])
        recall = len(selected & set(row["gold"])) / len(row["gold"])
        recalls.append(recall)
        tokens.append(sum(row["source_token_lengths_at_25"][:k]))
        selected_k.append(k)
        by_category.setdefault(row["category"], []).append(recall)
    fixed_recall = [row["recall_at_k"]["10"] for row in rows]
    fixed_tokens = [row["tokens_at_k"]["10"] for row in rows]
    return {
        "threshold_score5_over_score1": threshold,
        "exact_evidence_recall": mean(recalls),
        "fixed_k10_exact_evidence_recall": mean(fixed_recall),
        "recall_delta": mean(recalls) - mean(fixed_recall),
        "mean_k": mean(selected_k),
        "fraction_k5": mean(float(k == 5) for k in selected_k),
        "retrieved_source_tokens_regex_proxy": mean(tokens),
        "fixed_k10_source_tokens_regex_proxy": mean(fixed_tokens),
        "source_token_reduction": 1.0 - mean(tokens) / max(mean(fixed_tokens), 1e-12),
        "by_category": {
            category: mean(values) for category, values in sorted(by_category.items())
        },
        "paired_recall_delta": _bootstrap_mean(
            [value - fixed for value, fixed in zip(recalls, fixed_recall)],
            seed=BOOTSTRAP_SEED + 900,
        ),
    }


def _calibrate_adaptive_k(rows: list[dict]) -> dict:
    features = sorted({_adaptive_feature(row) for row in rows})
    thresholds = [-math.inf, *features]
    fixed_categories: dict[str, float] = {}
    for category in {row["category"] for row in rows}:
        selected = [row for row in rows if row["category"] == category]
        fixed_categories[category] = mean(
            row["recall_at_k"]["10"] for row in selected
        )
    feasible = []
    for threshold in thresholds:
        metrics = _adaptive_metrics(rows, threshold)
        if metrics["recall_delta"] < -0.005:
            continue
        if any(
            metrics["by_category"][category] - baseline < -0.02
            for category, baseline in fixed_categories.items()
        ):
            continue
        feasible.append(metrics)
    if not feasible:
        raise RuntimeError("no adaptive-k threshold satisfies development gates")
    return max(
        feasible,
        key=lambda item: (item["source_token_reduction"], item["recall_delta"]),
    )


def _public_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "_rows"}


def run(dataset: Path, *, split_seed: str = SPLIT_SEED) -> dict:
    started = time.perf_counter()
    data = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("LoCoMo dataset must contain multiple conversations")
    split = _split_ids(data, split_seed)
    development: dict[str, dict] = {}
    for offset, variant in enumerate(VARIANTS):
        development[variant] = _evaluate_variant(
            data, set(split["development"]), variant,
            seed=BOOTSTRAP_SEED + offset * 1000,
        )
    winner, selection = _select_variant(development)

    held_out_raw = _evaluate_variant(
        data, set(split["held_out"]), "raw", seed=BOOTSTRAP_SEED + 10_000,
    )
    held_out_winner = (
        held_out_raw
        if winner == "raw"
        else _evaluate_variant(
            data, set(split["held_out"]), winner,
            seed=BOOTSTRAP_SEED + 11_000,
        )
    )
    paired = _paired_delta(
        held_out_winner["_rows"], held_out_raw["_rows"],
        seed=BOOTSTRAP_SEED + 12_000,
    )

    adaptive_dev = _calibrate_adaptive_k(development[winner]["_rows"])
    adaptive_held_out = _adaptive_metrics(
        held_out_winner["_rows"],
        adaptive_dev["threshold_score5_over_score1"],
    )

    raw_test_metrics = held_out_raw["metrics"]
    winner_test_metrics = held_out_winner["metrics"]
    raw_test_categories = raw_test_metrics["by_category"]
    category_deltas = {
        category: (
            metrics["exact_evidence_recall_at_10"]
            - raw_test_categories[category]["exact_evidence_recall_at_10"]
        )
        for category, metrics in winner_test_metrics["by_category"].items()
    }
    representation_quality_success = bool(
        paired["exact_evidence_recall_at_10"]["mean"] >= 0.01
        and min(category_deltas.values(), default=0.0) >= -0.02
    )
    representation_footprint_success = bool(
        held_out_winner["index_tokens_regex_proxy"]
        <= held_out_raw["index_tokens_regex_proxy"] * 1.15
    )
    latency_observation = {
        "decision_eligible": False,
        "raw_query_p95_ms": raw_test_metrics["query_latency_ms"]["p95"],
        "winner_query_p95_ms": winner_test_metrics["query_latency_ms"]["p95"],
        "winner_minus_raw_p95_ms": (
            winner_test_metrics["query_latency_ms"]["p95"]
            - raw_test_metrics["query_latency_ms"]["p95"]
        ),
        "winner_over_raw_p95_ratio": (
            winner_test_metrics["query_latency_ms"]["p95"]
            / max(raw_test_metrics["query_latency_ms"]["p95"], 1e-12)
        ),
        "reason": (
            "single sequential in-process timing pass; report as exploratory "
            "until a repeated randomized paired benchmark is executed"
        ),
    }
    adaptive_success = bool(
        adaptive_held_out["recall_delta"] >= -0.005
        and adaptive_held_out["source_token_reduction"] >= 0.20
        and min(
            adaptive_held_out["by_category"][category]
            - winner_test_metrics["by_category"][category]["exact_evidence_recall_at_10"]
            for category in adaptive_held_out["by_category"]
        ) >= -0.02
    )

    status = _git_output("status", "--porcelain=v1", "-z") or ""
    return {
        "schema_version": SCHEMA,
        "experiment_id": (
            "locomo-sparse-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "measured_offline_paired_conversation_split_experiment",
        "execution": {
            "external_network_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "actual_external_cost_usd": 0.0,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": {
            "dataset": {"path": str(dataset.resolve()), "sha256": _sha256(dataset)},
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree": {
                "dirty": bool(status),
                "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": BOOTSTRAP_SEED,
            "dependency_versions": _dependency_versions(),
            "logical_cpu_count": os.cpu_count(),
        },
        "information_boundary": {
            "split_unit": "whole LoCoMo conversation",
            "split_seed": split_seed,
            **split,
            "development_use": "variant selection and adaptive-k calibration only",
            "held_out_use": "one locked representation plus raw baseline and one locked adaptive policy",
            "answers_or_gold_used_by_retriever": False,
        },
        "representation_experiment": {
            "hypothesis": (
                "speaker/date retrieval keys improve exact source evidence retrieval "
                "without changing returned source-turn identity"
            ),
            "development": {
                name: _public_result(result) for name, result in development.items()
            },
            "selection": {"winner": winner, **selection},
            "held_out": {
                "raw": _public_result(held_out_raw),
                "locked_winner": _public_result(held_out_winner),
                "paired_delta_winner_minus_raw": paired,
                "category_recall_at_10_deltas": category_deltas,
                "quality_success": representation_quality_success,
                "footprint_success": representation_footprint_success,
                "predeclared_success": bool(
                    representation_quality_success
                    and representation_footprint_success
                ),
                "latency_observation": latency_observation,
                "success_criteria": {
                    "recall_at_10_absolute_gain_min": 0.01,
                    "index_token_increase_max": 0.15,
                    "per_category_recall_loss_max": 0.02,
                    "latency_gate": (
                        "not applied in this quality experiment; requires a "
                        "separate repeated randomized paired benchmark"
                    ),
                },
            },
        },
        "adaptive_k_experiment": {
            "hypothesis": (
                "the BM25 score5/score1 ratio can select k=5 or k=10 while "
                "preserving recall and reducing source context"
            ),
            "development_calibration": adaptive_dev,
            "held_out": adaptive_held_out,
            "predeclared_success": adaptive_success,
            "success_criteria": {
                "recall_noninferiority_margin": -0.005,
                "source_token_reduction_min": 0.20,
                "per_category_recall_loss_max": 0.02,
            },
        },
        "interpretation_limits": [
            "This is sparse exact-evidence retrieval, not answer accuracy.",
            "Only ten LoCoMo conversations exist; the held-out split has five clusters.",
            "Regex tokens are a context/source proxy, not model tokenizer usage.",
            "BM25S latency excludes API, network, dense, graph, reranker, and reader costs.",
            "Sequential per-query wall-clock timings are exploratory and do not select the representation.",
            "A failed predeclared gate is a rejected hypothesis, not a reason to retune on held-out rows.",
        ],
    }


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-seed", default=SPLIT_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or (
        PROJECT_ROOT / "experiments" / "results"
        / f"offline-locomo-sparse-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    result = run(args.dataset.resolve(), split_seed=args.split_seed)
    _atomic_write(output.resolve(), result)
    held_out = result["representation_experiment"]["held_out"]
    print(json.dumps({
        "output": str(output.resolve()),
        "winner": result["representation_experiment"]["selection"]["winner"],
        "held_out_recall_at_10_delta": held_out[
            "paired_delta_winner_minus_raw"
        ]["exact_evidence_recall_at_10"]["mean"],
        "representation_success": held_out["predeclared_success"],
        "adaptive_k_success": result["adaptive_k_experiment"]["predeclared_success"],
        "external_network_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
