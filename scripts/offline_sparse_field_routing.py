"""Post-hoc, strictly offline LoCoMo sparse field-routing experiment.

This experiment probes a narrow hypothesis:

    the apparent benefit of a speaker-prefixed BM25S field is concentrated on
    questions that explicitly name a speaker known from the conversation
    metadata.

The router is deliberately small and gold-independent.  For each conversation
it normalizes the question and the speaker names in that conversation.  It
selects ``speaker_prefix`` only when a complete known-speaker token sequence is
present; all other questions use ``raw``.  The raw and unconditional
speaker-prefixed baselines are evaluated alongside the routed condition.  An
RRF two-field condition is also reported as exploratory, with its extra index
and token footprint made explicit.

This is an exploratory paired conversation-split measurement, not a causal
result and not a promotion decision.  Any apparent effect requires confirmation
on an unseen dataset or a preregistered unseen split.
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
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_locomo_sparse_baseline import (  # noqa: E402
    DEFAULT_DATASET,
    _TOKEN,
    _canonical_id,
    _gold_evidence,
    _sha256,
)
from scripts.offline_locomo_sparse_experiments import (  # noqa: E402
    CATEGORY,
    _split_ids,
    _turn_records,
)
from storage.bm25_index import BM25SBackend  # noqa: E402


SCHEMA = "hybridmind.offline-sparse-field-routing/v1"
FAILED_SCHEMA = "hybridmind.offline-sparse-field-routing-failure/v1"
SPLIT_SEED = "20260822-speaker-routing"
BOOTSTRAP_SEED = 20260822
RRF_K = 60
K_VALUES = (1, 5, 10, 25)
MAX_REPORTED_IDS = 25
SESSION_KEY = re.compile(r"^session_(\d+)$")
WORD = re.compile(r"\w+", re.UNICODE)
CONDITIONS = ("raw", "speaker_prefix", "routed", "rrf_multi_field")


class ValidationError(ValueError):
    """Raised when an input artifact cannot be scored without guessing."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _git_output(*args: str) -> str | None:
    """Read local git metadata only; this never contacts a remote."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("bm25s", "numpy", "PyStemmer"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def normalize_name_tokens(text: str) -> tuple[str, ...]:
    """Normalize a query/name to Unicode case-folded word tokens.

    Matching token sequences rather than raw substrings prevents false matches
    such as the speaker ``Ann`` appearing inside ``anniversary`` while still
    allowing punctuation and case differences in a question.
    """
    if not isinstance(text, str):
        raise ValidationError("speaker names and questions must be strings")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(WORD.findall(normalized))


def normalize_query(text: str) -> str:
    """Return the canonical token-space query representation used for routing."""
    return " ".join(normalize_name_tokens(text))


# Private spelling retained for small offline callers/tests that use the
# evaluator helpers directly.
_normalize_query = normalize_query


def known_speaker_names(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return deterministic non-empty speaker metadata names."""
    names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValidationError("turn records must be mappings")
        value = record.get("speaker", "")
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValidationError("speaker metadata must be a string")
        if value.strip() and normalize_name_tokens(value):
            names.add(" ".join(normalize_name_tokens(value)))
    return tuple(sorted(names))


def mentioned_speaker_names(
    query: str, speaker_names: Iterable[str],
) -> tuple[str, ...]:
    """Return known speaker names explicitly present in ``query``.

    The function accepts only retrieval-time inputs (the query and corpus
    metadata).  It intentionally has no evidence, answer, or annotation
    parameter, which makes accidental gold-dependent routing difficult.
    """
    query_tokens = normalize_name_tokens(query)
    if not query_tokens:
        return ()
    found: set[str] = set()
    for raw_name in speaker_names:
        if not isinstance(raw_name, str):
            raise ValidationError("speaker names must be strings")
        name_tokens = normalize_name_tokens(raw_name)
        if not name_tokens or len(name_tokens) > len(query_tokens):
            continue
        width = len(name_tokens)
        if any(
            query_tokens[offset : offset + width] == name_tokens
            for offset in range(len(query_tokens) - width + 1)
        ):
            found.add(" ".join(name_tokens))
    return tuple(sorted(found))


def query_mentions_known_speaker(
    query: str, speaker_names: Iterable[str],
) -> bool:
    """Return whether ``query`` explicitly contains a known speaker name."""
    return bool(mentioned_speaker_names(query, speaker_names))


def explicitly_mentions_known_speaker(
    query: str, speaker_names: Iterable[str],
) -> bool:
    """Readable alias for :func:`query_mentions_known_speaker`."""
    return query_mentions_known_speaker(query, speaker_names)


def route_representation(query: str, speaker_names: Iterable[str]) -> str:
    """Select a sparse field from query text and metadata only."""
    return "speaker_prefix" if query_mentions_known_speaker(query, speaker_names) else "raw"


def route_query(query: str, speaker_names: Iterable[str]) -> str:
    """Alias emphasizing that routing is performed at query time."""
    return route_representation(query, speaker_names)


_route_query = route_query


def select_sparse_field(query: str, speaker_names: Iterable[str]) -> str:
    """Compatibility alias for callers that prefer a selection verb."""
    return route_representation(query, speaker_names)


def _question_id(sample_id: str, qa_index: int, question: str) -> str:
    return hashlib.sha256(
        f"{sample_id}\0{qa_index}\0{question}".encode("utf-8")
    ).hexdigest()


def _field_text(record: Mapping[str, str], field: str) -> str:
    raw = record["text"]
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError("turn text must be a non-empty string")
    if field == "raw":
        return raw
    if field == "speaker_prefix":
        speaker = record.get("speaker", "")
        if not isinstance(speaker, str):
            raise ValidationError("speaker metadata must be a string")
        return f"{speaker}: {raw}" if speaker else raw
    raise ValidationError(f"unknown sparse field: {field}")


def _build_field_index(
    records: Sequence[Mapping[str, str]], field: str,
) -> tuple[BM25SBackend, dict[str, str], dict[str, Any]]:
    """Build one source-preserving BM25S field and return footprint metadata."""
    if field not in ("raw", "speaker_prefix"):
        raise ValidationError(f"unknown sparse field: {field}")
    rows: list[tuple[str, str]] = []
    key_to_source: dict[str, str] = {}
    index_tokens = 0
    source_tokens = 0
    for record in records:
        source_id = str(record.get("source_id") or "").strip()
        text = str(record.get("text") or "").strip()
        if not source_id or not text:
            raise ValidationError("every turn needs a non-empty source_id and text")
        key_id = f"{source_id}|{field}"
        if key_id in key_to_source:
            raise ValidationError(f"duplicate sparse retrieval key: {key_id}")
        key_text = _field_text(record, field)
        key_to_source[key_id] = source_id
        rows.append((key_id, key_text))
        index_tokens += len(_TOKEN.findall(key_text))
        source_tokens += len(_TOKEN.findall(text))
    if not rows:
        raise ValidationError("cannot build a sparse index over zero turns")
    index = BM25SBackend()
    index.add_batch(rows)
    return index, key_to_source, {
        "field": field,
        "indexed_keys": len(rows),
        "source_turns": len(records),
        "source_tokens_regex_proxy": source_tokens,
        "index_tokens_regex_proxy": index_tokens,
    }


def _search_field(
    index: BM25SBackend,
    key_to_source: Mapping[str, str],
    query: str,
    *,
    top_k: int,
) -> tuple[list[str], list[float], float]:
    started = time.perf_counter()
    # bm25s versions used by HybridMind can emit progress diagnostics on
    # stderr.  Suppress those diagnostics without touching provider/network IO.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stderr(sink):
            hits = index.search(query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    best_score: dict[str, float] = {}
    first_order: dict[str, int] = {}
    for order, hit in enumerate(hits):
        if not isinstance(hit, (tuple, list)) or len(hit) != 2:
            raise ValidationError("BM25S returned a malformed hit")
        key_id, raw_score = hit
        source_id = key_to_source.get(str(key_id))
        if source_id is None:
            raise ValidationError("BM25S returned an unknown retrieval key")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValidationError("BM25S returned a non-finite score")
        first_order.setdefault(source_id, order)
        best_score[source_id] = max(score, best_score.get(source_id, -math.inf))
    ranked = sorted(
        best_score,
        key=lambda source_id: (-best_score[source_id], first_order[source_id]),
    )
    return ranked[:top_k], [best_score[source_id] for source_id in ranked[:top_k]], elapsed_ms


def rrf_fuse(
    raw_ranked_ids: Sequence[str],
    speaker_ranked_ids: Sequence[str],
    *,
    rrf_k: int = RRF_K,
    top_k: int = MAX_REPORTED_IDS,
) -> tuple[list[str], list[float]]:
    """Fuse two source-ID rank lists with deterministic reciprocal rank fusion."""
    if not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValidationError("rrf_k must be a positive integer")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranked in (raw_ranked_ids, speaker_ranked_ids):
        seen: set[str] = set()
        for rank, source_id in enumerate(ranked, 1):
            source_id = str(source_id)
            if not source_id or source_id in seen:
                if not source_id:
                    raise ValidationError("RRF received an empty source ID")
                continue
            seen.add(source_id)
            first_seen.setdefault(source_id, order)
            order += 1
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores, key=lambda source_id: (-scores[source_id], first_seen[source_id]))
    ranked = ranked[:top_k]
    return ranked, [scores[source_id] for source_id in ranked]


def _metric_bootstrap(
    values: Sequence[float], *, seed: int, samples: int = 1000,
) -> dict[str, Any]:
    if not values:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "n": 0,
            "bootstrap_samples": samples,
        }
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


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return float(ordered[round((len(ordered) - 1) * quantile)])


def _score_ranking(
    ranked_ids: Sequence[str],
    ranked_scores: Sequence[float],
    gold: set[str],
    record_by_id: Mapping[str, Mapping[str, str]],
    latency_ms: float,
) -> dict[str, Any]:
    ranked = [str(source_id) for source_id in ranked_ids]
    if len(ranked) != len(set(ranked)):
        raise ValidationError("ranking contains duplicate source IDs")
    missing = [source_id for source_id in ranked if source_id not in record_by_id]
    if missing:
        raise ValidationError(f"ranking contains unknown source IDs: {missing[:3]}")
    first_rank = next(
        (rank for rank, source_id in enumerate(ranked, 1) if source_id in gold),
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
    return {
        "ranked_evidence_ids_at_25": ranked[:MAX_REPORTED_IDS],
        "scores_at_25": [float(score) for score in ranked_scores[:MAX_REPORTED_IDS]],
        "first_exact_evidence_rank": first_rank,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "recall_at_k": recall_at_k,
        "any_hit_at_k": any_hit_at_k,
        "all_hit_at_k": all_hit_at_k,
        "tokens_at_k": tokens_at_k,
        "latency_ms": float(latency_ms),
    }


def _condition_metrics(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValidationError("split has no scorable evidence-bearing questions")

    def metric_block(selected: Sequence[Mapping[str, Any]], seed_offset: int) -> dict[str, Any]:
        selected = list(selected)
        return {
            "n": len(selected),
            "mrr_first_exact_evidence": _metric_bootstrap(
                [float(row["reciprocal_rank"]) for row in selected],
                seed=seed + seed_offset + 1,
            ),
            "at_k": {
                str(k): {
                    "exact_evidence_recall": _metric_bootstrap(
                        [float(row["recall_at_k"][str(k)]) for row in selected],
                        seed=seed + seed_offset + k * 10 + 1,
                    ),
                    "any_exact_evidence_hit": _metric_bootstrap(
                        [float(row["any_hit_at_k"][str(k)]) for row in selected],
                        seed=seed + seed_offset + k * 10 + 2,
                    ),
                    "all_exact_evidence_hit": _metric_bootstrap(
                        [float(row["all_hit_at_k"][str(k)]) for row in selected],
                        seed=seed + seed_offset + k * 10 + 3,
                    ),
                    "retrieved_source_tokens_regex_proxy": _metric_bootstrap(
                        [float(row["tokens_at_k"][str(k)]) for row in selected],
                        seed=seed + seed_offset + k * 10 + 4,
                    ),
                }
                for k in K_VALUES
            },
            "query_latency_ms": {
                "mean": mean(float(row["latency_ms"]) for row in selected),
                "p50": _percentile([float(row["latency_ms"]) for row in selected], 0.50),
                "p95": _percentile([float(row["latency_ms"]) for row in selected], 0.95),
                "p99": _percentile([float(row["latency_ms"]) for row in selected], 0.99),
            },
        }

    metrics = metric_block(rows, 0)
    metrics["by_category"] = {
        category: metric_block(
            [row for row in rows if row["category"] == category],
            index * 100,
        )
        for index, category in enumerate(sorted({str(row["category"]) for row in rows}))
    }
    speaker_strata = {
        "true" if mentioned else "false": metric_block(
            [row for row in rows if bool(row["speaker_mentioned"]) is mentioned],
            1000 + int(mentioned) * 100,
        )
        for mentioned in (False, True)
        if any(bool(row["speaker_mentioned"]) is mentioned for row in rows)
    }
    metrics["by_speaker_mentioned"] = speaker_strata
    # Match the concise spelling used by the companion sparse failure report.
    metrics["by_speaker_mention"] = speaker_strata
    return metrics


def _status(a: float, b: float, *, lower_is_better: bool = False) -> str:
    if lower_is_better:
        if b < a:
            return "improved"
        if b > a:
            return "regressed"
        return "unchanged"
    if b > a:
        return "improved"
    if b < a:
        return "regressed"
    return "unchanged"


def _transition_block(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {str(row["question_id"]): row for row in baseline_rows}
    candidate = {str(row["question_id"]): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValidationError("paired conditions contain different questions")

    def summarize(selected_ids: Sequence[str]) -> dict[str, Any]:
        values = [
            (
                _status(
                    float(baseline[qid]["recall_at_k"]["10"]),
                    float(candidate[qid]["recall_at_k"]["10"]),
                ),
                _status(
                    float(baseline[qid]["reciprocal_rank"]),
                    float(candidate[qid]["reciprocal_rank"]),
                ),
            )
            for qid in selected_ids
        ]
        return {
            "n": len(values),
            "recall_at_10": {
                status: sum(item[0] == status for item in values)
                for status in ("improved", "regressed", "unchanged")
            },
            "mrr": {
                status: sum(item[1] == status for item in values)
                for status in ("improved", "regressed", "unchanged")
            },
        }

    all_ids = sorted(candidate)
    by_speaker = {
        "true" if mentioned else "false": summarize([
            qid for qid in all_ids
            if bool(candidate[qid]["speaker_mentioned"]) is mentioned
        ])
        for mentioned in (False, True)
        if any(bool(candidate[qid]["speaker_mentioned"]) is mentioned for qid in all_ids)
    }
    by_category = {
        category: summarize([
            qid for qid in all_ids if candidate[qid]["category"] == category
        ])
        for category in sorted({str(candidate[qid]["category"]) for qid in all_ids})
    }

    paired_changes = []
    for qid in all_ids:
        baseline_row = baseline[qid]
        candidate_row = candidate[qid]
        baseline_recall = float(baseline_row["recall_at_k"]["10"])
        candidate_recall = float(candidate_row["recall_at_k"]["10"])
        baseline_mrr = float(baseline_row["reciprocal_rank"])
        candidate_mrr = float(candidate_row["reciprocal_rank"])
        paired_changes.append({
            "question_id": qid,
            "sample_id": candidate_row["sample_id"],
            "category": candidate_row["category"],
            "speaker_mentioned": bool(candidate_row["speaker_mentioned"]),
            "baseline_recall_at_10": baseline_recall,
            "candidate_recall_at_10": candidate_recall,
            "recall_delta_at_10": candidate_recall - baseline_recall,
            "recall_status": _status(baseline_recall, candidate_recall),
            "baseline_mrr": baseline_mrr,
            "candidate_mrr": candidate_mrr,
            "mrr_delta": candidate_mrr - baseline_mrr,
            "mrr_status": _status(baseline_mrr, candidate_mrr),
            "baseline_first_exact_evidence_rank": baseline_row[
                "first_exact_evidence_rank"
            ],
            "candidate_first_exact_evidence_rank": candidate_row[
                "first_exact_evidence_rank"
            ],
        })

    recall_deltas = [
        float(candidate[qid]["recall_at_k"]["10"])
        - float(baseline[qid]["recall_at_k"]["10"])
        for qid in all_ids
    ]
    mrr_deltas = [
        float(candidate[qid]["reciprocal_rank"])
        - float(baseline[qid]["reciprocal_rank"])
        for qid in all_ids
    ]
    return {
        "n": len(all_ids),
        "paired_delta": {
            "exact_evidence_recall_at_10": _metric_bootstrap(
                recall_deltas, seed=BOOTSTRAP_SEED + 500,
            ),
            "mrr": _metric_bootstrap(mrr_deltas, seed=BOOTSTRAP_SEED + 501),
        },
        "question_level": summarize(all_ids),
        "paired_changes": paired_changes,
        "by_speaker_mentioned": by_speaker,
        "by_speaker_mention": by_speaker,
        "by_category": by_category,
    }


def _retrieval_input_fingerprint(
    data: Sequence[Mapping[str, Any]], sample_ids: Iterable[str],
) -> str:
    """Hash only documents, metadata, questions, and routing inputs.

    ``answer`` and ``evidence`` are deliberately never read in this function.
    The resulting digest is an immutable audit boundary for gold-independent
    retrieval and routing.
    """
    selected = set(sample_ids)
    payload: list[dict[str, Any]] = []
    for item in sorted(data, key=lambda row: str(row.get("sample_id") or "")):
        sample_id = str(item.get("sample_id") or "").strip()
        if sample_id not in selected:
            continue
        records = _turn_records(dict(item))
        speakers = known_speaker_names(records)
        payload.append({
            "sample_id": sample_id,
            "turns": [
                {
                    "source_id": record["source_id"],
                    "text": record["text"],
                    "speaker": record["speaker"],
                    "date": record["date"],
                }
                for record in records
            ],
            "known_speaker_names": list(speakers),
            "questions": [
                {
                    "question_id": _question_id(sample_id, index, question),
                    "question": question,
                    "normalized_query": normalize_query(question),
                    "mentioned_speakers": list(mentioned_speaker_names(question, speakers)),
                }
                for index, qa in enumerate(item.get("qa") or [])
                if isinstance(qa, Mapping)
                for question in [str(qa.get("question") or "").strip()]
            ],
        })
    return _value_sha256(payload)


def _footprint(
    raw_meta: Mapping[str, Any], speaker_meta: Mapping[str, Any], condition: str,
) -> dict[str, Any]:
    raw_tokens = int(raw_meta["index_tokens_regex_proxy"])
    speaker_tokens = int(speaker_meta["index_tokens_regex_proxy"])
    raw_keys = int(raw_meta["indexed_keys"])
    speaker_keys = int(speaker_meta["indexed_keys"])
    if condition == "raw":
        fields = ["raw"]
        keys = raw_keys
        tokens = raw_tokens
    elif condition == "speaker_prefix":
        fields = ["speaker_prefix"]
        keys = speaker_keys
        tokens = speaker_tokens
    elif condition in ("routed", "rrf_multi_field"):
        fields = ["raw", "speaker_prefix"]
        keys = raw_keys + speaker_keys
        tokens = raw_tokens + speaker_tokens
    else:
        raise ValidationError(f"unknown condition: {condition}")
    return {
        "fields": fields,
        "field_count": len(fields),
        "indexed_keys": keys,
        "index_tokens_regex_proxy": tokens,
        "index_token_ratio_vs_raw": tokens / max(raw_tokens, 1),
        "requires_two_indexes": len(fields) == 2,
        "doubled_index_token_footprint_note": (
            "This condition retains raw and speaker-prefix indexes; token counts "
            "include both fields and therefore exceed the raw baseline."
            if len(fields) == 2 else None
        ),
    }


def _validate_and_load(dataset: Path) -> tuple[list[dict], str]:
    if not isinstance(dataset, Path):
        raise ValidationError("dataset must be a pathlib.Path")
    if not dataset.exists() or not dataset.is_file():
        raise ValidationError(f"dataset does not exist: {dataset}")
    raw = dataset.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"dataset is not valid UTF-8 JSON: {dataset}") from exc
    if not isinstance(data, list) or len(data) < 2:
        raise ValidationError("LoCoMo dataset must contain at least two conversations")
    sample_ids: set[str] = set()
    question_ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValidationError("each LoCoMo conversation must be an object")
        sample_id = str(item.get("sample_id") or "").strip()
        if not sample_id or sample_id in sample_ids:
            raise ValidationError("LoCoMo sample IDs must be unique and non-empty")
        sample_ids.add(sample_id)
        records = _turn_records(item)
        if not isinstance(item.get("qa"), list):
            raise ValidationError(f"LoCoMo sample {sample_id} has no qa array")
        for qa_index, qa in enumerate(item["qa"]):
            if not isinstance(qa, Mapping):
                raise ValidationError(f"LoCoMo sample {sample_id} has malformed QA row")
            question = str(qa.get("question") or "").strip()
            if not question:
                raise ValidationError(f"LoCoMo sample {sample_id} has an empty question")
            question_id = _question_id(sample_id, qa_index, question)
            if question_id in question_ids:
                raise ValidationError(f"duplicate question ID: {question_id}")
            question_ids.add(question_id)
        # Trigger metadata validation before any evaluation work.
        known_speaker_names(records)
    return data, hashlib.sha256(raw).hexdigest()


def _evaluate_split(
    data: Sequence[Mapping[str, Any]],
    sample_ids: set[str],
    *,
    split_name: str,
    seed: int,
    strict_evidence: bool,
) -> dict[str, Any]:
    rows_by_condition: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}
    exclusions = {
        "invalid_annotation": 0,
        "unresolved_evidence": 0,
        "no_evidence": 0,
    }
    failure_rows: list[dict[str, Any]] = []
    routing_counts = {"raw": 0, "speaker_prefix": 0}
    source_turns = 0
    source_tokens = 0
    raw_meta_total = {"indexed_keys": 0, "index_tokens_regex_proxy": 0}
    speaker_meta_total = {"indexed_keys": 0, "index_tokens_regex_proxy": 0}
    question_rows: list[dict[str, Any]] = []

    for item in data:
        sample_id = str(item.get("sample_id") or "").strip()
        if sample_id not in sample_ids:
            continue
        records = _turn_records(dict(item))
        record_by_id = {record["source_id"]: record for record in records}
        source_turns += len(records)
        source_tokens += sum(len(_TOKEN.findall(record["text"])) for record in records)
        raw_index, raw_map, raw_meta = _build_field_index(records, "raw")
        speaker_index, speaker_map, speaker_meta = _build_field_index(records, "speaker_prefix")
        raw_meta_total["indexed_keys"] += int(raw_meta["indexed_keys"])
        raw_meta_total["index_tokens_regex_proxy"] += int(raw_meta["index_tokens_regex_proxy"])
        speaker_meta_total["indexed_keys"] += int(speaker_meta["indexed_keys"])
        speaker_meta_total["index_tokens_regex_proxy"] += int(speaker_meta["index_tokens_regex_proxy"])
        speakers = known_speaker_names(records)
        qa_rows = item.get("qa")
        if not isinstance(qa_rows, list):
            raise ValidationError(f"LoCoMo sample {sample_id} has no qa array")

        for qa_index, qa in enumerate(qa_rows):
            if not isinstance(qa, Mapping):
                raise ValidationError(f"LoCoMo sample {sample_id} has malformed QA row")
            question = str(qa.get("question") or "").strip()
            # The routing decision is made before reading evidence/answers.  Do
            # not move this below _gold_evidence: that would weaken the audit
            # boundary even though the final metric is paired with gold.
            mentioned = mentioned_speaker_names(question, speakers)
            routed_field = "speaker_prefix" if mentioned else "raw"
            routing_counts[routed_field] += 1
            qid = _question_id(sample_id, qa_index, question)

            gold, invalid = _gold_evidence(sample_id, qa.get("evidence", []))
            if invalid:
                exclusions["invalid_annotation"] += 1
                failure_rows.append({
                    "question_id": qid,
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "classification": "failed_invalid_annotation",
                    "details": sorted(str(value) for value in invalid),
                })
                if strict_evidence:
                    raise ValidationError(
                        f"invalid evidence annotation in {sample_id}/{qa_index}: {invalid}"
                    )
                continue
            if not gold:
                exclusions["no_evidence"] += 1
                failure_rows.append({
                    "question_id": qid,
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "classification": "failed_no_evidence",
                    "details": [],
                })
                continue
            missing = gold - set(record_by_id)
            if missing:
                exclusions["unresolved_evidence"] += 1
                failure_rows.append({
                    "question_id": qid,
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "classification": "failed_unresolved_evidence",
                    "details": sorted(missing),
                })
                if strict_evidence:
                    raise ValidationError(
                        f"unresolved evidence IDs in {sample_id}/{qa_index}: {sorted(missing)}"
                    )
                continue

            top_k = max(MAX_REPORTED_IDS, len(records))
            raw_ranked, raw_scores, raw_latency = _search_field(
                raw_index, raw_map, question, top_k=top_k,
            )
            speaker_ranked, speaker_scores, speaker_latency = _search_field(
                speaker_index, speaker_map, question, top_k=top_k,
            )
            rrf_ranked, rrf_scores = rrf_fuse(
                raw_ranked, speaker_ranked, top_k=MAX_REPORTED_IDS,
            )
            rank_by_condition = {
                "raw": (raw_ranked, raw_scores, raw_latency),
                "speaker_prefix": (speaker_ranked, speaker_scores, speaker_latency),
                "routed": (
                    speaker_ranked if routed_field == "speaker_prefix" else raw_ranked,
                    speaker_scores if routed_field == "speaker_prefix" else raw_scores,
                    speaker_latency if routed_field == "speaker_prefix" else raw_latency,
                ),
                "rrf_multi_field": (
                    rrf_ranked, rrf_scores, raw_latency + speaker_latency,
                ),
            }
            scored: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                ranked, scores, latency = rank_by_condition[condition]
                scored[condition] = _score_ranking(
                    ranked, scores, gold, record_by_id, latency,
                )
                scored[condition]["condition"] = condition
                rows_by_condition[condition].append({
                    "question_id": qid,
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "category": CATEGORY.get(qa.get("category"), "unknown"),
                    "speaker_mentioned": bool(mentioned),
                    "mentioned_speakers": list(mentioned),
                    "known_speaker_names": list(speakers),
                    "gold_evidence_ids": sorted(gold),
                    **scored[condition],
                })
            question_rows.append({
                "question_id": qid,
                "sample_id": sample_id,
                "qa_index": qa_index,
                "category": CATEGORY.get(qa.get("category"), "unknown"),
                "normalized_query": normalize_query(question),
                "speaker_mentioned": bool(mentioned),
                "mentioned_speakers": list(mentioned),
                "known_speaker_names": list(speakers),
                "routed_field": routed_field,
                "gold_evidence_ids": sorted(gold),
                "conditions": {
                    condition: {
                        "ranked_evidence_ids_at_25": scored[condition]["ranked_evidence_ids_at_25"],
                        "first_exact_evidence_rank": scored[condition]["first_exact_evidence_rank"],
                        "reciprocal_rank": scored[condition]["reciprocal_rank"],
                        "recall_at_k": scored[condition]["recall_at_k"],
                    }
                    for condition in CONDITIONS
                },
            })

    if not rows_by_condition["raw"]:
        raise ValidationError(f"{split_name} split has no scorable questions")
    metrics = {
        condition: _condition_metrics(rows_by_condition[condition], seed=seed + offset * 1000)
        for offset, condition in enumerate(CONDITIONS)
    }
    comparisons = {
        condition: _transition_block(
            rows_by_condition["raw"], rows_by_condition[condition],
        )
        for condition in CONDITIONS
        if condition != "raw"
    }
    raw_meta_total["source_turns"] = source_turns
    raw_meta_total["source_tokens_regex_proxy"] = source_tokens
    speaker_meta_total["source_turns"] = source_turns
    speaker_meta_total["source_tokens_regex_proxy"] = source_tokens
    footprints = {
        condition: _footprint(raw_meta_total, speaker_meta_total, condition)
        for condition in CONDITIONS
    }
    # Keep each row's gold and exact ranked source IDs in the report.  This is
    # intentionally an evidence-ID ledger, never an answer-overlap metric.
    return {
        "split_name": split_name,
        "sample_ids": sorted(sample_ids),
        "source_turns": source_turns,
        "source_tokens_regex_proxy": source_tokens,
        "exclusions": exclusions,
        "failure_rows": failure_rows,
        "routing": {
            "rule": "speaker_prefix iff normalized query contains a complete known speaker name; otherwise raw",
            "known_speakers_from_corpus_metadata_only": True,
            "answers_or_gold_used_for_routing": False,
            "question_count_before_evidence_exclusions": sum(routing_counts.values()),
            "selected_field_counts": routing_counts,
            "speaker_mentioned_count": routing_counts["speaker_prefix"],
            "speaker_not_mentioned_count": routing_counts["raw"],
        },
        "conditions": {
            condition: {
                "classification": (
                    "exploratory_multi_field_rrf"
                    if condition == "rrf_multi_field"
                    else "offline_exact_evidence_condition"
                ),
                "selection_eligible": False,
                "metrics": metrics[condition],
                "footprint": footprints[condition],
                "question_rows": rows_by_condition[condition],
            }
            for condition in CONDITIONS
        },
        "paired_comparisons_vs_raw": comparisons,
        "question_level_transitions": comparisons,
        "question_rows": question_rows,
    }


def _public_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "conditions": list(CONDITIONS),
        "router": {
            "normalization": "Unicode NFKC + casefold + Unicode word-token sequence",
            "match": "complete contiguous known-speaker token sequence",
            "gold_independent": True,
            "selected_field": "speaker_prefix iff explicit known speaker mention, else raw",
        },
        "rrf": {"enabled": True, "k": RRF_K, "classification": "exploratory_only"},
        "metrics": {"k_values": list(K_VALUES), "evidence_identity": "canonical exact evidence IDs"},
        "split": {"unit": "whole conversation", "development_and_held_out": True},
        "promotion": {
            "select_or_promote_on_this_dataset": False,
            "status": "post_hoc_exploratory",
            "confirmatory_requirement": "unseen dataset or preregistered unseen split",
        },
    }


def run(
    dataset: Path = DEFAULT_DATASET,
    *,
    split_seed: str = SPLIT_SEED,
    strict_evidence: bool = False,
) -> dict[str, Any]:
    """Run both conversation splits without provider or network calls."""
    started = time.perf_counter()
    dataset = Path(dataset).resolve()
    data, dataset_sha256 = _validate_and_load(dataset)
    split = _split_ids(data, split_seed)
    config = _public_config()
    config["split"]["split_seed"] = split_seed
    development = _evaluate_split(
        data,
        set(split["development"]),
        split_name="development",
        seed=BOOTSTRAP_SEED,
        strict_evidence=strict_evidence,
    )
    held_out = _evaluate_split(
        data,
        set(split["held_out"]),
        split_name="held_out",
        seed=BOOTSTRAP_SEED + 10_000,
        strict_evidence=strict_evidence,
    )
    source_path = Path(__file__).resolve()
    worktree_status = _git_output("status", "--porcelain=v1", "-z") or ""
    provenance = {
        "dataset": {"path": str(dataset), "sha256": dataset_sha256},
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_worktree": {
            "dirty": bool(worktree_status),
            "status_sha256": hashlib.sha256(worktree_status.encode("utf-8")).hexdigest(),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "logical_cpu_count": os.cpu_count(),
        "split_seed": split_seed,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "config_sha256": _value_sha256(config),
        "retrieval_input_sha256": {
            "development": _retrieval_input_fingerprint(data, split["development"]),
            "held_out": _retrieval_input_fingerprint(data, split["held_out"]),
        },
    }
    provenance["manifest_sha256"] = _value_sha256(provenance)
    result = {
        "schema_version": SCHEMA,
        "experiment_id": "locomo-sparse-field-routing-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "post_hoc_exploratory_offline_sparse_field_routing",
        "hypothesis": (
            "speaker-prefix benefit is concentrated when the normalized query "
            "explicitly names a known speaker from corpus metadata"
        ),
        "execution": {
            "strictly_offline": True,
            "external_network_calls": 0,
            "provider_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "actual_external_cost_usd": 0.0,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": provenance,
        "information_boundary": {
            "split_unit": "whole LoCoMo conversation",
            "split_seed": split_seed,
            "development": split["development"],
            "held_out": split["held_out"],
            "answers_or_gold_used_by_retriever": False,
            "answers_or_gold_used_by_router": False,
            "routing_inputs": ["question text", "speaker metadata in the same corpus conversation"],
            "gold_used_only_after_ranked_source_ids": True,
        },
        "experimental_status": {
            "status": "post_hoc_exploratory",
            "selection_or_promotion_performed": False,
            "selection_or_promotion_forbidden": True,
            "requires_unseen_confirmatory_dataset_or_split": True,
            "causal_claim_supported": False,
            "rrf_multi_field_is_exploratory": True,
        },
        "configuration": config,
        "development": development,
        "held_out": held_out,
        "splits": {"development": development, "held_out": held_out},
        "interpretation_limits": [
            "This is exact-evidence sparse retrieval, not answer accuracy.",
            "The split and strata are reported for exploration; no condition is selected or promoted.",
            "A speaker mention is a metadata/name token match, not proof of causal benefit.",
            "Routed and RRF conditions retain two sparse fields; their index token footprint is not raw-equivalent.",
            "Regex token counts are a source/context proxy, not model tokenizer usage.",
            "Any apparent effect requires confirmation on an unseen dataset or preregistered unseen split.",
        ],
    }
    return result


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a JSON artifact atomically without overwriting a receipt.

    ``os.replace`` would make a rerun silently destroy an existing immutable
    receipt.  A same-directory hard-link is the create-once commit point: it is
    atomic and fails if the destination already exists.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing immutable artifact: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    """Backward-compatible private alias used by script callers."""
    atomic_write_json(path, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-seed", default=SPLIT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-evidence",
        action="store_true",
        help="abort on malformed or unresolved evidence annotations instead of recording exclusions",
    )
    args = parser.parse_args(argv)
    output = args.output or (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / f"offline-sparse-field-routing-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    try:
        result = run(
            args.dataset,
            split_seed=args.split_seed,
            strict_evidence=args.strict_evidence,
        )
    except ValidationError as exc:
        dataset_path = args.dataset.resolve()
        source_path = Path(__file__).resolve()
        receipt = {
            "schema_version": FAILED_SCHEMA,
            "classification": "failed_validation",
            "status": "failed",
            "reason": str(exc),
            "dataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path) if dataset_path.is_file() else None,
            },
            "configuration": {
                "split_seed": args.split_seed,
                "strict_evidence": args.strict_evidence,
            },
            "execution": {
                "strictly_offline": True,
                "external_network_calls": 0,
                "provider_calls": 0,
                "embedding_calls": 0,
                "reranker_calls": 0,
                "reader_calls": 0,
                "actual_external_cost_usd": 0.0,
            },
            "provenance": {
                "source": {"path": str(source_path), "sha256": _sha256(source_path)},
                "git_commit": _git_output("rev-parse", "HEAD"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependency_versions": _dependency_versions(),
            },
            "claim_boundary": "No comparative retrieval result is valid for this receipt.",
        }
        atomic_write_json(output, receipt)
        print(json.dumps({
            "output": str(Path(output).resolve()),
            "classification": receipt["classification"],
            "status": "failed",
        }, sort_keys=True))
        return 2
    atomic_write_json(output, result)
    print(json.dumps({
        "output": str(Path(output).resolve()),
        "classification": result["classification"],
        "held_out_speaker_mentioned": result["held_out"]["routing"]["speaker_mentioned_count"],
        "held_out_routed_vs_raw_recall_at_10_delta": result["held_out"][
            "paired_comparisons_vs_raw"
        ]["routed"]["paired_delta"]["exact_evidence_recall_at_10"]["mean"],
        "external_network_calls": result["execution"]["external_network_calls"],
        "provider_calls": result["execution"]["provider_calls"],
        "selection_or_promotion_performed": result["experimental_status"][
            "selection_or_promotion_performed"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
