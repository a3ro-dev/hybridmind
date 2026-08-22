"""Strictly offline LongMemEval-S support-session retrieval benchmark.

This benchmark measures lexical retrieval only.  It compares a conventional
BM25S index over one concatenated document per session with turn-level BM25S
indexes whose ranked turns are aggregated back to session IDs without using
the gold labels.  The corpus for every question is that question's own
``haystack_sessions`` value; the answer text is deliberately never read.

The measured target is *support-session retrieval*.  ``answer_session_ids``
are session-level labels, not exact turn evidence.  This script therefore
does not claim exact-turn evidence recall, answer accuracy, a reader result,
or SOTA.  Conditions are reported as an exploratory comparison on the same
dataset; no winner is selected.

The script has no provider, embedding, reader, or network path.  It reads the
checked-in JSON, builds in-memory BM25S indexes, scores, and publishes one
create-once JSON report through an exclusive temporary file and atomic link.
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.bm25_index import BM25SBackend  # noqa: E402


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "memorybench"
    / "data"
    / "benchmarks"
    / "longmemeval"
    / "longmemeval_s.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "offline-longmemeval-session-retrieval-20260822.json"
)

SCHEMA = "hybridmind.offline-longmemeval-session-retrieval/v1"
FAILED_SCHEMA = "hybridmind.offline-longmemeval-session-retrieval-failure/v1"
CONDITIONS = ("whole_session", "turn_max", "turn_rrf")
METRIC_KS = (1, 5, 10)
FIXED_TOP_K = 10
RRF_K = 60
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_MARKER_RE = re.compile(
    r"(?ix)"
    r"(?:^|[\s\[])(?:has_answer|is_answer|gold|support_session)"
    r"\s*[:=]\s*(?:true|false|yes|no|1|0)\s*(?:\]|(?=\s|$))"
)


class DatasetNotRetrievalEvaluableError(ValueError):
    """Raised when a corpus contains no distractors at the declared cutoff."""

    def __init__(self, message: str, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = dict(audit)


@dataclass(frozen=True)
class Turn:
    """Validated source turn, retaining only fields allowed into retrieval."""

    session_id: str
    turn_index: int
    role: str
    content: str


@dataclass(frozen=True)
class Document:
    """An immutable retrieval document and its source session provenance."""

    document_id: str
    session_id: str
    text: str
    turn_index: int | None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{field} contains an invalid identifier: {value!r}")
    return value


def _clean_content(value: object, *, item_index: int, session_index: int, turn_index: int) -> str:
    """Return source content with known leaked answer markers removed.

    LongMemEval-S stores the answer marker as a separate ``has_answer`` field.
    That field is never copied into a document.  The small sanitizer also
    handles synthetic/legacy corpora that accidentally rendered that marker
    into content, while leaving ordinary words such as ``gold`` untouched.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "haystack turn content must be a non-empty string "
            f"(item={item_index}, session={session_index}, turn={turn_index})"
        )
    content = value.strip()
    content = _MARKER_RE.sub(" ", content)
    content = re.sub(r"[ \t]{2,}", " ", content)
    if not content.strip():
        raise ValueError(
            "haystack turn content is empty after leaked-marker removal "
            f"(item={item_index}, session={session_index}, turn={turn_index})"
        )
    return content.strip()


def _validate_example(item: object, *, item_index: int) -> dict[str, Any]:
    """Validate one example and return a retrieval-only immutable copy.

    The returned structure intentionally contains no answer text and no
    ``has_answer`` marker.  Gold session IDs are retained solely in a separate
    field for post-retrieval scoring.
    """

    if not isinstance(item, dict):
        raise ValueError(f"LongMemEval example {item_index} must be an object")

    question_id = _safe_identifier(item.get("question_id"), "question_id")
    question_type = item.get("question_type")
    if not isinstance(question_type, str) or not question_type.strip():
        raise ValueError(f"example {question_id} has an invalid question_type")
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"example {question_id} has an empty question")

    session_ids_value = item.get("haystack_session_ids")
    sessions_value = item.get("haystack_sessions")
    gold_value = item.get("answer_session_ids")
    if not isinstance(session_ids_value, list) or not session_ids_value:
        raise ValueError(f"example {question_id} has no haystack_session_ids list")
    if not isinstance(sessions_value, list) or not sessions_value:
        raise ValueError(f"example {question_id} has no haystack_sessions list")
    if len(session_ids_value) != len(sessions_value):
        raise ValueError(
            f"example {question_id} has mismatched session IDs and haystack sessions"
        )
    if not isinstance(gold_value, list) or not gold_value:
        raise ValueError(f"example {question_id} has no answer_session_ids list")

    session_ids = [
        _safe_identifier(value, f"haystack_session_ids[{index}]")
        for index, value in enumerate(session_ids_value)
    ]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(f"example {question_id} has duplicate haystack session IDs")

    gold_ids = [
        _safe_identifier(value, f"answer_session_ids[{index}]")
        for index, value in enumerate(gold_value)
    ]
    if len(gold_ids) != len(set(gold_ids)):
        raise ValueError(f"example {question_id} has duplicate answer session IDs")
    if not set(gold_ids).issubset(session_ids):
        missing = sorted(set(gold_ids) - set(session_ids))
        raise ValueError(
            f"example {question_id} has answer sessions absent from its haystack: {missing}"
        )

    dates_value = item.get("haystack_dates")
    if dates_value is not None:
        if not isinstance(dates_value, list) or len(dates_value) != len(session_ids):
            raise ValueError(f"example {question_id} has malformed haystack_dates")
        for date_index, date in enumerate(dates_value):
            if not isinstance(date, str) or not date.strip():
                raise ValueError(
                    f"example {question_id} has invalid haystack_dates[{date_index}]"
                )

    sessions: list[tuple[Turn, ...]] = []
    for session_index, (session_id, session) in enumerate(
        zip(session_ids, sessions_value)
    ):
        if not isinstance(session, list) or not session:
            raise ValueError(
                f"example {question_id} session {session_id} must be a non-empty list"
            )
        turns: list[Turn] = []
        for turn_index, turn in enumerate(session):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"example {question_id} session {session_id} turn {turn_index} "
                    "must be an object"
                )
            role = turn.get("role")
            if not isinstance(role, str) or role.strip() not in {"user", "assistant"}:
                raise ValueError(
                    f"example {question_id} session {session_id} turn {turn_index} "
                    "has an invalid role"
                )
            if "has_answer" in turn and not isinstance(turn["has_answer"], bool):
                raise ValueError(
                    f"example {question_id} session {session_id} turn {turn_index} "
                    "has a non-boolean has_answer marker"
                )
            content = _clean_content(
                turn.get("content"),
                item_index=item_index,
                session_index=session_index,
                turn_index=turn_index,
            )
            turns.append(
                Turn(
                    session_id=session_id,
                    turn_index=turn_index,
                    role=role.strip(),
                    content=content,
                )
            )
        sessions.append(tuple(turns))

    return {
        "question_id": question_id,
        "question_type": question_type.strip(),
        "question": question.strip(),
        "session_ids": tuple(session_ids),
        "sessions": tuple(sessions),
        "gold_session_ids": tuple(gold_ids),
    }


def validate_dataset(data: object) -> list[dict[str, Any]]:
    """Fail closed on malformed corpus, session IDs, turns, or gold labels."""

    if not isinstance(data, list) or not data:
        raise ValueError("LongMemEval-S dataset must be a non-empty JSON array")
    examples = [
        _validate_example(item, item_index=index) for index, item in enumerate(data)
    ]
    question_ids = [example["question_id"] for example in examples]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("LongMemEval-S question IDs must be unique")
    return examples


def retrieval_challenge_audit(
    examples: Sequence[Mapping[str, Any]], *, top_k: int = FIXED_TOP_K
) -> dict[str, Any]:
    """Describe whether support-session ranking can be distinguished at k."""
    if not examples:
        raise ValueError("retrieval challenge audit requires examples")
    session_counts = [len(example["session_ids"]) for example in examples]
    gold_counts = [len(example["gold_session_ids"]) for example in examples]
    non_gold_counts = [
        len(set(example["session_ids"]) - set(example["gold_session_ids"]))
        for example in examples
    ]
    return {
        "examples": len(examples),
        "top_k": top_k,
        "total_haystack_sessions": sum(session_counts),
        "total_gold_sessions": sum(gold_counts),
        "total_non_gold_sessions": sum(non_gold_counts),
        "examples_with_non_gold_sessions": sum(value > 0 for value in non_gold_counts),
        "examples_with_more_than_top_k_sessions": sum(
            value > top_k for value in session_counts
        ),
        "min_haystack_sessions": min(session_counts),
        "max_haystack_sessions": max(session_counts),
    }


def require_retrieval_challenge(
    examples: Sequence[Mapping[str, Any]], *, top_k: int = FIXED_TOP_K
) -> dict[str, Any]:
    audit = retrieval_challenge_audit(examples, top_k=top_k)
    failures: list[str] = []
    if audit["total_non_gold_sessions"] == 0:
        failures.append("every haystack session is gold; there are no distractors")
    if audit["examples_with_more_than_top_k_sessions"] == 0:
        failures.append(
            f"no example has more than top_k={top_k} sessions, so top-k returns the full haystack"
        )
    if failures:
        raise DatasetNotRetrievalEvaluableError(
            "LongMemEval file is oracle-context data, not a retrieval corpus: "
            + "; ".join(failures),
            audit,
        )
    return audit


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read and validate a dataset without changing it; return bytes SHA256."""

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read LongMemEval-S dataset {path}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed LongMemEval-S JSON at {path}: {exc}") from exc
    return validate_dataset(data), _sha256_bytes(raw)


def build_documents(example: Mapping[str, Any], condition: str) -> list[Document]:
    """Build retrieval documents from haystack content only.

    ``example`` must be the retrieval-only value returned by
    :func:`validate_dataset`.  No gold field is consulted here.  Whole-session
    concatenation is the conventional baseline.  The two turn conditions use
    the same raw turn content; they differ only in how turn ranks become
    support-session ranks.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"unknown retrieval condition: {condition}")
    session_ids = example.get("session_ids")
    sessions = example.get("sessions")
    if not isinstance(session_ids, tuple) or not isinstance(sessions, tuple):
        raise ValueError("build_documents requires a validated example")
    if len(session_ids) != len(sessions) or not session_ids:
        raise ValueError("validated example has malformed session structure")

    documents: list[Document] = []
    if condition == "whole_session":
        for session_id, turns in zip(session_ids, sessions):
            text = "\n\n".join(turn.content for turn in turns).strip()
            if not text:
                raise ValueError(f"session {session_id} has no document text")
            documents.append(
                Document(
                    document_id=session_id,
                    session_id=session_id,
                    text=text,
                    turn_index=None,
                )
            )
        return documents

    for session_id, turns in zip(session_ids, sessions):
        for turn in turns:
            documents.append(
                Document(
                    document_id=f"{session_id}::turn:{turn.turn_index}",
                    session_id=session_id,
                    text=turn.content,
                    turn_index=turn.turn_index,
                )
            )
    if not documents:
        raise ValueError("turn-level condition has no documents")
    return documents


def aggregate_turn_scores(
    ranked_turns: Iterable[tuple[str, float]],
    turn_to_session: Mapping[str, str],
    *,
    method: str,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Aggregate gold-independent ranked turns to deterministic session ranks.

    ``max`` assigns each session its highest turn score.  ``rrf`` sums
    ``1 / (rrf_k + rank)`` across the ranked turns.  Scores and IDs are
    validated so malformed backend output fails closed rather than silently
    corrupting metrics.
    """

    if method not in {"max", "rrf"}:
        raise ValueError(f"unknown turn aggregation method: {method}")
    if not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")

    rows = list(ranked_turns)
    aggregates: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    seen_turns: set[str] = set()
    for rank, pair in enumerate(rows, start=1):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("ranked turn output must contain (turn_id, score) pairs")
        turn_id, score = pair
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("ranked turn output contains an invalid turn ID")
        if turn_id in seen_turns:
            raise ValueError(f"ranked turn output contains duplicate turn ID: {turn_id}")
        seen_turns.add(turn_id)
        session_id = turn_to_session.get(turn_id)
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"ranked turn is absent from turn mapping: {turn_id}")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"ranked turn has a non-finite score: {turn_id}")
        score_float = float(score)
        if score_float < 0.0:
            raise ValueError(f"ranked turn has a negative score: {turn_id}")
        first_rank.setdefault(session_id, rank)
        if method == "max":
            aggregates[session_id] = max(aggregates.get(session_id, 0.0), score_float)
        else:
            aggregates[session_id] = aggregates.get(session_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )

    return sorted(
        aggregates.items(),
        key=lambda pair: (-pair[1], first_rank[pair[0]], pair[0]),
    )


def metrics_for_ranking(
    ranked_session_ids: Sequence[str],
    gold_session_ids: Sequence[str],
    *,
    top_k: int = FIXED_TOP_K,
) -> dict[str, Any]:
    """Compute support-session metrics from a ranked session-ID list."""

    if not isinstance(top_k, int) or top_k < FIXED_TOP_K:
        raise ValueError(f"top_k must be an integer >= {FIXED_TOP_K}")
    ranked = list(ranked_session_ids)
    gold = set(gold_session_ids)
    if not gold:
        raise ValueError("gold_session_ids must be non-empty")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked session IDs must be unique")
    if any(not isinstance(value, str) or not value for value in ranked):
        raise ValueError("ranked session IDs must be non-empty strings")

    recall: dict[str, float] = {}
    hit: dict[str, float] = {}
    all_gold: dict[str, float] = {}
    for k in METRIC_KS:
        selected = set(ranked[:k])
        matched = selected & gold
        recall[str(k)] = len(matched) / len(gold)
        hit[str(k)] = float(bool(matched))
        all_gold[str(k)] = float(gold.issubset(selected))
    first_rank = next(
        (rank for rank, session_id in enumerate(ranked[:top_k], start=1) if session_id in gold),
        None,
    )
    return {
        "support_session_recall_at_k": recall,
        "hit_at_k": hit,
        "all_gold_at_k": all_gold,
        "first_gold_rank_at_10": first_rank,
        "mrr_at_10": (1.0 / first_rank) if first_rank is not None else 0.0,
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return fmean(values) if values else 0.0


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-question support-session metrics."""

    def at_k(metric: str) -> dict[str, float]:
        return {
            str(k): _mean_metric(rows, f"{metric}@{k}") for k in METRIC_KS
        }

    summary = {
        "n_questions": len(rows),
        "support_session_recall_at_k": at_k("support_session_recall"),
        "hit_at_k": at_k("hit"),
        "all_gold_at_k": at_k("all_gold"),
        "mrr_at_10": _mean_metric(rows, "mrr_at_10"),
        "query_latency_ms": {
            "mean": _mean_metric(rows, "query_latency_ms"),
            "p50": _percentile([float(row["query_latency_ms"]) for row in rows], 0.50),
            "p95": _percentile([float(row["query_latency_ms"]) for row in rows], 0.95),
        },
    }
    # Keep explicit names beside the compact @k maps so a report consumer can
    # read the primary requested cutoffs without knowing the nested schema.
    for k in METRIC_KS:
        summary[f"support_session_recall_at_{k}"] = summary[
            "support_session_recall_at_k"
        ][str(k)]
        summary[f"hit_at_{k}"] = summary["hit_at_k"][str(k)]
        summary[f"all_gold_at_{k}"] = summary["all_gold_at_k"][str(k)]
    return summary


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = round((len(ordered) - 1) * quantile)
    return ordered[position]


def _git_output(*args: str) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
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


def _build_index(documents: Sequence[Document]) -> tuple[BM25SBackend, dict[str, Any], float]:
    """Build an in-memory BM25S index and report deterministic input footprint."""

    if not documents:
        raise ValueError("cannot build an empty BM25S index")
    index = BM25SBackend()
    rows = [(document.document_id, document.text) for document in documents]
    start = time.perf_counter()
    index.add_batch(rows)
    # BM25SBackend is lazy.  Force its private rebuild so build and query times
    # are not conflated; this does not persist or mutate any external state.
    rebuild = getattr(index, "_rebuild", None)
    if not callable(rebuild):
        raise RuntimeError("BM25SBackend does not expose its required in-memory rebuild")
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            rebuild()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    token_count = sum(len(index.tokenize(document.text)) for document in documents)
    text_bytes = sum(len(document.text.encode("utf-8")) for document in documents)
    id_bytes = sum(len(document.document_id.encode("utf-8")) for document in documents)
    terms = {
        token
        for document in documents
        for token in index.tokenize(document.text)
    }
    footprint = {
        "document_count": len(documents),
        "document_token_count": token_count,
        "document_text_bytes": text_bytes,
        "document_id_bytes": id_bytes,
        "unique_index_terms": len(terms),
        "estimated_index_input_bytes": text_bytes + id_bytes,
        "footprint_definition": (
            "UTF-8 source text and ID bytes plus token counts; this is an "
            "input-footprint proxy, not allocator RSS or serialized BM25S size"
        ),
    }
    return index, footprint, elapsed_ms


def _rank_condition(
    example: Mapping[str, Any], condition: str, *, top_k: int
) -> tuple[list[str], dict[str, Any], float, float]:
    documents = build_documents(example, condition)
    index, footprint, build_ms = _build_index(documents)
    query = example["question"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("validated question unexpectedly became empty")
    # Retrieve every source document before aggregation.  This is necessary
    # for max/RRF to see turns from all sessions; the fixed output cutoff is
    # still top_k (10 by default) at the session level.
    candidate_k = max(top_k, len(documents))
    query_start = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            ranked_documents = index.search(query, top_k=candidate_k)
    query_ms = (time.perf_counter() - query_start) * 1000.0
    if not isinstance(ranked_documents, list):
        raise ValueError("BM25SBackend returned malformed ranked output")

    if condition == "whole_session":
        ranked_sessions: list[tuple[str, float]] = []
        seen: set[str] = set()
        for pair in ranked_documents:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("BM25SBackend returned malformed ranked output")
            document_id, score = pair
            if not isinstance(document_id, str) or document_id not in example["session_ids"]:
                raise ValueError("whole-session BM25S output contains an unknown ID")
            if document_id in seen:
                raise ValueError("whole-session BM25S output contains duplicate IDs")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ValueError("whole-session BM25S output contains a non-finite score")
            seen.add(document_id)
            ranked_sessions.append((document_id, float(score)))
    else:
        turn_to_session = {document.document_id: document.session_id for document in documents}
        ranked_sessions = aggregate_turn_scores(
            ranked_documents,
            turn_to_session,
            method="max" if condition == "turn_max" else "rrf",
            rrf_k=RRF_K,
        )
    ranked_ids = [session_id for session_id, _ in ranked_sessions[:top_k]]
    return ranked_ids, footprint, build_ms, query_ms


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {
        "mrr_at_10": float(metrics["mrr_at_10"]),
        "first_gold_rank_at_10": metrics["first_gold_rank_at_10"],
    }
    prefixes = {
        "support_session_recall_at_k": "support_session_recall",
        "hit_at_k": "hit",
        "all_gold_at_k": "all_gold",
    }
    for metric_name, prefix in prefixes.items():
        values = metrics[metric_name]
        for k in METRIC_KS:
            flattened[f"{prefix}@{k}"] = float(values[str(k)])
    return flattened


def evaluate(
    dataset: Path,
    *,
    top_k: int = FIXED_TOP_K,
    mechanics_test_only: bool = False,
) -> dict[str, Any]:
    """Run all fixed exploratory conditions and return a provenance-rich report."""

    if not isinstance(top_k, int) or top_k < FIXED_TOP_K:
        raise ValueError(f"top_k must be an integer >= {FIXED_TOP_K}")
    started = time.perf_counter()
    examples, source_sha256 = load_dataset(Path(dataset))
    challenge_audit = retrieval_challenge_audit(examples, top_k=top_k)
    if not mechanics_test_only:
        challenge_audit = require_retrieval_challenge(examples, top_k=top_k)

    rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    footprints: dict[str, dict[str, int]] = {
        condition: {
            "document_count": 0,
            "document_token_count": 0,
            "document_text_bytes": 0,
            "document_id_bytes": 0,
            "unique_index_terms": 0,
            "estimated_index_input_bytes": 0,
        }
        for condition in CONDITIONS
    }
    question_rows: list[dict[str, Any]] = []

    for example in examples:
        question_id = example["question_id"]
        question_sha256 = _sha256_text(example["question"])
        post_retrieval_gold = example["gold_session_ids"]
        by_condition: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            ranked_ids, footprint, build_ms, query_ms = _rank_condition(
                example, condition, top_k=top_k
            )
            for key in footprints[condition]:
                if key == "unique_index_terms":
                    # The same term may recur in different per-question
                    # indexes; totals are intentionally additive except for
                    # this explicit per-index vocabulary footprint.
                    footprints[condition][key] += int(footprint[key])
                else:
                    footprints[condition][key] += int(footprint[key])
            score = metrics_for_ranking(
                ranked_ids, post_retrieval_gold, top_k=top_k
            )
            flat = _flatten_metrics(score)
            row = {
                "question_id": question_id,
                "question_sha256": question_sha256,
                "question_type": example["question_type"],
                "condition": condition,
                "ranked_session_ids": ranked_ids,
                "gold_session_ids": list(post_retrieval_gold),
                "haystack_session_count": len(example["session_ids"]),
                "haystack_turn_count": sum(len(session) for session in example["sessions"]),
                "index_build_ms": build_ms,
                "query_latency_ms": query_ms,
                "document_footprint": footprint,
                "metrics": score,
                **flat,
            }
            rows_by_condition[condition].append(row)
            by_condition[condition] = row
            question_rows.append(row)

    # Grouped summaries deliberately consume only post-retrieval rows.
    condition_summaries: dict[str, dict[str, Any]] = {}
    by_question_type: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for condition in CONDITIONS:
        condition_rows = rows_by_condition[condition]
        condition_summaries[condition] = {
            "metrics": _summarize_rows(condition_rows),
            "footprint": footprints[condition],
            "elapsed_ms": {
                "index_build_total": sum(float(row["index_build_ms"]) for row in condition_rows),
                "query_total": sum(float(row["query_latency_ms"]) for row in condition_rows),
                "total": sum(
                    float(row["index_build_ms"]) + float(row["query_latency_ms"])
                    for row in condition_rows
                ),
            },
        }
        for question_type in sorted({row["question_type"] for row in condition_rows}):
            selected = [row for row in condition_rows if row["question_type"] == question_type]
            by_question_type[condition][question_type] = _summarize_rows(selected)
        condition_summaries[condition]["by_question_type"] = by_question_type[condition]

    rows_by_key = {
        (row["question_id"], row["condition"]): row for row in question_rows
    }
    transitions: list[dict[str, Any]] = []
    for example in examples:
        question_id = example["question_id"]
        baseline = rows_by_key[(question_id, "whole_session")]
        for candidate in ("turn_max", "turn_rrf"):
            target = rows_by_key[(question_id, candidate)]
            base_top = set(baseline["ranked_session_ids"][:FIXED_TOP_K])
            target_top = set(target["ranked_session_ids"][:FIXED_TOP_K])
            transitions.append(
                {
                    "question_id": question_id,
                    "question_type": example["question_type"],
                    "from_condition": "whole_session",
                    "to_condition": candidate,
                    "support_session_recall_at_10_delta": (
                        target["support_session_recall@10"]
                        - baseline["support_session_recall@10"]
                    ),
                    "hit_at_10_delta": target["hit@10"] - baseline["hit@10"],
                    "mrr_at_10_delta": target["mrr_at_10"] - baseline["mrr_at_10"],
                    "top10_added_session_ids": sorted(target_top - base_top),
                    "top10_removed_session_ids": sorted(base_top - target_top),
                    "baseline_first_gold_rank_at_10": baseline["first_gold_rank_at_10"],
                    "candidate_first_gold_rank_at_10": target["first_gold_rank_at_10"],
                }
            )

    elapsed_seconds = time.perf_counter() - started
    report = {
        "schema_version": SCHEMA,
        "classification": "exploratory_offline_support_session_retrieval",
        "claim_boundary": {
            "target": "LongMemEval-S support-session retrieval",
            "gold_field": "answer_session_ids",
            "not_measured": [
                "exact-turn evidence recall",
                "answer accuracy",
                "reader quality",
                "SOTA or production quality",
            ],
        },
        "comparison": {
            "label": "exploratory",
            "winner_selected": False,
            "selection_dataset": None,
            "conditions": list(CONDITIONS),
        },
        "configuration": {
            "fixed_top_k": top_k,
            "metric_k_values": list(METRIC_KS),
            "whole_session_document": "concatenate cleaned haystack turn content per session",
            "turn_document": "one cleaned haystack content string per turn",
            "turn_aggregation": {
                "turn_max": "maximum BM25S score per session",
                "turn_rrf": "sum(1 / (60 + turn_rank)) per session",
                "rrf_k": RRF_K,
                "candidate_policy": "retrieve all indexed documents, then cut to fixed session top-k",
            },
            "metadata_enrichment": "none; role/date/has_answer fields are not indexed",
            "query_input": "question field only",
            "document_input": "question's haystack_sessions content only",
            "gold_usage": "answer_session_ids read only after ranking for metrics",
            "tokenizer": "BM25SBackend tokenizer (bm25s, English stopwords, optional PyStemmer)",
            "random_seed": None,
            "mechanics_test_only": mechanics_test_only,
        },
        "dataset": {
            "path": str(Path(dataset).resolve()),
            "sha256": source_sha256,
            "examples": len(examples),
            "question_types": {
                key: sum(example["question_type"] == key for example in examples)
                for key in sorted({example["question_type"] for example in examples})
            },
            "gold_scope": "support sessions in each question's haystack",
            "retrieval_challenge_audit": challenge_audit,
        },
        "execution": {
            "offline": True,
            "provider_calls": 0,
            "network_calls": 0,
            "external_network_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "actual_external_cost_usd": 0.0,
        },
        "provenance": {
            "dataset_sha256": source_sha256,
            "git_head": _git_output("rev-parse", "HEAD"),
            "git_status": "dirty" if _git_output("status", "--porcelain") else "clean",
            "dependencies": _dependency_versions(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "conditions": condition_summaries,
        "metrics": {
            condition: condition_summaries[condition]["metrics"]
            for condition in CONDITIONS
        },
        "question_level_rows": question_rows,
        "transitions": transitions,
        "elapsed_seconds": elapsed_seconds,
        "interpretation_limits": [
            "Every query uses only its own haystack sessions; no cross-question corpus is searched.",
            "answer_session_ids are session-level support labels, not exact turn evidence IDs.",
            "The answer text is not read by retrieval or validation.",
            "All conditions are exploratory and measured on this same dataset; no winner is selected.",
            "Zero provider/network calls are asserted by execution accounting, not a network firewall.",
            "A mechanics-only fixture may bypass the distractor gate but is never an admissible quality result.",
        ],
    }
    return report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create JSON exactly once through an exclusive temporary hard-link.

    The destination is intentionally create-once: an existing artifact is
    never overwritten.  JSON is fully written and fsynced to an exclusive
    temporary inode first; ``os.link`` then publishes that inode atomically
    and fails with ``FileExistsError`` if another run already owns the target.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path, follow_symlinks=False)
        except FileExistsError:
            raise
        finally:
            # On success the destination and temp name are two links to the
            # same fully-fsynced inode.  On failure this removes the scratch
            # inode without touching an existing destination.
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-k",
        type=int,
        default=FIXED_TOP_K,
        help=f"fixed session cutoff; must be >= {FIXED_TOP_K}",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.dataset.resolve(), top_k=args.top_k)
    except DatasetNotRetrievalEvaluableError as exc:
        dataset_path = args.dataset.resolve()
        source = Path(__file__).resolve()
        receipt = {
            "schema_version": FAILED_SCHEMA,
            "classification": "failed_dataset_admission",
            "status": "failed",
            "reason": str(exc),
            "retrieval_challenge_audit": exc.audit,
            "dataset": {
                "path": str(dataset_path),
                "sha256": _sha256_bytes(dataset_path.read_bytes()),
            },
            "execution": {
                "offline": True,
                "provider_calls": 0,
                "external_network_calls": 0,
                "embedding_calls": 0,
                "reranker_calls": 0,
                "reader_calls": 0,
                "actual_external_cost_usd": 0.0,
            },
            "provenance": {
                "source": {
                    "path": str(source),
                    "sha256": _sha256_bytes(source.read_bytes()),
                },
                "git_head": _git_output("rev-parse", "HEAD"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": _dependency_versions(),
            },
            "claim_boundary": "No retrieval metric is valid for this receipt.",
        }
        _atomic_write_json(args.output.resolve(), receipt)
        print(
            json.dumps(
                {"output": str(args.output.resolve()), "status": "failed"},
                sort_keys=True,
            )
        )
        return 2
    _atomic_write_json(args.output.resolve(), report)
    summary = {
        "output": str(args.output.resolve()),
        "examples": report["dataset"]["examples"],
        "provider_calls": report["execution"]["provider_calls"],
        "network_calls": report["execution"]["network_calls"],
        "conditions": {
            condition: report["conditions"][condition]["metrics"][
                "support_session_recall_at_k"
            ]["10"]
            for condition in CONDITIONS
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
