"""Measure the retrieval-conditioned effective-context frontier.

This benchmark measures whether a smaller retrieved prompt preserves exact
LoCoMo evidence IDs. It does not claim retrieval replaces, compresses, evicts,
or proportionally reduces a model's realized KV cache. Answer-string overlap
is diagnostic only and can never satisfy the quality gate.

The benchmark is read-only. It joins three existing artifacts:

* a LoCoMo evaluation ledger containing ranked HybridMind node IDs;
* the original LoCoMo conversations and evidence annotations; and
* the SQLite node store containing the retrieved text.

An optional architecture-only allocation estimate can be reported when model
parameters are supplied. It is explicitly labelled as the bytes that would be
allocated *if* all counted prompt tokens were materialized in a standard
attention cache, not a measured cache saving. Bytes per token are:

    2 * layers * kv_heads * head_dim * element_bytes

The factor of two accounts for keys and values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Protocol, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_LEDGER = Path("benchmarks/results/ledger_locomo_7d21dbb5fc67.jsonl")
DEFAULT_CHECKPOINT = Path(
    "memorybench/data/runs/hybridmind-locomo-fixed-20260726/checkpoint.json"
)
DEFAULT_DATASET = Path("memorybench/data/benchmarks/locomo/locomo10.json")
DEFAULT_DATABASE = Path("data/hybridmind.mind/store.db")
DEFAULT_OUTPUT = Path("benchmarks/results/kv_reduction_locomo.json")
DEFAULT_K_VALUES = (1, 3, 5, 10, 25)
DEFAULT_TOKENIZER = "BAAI/bge-m3"

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SESSION_RE = re.compile(r"^session_(\d+)$")


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""


class RegexTokenCounter:
    """Deterministic approximation for tests and dependency-free audits."""

    name = "regex-word-punctuation-proxy"

    def count(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))


class HuggingFaceTokenCounter:
    """Count tokens with an explicit Hugging Face tokenizer."""

    def __init__(self, model_name: str, *, allow_download: bool = False):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - repository dependency
            raise RuntimeError("transformers is required for Hugging Face token counting") from exc

        self.name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=not allow_download,
            trust_remote_code=False,
        )
        # Tokenization is used for counting only; no model forward pass occurs.
        self._tokenizer.model_max_length = 1_000_000_000
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        count = len(self._tokenizer.encode(text, add_special_tokens=False))
        self._cache[text] = count
        return count


@dataclass(frozen=True)
class QuestionContext:
    question_id: str
    sample_id: str
    question: str
    category: int
    full_context: str
    evidence_ids: tuple[str, ...]
    evidence_texts: tuple[str, ...]
    unresolved_evidence_ids: tuple[str, ...]


def _question_id(question: str, explicit_id: Any = None) -> str:
    if explicit_id:
        return str(explicit_id)
    return hashlib.sha1(question.encode()).hexdigest()[:12]


def _canonical_evidence_id(sample_id: str, evidence_id: Any) -> str:
    value = str(evidence_id).strip()
    return value if value.startswith("locomo:") else f"locomo:{sample_id}:{value}"


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _format_turn(message: dict[str, Any], date: str) -> str:
    speaker = str(message.get("speaker", "unknown"))
    text = str(message.get("text", "")).strip()
    if date:
        return f"[DATE: {date}] [SPEAKER: {speaker}] {text}"
    return f"[SPEAKER: {speaker}] {text}"


def load_question_contexts(
    dataset_path: Path,
    *,
    id_scheme: str = "ledger",
) -> dict[str, deque[QuestionContext]]:
    """Load LoCoMo questions, preserving duplicate question IDs as queues."""
    if id_scheme not in {"ledger", "ledger_v2", "memorybench"}:
        raise ValueError("id_scheme must be 'ledger', 'ledger_v2', or 'memorybench'")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_question_id: dict[str, deque[QuestionContext]] = defaultdict(deque)

    for conversation_index, record in enumerate(data):
        conversation = record.get("conversation", {})
        sample_id = str(record.get("sample_id", f"locomo_{conversation_index + 1}"))
        session_keys = sorted(
            (
                (int(match.group(1)), key)
                for key in conversation
                if (match := _SESSION_RE.match(key))
            ),
            key=lambda item: item[0],
        )

        formatted_turns: list[str] = []
        evidence_by_id: dict[str, str] = {}
        for _, session_key in session_keys:
            date = str(conversation.get(f"{session_key}_date_time", ""))
            messages = conversation.get(session_key, [])
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, dict) or not str(message.get("text", "")).strip():
                    continue
                formatted_turns.append(_format_turn(message, date))
                evidence_id = message.get("dia_id")
                if evidence_id:
                    evidence_by_id[str(evidence_id)] = str(message["text"]).strip()

        full_context = "\n".join(formatted_turns)
        for question_index, qa in enumerate(record.get("qa", [])):
            question = str(qa.get("question", ""))
            if id_scheme == "memorybench":
                qid = f"{sample_id}-q{question_index}"
            elif id_scheme == "ledger_v2":
                explicit_id = qa.get("question_id")
                qid = (
                    f"locomo:{sample_id}:{explicit_id}"
                    if explicit_id
                    else "locomo:" + hashlib.sha1(
                        f"{sample_id}\0{question_index}\0{question}".encode()
                    ).hexdigest()[:16]
                )
            else:
                qid = _question_id(question, qa.get("question_id"))
            raw_evidence_ids = tuple(str(item) for item in qa.get("evidence", []) if item)
            evidence_ids = tuple(
                _canonical_evidence_id(sample_id, item) for item in raw_evidence_ids
            )
            evidence_texts = tuple(
                evidence_by_id[evidence_id]
                for evidence_id in raw_evidence_ids
                if evidence_id in evidence_by_id
            )
            unresolved = tuple(
                evidence_id for evidence_id in raw_evidence_ids if evidence_id not in evidence_by_id
            )
            by_question_id[qid].append(
                QuestionContext(
                    question_id=qid,
                    sample_id=sample_id,
                    question=question,
                    category=int(qa.get("category", 0) or 0),
                    full_context=full_context,
                    evidence_ids=evidence_ids,
                    evidence_texts=evidence_texts,
                    unresolved_evidence_ids=unresolved,
                )
            )

    return by_question_id


def load_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ledger_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on ledger line {line_number}: {exc}") from exc
            if record.get("schema") != "hybridmind.eval-ledger/v2":
                raise ValueError(
                    f"Ledger line {line_number} is legacy/unattested; exact evidence-ID "
                    "evaluation requires hybridmind.eval-ledger/v2"
                )
            if record.get("status") != "completed":
                raise ValueError(
                    f"Ledger line {line_number} has status {record.get('status')!r}; "
                    "partial or failed runs are invalid"
                )
            if record.get("metric_basis") != "exact_evidence_id":
                raise ValueError(
                    f"Ledger line {line_number} does not declare exact_evidence_id relevance"
                )
            extra = record.get("extra") or {}
            required_extra = {
                "retrieved_evidence_ids_at_k",
                "retrieved_result_count_at_k",
                "evidence_tagged_result_count_at_k",
            }
            missing = sorted(required_extra - set(extra))
            if missing:
                raise ValueError(
                    f"Ledger line {line_number} lacks exact-evidence retrieval metadata: "
                    f"{', '.join(missing)}"
                )
            record["_retrieved_evidence_ids_at_k"] = extra["retrieved_evidence_ids_at_k"]
            record["_retrieved_result_count_at_k"] = extra["retrieved_result_count_at_k"]
            record["_evidence_tagged_result_count_at_k"] = extra[
                "evidence_tagged_result_count_at_k"
            ]
            overlap = record.get("answer_overlap_metrics") or {}
            record["answer_overlap_rank_post_rerank"] = overlap.get(
                "gold_rank_post_rerank"
            )
            records.append(record)
    return records


def _answer_tokens(answer: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z0-9']+", answer.lower()))
    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was", "it", "and", "or", "but"
    }
    return tokens - stopwords


def _is_answer_relevant(retrieved_text: str, answer: str) -> bool:
    """Match eval_locomo_retrieval.py's historical answer-overlap proxy."""
    text_lower = retrieved_text.lower()
    answer_lower = answer.lower()
    if answer_lower and answer_lower in text_lower:
        return True
    answer_tokens = _answer_tokens(answer)
    if not answer_tokens:
        return False
    text_tokens = set(re.findall(r"[A-Za-z0-9']+", text_lower))
    if len(answer_tokens) <= 3:
        return answer_tokens.issubset(text_tokens)
    overlap = len(answer_tokens & text_tokens)
    return overlap / len(answer_tokens) >= 0.7


def _checkpoint_result_evidence_ids(result: dict[str, Any], sample_id: str) -> set[str]:
    metadata = result.get("metadata") or {}
    values: list[Any] = []
    for key in ("evidence_id", "dia_id"):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return {
        _canonical_evidence_id(sample_id, value)
        for value in values
        if str(value).strip()
    }


def load_memorybench_checkpoint(
    checkpoint_path: Path,
    k_values: Sequence[int],
    *,
    ranking_strategy: str = "combined",
    lexical_pool_size: int = 500,
    lexical_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Stream completed MemoryBench searches into the ledger-compatible shape."""
    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - installed in the workspace
        raise RuntimeError("ijson is required to stream MemoryBench checkpoints") from exc

    if ranking_strategy not in {"combined", "local-lexical-rrf"}:
        raise ValueError("ranking_strategy must be 'combined' or 'local-lexical-rrf'")
    if ranking_strategy == "local-lexical-rrf":
        from engine.lexical_reranker import rerank_with_query_local_lexical_rrf
        from storage.bm25_index import BM25Index

        lexical_tokenize = BM25Index().tokenize
    else:
        rerank_with_query_local_lexical_rrf = None
        lexical_tokenize = None

    max_k = max(k_values)
    records: list[dict[str, Any]] = []
    with checkpoint_path.open("rb") as handle:
        for question_id, question in ijson.kvitems(handle, "questions"):
            search = question.get("phases", {}).get("search", {})
            if search.get("status") != "completed" or not isinstance(search.get("results"), list):
                continue
            # Historical MemoryBench providers could reorder API results by
            # timestamp before checkpointing them. Reconstruct the retrieval
            # order from the persisted final score so top-k remains meaningful.
            results = sorted(
                search["results"],
                key=lambda result: (
                    float(result.get("combined_score") or 0.0),
                    float(result.get("vector_score") or 0.0),
                ),
                reverse=True,
            )
            offline_rerank_duration_ms = None
            if ranking_strategy == "local-lexical-rrf":
                rerank_started = time.perf_counter()
                results = rerank_with_query_local_lexical_rrf(
                    str(question.get("question", "")),
                    results,
                    lexical_tokenize,
                    pool_size=lexical_pool_size,
                    lexical_weight=lexical_weight,
                    rrf_k=60,
                )
                offline_rerank_duration_ms = (time.perf_counter() - rerank_started) * 1000
            results = results[:max_k]
            answer = str(question.get("groundTruth", ""))
            answer_overlap_rank = None
            for rank, result in enumerate(results, 1):
                if _is_answer_relevant(str(result.get("text", "")), answer):
                    answer_overlap_rank = rank
                    break
            sample_id = str(question_id).rsplit("-q", 1)[0]
            records.append(
                {
                    "question_id": str(question_id),
                    "answer_overlap_rank_post_rerank": answer_overlap_rank,
                    "retrieved_ids_at_k": {
                        str(k): [str(result.get("node_id") or result.get("id") or "") for result in results[:k]]
                        for k in k_values
                    },
                    "_retrieved_texts_at_k": {
                        str(k): [str(result.get("text", "")) for result in results[:k]]
                        for k in k_values
                    },
                    "_retrieved_evidence_ids_at_k": {
                        str(k): sorted(set().union(*(
                            _checkpoint_result_evidence_ids(result, sample_id)
                            for result in results[:k]
                        ))) if results[:k] else []
                        for k in k_values
                    },
                    "_retrieved_result_count_at_k": {
                        str(k): len(results[:k]) for k in k_values
                    },
                    "_evidence_tagged_result_count_at_k": {
                        str(k): sum(
                            bool(_checkpoint_result_evidence_ids(result, sample_id))
                            for result in results[:k]
                        )
                        for k in k_values
                    },
                    "_search_duration_ms": (
                        float(search["durationMs"]) if search.get("durationMs") is not None else None
                    ),
                    "_offline_rerank_duration_ms": offline_rerank_duration_ms,
                }
            )
    return records


def _exact_source_recall_by_question(
    records: Sequence[dict[str, Any]],
    dataset_path: Path,
    *,
    k: int,
) -> dict[str, float]:
    question_queues = load_question_contexts(dataset_path, id_scheme="memorybench")
    recall_by_question: dict[str, float] = {}
    for record in records:
        question_id = str(record.get("question_id", ""))
        queue = question_queues.get(question_id)
        if not queue:
            continue
        context = queue.popleft()
        evidence = [_normalize_text(text) for text in context.evidence_texts if text]
        if not evidence:
            continue
        retrieved_texts = record.get("_retrieved_texts_at_k", {}).get(str(k), [])
        normalized_retrieved = _normalize_text("\n".join(str(text) for text in retrieved_texts))
        recall_by_question[question_id] = mean(
            1.0 if source_text in normalized_retrieved else 0.0
            for source_text in evidence
        )
    return recall_by_question


def _paired_ranking_hypothesis(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    variant_records: Sequence[dict[str, Any]],
    k: int,
    lexical_pool_size: int,
    lexical_weight: float,
    min_improvement: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_records = load_memorybench_checkpoint(
        checkpoint_path,
        (k,),
        ranking_strategy="combined",
    )
    baseline = _exact_source_recall_by_question(baseline_records, dataset_path, k=k)
    variant = _exact_source_recall_by_question(variant_records, dataset_path, k=k)
    paired_ids = sorted(baseline.keys() & variant.keys())
    differences = [variant[question_id] - baseline[question_id] for question_id in paired_ids]
    delta = _bootstrap_ci(
        differences,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "k": k,
        "statement": (
            f"A {lexical_weight:.0%}/{1.0 - lexical_weight:.0%} fusion of query-local "
            f"lexical rank and existing candidate rank improves exact-source Recall@{k} "
            f"by at least {min_improvement:.0%}."
        ),
        "metric": f"paired exact-source Recall@{k}",
        "baseline_mean": mean(baseline[question_id] for question_id in paired_ids),
        "variant_mean": mean(variant[question_id] for question_id in paired_ids),
        "improvement": delta,
        "minimum_improvement": min_improvement,
        "improved_questions": sum(difference > 0 for difference in differences),
        "regressed_questions": sum(difference < 0 for difference in differences),
        "unchanged_questions": sum(difference == 0 for difference in differences),
        "passed": bool(differences) and float(delta["ci95_low"]) >= min_improvement,
        "confirmatory": False,
        "confirmation_limit": (
            "The same partial checkpoint informed hypothesis selection, so this is a paired "
            "retrospective estimate rather than an independent holdout confirmation."
        ),
    }


def _iter_retrieved_ids(records: Iterable[dict[str, Any]], k_values: Sequence[int]) -> Iterable[str]:
    for record in records:
        retrieved = record.get("retrieved_ids_at_k", {})
        for k in k_values:
            for node_id in retrieved.get(str(k), []):
                if node_id:
                    yield str(node_id)


def load_node_texts(
    database_path: Path,
    node_ids: Iterable[str],
    *,
    batch_size: int = 800,
) -> dict[str, str]:
    """Resolve node texts through a read-only SQLite connection."""
    unique_ids = sorted(set(node_ids))
    database_uri = f"file:{database_path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        texts: dict[str, str] = {}
        for start in range(0, len(unique_ids), batch_size):
            batch = unique_ids[start : start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT id, text FROM nodes WHERE id IN ({placeholders})"
            for node_id, text in connection.execute(query, batch):
                texts[str(node_id)] = str(text)
        return texts
    finally:
        connection.close()


def kv_bytes_per_token(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    element_bytes: float,
) -> float:
    values = (layers, kv_heads, head_dim, element_bytes)
    if any(value <= 0 for value in values):
        raise ValueError("KV architecture parameters must all be positive")
    return 2.0 * layers * kv_heads * head_dim * element_bytes


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p95": math.nan}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": _percentile(values, 0.95),
    }


def _bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    rng = random.Random(seed)
    n = len(values)
    resampled_means = []
    for _ in range(n_resamples):
        resampled_means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    resampled_means.sort()
    low_index = int(0.025 * n_resamples)
    high_index = max(low_index, int(0.975 * n_resamples) - 1)
    return {
        "n": n,
        "mean": mean(values),
        "ci95_low": resampled_means[low_index],
        "ci95_high": resampled_means[high_index],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_node_texts(node_texts: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for node_id, text in sorted(node_texts.items()):
        digest.update(node_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_retrieved_texts(records: Sequence[dict[str, Any]], k_values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    max_k = str(max(k_values))
    for record in records:
        digest.update(str(record.get("question_id", "")).encode("utf-8"))
        digest.update(b"\0")
        for text in record.get("_retrieved_texts_at_k", {}).get(max_k, []):
            digest.update(str(text).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def evaluate_frontier(
    *,
    ledger_path: Path | None,
    dataset_path: Path,
    database_path: Path | None,
    token_counter: TokenCounter,
    checkpoint_path: Path | None = None,
    checkpoint_ranking: str = "combined",
    local_lexical_pool_size: int = 500,
    local_lexical_weight: float = 0.5,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    hypothesis_k: int = 10,
    min_context_reduction: float = 0.90,
    min_exact_evidence_recall: float = 0.80,
    min_answer_proxy_hit: float | None = None,
    min_node_resolution: float = 0.99,
    min_exact_source_recall_improvement: float = 0.05,
    bootstrap_resamples: int = 2_000,
    seed: int = 42,
    absolute_kv_bytes_per_token: float | None = None,
    max_retrieved_tokens: int | None = None,
    max_search_latency_ms: float | None = None,
    max_offline_rerank_latency_ms: float | None = None,
) -> dict[str, Any]:
    k_values = tuple(sorted(set(int(k) for k in k_values)))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k values must be positive integers")
    if hypothesis_k not in k_values:
        raise ValueError("hypothesis_k must be included in k_values")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    # Backward-compatible argument name, but it now thresholds exact evidence
    # recall. Answer overlap never controls pass/fail.
    if min_answer_proxy_hit is not None:
        min_exact_evidence_recall = min_answer_proxy_hit

    if checkpoint_path is not None:
        ledger_records = load_memorybench_checkpoint(
            checkpoint_path,
            k_values,
            ranking_strategy=checkpoint_ranking,
            lexical_pool_size=local_lexical_pool_size,
            lexical_weight=local_lexical_weight,
        )
        question_queues = load_question_contexts(dataset_path, id_scheme="memorybench")
        node_texts: dict[str, str] = {}
        source_inputs = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_result_order": checkpoint_ranking,
            "local_lexical_pool_size": local_lexical_pool_size,
            "local_lexical_weight": local_lexical_weight,
            "retrieved_text_sha256": _sha256_retrieved_texts(ledger_records, k_values),
        }
    else:
        if ledger_path is None or database_path is None:
            raise ValueError("ledger_path and database_path are required without checkpoint_path")
        ledger_records = load_ledger(ledger_path)
        question_queues = load_question_contexts(dataset_path, id_scheme="ledger_v2")
        requested_ids = list(_iter_retrieved_ids(ledger_records, k_values))
        node_texts = load_node_texts(database_path, requested_ids)
        source_inputs = {
            "ledger": str(ledger_path),
            "ledger_sha256": _sha256_file(ledger_path),
            "database": str(database_path),
            "database_size_bytes": database_path.stat().st_size,
            "retrieved_node_text_sha256": _sha256_node_texts(node_texts),
        }

    per_k: dict[int, dict[str, list[float]]] = {
        k: {
            "retrieved_tokens": [],
            "context_reduction": [],
            "answer_proxy_hit_all": [],
            "answer_proxy_hit_with_evidence": [],
            "exact_source_recall": [],
            "exact_source_any_hit": [],
            "exact_source_all_hit": [],
            "exact_evidence_id_recall": [],
            "exact_evidence_id_any_hit": [],
            "exact_evidence_id_all_hit": [],
        }
        for k in k_values
    }
    full_context_tokens: list[float] = []
    matched_records = 0
    unmatched_ledger_records = 0
    unresolved_evidence_ids = 0
    total_retrieved_id_references = 0
    resolved_retrieved_id_references = 0
    search_durations_ms: list[float] = []
    offline_rerank_durations_ms: list[float] = []
    retrieved_result_references = 0
    evidence_tagged_result_references = 0

    for record in ledger_records:
        qid = str(record.get("question_id", ""))
        queue = question_queues.get(qid)
        if not queue:
            unmatched_ledger_records += 1
            continue
        context = queue.popleft()
        matched_records += 1
        unresolved_evidence_ids += len(context.unresolved_evidence_ids)
        full_tokens = token_counter.count(context.full_context)
        if full_tokens <= 0:
            continue
        full_context_tokens.append(float(full_tokens))

        answer_overlap_rank = record.get("answer_overlap_rank_post_rerank")
        retrieved_by_k = record.get("retrieved_ids_at_k", {})
        retrieved_texts_by_k = record.get("_retrieved_texts_at_k")
        retrieved_evidence_by_k = record.get("_retrieved_evidence_ids_at_k", {})
        result_count_by_k = record.get("_retrieved_result_count_at_k", {})
        tagged_count_by_k = record.get("_evidence_tagged_result_count_at_k", {})
        if record.get("_search_duration_ms") is not None:
            search_durations_ms.append(float(record["_search_duration_ms"]))
        if record.get("_offline_rerank_duration_ms") is not None:
            offline_rerank_durations_ms.append(float(record["_offline_rerank_duration_ms"]))
        normalized_evidence = [_normalize_text(text) for text in context.evidence_texts if text]

        for k in k_values:
            retrieved_ids = [str(node_id) for node_id in retrieved_by_k.get(str(k), []) if node_id]
            total_retrieved_id_references += len(retrieved_ids)
            if retrieved_texts_by_k is not None:
                texts = [str(text) for text in retrieved_texts_by_k.get(str(k), [])]
                resolved_retrieved_id_references += len(texts)
            else:
                texts = []
                for node_id in retrieved_ids:
                    text = node_texts.get(node_id)
                    if text is not None:
                        resolved_retrieved_id_references += 1
                        texts.append(text)
            retrieved_context = "\n".join(texts)
            retrieved_tokens = token_counter.count(retrieved_context)
            reduction = 1.0 - (retrieved_tokens / full_tokens)
            proxy_hit = (
                1.0
                if answer_overlap_rank is not None and int(answer_overlap_rank) <= k
                else 0.0
            )
            gold_evidence_ids = set(context.evidence_ids)
            retrieved_evidence_ids = {
                str(value) for value in retrieved_evidence_by_k.get(str(k), []) if value
            }
            exact_evidence_hits = gold_evidence_ids & retrieved_evidence_ids
            exact_evidence_recall = (
                len(exact_evidence_hits) / len(gold_evidence_ids)
                if gold_evidence_ids
                else math.nan
            )
            if k == max(k_values):
                retrieved_result_references += int(result_count_by_k.get(str(k), 0))
                evidence_tagged_result_references += int(tagged_count_by_k.get(str(k), 0))

            per_k[k]["retrieved_tokens"].append(float(retrieved_tokens))
            per_k[k]["context_reduction"].append(reduction)
            per_k[k]["answer_proxy_hit_all"].append(proxy_hit)

            if normalized_evidence:
                per_k[k]["answer_proxy_hit_with_evidence"].append(proxy_hit)
                normalized_retrieved = _normalize_text(retrieved_context)
                source_hits = [1.0 if evidence in normalized_retrieved else 0.0 for evidence in normalized_evidence]
                per_k[k]["exact_source_recall"].append(mean(source_hits))
                per_k[k]["exact_source_any_hit"].append(1.0 if any(source_hits) else 0.0)
                per_k[k]["exact_source_all_hit"].append(1.0 if all(source_hits) else 0.0)
            if gold_evidence_ids:
                per_k[k]["exact_evidence_id_recall"].append(exact_evidence_recall)
                per_k[k]["exact_evidence_id_any_hit"].append(
                    1.0 if exact_evidence_hits else 0.0
                )
                per_k[k]["exact_evidence_id_all_hit"].append(
                    1.0 if exact_evidence_hits == gold_evidence_ids else 0.0
                )

    unmatched_dataset_questions = sum(len(queue) for queue in question_queues.values())
    node_resolution_rate = (
        resolved_retrieved_id_references / total_retrieved_id_references
        if total_retrieved_id_references
        else 0.0
    )
    evidence_metadata_coverage = (
        evidence_tagged_result_references / retrieved_result_references
        if retrieved_result_references
        else 0.0
    )
    if not ledger_records:
        raise ValueError("Source contains no completed retrieval records")
    if unmatched_ledger_records or matched_records != len(ledger_records):
        raise ValueError(
            "Question IDs do not join exactly to the dataset; refusing a partial frontier"
        )
    if unresolved_evidence_ids:
        raise ValueError(
            f"Dataset contains {unresolved_evidence_ids} unresolved gold evidence IDs"
        )
    if not retrieved_result_references or evidence_metadata_coverage < 1.0:
        raise ValueError(
            "Retrieved candidates lack complete stable evidence-ID metadata; "
            "legacy checkpoints/ledgers are not valid for this evaluator"
        )

    frontier: dict[str, Any] = {}
    for k in k_values:
        metrics = per_k[k]
        summary: dict[str, Any] = {
            "retrieved_context_tokens": _distribution(metrics["retrieved_tokens"]),
            "retrieval_conditioned_context_token_reduction": _bootstrap_ci(
                metrics["context_reduction"], n_resamples=bootstrap_resamples, seed=seed + k
            ),
            "answer_overlap_proxy_hit_all": _bootstrap_ci(
                metrics["answer_proxy_hit_all"], n_resamples=bootstrap_resamples, seed=seed + 100 + k
            ),
            "answer_overlap_proxy_hit_with_gold_evidence": _bootstrap_ci(
                metrics["answer_proxy_hit_with_evidence"],
                n_resamples=bootstrap_resamples,
                seed=seed + 200 + k,
            ),
            "exact_source_recall": _bootstrap_ci(
                metrics["exact_source_recall"], n_resamples=bootstrap_resamples, seed=seed + 300 + k
            ),
            "exact_source_any_hit": _bootstrap_ci(
                metrics["exact_source_any_hit"], n_resamples=bootstrap_resamples, seed=seed + 400 + k
            ),
            "exact_source_all_hit": _bootstrap_ci(
                metrics["exact_source_all_hit"], n_resamples=bootstrap_resamples, seed=seed + 500 + k
            ),
            "exact_evidence_id_recall": _bootstrap_ci(
                metrics["exact_evidence_id_recall"],
                n_resamples=bootstrap_resamples,
                seed=seed + 600 + k,
            ),
            "exact_evidence_id_any_hit": _bootstrap_ci(
                metrics["exact_evidence_id_any_hit"],
                n_resamples=bootstrap_resamples,
                seed=seed + 700 + k,
            ),
            "exact_evidence_id_all_hit": _bootstrap_ci(
                metrics["exact_evidence_id_all_hit"],
                n_resamples=bootstrap_resamples,
                seed=seed + 800 + k,
            ),
        }
        if absolute_kv_bytes_per_token is not None:
            summary["model_kv_allocation_if_all_tokens_materialized"] = {
                "full_context_mean": mean(full_context_tokens) * absolute_kv_bytes_per_token,
                "retrieved_context_mean": mean(metrics["retrieved_tokens"]) * absolute_kv_bytes_per_token,
                "bytes_per_token": absolute_kv_bytes_per_token,
                "measured": False,
            }
        frontier[str(k)] = summary

    hypothesis_metrics = frontier[str(hypothesis_k)]
    observed_reduction = hypothesis_metrics[
        "retrieval_conditioned_context_token_reduction"
    ]["mean"]
    observed_exact_recall = hypothesis_metrics["exact_evidence_id_recall"]["mean"]
    data_valid = (
        node_resolution_rate >= min_node_resolution
        and evidence_metadata_coverage == 1.0
        and unmatched_ledger_records == 0
        and unresolved_evidence_ids == 0
        and matched_records == len(ledger_records)
    )
    hypothesis_passed = (
        data_valid
        and observed_reduction >= min_context_reduction
        and observed_exact_recall >= min_exact_evidence_recall
    )
    search_latency_summary = _distribution(search_durations_ms)
    rerank_latency_summary = _distribution(offline_rerank_durations_ms)
    retrieved_token_summary = hypothesis_metrics["retrieved_context_tokens"]
    budget_checks = {
        "retrieved_tokens_p95": {
            "cap": max_retrieved_tokens,
            "observed": retrieved_token_summary["p95"],
            "passed": (
                None
                if max_retrieved_tokens is None
                else bool(retrieved_token_summary["n"] and retrieved_token_summary["p95"] <= max_retrieved_tokens)
            ),
        },
        "search_latency_ms_p95": {
            "cap": max_search_latency_ms,
            "observed": search_latency_summary["p95"],
            "passed": (
                None
                if max_search_latency_ms is None
                else bool(search_latency_summary["n"] and search_latency_summary["p95"] <= max_search_latency_ms)
            ),
        },
        "offline_rerank_latency_ms_p95": {
            "cap": max_offline_rerank_latency_ms,
            "observed": rerank_latency_summary["p95"],
            "passed": (
                None
                if max_offline_rerank_latency_ms is None
                else bool(rerank_latency_summary["n"] and rerank_latency_summary["p95"] <= max_offline_rerank_latency_ms)
            ),
        },
    }
    budget_passed = all(
        check["passed"] is not False for check in budget_checks.values()
    )
    hypothesis_passed = hypothesis_passed and budget_passed

    limitations = [
        "Context-token reduction is prompt-side only; it is not measured KV-cache eviction, compression, or replacement.",
        "The answer-text overlap signal is weaker than downstream QA accuracy.",
        "Exact-source metrics require the original annotated turn text and do not credit paraphrased extracted facts.",
        "Search latency is recorded only when the source checkpoint reports it; model inference latency, throughput, memory, energy, and provider cost are not inferred.",
    ]
    if checkpoint_path is not None:
        limitations.extend(
            [
                "The MemoryBench checkpoint is partial and contains only completed historical searches.",
                "The checkpoint has no non-null cross-encoder scores, so lexical reranking is evaluated before any neural reranker interaction.",
            ]
        )
    else:
        limitations.append(
            "The current ledger omits sample_id, so duplicate question hashes are matched in dataset order."
        )

    result = {
        "schema": "hybridmind.retrieval-conditioned-context/v2",
        "benchmark": "locomo_retrieval_conditioned_effective_context_frontier",
        "hypothesis": {
            "k": hypothesis_k,
            "statement": (
                f"At k={hypothesis_k}, HybridMind retrieves at least "
                f"{min_exact_evidence_recall:.0%} of exact LoCoMo evidence IDs while reducing "
                f"retrieval-conditioned prompt tokens by at least {min_context_reduction:.0%} "
                "versus full-history prompting."
            ),
            "quality_metric": "exact evidence-ID recall",
            "observed_context_token_reduction": observed_reduction,
            "observed_exact_evidence_id_recall": observed_exact_recall,
            "data_valid": data_valid,
            "budget_valid": budget_passed,
            "passed": hypothesis_passed,
        },
        "inputs": {
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256_file(dataset_path),
            "tokenizer": token_counter.name,
            "k_values": list(k_values),
            "bootstrap_resamples": bootstrap_resamples,
            "seed": seed,
            **source_inputs,
        },
        "coverage": {
            "ledger_records": len(ledger_records),
            "matched_records": matched_records,
            "unmatched_ledger_records": unmatched_ledger_records,
            "unmatched_dataset_questions": unmatched_dataset_questions,
            "unresolved_gold_evidence_ids": unresolved_evidence_ids,
            "unique_retrieved_nodes_resolved": len(node_texts),
            "retrieved_node_reference_resolution_rate": node_resolution_rate,
            "retrieved_result_evidence_metadata_coverage": evidence_metadata_coverage,
        },
        "full_context_tokens": _distribution(full_context_tokens),
        "search_duration_ms": search_latency_summary,
        "offline_rerank_duration_ms": rerank_latency_summary,
        "budget_caps": budget_checks,
        "resource_and_cost_denominators": {
            "questions_in_source": len(ledger_records),
            "questions_matched": matched_records,
            "full_context_token_observations": len(full_context_tokens),
            "search_latency_observations": len(search_durations_ms),
            "offline_rerank_latency_observations": len(offline_rerank_durations_ms),
            "provider_cost": {"measured": False, "currency": None, "amount": None},
            "peak_accelerator_memory": {"measured": False, "bytes": None},
            "energy": {"measured": False, "joules": None},
        },
        "frontier": frontier,
        "limitations": limitations,
    }
    if checkpoint_path is not None and checkpoint_ranking == "local-lexical-rrf":
        result["ranking_hypothesis"] = _paired_ranking_hypothesis(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            variant_records=ledger_records,
            k=hypothesis_k,
            lexical_pool_size=local_lexical_pool_size,
            lexical_weight=local_lexical_weight,
            min_improvement=min_exact_source_recall_improvement,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 700 + hypothesis_k,
        )
    return result


def _parse_k_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("k values must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("k values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-ranking",
        choices=("combined", "local-lexical-rrf"),
        default="combined",
    )
    parser.add_argument("--local-lexical-pool-size", type=int, default=500)
    parser.add_argument("--local-lexical-weight", type=float, default=0.5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-values", type=_parse_k_values, default=DEFAULT_K_VALUES)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HF model name or 'regex'")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--hypothesis-k", type=int, default=10)
    parser.add_argument("--min-context-reduction", type=float, default=0.90)
    parser.add_argument("--min-exact-evidence-recall", type=float, default=0.80)
    parser.add_argument(
        "--min-answer-proxy-hit",
        type=float,
        dest="legacy_exact_recall_threshold",
        help="Deprecated alias; value thresholds exact evidence-ID recall, never answer overlap",
    )
    parser.add_argument("--min-node-resolution", type=float, default=0.99)
    parser.add_argument("--min-exact-source-recall-improvement", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-layers", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--element-bytes", type=float)
    parser.add_argument("--max-retrieved-tokens", type=int)
    parser.add_argument("--max-search-latency-ms", type=float)
    parser.add_argument("--max-offline-rerank-latency-ms", type=float)
    return parser.parse_args()


def _build_token_counter(args: argparse.Namespace) -> TokenCounter:
    if args.tokenizer == "regex":
        return RegexTokenCounter()
    try:
        return HuggingFaceTokenCounter(
            args.tokenizer,
            allow_download=args.allow_tokenizer_download,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Tokenizer {args.tokenizer!r} is not cached. Pass --allow-tokenizer-download "
            "or select an explicitly available tokenizer."
        ) from exc


def _absolute_kv_bytes_per_token(args: argparse.Namespace) -> float | None:
    values = (args.kv_layers, args.kv_heads, args.head_dim, args.element_bytes)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "Absolute KV estimates require --kv-layers, --kv-heads, --head-dim, and --element-bytes"
        )
    return kv_bytes_per_token(
        layers=args.kv_layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        element_bytes=args.element_bytes,
    )


def _print_summary(result: dict[str, Any]) -> None:
    print("k  context_token_reduction  exact_evidence_recall  answer_overlap_diagnostic")
    for k, metrics in result["frontier"].items():
        reduction = metrics["retrieval_conditioned_context_token_reduction"]["mean"]
        proxy = metrics["answer_overlap_proxy_hit_with_gold_evidence"]["mean"]
        exact = metrics["exact_evidence_id_recall"]["mean"]
        print(f"{int(k):>2} {reduction:>23.2%} {exact:>22.2%} {proxy:>26.2%}")
    hypothesis = result["hypothesis"]
    verdict = "PASSED" if hypothesis["passed"] else "FAILED"
    print(f"\nRetrieval-conditioned context hypothesis: {verdict}")
    print(
        f"Observed at k={hypothesis['k']}: "
        f"context-token reduction={hypothesis['observed_context_token_reduction']:.2%}, "
        f"exact evidence-ID recall={hypothesis['observed_exact_evidence_id_recall']:.2%}"
    )
    ranking_hypothesis = result.get("ranking_hypothesis")
    if ranking_hypothesis:
        ranking_verdict = "PASSED" if ranking_hypothesis["passed"] else "FAILED"
        improvement = ranking_hypothesis["improvement"]
        print(f"\nRanking hypothesis: {ranking_verdict}")
        print(
            f"Exact-source Recall@{ranking_hypothesis['k']}: "
            f"{ranking_hypothesis['baseline_mean']:.2%} -> "
            f"{ranking_hypothesis['variant_mean']:.2%}; "
            f"paired delta={improvement['mean']:.2%} "
            f"(95% CI {improvement['ci95_low']:.2%} to {improvement['ci95_high']:.2%})"
        )


def main() -> None:
    args = parse_args()
    token_counter = _build_token_counter(args)
    result = evaluate_frontier(
        ledger_path=args.ledger,
        dataset_path=args.dataset,
        database_path=args.database,
        token_counter=token_counter,
        checkpoint_path=args.checkpoint,
        checkpoint_ranking=args.checkpoint_ranking,
        local_lexical_pool_size=args.local_lexical_pool_size,
        local_lexical_weight=args.local_lexical_weight,
        k_values=args.k_values,
        hypothesis_k=args.hypothesis_k,
        min_context_reduction=args.min_context_reduction,
        min_exact_evidence_recall=(
            args.legacy_exact_recall_threshold
            if args.legacy_exact_recall_threshold is not None
            else args.min_exact_evidence_recall
        ),
        min_node_resolution=args.min_node_resolution,
        min_exact_source_recall_improvement=args.min_exact_source_recall_improvement,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        absolute_kv_bytes_per_token=_absolute_kv_bytes_per_token(args),
        max_retrieved_tokens=args.max_retrieved_tokens,
        max_search_latency_ms=args.max_search_latency_ms,
        max_offline_rerank_latency_ms=args.max_offline_rerank_latency_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print_summary(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
