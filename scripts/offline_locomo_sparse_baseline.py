"""Exact-evidence LoCoMo BM25S baseline with zero provider calls.

This measures a conversation-scoped sparse retriever on the checked-in LoCoMo
file. It is a real dataset baseline, but not a HybridMind end-to-end result:
there is no API, dense embedding, graph, reranker, or reader in this process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.bm25_index import BM25SBackend


DEFAULT_DATASET = (
    PROJECT_ROOT / "memorybench" / "data" / "benchmarks" / "locomo" / "locomo10.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "results" / "offline_locomo_bm25s.json"
K_VALUES = (1, 3, 5, 10, 25, 50, 100)
_SESSION_KEY = re.compile(r"^session_(\d+)$")
_DIA_ID = re.compile(r"D\d+:\d+")
_TOKEN = re.compile(r"\b\w+\b", re.UNICODE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_id(sample_id: str, dia_id: object) -> str:
    value = str(dia_id).strip()
    if not value:
        raise ValueError("empty LoCoMo evidence ID")
    return value if value.startswith("locomo:") else f"locomo:{sample_id}:{value}"


def _gold_evidence(sample_id: str, values: object) -> tuple[set[str], list[str]]:
    """Parse composite annotation cells without guessing malformed fragments."""
    if not isinstance(values, list):
        return set(), [repr(values)]
    result: set[str] = set()
    invalid: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        matches = list(_DIA_ID.finditer(text))
        remainder = _DIA_ID.sub("", text)
        if remainder.strip(" ;,\t\r\n"):
            invalid.append(text)
            continue
        if not matches:
            invalid.append(text)
            continue
        result.update(_canonical_id(sample_id, match.group(0)) for match in matches)
    return result, invalid


def _turns(item: dict) -> list[tuple[str, str]]:
    sample_id = str(item.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("LoCoMo item has no sample_id")
    conversation = item.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError(f"LoCoMo sample {sample_id} has no conversation object")
    keys = sorted(
        (key for key in conversation if _SESSION_KEY.fullmatch(key)),
        key=lambda key: int(_SESSION_KEY.fullmatch(key).group(1)),
    )
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in keys:
        messages = conversation.get(key)
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            evidence_id = _canonical_id(
                sample_id, message.get("dia_id") or f"{key}:{index}"
            )
            if evidence_id in seen:
                raise ValueError(f"duplicate evidence ID {evidence_id}")
            seen.add(evidence_id)
            rows.append((evidence_id, text))
    if not rows:
        raise ValueError(f"LoCoMo sample {sample_id} has no indexable turns")
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _bootstrap(values: list[float], *, seed: int, samples: int = 2000) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    rng = np.random.default_rng(seed)
    estimates = np.sort(
        array[rng.integers(0, count, size=(samples, count))].mean(axis=1)
    )
    low = float(estimates[math.floor(0.025 * (samples - 1))])
    high = float(estimates[math.ceil(0.975 * (samples - 1))])
    return {
        "mean": _mean(values),
        "ci95_low": low,
        "ci95_high": high,
        "n": count,
        "bootstrap_samples": samples,
    }


def evaluate(
    dataset: Path, *, seed: int = 20260813, strict_evidence: bool = False
) -> dict:
    started = time.perf_counter()
    data = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("LoCoMo dataset must be a non-empty array")

    per_k = {
        k: {"recall": [], "any_hit": [], "all_hit": [], "source_reduction": []}
        for k in K_VALUES
    }
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    evidence_questions = 0
    questions_total = 0
    turn_count = 0
    source_tokens = 0
    invalid_annotation_questions: list[dict] = []
    unresolved_evidence_questions: list[dict] = []

    for item in data:
        sample_id = str(item.get("sample_id") or "").strip()
        turns = _turns(item)
        turn_count += len(turns)
        corpus_ids = {node_id for node_id, _ in turns}
        corpus_tokens = sum(len(_TOKEN.findall(text)) for _, text in turns)
        source_tokens += corpus_tokens
        index = BM25SBackend()
        index.add_batch(turns)

        questions = item.get("qa")
        if not isinstance(questions, list):
            raise ValueError(f"LoCoMo sample {sample_id} has no qa array")
        for qa in questions:
            if not isinstance(qa, dict):
                continue
            questions_total += 1
            gold, invalid_fragments = _gold_evidence(
                sample_id, qa.get("evidence", [])
            )
            if invalid_fragments:
                invalid_annotation_questions.append(
                    {
                        "sample_id": sample_id,
                        "question_sha256": hashlib.sha256(
                            str(qa.get("question") or "").encode("utf-8")
                        ).hexdigest(),
                        "invalid_fragments": invalid_fragments,
                    }
                )
                continue
            if not gold:
                continue
            query = str(qa.get("question") or "").strip()
            if not query:
                raise ValueError(f"LoCoMo sample {sample_id} has an empty question")
            missing = gold - corpus_ids
            if missing:
                unresolved_evidence_questions.append(
                    {
                        "sample_id": sample_id,
                        "question_sha256": hashlib.sha256(
                            query.encode("utf-8")
                        ).hexdigest(),
                        "unresolved_evidence_ids": sorted(missing),
                    }
                )
                if strict_evidence:
                    raise ValueError(
                        f"LoCoMo sample {sample_id} has unresolved evidence IDs: {sorted(missing)}"
                    )
                continue
            evidence_questions += 1
            query_start = time.perf_counter()
            # bm25s currently emits one tqdm line per query. Suppress only that
            # progress stream so the machine-readable run log stays bounded.
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stderr(sink):
                    ranked = index.search(
                        query, top_k=min(max(K_VALUES), len(turns))
                    )
            latencies_ms.append((time.perf_counter() - query_start) * 1000.0)
            ranked_ids = [node_id for node_id, _ in ranked]
            first_rank = next(
                (rank for rank, node_id in enumerate(ranked_ids, 1) if node_id in gold),
                None,
            )
            reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

            text_by_id = dict(turns)
            for k in K_VALUES:
                selected = ranked_ids[:k]
                hits = gold & set(selected)
                retrieved_tokens = sum(
                    len(_TOKEN.findall(text_by_id[node_id])) for node_id in selected
                )
                per_k[k]["recall"].append(len(hits) / len(gold))
                per_k[k]["any_hit"].append(float(bool(hits)))
                per_k[k]["all_hit"].append(float(hits == gold))
                per_k[k]["source_reduction"].append(
                    1.0 - min(retrieved_tokens, corpus_tokens) / max(corpus_tokens, 1)
                )

    if evidence_questions == 0:
        raise ValueError("LoCoMo dataset has no evidence-bearing questions")

    latency_sorted = sorted(latencies_ms)
    percentile = lambda p: latency_sorted[round((len(latency_sorted) - 1) * p)]
    return {
        "schema_version": "hybridmind-offline-locomo-sparse/v1",
        "classification": "measured_offline_dataset_baseline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "external_network_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "actual_external_cost_usd": 0.0,
        },
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": _sha256(dataset),
            "samples": len(data),
            "turns": turn_count,
            "questions_total": questions_total,
            "evidence_questions": evidence_questions,
            "excluded_questions_invalid_evidence_annotation": len(
                invalid_annotation_questions
            ),
            "invalid_evidence_annotations": invalid_annotation_questions,
            "excluded_questions_unresolved_evidence": len(
                unresolved_evidence_questions
            ),
            "unresolved_evidence_annotations": unresolved_evidence_questions,
            "source_tokens_regex_proxy": source_tokens,
        },
        "retriever": {
            "name": "BM25SBackend",
            "scope": "one index per LoCoMo conversation",
            "query": "question text only",
            "documents": "raw turn text only",
            "relevance": "exact canonical evidence ID",
        },
        "metrics": {
            "mrr_first_exact_evidence": _bootstrap(
                reciprocal_ranks, seed=seed + 1
            ),
            "at_k": {
                str(k): {
                    "exact_evidence_recall": _bootstrap(
                        values["recall"], seed=seed + k * 10 + 1
                    ),
                    "exact_evidence_any_hit": _bootstrap(
                        values["any_hit"], seed=seed + k * 10 + 2
                    ),
                    "exact_evidence_all_hit": _bootstrap(
                        values["all_hit"], seed=seed + k * 10 + 3
                    ),
                    "prompt_source_reduction_regex_token_proxy": _bootstrap(
                        values["source_reduction"], seed=seed + k * 10 + 4
                    ),
                }
                for k, values in per_k.items()
            },
            "query_latency_ms": {
                "p50": percentile(0.50),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
                "mean": _mean(latencies_ms),
                "samples": len(latencies_ms),
            },
        },
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_limits": [
            "This is a real LoCoMo exact-evidence sparse baseline, not a HybridMind end-to-end result.",
            "It excludes the API, dense embeddings, graph, cross-encoder, decomposition, salience, and reader.",
            "Regex token counts are a prompt-source proxy, not model tokenizer counts or KV-cache allocation.",
            "No answer accuracy is measured.",
        ],
    }


def _atomic_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--strict-evidence",
        action="store_true",
        help="abort instead of transparently excluding broken annotations",
    )
    args = parser.parse_args(argv)
    report = evaluate(
        args.dataset.resolve(),
        seed=args.seed,
        strict_evidence=args.strict_evidence,
    )
    _atomic_write(args.output.resolve(), report)
    at_10 = report["metrics"]["at_k"]["10"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "external_network_calls": 0,
                "evidence_questions": report["dataset"]["evidence_questions"],
                "exact_evidence_recall_at_10": at_10["exact_evidence_recall"]["mean"],
                "exact_evidence_any_hit_at_10": at_10["exact_evidence_any_hit"]["mean"],
                "source_reduction_at_10": at_10[
                    "prompt_source_reduction_regex_token_proxy"
                ]["mean"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
