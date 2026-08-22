"""Cluster exact-evidence failures for raw versus speaker-prefixed LoCoMo BM25S.

This is an analysis-only companion to ``offline_locomo_sparse_experiments``.
Retrieval is delegated to that source-preserving evaluator.  Gold evidence is
read only after both conditions have produced ranked source IDs, for paired
scoring, diagnostics, and bounded audit examples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

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
    BOOTSTRAP_SEED,
    SPLIT_SEED,
    _evaluate_variant,
    _split_ids,
    _turn_records,
)


SCHEMA = "hybridmind.offline-locomo-sparse-failure-analysis/v1"
SESSION_KEY = re.compile(r"^session_(\d+)$")
MAX_TOP_IDS = 25
DEFAULT_REPRESENTATIVE_LIMIT = 6


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
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


def _question_id(sample_id: str, qa_index: int, question: str) -> str:
    return hashlib.sha256(
        f"{sample_id}\0{qa_index}\0{question}".encode("utf-8")
    ).hexdigest()


def _sample_ids(data: list[dict], *, split_seed: str, all_conversations: bool) -> dict[str, Any]:
    ids = sorted(str(item.get("sample_id") or "").strip() for item in data)
    if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("LoCoMo sample IDs must be unique and non-empty")
    if all_conversations:
        return {
            "selection": "all_declared_conversations",
            "split_seed": None,
            "development": [],
            "held_out": ids,
            "selected": ids,
        }
    split = _split_ids(data, split_seed)
    return {
        "selection": "held_out_conversations",
        "split_seed": split_seed,
        **split,
        "selected": split["held_out"],
    }


def _retrieval_input_fingerprint(
    data: list[dict], sample_ids: Iterable[str], variant: str,
) -> str:
    """Hash only documents and questions that enter retrieval.

    This deliberately does not inspect ``answer`` or ``evidence``.  It is
    emitted twice, once for each representation, to make a gold-free input
    boundary auditable.
    """
    selected = set(sample_ids)
    payload: list[dict[str, Any]] = []
    from scripts.offline_locomo_sparse_experiments import _keys

    for item in sorted(data, key=lambda row: str(row.get("sample_id") or "")):
        sample_id = str(item.get("sample_id") or "").strip()
        if sample_id not in selected:
            continue
        records = _turn_records(item)
        keys = [
            {"key_id": f"{record['source_id']}|{key_type}", "text": key_text}
            for record in records
            for key_type, key_text in _keys(record, variant)
        ]
        questions = [
            {
                "question_id": _question_id(sample_id, index, str(qa.get("question") or "").strip()),
                "question": str(qa.get("question") or "").strip(),
            }
            for index, qa in enumerate(item.get("qa") or [])
            if isinstance(qa, dict)
        ]
        payload.append({"sample_id": sample_id, "keys": keys, "questions": questions})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _question_metadata(data: list[dict], sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Build post-retrieval audit metadata, including gold source text."""
    metadata: dict[str, dict[str, Any]] = {}
    for item in data:
        sample_id = str(item.get("sample_id") or "").strip()
        if sample_id not in sample_ids:
            continue
        records = _turn_records(item)
        record_by_id = {record["source_id"]: record for record in records}
        for qa_index, qa in enumerate(item.get("qa") or []):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            question_id = _question_id(sample_id, qa_index, question)
            gold, invalid = _gold_evidence(sample_id, qa.get("evidence", []))
            if invalid or not gold or not gold.issubset(record_by_id):
                continue
            metadata[question_id] = {
                "question": question,
                "category": {
                    1: "single-hop", 2: "temporal", 3: "multi-hop",
                    4: "world-knowledge", 5: "adversarial",
                }.get(qa.get("category"), "unknown"),
                "sample_id": sample_id,
                "gold": sorted(gold),
                "records": record_by_id,
            }
    return metadata


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(text)}


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = " ".join(text.casefold().split())
    normalized_phrase = " ".join(phrase.casefold().split())
    if not normalized_phrase or len(_tokens(normalized_phrase)) < 1:
        return False
    return bool(re.search(
        rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text,
    ))


def _first_rank(row: dict[str, Any]) -> int | None:
    for rank, source_id in enumerate(row["ranked_ids_at_25"], 1):
        if source_id in set(row["gold"]):
            return rank
    return None


def _rank(row: dict[str, Any], source_id: str) -> int | None:
    try:
        return row["ranked_ids_at_25"].index(source_id) + 1
    except ValueError:
        return None


def _status(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "unchanged"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


def _overlap_diagnostics(question: str, gold_text: str) -> dict[str, Any]:
    question_terms = _tokens(question)
    evidence_terms = _tokens(gold_text)
    intersection = question_terms & evidence_terms
    union = question_terms | evidence_terms
    return {
        "question_term_count": len(question_terms),
        "gold_evidence_term_count": len(evidence_terms),
        "shared_term_count": len(intersection),
        "question_term_recall": len(intersection) / max(len(question_terms), 1),
        "jaccard": len(intersection) / max(len(union), 1),
    }


def _representative(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    records = metadata["records"]
    gold_audit = [
        {
            "evidence_id": source_id,
            "text": records[source_id]["text"][:240],
            "speaker": records[source_id]["speaker"],
        }
        for source_id in metadata["gold"]
    ]
    return {
        "question_id": row["question_id"],
        "sample_id": metadata["sample_id"],
        "category": metadata["category"],
        "question": metadata["question"][:240],
        "gold_evidence": gold_audit,
        "raw_top_ids_at_10": row["raw_top_ids_at_10"],
        "speaker_prefix_top_ids_at_10": row["speaker_prefix_top_ids_at_10"],
        "raw_first_gold_rank": row["raw_first_gold_rank"],
        "speaker_prefix_first_gold_rank": row["speaker_prefix_first_gold_rank"],
        "raw_recall_at_10": row["raw_recall_at_10"],
        "speaker_prefix_recall_at_10": row["speaker_prefix_recall_at_10"],
        "lexical": row["lexical"],
    }


def _validate_rows(raw_rows: list[dict], speaker_rows: list[dict]) -> None:
    raw_ids = [row.get("question_id") for row in raw_rows]
    speaker_ids = [row.get("question_id") for row in speaker_rows]
    if not raw_ids or len(raw_ids) != len(set(raw_ids)) or raw_ids != speaker_ids:
        raise ValueError("raw and speaker-prefix rows are not an identical ordered pair")
    for row in (*raw_rows, *speaker_rows):
        ranked = row.get("ranked_ids_at_25")
        gold = row.get("gold")
        if not isinstance(ranked, list) or len(ranked) > MAX_TOP_IDS:
            raise ValueError("retrieval rows must contain at most 25 ranked IDs")
        if len(ranked) != len(set(ranked)) or not isinstance(gold, list) or not gold:
            raise ValueError("retrieval rows have malformed exact-evidence IDs")


def _paired_failure_analysis(
    raw_rows: list[dict], speaker_rows: list[dict], metadata: dict[str, dict[str, Any]],
    *, representative_limit: int = DEFAULT_REPRESENTATIVE_LIMIT,
) -> dict[str, Any]:
    _validate_rows(raw_rows, speaker_rows)
    speaker_by_id = {row["question_id"]: row for row in speaker_rows}
    if set(metadata) != {row["question_id"] for row in raw_rows}:
        raise ValueError("post-retrieval metadata does not exactly cover paired rows")

    changes: list[dict[str, Any]] = []
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlap_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mention_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_transition = Counter()
    catastrophic = Counter()
    representatives: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for raw in sorted(raw_rows, key=lambda row: row["question_id"]):
        question_id = raw["question_id"]
        speaker = speaker_by_id[question_id]
        info = metadata[question_id]
        if raw["gold"] != info["gold"] or speaker["gold"] != info["gold"]:
            raise ValueError("paired evaluator gold IDs disagree with audit metadata")
        raw_recall = float(raw["recall_at_k"]["10"])
        speaker_recall = float(speaker["recall_at_k"]["10"])
        raw_mrr = float(raw["reciprocal_rank"])
        speaker_mrr = float(speaker["reciprocal_rank"])
        recall_delta = speaker_recall - raw_recall
        mrr_delta = speaker_mrr - raw_mrr
        status = _status(recall_delta)
        raw_first = _first_rank(raw)
        speaker_first = _first_rank(speaker)
        raw_any = bool(raw["any_hit_at_k"]["10"])
        speaker_any = bool(speaker["any_hit_at_k"]["10"])
        raw_all = bool(raw["all_hit_at_k"]["10"])
        speaker_all = bool(speaker["all_hit_at_k"]["10"])
        catastrophic[f"raw_{'hit' if raw_any else 'miss'}_speaker_{'hit' if speaker_any else 'miss'}_at_10"] += 1
        if raw_all and not speaker_all:
            catastrophic["all_gold_hit_to_not_all_gold_hit_at_10"] += 1
        gold_text = " ".join(info["records"][source_id]["text"] for source_id in info["gold"])
        lexical = _overlap_diagnostics(info["question"], gold_text)
        speakers = sorted({
            info["records"][source_id]["speaker"]
            for source_id in info["gold"]
            if info["records"][source_id]["speaker"]
        })
        lexical["gold_speakers"] = speakers
        lexical["speaker_named_in_question"] = any(
            _contains_phrase(info["question"], name) for name in speakers
        )
        lexical["speaker_name_present_in_evidence_text"] = any(
            _contains_phrase(info["records"][source_id]["text"], name)
            for source_id in info["gold"] for name in speakers
        )
        lexical_bucket = (
            "none" if lexical["question_term_recall"] == 0 else
            "low_0_to_25" if lexical["question_term_recall"] <= 0.25 else
            "medium_25_to_50" if lexical["question_term_recall"] <= 0.50 else
            "high_over_50"
        )
        mention_bucket = "speaker_named" if lexical["speaker_named_in_question"] else "speaker_not_named"
        overlap_rows[lexical_bucket].append({"delta": recall_delta, "status": status})
        mention_rows[mention_bucket].append({"delta": recall_delta, "status": status})

        for source_id in info["gold"]:
            raw_rank = _rank(raw, source_id)
            speaker_rank = _rank(speaker, source_id)
            if raw_rank is None and speaker_rank is not None:
                transition = "raw_miss_speaker_hit"
            elif raw_rank is not None and speaker_rank is None:
                transition = "raw_hit_speaker_miss"
            elif raw_rank is not None and speaker_rank is not None and speaker_rank < raw_rank:
                transition = "improved_rank"
            elif raw_rank is not None and speaker_rank is not None and speaker_rank > raw_rank:
                transition = "regressed_rank"
            else:
                transition = "unchanged_rank_or_both_missed"
            evidence_transition[transition] += 1

        change = {
            "question_id": question_id,
            "sample_id": info["sample_id"],
            "category": info["category"],
            "status": status,
            "raw_recall_at_10": raw_recall,
            "speaker_prefix_recall_at_10": speaker_recall,
            "recall_delta_speaker_minus_raw": recall_delta,
            "raw_mrr": raw_mrr,
            "speaker_prefix_mrr": speaker_mrr,
            "mrr_delta_speaker_minus_raw": mrr_delta,
            "raw_first_gold_rank": raw_first,
            "speaker_prefix_first_gold_rank": speaker_first,
            "first_gold_rank_delta": (
                None if raw_first is None or speaker_first is None
                else speaker_first - raw_first
            ),
            "raw_top_ids_at_10": raw["ranked_ids_at_25"][:10],
            "speaker_prefix_top_ids_at_10": speaker["ranked_ids_at_25"][:10],
            "lexical": lexical,
        }
        changes.append(change)
        category_rows[info["category"]].append(change)
        representatives[status].append(
            _representative({**change}, info)
        )

    def grouped(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        result = {}
        for key in sorted(rows):
            selected = rows[key]
            counts = Counter(row["status"] for row in selected)
            result[key] = {
                "n": len(selected),
                "counts": {status: counts.get(status, 0) for status in ("improved", "regressed", "unchanged")},
                "mean_recall_delta": _mean(row["delta"] for row in selected),
            }
        return result

    category_summary = {}
    for category in sorted(category_rows):
        rows = category_rows[category]
        counts = Counter(row["status"] for row in rows)
        category_summary[category] = {
            "n": len(rows),
            "counts": {status: counts.get(status, 0) for status in ("improved", "regressed", "unchanged")},
            "mean_recall_delta_at_10": _mean(row["recall_delta_speaker_minus_raw"] for row in rows),
            "mean_mrr_delta": _mean(row["mrr_delta_speaker_minus_raw"] for row in rows),
            "mean_first_gold_rank_delta_observed": _mean(
                row["first_gold_rank_delta"] for row in rows
                if row["first_gold_rank_delta"] is not None
            ),
        }

    return {
        "n_questions": len(changes),
        "paired_changes": changes,
        "category_summary": category_summary,
        "catastrophic_miss_transitions_at_10": dict(sorted(catastrophic.items())),
        "evidence_rank_transitions_within_top_25": dict(sorted(evidence_transition.items())),
        "lexical_failure_clusters": {
            "by_question_term_recall_bucket": grouped(overlap_rows),
            "by_speaker_mention": grouped(mention_rows),
        },
        "representatives": {
            status: representatives.get(status, [])[:representative_limit]
            for status in ("improved", "regressed", "unchanged")
        },
    }


def run(
    dataset: Path, *, split_seed: str = SPLIT_SEED,
    all_conversations: bool = False,
    representative_limit: int = DEFAULT_REPRESENTATIVE_LIMIT,
) -> dict[str, Any]:
    started = time.perf_counter()
    if representative_limit < 0 or representative_limit > 20:
        raise ValueError("representative_limit must be between 0 and 20")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("LoCoMo dataset must contain multiple conversations")
    scope = _sample_ids(data, split_seed=split_seed, all_conversations=all_conversations)
    selected = set(scope["selected"])
    raw_input_hash = _retrieval_input_fingerprint(data, selected, "raw")
    speaker_input_hash = _retrieval_input_fingerprint(data, selected, "speaker_prefix")
    raw = _evaluate_variant(data, selected, "raw", seed=BOOTSTRAP_SEED + 41_000)
    speaker = _evaluate_variant(data, selected, "speaker_prefix", seed=BOOTSTRAP_SEED + 42_000)
    metadata = _question_metadata(data, selected)
    analysis = _paired_failure_analysis(
        raw["_rows"], speaker["_rows"], metadata,
        representative_limit=representative_limit,
    )
    status = _git_output("status", "--porcelain=v1", "-z") or ""
    source_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA,
        "classification": "measured_offline_exact_evidence_failure_clustering",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0,
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
            "source": {"path": str(source_path), "sha256": _sha256(source_path)},
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree": {
                "dirty": bool(status),
                "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "selection_seed": split_seed,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "scope": scope,
        "information_boundary": {
            "retrieval_conditions": ["raw", "speaker_prefix"],
            "retrieval_inputs_exclude_answer_and_evidence": True,
            "retrieval_input_sha256": {
                "raw": raw_input_hash,
                "speaker_prefix": speaker_input_hash,
            },
            "gold_used_after_retrieval_for": [
                "exact evidence scoring", "rank transitions", "lexical diagnostics", "audit representatives",
            ],
        },
        "conditions": {
            "raw": {key: value for key, value in raw.items() if key != "_rows"},
            "speaker_prefix": {key: value for key, value in speaker.items() if key != "_rows"},
        },
        "analysis": analysis,
        "interpretation_limits": [
            "This is sparse exact-evidence retrieval, not answer accuracy or end-to-end HybridMind evaluation.",
            "Failure clusters are descriptive and do not establish causal superiority of speaker prefixes.",
            "Only the declared held-out conversations are analyzed unless --all-conversations is used.",
            "Ranks beyond the source-preserving evaluator's top-25 output are represented as misses.",
            "Regex term overlap and exact speaker-name matching are deterministic diagnostics, not semantic attribution.",
            "Per-query wall-clock timings are excluded from failure clustering and require a separate repeated paired benchmark.",
        ],
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
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
    parser.add_argument("--all-conversations", action="store_true")
    parser.add_argument("--representative-limit", type=int, default=DEFAULT_REPRESENTATIVE_LIMIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or (
        PROJECT_ROOT / "experiments" / "results"
        / f"offline-locomo-sparse-failure-analysis-v1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report = run(
        args.dataset.resolve(), split_seed=args.split_seed,
        all_conversations=args.all_conversations,
        representative_limit=args.representative_limit,
    )
    _atomic_write(output.resolve(), report)
    print(json.dumps({
        "output": str(output.resolve()),
        "provider_calls": 0,
        "n_questions": report["analysis"]["n_questions"],
        "catastrophic_miss_transitions_at_10": report["analysis"]["catastrophic_miss_transitions_at_10"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
