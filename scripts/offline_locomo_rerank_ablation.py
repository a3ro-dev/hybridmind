"""Strictly offline fixed-pool LoCoMo cross-encoder reranking ablation.

This is an exploratory mechanism test, not a production configuration change.
It compares two fixed candidate generators (raw BM25S and a speaker-prefixed
BM25S field) before and after one fixed local cross-encoder.  The candidate
pool is selected without gold evidence and is never changed after scoring.

The default model path is a checked-in local Hugging Face snapshot.  Loading
is deliberately local-only: this module sets the offline Transformers/HF
environment flags and passes ``local_files_only=True`` to every model loader.
No provider, embedding, reader, or network calls are made by this script.
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
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

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
    _split_ids,
    _turn_records,
)
from storage.bm25_index import BM25SBackend  # noqa: E402


SCHEMA = "hybridmind.offline-locomo-rerank-ablation/v1"
FAILED_SCHEMA = "hybridmind.offline-locomo-rerank-ablation-failure/v1"
DEFAULT_MODEL_PATH = Path(
    r"C:\Users\akshat\.cache\huggingface\hub\models--cross-encoder--ms-marco-MiniLM-L-6-v2"
    r"\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8"
)
DEFAULT_POOL_SIZE = 25
DEFAULT_TOP_K = 10
SPLIT_SEED = "20260822-rerank-fixed-pool"
BOOTSTRAP_SEED = 20260822
CATEGORY = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "world-knowledge",
    5: "adversarial",
}
CONDITIONS = ("raw", "speaker_prefix")


class ValidationError(ValueError):
    """Raised when the experiment cannot proceed without guessing."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("bm25s", "numpy", "PyStemmer", "torch", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _question_id(sample_id: str, qa_index: int, question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"locomo:{sample_id}:q{qa_index}:{digest}"


def _category(value: object) -> str:
    try:
        return CATEGORY.get(int(value), f"unknown-{value}")
    except (TypeError, ValueError):
        return "unknown"


def _field_text(record: Mapping[str, str], field: str) -> str:
    text = str(record.get("text") or "").strip()
    if field == "raw":
        return text
    if field == "speaker_prefix":
        speaker = str(record.get("speaker") or "").strip()
        return f"{speaker}: {text}" if speaker else text
    raise ValidationError(f"unknown candidate field: {field}")


def _build_index(records: Sequence[Mapping[str, str]], field: str) -> tuple[BM25SBackend, dict[str, str]]:
    pairs = []
    text_by_id: dict[str, str] = {}
    for record in records:
        evidence_id = str(record.get("source_id") or "").strip()
        if not evidence_id.startswith("locomo:"):
            raise ValidationError("candidate records must use canonical LoCoMo evidence IDs")
        text = _field_text(record, field)
        if not text:
            raise ValidationError(f"empty {field} candidate text for {evidence_id}")
        if evidence_id in text_by_id:
            raise ValidationError(f"duplicate candidate evidence ID: {evidence_id}")
        text_by_id[evidence_id] = text
        pairs.append((evidence_id, text))
    index = BM25SBackend()
    index.add_batch(pairs)
    return index, text_by_id


def _retrieve(
    index: BM25SBackend,
    query: str,
    *,
    pool_size: int,
    known_ids: set[str],
) -> tuple[list[str], list[float], float, int]:
    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stderr(sink):
            hits = index.search(query, top_k=min(pool_size, len(known_ids)))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ranked: list[tuple[str, float, int]] = []
    seen: set[str] = set()
    for order, hit in enumerate(hits):
        if not isinstance(hit, (tuple, list)) or len(hit) != 2:
            raise ValidationError("BM25S returned a malformed hit")
        evidence_id, score = str(hit[0]), float(hit[1])
        if evidence_id not in known_ids:
            raise ValidationError(f"BM25S returned unknown evidence ID: {evidence_id}")
        if evidence_id in seen:
            raise ValidationError(f"BM25S returned duplicate evidence ID: {evidence_id}")
        if not math.isfinite(score):
            raise ValidationError("BM25S returned a non-finite score")
        seen.add(evidence_id)
        ranked.append((evidence_id, score, order))
    # Retain deterministic rank order for exact-score ties.  The source order
    # is an input to the fixed candidate generator, never a gold signal.
    ranked.sort(key=lambda row: (-row[1], row[2], row[0]))
    target = min(pool_size, len(known_ids))
    padding = sorted(known_ids - seen)[: max(0, target - len(ranked))]
    ranked.extend(
        (evidence_id, 0.0, len(ranked) + offset)
        for offset, evidence_id in enumerate(padding)
    )
    selected = ranked[:target]
    return (
        [row[0] for row in selected],
        [row[1] for row in selected],
        elapsed_ms,
        len(padding),
    )


class LocalMiniLMScorer:
    """Local-only cross-encoder scorer for ``query, candidate_text`` pairs."""

    def __init__(self, model_path: Path, *, batch_size: int = 16) -> None:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "1"
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_dir():
            raise ValidationError(f"local reranker snapshot does not exist: {self.model_path}")
        self.batch_size = max(1, int(batch_size))
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ValidationError(f"local reranker dependencies unavailable: {exc}") from exc
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path), local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ValidationError(f"local reranker snapshot failed to load: {exc}") from exc
        self.model.eval()
        self.call_count = 0
        self.last_duration_ms = 0.0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not isinstance(query, str) or not all(isinstance(text, str) for text in texts):
            raise ValidationError("reranker inputs must be strings")
        started = time.perf_counter()
        values: list[float] = []
        with self._torch.inference_mode():
            for offset in range(0, len(texts), self.batch_size):
                batch = list(texts[offset : offset + self.batch_size])
                encoded = self.tokenizer(
                    [query] * len(batch), batch,
                    padding=True, truncation=True, max_length=512,
                    return_tensors="pt",
                )
                logits = self.model(**encoded).logits.reshape(-1).detach().cpu().tolist()
                values.extend(float(value) for value in logits)
        self.call_count += 1
        self.last_duration_ms = (time.perf_counter() - started) * 1000.0
        if len(values) != len(texts) or not all(math.isfinite(value) for value in values):
            raise ValidationError("local reranker returned malformed or non-finite scores")
        return values


def _rerank(
    scorer: Any,
    query: str,
    candidate_ids: Sequence[str],
    candidate_texts: Mapping[str, str],
) -> tuple[list[str], list[float], float]:
    started = time.perf_counter()
    values = scorer.score(query, [candidate_texts[evidence_id] for evidence_id in candidate_ids])
    values = [float(value) for value in values]
    if len(values) != len(candidate_ids):
        raise ValidationError("reranker returned a score count different from the candidate pool")
    if not all(math.isfinite(value) for value in values):
        raise ValidationError("reranker returned a non-finite score")
    ranked = sorted(
        zip(candidate_ids, values, range(len(candidate_ids))),
        key=lambda row: (-row[1], row[2], row[0]),
    )
    return [row[0] for row in ranked], [row[1] for row in ranked], (time.perf_counter() - started) * 1000.0


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float],
    *,
    seed: int,
    samples: int = 4000,
) -> dict[str, Any]:
    """Question-weighted mean with whole-conversation cluster resampling."""
    if not rows:
        return {
            "mean": None, "ci95_low": None, "ci95_high": None,
            "n": 0, "n_clusters": 0,
            "bootstrap_unit": "whole conversation",
        }
    cluster_rows: dict[str, list[float]] = {}
    for row in rows:
        cluster = str(row["sample_id"])
        cluster_rows.setdefault(cluster, []).append(float(value(row)))
    clusters = sorted(cluster_rows)
    array = np.asarray(
        [item for cluster in clusters for item in cluster_rows[cluster]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = []
    sampled_cluster_indexes = rng.integers(
        0, len(clusters), size=(samples, len(clusters)),
    )
    for sampled in sampled_cluster_indexes:
        sampled_values = [
            item
            for cluster_index in sampled
            for item in cluster_rows[clusters[int(cluster_index)]]
        ]
        estimates.append(float(np.mean(sampled_values)))
    estimates = np.sort(np.asarray(estimates, dtype=np.float64))
    return {
        "mean": float(array.mean()),
        "ci95_low": float(estimates[math.floor(0.025 * (samples - 1))]),
        "ci95_high": float(estimates[math.ceil(0.975 * (samples - 1))]),
        "n": int(len(array)),
        "n_clusters": len(clusters),
        "bootstrap_samples": samples,
        "bootstrap_unit": "whole conversation",
    }


def _metrics(rows: Sequence[Mapping[str, Any]], *, condition: str, split: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("split") == split and row.get("status") == "ok"]
    categories: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in selected}):
        cat_rows = [row for row in selected if str(row["category"]) == category]
        categories[category] = {
            "n": len(cat_rows),
            "pre_recall_at_10": _cluster_bootstrap(
                cat_rows,
                lambda row: float(row["conditions"][condition]["pre_recall_at_10"]),
                seed=BOOTSTRAP_SEED + len(category),
            ),
            "post_recall_at_10": _cluster_bootstrap(
                cat_rows,
                lambda row: float(row["conditions"][condition]["post_recall_at_10"]),
                seed=BOOTSTRAP_SEED + 100 + len(category),
            ),
            "paired_delta_recall_at_10": _cluster_bootstrap(
                cat_rows,
                lambda row: (
                    float(row["conditions"][condition]["post_recall_at_10"])
                    - float(row["conditions"][condition]["pre_recall_at_10"])
                ),
                seed=BOOTSTRAP_SEED + 200 + len(category),
            ),
        }
    return {
        "split": split,
        "condition": condition,
        "n_ok": len(selected),
        "n_failed": sum(1 for row in rows if row.get("split") == split and row.get("status") != "ok"),
        "candidate_pool_oracle_recall": _cluster_bootstrap(
            selected,
            lambda row: float(row["conditions"][condition]["oracle_recall_at_pool"]),
            seed=BOOTSTRAP_SEED + 1,
        ),
        "pre_fixed_pool_recall_at_10": _cluster_bootstrap(
            selected,
            lambda row: float(row["conditions"][condition]["pre_recall_at_10"]),
            seed=BOOTSTRAP_SEED + 2,
        ),
        "post_fixed_pool_recall_at_10": _cluster_bootstrap(
            selected,
            lambda row: float(row["conditions"][condition]["post_recall_at_10"]),
            seed=BOOTSTRAP_SEED + 3,
        ),
        "paired_delta_recall_at_10": _cluster_bootstrap(
            selected,
            lambda row: (
                float(row["conditions"][condition]["post_recall_at_10"])
                - float(row["conditions"][condition]["pre_recall_at_10"])
            ),
            seed=BOOTSTRAP_SEED + 4,
        ),
        "pre_mrr": _cluster_bootstrap(
            selected,
            lambda row: float(row["conditions"][condition]["pre_reciprocal_rank"]),
            seed=BOOTSTRAP_SEED + 5,
        ),
        "post_mrr": _cluster_bootstrap(
            selected,
            lambda row: float(row["conditions"][condition]["post_reciprocal_rank"]),
            seed=BOOTSTRAP_SEED + 6,
        ),
        "paired_delta_mrr": _cluster_bootstrap(
            selected,
            lambda row: (
                float(row["conditions"][condition]["post_reciprocal_rank"])
                - float(row["conditions"][condition]["pre_reciprocal_rank"])
            ),
            seed=BOOTSTRAP_SEED + 7,
        ),
        "by_category": categories,
    }


def _snapshot_manifest(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise ValidationError(f"local reranker snapshot does not exist: {model_path}")
    files: list[dict[str, Any]] = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        files.append({
            "path": str(path.relative_to(model_path)).replace("\\", "/"),
            "bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        })
    if not files:
        raise ValidationError("local reranker snapshot has no files")
    return {
        "path": str(model_path.resolve()),
        "files": files,
        "manifest_sha256": _value_sha256(files),
    }


def _validate_pool(pool_size: int, top_k: int) -> None:
    if top_k != DEFAULT_TOP_K:
        raise ValidationError(f"final top_k is fixed at {DEFAULT_TOP_K}")
    if pool_size < top_k:
        raise ValidationError(
            "fixed candidate pool must be at least final top_k and both must be positive"
        )


def _machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def evaluate(
    dataset: Path,
    *,
    scorer: Any | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    pool_size: int = DEFAULT_POOL_SIZE,
    top_k: int = DEFAULT_TOP_K,
    split_seed: str = SPLIT_SEED,
    batch_size: int = 16,
) -> dict[str, Any]:
    _validate_pool(pool_size, top_k)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    dataset = Path(dataset)
    data = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        raise ValidationError("LoCoMo dataset must contain at least two conversations")
    split = _split_ids(data, split_seed)
    split_by_id = {sample_id: name for name, ids in split.items() for sample_id in ids}
    if scorer is None:
        scorer = LocalMiniLMScorer(Path(model_path), batch_size=batch_size)
    rows: list[dict[str, Any]] = []
    build_timing: dict[str, float] = {condition: 0.0 for condition in CONDITIONS}
    rerank_calls = 0
    rerank_ms: list[float] = []
    for item in data:
        sample_id = str(item.get("sample_id") or "").strip()
        if not sample_id or sample_id not in split_by_id:
            raise ValidationError("LoCoMo sample IDs must be non-empty and appear in the split")
        split_name = split_by_id[sample_id]
        records = _turn_records(item)
        known_ids = {str(record["source_id"]) for record in records}
        authoritative_texts = {
            str(record["source_id"]): str(record["text"])
            for record in records
        }
        indexes: dict[str, BM25SBackend] = {}
        for condition in CONDITIONS:
            started = time.perf_counter()
            indexes[condition], _ = _build_index(records, condition)
            build_timing[condition] += (time.perf_counter() - started) * 1000.0
        questions = item.get("qa")
        if not isinstance(questions, list):
            raise ValidationError(f"LoCoMo sample {sample_id} has no qa array")
        for qa_index, qa in enumerate(questions):
            base: dict[str, Any] = {
                "question_id": None,
                "sample_id": sample_id,
                "split": split_name,
                "category": "unknown",
                "status": "failed",
                "failure": None,
                "gold_evidence_ids": [],
                "conditions": {},
            }
            if not isinstance(qa, Mapping):
                base["failure"] = {"classification": "failed_malformed_question", "details": ["question is not an object"]}
                rows.append(base)
                continue
            question = str(qa.get("question") or "").strip()
            base["question_id"] = _question_id(sample_id, qa_index, question)
            base["question_sha256"] = hashlib.sha256(question.encode("utf-8")).hexdigest()
            base["category"] = _category(qa.get("category"))
            if not question:
                base["failure"] = {"classification": "failed_empty_question", "details": []}
                rows.append(base)
                continue
            gold, invalid = _gold_evidence(sample_id, qa.get("evidence", []))
            base["gold_evidence_ids"] = sorted(gold)
            if invalid:
                base["failure"] = {"classification": "failed_invalid_annotation", "details": invalid}
                rows.append(base)
                continue
            missing = sorted(gold - known_ids)
            if not gold:
                base["failure"] = {"classification": "failed_missing_gold", "details": []}
                rows.append(base)
                continue
            if missing:
                base["failure"] = {"classification": "failed_unresolved_gold", "details": missing}
                rows.append(base)
                continue
            try:
                for condition in CONDITIONS:
                    candidate_ids, candidate_scores, retrieval_ms, padded_zero_score = _retrieve(
                        indexes[condition], question, pool_size=pool_size, known_ids=known_ids,
                    )
                    post_ids, post_scores, rerank_duration_ms = _rerank(
                        scorer, question, candidate_ids, authoritative_texts,
                    )
                    rerank_calls += 1
                    rerank_ms.append(rerank_duration_ms)
                    gold_set = set(gold)
                    pre_top = candidate_ids[:top_k]
                    post_top = post_ids[:top_k]
                    def recall(ids: Sequence[str]) -> float:
                        return len(gold_set.intersection(ids)) / len(gold_set)
                    def rr(ids: Sequence[str]) -> float:
                        rank = next((index for index, value in enumerate(ids, 1) if value in gold_set), None)
                        return 1.0 / rank if rank else 0.0
                    base["conditions"][condition] = {
                        "candidate_pool_size": len(candidate_ids),
                        "candidate_pool_ceiling": pool_size,
                        "zero_score_padding_count": padded_zero_score,
                        "candidate_evidence_ids": list(candidate_ids),
                        "candidate_bm25_scores": list(candidate_scores),
                        "oracle_recall_at_pool": recall(candidate_ids),
                        "pre_ranked_evidence_ids_at_10": list(pre_top),
                        "pre_recall_at_10": recall(pre_top),
                        "pre_reciprocal_rank": rr(pre_top),
                        "post_ranked_evidence_ids_at_10": list(post_top),
                        "post_rerank_scores_at_10": list(post_scores[:top_k]),
                        "post_recall_at_10": recall(post_top),
                        "post_reciprocal_rank": rr(post_top),
                        "timing_ms": {
                            "candidate_generation": retrieval_ms,
                            "reranking": rerank_duration_ms,
                        },
                    }
                base["status"] = "ok"
            except Exception as exc:
                base["failure"] = {
                    "classification": "failed_retrieval_or_reranking",
                    "details": [f"{type(exc).__name__}: {exc}"],
                }
            rows.append(base)
    snapshot = _snapshot_manifest(Path(model_path)) if isinstance(scorer, LocalMiniLMScorer) else {
        "path": "injected_scorer",
        "files": [],
        "manifest_sha256": None,
    }
    summaries = {
        split_name: {
            condition: _metrics(rows, condition=condition, split=split_name)
            for condition in CONDITIONS
        }
        for split_name in ("development", "held_out")
    }
    return {
        "schema_version": SCHEMA,
        "classification": "exploratory_offline_fixed_pool_cross_encoder_ablation",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "candidate_generators": list(CONDITIONS),
            "pool_size": pool_size,
            "final_top_k": top_k,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "selection_or_promotion_performed": False,
            "model_or_hyperparameter_tuning_performed": False,
            "fixed_pool_before_and_after_reranking": True,
            "reranker_input": "authoritative raw turn text for both candidate generators",
            "batch_size": int(batch_size),
        },
        "split": {
            "seed": split_seed,
            "unit": "whole conversation",
            "development_sample_ids": split["development"],
            "held_out_sample_ids": split["held_out"],
            "disjoint": set(split["development"]).isdisjoint(split["held_out"]),
        },
        "summaries": summaries,
        "question_rows": rows,
        "failure_ledger": [row for row in rows if row["status"] != "ok"],
        "provenance": {
            "dataset": {"path": str(dataset.resolve()), "sha256": _sha256(dataset)},
            "source": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
            "git": {"commit": _git_output("rev-parse", "HEAD"), "status_sha256": _value_sha256(_git_output("status", "--short"))},
            "machine": _machine(),
            "dependencies": _dependency_versions(),
            "model_snapshot": snapshot,
            "config_sha256": _value_sha256({"pool_size": pool_size, "top_k": top_k, "split_seed": split_seed, "model": snapshot["manifest_sha256"]}),
        },
        "timing": {
            "bm25_index_build_ms_by_condition": build_timing,
            "rerank_calls": rerank_calls,
            "rerank_ms": {
                "n": len(rerank_ms),
                "mean": float(np.mean(rerank_ms)) if rerank_ms else None,
                "p50": float(np.percentile(rerank_ms, 50)) if rerank_ms else None,
                "p95": float(np.percentile(rerank_ms, 95)) if rerank_ms else None,
            },
        },
        "execution": {
            "strictly_offline": True,
            "external_network_calls": 0,
            "provider_calls": 0,
            "embedding_calls": 0,
            "reader_calls": 0,
            "reranker_calls": rerank_calls,
            "actual_external_cost_usd": 0.0,
            "offline_environment_flags": {
                key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
            },
        },
        "claim_boundaries": [
            "This is an exploratory LoCoMo measurement, not a production winner selection.",
            "Candidate-pool oracle recall is reported separately; post-rerank scores cannot recover evidence absent from the fixed pool.",
            "The result does not establish generalization beyond this dataset and conversation-disjoint split.",
            "No claim is made about answer quality, external providers, or embedding performance.",
        ],
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a JSON artifact once; never overwrite an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _failure_receipt(dataset: Path, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": FAILED_SCHEMA,
        "status": "failed",
        "classification": "exploratory_offline_fixed_pool_cross_encoder_ablation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reason": f"{type(exc).__name__}: {exc}",
        "execution": {
            "strictly_offline": True,
            "external_network_calls": 0,
            "provider_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "actual_external_cost_usd": 0.0,
        },
        "dataset": {"path": str(Path(dataset).resolve()), "sha256": _sha256(Path(dataset)) if Path(dataset).is_file() else None},
        "claim_boundaries": ["Failed runs are retained as receipts and are not evidence of a retrieval result."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--split-seed", default=SPLIT_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            args.dataset,
            model_path=args.model_path,
            pool_size=args.pool_size,
            top_k=args.top_k,
            split_seed=args.split_seed,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        receipt = _failure_receipt(args.dataset, exc)
        atomic_write_json(args.output, receipt)
        print(f"offline rerank ablation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result["status"], "rows": len(result["question_rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
