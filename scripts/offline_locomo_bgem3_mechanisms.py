"""Offline LoCoMo BGE-M3 sparse and late-interaction mechanism ablation.

This file measures BGE-M3's learned sparse weights and ColBERT-style token
MaxSim only.  Its 1024-dimensional dense output is deliberately excluded:
HybridMind production embeddings are native 4096-dimensional vectors.
The default loader is local-only and refuses an ambiguous Hugging Face cache.
No provider, network, reader, or answer calls are made here.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_locomo_rerank_ablation import (  # noqa: E402
    _category,
    _cluster_bootstrap,
    _question_id,
    _snapshot_manifest,
)
from scripts.offline_locomo_sparse_baseline import (  # noqa: E402
    DEFAULT_DATASET,
    _canonical_id,
    _gold_evidence,
    _sha256,
)
from scripts.offline_locomo_sparse_experiments import _split_ids, _turn_records  # noqa: E402
from storage.bm25_index import BM25SBackend  # noqa: E402


SCHEMA = "hybridmind.offline-locomo-bgem3-mechanisms/v1"
FAILED_SCHEMA = "hybridmind.offline-locomo-bgem3-mechanisms-failure/v1"
DEFAULT_MODEL_ROOT = Path(r"C:\Users\akshat\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots")
DEFAULT_COMPLETE_SNAPSHOT = DEFAULT_MODEL_ROOT / "5617a9f61b028005a4858fdac845db406aefb181"
POOL_SIZE = 25
TOP_K = 10
SPLIT_SEED = "20260822-bgem3-mechanisms"
BOOTSTRAP_SEED = 20260822
CONDITIONS = ("bm25s_speaker_prefix", "bgem3_learned_sparse")


class ValidationError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _deps() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("FlagEmbedding", "torch", "transformers", "numpy", "bm25s"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def resolve_model_snapshot(model_path: Path | None = None) -> Path:
    """Resolve exactly one local snapshot; never select an arbitrary cache entry."""
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValidationError(f"explicit BGE-M3 snapshot does not exist: {path}")
        candidates = ["config.json", "tokenizer.json", "tokenizer_config.json", "sparse_linear.pt", "colbert_linear.pt"]
        missing = [name for name in candidates if not (path / name).is_file()]
        if not ((path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()):
            missing.append("model.safetensors or pytorch_model.bin")
        if missing:
            raise ValidationError(f"BGE-M3 snapshot is incomplete; missing {missing}: {path}")
        return path
    paths = sorted(p.resolve() for p in DEFAULT_MODEL_ROOT.glob("*") if p.is_dir())
    if len(paths) != 1:
        raise ValidationError(
            f"BGE-M3 cache has {len(paths)} snapshots; pass --model-path explicitly"
        )
    return paths[0]


def _offline_flags() -> dict[str, str]:
    flags = {key: "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ.update(flags)
    return flags


def _reject_dense_output(value: Any) -> None:
    """Guard against accidentally treating BGE-M3's 1024-d dense vector as production."""
    if value is None:
        return
    array = np.asarray(value)
    if array.ndim == 1:
        dimension = int(array.shape[0])
    elif array.ndim >= 2:
        dimension = int(array.shape[-1])
    else:
        raise ValidationError("malformed dense output")
    raise ValidationError(
        f"BGE-M3 dense output dimension {dimension} is excluded; production requires native 4096"
    )


class LocalBGEM3Encoder:
    """FlagEmbedding wrapper using only learned sparse and ColBERT outputs."""

    def __init__(self, model_path: Path, *, batch_size: int = 8) -> None:
        self.model_path = resolve_model_snapshot(model_path)
        _offline_flags()
        # FlagEmbedding's BGEM3FlagModel constructor does not expose
        # ``local_files_only``; an explicit snapshot path plus the three HF
        # offline flags is the supported equivalent and fails closed on cache
        # misses rather than permitting a Hub lookup.
        self.batch_size = max(1, int(batch_size))
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ValidationError(f"FlagEmbedding unavailable: {exc}") from exc
        try:
            self.model = BGEM3FlagModel(
                str(self.model_path), use_fp16=False, batch_size=self.batch_size,
                return_dense=False, return_sparse=True, return_colbert_vecs=True,
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ValidationError(f"local BGE-M3 snapshot failed to load: {exc}") from exc
        self.call_count = 0

    def encode(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        if not all(isinstance(text, str) for text in texts):
            raise ValidationError("encoder inputs must be strings")
        started = time.perf_counter()
        output = self.model.encode(
            list(texts), batch_size=self.batch_size, max_length=512,
            return_dense=False, return_sparse=True, return_colbert_vecs=True,
        )
        if isinstance(output, Mapping) and "dense_vecs" in output:
            # This branch is defensive: the dense output is never consumed.
            _reject_dense_output(output["dense_vecs"])
        sparse = output.get("lexical_weights") if isinstance(output, Mapping) else None
        colbert = output.get("colbert_vecs") if isinstance(output, Mapping) else None
        if not isinstance(sparse, Sequence) or not isinstance(colbert, Sequence):
            raise ValidationError("BGE-M3 did not return sparse and ColBERT outputs")
        self.call_count += 1
        elapsed = (time.perf_counter() - started) * 1000.0
        return [{"sparse": dict(sparse[i]), "colbert": np.asarray(colbert[i], dtype=np.float32),
                 "elapsed_ms": elapsed} for i in range(len(texts))]


def _validate_encoded(encoded: Mapping[str, Any]) -> tuple[dict[str, float], np.ndarray]:
    sparse = encoded.get("sparse")
    colbert = encoded.get("colbert")
    if not isinstance(sparse, Mapping):
        raise ValidationError("malformed sparse encoding")
    weights = {str(key): float(value) for key, value in sparse.items()}
    if not all(math.isfinite(value) for value in weights.values()):
        raise ValidationError("non-finite sparse weight")
    vectors = np.asarray(colbert, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValidationError("malformed ColBERT token vectors")
    if not np.isfinite(vectors).all():
        raise ValidationError("non-finite ColBERT token vector")
    return weights, vectors


def learned_sparse_score(query: Mapping[str, Any], document: Mapping[str, Any]) -> float:
    q, d = _validate_encoded(query)[0], _validate_encoded(document)[0]
    return float(sum(value * d.get(token, 0.0) for token, value in q.items()))


def maxsim_score(query: Mapping[str, Any], document: Mapping[str, Any]) -> float:
    """ColBERT-style MaxSim: mean over query tokens of max document similarity."""
    q = _validate_encoded(query)[1]
    d = _validate_encoded(document)[1]
    if q.shape[1] != d.shape[1]:
        raise ValidationError("query/document ColBERT dimensions differ")
    similarities = q @ d.T
    return float(np.max(similarities, axis=1).mean())


def _build_bm25(records: Sequence[Mapping[str, str]]) -> BM25SBackend:
    pairs = []
    for record in records:
        evidence_id = str(record.get("source_id") or "")
        speaker = str(record.get("speaker") or "").strip()
        text = str(record.get("text") or "").strip()
        if not evidence_id.startswith("locomo:") or not text:
            raise ValidationError("BM25 records must have canonical IDs and raw text")
        pairs.append((evidence_id, f"{speaker}: {text}" if speaker else text))
    index = BM25SBackend()
    index.add_batch(pairs)
    return index


def _bm25_ids(index: BM25SBackend, query: str, ids: set[str]) -> tuple[list[str], float, int]:
    started = time.perf_counter()
    hits = index.search(query, top_k=min(POOL_SIZE, len(ids)))
    seen: set[str] = set()
    ranked: list[str] = []
    for item in hits:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValidationError("malformed BM25 result")
        evidence_id = str(item[0])
        if evidence_id not in ids or evidence_id in seen:
            raise ValidationError("BM25 returned unknown or duplicate evidence ID")
        seen.add(evidence_id)
        ranked.append(evidence_id)
    padded = _pad_pool(ranked, ids)
    return padded, (time.perf_counter() - started) * 1000.0, max(0, len(padded) - len(ranked))


def _pad_pool(ranked: Sequence[str], all_ids: set[str]) -> list[str]:
    target = min(POOL_SIZE, len(all_ids))
    unique = list(dict.fromkeys(ranked))
    if not set(unique) <= all_ids:
        raise ValidationError("candidate pool contains an unknown evidence ID")
    return (unique + sorted(all_ids - set(unique)))[:target]


def _sparse_ids(query: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], int]:
    ranked = sorted(((learned_sparse_score(query, value), evidence_id)
                     for evidence_id, value in documents.items()), key=lambda row: (-row[0], row[1]))
    selected = ranked[:POOL_SIZE]
    return _pad_pool([evidence_id for _, evidence_id in selected], set(documents)), sum(
        score == 0.0 for score, _evidence_id in selected
    )


def _metrics(rows: Sequence[Mapping[str, Any]], condition: str, split: str) -> dict[str, Any]:
    selected = [r for r in rows if r.get("split") == split and r.get("status") == "ok"]
    def recall(row: Mapping[str, Any], stage: str) -> float:
        return float(row["conditions"][condition][stage]["recall_at_10"])
    def mrr(row: Mapping[str, Any], stage: str) -> float:
        return float(row["conditions"][condition][stage]["mrr_at_10"])
    return {
        "split": split, "condition": condition, "n_ok": len(selected),
        "n_failed": sum(1 for r in rows if r.get("split") == split and r.get("status") != "ok"),
        "candidate_pool_oracle_recall": _cluster_bootstrap(
            selected, lambda r: float(r["conditions"][condition]["oracle_recall_at_pool"]),
            seed=BOOTSTRAP_SEED + 1),
        "pre_recall_at_10": _cluster_bootstrap(selected, lambda r: recall(r, "pre"), seed=BOOTSTRAP_SEED + 2),
        "post_recall_at_10": _cluster_bootstrap(selected, lambda r: recall(r, "post"), seed=BOOTSTRAP_SEED + 3),
        "paired_delta_recall_at_10": _cluster_bootstrap(
            selected, lambda r: recall(r, "post") - recall(r, "pre"), seed=BOOTSTRAP_SEED + 4),
        "pre_mrr_at_10": _cluster_bootstrap(selected, lambda r: mrr(r, "pre"), seed=BOOTSTRAP_SEED + 5),
        "post_mrr_at_10": _cluster_bootstrap(selected, lambda r: mrr(r, "post"), seed=BOOTSTRAP_SEED + 6),
        "paired_delta_mrr_at_10": _cluster_bootstrap(
            selected, lambda r: mrr(r, "post") - mrr(r, "pre"), seed=BOOTSTRAP_SEED + 7),
        "by_category": {
            category: _cluster_bootstrap(
                [r for r in selected if r["category"] == category], lambda r: recall(r, "post"),
                seed=BOOTSTRAP_SEED + len(category))
            for category in sorted({str(r["category"]) for r in selected})
        },
    }


def _condition_comparison(
    rows: Sequence[Mapping[str, Any]], split: str, *,
    left: str = "bgem3_learned_sparse", right: str = "bm25s_speaker_prefix",
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("split") == split and row.get("status") == "ok"]
    comparisons = {
        "candidate_pool_oracle_recall": lambda row: (
            row["conditions"][left]["oracle_recall_at_pool"]
            - row["conditions"][right]["oracle_recall_at_pool"]
        ),
        "pre_recall_at_10": lambda row: (
            row["conditions"][left]["pre"]["recall_at_10"]
            - row["conditions"][right]["pre"]["recall_at_10"]
        ),
        "post_maxsim_recall_at_10": lambda row: (
            row["conditions"][left]["post"]["recall_at_10"]
            - row["conditions"][right]["post"]["recall_at_10"]
        ),
        "post_maxsim_mrr_at_10": lambda row: (
            row["conditions"][left]["post"]["mrr_at_10"]
            - row["conditions"][right]["post"]["mrr_at_10"]
        ),
    }
    return {
        "left": left,
        "right": right,
        "paired_deltas": {
            name: _cluster_bootstrap(selected, value, seed=BOOTSTRAP_SEED + 100 + offset)
            for offset, (name, value) in enumerate(comparisons.items())
        },
    }


def _stage(ids: Sequence[str], gold: set[str]) -> dict[str, Any]:
    selected = list(ids[:TOP_K])
    hits = gold.intersection(selected)
    rank = next((i for i, value in enumerate(selected, 1) if value in gold), None)
    return {"evidence_ids": selected, "recall_at_10": len(hits) / len(gold),
            "mrr_at_10": 1.0 / rank if rank else 0.0}


def evaluate(dataset: Path, *, encoder: Any, split_seed: str = SPLIT_SEED,
             batch_size: int = 8) -> dict[str, Any]:
    _offline_flags()
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        raise ValidationError("LoCoMo dataset must contain at least two conversations")
    split = _split_ids(data, split_seed)
    split_by_id = {sample_id: name for name, ids in split.items() for sample_id in ids}
    rows: list[dict[str, Any]] = []
    index_ms = query_ms = rerank_ms = encode_ms = 0.0
    sparse_postings = token_vector_bytes = 0
    for item in data:
        sample_id = str(item.get("sample_id") or "")
        records = _turn_records(item)
        ids = {str(record["source_id"]) for record in records}
        documents = {
            str(record["source_id"]): (
                f"{str(record.get('speaker') or '').strip()}: {str(record['text']).strip()}"
                if str(record.get("speaker") or "").strip()
                else str(record["text"]).strip()
            )
            for record in records
        }
        started = time.perf_counter(); bm25 = _build_bm25(records); index_ms += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter(); encoded_values = encoder.encode(list(documents.values())); encode_ms += (time.perf_counter() - started) * 1000.0
        if len(encoded_values) != len(documents):
            raise ValidationError("encoder returned wrong document count")
        docs = dict(zip(documents, encoded_values))
        for value in docs.values():
            sparse, vectors = _validate_encoded(value)
            sparse_postings += len(sparse); token_vector_bytes += int(vectors.nbytes)
        questions = item.get("qa")
        if not isinstance(questions, list):
            raise ValidationError(f"LoCoMo sample {sample_id} has no qa array")
        sample_rows: list[dict[str, Any]] = []
        valid_questions: list[tuple[dict[str, Any], str]] = []
        for qa_index, qa in enumerate(questions):
            base = {"question_id": _question_id(sample_id, qa_index, ""), "sample_id": sample_id, "split": split_by_id.get(sample_id),
                    "category": _category(qa.get("category")) if isinstance(qa, Mapping) else "unknown",
                    "status": "failed", "failure": None, "gold_evidence_ids": [], "conditions": {}}
            if not isinstance(qa, Mapping):
                base["failure"] = {"classification": "failed_malformed_question", "details": []}; sample_rows.append(base); continue
            question = str(qa.get("question") or "").strip(); base["question_id"] = _question_id(sample_id, qa_index, question)
            gold, invalid = _gold_evidence(sample_id, qa.get("evidence", [])); base["gold_evidence_ids"] = sorted(gold)
            if not question or invalid or not gold or not gold <= ids:
                base["failure"] = {"classification": "failed_invalid_or_unresolved_evidence", "details": invalid or sorted(gold - ids)}
                sample_rows.append(base); continue
            sample_rows.append(base)
            valid_questions.append((base, question))
        started = time.perf_counter()
        query_encodings = encoder.encode([question for _base, question in valid_questions]) if valid_questions else []
        encode_ms += (time.perf_counter() - started) * 1000.0
        if len(query_encodings) != len(valid_questions):
            raise ValidationError("encoder returned wrong query count")
        for (base, question), query_encoded in zip(valid_questions, query_encodings):
            gold = set(base["gold_evidence_ids"])
            try:
                started = time.perf_counter(); bm_ids = _bm25_ids(bm25, question, ids); query_ms += (time.perf_counter() - started) * 1000.0
                started = time.perf_counter(); sparse_ids, sparse_zero_fill = _sparse_ids(query_encoded, docs); sparse_query_ms = (time.perf_counter() - started) * 1000.0; query_ms += sparse_query_ms
                for condition, candidate_ids in (("bm25s_speaker_prefix", bm_ids[0]), ("bgem3_learned_sparse", sparse_ids)):
                    started = time.perf_counter()
                    positions = {evidence_id: position for position, evidence_id in enumerate(candidate_ids)}
                    maxsim_ranked = sorted(((maxsim_score(query_encoded, docs[evidence_id]), evidence_id)
                                            for evidence_id in candidate_ids), key=lambda row: (-row[0], positions[row[1]], row[1]))
                    condition_rerank_ms = (time.perf_counter() - started) * 1000.0
                    rerank_ms += condition_rerank_ms
                    candidate = {"evidence_ids": list(candidate_ids), "retrieval_ms": bm_ids[1] if condition.startswith("bm25") else 0.0,
                                 "sparse_query_ms": sparse_query_ms if condition == "bgem3_learned_sparse" else 0.0,
                                 "rerank_ms": condition_rerank_ms}
                    base["conditions"][condition] = {
                        "candidate_pool_size": len(candidate_ids),
                        "zero_score_padding_count": bm_ids[2] if condition.startswith("bm25") else sparse_zero_fill,
                        "candidate_pool_budget": min(POOL_SIZE, len(ids)), "deterministic_padding": "sorted canonical evidence IDs if backend returns fewer than budget",
                        "candidate_pool_evidence_ids": list(candidate_ids), "oracle_recall_at_pool": len(gold.intersection(candidate_ids)) / len(gold),
                        "pre": _stage(candidate_ids, gold), "post": _stage([evidence_id for _, evidence_id in maxsim_ranked], gold),
                        "timing_ms": candidate,
                    }
                base["status"] = "ok"
            except Exception as exc:
                base["failure"] = {"classification": "failed_retrieval_or_encoding", "details": [f"{type(exc).__name__}: {exc}"]}
        rows.extend(sample_rows)
    model_path = getattr(encoder, "model_path", None)
    snapshot = _snapshot_manifest(Path(model_path)) if model_path else {"path": "injected_encoder", "files": [], "manifest_sha256": None}
    if model_path:
        license_path = Path(model_path) / "README.md"
        snapshot["license"] = "MIT" if license_path.is_file() and "license: mit" in license_path.read_text(encoding="utf-8", errors="replace").lower() else "unverified"
        snapshot["license_source_sha256"] = _sha256(license_path) if license_path.is_file() else None
    machine = {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count()}
    dependencies = _deps()
    summaries = {name: {condition: _metrics(rows, condition, name) for condition in CONDITIONS} for name in ("development", "held_out")}
    comparisons = {name: _condition_comparison(rows, name) for name in ("development", "held_out")}
    local_model_encode_batches = getattr(encoder, "call_count", None)
    if local_model_encode_batches is None and hasattr(encoder, "calls"):
        local_model_encode_batches = len(encoder.calls)
    return {"schema_version": SCHEMA, "classification": "exploratory_offline_bgem3_mechanism_ablation", "status": "complete",
            "generated_at": datetime.now(timezone.utc).isoformat(), "experiment": {
                "candidate_generators": list(CONDITIONS), "pool_size": POOL_SIZE, "final_top_k": TOP_K,
                "reranker": "BGE-M3 ColBERT-style MaxSim", "candidate_pool_gold_blind": True,
                "document_representation": "speaker-prefixed authoritative turn content for both candidate generators and MaxSim",
                "maxsim_input": "source-identical speaker-prefixed authoritative turn representation", "selection_or_promotion_performed": False,
                "dense_output_used": False, "dense_output_dimension_excluded": 1024, "production_embedding_contract_dimension": 4096,
                "claim_boundary": "BGE-M3 mechanisms only; not a SPLADE++ or ColBERTv2 reproduction", "batch_size": batch_size},
            "split": {"seed": split_seed, "unit": "whole conversation", "development_sample_ids": split["development"], "held_out_sample_ids": split["held_out"], "disjoint": set(split["development"]).isdisjoint(split["held_out"])},
            "summaries": summaries, "comparisons": comparisons,
            "question_rows": rows, "failure_ledger": [r for r in rows if r["status"] != "ok"],
            "storage": {"learned_sparse_postings": sparse_postings, "colbert_token_vector_bytes": token_vector_bytes},
            "provenance": {"dataset": {"path": str(Path(dataset).resolve()), "sha256": _sha256(Path(dataset))}, "source": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))}, "git": {"commit": _git("rev-parse", "HEAD"), "status_sha256": _value_sha256(_git("status", "--short"))}, "platform": {**machine, "sha256": _value_sha256(machine)}, "dependencies": {"versions": dependencies, "sha256": _value_sha256(dependencies)}, "model_snapshot": snapshot, "config_sha256": _value_sha256({"pool_size": POOL_SIZE, "top_k": TOP_K, "split_seed": split_seed})},
            "timing_ms": {"encoding": encode_ms, "index_build": index_ms, "query": query_ms, "rerank": rerank_ms},
            "execution": {"strictly_offline": True, "external_network_calls": 0, "provider_calls": 0, "remote_embedding_calls": 0,
                          "local_model_encode_batches": local_model_encode_batches, "reader_calls": 0,
                          "actual_external_cost_usd": 0.0, "offline_environment_flags": _offline_flags()},
            "claim_boundaries": ["This is a mechanism ablation, not a production configuration or winner selection.", "BGE-M3 1024-d dense output was not used and is incompatible with HybridMind's native 4096-d runtime contract.", "Candidate-pool oracle recall is separate from final Recall@10/MRR@10; MaxSim cannot recover absent evidence.", "No claims are made about SPLADE++, ColBERTv2, answer quality, or generalization beyond the conversation-disjoint LoCoMo split."]}


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        temp.unlink(missing_ok=True)


def failure_receipt(dataset: Path, exc: Exception) -> dict[str, Any]:
    return {"schema_version": FAILED_SCHEMA, "status": "failed", "classification": "exploratory_offline_bgem3_mechanism_ablation", "generated_at": datetime.now(timezone.utc).isoformat(), "reason": f"{type(exc).__name__}: {exc}", "dataset": {"path": str(Path(dataset).resolve()), "sha256": _sha256(Path(dataset)) if Path(dataset).is_file() else None}, "execution": {"strictly_offline": True, "external_network_calls": 0, "provider_calls": 0, "actual_external_cost_usd": 0.0}, "claim_boundaries": ["Failed runs are retained as receipts and are not evidence of a retrieval result."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--model-path", type=Path); parser.add_argument("--batch-size", type=int, default=8); args = parser.parse_args(argv)
    try:
        encoder = LocalBGEM3Encoder(resolve_model_snapshot(args.model_path), batch_size=args.batch_size)
        report = evaluate(args.dataset, encoder=encoder, batch_size=args.batch_size)
    except Exception as exc:
        atomic_write_json(args.output, failure_receipt(args.dataset, exc)); print(f"offline BGE-M3 mechanisms failed: {type(exc).__name__}: {exc}", file=sys.stderr); return 2
    atomic_write_json(args.output, report); print(json.dumps({"output": str(args.output), "status": report["status"], "rows": len(report["question_rows"])}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
