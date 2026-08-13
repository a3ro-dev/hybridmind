"""Immutable per-run evaluation ledgers and retrieval metric helpers.

Each evaluator run receives a unique ledger and a sidecar manifest.  Existing
artifacts are never truncated or relabelled.  The manifest carries the exact
resolved evaluator configuration plus dataset/runtime provenance; every row
is tied back to it by ``run_id`` and ``manifest_sha256``.

Retrieval metrics deliberately distinguish exact evidence-ID relevance from
answer-text overlap.  ``gold_in_pool_pre_rerank`` is ``None`` unless a caller
actually supplies the pre-rerank pool; post-rerank results are never presented
as pre-rerank evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

RESULTS_DIR = Path(os.getenv("HYBRIDMIND_EVAL_RESULTS_DIR", "benchmarks/results"))
LEDGER_SCHEMA = "hybridmind.eval-ledger/v2"
MANIFEST_SCHEMA = "hybridmind.eval-run/v2"
COMPLETION_SCHEMA = "hybridmind.eval-run-completion/v2"

DEFAULT_K_LIST: Tuple[int, ...] = (1, 3, 5, 10, 25)
DEFAULT_SEED = 42


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable short hash of a resolved run configuration."""
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()[:12]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: str | Path = ".") -> Optional[str]:
    """Return the checked-out commit without making provenance collection fatal."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_worktree_state(project_root: str | Path = ".") -> Dict[str, Any]:
    """Record whether the commit alone is sufficient to reproduce this run."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=project_root,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(status).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {"dirty": None, "status_sha256": None}


def _source_provenance() -> Dict[str, Any]:
    sources: Dict[str, str] = {}
    for candidate in (Path(__file__).resolve(), Path(sys.argv[0]).resolve()):
        if candidate.is_file():
            sources[str(candidate)] = sha256_file(candidate)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": sources,
    }


def dataset_provenance(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _first_relevant_rank(results: List[dict], relevance_fn: Callable[[dict], bool]) -> Optional[int]:
    for rank, result in enumerate(results, 1):
        if relevance_fn(result):
            return rank
    return None


def compute_pool_metrics(
    results: List[dict],
    relevance_fn: Callable[[dict], bool],
    k_list: Tuple[int, ...] = DEFAULT_K_LIST,
    *,
    pre_rerank_results: Optional[List[dict]] = None,
    metric_basis: str = "exact_evidence_id",
) -> Dict[str, Any]:
    """Compute honest pre/post-pool metrics for one relevance definition.

    ``results`` must be the final ordered API response.  A pre-rerank presence
    or rank is emitted only when ``pre_rerank_results`` is explicitly supplied.
    This prevents the historical bug where a post-rerank rank was also labelled
    as pre-rerank pool membership.
    """
    retrieved_ids_at_k = {
        str(k): [r.get("node_id") or r.get("id") for r in results[:k]]
        for k in k_list
    }
    post_rank = _first_relevant_rank(results, relevance_fn)
    pre_rank = (
        _first_relevant_rank(pre_rerank_results, relevance_fn)
        if pre_rerank_results is not None
        else None
    )
    return {
        "metric_basis": metric_basis,
        "retrieved_ids_at_k": retrieved_ids_at_k,
        "gold_in_pool_pre_rerank": pre_rank is not None if pre_rerank_results is not None else None,
        "gold_rank_pre_rerank": pre_rank,
        "gold_rank_post_rerank": post_rank,
        "hit_at_k": {
            str(k): bool(post_rank is not None and post_rank <= k)
            for k in k_list
        },
    }


def empty_pool_metrics(
    k_list: Tuple[int, ...] = DEFAULT_K_LIST,
    *,
    metric_basis: str = "exact_evidence_id",
) -> Dict[str, Any]:
    return compute_pool_metrics([], lambda _result: False, k_list, metric_basis=metric_basis)


class LedgerWriter:
    """Thread-safe immutable JSONL writer for exactly one evaluation run."""

    def __init__(
        self,
        benchmark: str,
        config: Mapping[str, Any],
        seed: int = DEFAULT_SEED,
        *,
        provenance: Optional[Mapping[str, Any]] = None,
        results_dir: str | Path | None = None,
        run_id: Optional[str] = None,
    ):
        self.benchmark = benchmark
        self.config = dict(config)
        self.config_hash = config_hash(self.config)
        self.seed = seed
        self.run_id = run_id or (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        self.results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stem = f"ledger_{benchmark}_{self.config_hash}_{self.run_id}"
        self.path = self.results_dir / f"{stem}.jsonl"
        self.manifest_path = self.results_dir / f"{stem}.manifest.json"
        self.completion_path = self.results_dir / f"{stem}.completion.json"
        self._lock = threading.Lock()
        self._sequence = 0
        self._finalized = False
        self._question_ids: set[str] = set()

        # Exclusive creation is the critical no-overwrite/no-interleave guard.
        self.path.open("x", encoding="utf-8").close()
        manifest_payload = {
            "schema": MANIFEST_SCHEMA,
            "benchmark": benchmark,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "config": self.config,
            "config_hash": self.config_hash,
            "seed": seed,
            "provenance": {
                "git_commit": git_commit(),
                "git_worktree": git_worktree_state(),
                "runtime": _source_provenance(),
                **dict(provenance or {}),
            },
            "ledger_file": self.path.name,
            "completion_file": self.completion_path.name,
        }
        self.manifest_sha256 = hashlib.sha256(
            _canonical_json(manifest_payload).encode("utf-8")
        ).hexdigest()
        manifest = {**manifest_payload, "manifest_sha256": self.manifest_sha256}
        with self.manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write(
        self,
        *,
        question_id: str,
        question_type: str,
        gold_evidence_ids: List[str],
        pool_metrics: Mapping[str, Any],
        raw_llm_answer: str = "",
        judged_correct: Optional[bool] = None,
        judge_rationale: str = "",
        prompt_version: str = "",
        status: str = "completed",
        answer_status: str = "not_requested",
        judge_method: str = "not_run",
        answer_overlap_metrics: Optional[Mapping[str, Any]] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not question_id:
            raise ValueError("question_id must be non-empty")
        record = {
            "schema": LEDGER_SCHEMA,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "question_id": question_id,
            "question_type": question_type,
            "gold_evidence_ids": list(gold_evidence_ids),
            "retrieved_ids_at_k": pool_metrics["retrieved_ids_at_k"],
            "metric_basis": pool_metrics.get("metric_basis", "exact_evidence_id"),
            "gold_in_pool_pre_rerank": pool_metrics.get("gold_in_pool_pre_rerank"),
            "gold_rank_pre_rerank": pool_metrics.get("gold_rank_pre_rerank"),
            "gold_rank_post_rerank": pool_metrics.get("gold_rank_post_rerank"),
            "hit_at_k": pool_metrics.get("hit_at_k", {}),
            "answer_overlap_metrics": dict(answer_overlap_metrics) if answer_overlap_metrics else None,
            "raw_llm_answer": raw_llm_answer,
            "answer_status": answer_status,
            "judged_correct": judged_correct,
            "judge_method": judge_method,
            "judge_rationale": judge_rationale,
            "prompt_version": prompt_version,
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
            "config_hash": self.config_hash,
            "seed": self.seed,
        }
        if extra:
            record["extra"] = dict(extra)
        try:
            import eval_common

            record["budget"] = eval_common.active_budget_provenance()
        except ImportError:
            record["budget"] = None

        with self._lock:
            if self._finalized:
                raise RuntimeError("cannot append to a finalized evaluation ledger")
            if question_id in self._question_ids:
                raise ValueError(f"duplicate question_id in evaluation run: {question_id}")
            self._question_ids.add(question_id)
            self._sequence += 1
            record["sequence"] = self._sequence
            encoded = json.dumps(record, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

    def finalize(
        self,
        *,
        status: str,
        summary: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Seal this run with an immutable ledger checksum and row count."""
        if status not in {"completed", "failed"}:
            raise ValueError("completion status must be 'completed' or 'failed'")
        with self._lock:
            if self._finalized:
                raise RuntimeError("evaluation ledger is already finalized")
            expected_rows = (summary or {}).get("n")
            if status == "completed" and expected_rows is not None and expected_rows != self._sequence:
                raise ValueError(
                    f"completed run claims n={expected_rows} but ledger contains "
                    f"{self._sequence} rows"
                )
            receipt = {
                "schema": COMPLETION_SCHEMA,
                "benchmark": self.benchmark,
                "run_id": self.run_id,
                "manifest_sha256": self.manifest_sha256,
                "ledger_file": self.path.name,
                "ledger_sha256": sha256_file(self.path),
                "row_count": self._sequence,
                "status": status,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": dict(summary or {}),
            }
            try:
                import eval_common

                receipt["budget"] = eval_common.active_budget_provenance()
            except ImportError:
                receipt["budget"] = None
            with self.completion_path.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._finalized = True
        return self.completion_path

    def finalize_failure(
        self,
        *,
        reason: str,
        error_type: str,
        expected_questions: Optional[int] = None,
    ) -> Path:
        """Seal a partial run as failed without recording sensitive exception text."""
        summary = {
            "reason": reason,
            "error_type": error_type,
            "rows_written": self._sequence,
            "expected_questions": expected_questions,
        }
        return self.finalize(status="failed", summary=summary)
