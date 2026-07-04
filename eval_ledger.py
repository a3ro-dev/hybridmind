"""
Phase 6.0 measurement ledger — shared per-question JSONL instrumentation.

Wired into eval_locomo_retrieval.py, eval_longmemeval_retrieval.py, and
eval_musique_retrieval.py (see docs/PHASE_6_REALISTIC.md §2). Every run emits
one JSONL row per evaluated question to
benchmarks/results/ledger_<benchmark>_<confighash>.jsonl so that per-question
attribution, bootstrap CIs, and paired permutation tests (eval_stats.py) have
a stable substrate to work from.

Kept as a flat top-level module (eval_ledger.py) to match this repo's existing
eval_common.py / eval_*.py layout rather than introducing a new eval/ package.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

RESULTS_DIR = Path("benchmarks/results")

DEFAULT_K_LIST: Tuple[int, ...] = (1, 3, 5, 10, 25)

# Fixed seed for every eval run in this phase (docs/PHASE_6_REALISTIC.md §6.0.2).
DEFAULT_SEED = 42


def config_hash(config: Dict[str, Any]) -> str:
    """Stable short hash of a resolved run config (sorted-key JSON -> sha256[:12])."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def compute_pool_metrics(
    results: List[dict],
    relevance_fn: Callable[[dict], bool],
    k_list: Tuple[int, ...] = DEFAULT_K_LIST,
) -> Dict[str, Any]:
    """
    Derive retrieved-ids-at-k and gold rank from a post-rerank, rank-ordered
    candidate list.

    The caller must request enough candidates (top_k >= rerank_pool, and
    >= max(k_list)) so `results` reflects the full pool that entered
    reranking — otherwise gold_in_pool_pre_rerank is meaningless.

    relevance_fn: predicate(candidate_dict) -> bool, benchmark-specific
    (answer-overlap for LoCoMo/LongMemEval, supporting-paragraph-id for MuSiQue).
    """
    retrieved_ids_at_k = {
        str(k): [r.get("node_id") or r.get("id") for r in results[:k]]
        for k in k_list
    }

    gold_rank: Optional[int] = None
    for rank, r in enumerate(results, 1):
        if relevance_fn(r):
            gold_rank = rank
            break

    return {
        "retrieved_ids_at_k": retrieved_ids_at_k,
        "gold_in_pool_pre_rerank": gold_rank is not None,
        "gold_rank_post_rerank": gold_rank,
    }


class LedgerWriter:
    """Append-only JSONL writer for one benchmark run (one config_hash)."""

    def __init__(self, benchmark: str, config: Dict[str, Any], seed: int = DEFAULT_SEED):
        self.benchmark = benchmark
        self.config = config
        self.config_hash = config_hash(config)
        self.seed = seed
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RESULTS_DIR / f"ledger_{benchmark}_{self.config_hash}.jsonl"
        # Truncate: a stale row from a prior run under the same config_hash
        # would silently corrupt "two consecutive runs are bit-identical"
        # (the 6.0 acceptance gate).
        self.path.write_text("")

    def write(
        self,
        *,
        question_id: str,
        question_type: str,
        gold_evidence_ids: List[str],
        pool_metrics: Dict[str, Any],
        raw_llm_answer: str,
        judged_correct: bool,
        judge_rationale: str,
        prompt_version: str = "",
    ) -> None:
        record = {
            "question_id": question_id,
            "question_type": question_type,
            "gold_evidence_ids": list(gold_evidence_ids),
            "retrieved_ids_at_k": pool_metrics["retrieved_ids_at_k"],
            "gold_in_pool_pre_rerank": pool_metrics["gold_in_pool_pre_rerank"],
            "gold_rank_post_rerank": pool_metrics["gold_rank_post_rerank"],
            "raw_llm_answer": raw_llm_answer,
            "judged_correct": judged_correct,
            "judge_rationale": judge_rationale,
            "prompt_version": prompt_version,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "timestamp": time.time(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
