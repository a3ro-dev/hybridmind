"""
Statistical protocol (docs/EVALUATION.md section 4).

- Bootstrap 95% CI (1e4 resamples over questions) for accuracy / Hit@k.
- Paired permutation test comparing two ledger files on per-question outcomes.

Kept as a flat top-level module (eval_stats.py) to match this repo's existing
eval_common.py / eval_ledger.py / eval_*.py layout — invoke as a script rather
than `python -m eval.stats` (there is no eval/ package in this repo).

Usage:
  python eval_stats.py ci <ledger.jsonl> [--metric accuracy|hit1|hit3|hit5|hit10|hit25]
  python eval_stats.py compare <ledger_a.jsonl> <ledger_b.jsonl> [--metric ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

N_RESAMPLES = 10_000
N_PERMUTATIONS = 10_000
DEFAULT_SEED = 42

HIT_K_VALUES = (1, 3, 5, 10, 25)


def load_ledger(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_evaluated(record: dict) -> bool:
    """True if this row actually ran the answering LLM (not a retrieval-only pass)."""
    return record.get("judge_rationale") != "not evaluated (retrieval-only run, pass --with-answers)"


def metric_values(records: List[dict], metric: str) -> Dict[str, float]:
    """
    Return {question_id: value} for the requested metric.

    metric == "accuracy": 1.0/0.0 from judged_correct, only over rows that
      actually ran the answering LLM (retrieval-only rows are excluded, not
      counted as wrong — they carry no reading-stage signal at all).
    metric == "hitK": derived from gold_rank_post_rerank (1 if a gold-relevant
      candidate was found within the top K of the returned pool, else 0). This
      is a post-rerank Hit@k, consistent with the ledger's own definition of
      gold_rank_post_rerank.
    """
    if metric == "accuracy":
        return {
            r["question_id"]: (1.0 if r.get("judged_correct") else 0.0)
            for r in records
            if _is_evaluated(r)
        }
    if metric.startswith("hit") and metric[3:].isdigit():
        k = int(metric[3:])
        out = {}
        for r in records:
            rank = r.get("gold_rank_post_rerank")
            out[r["question_id"]] = 1.0 if (rank is not None and rank <= k) else 0.0
        return out
    raise ValueError(f"Unknown metric: {metric!r} (expected 'accuracy' or 'hitK' for K in {HIT_K_VALUES})")


def bootstrap_ci(
    values: List[float],
    n_resamples: int = N_RESAMPLES,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Percentile bootstrap CI over the question axis."""
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    point_mean = sum(values) / n
    resample_means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(resample) / n)
    resample_means.sort()
    alpha = (1 - ci) / 2
    lo = resample_means[int(alpha * n_resamples)]
    hi = resample_means[int((1 - alpha) * n_resamples) - 1]
    return {"mean": point_mean, "ci_lo": lo, "ci_hi": hi, "n": n}


def paired_permutation_test(
    values_a: Dict[str, float],
    values_b: Dict[str, float],
    n_permutations: int = N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """
    Paired permutation test on the intersection of question_ids present in
    both ledgers. Two-sided: null hypothesis is that per-question outcomes
    are exchangeable between A and B.
    """
    shared_ids = sorted(set(values_a) & set(values_b))
    if not shared_ids:
        raise ValueError("No shared question_id between the two ledgers — nothing to compare")

    a = [values_a[qid] for qid in shared_ids]
    b = [values_b[qid] for qid in shared_ids]
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    observed = sum(diffs) / n

    rng = random.Random(seed)
    count_ge = 0
    for _ in range(n_permutations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= abs(observed):
            count_ge += 1
    p_value = count_ge / n_permutations

    return {
        "n_paired": n,
        "mean_a": sum(a) / n,
        "mean_b": sum(b) / n,
        "observed_diff": observed,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def _print_ci(label: str, result: Dict[str, float]) -> None:
    print(f"{label}: mean={result['mean']:.4f}  95% CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]  n={result['n']}")


def cmd_ci(args) -> None:
    records = load_ledger(args.ledger)
    metrics = [args.metric] if args.metric else ["accuracy", *[f"hit{k}" for k in HIT_K_VALUES]]
    for metric in metrics:
        try:
            values = list(metric_values(records, metric).values())
        except ValueError as e:
            print(f"{metric}: {e}")
            continue
        result = bootstrap_ci(values, n_resamples=args.n_resamples, seed=args.seed)
        _print_ci(metric, result)


def cmd_compare(args) -> None:
    records_a = load_ledger(args.ledger_a)
    records_b = load_ledger(args.ledger_b)
    metrics = [args.metric] if args.metric else ["accuracy", *[f"hit{k}" for k in HIT_K_VALUES]]

    n_a = len(records_a)
    print(f"LoCoMo-style minimum detectable effect note: with n={n_a} questions, treat any "
          f"gain under ~2.5pts (see docs/EVALUATION.md section 4) as noise even if p<0.05.\n")

    for metric in metrics:
        try:
            values_a = metric_values(records_a, metric)
            values_b = metric_values(records_b, metric)
            ci_a = bootstrap_ci(list(values_a.values()), n_resamples=args.n_resamples, seed=args.seed)
            ci_b = bootstrap_ci(list(values_b.values()), n_resamples=args.n_resamples, seed=args.seed)
            test = paired_permutation_test(values_a, values_b, n_permutations=args.n_perm, seed=args.seed)
        except ValueError as e:
            print(f"{metric}: {e}")
            continue
        print(f"--- {metric} ---")
        _print_ci("  A", ci_a)
        _print_ci("  B", ci_b)
        print(f"  paired diff (A-B) = {test['observed_diff']:+.4f}  p={test['p_value']:.4f}  "
              f"{'SIGNIFICANT' if test['significant_at_0.05'] else 'not significant'} (n_paired={test['n_paired']})")
        print()


def parse_args():
    p = argparse.ArgumentParser(description="Phase 6.0 ledger statistics")
    sub = p.add_subparsers(dest="command", required=True)

    p_ci = sub.add_parser("ci", help="Bootstrap 95%% CI for one ledger")
    p_ci.add_argument("ledger")
    p_ci.add_argument("--metric", default=None, help="accuracy|hit1|hit3|hit5|hit10|hit25 (default: all)")
    p_ci.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    p_ci.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_ci.set_defaults(func=cmd_ci)

    p_cmp = sub.add_parser("compare", help="Paired permutation test between two ledgers")
    p_cmp.add_argument("ledger_a")
    p_cmp.add_argument("ledger_b")
    p_cmp.add_argument("--metric", default=None, help="accuracy|hit1|hit3|hit5|hit10|hit25 (default: all)")
    p_cmp.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    p_cmp.add_argument("--n-perm", type=int, default=N_PERMUTATIONS)
    p_cmp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_cmp.set_defaults(func=cmd_compare)

    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
