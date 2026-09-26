"""Retrieval metrics and aggregation for LoCoMo / LongMemEval scopes.

Pure functions over evidence IDs (``TurnRecord.evidence_id``); zero provider
calls, no config reads. Per-question metrics fail closed (``ValueError``) on
empty gold, gold outside the corpus, or ranked IDs outside the corpus: the
caller must skip such questions explicitly with a reason (``MetricRow``), so
every denominator is reported, never implied.

Official denominators are encoded as skip reasons by ``scoring_gold``:
- LoCoMo evidence metrics: questions with non-empty resolved gold.
- LongMemEval official turn/session metrics: not ``_abs`` and at least one
  user-side ``has_answer`` turn (``Question.gold_user_evidence``) -> 419 on S-cleaned.

Attribution
-----------
LoCoMo recall (``locomo_recall_acc``, ``locomo_recall_official``) is a clean
reimplementation of the metric as defined by the LoCoMo evaluation code
(https://github.com/snap-research/locomo @ 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376:
task_eval/evaluation.py::eval_question_answering per-row recall,
task_eval/evaluate_qa.py per-row rounding, task_eval/evaluation_stats.py
aggregation). That repository is CC-BY-NC-4.0, so no code is copied; behaviour
equality is checked by tests on the definition (per-row fraction of gold
dia_ids in the context, session-id contexts matched by session number, empty
evidence scores 1.0, rows rounded to 3 decimals, categories averaged over all
rows). LoCoMo: Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang (ACL 2024).

LongMemEval metrics are *imported* (not copied) from
``scripts/reproduce_longmemeval_retrieval.py``, which vendors
https://github.com/xiaowu0162/LongMemEval src/retrieval/eval_utils.py
(logic identical at 9e0b455f4ef0e2ab8f2e582289761153549043fc; MIT,
"Copyright (c) 2024 Di Wu").

Bootstrap CIs reuse ``eval_stats.bootstrap_ci`` (question level) and
``scripts.offline_budgeted_evidence.cluster_bootstrap`` (conversation clusters).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from eval_stats import DEFAULT_SEED, N_RESAMPLES, bootstrap_ci
from scripts.offline_budgeted_evidence import cluster_bootstrap
from scripts.reproduce_longmemeval_retrieval import evaluate_retrieval, evaluate_retrieval_turn2session

DEVIATIONS = [
    "locomo_recall_acc: an empty context list with non-empty evidence scores 0.0; the official "
    "code indexes context[0] and raises IndexError.",
    "locomo_recall_official replicates analyze_aggr_acc exactly: rows without evidence count in "
    "the denominator but add 0 to the numerator (evaluation.py gives them recall 1, which the "
    "aggregate ignores). The macro_mean path instead skips them with an explicit reason.",
    "LongMemEval turn metrics raise on empty gold, gold or ranked ids outside the corpus, and "
    "duplicate ids; the official code silently scores such inputs (recall_all of empty gold = 1).",
    "turn2session: corpus ids are rewritten to '{session}_{corpus position}' so the official "
    "strip_turn_id recovers the caller's session key; the official function then runs unchanged.",
]


# ---------------------------------------------------------------- LoCoMo official recall


def locomo_recall_acc(context_ids: Sequence[str], evidence: Sequence[str]) -> float:
    """LoCoMo per-row evidence recall: share of gold dia_ids present in the context.

    When the context lists sessions (``S3``) instead of turns, a gold ``D3:7``
    counts if its session number is listed. Rows without evidence score 1.0,
    matching the LoCoMo definition (callers exclude them from our means).
    """
    if not evidence:
        return 1.0
    if not context_ids:
        return 0.0
    if context_ids[0].startswith("S"):
        listed = {c[1:] for c in context_ids}
        found = sum(1 for dia in evidence if dia.split(":")[0][1:] in listed)
    else:
        listed = set(context_ids)
        found = sum(1 for dia in evidence if dia in listed)
    return found / len(evidence)


def locomo_recall_official(rows: Iterable[Tuple[Any, Sequence[str], Sequence[str]]]) -> Dict[str, Any]:
    """Official aggregate over ``(category, evidence, context_ids)`` rows (every QA row).

    ``by_category[c] = sum(round(recall, 3) over rows with evidence) / rows in c``;
    ``overall`` uses the same numerator over all rows.
    """
    total: Counter = Counter()
    recall_sum: Dict[Any, float] = {}
    for category, evidence, context_ids in rows:
        total[category] += 1
        if len(evidence) > 0:
            value = round(locomo_recall_acc(context_ids, evidence), 3)
            recall_sum[category] = recall_sum.get(category, 0.0) + value
    n = sum(total.values())
    if n == 0:
        raise ValueError("no rows")
    return {
        "overall": sum(recall_sum.values()) / n,
        "by_category": {c: recall_sum[c] / total[c] for c in sorted(recall_sum, key=str)},
        "n": n,
        "n_by_category": {c: total[c] for c in sorted(total, key=str)},
    }


# ---------------------------------------------------------------- LongMemEval official metrics


def _require_gold(gold: Sequence[str]) -> None:
    if len(gold) == 0:
        raise ValueError("empty gold: skip this question with an explicit reason")


def _rankings(ranked: Sequence[str], corpus_ids: Sequence[str], gold: Sequence[str]) -> List[int]:
    index = {cid: i for i, cid in enumerate(corpus_ids)}
    if len(index) != len(corpus_ids):
        raise ValueError("corpus_ids must be unique")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked ids must be unique")
    outside = [x for x in list(ranked) + list(gold) if x not in index]
    if outside:
        raise ValueError(f"ids outside the corpus: {outside[:5]}")
    return [index[x] for x in ranked]


def turn_metrics_at_k(ranked: Sequence[str], gold: Sequence[str], corpus_ids: Sequence[str], k: int) -> Dict[str, float]:
    """Official ``recall_any@k``, ``recall_all@k``, ``ndcg_any@k`` at item (turn) level."""
    _require_gold(gold)
    ra, rl, nd = evaluate_retrieval(_rankings(ranked, corpus_ids, gold), list(gold), list(corpus_ids), k=k)
    return {"recall_any": float(ra), "recall_all": float(rl), "ndcg_any": float(nd)}


def turn2session_metrics_at_k(
    ranked: Sequence[str],
    gold: Sequence[str],
    corpus_ids: Sequence[str],
    session_of: Mapping[str, str],
    k: int,
) -> Dict[str, float]:
    """Official turn2session metrics: grow k until the top list spans k unique sessions.

    Gold sessions are the sessions of the gold turns (official convention). For exact
    LongMemEval parity take ``corpus_ids`` and ``session_of`` from
    ``conversational_data.longmemeval_official_turn_view`` (user turns, 'noans' relabel).
    """
    _require_gold(gold)
    rankings = _rankings(ranked, corpus_ids, gold)
    official = [f"{session_of[c]}_{i}" for i, c in enumerate(corpus_ids)]
    position = {c: i for i, c in enumerate(corpus_ids)}
    correct = [official[position[g]] for g in gold]
    ra, rl, nd = evaluate_retrieval_turn2session(rankings, correct, official, k=k)
    return {"recall_any": float(ra), "recall_all": float(rl), "ndcg_any": float(nd)}


def session_metrics_at_k(
    ranked: Sequence[str],
    session_of: Mapping[str, str],
    gold_sessions: Sequence[str],
    k: int,
) -> Dict[str, float]:
    """Plain top-k turns (no k growth): ``session_hit`` (any gold session) and ``session_recall_all``."""
    _require_gold(gold_sessions)
    hit = {session_of[t] for t in ranked[:k]}
    return {
        "session_hit": float(any(s in hit for s in gold_sessions)),
        "session_recall_all": float(all(s in hit for s in gold_sessions)),
    }


# ---------------------------------------------------------------- budgeted / channel analysis


def budgeted_coverage(packed_ids: Iterable[str], gold: Sequence[str]) -> Dict[str, float]:
    """E-series coverage of a packed context: all gold packed, gold fraction, any gold."""
    _require_gold(gold)
    packed = set(packed_ids)
    hit = sum(g in packed for g in set(gold))
    n_gold = len(set(gold))
    return {"complete_coverage": float(hit == n_gold), "recall": hit / n_gold, "any_hit": float(hit > 0)}


def _recall(top: Sequence[str], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    return len(gold_set & set(top)) / len(gold_set)


def complementarity(channel_rankings: Mapping[str, Sequence[str]], gold: Sequence[str], k: int) -> Dict[str, Any]:
    """How much each channel's top-k adds: hits, unique hits, overlap, oracle union.

    ``pairwise_jaccard`` compares the channels' top-k id sets (1.0 when both are empty).
    Id lists keep gold order.
    """
    _require_gold(gold)
    tops = {name: set(channel_rankings[name][:k]) for name in sorted(channel_rankings)}
    hits = {name: [g for g in gold if g in top] for name, top in tops.items()}
    unique = {
        name: [g for g in hits[name] if not any(g in tops[other] for other in tops if other != name)]
        for name in tops
    }
    jaccard = {
        f"{a}|{b}": (len(tops[a] & tops[b]) / len(tops[a] | tops[b])) if tops[a] | tops[b] else 1.0
        for a, b in combinations(tops, 2)
    }
    union = set().union(*tops.values())
    return {
        "k": k,
        "hits": hits,
        "unique_hits": unique,
        "pairwise_jaccard": jaccard,
        "oracle_union_recall_all": float(all(g in union for g in gold)),
        "oracle_union_recall_any": float(any(g in union for g in gold)),
        "oracle_union_recall": _recall(list(union), gold),
    }


def fusion_regret(
    fused_ids: Sequence[str],
    channel_rankings: Mapping[str, Sequence[str]],
    gold: Sequence[str],
    k: int,
) -> Dict[str, Any]:
    """Gold-fraction recall@k lost by fusion versus the best channel and the oracle union.

    Negative regret means fusion beat the reference. ``lost_gold`` lists gold ids some
    channel had in its top-k that the fused top-k dropped. Best-channel ties break by name.
    """
    _require_gold(gold)
    fused_top = list(fused_ids[:k])
    fused = _recall(fused_top, gold)
    per = {name: _recall(channel_rankings[name][:k], gold) for name in sorted(channel_rankings)}
    if not per:
        raise ValueError("no channels")
    best = min(per, key=lambda name: (-per[name], name))
    union = set().union(*(set(r[:k]) for r in channel_rankings.values()))
    union_recall = _recall(list(union), gold)
    return {
        "k": k,
        "fused_recall": fused,
        "channel_recall": per,
        "best_channel": best,
        "best_channel_recall": per[best],
        "regret_vs_best_channel": per[best] - fused,
        "oracle_union_recall": union_recall,
        "regret_vs_oracle_union": union_recall - fused,
        "lost_gold": [g for g in gold if g in union and g not in fused_top],
    }


# ---------------------------------------------------------------- aggregation


PROTOCOLS = ("locomo", "locomo_audited", "lme_official", "lme_any_role")


def scoring_gold(question: Any, protocol: str) -> Tuple[Tuple[str, ...], Optional[str]]:
    """``(gold, skip_reason)`` for a ``conversational_data.Question``; reason None = score it.

    - ``locomo``: resolved non-empty ``gold_evidence`` (1,981 of 1,986 on locomo10).
    - ``locomo_audited``: also skips rows with a malformed/unresolved annotation (E1: 1,977).
    - ``lme_official``: skips ``_abs``; gold = user-side has_answer turns
      (official run_retrieval.py denominator).
    - ``lme_any_role``: skips ``_abs``; gold = has_answer turns of any role (E-series).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    if protocol.startswith("locomo"):
        if protocol == "locomo_audited" and question.malformed_evidence:
            return (), "malformed_evidence"
        gold = tuple(question.gold_evidence)
        return gold, (None if gold else "no_gold_evidence")
    if question.abstention:
        return (), "abstention"
    gold = tuple(question.gold_user_evidence if protocol == "lme_official" else question.gold_evidence)
    return gold, (None if gold else "no_answer_turn")


@dataclass(frozen=True)
class MetricRow:
    """One question's metric values, or the reason it was not scored."""

    qid: str
    cluster: str  # conversation / scope key for the cluster bootstrap
    category: str
    values: Mapping[str, float] = field(default_factory=dict)
    skip_reason: Optional[str] = None


def macro_mean(rows: Sequence[MetricRow], metric: str) -> Dict[str, Any]:
    """Mean over scored rows with explicit denominators and skip reasons (mean None if n=0)."""
    scored = [float(r.values[metric]) for r in rows if r.skip_reason is None]
    skipped = Counter(r.skip_reason for r in rows if r.skip_reason is not None)
    return {
        "mean": sum(scored) / len(scored) if scored else None,
        "n": len(scored),
        "n_total": len(rows),
        "n_skipped": sum(skipped.values()),
        "skipped": dict(sorted(skipped.items())),
    }


def by_category(rows: Sequence[MetricRow], metric: str) -> Dict[str, Dict[str, Any]]:
    """``macro_mean`` for ``all`` and each category (sorted)."""
    out = {"all": macro_mean(rows, metric)}
    for category in sorted({r.category for r in rows}):
        out[category] = macro_mean([r for r in rows if r.category == category], metric)
    return out


def _index(rows: Sequence[MetricRow]) -> Dict[str, MetricRow]:
    out = {r.qid: r for r in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate qid in rows")
    return out


def bootstrap(
    rows: Sequence[MetricRow],
    metric: str,
    *,
    baseline: Optional[Sequence[MetricRow]] = None,
    cluster: bool = False,
    seed: int = DEFAULT_SEED,
    n_resamples: int = N_RESAMPLES,
) -> Dict[str, Any]:
    """Seeded percentile-bootstrap 95% CI of the mean, or of the paired mean difference.

    With ``baseline`` the arms must cover the same qids (and clusters); a question is
    scored only when neither arm skipped it. ``cluster=True`` resamples conversations
    (``cluster_bootstrap``), otherwise questions (``eval_stats.bootstrap_ci``).
    """
    arm = _index(rows)
    base = _index(baseline) if baseline is not None else None
    if base is not None and set(base) != set(arm):
        raise ValueError("paired arms must cover the same qids")
    pairs: List[Tuple[str, float]] = []
    skipped: Counter = Counter()
    for qid in sorted(arm):
        row, ref = arm[qid], (base[qid] if base is not None else None)
        reason = row.skip_reason or (ref.skip_reason if ref is not None else None)
        if reason is not None:
            skipped[reason] += 1
            continue
        if ref is not None and ref.cluster != row.cluster:
            raise ValueError(f"{qid}: cluster differs between arms")
        value = float(row.values[metric]) - (float(ref.values[metric]) if ref is not None else 0.0)
        pairs.append((row.cluster, value))
    if not pairs:
        raise ValueError("no scored rows")
    if cluster:
        ci = cluster_bootstrap(pairs, seed, samples=n_resamples)
        ci = {"mean": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"], "n": ci["n"], "clusters": ci["clusters"]}
    else:
        ci = bootstrap_ci([v for _, v in pairs], n_resamples=n_resamples, seed=seed)
        ci["clusters"] = len({c for c, _ in pairs})
    return {
        **ci,
        "paired": base is not None,
        "method": "conversation-cluster" if cluster else "question",
        "seed": seed,
        "n_resamples": n_resamples,
        "n_skipped": sum(skipped.values()),
        "skipped": dict(sorted(skipped.items())),
    }
