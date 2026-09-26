"""Engine-driven channel ablations for the tri-signal retriever.

Runs ``engine.trisignal.TriSignalRetriever`` (the same code behind
``POST /retrieve``) over official LoCoMo and LongMemEval-S (cleaned) scopes and
scores every arm with exact evidence IDs:

- turn recall_any@k / recall_all@k (complete evidence) / ndcg_any@k,
- session hit@k and session recall_all@k,
- LoCoMo official evidence recall (fraction of gold dia_ids in the top-k),
- budgeted complete coverage after evidence packing,
- channel complementarity and fusion regret for the all-channel arm.

Dense sources:
  --dense none            offline arms only (sparse, graph)
  --dense emg-reference   LoCoMo only: EMG's shipped text-embedding-3-small
                          vectors, frozen LLM query keys and LLM entity graph
                          (reference track; 1536-d, NOT the 4096-d runtime model)
  --dense cache:PATH      4096-d vectors replayed from an EmbeddingCache; any
                          miss fails the run (fill it on the VPS first)

Every run writes an immutable ledger (manifest, rows, completion) plus a JSON
summary. Provider calls: zero for all three dense sources.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmarks import conversational_data as data
from benchmarks import conversational_metrics as M
from engine.evidence import assemble_evidence
from engine.trisignal import RetrievalConfig, ScopeIndex, TriSignalRetriever, config_sha256
from eval_ledger import LedgerWriter, compute_pool_metrics

ROOT = Path(__file__).resolve().parent
DEFAULT_PATHS = {
    "locomo": ROOT / "memorybench/data/benchmarks/locomo/locomo10.json",
    "longmemeval": ROOT / "memorybench/data/benchmarks/longmemeval/longmemeval_s_cleaned.json",
}
KS = (1, 5, 10, 25, 50)
BUDGETS = (1024, 2048, 4096)
EMG_LLM_EXTRACTOR = "llm:emg-gpt-3.5-turbo-v4"

BASE = RetrievalConfig(top_k=50, channel_k=100)
ARMS: Dict[str, RetrievalConfig] = {
    "dense": replace(BASE, channels=("dense",)),
    "sparse": replace(BASE, channels=("sparse",)),
    "graph": replace(BASE, channels=("graph",)),
    "graph-ppr": replace(BASE, channels=("graph",), graph_method="ppr"),
    "dense+sparse": replace(BASE, channels=("dense", "sparse")),
    "sparse+graph": replace(BASE, channels=("sparse", "graph")),
    "dense+graph": replace(BASE, channels=("dense", "graph")),
    "all": replace(BASE, channels=("dense", "sparse", "graph")),
    "all-dbsf": replace(BASE, channels=("dense", "sparse", "graph"), fusion="dbsf"),
    "all-zscore": replace(BASE, channels=("dense", "sparse", "graph"), fusion="zscore"),
    "all-ppr-seeded": replace(
        BASE, channels=("dense", "sparse", "graph"), graph_method="ppr", ppr_passage_seed_channel="dense",
    ),
}
OFFLINE_ARMS = ("sparse", "graph", "graph-ppr", "sparse+graph")


class ReferenceDense:
    """EMG reference artifacts for one LoCoMo conversation (no network)."""

    def __init__(self, sample_id: str, corpus, questions) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import reproduce_emg_locomo as repro  # noqa: E402  (pinned upstream loader)

        inputs = repro.load_inputs(sample_id)
        # Query vectors and keys are aligned by QA position; prove the order.
        if [q.question for q in questions] != [str(r["question"]) for r in inputs.sample["qa"]]:
            raise ValueError(f"{sample_id}: dataset questions do not match the EMG artifact order")
        index = inputs.index
        by_dia = {}
        for mid, vector in zip(index.memory_ids, index.vectors):
            memory = inputs.memory_graph.memories[mid]
            by_dia[memory.dia_id] = np.asarray(vector, dtype=np.float32)
        self.vectors = {t.node_id: by_dia[t.evidence_id] for t in corpus.turns}
        self.query_vectors = inputs.query_vectors
        self.qkeys = inputs.qkeys
        mentions: Dict[str, list] = {}
        dia_to_node = {t.evidence_id: t.node_id for t in corpus.turns}
        from engine.entity_extraction import EntityMention

        graph = inputs.entity_graph
        for edge in graph.edges:
            entity = graph.entities[edge.entity_id]
            node = dia_to_node[graph.memories[edge.memory_id].dia_id]
            mentions.setdefault(node, []).append(EntityMention(key=entity.key, value=entity.value, type=entity.type))
        self.mentions = {t.node_id: mentions.get(t.node_id, []) for t in corpus.turns}


def _scope(conv, reference: Optional[ReferenceDense]):
    if reference is not None:
        return ScopeIndex(
            conv.corpus,
            vectors=lambda: reference.vectors,
            expected_dim=None,
            stored_mentions=lambda extractor: reference.mentions,
        )
    return ScopeIndex(conv.corpus, vectors=None)


def evaluate_question(
    retriever: TriSignalRetriever,
    scope: ScopeIndex,
    question: data.Question,
    arms: Mapping[str, RetrievalConfig],
    protocol: str,
    *,
    query_vector: Optional[np.ndarray],
    query_keys: Optional[Sequence[str]],
) -> Dict[str, Any]:
    corpus = scope.corpus
    gold, skip = M.scoring_gold(question, protocol)
    session_of = {t.evidence_id: t.session_key for t in corpus.turns}
    corpus_ids = [t.evidence_id for t in corpus.turns]
    # Sessions are derived from the scored gold turns so session and turn
    # metrics share one denominator.
    gold_sessions = sorted({session_of[g] for g in gold}) if skip is None else []
    out: Dict[str, Any] = {"skip_reason": skip, "arms": {}}
    for name, config in arms.items():
        needs_vector = "dense" in config.channels
        result = retriever.retrieve(
            scope,
            question.question,
            config,
            query_vector=query_vector if needs_vector else None,
            query_keys=query_keys if "graph" in config.channels else None,
        )
        ranked = [corpus.turn(nid).evidence_id for nid, _ in result.fused]
        arm: Dict[str, Any] = {"ranked": ranked[:50], "config_sha256": result.trace["resolved_config_sha256"]}
        if skip is None:
            values: Dict[str, float] = {}
            for k in KS:
                for key, value in M.turn_metrics_at_k(ranked, gold, corpus_ids, k).items():
                    values[f"{key}@{k}"] = value
                for key, value in M.session_metrics_at_k(ranked, session_of, gold_sessions, k).items():
                    values[f"{key}@{k}"] = value
                if protocol.startswith("locomo"):
                    values[f"locomo_recall@{k}"] = M.locomo_recall_acc(ranked[:k], gold)
            for budget in BUDGETS:
                for strategy in ("turn", "propagate"):
                    pack = assemble_evidence(corpus, result.fused, strategy=strategy, budget_tokens=budget)
                    cov = M.budgeted_coverage([i.evidence_id for i in pack.items], gold)
                    values[f"complete@{strategy}{budget}"] = cov["complete_coverage"]
            arm["metrics"] = values
            if len([c for c in result.channels.values() if c.executed]) == 3:
                channel_ids = {
                    c: [corpus.turn(nid).evidence_id for nid, _ in run.ranking]
                    for c, run in result.channels.items()
                }
                arm["complementarity@25"] = M.complementarity(channel_ids, gold, 25)
                arm["fusion_regret@25"] = M.fusion_regret(ranked, channel_ids, gold, 25)
        out["arms"][name] = arm
    return out


def summarize(rows: List[Dict[str, Any]], arms: Iterable[str], baseline: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for arm in arms:
        metric_names = sorted({m for r in rows for m in r["arms"][arm].get("metrics", {})})
        per_metric = {}
        for metric in metric_names:
            metric_rows = [
                M.MetricRow(
                    qid=r["qid"], cluster=r["cluster"], category=str(r["category"]),
                    values={metric: r["arms"][arm]["metrics"][metric]} if "metrics" in r["arms"][arm] else {},
                    skip_reason=r["skip_reason"],
                )
                for r in rows
            ]
            entry = M.bootstrap(metric_rows, metric, cluster=True)
            if arm != baseline:
                base_rows = [
                    replace(row, values={metric: r["arms"][baseline]["metrics"][metric]} if "metrics" in r["arms"][baseline] else {})
                    for row, r in zip(metric_rows, rows)
                ]
                entry["vs_" + baseline] = M.bootstrap(metric_rows, metric, baseline=base_rows, cluster=True)
            entry["by_category"] = {c: v["mean"] for c, v in M.by_category(metric_rows, metric).items()}
            per_metric[metric] = entry
        comp = [r["arms"][arm]["complementarity@25"] for r in rows if "complementarity@25" in r["arms"][arm]]
        if comp:
            per_metric["complementarity@25"] = {
                "oracle_union_recall_all": float(np.mean([c["oracle_union_recall_all"] for c in comp])),
                # Gold turns that only this channel retrieved in its top 25 (summed over questions).
                "unique_gold_hits": {ch: sum(len(c["unique_hits"].get(ch, ())) for c in comp) for ch in ("dense", "sparse", "graph")},
                "gold_hits": {ch: sum(len(c["hits"].get(ch, ())) for c in comp) for ch in ("dense", "sparse", "graph")},
            }
            regret = [r["arms"][arm]["fusion_regret@25"] for r in rows if "fusion_regret@25" in r["arms"][arm]]
            per_metric["fusion_regret@25"] = {
                "mean_regret_vs_oracle_union": float(np.mean([g["regret_vs_oracle_union"] for g in regret])),
                "mean_regret_vs_best_channel": float(np.mean([g["regret_vs_best_channel"] for g in regret])),
            }
        summary[arm] = per_metric
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_PATHS), required=True)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--arms", default=",".join(OFFLINE_ARMS))
    parser.add_argument("--baseline", default="sparse")
    parser.add_argument("--dense", default="none", help="none | emg-reference | cache:PATH")
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--query-instruction", default="")
    parser.add_argument("--graph-extractor", default=None, help="override; emg-reference uses the EMG LLM graph")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/results")
    args = parser.parse_args(argv)

    path = args.path or DEFAULT_PATHS[args.dataset]
    identity = data.dataset_identity(path)
    arms = {name: ARMS[name] for name in args.arms.split(",")}
    if args.baseline not in arms:
        parser.error("--baseline must be one of the selected arms")
    if args.dense == "none" and any("dense" in c.channels for c in arms.values()):
        parser.error("dense arms need --dense emg-reference or --dense cache:PATH")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.dense == "emg-reference":
        if args.dataset != "locomo":
            parser.error("emg-reference vectors exist only for LoCoMo")
        if identity.get("known") not in {"locomo10", "emg_locomo10"}:
            parser.error("emg-reference requires the official locomo10.json (EMG artifacts are aligned to it)")
        arms = {n: replace(c, graph_extractor=EMG_LLM_EXTRACTOR, graph_query_extractor=EMG_LLM_EXTRACTOR) for n, c in arms.items()}
    elif args.graph_extractor:
        arms = {n: replace(c, graph_extractor=args.graph_extractor) for n, c in arms.items()}
    protocol = args.protocol or ("locomo" if args.dataset == "locomo" else "lme_any_role")

    cache = None
    if args.dense.startswith("cache:"):
        from engine.embedding_cache import CacheItem, EmbeddingCache

        cache = EmbeddingCache(args.dense[len("cache:"):], model_id=args.embed_model, expected_dim=4096)

    conversations = (
        data.load_locomo(path) if args.dataset == "locomo" else data.load_longmemeval(path, limit=args.limit)
    )
    config = {
        "dataset": args.dataset, "protocol": protocol, "dense": args.dense, "embed_model": args.embed_model,
        "query_instruction": args.query_instruction, "query_style": "qwen3", "ks": KS, "budgets": BUDGETS,
        "label": args.label, "baseline": args.baseline, "limit": args.limit, "path": str(path),
        "arms": {n: {"config": c.resolved(), "sha256": config_sha256(c)} for n, c in arms.items()},
    }
    ledger = LedgerWriter(
        f"trisignal_{args.dataset}", config,
        provenance={"dataset": identity, "python": platform.python_version()},
        results_dir=args.out_dir / "ledgers",
    )
    retriever = TriSignalRetriever()
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for conv in conversations:
            if args.dataset == "locomo" and args.limit and len({r["cluster"] for r in rows}) >= args.limit:
                break
            reference = (
                ReferenceDense(conv.scope_key.split(":", 1)[1], conv.corpus, conv.questions)
                if args.dense == "emg-reference" else None
            )
            if cache is not None:
                items = [CacheItem("doc", t.search_text) for t in conv.corpus.turns]
                vectors = cache.require_all(items)
                ids = [t.node_id for t in conv.corpus.turns]
                keyed = {nid: vectors[cache.key(it)] for nid, it in zip(ids, items)}
                scope = ScopeIndex(conv.corpus, vectors=lambda keyed=keyed: keyed, expected_dim=4096)
            else:
                scope = _scope(conv, reference)
            for index, question in enumerate(conv.questions):
                query_vector = None
                if reference is not None:
                    query_vector = reference.query_vectors[index]
                elif cache is not None:
                    item = CacheItem("query", question.question, args.query_instruction)
                    query_vector = cache.require_all([item])[cache.key(item)]
                query_keys = sorted(reference.qkeys[index]) if reference is not None else None
                result = evaluate_question(
                    retriever, scope, question, arms, protocol, query_vector=query_vector, query_keys=query_keys,
                )
                row = {
                    "qid": question.qid, "cluster": conv.scope_key, "category": question.category_name,
                    **result,
                }
                rows.append(row)
                gold_ids = set(M.scoring_gold(question, protocol)[0])
                baseline_ranked = row["arms"][args.baseline]["ranked"]
                ledger.write(
                    question_id=question.qid,
                    question_type=question.category_name,
                    gold_evidence_ids=sorted(gold_ids),
                    # Ledger pool metrics describe the baseline arm; every arm's
                    # ranking and metrics are in ``extra``.
                    pool_metrics=compute_pool_metrics(
                        [{"node_id": e} for e in baseline_ranked],
                        lambda r: r["node_id"] in gold_ids,
                        [k for k in KS if k <= 25],
                    ),
                    status="completed",
                    extra={"skip_reason": row["skip_reason"], "arms": row["arms"]},
                )
        if not any(r["skip_reason"] is None for r in rows):
            raise ValueError("no scored questions")
        summary = {
            "label": args.label,
            "dataset": identity,
            "protocol": protocol,
            "dense": args.dense,
            "provider_calls": 0,
            "n_questions": len(rows),
            "n_scored": sum(1 for r in rows if r["skip_reason"] is None),
            "elapsed_s": round(time.perf_counter() - started, 1),
            "arms": {n: {"config_sha256": config_sha256(c)} for n, c in arms.items()},
            "results": summarize(rows, arms, args.baseline),
            "ledger": ledger.path.name,
        }
        out = args.out_dir / f"trisignal-{args.dataset}-{args.label}.json"
        with out.open("xb") as handle:  # never overwrite an earlier result
            handle.write((json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"))
        ledger.finalize(status="completed", summary={"n": len(rows), "label": args.label})
    except Exception as exc:
        ledger.finalize_failure(reason="evaluation aborted", error_type=type(exc).__name__)
        raise
    print(f"wrote {out} ({len(rows)} questions, {summary['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
