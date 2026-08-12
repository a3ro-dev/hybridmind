"""
Quick retrieval-only LoCoMo evaluation.
Calls the live HybridMind /search/hybrid endpoint for a sample of questions
and computes Hit@k / MRR using answer-text overlap as relevance signal.

Usage (single config):
  python eval_locomo_retrieval.py [--vector-weight 0.5] [--graph-weight 0.15] ...

Usage (grid sweep — vary vector_weight × bm25_boost × rerank_pool):
  python eval_locomo_retrieval.py --sweep

Optional filters:
  --category single-hop|multi-hop|temporal|world-knowledge|adversarial
  --n 5        samples per category (default 5)
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import re
import httpx
from pathlib import Path
from statistics import mean

import eval_common
import eval_ledger
from engine.query_router import route_query

BASE_URL = os.getenv("HYBRIDMIND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")

CATEGORY_MAP = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "world-knowledge", 5: "adversarial"}


def parse_args():
    p = argparse.ArgumentParser(description="LoCoMo retrieval eval")
    p.add_argument("--vector-weight",    type=float, default=0.5)
    p.add_argument("--graph-weight",     type=float, default=0.15)
    p.add_argument("--bm25-boost",       type=float, default=0.35)
    p.add_argument("--overlap-threshold",type=float, default=0.15)
    p.add_argument("--rerank-pool",      type=int,   default=25)
    p.add_argument("--top-k",            type=int,   default=15)
    p.add_argument(
        "--search-mode",
        choices=["hybrid", "vector_only", "sparse_only", "vector_sparse", "graph_only"],
        default="hybrid",
        help="Controlled retrieval mode used by scripts/ablation_matrix.py",
    )
    p.add_argument(
        "--route-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow query routing to override explicit weights (disable for ablations)",
    )
    p.add_argument(
        "--track-access",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mutate access-frequency state during retrieval (off for deterministic runs)",
    )
    p.add_argument(
        "--graph-anchor-strategy",
        choices=["explicit", "vector_top1"],
        default="explicit",
        help="For graph_only, use --anchor-node-id or a separately recorded vector top-1 seed",
    )
    p.add_argument("--anchor-node-id", action="append", default=[])
    p.add_argument("--category",         type=str,   default=None,
                   help="Filter to one category: single-hop, multi-hop, temporal, world-knowledge, adversarial")
    p.add_argument("--n",                type=int,   default=5,
                   help="Samples per category (default 5)")
    p.add_argument("--sweep",            action="store_true",
                   help="Grid-search over key params and print ranked table")
    p.add_argument("--with-answers",     action="store_true",
                   help="Also run LLM QA-answering + accuracy scoring on top of retrieval")
    p.add_argument("--answer-model",     type=str,   default=None,
                   help="Override the Z.AI answering model (default: HYBRIDMIND_QA_MODEL or glm-4.6)")
    p.add_argument("--decompose-multihop", dest="decompose_multihop", action="store_true", default=True,
                   help="6.2.2: decompose multihop-routed queries into sub-questions (default on for eval)")
    p.add_argument("--no-decompose-multihop", dest="decompose_multihop", action="store_false")
    return p.parse_args()


def load_questions(samples_per_category: int = 5, category_filter: str | None = None):
    data = json.loads(LOCOMO_PATH.read_text())
    questions = []
    counts: dict = {}
    for conv in data:
        for qa in conv.get("qa", []):
            cat = CATEGORY_MAP.get(qa.get("category", 0), "unknown")
            if category_filter and cat != category_filter:
                continue
            if samples_per_category > 0 and counts.get(cat, 0) >= samples_per_category:
                continue
            counts[cat] = counts.get(cat, 0) + 1
            ans = qa.get("answer") or qa.get("adversarial_answer", "")
            question_text = qa["question"]
            questions.append({
                "question_id": qa.get("question_id") or hashlib.sha1(question_text.encode()).hexdigest()[:12],
                "question": question_text,
                "answer": str(ans).strip(),
                "category": cat,
                "evidence": qa.get("evidence", []),
            })
    return questions


def answer_tokens(answer: str) -> set:
    tokens = set(re.findall(r"[A-Za-z0-9']+", answer.lower()))
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was", "it", "and", "or", "but"}
    return tokens - stopwords


def is_relevant(retrieved_text: str, answer: str) -> bool:
    text_lower = retrieved_text.lower()
    answer_lower = answer.lower()
    if answer_lower in text_lower:
        return True
    answer_toks = answer_tokens(answer)
    if not answer_toks:
        return False
    text_toks = set(re.findall(r"[A-Za-z0-9']+", text_lower))
    if len(answer_toks) <= 3:
        return answer_toks.issubset(text_toks)
    overlap = len(answer_toks & text_toks)
    return overlap / len(answer_toks) >= 0.7


def run_eval(
    questions: list,
    client: httpx.Client,
    *,
    vector_weight: float = 0.5,
    graph_weight: float = 0.15,
    bm25_boost: float = 0.35,
    overlap_threshold: float = 0.15,
    rerank_pool: int = 25,
    top_k: int = 15,
    verbose: bool = True,
    with_answers: bool = False,
    answer_model: str | None = None,
    decompose_multihop: bool = True,
    search_mode: str = "hybrid",
    route_weights: bool = True,
    track_access: bool = False,
    graph_anchor_strategy: str = "explicit",
    anchor_node_ids: list[str] | None = None,
) -> dict:
    all_hits_1, all_hits_5, all_hits_10, all_mrr, all_prec_5, all_prec_10 = [], [], [], [], [], []
    all_correct: list = []
    results_by_category: dict = {}

    ledger_config = {
        "benchmark": "locomo",
        "vector_weight": vector_weight,
        "graph_weight": graph_weight,
        "bm25_boost": bm25_boost,
        "overlap_threshold": overlap_threshold,
        "rerank_pool": rerank_pool,
        "top_k": top_k,
        "with_answers": with_answers,
        "answer_model": answer_model or eval_common.DEFAULT_ANSWER_MODEL,
        "search_mode": search_mode,
        "route_weights": route_weights,
        "track_access": track_access,
        "graph_anchor_strategy": graph_anchor_strategy,
        "anchor_node_ids": list(anchor_node_ids or []),
    }
    ledger = eval_ledger.LedgerWriter("locomo", ledger_config)
    # Request enough candidates that the full pre-rerank pool is visible for ledger metrics.
    pool_top_k = max(top_k, rerank_pool, max(eval_ledger.DEFAULT_K_LIST))

    for i, q in enumerate(questions):
        question = q["question"]
        answer = q["answer"]
        category = q["category"]

        if verbose:
            safe_q = question[:80].encode("ascii", "backslashreplace").decode("ascii")
            safe_a = answer[:80].encode("ascii", "backslashreplace").decode("ascii")
            print(f"[{i+1}/{len(questions)}] [{category}] {safe_q}...")
            print(f"  Answer: {safe_a}...")

        qtype = route_query(question)["type"]

        def _post(q_text: str) -> list:
            anchors = list(anchor_node_ids or [])
            if search_mode == "graph_only" and not anchors and graph_anchor_strategy == "vector_top1":
                seed = client.post("/search/hybrid", json={
                    "query_text": q_text,
                    "top_k": 1,
                    "min_score": 0.0,
                    "vector_weight": 1.0,
                    "graph_weight": 0.0,
                    "bm25_boost_weight": 0.0,
                    "rerank_pool": 1,
                    "search_mode": "vector_only",
                    "route_weights": False,
                    "track_access": False,
                })
                seed.raise_for_status()
                seed_results = seed.json().get("results", [])
                if seed_results:
                    anchors = [seed_results[0]["node_id"]]
            if search_mode == "graph_only" and not anchors:
                raise ValueError(
                    "graph_only requires --anchor-node-id or "
                    "--graph-anchor-strategy vector_top1"
                )

            payload = {
                "query_text": q_text,
                "top_k": pool_top_k,
                "min_score": 0.0,
                "vector_weight": vector_weight,
                "graph_weight": graph_weight,
                "bm25_boost_weight": bm25_boost,
                "overlap_threshold": overlap_threshold,
                "rerank_pool": rerank_pool,
                "search_mode": search_mode,
                "route_weights": route_weights,
                "track_access": track_access,
            }
            if anchors:
                payload["anchor_node_ids"] = anchors
            resp = client.post("/search/hybrid", json=payload)
            resp.raise_for_status()
            return resp.json().get("results", [])

        try:
            res_list = eval_common.retrieve_with_decomposition(
                question, qtype, _post, decompose_enabled=decompose_multihop
            )
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            continue

        for k, hits_list, prec_list in [
            (1, all_hits_1, None),
            (5, all_hits_5, all_prec_5),
            (10, all_hits_10, all_prec_10),
        ]:
            top_k_res = res_list[:k]
            relevant = [r for r in top_k_res if is_relevant(r.get("text", ""), answer)]
            hit = 1.0 if relevant else 0.0
            hits_list.append(hit)
            if prec_list is not None:
                prec_list.append(len(relevant) / k)

        mrr = 0.0
        for rank, r in enumerate(res_list[:10], 1):
            if is_relevant(r.get("text", ""), answer):
                mrr = 1.0 / rank
                break
        all_mrr.append(mrr)

        hypothesis, judged_correct, judge_rationale = "", False, "not evaluated (retrieval-only run, pass --with-answers)"
        prompt_version = ""
        if with_answers:
            snippets = [r.get("text", "") for r in res_list[:10] if r.get("text")]
            hypothesis, prompt_version = eval_common.answer_question(
                question, snippets, question_type=qtype, model=answer_model
            )
            judged_correct, judge_rationale = eval_common.judge_correct_normalized(hypothesis, answer)
            all_correct.append(1.0 if judged_correct else 0.0)
            if verbose:
                print(f"  QA answer: {hypothesis[:100]!r} -> {'CORRECT' if judged_correct else 'WRONG'}")

        pool_metrics = eval_ledger.compute_pool_metrics(
            res_list, lambda r: is_relevant(r.get("text", ""), answer)
        )
        ledger.write(
            question_id=q["question_id"],
            question_type=qtype,
            gold_evidence_ids=[str(e) for e in q.get("evidence", [])],
            pool_metrics=pool_metrics,
            raw_llm_answer=hypothesis,
            judged_correct=judged_correct,
            judge_rationale=judge_rationale,
            prompt_version=prompt_version,
        )

        if verbose:
            print(f"  Hit@1={all_hits_1[-1]:.0%} Hit@5={all_hits_5[-1]:.0%} Hit@10={all_hits_10[-1]:.0%} MRR={mrr:.3f}")
            for rank, r in enumerate(res_list[:3], 1):
                text = r.get("text", "")[:120]
                vs = r.get("vector_score", 0)
                gs = r.get("graph_score", 0)
                gg = r.get("graph_gate", 1.0)
                egs = r.get("effective_graph_score", gs)
                cs = r.get("combined_score", 0)
                rel = "[OK]" if is_relevant(text, answer) else "[NO]"
                print(f"    #{rank} {rel} v={vs:.3f} g={gs:.3f} gate={gg:.3f} efg={egs:.3f} c={cs:.3f} {text[:100]}")

        if category not in results_by_category:
            results_by_category[category] = {"hits": [], "mrrs": [], "correct": []}
        results_by_category[category]["hits"].append(all_hits_5[-1])
        results_by_category[category]["mrrs"].append(mrr)
        if with_answers:
            results_by_category[category]["correct"].append(all_correct[-1])

        time.sleep(0.05)

    if not all_hits_1:
        return {}

    n = len(all_hits_1)
    summary = {
        "hit_at_1":       mean(all_hits_1),
        "hit_at_5":       mean(all_hits_5),
        "hit_at_10":      mean(all_hits_10),
        "precision_at_5": mean(all_prec_5),
        "precision_at_10":mean(all_prec_10),
        "mrr":            mean(all_mrr),
        "n":              n,
        "by_category":    {
            cat: {
                "hit5": mean(v["hits"]), "mrr": mean(v["mrrs"]),
                **({"accuracy": mean(v["correct"])} if with_answers and v["correct"] else {}),
            }
            for cat, v in results_by_category.items()
        },
    }
    if with_answers and all_correct:
        summary["accuracy"] = mean(all_correct)
    return summary


def print_summary(summary: dict, label: str = ""):
    n = summary.get("n", 0)
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Questions evaluated: {n}")
    print(f"  Hit@1:  {summary['hit_at_1']:.4f}  ({round(summary['hit_at_1']*n)}/{n})")
    print(f"  Hit@5:  {summary['hit_at_5']:.4f}  ({round(summary['hit_at_5']*n)}/{n})")
    print(f"  Hit@10: {summary['hit_at_10']:.4f}  ({round(summary['hit_at_10']*n)}/{n})")
    print(f"  Prec@5:  {summary['precision_at_5']:.4f}")
    print(f"  Prec@10: {summary['precision_at_10']:.4f}")
    print(f"  MRR:    {summary['mrr']:.4f}")
    if "accuracy" in summary:
        print(f"  Accuracy (LLM QA): {summary['accuracy']:.4f}  ({round(summary['accuracy']*n)}/{n})")
    print()
    print("  --- By Category ---")
    for cat in sorted(summary["by_category"].keys()):
        c = summary["by_category"][cat]
        acc_str = f" Accuracy={c['accuracy']:.4f}" if "accuracy" in c else ""
        print(f"  {cat}: Hit@5={c['hit5']:.4f} MRR={c['mrr']:.4f}{acc_str}")
    print()
    acc_str = f" Accuracy={summary['accuracy']:.3f}" if "accuracy" in summary else ""
    print(f"  SUMMARY: Hit@1={summary['hit_at_1']:.3f} Hit@5={summary['hit_at_5']:.3f} "
          f"Hit@10={summary['hit_at_10']:.3f} Prec@5={summary['precision_at_5']:.3f} "
          f"Prec@10={summary['precision_at_10']:.3f} MRR={summary['mrr']:.4f}{acc_str}")


def sweep(questions: list, client: httpx.Client):
    """Coarse grid search over the highest-impact dimensions."""
    grid = {
        "vector_weight":     [0.4, 0.5, 0.6, 0.7],
        "bm25_boost":        [0.25, 0.35, 0.5],
        "rerank_pool":       [10, 25, 40],   # 10 <= top_k disables reranker
        "graph_weight":      [0.0, 0.1, 0.2],
        "overlap_threshold": [0.10, 0.15, 0.25],
    }
    # Fixed dims during sweep
    top_k = 15

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)
    print(f"Sweep: {total} configurations × {len(questions)} questions ...\n")

    results = []
    for idx, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        label = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"[{idx:3d}/{total}] {label}", end="  ", flush=True)
        s = run_eval(
            questions, client,
            vector_weight=cfg["vector_weight"],
            graph_weight=cfg["graph_weight"],
            bm25_boost=cfg["bm25_boost"],
            overlap_threshold=cfg["overlap_threshold"],
            rerank_pool=cfg["rerank_pool"],
            top_k=top_k,
            verbose=False,
        )
        if not s:
            print("SKIP")
            continue
        h1, h5, mrr = s["hit_at_1"], s["hit_at_5"], s["mrr"]
        print(f"Hit@1={h1:.3f} Hit@5={h5:.3f} MRR={mrr:.4f}")
        results.append({"cfg": cfg, **s})

    if not results:
        print("No results.")
        return

    # Rank by Hit@5 (primary) then MRR (tiebreak)
    results.sort(key=lambda r: (-r["hit_at_5"], -r["mrr"]))

    print(f"\n{'='*90}")
    print("  TOP-10 CONFIGURATIONS (ranked by Hit@5 then MRR)")
    print(f"{'='*90}")
    print(f"  {'#':>3}  {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>7}  Config")
    print(f"  {'-'*85}")
    for i, r in enumerate(results[:10], 1):
        cfg_str = " ".join(f"{k}={v}" for k, v in r["cfg"].items())
        print(f"  {i:>3}  {r['hit_at_1']:>6.3f} {r['hit_at_5']:>6.3f} {r['hit_at_10']:>7.3f} "
              f"{r['mrr']:>7.4f}  {cfg_str}")

    best = results[0]
    print(f"\n  BEST CONFIG:")
    for k, v in best["cfg"].items():
        print(f"    {k} = {v}")
    print(f"\n  Metrics: Hit@1={best['hit_at_1']:.3f} Hit@5={best['hit_at_5']:.3f} "
          f"Hit@10={best['hit_at_10']:.3f} MRR={best['mrr']:.4f}")
    print()
    print("  To lock these into config.py, update the defaults in Settings and")
    print("  set HYBRIDMIND_DEFAULT_VECTOR_WEIGHT etc. in your .env file.")


def main():
    args = parse_args()

    if not LOCOMO_PATH.exists():
        print(f"ERROR: {LOCOMO_PATH} not found")
        sys.exit(1)

    questions = load_questions(samples_per_category=args.n, category_filter=args.category)
    if not questions:
        print("ERROR: No questions loaded from LoCoMo data")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions:")
    cats: dict = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print()

    client = httpx.Client(base_url=BASE_URL, timeout=300.0)
    try:
        if args.sweep:
            sweep(questions, client)
        else:
            summary = run_eval(
                questions, client,
                vector_weight=args.vector_weight,
                graph_weight=args.graph_weight,
                bm25_boost=args.bm25_boost,
                overlap_threshold=args.overlap_threshold,
                rerank_pool=args.rerank_pool,
                top_k=args.top_k,
                verbose=True,
                decompose_multihop=args.decompose_multihop,
                with_answers=args.with_answers,
                answer_model=args.answer_model,
                search_mode=args.search_mode,
                route_weights=args.route_weights,
                track_access=args.track_access,
                graph_anchor_strategy=args.graph_anchor_strategy,
                anchor_node_ids=args.anchor_node_id,
            )
            if summary:
                print_summary(summary, label="RETRIEVAL EVALUATION RESULTS")
    finally:
        client.close()


if __name__ == "__main__":
    main()
