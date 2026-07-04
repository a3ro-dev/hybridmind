"""
Retrieval-only LongMemEval evaluation for HybridMind.

Calls /search/hybrid for each question and measures Hit@k / MRR using
answer-text overlap as the relevance signal — matching the LoCoMo eval pattern.

The LongMemEval benchmark tests long-context episodic memory across multiple
conversation sessions (up to 500 sessions per question).

Data:
  Download LongMemEval from: https://github.com/tiger-ai-lab/LongMemEval
  Expected path: memorybench/data/benchmarks/longmemeval/longmemeval_s.json
  (The "_s" split is the single-session variant; "_m" is multi-session.)

Usage:
  python eval_longmemeval_retrieval.py [--split s|m] [--n 20] [--sweep]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

import httpx

import eval_common
import eval_ledger
from engine.query_router import route_query

BASE_URL = "http://127.0.0.1:8000"
DATA_DIR = Path("memorybench/data/benchmarks/longmemeval")

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "adversarial-irrelevant",
]


def parse_args():
    p = argparse.ArgumentParser(description="LongMemEval retrieval eval")
    p.add_argument("--split",             choices=["s", "m"], default="s",
                   help="Dataset split: s=single-session, m=multi-session (default: s)")
    p.add_argument("--vector-weight",     type=float, default=0.5)
    p.add_argument("--graph-weight",      type=float, default=0.15)
    p.add_argument("--bm25-boost",        type=float, default=0.35)
    p.add_argument("--overlap-threshold", type=float, default=0.15)
    p.add_argument("--rerank-pool",       type=int,   default=25)
    p.add_argument("--top-k",            type=int,   default=15)
    p.add_argument("--n",                type=int,   default=20,
                   help="Max questions to evaluate (default 20)")
    p.add_argument("--question-type",    type=str,   default=None,
                   help="Filter to one question type")
    p.add_argument("--sweep",            action="store_true",
                   help="Grid-search key params")
    p.add_argument("--base-url",         default=BASE_URL)
    p.add_argument("--with-answers",     action="store_true",
                   help="Also run LLM QA-answering + accuracy scoring on top of retrieval")
    p.add_argument("--answer-model",     type=str,   default=None)
    return p.parse_args()


def load_questions(split: str = "s", n: int = 20, question_type_filter: str | None = None):
    path = DATA_DIR / f"longmemeval_{split}.json"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Download LongMemEval from: https://github.com/tiger-ai-lab/LongMemEval")
        print(f"Place longmemeval_{split}.json in {DATA_DIR}/")
        sys.exit(1)

    data = json.loads(path.read_text())
    questions = []
    for item in data:
        qt = item.get("question_type", "unknown")
        if question_type_filter and qt != question_type_filter:
            continue
        # Best-effort gold evidence ids: LongMemEval doesn't guarantee a
        # dedicated field, so fall back across the names seen in released
        # variants of the dataset before giving up with an empty list.
        evidence_ids = (
            item.get("evidence")
            or item.get("answer_session_ids")
            or item.get("haystack_session_ids")
            or []
        )
        questions.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "answer": str(item.get("answer", "")).strip(),
            "question_type": qt,
            "question_date": item.get("question_date", ""),
            "evidence_ids": [str(e) for e in evidence_ids],
        })
        if len(questions) >= n:
            break
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
    base_url: str = BASE_URL,
    with_answers: bool = False,
    answer_model: str | None = None,
):
    hit1 = hit5 = hit10 = p5 = p10 = mrr = 0.0
    correct_sum = 0.0
    by_type: dict = {}

    ledger_config = {
        "benchmark": "longmemeval",
        "vector_weight": vector_weight,
        "graph_weight": graph_weight,
        "bm25_boost": bm25_boost,
        "overlap_threshold": overlap_threshold,
        "rerank_pool": rerank_pool,
        "top_k": top_k,
        "with_answers": with_answers,
        "answer_model": answer_model or eval_common.DEFAULT_ANSWER_MODEL,
    }
    ledger = eval_ledger.LedgerWriter("longmemeval", ledger_config)
    pool_top_k = max(top_k, rerank_pool, max(eval_ledger.DEFAULT_K_LIST))

    for q in questions:
        try:
            resp = client.post(
                f"{base_url}/search/hybrid",
                json={
                    "query_text": q["question"],
                    "top_k": pool_top_k,
                    "min_score": 0.0,
                    "vector_weight": vector_weight,
                    "graph_weight": graph_weight,
                    "bm25_boost_weight": bm25_boost,
                    "overlap_threshold": overlap_threshold,
                    "rerank_pool": rerank_pool,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as e:
            print(f"  ERROR {q['question_id']}: {e}")
            results = []

        relevance = [is_relevant(r.get("text", ""), q["answer"]) for r in results]

        # Hit@k
        hit1 += any(relevance[:1])
        hit5 += any(relevance[:5])
        hit10 += any(relevance[:10])

        # Precision@k
        p5 += sum(relevance[:5]) / 5
        p10 += sum(relevance[:10]) / 10

        # MRR
        for rank, rel in enumerate(relevance[:10], 1):
            if rel:
                mrr += 1.0 / rank
                break

        qtype = route_query(q["question"])["type"]
        hypothesis, judged_correct, judge_rationale = "", False, "not evaluated (retrieval-only run, pass --with-answers)"
        prompt_version = ""
        if with_answers:
            snippets = [r.get("text", "") for r in results[:10] if r.get("text")]
            hypothesis, prompt_version = eval_common.answer_question(
                q["question"], snippets, question_type=qtype, question_date=q.get("question_date", ""), model=answer_model
            )
            judged_correct, judge_rationale = eval_common.judge_correct_normalized(hypothesis, q["answer"])
            correct_sum += 1.0 if judged_correct else 0.0

        pool_metrics = eval_ledger.compute_pool_metrics(
            results, lambda r: is_relevant(r.get("text", ""), q["answer"])
        )
        ledger.write(
            question_id=q["question_id"],
            question_type=qtype,
            gold_evidence_ids=q.get("evidence_ids", []),
            pool_metrics=pool_metrics,
            raw_llm_answer=hypothesis,
            judged_correct=judged_correct,
            judge_rationale=judge_rationale,
            prompt_version=prompt_version,
        )

        # By question type
        qt = q.get("question_type", "unknown")
        by_type.setdefault(qt, {"n": 0, "hit5": 0, "mrr": 0})
        by_type[qt]["n"] += 1
        by_type[qt]["hit5"] += any(relevance[:5])
        by_type[qt]["mrr"] += sum(1.0 / r for r, rel in enumerate(relevance[:10], 1) if rel)

    n = len(questions) or 1
    result = {
        "hit_at_1": round(hit1 / n, 3),
        "hit_at_5": round(hit5 / n, 3),
        "hit_at_10": round(hit10 / n, 3),
        "precision_at_5": round(p5 / n, 3),
        "precision_at_10": round(p10 / n, 3),
        "mrr": round(mrr / n, 3),
        "n": n,
        "by_type": {
            qt: {
                "hit5": round(v["hit5"] / v["n"], 3) if v["n"] else 0,
                "mrr": round(v["mrr"] / v["n"], 3) if v["n"] else 0,
                "n": v["n"],
            }
            for qt, v in by_type.items()
        },
    }
    if with_answers:
        result["accuracy"] = round(correct_sum / n, 3)
    return result


def print_results(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"  N        : {metrics['n']}")
    print(f"  Hit@1    : {metrics['hit_at_1']:.1%}")
    print(f"  Hit@5    : {metrics['hit_at_5']:.1%}")
    print(f"  Hit@10   : {metrics['hit_at_10']:.1%}")
    print(f"  P@5      : {metrics['precision_at_5']:.1%}")
    print(f"  P@10     : {metrics['precision_at_10']:.1%}")
    print(f"  MRR      : {metrics['mrr']:.3f}")
    if "accuracy" in metrics:
        print(f"  Accuracy (LLM QA): {metrics['accuracy']:.1%}")
    if metrics.get("by_type"):
        print("  By type:")
        for qt, v in sorted(metrics["by_type"].items()):
            print(f"    {qt:<40} Hit@5={v['hit5']:.0%}  MRR={v['mrr']:.3f}  (n={v['n']})")


def main():
    args = parse_args()
    questions = load_questions(
        split=args.split,
        n=args.n,
        question_type_filter=args.question_type,
    )
    print(f"LongMemEval retrieval eval — {len(questions)} questions (split={args.split})")

    with httpx.Client(timeout=60) as client:
        # Check server is up
        try:
            client.get(f"{args.base_url}/live").raise_for_status()
        except Exception as e:
            print(f"ERROR: HybridMind not reachable at {args.base_url}: {e}")
            sys.exit(1)

        if args.sweep:
            configs = []
            for vw in [0.4, 0.5, 0.6]:
                for gw in [0.0, 0.1, 0.2]:
                    for rp in [15, 25]:
                        configs.append((vw, gw, args.bm25_boost, rp))
            results = []
            for vw, gw, bb, rp in configs:
                m = run_eval(questions, client, vector_weight=vw, graph_weight=gw,
                             bm25_boost=bb, rerank_pool=rp, top_k=args.top_k, base_url=args.base_url)
                results.append((vw, gw, bb, rp, m))
                print(f"  vw={vw} gw={gw} bb={bb} rp={rp} → Hit@5={m['hit_at_5']:.1%} MRR={m['mrr']:.3f}")
            results.sort(key=lambda x: (-x[4]["hit_at_5"], -x[4]["mrr"]))
            print("\nTop 3 configs:")
            for vw, gw, bb, rp, m in results[:3]:
                print(f"  vw={vw} gw={gw} bb={bb} rp={rp} → Hit@5={m['hit_at_5']:.1%} MRR={m['mrr']:.3f}")
        else:
            metrics = run_eval(
                questions, client,
                vector_weight=args.vector_weight,
                graph_weight=args.graph_weight,
                bm25_boost=args.bm25_boost,
                overlap_threshold=args.overlap_threshold,
                rerank_pool=args.rerank_pool,
                top_k=args.top_k,
                base_url=args.base_url,
                with_answers=args.with_answers,
                answer_model=args.answer_model,
            )
            print_results(metrics, label="LongMemEval Retrieval Results")
            # Save
            out = Path("benchmarks/results/longmemeval_retrieval.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, indent=2))
            print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
