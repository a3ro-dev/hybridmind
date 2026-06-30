"""
Retrieval-only MuSiQue multi-hop evaluation for HybridMind.

MuSiQue (Multi-hop Questions via Single-hop Question Chains) tests 2-4 hop
reasoning over paragraphs. Relevance is keyed on supporting paragraph IDs
rather than answer string overlap.

Data:
  Download from: https://huggingface.co/datasets/allenai/musique
  or: https://github.com/StonyBrookNLP/musique
  Expected path: benchmarks/data/musique/musique_ans_v1.0_dev.jsonl

Usage:
  python eval_musique_retrieval.py [--n 100] [--n-hops 2|3|4] [--sweep]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
DATA_DIR = Path("benchmarks/data/musique")


def parse_args():
    p = argparse.ArgumentParser(description="MuSiQue multi-hop retrieval eval")
    p.add_argument("--data-file",         default="musique_ans_v1.0_dev.jsonl")
    p.add_argument("--n",                 type=int, default=100,
                   help="Number of questions to evaluate (default 100)")
    p.add_argument("--n-hops",            type=int, default=None,
                   help="Filter to questions with this many hops (2/3/4)")
    p.add_argument("--vector-weight",     type=float, default=0.5)
    p.add_argument("--graph-weight",      type=float, default=0.2,
                   help="Graph weight (higher helps multi-hop; default 0.2)")
    p.add_argument("--bm25-boost",        type=float, default=0.35)
    p.add_argument("--overlap-threshold", type=float, default=0.10)
    p.add_argument("--rerank-pool",       type=int,   default=40)
    p.add_argument("--top-k",            type=int,   default=20)
    p.add_argument("--sweep",            action="store_true")
    p.add_argument("--base-url",         default=BASE_URL)
    return p.parse_args()


def load_questions(data_file: str, n: int = 100, n_hops_filter: int | None = None):
    path = DATA_DIR / data_file
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Download MuSiQue from: https://huggingface.co/datasets/allenai/musique")
        print(f"Place {data_file} in {DATA_DIR}/")
        sys.exit(1)

    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("answerable", True):
                continue

            # n_hops can be inferred from decomposition or id prefix
            n_hops = item.get("n_hops", None)
            if n_hops is None:
                qid = item.get("id", "")
                if "2hop" in qid:
                    n_hops = 2
                elif "3hop" in qid:
                    n_hops = 3
                elif "4hop" in qid:
                    n_hops = 4

            if n_hops_filter is not None and n_hops != n_hops_filter:
                continue

            # Supporting paragraph IDs (used as ground truth for retrieval)
            supporting_ids = set()
            for sp in item.get("supports_facts", []):
                if isinstance(sp, dict):
                    pid = sp.get("paragraph_id") or sp.get("id")
                    if pid:
                        supporting_ids.add(str(pid))

            # Paragraphs that must be retrieved
            paragraphs = item.get("paragraphs", [])

            questions.append({
                "question_id": item.get("id", ""),
                "question": item["question"],
                "answer": str(item.get("answer", "")).strip(),
                "n_hops": n_hops,
                "supporting_ids": supporting_ids,
                "paragraphs": paragraphs,
            })
            if len(questions) >= n:
                break

    return questions


def is_relevant_by_id(retrieved_metadata: dict, supporting_ids: set) -> bool:
    """Check if a retrieved node corresponds to a supporting paragraph."""
    pid = (retrieved_metadata.get("paragraph_id")
           or retrieved_metadata.get("musique_id")
           or retrieved_metadata.get("source_id", ""))
    return str(pid) in supporting_ids


def is_relevant_by_text(retrieved_text: str, answer: str) -> bool:
    """Fallback: token-overlap between retrieved text and gold answer."""
    if not answer:
        return False
    text_lower = retrieved_text.lower()
    ans_lower = answer.lower()
    if ans_lower in text_lower:
        return True
    ans_toks = set(re.findall(r"[A-Za-z0-9']+", ans_lower)) - \
               {"the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was"}
    if not ans_toks:
        return False
    text_toks = set(re.findall(r"[A-Za-z0-9']+", text_lower))
    overlap = len(ans_toks & text_toks)
    return overlap / len(ans_toks) >= 0.7


def run_eval(
    questions: list,
    client: httpx.Client,
    *,
    vector_weight: float = 0.5,
    graph_weight: float = 0.2,
    bm25_boost: float = 0.35,
    overlap_threshold: float = 0.10,
    rerank_pool: int = 40,
    top_k: int = 20,
    base_url: str = BASE_URL,
):
    hit1 = hit5 = hit10 = mrr = 0.0
    # Multi-hop specific: did we retrieve ALL supporting paragraphs?
    recall_full = 0.0
    by_hops: dict = {}

    for q in questions:
        supporting_ids = q["supporting_ids"]

        try:
            resp = client.post(
                f"{base_url}/search/hybrid",
                json={
                    "query_text": q["question"],
                    "top_k": top_k,
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

        # Check relevance: prefer ID match, fall back to answer overlap
        relevance = []
        retrieved_ids = set()
        for r in results:
            meta = r.get("metadata", {})
            if supporting_ids and is_relevant_by_id(meta, supporting_ids):
                relevance.append(True)
                pid = meta.get("paragraph_id") or meta.get("musique_id") or meta.get("source_id", "")
                retrieved_ids.add(str(pid))
            elif is_relevant_by_text(r.get("text", ""), q["answer"]):
                relevance.append(True)
            else:
                relevance.append(False)

        hit1 += any(relevance[:1])
        hit5 += any(relevance[:5])
        hit10 += any(relevance[:10])
        for rank, rel in enumerate(relevance[:10], 1):
            if rel:
                mrr += 1.0 / rank
                break

        # Multi-hop: did we surface all required paragraphs?
        if supporting_ids:
            recall_full += int(supporting_ids.issubset(retrieved_ids))

        hk = q.get("n_hops") or "?"
        by_hops.setdefault(hk, {"n": 0, "hit5": 0, "mrr": 0})
        by_hops[hk]["n"] += 1
        by_hops[hk]["hit5"] += any(relevance[:5])
        by_hops[hk]["mrr"] += sum(1.0 / r for r, rel in enumerate(relevance[:10], 1) if rel)

    n = len(questions) or 1
    return {
        "hit_at_1": round(hit1 / n, 3),
        "hit_at_5": round(hit5 / n, 3),
        "hit_at_10": round(hit10 / n, 3),
        "mrr": round(mrr / n, 3),
        "full_recall": round(recall_full / n, 3),
        "n": n,
        "by_hops": {
            str(hk): {
                "hit5": round(v["hit5"] / v["n"], 3) if v["n"] else 0,
                "mrr": round(v["mrr"] / v["n"], 3) if v["n"] else 0,
                "n": v["n"],
            }
            for hk, v in by_hops.items()
        },
    }


def print_results(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"  N            : {metrics['n']}")
    print(f"  Hit@1        : {metrics['hit_at_1']:.1%}")
    print(f"  Hit@5        : {metrics['hit_at_5']:.1%}")
    print(f"  Hit@10       : {metrics['hit_at_10']:.1%}")
    print(f"  MRR          : {metrics['mrr']:.3f}")
    print(f"  Full Recall  : {metrics['full_recall']:.1%}  (all supporting paragraphs found)")
    if metrics.get("by_hops"):
        print("  By hop count:")
        for hk in sorted(metrics["by_hops"], key=lambda x: str(x)):
            v = metrics["by_hops"][str(hk)]
            print(f"    {hk}-hop  Hit@5={v['hit5']:.0%}  MRR={v['mrr']:.3f}  (n={v['n']})")


def main():
    args = parse_args()
    questions = load_questions(args.data_file, args.n, args.n_hops)
    print(f"MuSiQue multi-hop retrieval eval — {len(questions)} questions")

    with httpx.Client(timeout=60) as client:
        try:
            client.get(f"{args.base_url}/live").raise_for_status()
        except Exception as e:
            print(f"ERROR: HybridMind not reachable at {args.base_url}: {e}")
            sys.exit(1)

        if args.sweep:
            configs = [
                (0.5, 0.2, 0.35, 40),
                (0.4, 0.3, 0.35, 40),
                (0.6, 0.1, 0.35, 25),
                (0.3, 0.4, 0.25, 40),
            ]
            for vw, gw, bb, rp in configs:
                m = run_eval(questions, client, vector_weight=vw, graph_weight=gw,
                             bm25_boost=bb, rerank_pool=rp, top_k=args.top_k, base_url=args.base_url)
                print(f"  vw={vw} gw={gw} bb={bb} rp={rp} → Hit@5={m['hit_at_5']:.1%} MRR={m['mrr']:.3f} FullRecall={m['full_recall']:.1%}")
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
            )
            print_results(metrics, label="MuSiQue Multi-Hop Retrieval Results")
            out = Path("benchmarks/results/musique_retrieval.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, indent=2))
            print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
