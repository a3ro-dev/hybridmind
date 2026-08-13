"""
Retrieval-only LongMemEval evaluation for HybridMind.

Calls /search/hybrid for each question and measures primary Hit@k / MRR using
gold evidence/session IDs. Answer-text overlap is a separately labelled proxy.

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
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

import httpx

import eval_common
import eval_ledger
from config import settings
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


class EvaluationRunError(RuntimeError):
    pass


def parse_args():
    p = argparse.ArgumentParser(description="LongMemEval retrieval eval")
    p.add_argument("--split",             choices=["s", "m"], default="s",
                   help="Dataset split: s=single-session, m=multi-session (default: s)")
    p.add_argument("--vector-weight",     type=float, default=0.5)
    p.add_argument("--graph-weight",      type=float, default=0.15)
    p.add_argument("--bm25-boost",        type=float, default=0.35)
    p.add_argument("--overlap-threshold", type=float, default=0.15)
    p.add_argument(
        "--rerank-pool", type=int, default=25,
        help="Hard cross-encoder candidate cap; 0 disables, positive must be >= top-k",
    )
    p.add_argument("--top-k",            type=int,   default=15)
    p.add_argument(
        "--fusion-mode", choices=["rrf", "linear", "mlp"], default=None,
        help="Override the server fusion mode; controlled ablations pin rrf",
    )
    p.add_argument(
        "--search-mode",
        choices=["hybrid", "vector_only", "sparse_only", "vector_sparse", "graph_only"],
        default="hybrid",
    )
    p.add_argument("--route-weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--track-access", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--graph-anchor-strategy", choices=["explicit", "vector_top1"], default="explicit"
    )
    p.add_argument("--anchor-node-id", action="append", default=[])
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
    p.add_argument("--decompose-multihop", dest="decompose_multihop", action="store_true", default=False,
                   help="Opt in to paid/provider-backed multihop decomposition")
    p.add_argument("--no-decompose-multihop", dest="decompose_multihop", action="store_false")
    eval_common.add_budget_arguments(p)
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
        # Never treat the full haystack as gold. Released variants use either
        # explicit evidence or answer_session_ids for supporting sessions.
        evidence_ids = (
            item.get("evidence")
            or item.get("answer_session_ids")
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
    """Diagnostic answer-overlap proxy, not gold-evidence relevance."""
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


def candidate_evidence_ids(result: dict) -> set[str]:
    metadata = result.get("metadata") or {}
    found: set[str] = set()
    for key in ("evidence_id", "session_id", "sessionId", "source_id"):
        value = metadata.get(key)
        values = value if isinstance(value, list) else [value]
        found.update(str(item) for item in values if item is not None and str(item).strip())
    return found


def is_exact_evidence(result: dict, evidence_ids: set[str]) -> bool:
    return bool(evidence_ids & candidate_evidence_ids(result))


def _reranker_executed(results: list[dict]) -> bool:
    return bool(results) and all(
        result.get("rerank_attempted") is True
        and result.get("rerank_applied") is True
        and result.get("rerank_failure_type") is None
        for result in results
    )


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
    split: str = "s",
    with_answers: bool = False,
    answer_model: str | None = None,
    decompose_multihop: bool = False,
    search_mode: str = "hybrid",
    route_weights: bool = True,
    track_access: bool = False,
    graph_anchor_strategy: str = "explicit",
    anchor_node_ids: list[str] | None = None,
    fusion_mode: str | None = None,
):
    if top_k < 10:
        raise ValueError("LongMemEval evaluation requires top_k >= 10")
    eval_common.validate_rerank_pool(top_k=top_k, rerank_pool=rerank_pool)
    exact_hit1 = exact_hit5 = exact_hit10 = exact_mrr = 0.0
    exact_recall10 = exact_all_hit10 = 0.0
    overlap_hit1 = overlap_hit5 = overlap_hit10 = overlap_mrr = 0.0
    exact_n = 0
    correct_sum = 0.0
    answer_n = 0
    retrieved_candidate_count = 0
    evidence_tagged_candidate_count = 0
    by_type: dict = {}

    ledger_config = {
        "benchmark": "longmemeval",
        "vector_weight": vector_weight,
        "graph_weight": graph_weight,
        "bm25_boost": bm25_boost,
        "overlap_threshold": overlap_threshold,
        "rerank_pool": rerank_pool,
        "fusion_mode": fusion_mode,
        "top_k": top_k,
        "with_answers": with_answers,
        "answer_model": answer_model or eval_common.DEFAULT_ANSWER_MODEL,
        "decompose_multihop": decompose_multihop,
        "api_base_url": base_url,
        "split": split,
        "reranker_expected": search_mode == "hybrid" and rerank_pool > 0,
        "metric_primary": "exact_evidence_id",
        "answer_overlap_role": "diagnostic_proxy_only",
        "search_mode": search_mode,
        "route_weights": route_weights,
        "track_access": track_access,
        "graph_anchor_strategy": graph_anchor_strategy,
        "anchor_node_ids": list(anchor_node_ids or []),
        "budget": eval_common.active_budget_provenance(),
    }
    dataset_path = DATA_DIR / f"longmemeval_{split}.json"
    # Callers using an in-memory fixture may not have the default split path.
    provenance = {"api_base_url": base_url}
    if dataset_path.is_file():
        provenance["dataset"] = eval_ledger.dataset_provenance(dataset_path)
    if os.getenv("HYBRIDMIND_ABLATION_CONFIG_HASH") or os.getenv("HYBRIDMIND_ABLATION_MODE"):
        ledger_config["ablation"] = {
            "plan_hash": os.getenv("HYBRIDMIND_ABLATION_CONFIG_HASH", ""),
            "mode": os.getenv("HYBRIDMIND_ABLATION_MODE", ""),
            "resolved_settings_sha256": os.getenv(
                "HYBRIDMIND_ABLATION_SETTINGS_SHA256", ""
            ),
        }
    ledger = eval_ledger.LedgerWriter("longmemeval", ledger_config, provenance=provenance)
    metric_k = tuple(k for k in eval_ledger.DEFAULT_K_LIST if k <= top_k)

    for q in questions:
        qtype = route_query(q["question"])["type"]

        def _post(q_text: str) -> list:
            anchors = list(anchor_node_ids or [])
            if search_mode == "graph_only" and not anchors and graph_anchor_strategy == "vector_top1":
                eval_common.record_retrieval_query()
                seed_payload = {
                    "query_text": q_text, "top_k": 1, "rerank_pool": 0,
                    "vector_weight": 1.0, "graph_weight": 0.0,
                    "bm25_boost_weight": 0.0, "search_mode": "vector_only",
                    "route_weights": False, "track_access": False,
                }
                if fusion_mode is not None:
                    seed_payload["fusion_mode"] = fusion_mode
                seed = client.post(
                    f"{base_url}/search/hybrid",
                    json=seed_payload,
                    timeout=eval_common.live_request_timeout(30.0),
                )
                eval_common.record_retrieval_response()
                seed.raise_for_status()
                seed_results = seed.json().get("results", [])
                if seed_results:
                    anchors = [seed_results[0]["node_id"]]
            if search_mode == "graph_only" and not anchors:
                raise ValueError("graph_only requires explicit anchors or vector_top1")
            payload = {
                "query_text": q_text,
                "top_k": top_k,
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
            if fusion_mode is not None:
                payload["fusion_mode"] = fusion_mode
            if anchors:
                payload["anchor_nodes"] = anchors
            eval_common.record_retrieval_query()
            resp = client.post(
                f"{base_url}/search/hybrid",
                json=payload,
                timeout=eval_common.live_request_timeout(30.0),
            )
            eval_common.record_retrieval_response()
            resp.raise_for_status()
            return resp.json().get("results", [])

        try:
            results = eval_common.retrieve_with_decomposition(
                q["question"], qtype, _post, decompose_enabled=decompose_multihop
            )
        except Exception as e:
            print(f"  ERROR {q['question_id']}: {eval_common.sanitized_error(e)}")
            ledger.write(
                question_id=q["question_id"], question_type=qtype,
                gold_evidence_ids=q.get("evidence_ids", []),
                pool_metrics=eval_ledger.empty_pool_metrics(metric_k),
                status="retrieval_error", answer_status="not_run",
                error_type=type(e).__name__, error_message=eval_common.sanitized_error(e),
            )
            ledger.finalize_failure(
                reason="retrieval_error",
                error_type=type(e).__name__,
                expected_questions=len(questions),
            )
            raise EvaluationRunError(
                f"LongMemEval retrieval failed for {q['question_id']}; no score is valid"
            ) from e

        if search_mode == "hybrid" and rerank_pool > 0 and results and not _reranker_executed(results):
            error = "reranker was required but API results contain no rerank_score"
            ledger.write(
                question_id=q["question_id"], question_type=qtype,
                gold_evidence_ids=q.get("evidence_ids", []),
                pool_metrics=eval_ledger.empty_pool_metrics(metric_k),
                status="reranker_not_executed", answer_status="not_run",
                error_type="RerankerExecutionError", error_message=error,
            )
            ledger.finalize_failure(
                reason="reranker_not_executed",
                error_type="RerankerExecutionError",
                expected_questions=len(questions),
            )
            raise EvaluationRunError(error)

        evidence_ids = set(q.get("evidence_ids", []))
        retrieved_candidate_count += len(results)
        evidence_tagged_candidate_count += sum(bool(candidate_evidence_ids(r)) for r in results)
        relevance = [is_exact_evidence(r, evidence_ids) for r in results]
        overlap_relevance = [is_relevant(r.get("text", ""), q["answer"]) for r in results]

        if evidence_ids:
            exact_n += 1
            exact_hit1 += any(relevance[:1])
            exact_hit5 += any(relevance[:5])
            exact_hit10 += any(relevance[:10])
            exact_mrr += next((1.0 / rank for rank, rel in enumerate(relevance[:10], 1) if rel), 0.0)
            retrieved_evidence_at_10 = set().union(
                *(candidate_evidence_ids(result) for result in results[:10])
            ) if results[:10] else set()
            covered_evidence = evidence_ids & retrieved_evidence_at_10
            exact_recall10 += len(covered_evidence) / len(evidence_ids)
            exact_all_hit10 += float(covered_evidence == evidence_ids)
        overlap_hit1 += any(overlap_relevance[:1])
        overlap_hit5 += any(overlap_relevance[:5])
        overlap_hit10 += any(overlap_relevance[:10])
        overlap_mrr += next((1.0 / rank for rank, rel in enumerate(overlap_relevance[:10], 1) if rel), 0.0)

        hypothesis, judged_correct = "", None
        judge_rationale = "not evaluated (retrieval-only run, pass --with-answers)"
        answer_status, judge_method = "not_requested", "not_run"
        prompt_version = ""
        if with_answers:
            snippets = [r.get("text", "") for r in results[:10] if r.get("text")]
            answer_result = eval_common.answer_question_with_status(
                q["question"], snippets, question_type=qtype, question_date=q.get("question_date", ""), model=answer_model
            )
            hypothesis, prompt_version, answer_status = (
                answer_result.answer, answer_result.prompt_version, answer_result.status
            )
            if answer_status in {"provider_unavailable", "provider_error"}:
                ledger.write(
                    question_id=q["question_id"], question_type=qtype,
                    gold_evidence_ids=sorted(evidence_ids),
                    pool_metrics=eval_ledger.compute_pool_metrics(
                        results, lambda r: is_exact_evidence(r, evidence_ids), metric_k
                    ),
                    status="answer_error", answer_status=answer_status,
                    error_type="AnswerProviderError", error_message=answer_result.error or answer_status,
                    prompt_version=prompt_version,
                )
                ledger.finalize_failure(
                    reason="answer_provider_error",
                    error_type="AnswerProviderError",
                    expected_questions=len(questions),
                )
                raise EvaluationRunError(
                    f"answer provider failed for {q['question_id']}: {answer_status}"
                )
            judged_correct, judge_rationale = eval_common.judge_correct_normalized(hypothesis, q["answer"])
            judge_method = "deterministic_normalized_answer_overlap_v1"
            correct_sum += 1.0 if judged_correct else 0.0
            answer_n += 1

        exact_metrics = eval_ledger.compute_pool_metrics(
            results, lambda r: is_exact_evidence(r, evidence_ids), metric_k,
            metric_basis="exact_evidence_id",
        )
        overlap_metrics = eval_ledger.compute_pool_metrics(
            results, lambda r: is_relevant(r.get("text", ""), q["answer"]), metric_k,
            metric_basis="answer_text_overlap_proxy",
        )
        ledger.write(
            question_id=q["question_id"],
            question_type=qtype,
            gold_evidence_ids=q.get("evidence_ids", []),
            pool_metrics=exact_metrics,
            answer_overlap_metrics=overlap_metrics,
            raw_llm_answer=hypothesis,
            judged_correct=judged_correct,
            judge_rationale=judge_rationale,
            prompt_version=prompt_version,
            answer_status=answer_status,
            judge_method=judge_method,
            extra={
                "exact_evidence_available": bool(evidence_ids),
                "reranker_executed": _reranker_executed(results),
                "retrieved_evidence_ids_at_10": sorted(
                    set().union(*(candidate_evidence_ids(r) for r in results[:10]))
                    if results[:10] else set()
                ),
                "gold_evidence_recall_at_10": (
                    len(
                        evidence_ids
                        & set().union(*(candidate_evidence_ids(r) for r in results[:10]))
                    ) / len(evidence_ids)
                    if evidence_ids and results[:10] else 0.0 if evidence_ids else None
                ),
            },
        )

        # By question type
        qt = q.get("question_type", "unknown")
        by_type.setdefault(qt, {"n": 0, "exact_n": 0, "hit5": 0, "mrr": 0, "overlap_hit5": 0})
        by_type[qt]["n"] += 1
        if evidence_ids:
            by_type[qt]["exact_n"] += 1
            by_type[qt]["hit5"] += any(relevance[:5])
            by_type[qt]["mrr"] += next((1.0 / r for r, rel in enumerate(relevance[:10], 1) if rel), 0.0)
        by_type[qt]["overlap_hit5"] += any(overlap_relevance[:5])

    n = len(questions)
    if not n:
        ledger.finalize_failure(
            reason="zero_questions", error_type="EvaluationRunError", expected_questions=0
        )
        raise EvaluationRunError("LongMemEval run contained zero questions")
    if not exact_n:
        ledger.finalize_failure(
            reason="no_exact_evidence",
            error_type="EvaluationRunError",
            expected_questions=n,
        )
        raise EvaluationRunError("LongMemEval run contained no usable gold evidence IDs")
    if not retrieved_candidate_count or not evidence_tagged_candidate_count:
        ledger.finalize_failure(
            reason="unverified_corpus_metadata",
            error_type="EvaluationRunError",
            expected_questions=n,
        )
        raise EvaluationRunError(
            "LongMemEval corpus evidence/session metadata cannot be verified from retrieved candidates"
        )
    result = {
        "metric_basis": "exact_evidence_id",
        "hit_at_1": round(exact_hit1 / exact_n, 3),
        "hit_at_5": round(exact_hit5 / exact_n, 3),
        "hit_at_10": round(exact_hit10 / exact_n, 3),
        "mrr": round(exact_mrr / exact_n, 3),
        "gold_evidence_recall_at_10": round(exact_recall10 / exact_n, 3),
        "all_gold_evidence_hit_at_10": round(exact_all_hit10 / exact_n, 3),
        "n": n,
        "n_exact_evidence": exact_n,
        "answer_overlap_proxy": {
            "hit_at_1": round(overlap_hit1 / n, 3),
            "hit_at_5": round(overlap_hit5 / n, 3),
            "hit_at_10": round(overlap_hit10 / n, 3),
            "mrr": round(overlap_mrr / n, 3),
        },
        "ledger_path": str(ledger.path),
        "manifest_path": str(ledger.manifest_path),
        "completion_path": str(ledger.completion_path),
        "budget": eval_common.active_budget_provenance(),
        "by_type": {
            qt: {
                "hit5": round(v["hit5"] / v["exact_n"], 3) if v["exact_n"] else None,
                "mrr": round(v["mrr"] / v["exact_n"], 3) if v["exact_n"] else None,
                "answer_overlap_hit5_proxy": round(v["overlap_hit5"] / v["n"], 3) if v["n"] else 0,
                "n": v["n"],
                "n_exact_evidence": v["exact_n"],
            }
            for qt, v in by_type.items()
        },
    }
    if with_answers:
        result["accuracy"] = round(correct_sum / answer_n, 3) if answer_n else None
    ledger.finalize(status="completed", summary=result)
    return result


def print_results(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"  N        : {metrics['n']} ({metrics.get('n_exact_evidence', 0)} with exact evidence)")
    print(f"  Basis    : {metrics.get('metric_basis', 'unknown')}")
    print(f"  Hit@1    : {metrics['hit_at_1']:.1%}")
    print(f"  Hit@5    : {metrics['hit_at_5']:.1%}")
    print(f"  Hit@10   : {metrics['hit_at_10']:.1%}")
    print(f"  MRR      : {metrics['mrr']:.3f}")
    proxy = metrics.get("answer_overlap_proxy", {})
    if proxy:
        print(f"  Answer-overlap diagnostic only: Hit@5={proxy['hit_at_5']:.1%} MRR={proxy['mrr']:.3f}")
    if "accuracy" in metrics:
        print(f"  Accuracy (LLM QA): {metrics['accuracy']:.1%}")
    if metrics.get("by_type"):
        print("  By type:")
        for qt, v in sorted(metrics["by_type"].items()):
            exact = (
                f"Hit@5={v['hit5']:.0%} MRR={v['mrr']:.3f}"
                if v["hit5"] is not None else "exact evidence unavailable"
            )
            print(f"    {qt:<40} {exact}  (n={v['n']}, exact={v['n_exact_evidence']})")


def main():
    args = parse_args()
    eval_common.validate_rerank_pool(top_k=args.top_k, rerank_pool=args.rerank_pool)
    questions = load_questions(
        split=args.split,
        n=args.n,
        question_type_filter=args.question_type,
    )
    print(f"LongMemEval retrieval eval — {len(questions)} questions (split={args.split})")

    budget = eval_common.budget_from_args(args)
    if not args.execute:
        print("DRY RUN: no API/provider calls made. Re-run with --execute.")
        print(json.dumps({"questions": len(questions), "budget_ceilings": budget.ceilings()}, indent=2))
        return
    eval_common.enforce_priced_llm_budget(
        args,
        answer_requested=args.with_answers,
        decomposition_requested=args.decompose_multihop,
    )

    with budget.activate(), httpx.Client(
        timeout=300.0, headers=eval_common.api_headers()
    ) as client:
        # Check server is up
        try:
            client.get(
                f"{args.base_url}/live",
                timeout=eval_common.live_request_timeout(30.0),
            ).raise_for_status()
            eval_common.record_retrieval_response()
        except Exception as e:
            print(
                f"ERROR: HybridMind not reachable at {args.base_url}: "
                f"{eval_common.sanitized_error(e)}"
            )
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
                             bm25_boost=bb, rerank_pool=rp, top_k=args.top_k,
                             base_url=args.base_url, split=args.split)
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
                split=args.split,
                decompose_multihop=args.decompose_multihop,
                with_answers=args.with_answers,
                answer_model=args.answer_model,
                search_mode=args.search_mode,
                route_weights=args.route_weights,
                track_access=args.track_access,
                graph_anchor_strategy=args.graph_anchor_strategy,
                anchor_node_ids=args.anchor_node_id,
                fusion_mode=args.fusion_mode,
            )
            print_results(metrics, label="LongMemEval Retrieval Results")
            # Save
            out = Path(metrics["ledger_path"]).with_suffix(".summary.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
                handle.write("\n")
            print(f"\nResults saved to {out}")
    print("Budget usage:", json.dumps(budget.usage(), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except eval_common.EvaluationBudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
