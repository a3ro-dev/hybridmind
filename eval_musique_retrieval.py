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
import os
import re
import sys
import time
from pathlib import Path

import httpx

import eval_common
import eval_ledger
from config import settings
from engine.query_router import route_query

BASE_URL = "http://127.0.0.1:8000"
DATA_DIR = Path("benchmarks/data/musique")


class EvaluationRunError(RuntimeError):
    pass


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
    p.add_argument(
        "--rerank-pool", type=int, default=40,
        help="Hard cross-encoder candidate cap; 0 disables, positive must be >= top-k",
    )
    p.add_argument("--top-k",            type=int,   default=20)
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
    p.add_argument("--sweep",            action="store_true")
    p.add_argument("--base-url",         default=BASE_URL)
    p.add_argument("--with-answers",     action="store_true",
                   help="Also run LLM QA-answering + accuracy scoring on top of retrieval")
    p.add_argument("--answer-model",     type=str,   default=None)
    p.add_argument("--decompose-multihop", dest="decompose_multihop", action="store_true", default=False,
                   help="Opt in to paid/provider-backed multihop decomposition")
    p.add_argument("--no-decompose-multihop", dest="decompose_multihop", action="store_false")
    eval_common.add_budget_arguments(p)
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
    """Diagnostic answer-overlap proxy; never a gold paragraph match."""
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
    graph_weight: float = 0.2,
    bm25_boost: float = 0.35,
    overlap_threshold: float = 0.10,
    rerank_pool: int = 40,
    top_k: int = 20,
    base_url: str = BASE_URL,
    with_answers: bool = False,
    answer_model: str | None = None,
    decompose_multihop: bool = False,
    data_file: str = "musique_ans_v1.0_dev.jsonl",
    search_mode: str = "hybrid",
    route_weights: bool = True,
    track_access: bool = False,
    graph_anchor_strategy: str = "explicit",
    anchor_node_ids: list[str] | None = None,
    fusion_mode: str | None = None,
):
    if top_k < 10:
        raise ValueError("MuSiQue evaluation requires top_k >= 10")
    eval_common.validate_rerank_pool(top_k=top_k, rerank_pool=rerank_pool)
    hit1 = hit5 = hit10 = mrr = 0.0
    overlap_hit1 = overlap_hit5 = overlap_hit10 = overlap_mrr = 0.0
    correct_sum = 0.0
    answer_n = 0
    retrieved_candidate_count = 0
    paragraph_tagged_candidate_count = 0
    supporting_recall10 = all_supporting_hit10 = 0.0
    by_hops: dict = {}

    ledger_config = {
        "benchmark": "musique",
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
        "data_file": data_file,
        "reranker_expected": search_mode == "hybrid" and rerank_pool > 0,
        "metric_primary": "exact_supporting_paragraph_id",
        "answer_overlap_role": "diagnostic_proxy_only",
        "search_mode": search_mode,
        "route_weights": route_weights,
        "track_access": track_access,
        "graph_anchor_strategy": graph_anchor_strategy,
        "anchor_node_ids": list(anchor_node_ids or []),
        "budget": eval_common.active_budget_provenance(),
    }
    if os.getenv("HYBRIDMIND_ABLATION_CONFIG_HASH") or os.getenv("HYBRIDMIND_ABLATION_MODE"):
        ledger_config["ablation"] = {
            "plan_hash": os.getenv("HYBRIDMIND_ABLATION_CONFIG_HASH", ""),
            "mode": os.getenv("HYBRIDMIND_ABLATION_MODE", ""),
            "resolved_settings_sha256": os.getenv(
                "HYBRIDMIND_ABLATION_SETTINGS_SHA256", ""
            ),
        }
    provenance = {"api_base_url": base_url}
    dataset_path = DATA_DIR / data_file
    if dataset_path.is_file():
        provenance["dataset"] = eval_ledger.dataset_provenance(dataset_path)
    ledger = eval_ledger.LedgerWriter("musique", ledger_config, provenance=provenance)
    metric_k = tuple(k for k in eval_ledger.DEFAULT_K_LIST if k <= top_k)

    for q in questions:
        supporting_ids = q["supporting_ids"]
        qtype = route_query(q["question"])["type"]
        execution_traces: list[dict] = []

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
                seed_body = seed.json()
                execution_traces.append({
                    "request_role": "graph_seed",
                    "trace": eval_common.validate_search_execution(
                        seed_body,
                        expected_request=seed_payload,
                        require_reranker=False,
                    ),
                })
                seed_results = seed_body.get("results", [])
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
            body = resp.json()
            execution_traces.append({
                "request_role": "retrieval",
                "trace": eval_common.validate_search_execution(
                    body,
                    expected_request=payload,
                    require_reranker=(search_mode == "hybrid" and rerank_pool > 0),
                ),
            })
            return body.get("results", [])

        try:
            results = eval_common.retrieve_with_decomposition(
                q["question"], qtype, _post, decompose_enabled=decompose_multihop
            )
        except Exception as e:
            print(f"  ERROR {q['question_id']}: {eval_common.sanitized_error(e)}")
            ledger.write(
                question_id=q["question_id"], question_type=qtype,
                gold_evidence_ids=sorted(supporting_ids),
                pool_metrics=eval_ledger.empty_pool_metrics(metric_k),
                status="retrieval_error", answer_status="not_run",
                error_type=type(e).__name__, error_message=eval_common.sanitized_error(e),
                extra={"search_execution_traces": execution_traces},
            )
            ledger.finalize_failure(
                reason="retrieval_error",
                error_type=type(e).__name__,
                expected_questions=len(questions),
            )
            raise EvaluationRunError(
                f"MuSiQue retrieval failed for {q['question_id']}; no score is valid"
            ) from e

        if search_mode == "hybrid" and rerank_pool > 0 and results and not _reranker_executed(results):
            error = "reranker was required but API results contain no rerank_score"
            ledger.write(
                question_id=q["question_id"], question_type=qtype,
                gold_evidence_ids=sorted(supporting_ids),
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

        # Exact supporting-paragraph relevance is primary. Answer overlap is a
        # separate diagnostic and can never upgrade an exact miss to a hit.
        relevance = []
        overlap_relevance = []
        retrieved_ids_at_10 = set()
        for rank, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            retrieved_candidate_count += 1
            paragraph_tagged_candidate_count += int(bool(
                meta.get("paragraph_id") or meta.get("musique_id") or meta.get("source_id")
            ))
            if supporting_ids and is_relevant_by_id(meta, supporting_ids):
                relevance.append(True)
                pid = meta.get("paragraph_id") or meta.get("musique_id") or meta.get("source_id", "")
                if rank <= 10:
                    retrieved_ids_at_10.add(str(pid))
            else:
                relevance.append(False)
            overlap_relevance.append(is_relevant_by_text(r.get("text", ""), q["answer"]))

        hit1 += any(relevance[:1])
        hit5 += any(relevance[:5])
        hit10 += any(relevance[:10])
        for rank, rel in enumerate(relevance[:10], 1):
            if rel:
                mrr += 1.0 / rank
                break
        overlap_hit1 += any(overlap_relevance[:1])
        overlap_hit5 += any(overlap_relevance[:5])
        overlap_hit10 += any(overlap_relevance[:10])
        overlap_mrr += next(
            (1.0 / rank for rank, rel in enumerate(overlap_relevance[:10], 1) if rel),
            0.0,
        )

        if supporting_ids:
            covered_support = supporting_ids & retrieved_ids_at_10
            supporting_recall10 += len(covered_support) / len(supporting_ids)
            all_supporting_hit10 += float(covered_support == supporting_ids)

        hypothesis, judged_correct = "", None
        judge_rationale = "not evaluated (retrieval-only run, pass --with-answers)"
        answer_status, judge_method = "not_requested", "not_run"
        prompt_version = ""
        if with_answers:
            snippets = [r.get("text", "") for r in results[:10] if r.get("text")]
            answer_result = eval_common.answer_question_with_status(
                q["question"], snippets, question_type=qtype, model=answer_model
            )
            hypothesis, prompt_version, answer_status = (
                answer_result.answer, answer_result.prompt_version, answer_result.status
            )
            if answer_status in {"provider_unavailable", "provider_error"}:
                ledger.write(
                    question_id=q["question_id"], question_type=qtype,
                    gold_evidence_ids=sorted(supporting_ids),
                    pool_metrics=eval_ledger.compute_pool_metrics(
                        results,
                        lambda r: is_relevant_by_id(r.get("metadata", {}), supporting_ids),
                        metric_k,
                        metric_basis="exact_supporting_paragraph_id",
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

        pool_metrics = eval_ledger.compute_pool_metrics(
            results,
            lambda r: is_relevant_by_id(r.get("metadata", {}), supporting_ids),
            metric_k,
            metric_basis="exact_supporting_paragraph_id",
        )
        overlap_metrics = eval_ledger.compute_pool_metrics(
            results,
            lambda r: is_relevant_by_text(r.get("text", ""), q["answer"]),
            metric_k,
            metric_basis="answer_text_overlap_proxy",
        )
        ledger.write(
            question_id=q["question_id"],
            question_type=qtype,
            gold_evidence_ids=sorted(supporting_ids),
            pool_metrics=pool_metrics,
            answer_overlap_metrics=overlap_metrics,
            raw_llm_answer=hypothesis,
            judged_correct=judged_correct,
            judge_rationale=judge_rationale,
            prompt_version=prompt_version,
            answer_status=answer_status,
            judge_method=judge_method,
            extra={
                "reranker_executed": _reranker_executed(results),
                "retrieved_supporting_ids_at_10": sorted(retrieved_ids_at_10),
                "supporting_paragraph_recall_at_10": (
                    len(supporting_ids & retrieved_ids_at_10) / len(supporting_ids)
                    if supporting_ids else None
                ),
                "all_supporting_paragraphs_hit_at_10": (
                    supporting_ids.issubset(retrieved_ids_at_10)
                    if supporting_ids else None
                ),
                "search_execution_traces": execution_traces,
            },
        )

        hk = q.get("n_hops") or "?"
        by_hops.setdefault(hk, {"n": 0, "hit5": 0, "mrr": 0, "overlap_hit5": 0})
        by_hops[hk]["n"] += 1
        by_hops[hk]["hit5"] += any(relevance[:5])
        by_hops[hk]["mrr"] += next((1.0 / r for r, rel in enumerate(relevance[:10], 1) if rel), 0.0)
        by_hops[hk]["overlap_hit5"] += any(overlap_relevance[:5])

    n = len(questions)
    if not n:
        ledger.finalize_failure(
            reason="zero_questions", error_type="EvaluationRunError", expected_questions=0
        )
        raise EvaluationRunError("MuSiQue run contained zero questions")
    if any(not q["supporting_ids"] for q in questions):
        ledger.finalize_failure(
            reason="missing_supporting_paragraph_ids",
            error_type="EvaluationRunError",
            expected_questions=n,
        )
        raise EvaluationRunError("MuSiQue run contains questions without supporting paragraph IDs")
    if not retrieved_candidate_count or not paragraph_tagged_candidate_count:
        ledger.finalize_failure(
            reason="unverified_corpus_metadata",
            error_type="EvaluationRunError",
            expected_questions=n,
        )
        raise EvaluationRunError(
            "MuSiQue corpus paragraph metadata cannot be verified from retrieved candidates"
        )
    result = {
        "metric_basis": "exact_supporting_paragraph_id",
        "hit_at_1": round(hit1 / n, 3),
        "hit_at_5": round(hit5 / n, 3),
        "hit_at_10": round(hit10 / n, 3),
        "mrr": round(mrr / n, 3),
        "supporting_paragraph_recall_at_10": round(supporting_recall10 / n, 3),
        "all_supporting_paragraphs_hit_at_10": round(all_supporting_hit10 / n, 3),
        "n": n,
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
        "by_hops": {
            str(hk): {
                "hit5": round(v["hit5"] / v["n"], 3) if v["n"] else 0,
                "mrr": round(v["mrr"] / v["n"], 3) if v["n"] else 0,
                "answer_overlap_hit5_proxy": round(v["overlap_hit5"] / v["n"], 3) if v["n"] else 0,
                "n": v["n"],
            }
            for hk, v in by_hops.items()
        },
    }
    if with_answers:
        result["accuracy"] = round(correct_sum / answer_n, 3) if answer_n else None
    ledger.finalize(status="completed", summary=result)
    return result


def print_results(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"  N            : {metrics['n']}")
    print(f"  Hit@1        : {metrics['hit_at_1']:.1%}")
    print(f"  Hit@5        : {metrics['hit_at_5']:.1%}")
    print(f"  Hit@10       : {metrics['hit_at_10']:.1%}")
    print(f"  MRR          : {metrics['mrr']:.3f}")
    print(
        f"  Support R@10 : {metrics['supporting_paragraph_recall_at_10']:.1%}  "
        f"All@10={metrics['all_supporting_paragraphs_hit_at_10']:.1%}"
    )
    proxy = metrics.get("answer_overlap_proxy", {})
    if proxy:
        print(f"  Answer-overlap diagnostic only: Hit@5={proxy['hit_at_5']:.1%} MRR={proxy['mrr']:.3f}")
    if "accuracy" in metrics:
        print(f"  Accuracy (LLM QA): {metrics['accuracy']:.1%}")
    if metrics.get("by_hops"):
        print("  By hop count:")
        for hk in sorted(metrics["by_hops"], key=lambda x: str(x)):
            v = metrics["by_hops"][str(hk)]
            print(f"    {hk}-hop  Hit@5={v['hit5']:.0%}  MRR={v['mrr']:.3f}  (n={v['n']})")


def main():
    args = parse_args()
    eval_common.validate_rerank_pool(top_k=args.top_k, rerank_pool=args.rerank_pool)
    questions = load_questions(args.data_file, args.n, args.n_hops)
    print(f"MuSiQue multi-hop retrieval eval — {len(questions)} questions")

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
            configs = [
                (0.5, 0.2, 0.35, 40),
                (0.4, 0.3, 0.35, 40),
                (0.6, 0.1, 0.35, 25),
                (0.3, 0.4, 0.25, 40),
            ]
            for vw, gw, bb, rp in configs:
                m = run_eval(questions, client, vector_weight=vw, graph_weight=gw,
                             bm25_boost=bb, rerank_pool=rp, top_k=args.top_k,
                             base_url=args.base_url, data_file=args.data_file)
                print(
                    f"  vw={vw} gw={gw} bb={bb} rp={rp} → "
                    f"Hit@5={m['hit_at_5']:.1%} MRR={m['mrr']:.3f} "
                    f"SupportR@10={m['supporting_paragraph_recall_at_10']:.1%}"
                )
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
                decompose_multihop=args.decompose_multihop,
                with_answers=args.with_answers,
                answer_model=args.answer_model,
                data_file=args.data_file,
                search_mode=args.search_mode,
                route_weights=args.route_weights,
                track_access=args.track_access,
                graph_anchor_strategy=args.graph_anchor_strategy,
                anchor_node_ids=args.anchor_node_id,
                fusion_mode=args.fusion_mode,
            )
            print_results(metrics, label="MuSiQue Multi-Hop Retrieval Results")
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
