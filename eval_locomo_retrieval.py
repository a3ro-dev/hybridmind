"""
Conversation-scoped LoCoMo retrieval evaluation.

Primary Hit@k/MRR use stable gold dialogue evidence IDs.  Answer-text overlap
is retained only as an explicitly labelled diagnostic proxy.

Usage (single config):
  python eval_locomo_retrieval.py [--vector-weight 0.5] [--graph-weight 0.15] ...

The historical same-set ``--sweep`` is quarantined because selecting a
configuration on evaluation questions is not held-out evidence.

Optional filters:
  --category single-hop|multi-hop|temporal|world-knowledge|adversarial
  --n 5        samples per category (default 5)
"""
import argparse
import hashlib
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
from config import settings
from engine.query_router import route_query

BASE_URL = os.getenv("HYBRIDMIND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LOCOMO_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")

CATEGORY_MAP = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "world-knowledge", 5: "adversarial"}


class EvaluationRunError(RuntimeError):
    """A benchmark infrastructure failure that must not become a score."""


def canonical_evidence_id(sample_id: str, evidence: object) -> str:
    value = str(evidence).strip()
    if value.startswith("locomo:"):
        return value
    return f"locomo:{sample_id}:{value}"


def parse_args():
    p = argparse.ArgumentParser(description="LoCoMo retrieval eval")
    p.add_argument("--vector-weight",    type=float, default=0.5)
    p.add_argument("--graph-weight",     type=float, default=0.15)
    p.add_argument("--bm25-boost",       type=float, default=0.35)
    p.add_argument("--overlap-threshold",type=float, default=0.15)
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
        help="Controlled retrieval mode used by scripts/ablation_matrix.py",
    )
    p.add_argument(
        "--route-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable routing behavior for omitted controls; explicit weights remain authoritative",
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
    p.add_argument(
        "--sweep", action="store_true",
        help="Quarantined: same-set benchmark tuning is not valid held-out evidence",
    )
    p.add_argument("--with-answers",     action="store_true",
                   help="Also run LLM QA-answering + accuracy scoring on top of retrieval")
    p.add_argument("--answer-model",     type=str,   default=None,
                   help="Override the Z.AI answering model (default: HYBRIDMIND_QA_MODEL or glm-4.6)")
    p.add_argument("--decompose-multihop", dest="decompose_multihop", action="store_true", default=False,
                   help="Opt in to paid/provider-backed multihop decomposition")
    p.add_argument("--no-decompose-multihop", dest="decompose_multihop", action="store_false")
    eval_common.add_budget_arguments(p)
    return p.parse_args()


def load_questions(samples_per_category: int = 5, category_filter: str | None = None):
    data = json.loads(LOCOMO_PATH.read_text())
    questions = []
    counts: dict = {}
    for conv_index, conv in enumerate(data):
        sample_id = str(conv.get("sample_id") or f"locomo_{conv_index}")
        for qa_index, qa in enumerate(conv.get("qa", [])):
            cat = CATEGORY_MAP.get(qa.get("category", 0), "unknown")
            if category_filter and cat != category_filter:
                continue
            if samples_per_category > 0 and counts.get(cat, 0) >= samples_per_category:
                continue
            counts[cat] = counts.get(cat, 0) + 1
            ans = qa.get("answer") or qa.get("adversarial_answer", "")
            question_text = qa["question"]
            evidence = [
                canonical_evidence_id(sample_id, value)
                for value in qa.get("evidence", [])
                if str(value).strip()
            ]
            source_question_id = qa.get("question_id")
            stable_question_id = (
                f"locomo:{sample_id}:{source_question_id}"
                if source_question_id
                else "locomo:" + hashlib.sha1(
                    f"{sample_id}\0{qa_index}\0{question_text}".encode()
                ).hexdigest()[:16]
            )
            questions.append({
                "question_id": stable_question_id,
                "question": question_text,
                "answer": str(ans).strip(),
                "category": cat,
                "sample_id": sample_id,
                "evidence": evidence,
            })
    return questions


def answer_tokens(answer: str) -> set:
    tokens = set(re.findall(r"[A-Za-z0-9']+", answer.lower()))
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was", "it", "and", "or", "but"}
    return tokens - stopwords


def is_relevant(retrieved_text: str, answer: str) -> bool:
    """Legacy answer-overlap proxy.  Never use as exact evidence relevance."""
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


def candidate_evidence_ids(result: dict, sample_id: str = "") -> set[str]:
    metadata = result.get("metadata") or {}
    values = []
    for key in ("evidence_id", "dia_id"):
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value is not None:
            values.append(value)
    ids = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        ids.add(text if text.startswith("locomo:") else canonical_evidence_id(sample_id, text))
    return ids


def is_exact_evidence(result: dict, gold_evidence_ids: set[str], sample_id: str) -> bool:
    return bool(gold_evidence_ids & candidate_evidence_ids(result, sample_id))


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
    verbose: bool = True,
    with_answers: bool = False,
    answer_model: str | None = None,
    decompose_multihop: bool = False,
    search_mode: str = "hybrid",
    route_weights: bool = True,
    track_access: bool = False,
    graph_anchor_strategy: str = "explicit",
    anchor_node_ids: list[str] | None = None,
    require_reranker: bool | None = None,
    fusion_mode: str | None = None,
) -> dict:
    if top_k < 10:
        raise ValueError("LoCoMo evaluation requires top_k >= 10 for Hit@10")
    eval_common.validate_rerank_pool(top_k=top_k, rerank_pool=rerank_pool)
    reranker_expected = (
        search_mode == "hybrid" and rerank_pool > 0
        if require_reranker is None
        else require_reranker
    )
    exact_hits = {1: [], 5: [], 10: []}
    overlap_hits = {1: [], 5: [], 10: []}
    exact_mrrs: list[float] = []
    exact_recall_at_10: list[float] = []
    all_evidence_hit_at_10: list[float] = []
    overlap_mrrs: list[float] = []
    all_correct: list[float] = []
    results_by_category: dict = {}
    failure_count = 0
    retrieved_candidate_count = 0
    evidence_tagged_candidate_count = 0

    ledger_config = {
        "benchmark": "locomo",
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
        "search_mode": search_mode,
        "route_weights": route_weights,
        "track_access": track_access,
        "graph_anchor_strategy": graph_anchor_strategy,
        "anchor_node_ids": list(anchor_node_ids or []),
        "request_top_k": top_k,
        "reranker_expected": reranker_expected,
        "metric_primary": "exact_evidence_id",
        "answer_overlap_role": "diagnostic_proxy_only",
        "budget": eval_common.active_budget_provenance(),
    }
    ablation_plan_hash = os.getenv("HYBRIDMIND_ABLATION_CONFIG_HASH", "").strip()
    ablation_mode = os.getenv("HYBRIDMIND_ABLATION_MODE", "").strip()
    if ablation_plan_hash or ablation_mode:
        ledger_config["ablation"] = {
            "plan_hash": ablation_plan_hash,
            "mode": ablation_mode,
            "resolved_settings_sha256": os.getenv(
                "HYBRIDMIND_ABLATION_SETTINGS_SHA256", ""
            ).strip(),
        }
    ledger = eval_ledger.LedgerWriter(
        "locomo",
        ledger_config,
        provenance={
            "dataset": eval_ledger.dataset_provenance(LOCOMO_PATH),
            "api_base_url": BASE_URL,
        },
    )
    metric_k = tuple(k for k in eval_ledger.DEFAULT_K_LIST if k <= top_k)

    for i, q in enumerate(questions):
        question = q["question"]
        answer = q["answer"]
        category = q["category"]
        sample_id = q.get("sample_id", "")
        gold_evidence_ids = set(q.get("evidence", []))

        if verbose:
            safe_q = question[:80].encode("ascii", "backslashreplace").decode("ascii")
            safe_a = answer[:80].encode("ascii", "backslashreplace").decode("ascii")
            print(f"[{i+1}/{len(questions)}] [{category}] {safe_q}...")
            print(f"  Answer: {safe_a}...")

        qtype = route_query(question)["type"]

        def _post(q_text: str) -> list:
            anchors = list(anchor_node_ids or [])
            if search_mode == "graph_only" and not anchors and graph_anchor_strategy == "vector_top1":
                eval_common.record_retrieval_query()
                seed_payload = {
                    "query_text": q_text,
                    "top_k": 1,
                    "min_score": 0.0,
                    "vector_weight": 1.0,
                    "graph_weight": 0.0,
                    "bm25_boost_weight": 0.0,
                    "rerank_pool": 0,
                    "search_mode": "vector_only",
                    "route_weights": False,
                    "track_access": False,
                    "filter_metadata": {"benchmark_sample_id": sample_id},
                }
                if fusion_mode is not None:
                    seed_payload["fusion_mode"] = fusion_mode
                seed = client.post(
                    "/search/hybrid", json=seed_payload,
                    timeout=eval_common.live_request_timeout(300.0),
                )
                eval_common.record_retrieval_response()
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
                "filter_metadata": {"benchmark_sample_id": sample_id},
            }
            if fusion_mode is not None:
                payload["fusion_mode"] = fusion_mode
            if anchors:
                payload["anchor_nodes"] = anchors
            eval_common.record_retrieval_query()
            resp = client.post(
                "/search/hybrid", json=payload,
                timeout=eval_common.live_request_timeout(300.0),
            )
            eval_common.record_retrieval_response()
            resp.raise_for_status()
            return resp.json().get("results", [])

        try:
            res_list = eval_common.retrieve_with_decomposition(
                question, qtype, _post, decompose_enabled=decompose_multihop
            )
        except Exception as e:
            failure_count += 1
            ledger.write(
                question_id=q["question_id"],
                question_type=qtype,
                gold_evidence_ids=sorted(gold_evidence_ids),
                pool_metrics=eval_ledger.empty_pool_metrics(metric_k),
                status="retrieval_error",
                answer_status="not_run",
                error_type=type(e).__name__,
                error_message=eval_common.sanitized_error(e),
                extra={"sample_id": sample_id},
            )
            if verbose:
                print(f"  ERROR: {eval_common.sanitized_error(e)}")
            ledger.finalize_failure(
                reason="retrieval_error",
                error_type=type(e).__name__,
                expected_questions=len(questions),
            )
            raise EvaluationRunError(
                f"LoCoMo retrieval failed for {q['question_id']}; no score is valid"
            ) from e

        if any((result.get("metadata") or {}).get("benchmark_sample_id") != sample_id for result in res_list):
            error = "server returned a result outside the requested LoCoMo conversation"
            ledger.write(
                question_id=q["question_id"], question_type=qtype,
                gold_evidence_ids=sorted(gold_evidence_ids),
                pool_metrics=eval_ledger.empty_pool_metrics(metric_k),
                status="scope_violation", answer_status="not_run",
                error_type="ConversationScopeError", error_message=error,
                extra={"sample_id": sample_id},
            )
            ledger.finalize_failure(
                reason="conversation_scope_violation",
                error_type="ConversationScopeError",
                expected_questions=len(questions),
            )
            raise EvaluationRunError(error)
        if reranker_expected and res_list and not _reranker_executed(res_list):
            error = "reranker was required but API results contain no rerank_score"
            ledger.write(
                question_id=q["question_id"], question_type=qtype,
                gold_evidence_ids=sorted(gold_evidence_ids),
                pool_metrics=eval_ledger.empty_pool_metrics(metric_k),
                status="reranker_not_executed", answer_status="not_run",
                error_type="RerankerExecutionError", error_message=error,
                extra={"sample_id": sample_id},
            )
            ledger.finalize_failure(
                reason="reranker_not_executed",
                error_type="RerankerExecutionError",
                expected_questions=len(questions),
            )
            raise EvaluationRunError(error)

        retrieved_candidate_count += len(res_list)
        evidence_tagged_candidate_count += sum(
            bool(candidate_evidence_ids(result, sample_id)) for result in res_list
        )

        exact_metrics = eval_ledger.compute_pool_metrics(
            res_list,
            lambda result: is_exact_evidence(result, gold_evidence_ids, sample_id),
            metric_k,
            metric_basis="exact_evidence_id",
        )
        overlap_metrics = eval_ledger.compute_pool_metrics(
            res_list,
            lambda result: is_relevant(result.get("text", ""), answer),
            metric_k,
            metric_basis="answer_text_overlap_proxy",
        )
        retrieved_evidence_ids_at_k = {
            str(k): sorted(set().union(*(
                candidate_evidence_ids(result, sample_id) for result in res_list[:k]
            ))) if res_list[:k] else []
            for k in metric_k
        }
        evidence_tagged_results_at_k = {
            str(k): sum(bool(candidate_evidence_ids(result, sample_id)) for result in res_list[:k])
            for k in metric_k
        }
        exact_eligible = bool(gold_evidence_ids)
        if exact_eligible:
            for k in exact_hits:
                exact_hits[k].append(float(exact_metrics["hit_at_k"][str(k)]))
            exact_rank = exact_metrics["gold_rank_post_rerank"]
            exact_mrrs.append(1.0 / exact_rank if exact_rank and exact_rank <= 10 else 0.0)
            retrieved_evidence_at_10 = set().union(
                *(candidate_evidence_ids(result, sample_id) for result in res_list[:10])
            ) if res_list[:10] else set()
            covered = gold_evidence_ids & retrieved_evidence_at_10
            exact_recall_at_10.append(len(covered) / len(gold_evidence_ids))
            all_evidence_hit_at_10.append(float(covered == gold_evidence_ids))
        for k in overlap_hits:
            overlap_hits[k].append(float(overlap_metrics["hit_at_k"][str(k)]))
        overlap_rank = overlap_metrics["gold_rank_post_rerank"]
        overlap_mrrs.append(1.0 / overlap_rank if overlap_rank and overlap_rank <= 10 else 0.0)

        hypothesis, judged_correct = "", None
        judge_rationale = "not evaluated (retrieval-only run, pass --with-answers)"
        answer_status = "not_requested"
        judge_method = "not_run"
        prompt_version = ""
        if with_answers:
            snippets = [r.get("text", "") for r in res_list[:10] if r.get("text")]
            answer_result = eval_common.answer_question_with_status(
                question, snippets, question_type=qtype, model=answer_model
            )
            hypothesis = answer_result.answer
            prompt_version = answer_result.prompt_version
            answer_status = answer_result.status
            if answer_status in {"provider_unavailable", "provider_error"}:
                ledger.write(
                    question_id=q["question_id"], question_type=qtype,
                    gold_evidence_ids=sorted(gold_evidence_ids), pool_metrics=exact_metrics,
                    answer_overlap_metrics=overlap_metrics, raw_llm_answer=hypothesis,
                    status="answer_error", answer_status=answer_status,
                    prompt_version=prompt_version, error_type="AnswerProviderError",
                    error_message=answer_result.error or answer_status,
                    extra={"sample_id": sample_id, "exact_evidence_available": exact_eligible},
                )
                ledger.finalize_failure(
                    reason="answer_provider_error",
                    error_type="AnswerProviderError",
                    expected_questions=len(questions),
                )
                raise EvaluationRunError(
                    f"answer provider failed for {q['question_id']}: {answer_status}"
                )
            judged_correct, judge_rationale = eval_common.judge_correct_normalized(hypothesis, answer)
            judge_method = "deterministic_normalized_answer_overlap_v1"
            all_correct.append(1.0 if judged_correct else 0.0)
            if verbose:
                print(f"  QA answer: {hypothesis[:100]!r} -> {'CORRECT' if judged_correct else 'WRONG'}")

        ledger.write(
            question_id=q["question_id"],
            question_type=qtype,
            gold_evidence_ids=sorted(gold_evidence_ids),
            pool_metrics=exact_metrics,
            answer_overlap_metrics=overlap_metrics,
            raw_llm_answer=hypothesis,
            judged_correct=judged_correct,
            judge_rationale=judge_rationale,
            prompt_version=prompt_version,
            answer_status=answer_status,
            judge_method=judge_method,
            extra={
                "sample_id": sample_id,
                "exact_evidence_available": exact_eligible,
                "retrieved_evidence_ids_at_k": retrieved_evidence_ids_at_k,
                "retrieved_result_count_at_k": {
                    str(k): len(res_list[:k]) for k in metric_k
                },
                "evidence_tagged_result_count_at_k": evidence_tagged_results_at_k,
                "gold_evidence_recall_at_10": exact_recall_at_10[-1] if exact_eligible else None,
                "all_gold_evidence_hit_at_10": all_evidence_hit_at_10[-1] if exact_eligible else None,
                "reranker_executed": _reranker_executed(res_list),
            },
        )

        if verbose:
            if exact_eligible:
                print(
                    f"  Exact evidence Hit@1={exact_hits[1][-1]:.0%} "
                    f"Hit@5={exact_hits[5][-1]:.0%} Hit@10={exact_hits[10][-1]:.0%} "
                    f"MRR={exact_mrrs[-1]:.3f}"
                )
            print(f"  Answer-overlap proxy Hit@5={overlap_hits[5][-1]:.0%}")
            for rank, r in enumerate(res_list[:3], 1):
                text = r.get("text", "")[:120]
                vs = r.get("vector_score", 0)
                gs = r.get("graph_score", 0)
                gg = r.get("graph_gate", 1.0)
                egs = r.get("effective_graph_score", gs)
                cs = r.get("combined_score", 0)
                rel = "[OK]" if is_exact_evidence(r, gold_evidence_ids, sample_id) else "[NO]"
                print(f"    #{rank} {rel} v={vs:.3f} g={gs:.3f} gate={gg:.3f} efg={egs:.3f} c={cs:.3f} {text[:100]}")

        if category not in results_by_category:
            results_by_category[category] = {"hits": [], "mrrs": [], "correct": [], "overlap_hits": []}
        if exact_eligible:
            results_by_category[category]["hits"].append(exact_hits[5][-1])
            results_by_category[category]["mrrs"].append(exact_mrrs[-1])
        results_by_category[category]["overlap_hits"].append(overlap_hits[5][-1])
        if with_answers:
            results_by_category[category]["correct"].append(all_correct[-1])

        time.sleep(0.05)

    n = len(questions)
    if n == 0:
        ledger.finalize_failure(
            reason="zero_questions", error_type="EvaluationRunError", expected_questions=0
        )
        raise EvaluationRunError("LoCoMo run contained zero questions")
    n_exact = len(exact_mrrs)
    if n_exact == 0:
        ledger.finalize_failure(
            reason="no_exact_evidence",
            error_type="EvaluationRunError",
            expected_questions=n,
        )
        raise EvaluationRunError("LoCoMo run contained no questions with exact evidence IDs")
    if retrieved_candidate_count == 0:
        ledger.finalize_failure(
            reason="zero_retrieved_candidates",
            error_type="EvaluationRunError",
            expected_questions=n,
        )
        raise EvaluationRunError(
            "LoCoMo returned zero candidates for the entire run; corpus ingestion cannot be verified"
        )
    if evidence_tagged_candidate_count == 0:
        ledger.finalize_failure(
            reason="missing_evidence_metadata",
            error_type="EvaluationRunError",
            expected_questions=n,
        )
        raise EvaluationRunError(
            "LoCoMo candidates contain no stable evidence IDs; re-run scripts/ingest_locomo.py"
        )
    summary = {
        "metric_basis": "exact_evidence_id",
        "hit_at_1": mean(exact_hits[1]),
        "hit_at_5": mean(exact_hits[5]),
        "hit_at_10": mean(exact_hits[10]),
        "mrr": mean(exact_mrrs),
        "gold_evidence_recall_at_10": mean(exact_recall_at_10),
        "all_gold_evidence_hit_at_10": mean(all_evidence_hit_at_10),
        "n":              n,
        "n_exact_evidence": n_exact,
        "failures": failure_count,
        "answer_overlap_proxy": {
            "hit_at_1": mean(overlap_hits[1]),
            "hit_at_5": mean(overlap_hits[5]),
            "hit_at_10": mean(overlap_hits[10]),
            "mrr": mean(overlap_mrrs),
        },
        "ledger_path": str(ledger.path),
        "manifest_path": str(ledger.manifest_path),
        "completion_path": str(ledger.completion_path),
        "budget": eval_common.active_budget_provenance(),
        "by_category":    {
            cat: {
                **({"hit5": mean(v["hits"]), "mrr": mean(v["mrrs"])} if v["hits"] else {}),
                "answer_overlap_hit5_proxy": mean(v["overlap_hits"]),
                **({"accuracy": mean(v["correct"])} if with_answers and v["correct"] else {}),
            }
            for cat, v in results_by_category.items()
        },
    }
    if with_answers and all_correct:
        summary["accuracy"] = mean(all_correct)
    ledger.finalize(status="completed", summary=summary)
    return summary


def print_summary(summary: dict, label: str = ""):
    n = summary.get("n", 0)
    n_exact = summary.get("n_exact_evidence", n)
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Questions evaluated: {n} ({n_exact} with exact gold evidence)")
    print(f"  Primary metric basis: {summary['metric_basis']}")
    print(f"  Hit@1:  {summary['hit_at_1']:.4f}  ({round(summary['hit_at_1']*n_exact)}/{n_exact})")
    print(f"  Hit@5:  {summary['hit_at_5']:.4f}  ({round(summary['hit_at_5']*n_exact)}/{n_exact})")
    print(f"  Hit@10: {summary['hit_at_10']:.4f}  ({round(summary['hit_at_10']*n_exact)}/{n_exact})")
    print(f"  MRR:    {summary['mrr']:.4f}")
    print(f"  Gold evidence Recall@10: {summary['gold_evidence_recall_at_10']:.4f}")
    print(f"  All-evidence Hit@10: {summary['all_gold_evidence_hit_at_10']:.4f}")
    proxy = summary["answer_overlap_proxy"]
    print(f"  Answer-overlap diagnostic only: Hit@5={proxy['hit_at_5']:.4f} MRR={proxy['mrr']:.4f}")
    if "accuracy" in summary:
        print(f"  Accuracy (LLM QA): {summary['accuracy']:.4f}  ({round(summary['accuracy']*n)}/{n})")
    print()
    print("  --- By Category ---")
    for cat in sorted(summary["by_category"].keys()):
        c = summary["by_category"][cat]
        acc_str = f" Accuracy={c['accuracy']:.4f}" if "accuracy" in c else ""
        exact_str = (
            f"Exact Hit@5={c['hit5']:.4f} MRR={c['mrr']:.4f} "
            if "hit5" in c else "Exact evidence unavailable "
        )
        print(f"  {cat}: {exact_str}OverlapProxy@5={c['answer_overlap_hit5_proxy']:.4f}{acc_str}")
    print()
    acc_str = f" Accuracy={summary['accuracy']:.3f}" if "accuracy" in summary else ""
    print(f"  SUMMARY: Hit@1={summary['hit_at_1']:.3f} Hit@5={summary['hit_at_5']:.3f} "
          f"Hit@10={summary['hit_at_10']:.3f} MRR={summary['mrr']:.4f}{acc_str}")


def sweep(questions: list, client: httpx.Client | None):
    """Reject the historical same-set hyperparameter sweep."""
    del questions, client
    raise SystemExit(
        "LoCoMo --sweep is quarantined: same-set tuning on evaluation "
        "questions is not held-out evidence. Use a disjoint tuning dataset "
        "and a frozen config artifact."
    )


def main():
    args = parse_args()
    eval_common.validate_rerank_pool(top_k=args.top_k, rerank_pool=args.rerank_pool)
    if args.sweep:
        # Reject before dataset, API, embedding, or LLM access.
        sweep([], None)

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

    client = httpx.Client(
        base_url=BASE_URL, timeout=300.0, headers=eval_common.api_headers()
    )
    try:
        with budget.activate():
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
                    fusion_mode=args.fusion_mode,
                )
                if summary:
                    print_summary(summary, label="RETRIEVAL EVALUATION RESULTS")
        print("Budget usage:", json.dumps(budget.usage(), sort_keys=True))
    except eval_common.EvaluationBudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        print(json.dumps(budget.usage(), sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from exc
    finally:
        client.close()


if __name__ == "__main__":
    main()
