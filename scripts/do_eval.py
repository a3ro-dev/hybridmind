"""Small fail-closed LoCoMo QA smoke test.

This is not a headline benchmark. It samples 20 questions from five
conversations, scopes retrieval to the matching conversation, generates an
answer through ``eval_common`` and labels the deterministic normalized-overlap
heuristic honestly. Provider/search failures terminate the run instead of
becoming answers or zeros.
"""
from __future__ import annotations

import json
import os
import random
import sys
import urllib.request
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_common
from config import settings

DATA_PATH = Path("memorybench/data/benchmarks/locomo/locomo10.json")
BASE_URL = os.getenv("HYBRIDMIND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    eval_common.add_budget_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    all_qa = []
    for element in data[:5]:
        sample_id = str(element.get("sample_id"))
        for item in element.get("qa", []):
            all_qa.append((sample_id, item.get("question"), str(item.get("answer"))))

    random.seed(42)
    sample = random.sample(all_qa, min(20, len(all_qa)))
    budget = eval_common.budget_from_args(args)
    if not args.execute:
        print("DRY RUN: no API/provider calls made. Re-run with --execute.")
        print(json.dumps({"questions": len(sample), "budget_ceilings": budget.ceilings()}, indent=2))
        return 0
    eval_common.enforce_priced_llm_budget(args, answer_requested=True)
    print(f"=== QA smoke test: {len(sample)} conversation-scoped questions ===")

    correct_count = 0
    with budget.activate():
        for index, (sample_id, question, gold_answer) in enumerate(sample, 1):
            eval_common.record_retrieval_query()
            request = urllib.request.Request(
            f"{BASE_URL}/search/hybrid",
            data=json.dumps({
                "query_text": question,
                "top_k": 10,
                "rerank_pool": 25,
                "filter_metadata": {"benchmark_sample_id": sample_id},
                "track_access": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", **eval_common.api_headers()},
        )
            with urllib.request.urlopen(
                request, timeout=eval_common.live_request_timeout(300.0)
            ) as response:
                search_data = json.loads(response.read().decode("utf-8"))
            eval_common.record_retrieval_response()
            results = search_data.get("results", [])
            if results and any(
                (result.get("metadata") or {}).get("benchmark_sample_id") != sample_id
                for result in results
            ):
                raise RuntimeError(f"conversation scope violation for {sample_id}")

            answer_result = eval_common.answer_question_with_status(
                question,
                [result.get("text", "") for result in results if result.get("text")],
            )
            if answer_result.status in {"provider_unavailable", "provider_error"}:
                raise RuntimeError(
                    f"answer provider failed for {sample_id}: {answer_result.status}"
                )
            is_correct, rationale = eval_common.judge_correct_normalized(
                answer_result.answer, gold_answer
            )
            correct_count += int(is_correct)
            print(
                f"[{index:02d}/{len(sample)}] {sample_id} "
                f"deterministic_normalized_answer_overlap_v1="
                f"{'CORRECT' if is_correct else 'INCORRECT'} ({rationale})"
            )

    accuracy = correct_count / len(sample) if sample else 0.0
    print(f"SMOKE-TEST HEURISTIC ACCURACY: {correct_count}/{len(sample)} ({accuracy:.1%})")
    print("Budget usage:", json.dumps(budget.usage(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except eval_common.EvaluationBudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"QA smoke test failed: {eval_common.sanitized_error(exc)}", file=sys.stderr)
        raise SystemExit(1) from exc
