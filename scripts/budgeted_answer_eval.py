"""Budget-controlled answer evaluation with the official LongMemEval reader and judge prompts.

Arms: `none` (closed book), `oracle` (answer sessions for LongMemEval, gold turns for
LoCoMo), `full` (whole haystack), or `<rep>|<strategy>|<budget>` packed exactly as in
scripts/offline_budgeted_evidence.py. Context is rendered in the official LongMemEval
`nl` format (sessions sorted chronologically) and read with the official chain-of-thought
template at temperature 0.

Default is a DRY RUN: every prompt is built, token usage is counted with the declared proxy,
and the planned usage for a `hybridmind-live-eval-plan/v1` is written. Zero network calls.
`--execute --plan <plan.json>` validates the plan (offline report hash, host, resources,
priced ceiling) and stops at the plan's usage ceiling. Every question becomes a ledger row;
any provider failure stops the run with a failed completion receipt (protocol §7).

Prompts are copied verbatim from github.com/xiaowu0162/LongMemEval (MIT):
src/generation/run_generation.py and src/evaluation/evaluate_qa.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_budgeted_evidence import (  # noqa: E402
    Conv, load, load_longmemeval, ntok, pack, render, strategy_units,
)

READER_COT = (  # run_generation.py, retrieval + CoT, no fact expansion
    "I will give you several history chats between you and a user. Please answer the question based on the "
    "relevant chat history. Answer the question step by step: first extract all the relevant information, and "
    "then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n"
    "Question: {}\nAnswer (step by step):"
)
READER_NONE = "{}Answer step by step."  # run_generation.py, no-retrieval + CoT
_J = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
      "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
      "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
      "the response only contains a subset of the information required by the answer, answer no. ")
JUDGE = {  # evaluate_qa.py get_anscheck_prompt, verbatim
    "default": _J + "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": _J + "In addition, do not penalize off-by-one errors for the number of days. If the "
    "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\n"
    "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
    "information along with an updated answer, the response should be considered as correct as long as the updated "
    "answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
    "response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
    "The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please "
    "answer yes if the model correctly identifies the question as unanswerable. The model could say that the "
    "information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: "
    "{}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only.",
}
# LoCoMo has no official LLM judge. Deviation (documented): LongMemEval's strict templates,
# temporal template for temporal questions, abstention template for adversarial questions.
LOCOMO_ADVERSARIAL_EXPLANATION = "The conversation does not contain this information."
READER_MAX_TOKENS, JUDGE_MAX_TOKENS = 800, 10  # official: 800 with CoT; judge max_tokens 10
PROMPT_VERSION = "longmemeval-official-cot-v1"
RESPONSE_SLOT = "<<MODEL_RESPONSE>>"  # filled after the reader answers


def judge_prompt(qa: dict) -> str:
    if qa["abstention"]:
        explanation = qa["answer"] if qa["answer"] and qa["cat"] != "adversarial" else LOCOMO_ADVERSARIAL_EXPLANATION
        return JUDGE["abstention"].format(qa["q"], explanation, RESPONSE_SLOT)
    key = qa["cat"] if qa["cat"] in JUDGE else ("temporal-reasoning" if qa["cat"] == "temporal" else "default")
    return JUDGE[key].format(qa["q"], qa["answer"], RESPONSE_SLOT)


def render_history(conv: Conv, idxs: list[int]) -> str:
    """Official `nl` history: selected turns grouped by session, sessions in time order."""
    by_session: dict[int, list[int]] = {}
    for i in sorted(idxs):
        by_session.setdefault(conv.turns[i]["session"], []).append(i)
    out = ""
    for n, s in enumerate(sorted(by_session, key=lambda s: (conv.turns[by_session[s][0]]["date"], s)
                                 if conv.lme else s), start=1):
        body = "".join(f"\n\n{conv.turns[i]['speaker']}: {render_value(conv.turns[i])}" for i in by_session[s])
        out += f"\n### Session {n}:\nSession Date: {conv.turns[by_session[s][0]]['date']}\nSession Content:\n{body}\n"
    return out


def render_value(t: dict) -> str:
    return f"{t['text']} [shares image: {t['caption']}]" if t["caption"] else t["text"]


def select(arm: str, conv: Conv, qa: dict) -> list[int] | None:
    if arm == "none":
        return None
    if arm == "full":
        return list(range(len(conv.turns)))
    if arm == "oracle":
        if conv.lme:
            return [i for i, t in enumerate(conv.turns) if t["session"] in set(qa["answer_sessions"])]
        return [conv.by_id[g] for g in qa["gold"]]
    _, strategy, budget = arm.split("|")
    return pack(strategy_units(strategy, conv, qa["q"]), conv.cost, int(budget))


def build(args) -> list[dict]:
    convs = load_longmemeval(args.dataset) if args.kind == "longmemeval" else load(args.dataset)
    items = [(c, qa) for c in convs for qa in c["qs"]]
    if args.kind == "locomo" and not args.include_adversarial:
        items = [(c, qa) for c, qa in items if qa["cat"] != "adversarial"]
    if args.sample:  # deterministic, stratified by category
        by_cat: dict[str, list] = {}
        for c, qa in items:
            by_cat.setdefault(qa["cat"], []).append((c, qa))
        items = []
        total = sum(len(v) for v in by_cat.values())
        for cat, group in sorted(by_cat.items()):
            group.sort(key=lambda x: hashlib.sha256(f"{args.sample_seed}:{x[1]['qid']}".encode()).hexdigest())
            items += group[: max(1, round(args.sample * len(group) / total))]
    rows, cache = [], {}
    rep = args.arm.split("|")[0] if "|" in args.arm else "raw"
    for c, qa in items:
        if c["sid"] not in cache:
            cache.clear()
            conv = cache[c["sid"]] = Conv(c, rep)
            conv.lme = args.kind == "longmemeval"
        conv = cache[c["sid"]]
        idxs = select(args.arm, conv, qa)
        if idxs is None:
            prompt = READER_NONE.format(qa["q"])
        else:
            prompt = READER_COT.format(render_history(conv, idxs), qa["question_date"] or "unknown", qa["q"])
        gold = {conv.by_id[g] for g in qa["gold"]}
        packed = set(idxs or [])
        rows.append({
            "question_id": qa["qid"], "category": qa["cat"], "abstention": qa["abstention"],
            "question": qa["q"], "gold_answer": qa["answer"], "arm": args.arm,
            "packed_evidence_ids": sorted(conv.turns[i]["id"] for i in packed),
            "gold_evidence_ids": sorted(qa["gold"]),
            "complete_coverage": bool(gold) and gold <= packed, "any_hit": bool(gold & packed),
            "context_proxy_tokens": sum(conv.cost[i] for i in packed),
            "reader_prompt": prompt, "reader_prompt_proxy_tokens": ntok(prompt),
            "judge_template": judge_prompt(qa),
        })
    return rows


def planned_usage(rows: list[dict]) -> dict:
    reader_in = sum(r["reader_prompt_proxy_tokens"] for r in rows)
    judge_in = sum(ntok(r["judge_template"]) + READER_MAX_TOKENS for r in rows)
    return {"queries": len(rows), "llm_calls": 2 * len(rows), "reader_input_tokens": reader_in + judge_in,
            "reader_output_tokens": len(rows) * (READER_MAX_TOKENS + JUDGE_MAX_TOKENS)}


def _ledger_ok_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["question_id"] for r in rows if r.get("status") == "ok"}


def run_stage(args, rows: list[dict]) -> int:
    """Reader or judge pass, bound to its own validated plan; stops at the ceiling or first failure."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from engine import llm_client
    from engine.resource_accounting import load_and_validate_live_plan

    plan, gate = load_and_validate_live_plan(args.plan)
    if "zai" not in plan["providers"]:
        raise SystemExit("plan does not admit the zai provider")
    ceiling = plan["usage_ceiling"]
    if args.stage == "judge":  # judge the reader's ok answers only
        answers = {}
        for line in args.answers.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    answers[r["question_id"]] = r
        rows = [dict(answers[r["question_id"]], judge_template=r["judge_template"]) for r in rows
                if r["question_id"] in answers]
    done = _ledger_ok_ids(args.output)
    todo = [r for r in rows if r["question_id"] not in done]
    model = args.model if args.stage == "reader" else args.judge_model
    max_tokens = READER_MAX_TOKENS if args.stage == "reader" else JUDGE_MAX_TOKENS
    manifest = {
        "schema": "hybridmind.budgeted_answer_eval.v2", "stage": args.stage, "arm": args.arm, "kind": args.kind,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(), "gate": vars(gate),
        "provider": "zai", "model": model, "thinking": "disabled", "temperature": 0, "max_tokens": max_tokens,
        "prompt_version": PROMPT_VERSION, "judge_label": "yes-substring (official)",
        "answers_sha256": hashlib.sha256(args.answers.read_bytes()).hexdigest() if args.stage == "judge" else None,
        "workers": args.workers, "questions": len(rows), "resumed_ok": len(done),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    # Spend is cumulative per plan across invocations (one arm per run must not reset the ceiling).
    spent_path = args.plan.with_suffix(".spent.json")
    used = {"llm_calls": 0, "reader_input_tokens": 0, "reader_output_tokens": 0}
    if spent_path.exists():
        used.update(json.loads(spent_path.read_text(encoding="utf-8")))
    lock = threading.Lock()
    state = {"status": "completed"}
    ledger = args.output.open("a", encoding="utf-8", newline="\n")

    def one(row: dict) -> None:
        if args.stage == "reader":
            prompt = row["reader_prompt"]
        else:
            prompt = row["judge_template"].replace(RESPONSE_SLOT, row["reader_answer"], 1)
        need = {"llm_calls": 1, "reader_input_tokens": 2 * ntok(prompt), "reader_output_tokens": max_tokens}
        with lock:
            if state["status"] != "completed":
                return
            if any(used[k] + need[k] > ceiling[k] for k in need):
                state["status"] = "budget_exhausted"
                return
            for k in need:
                used[k] += need[k]  # reserve; settled to provider-reported usage below
        t0, usage = time.perf_counter(), {}
        text = llm_client.chat_completion(
            [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0, model=model,
            preferred="zai", allow_fallback=False, usage=usage, zai_thinking=False)
        record = {k: v for k, v in row.items() if k != "reader_prompt"}
        if args.stage == "reader":
            record["reader_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            record.update(reader_answer=text, reader_usage=usage)
        else:
            record.update(judge_raw=text, judge_usage=usage, label=None if text is None else "yes" in text.lower())
        record.update(seconds=time.perf_counter() - t0, status="ok" if text is not None else "provider_error")
        with lock:
            for key, field in (("reader_input_tokens", "prompt_tokens"), ("reader_output_tokens", "completion_tokens")):
                used[key] += int(usage.get(field) or need[key]) - need[key]
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            if record["status"] != "ok" and state["status"] == "completed":
                state["status"] = "provider_failure"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    ledger.close()
    spent_path.write_text(json.dumps(used), encoding="utf-8")
    body = args.output.read_bytes()
    receipt = {"stage": args.stage, "status": state["status"], "usage": used,
               "ledger_sha256": hashlib.sha256(body).hexdigest(), "rows": body.count(b"\n"),
               "ok_rows": len(_ledger_ok_ids(args.output)), "finished_at": datetime.now(timezone.utc).isoformat()}
    args.output.with_suffix(".completion.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt))
    return 0 if state["status"] == "completed" else 3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=["longmemeval", "locomo"], required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--arm", required=True, help="none | oracle | full | <rep>|<strategy>|<budget>")
    p.add_argument("--sample", type=int, default=0, help="stratified deterministic subset size (0 = all)")
    p.add_argument("--sample-seed", default="20260925-answers")
    p.add_argument("--question-ids", type=Path, help="file with one question_id per line (overrides --sample)")
    p.add_argument("--include-adversarial", action="store_true", help="LoCoMo category 5 (abstention-judged)")
    p.add_argument("--output", type=Path, required=True, help="stage ledger .jsonl (dry run writes .plan-usage.json)")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--stage", choices=["reader", "judge"], default="reader")
    p.add_argument("--answers", type=Path, help="reader ledger to judge (judge stage)")
    p.add_argument("--plan", type=Path)
    p.add_argument("--model", default="glm-4.7-flash", help="Z.AI reader model")
    p.add_argument("--judge-model", default="glm-4.6", help="Z.AI judge model")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    rows = build(args)
    if args.question_ids:
        wanted = [q.strip() for q in args.question_ids.read_text(encoding="utf-8").splitlines() if q.strip()]
        by_id = {r["question_id"]: r for r in rows}
        rows = [by_id[q] for q in wanted if q in by_id]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        with_gold = max(1, sum(bool(r["gold_evidence_ids"]) for r in rows))
        summary = {"arm": args.arm, "questions": len(rows), "planned_usage_proxy_tokens": planned_usage(rows),
                   "complete_coverage": sum(r["complete_coverage"] for r in rows) / with_gold,
                   "mean_context_proxy_tokens": sum(r["context_proxy_tokens"] for r in rows) / max(1, len(rows)),
                   "network_calls": 0}
        args.output.with_suffix(".plan-usage.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps(summary))
        return 0
    if not args.plan or (args.stage == "judge" and not args.answers):
        raise SystemExit("--execute requires --plan (and --answers for the judge stage)")
    return run_stage(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
