"""Offline checks for the plan-bound reader/judge stages (provider mocked, zero calls)."""

import argparse
import json
from pathlib import Path

from scripts import budgeted_answer_eval as bae


def _rows(n):
    return [{"question_id": f"q{i}", "category": "single-session-user", "abstention": False, "question": "Q?",
             "gold_answer": "A", "arm": "raw|turn|4096", "packed_evidence_ids": [], "gold_evidence_ids": [],
             "complete_coverage": False, "any_hit": False, "context_proxy_tokens": 0,
             "reader_prompt": "history ... Question: Q?", "reader_prompt_proxy_tokens": 5,
             "judge_template": bae.JUDGE["default"].format("Q?", "A", bae.RESPONSE_SLOT)} for i in range(n)]


def _args(tmp_path: Path, stage: str, output: str, answers=None):
    data = tmp_path / "d.json"
    data.write_text("[]")
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    return argparse.Namespace(stage=stage, plan=plan, answers=answers, dataset=data, arm="raw|turn|4096",
                              kind="longmemeval", output=tmp_path / output, model="m", judge_model="j", workers=2)


def _gate(monkeypatch, llm_calls):
    ceiling = {"llm_calls": llm_calls, "reader_input_tokens": 10**6, "reader_output_tokens": 10**6}
    monkeypatch.setattr("engine.resource_accounting.load_and_validate_live_plan",
                        lambda p: ({"providers": ["zai"], "usage_ceiling": ceiling}, argparse.Namespace(ok=True)))


def test_reader_then_judge_labels_and_stops_at_ceiling(tmp_path, monkeypatch):
    calls = []

    def fake(messages, **kw):
        calls.append(kw["model"])
        kw["usage"].update(prompt_tokens=7, completion_tokens=3)
        return "yes" if kw["model"] == "j" else "A because ..."

    monkeypatch.setattr("engine.llm_client.chat_completion", fake)
    _gate(monkeypatch, llm_calls=3)
    reader = _args(tmp_path, "reader", "r.jsonl")
    assert bae.run_stage(reader, _rows(5)) == 3  # ceiling of 3 calls stops the pass
    assert len(bae._ledger_ok_ids(reader.output)) == 3
    _gate(monkeypatch, llm_calls=100)
    judge = _args(tmp_path, "judge", "j.jsonl", answers=reader.output)
    assert bae.run_stage(judge, _rows(5)) == 0
    judged = [json.loads(line) for line in judge.output.read_text().splitlines()]
    assert len(judged) == 3 and all(r["label"] is True for r in judged)
    assert calls.count("j") == 3 and calls.count("m") == 3


def test_ceiling_is_cumulative_across_invocations(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.llm_client.chat_completion", lambda m, **kw: "answer")
    _gate(monkeypatch, llm_calls=3)
    first = _args(tmp_path, "reader", "arm1.jsonl")
    assert bae.run_stage(first, _rows(2)) == 0
    second = _args(tmp_path, "reader", "arm2.jsonl")  # same plan, different arm ledger
    assert bae.run_stage(second, _rows(2)) == 3  # only 1 call left under the shared plan
    assert len(bae._ledger_ok_ids(second.output)) == 1


def test_provider_failure_stops_and_resume_skips_ok_rows(tmp_path, monkeypatch):
    outcomes = iter(["ok answer", None])
    monkeypatch.setattr("engine.llm_client.chat_completion", lambda m, **kw: next(outcomes, "late"))
    _gate(monkeypatch, llm_calls=100)
    args = _args(tmp_path, "reader", "r.jsonl")
    args.workers = 1
    assert bae.run_stage(args, _rows(4)) == 3  # stopped on the failed row
    receipt = json.loads(args.output.with_suffix(".completion.json").read_text())
    assert receipt["status"] == "provider_failure" and receipt["ok_rows"] == 1
    monkeypatch.setattr("engine.llm_client.chat_completion", lambda m, **kw: "resumed")
    assert bae.run_stage(args, _rows(4)) == 0  # resume answers only the 3 unfinished questions
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert sum(r["reader_answer"] == "resumed" for r in rows) == 3
