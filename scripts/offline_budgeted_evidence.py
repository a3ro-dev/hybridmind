"""Budgeted evidence selection on LoCoMo: coverage at equal rendered-token cost.

Zero provider calls. Every strategy is charged for the exact context string a
reader would receive (session date + speaker + text + image caption), so a
method that "wins" by retrieving more text is caught by the budget, not hidden
by a Recall@k cut-off.

Primary metric: complete-evidence coverage (all gold turn IDs inside the packed
context). Secondary: fractional evidence recall, any-hit, catastrophic miss.
Paired deltas use a conversation-cluster bootstrap because questions share
histories. LoCoMo has been inspected before: results here are EXPLORATORY.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.bm25_index import BM25SBackend  # noqa: E402

DATASET = PROJECT_ROOT / "memorybench" / "data" / "benchmarks" / "locomo" / "locomo10.json"
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
_SESSION = re.compile(r"^session_(\d+)$")
_DIA = re.compile(r"D\d+:\d+")
_TOK = re.compile(r"\w+|[^\w\s]")  # declared token proxy (no tokenizer download)


def ntok(text: str) -> int:
    return len(_TOK.findall(text))


def load(dataset: Path) -> list[dict]:
    """Conversations as ordered turn lists plus scored questions."""
    convs = []
    for item in json.loads(dataset.read_text(encoding="utf-8")):
        sid, conv = item["sample_id"], item["conversation"]
        keys = sorted((k for k in conv if _SESSION.match(k)), key=lambda k: int(_SESSION.match(k).group(1)))
        turns = []
        for key in keys:
            s = int(_SESSION.match(key).group(1))
            date = str(conv.get(f"{key}_date_time") or "")
            for pos, m in enumerate(conv[key]):
                text = str(m.get("text") or "").strip()
                if not text:
                    continue
                cap = str(m.get("blip_caption") or "").strip()
                turns.append({
                    "id": str(m["dia_id"]), "session": s, "pos": pos, "date": date,
                    "speaker": str(m.get("speaker") or ""), "text": text, "caption": cap,
                })
        ids = {t["id"] for t in turns}
        qs = []
        for qa in item["qa"]:
            raw = [str(e) for e in qa.get("evidence", [])]
            if any(_DIA.sub("", e).strip(" ;,\t\r\n") or not _DIA.search(e) for e in raw):
                continue  # malformed annotation: excluded, counted below
            gold = sorted({g for e in raw for g in _DIA.findall(e)})
            if not gold or not set(gold) <= ids:
                continue
            qs.append({"q": str(qa["question"]), "gold": gold, "cat": CATEGORY[qa["category"]]})
        convs.append({"sid": sid, "turns": turns, "qs": qs, "n_qa": len(item["qa"])})
    return convs


def render(t: dict) -> str:
    cap = f" [shares image: {t['caption']}]" if t["caption"] else ""
    return f"({t['date']}) {t['speaker']}: {t['text']}{cap}"


def index_text(t: dict, rep: str) -> str:
    if rep == "raw":
        return t["text"]
    if rep == "spk":
        return f"{t['speaker']}: {t['text']}"
    if rep == "spk_cap":
        return f"{t['speaker']}: {t['text']} {t['caption']}".strip()
    raise ValueError(rep)


class Conv:
    def __init__(self, conv: dict, rep: str):
        self.turns = conv["turns"]
        self.by_id = {t["id"]: i for i, t in enumerate(self.turns)}
        self.cost = [ntok(render(t)) for t in self.turns]
        self.bm = BM25SBackend()
        self.bm.add_batch([(t["id"], index_text(t, rep)) for t in self.turns])
        sessions = defaultdict(list)
        for i, t in enumerate(self.turns):
            sessions[t["session"]].append(i)
        self.sessions = dict(sessions)
        self.sbm = BM25SBackend()
        self.sbm.add_batch([(str(s), " ".join(index_text(self.turns[i], rep) for i in ix)) for s, ix in self.sessions.items()])

    # BM25S tie order depends on the requested k (47/1,977 LoCoMo top-10 sets
    # differ between k=10 and k=all), so ties break chronologically here.
    @functools.lru_cache(maxsize=4)
    def ranked_turns(self, q: str) -> list[int]:
        hits = self.bm.search(q, top_k=len(self.turns))
        return [self.by_id[i] for i, _ in sorted(hits, key=lambda h: (-h[1], self.by_id[h[0]]))]

    def ranked_sessions(self, q: str) -> list[int]:
        hits = self.sbm.search(q, top_k=len(self.sessions))
        return [int(sid) for sid, _ in sorted(hits, key=lambda h: (-h[1], int(h[0])))]


def pack(units: list[list[int]], cost: list[int], budget: int) -> list[int]:
    """Greedy in rank order; a unit that does not fit is skipped, not truncated."""
    chosen, seen, used = [], set(), 0
    for unit in units:
        new = [i for i in unit if i not in seen]
        c = sum(cost[i] for i in new)
        if not new or used + c > budget:
            continue
        chosen += new
        seen.update(new)
        used += c
    return chosen


def strategy_units(name: str, conv: Conv, q: str) -> list[list[int]]:
    if name == "turn":
        return [[i] for i in conv.ranked_turns(q)]
    if name.startswith("nbr"):  # nbr1 = hit plus +/-1 turns in the same session
        w = int(name[3:])
        units = []
        for i in conv.ranked_turns(q):
            s = conv.turns[i]["session"]
            units.append([j for j in range(i - w, i + w + 1)
                          if 0 <= j < len(conv.turns) and conv.turns[j]["session"] == s])
        return units
    if name.startswith("next"):  # hit plus the following w turns (reply direction)
        w = int(name[4:])
        units = []
        for i in conv.ranked_turns(q):
            s = conv.turns[i]["session"]
            units.append([j for j in range(i, i + w + 1)
                          if j < len(conv.turns) and conv.turns[j]["session"] == s])
        return units
    if name == "session":
        return [conv.sessions[s] for s in conv.ranked_sessions(q)]
    if name.startswith("cap"):  # at most m turns per session on the first pass, then the rest
        m = int(name[3:])
        ranked, per, first, rest = conv.ranked_turns(q), defaultdict(int), [], []
        for i in ranked:
            s = conv.turns[i]["session"]
            (first if per[s] < m else rest).append([i])
            per[s] += 1
        return first + rest
    raise ValueError(name)


def cluster_bootstrap(rows: list[tuple[str, float]], seed: int, samples: int = 4000) -> dict:
    """Mean over questions; resample conversations (clusters) with replacement."""
    by = defaultdict(list)
    for c, v in rows:
        by[c].append(v)
    keys = sorted(by)
    sums = np.array([sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(samples, len(keys)))
    est = np.sort(sums[idx].sum(1) / cnts[idx].sum(1))
    return {"mean": float(sums.sum() / cnts.sum()), "lo": float(est[int(0.025 * (samples - 1))]),
            "hi": float(est[math.ceil(0.975 * (samples - 1))]), "n": int(cnts.sum()), "clusters": len(keys)}


def run(args) -> dict:
    t0 = time.perf_counter()
    convs = load(args.dataset)
    strategies = args.strategies.split(",")
    budgets = [int(b) for b in args.budgets.split(",")]
    rows = []  # one per (question, rep, strategy, budget)
    for rep in args.reps.split(","):
        for conv in convs:
            c = Conv(conv, rep)
            for qi, qa in enumerate(conv["qs"]):
                gold = {c.by_id[g] for g in qa["gold"]}
                for st in strategies:
                    units = strategy_units(st, c, qa["q"])
                    for b in budgets:
                        sel = set(pack(units, c.cost, b))
                        hit = len(gold & sel)
                        rows.append({
                            "conv": conv["sid"], "qi": qi, "cat": qa["cat"], "rep": rep, "strategy": st,
                            "budget": b, "n_gold": len(gold), "hit": hit,
                            "tokens": sum(c.cost[i] for i in sel),
                        })
    summary = defaultdict(dict)
    base_key = (args.baseline_rep, args.baseline_strategy)
    index = {(r["rep"], r["strategy"], r["budget"], r["conv"], r["qi"]): r for r in rows}
    for rep in args.reps.split(","):
        for st in strategies:
            for b in budgets:
                sel = [r for r in rows if r["rep"] == rep and r["strategy"] == st and r["budget"] == b]
                for cat in ["all", "multi-hop", "temporal", "open-domain", "single-hop", "adversarial"]:
                    part = [r for r in sel if cat == "all" or r["cat"] == cat]
                    if not part:
                        continue
                    allhit = [(r["conv"], float(r["hit"] == r["n_gold"])) for r in part]
                    base = [index[(*base_key, b, r["conv"], r["qi"])] for r in part]
                    delta = [(r["conv"], float(r["hit"] == r["n_gold"]) - float(x["hit"] == x["n_gold"]))
                             for r, x in zip(part, base)]
                    summary[f"{rep}|{st}|{b}"][cat] = {
                        "n": len(part),
                        "complete_coverage": cluster_bootstrap(allhit, args.seed),
                        "recall": float(np.mean([r["hit"] / r["n_gold"] for r in part])),
                        "any_hit": float(np.mean([r["hit"] > 0 for r in part])),
                        "catastrophic_miss": float(np.mean([r["hit"] == 0 for r in part])),
                        "mean_tokens": float(np.mean([r["tokens"] for r in part])),
                        "delta_complete_vs_baseline": cluster_bootstrap(delta, args.seed + 1),
                    }
    return {
        "schema": "hybridmind.offline_budgeted_evidence.v1",
        "status": "exploratory",
        "dataset": {"path": str(args.dataset.relative_to(PROJECT_ROOT)),
                    "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    "questions_total": sum(c["n_qa"] for c in convs),
                    "questions_scored": sum(len(c["qs"]) for c in convs)},
        "token_proxy": _TOK.pattern,
        "render_format": "(session date) speaker: text [shares image: caption]",
        "baseline": {"rep": args.baseline_rep, "strategy": args.baseline_strategy},
        "config": vars(args) | {"dataset": str(args.dataset)},
        "provider_calls": 0,
        "wall_seconds": time.perf_counter() - t0,
        "summary": summary,
        "rows": rows if args.keep_rows else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--reps", default="raw,spk,spk_cap")
    p.add_argument("--strategies", default="turn,nbr1,next1,cap2,session")
    p.add_argument("--budgets", default="256,512,1024,2048,4096")
    p.add_argument("--baseline-rep", default="raw")
    p.add_argument("--baseline-strategy", default="turn")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--keep-rows", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    for key, cats in result["summary"].items():
        a = cats["all"]
        cc, d = a["complete_coverage"], a["delta_complete_vs_baseline"]
        print(f"{key:28s} complete={cc['mean']:.3f} [{cc['lo']:.3f},{cc['hi']:.3f}] "
              f"recall={a['recall']:.3f} tok={a['mean_tokens']:.0f} d={d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")


if __name__ == "__main__":
    main()
