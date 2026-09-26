"""Faithful reproduction of LongMemEval's official flat-BM25 retrieval evaluation.

The corpus construction, BM25 scoring, ranking and metric functions below are copied from
github.com/xiaowu0162/LongMemEval (MIT License, Copyright (c) Di Wu et al.):
src/retrieval/run_retrieval.py (`process_item_flat_index`, `run_flat_retrieval` flat-bm25 branch)
and src/retrieval/eval_utils.py. Only I/O and aggregation are ours. Averages skip `_abs`
questions and questions without a user-side has_answer turn, exactly as the official script.
Zero network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ijson
import numpy as np
from rank_bm25 import BM25Okapi


# ---- verbatim from src/retrieval/run_retrieval.py (MIT) ----
def process_item_flat_index(data, granularity, sess_id, timestamp):
    corpus = []

    if granularity == 'session':
        text = ' '.join([interact['content'] for interact in data if interact['role'] == 'user'])
        corpus.append(text)
        ids = [sess_id]
        if 'answer' in sess_id and all([not turn['has_answer'] for turn in [x for x in data if x['role'] == 'user']]):
            ids = [sess_id.replace('answer', 'noans')]
    elif granularity == 'turn':
        ids = []
        for i_turn, turn in enumerate(data):
            if turn['role'] == 'user':
                corpus.append(turn['content'])
                if 'answer' not in sess_id:
                    ids.append(sess_id + '_' + str(i_turn+1))
                else:
                    assert 'has_answer' in turn
                    assert turn['has_answer'] in [True, False]
                    if turn['has_answer']:
                        ids.append(sess_id + '_' + str(i_turn+1))
                    else:
                        ids.append((sess_id + '_' + str(i_turn+1)).replace('answer', 'noans'))
                        assert 'answer' not in ids[-1]
    else:
        raise NotImplementedError

    return corpus, ids, [timestamp for _ in corpus]


def run_flat_bm25(query, corpus):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split(" "))
    return np.argsort(scores)[::-1]


# ---- verbatim from src/retrieval/eval_utils.py (MIT; np.asfarray -> np.asarray(float) for NumPy 2) ----
def dcg(relevances, k):
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.


def ndcg(rankings, correct_docs, corpus_ids, k=10):
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(sorted_relevances, k)
    if ideal_dcg == 0:
        return 0.
    return actual_dcg / ideal_dcg


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(doc in recalled_docs for doc in correct_docs))
    recall_all = float(all(doc in recalled_docs for doc in correct_docs))
    ndcg_score = ndcg(rankings, correct_docs, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


def evaluate_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=10):
    def strip_turn_id(docid):
        return '_'.join(docid.split('_')[:-1])
    correct_docs = list(set([strip_turn_id(x) for x in correct_docs]))
    corpus_ids = [strip_turn_id(x) for x in corpus_ids]
    effective_k = k
    unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])
    while effective_k <= len(corpus_ids) and len(unique_docids) < k:
        effective_k += 1
        unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])
    return evaluate_retrieval(rankings, correct_docs, corpus_ids, k=effective_k)
# ---- end verbatim ----


KS = (1, 3, 5, 10, 30, 50)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--granularity", choices=["session", "turn"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    per_q, ignored_abs, ignored_no_target = [], [], []
    with args.dataset.open("rb") as handle:
        for entry in ijson.items(handle, "item"):
            corpus, corpus_ids = [], []
            for sid, sess, ts in zip(entry["haystack_session_ids"], entry["haystack_sessions"], entry["haystack_dates"]):
                items, ids, _ = process_item_flat_index(sess, args.granularity, sid, ts)
                corpus += items
                corpus_ids += ids
            correct_docs = list(set([doc_id for doc_id in corpus_ids if "answer" in doc_id]))
            rankings = run_flat_bm25(entry["question"], corpus)
            metrics = {"session": {}, "turn": {}}
            for k in KS:
                ra, rl, nd = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
                metrics[args.granularity].update({f"recall_any@{k}": ra, f"recall_all@{k}": rl, f"ndcg_any@{k}": nd})
                if args.granularity == "turn":
                    ra, rl, nd = evaluate_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=k)
                    metrics["session"].update({f"recall_any@{k}": ra, f"recall_all@{k}": rl, f"ndcg_any@{k}": nd})
            qid = entry["question_id"]
            if "_abs" in qid:
                ignored_abs.append(qid)
            elif not any(t.get("has_answer") for s in entry["haystack_sessions"] for t in s if t["role"] == "user"):
                ignored_no_target.append(qid)
            per_q.append({"question_id": qid, "question_type": entry["question_type"], "metrics": metrics,
                          "scored": "_abs" not in qid and qid not in ignored_no_target})
    scored = [r for r in per_q if r["scored"]]
    summary = {g: {m: float(np.mean([r["metrics"][g][m] for r in scored])) for m in scored[0]["metrics"][g]}
               for g in ("session", "turn") if scored[0]["metrics"][g]}
    result = {"schema": "hybridmind.longmemeval_official_retrieval_repro.v1", "retriever": "flat-bm25",
              "granularity": args.granularity,
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "questions": len(per_q), "scored": len(scored), "ignored_abstention": len(ignored_abs),
              "ignored_no_user_target": sorted(ignored_no_target), "summary": summary, "per_question": per_q,
              "provenance": "official LongMemEval code (MIT) vendored verbatim; provider_calls=0"}
    args.output.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("granularity", "questions", "scored", "ignored_abstention")}
                     | {"ignored_no_user_target": len(ignored_no_target), "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
