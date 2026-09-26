"""Conversational benchmark loaders and metrics: hand-computed fixtures, zero network."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from benchmarks import conversational_data as data
from benchmarks.conversational_data import (
    load_locomo,
    load_longmemeval,
    longmemeval_official_turn_view,
    parse_locomo_datetime,
    parse_longmemeval_datetime,
    split_locomo_evidence,
)
from benchmarks.conversational_metrics import (
    MetricRow,
    bootstrap,
    budgeted_coverage,
    by_category,
    complementarity,
    fusion_regret,
    locomo_recall_acc,
    locomo_recall_official,
    macro_mean,
    scoring_gold,
    session_metrics_at_k,
    turn2session_metrics_at_k,
    turn_metrics_at_k,
)

ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "memorybench/data/benchmarks/locomo/locomo10.json"
EMG_LOCOMO = ROOT / "tmp/upstream/em_graph_memory/data/locomo10.json"
LME_ORACLE = ROOT / "memorybench/data/benchmarks/longmemeval/longmemeval_s.json"


# ---------------------------------------------------------------- LoCoMo official recall


def test_locomo_recall_acc_matches_official_row_logic() -> None:
    assert locomo_recall_acc(["D1:1", "D2:3"], ["D1:1", "D1:2"]) == 0.5
    assert locomo_recall_acc(["D1:1"], []) == 1.0  # official: rows without evidence score 1
    assert locomo_recall_acc(["S1", "S3"], ["D1:5", "D2:1", "D3:9"]) == pytest.approx(2 / 3)
    assert locomo_recall_acc([], ["D1:1"]) == 0.0
    # A raw malformed string never matches, exactly as upstream.
    assert locomo_recall_acc(["D8:6", "D9:17"], ["D8:6; D9:17"]) == 0.0


def test_locomo_recall_official_aggregate_counts_empty_rows_in_denominator() -> None:
    result = locomo_recall_official([
        (1, ["D1:1", "D1:2"], ["D1:1"]),  # 0.5
        (1, [], ["D1:1"]),  # denominator only
        (2, ["D1:1", "D1:2", "D1:3"], ["D1:1"]),  # round(1/3, 3) = 0.333
    ])
    assert result["by_category"] == {1: 0.25, 2: pytest.approx(0.333)}
    assert result["overall"] == pytest.approx((0.5 + 0.333) / 3)
    assert result["n"] == 3 and result["n_by_category"] == {1: 2, 2: 1}


# ---------------------------------------------------------------- LongMemEval official metrics

CORPUS = ["a", "b", "c", "d", "e"]


def test_turn_metrics_hand_computed() -> None:
    # top2 = c, a; relevances c=0, a=1 -> DCG 1/log2(2) = 1; ideal [1, 1] -> 2.
    assert turn_metrics_at_k(["c", "a", "e"], ["a", "d"], CORPUS, 2) == {
        "recall_any": 1.0, "recall_all": 0.0, "ndcg_any": 0.5}
    assert turn_metrics_at_k(["c", "a", "e"], ["a", "d"], CORPUS, 3)["ndcg_any"] == 0.5
    assert turn_metrics_at_k(["a", "d"], ["a", "d"], CORPUS, 1) == {
        "recall_any": 1.0, "recall_all": 0.0, "ndcg_any": 1.0}
    assert turn_metrics_at_k(["d", "a"], ["a", "d"], CORPUS, 2)["recall_all"] == 1.0


@pytest.mark.parametrize("ranked,gold,corpus", [
    (["a"], [], CORPUS),  # empty gold
    (["z"], ["a"], CORPUS),  # ranked id outside corpus
    (["a", "a"], ["a"], CORPUS),  # duplicate ranked id
    (["a"], ["z"], CORPUS),  # gold outside corpus
    (["a"], ["a"], ["a", "a"]),  # duplicate corpus id
])
def test_turn_metrics_fail_closed(ranked, gold, corpus) -> None:
    with pytest.raises(ValueError):
        turn_metrics_at_k(ranked, gold, corpus, 1)


TURNS = ["t1", "t2", "t3", "t4", "t5", "t6"]
SESSION_OF = {"t1": "sess_a", "t2": "sess_a", "t3": "sess_b", "t4": "sess_b", "t5": "sess_c", "t6": "sess_c"}


def test_turn2session_grows_effective_k_to_unique_sessions() -> None:
    ranked = ["t1", "t2", "t3", "t5"]
    # k=2: top2 spans 1 session -> effective k 3 spans {a, b}; gold session c missed.
    assert turn2session_metrics_at_k(ranked, ["t5"], TURNS, SESSION_OF, 2) == {
        "recall_any": 0.0, "recall_all": 0.0, "ndcg_any": 0.0}
    # k=3: effective k 4 reaches sess_c at rank 4 -> DCG 1/log2(4) = 0.5; every sess_c turn
    # is relevant in the official turn2session NDCG, so ideal = 1 + 1 = 2.
    assert turn2session_metrics_at_k(ranked, ["t5"], TURNS, SESSION_OF, 3) == {
        "recall_any": 1.0, "recall_all": 1.0, "ndcg_any": 0.25}
    # Plain top-3 turns do not reach sess_c: the growth is what differs.
    assert session_metrics_at_k(ranked, SESSION_OF, ["sess_c"], 3)["session_hit"] == 0.0


def test_session_metrics_hand_computed() -> None:
    ranked = ["t1", "t3", "t5"]
    assert session_metrics_at_k(ranked, SESSION_OF, ["sess_b", "sess_c"], 2) == {
        "session_hit": 1.0, "session_recall_all": 0.0}
    assert session_metrics_at_k(ranked, SESSION_OF, ["sess_b", "sess_c"], 3) == {
        "session_hit": 1.0, "session_recall_all": 1.0}
    with pytest.raises(ValueError):
        session_metrics_at_k(ranked, SESSION_OF, [], 3)


# ---------------------------------------------------------------- budgeted / channel analysis


def test_budgeted_coverage() -> None:
    assert budgeted_coverage(["a", "b"], ["a", "c"]) == {"complete_coverage": 0.0, "recall": 0.5, "any_hit": 1.0}
    assert budgeted_coverage(["a", "c", "x"], ["a", "c"]) == {"complete_coverage": 1.0, "recall": 1.0, "any_hit": 1.0}
    assert budgeted_coverage([], ["a"]) == {"complete_coverage": 0.0, "recall": 0.0, "any_hit": 0.0}
    with pytest.raises(ValueError):
        budgeted_coverage(["a"], [])


CHANNELS = {"sparse": ["b", "d", "e"], "dense": ["a", "b", "c"], "graph": ["f"]}
GOLD = ["a", "d", "g"]


def test_complementarity_hand_computed() -> None:
    result = complementarity(CHANNELS, GOLD, 2)
    assert result["hits"] == {"dense": ["a"], "graph": [], "sparse": ["d"]}
    assert result["unique_hits"] == {"dense": ["a"], "graph": [], "sparse": ["d"]}
    # dense {a, b} vs sparse {b, d}: 1 / 3.
    assert result["pairwise_jaccard"] == {"dense|graph": 0.0, "dense|sparse": pytest.approx(1 / 3), "graph|sparse": 0.0}
    assert result["oracle_union_recall_all"] == 0.0
    assert result["oracle_union_recall_any"] == 1.0
    assert result["oracle_union_recall"] == pytest.approx(2 / 3)
    assert complementarity({"x": [], "y": []}, ["a"], 5)["pairwise_jaccard"] == {"x|y": 1.0}


def test_fusion_regret_hand_computed() -> None:
    result = fusion_regret(["b", "a", "x"], CHANNELS, GOLD, 2)
    assert result["fused_recall"] == pytest.approx(1 / 3)
    assert result["channel_recall"] == {"dense": pytest.approx(1 / 3), "graph": 0.0, "sparse": pytest.approx(1 / 3)}
    assert result["best_channel"] == "dense"  # tie with sparse breaks by name
    assert result["regret_vs_best_channel"] == 0.0
    assert result["oracle_union_recall"] == pytest.approx(2 / 3)
    assert result["regret_vs_oracle_union"] == pytest.approx(1 / 3)
    assert result["lost_gold"] == ["d"]
    assert fusion_regret(["a", "d"], CHANNELS, GOLD, 2)["regret_vs_best_channel"] == pytest.approx(-1 / 3)


# ---------------------------------------------------------------- aggregation

ROWS = [
    MetricRow("q1", "c1", "A", {"m": 1.0}),
    MetricRow("q2", "c1", "A", {"m": 0.0}),
    MetricRow("q3", "c2", "B", {"m": 1.0}),
    MetricRow("q4", "c2", "B", skip_reason="empty_gold"),
    MetricRow("q5", "c3", "A", skip_reason="abstention"),
]


def test_macro_mean_and_by_category_report_denominators() -> None:
    assert macro_mean(ROWS, "m") == {"mean": pytest.approx(2 / 3), "n": 3, "n_total": 5, "n_skipped": 2,
                                     "skipped": {"abstention": 1, "empty_gold": 1}}
    cats = by_category(ROWS, "m")
    assert list(cats) == ["all", "A", "B"]
    assert cats["A"]["mean"] == 0.5 and cats["A"]["skipped"] == {"abstention": 1}
    assert cats["B"] == {"mean": 1.0, "n": 1, "n_total": 2, "n_skipped": 1, "skipped": {"empty_gold": 1}}
    assert macro_mean([ROWS[3]], "m")["mean"] is None
    with pytest.raises(KeyError):
        macro_mean([MetricRow("q", "c", "A", {"other": 1.0})], "m")


def test_bootstrap_question_and_cluster() -> None:
    q = bootstrap(ROWS, "m", seed=7, n_resamples=500)
    assert q["mean"] == pytest.approx(2 / 3) and q["n"] == 3 and q["clusters"] == 2
    assert q["ci_lo"] <= q["mean"] <= q["ci_hi"] and q["n_skipped"] == 2
    assert q == bootstrap(ROWS, "m", seed=7, n_resamples=500)  # seeded
    c = bootstrap(ROWS, "m", cluster=True, seed=7, n_resamples=500)
    assert c["method"] == "conversation-cluster" and c["mean"] == pytest.approx(2 / 3) and c["clusters"] == 2
    assert 0.0 <= c["ci_lo"] <= c["ci_hi"] <= 1.0


def test_paired_bootstrap() -> None:
    arm = [MetricRow(f"q{i}", f"c{i % 2}", "A", {"m": 1.0}) for i in range(6)]
    base = [MetricRow(f"q{i}", f"c{i % 2}", "A", {"m": 0.0}) for i in range(5)]
    base.append(MetricRow("q5", "c1", "A", skip_reason="retrieval_failed"))
    for cluster in (False, True):
        result = bootstrap(arm, "m", baseline=base, cluster=cluster, n_resamples=200)
        assert (result["mean"], result["ci_lo"], result["ci_hi"], result["n"]) == (1.0, 1.0, 1.0, 5)
        assert result["paired"] and result["skipped"] == {"retrieval_failed": 1}
    with pytest.raises(ValueError):
        bootstrap(arm, "m", baseline=base[:-1])  # different qid sets
    with pytest.raises(ValueError):
        bootstrap(arm[:1], "m", baseline=[MetricRow("q0", "other", "A", {"m": 0.0})])


# ---------------------------------------------------------------- data: parsing helpers


def test_date_parsers() -> None:
    assert parse_locomo_datetime("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"
    assert parse_locomo_datetime("12:09 am on 13 September, 2023") == "2023-09-13T00:09:00"
    assert parse_longmemeval_datetime("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00"
    with pytest.raises(ValueError):
        parse_locomo_datetime("May 8, 2023")


def test_split_locomo_evidence() -> None:
    raw = ["D8:6; D9:17", "D", "D1:1", "D:11:26", "D30:05", "D1:1", "D9:1 D4:4"]
    known = {"D8:6", "D9:17", "D1:1", "D9:1", "D4:4", "D30:5"}
    assert split_locomo_evidence(raw, known) == (
        ("D8:6", "D9:17", "D1:1", "D9:1", "D4:4"),
        ("D", "D:11:26", "D30:05"),
    )


# ---------------------------------------------------------------- data: LongMemEval


def _lme_item(qid, answer, session_ids, dates, sessions, answer_sessions) -> dict:
    return {"question_id": qid, "question_type": "multi-session", "question": "What pet did I adopt?",
            "question_date": "2023/05/30 (Tue) 23:40", "answer": answer, "answer_session_ids": answer_sessions,
            "haystack_session_ids": session_ids, "haystack_dates": dates, "haystack_sessions": sessions}


@pytest.fixture()
def lme_file(tmp_path: Path) -> Path:
    small = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    answer = [
        {"role": "user", "content": "I adopted a cat", "has_answer": True},
        {"role": "assistant", "content": "Nice cat", "has_answer": True},
        {"role": "user", "content": "bye", "has_answer": False},
    ]
    items = [
        _lme_item("q1", 42, ["s1", "answer_s2", "s1"],
                  ["2023/05/21 (Sun) 10:00", "2023/05/20 (Sat) 09:00", "2023/05/22 (Mon) 08:00"],
                  [small, answer, small], ["answer_s2"]),
        _lme_item("q2_abs", "none", ["answer_s3"], ["2023/05/20 (Sat) 09:00"], [answer], ["answer_s3"]),
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_load_longmemeval_synthetic(lme_file: Path) -> None:
    conv, abstain = list(load_longmemeval(lme_file))
    assert conv.scope_key == "lme:q1" and len(conv.corpus) == 7
    # Chronological: answer_s2 (05-20) < s1 (05-21) < repeated s1 (05-22).
    assert [t.node_id for t in conv.corpus.turns] == [
        "lme:q1:answer_s2:0", "lme:q1:answer_s2:1", "lme:q1:answer_s2:2",
        "lme:q1:s1:0", "lme:q1:s1:1", "lme:q1:s1@2:0", "lme:q1:s1@2:1"]
    first = conv.corpus.turns[0]
    assert (first.evidence_id, first.session_key, first.speaker, first.order) == (
        "answer_s2:0", "answer_s2", "user", ("2023-05-20T09:00:00", 1, 0))
    assert first.metadata == {"has_answer": True, "role": "user", "session_id": "answer_s2",
                              "session_position": 1, "turn_index": 0}
    assert conv.corpus.turn("lme:q1:s1@2:0").metadata["session_id"] == "s1"
    q = conv.questions[0]
    assert (q.qid, q.answer, q.category, q.abstention) == ("q1", "42", "multi-session", False)
    assert q.gold_evidence == ("answer_s2:0", "answer_s2:1")
    assert q.gold_user_evidence == ("answer_s2:0",)
    assert q.gold_sessions == ("answer_s2",)
    assert q.question_date == "2023-05-30T23:40:00"
    assert abstain.questions[0].abstention is True


def test_load_longmemeval_limit_and_filter(lme_file: Path) -> None:
    assert [c.scope_key for c in load_longmemeval(lme_file, limit=1)] == ["lme:q1"]
    assert list(load_longmemeval(lme_file, limit=0)) == []
    assert [c.scope_key for c in load_longmemeval(lme_file, question_ids=["q2_abs"])] == ["lme:q2_abs"]
    with pytest.raises(ValueError, match="missing_q"):
        list(load_longmemeval(lme_file, question_ids=["q1", "missing_q"]))


def test_official_turn_view_relabels_noans_turns(lme_file: Path) -> None:
    conv, abstain = list(load_longmemeval(lme_file))
    ids, session_of = longmemeval_official_turn_view(conv.corpus)
    assert ids == ["answer_s2:0", "answer_s2:2", "s1:0", "s1@2:0"]  # user turns only
    assert session_of == {"answer_s2:0": "answer_s2", "answer_s2:2": "noans_s2", "s1:0": "s1", "s1@2:0": "s1"}
    gold, skip = scoring_gold(conv.questions[0], "lme_official")
    assert (gold, skip) == (("answer_s2:0",), None)
    ranked = ["answer_s2:2", "s1:0", "answer_s2:0"]
    # A non-evidence turn of the answer session is not a session hit (official 'noans' label).
    assert turn2session_metrics_at_k(ranked, gold, ids, session_of, 2)["recall_any"] == 0.0
    three = turn2session_metrics_at_k(ranked, gold, ids, session_of, 3)
    assert (three["recall_all"], three["ndcg_any"]) == (1.0, pytest.approx(1 / math.log2(3)))
    assert scoring_gold(conv.questions[0], "lme_any_role") == (("answer_s2:0", "answer_s2:1"), None)
    assert scoring_gold(abstain.questions[0], "lme_official") == ((), "abstention")


def test_oracle_file_is_refused(lme_file: Path, monkeypatch) -> None:
    size, digest = lme_file.stat().st_size, hashlib.sha256(lme_file.read_bytes()).hexdigest()
    monkeypatch.setitem(data.LONGMEMEVAL_S_ORACLE_INVALID, "bytes", size)
    with pytest.raises(ValueError, match="oracle"):
        load_longmemeval(lme_file)  # refused at call time, before iteration
    monkeypatch.setitem(data.LONGMEMEVAL_S_ORACLE_INVALID, "sha256", digest)
    with pytest.raises(ValueError, match="oracle"):
        data.dataset_identity(lme_file)


@pytest.mark.skipif(not LME_ORACLE.is_file(), reason="longmemeval_s.json not present")
def test_real_oracle_file_is_refused() -> None:
    with pytest.raises(ValueError, match="oracle"):
        data.dataset_identity(LME_ORACLE)
    with pytest.raises(ValueError, match="oracle"):
        next(load_longmemeval(LME_ORACLE))


# ---------------------------------------------------------------- data: LoCoMo


def test_load_locomo_synthetic(tmp_path: Path) -> None:
    item = {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Ann", "speaker_b": "Bo",
            "session_2_date_time": "9:00 am on 2 May, 2023",
            "session_2": [{"speaker": "Bo", "dia_id": "D2:1", "text": "later"}],
            "session_1_date_time": "1:56 pm on 1 May, 2023",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": " Look! "},
                {"speaker": "Bo", "dia_id": "D1:2", "text": "Nice", "blip_caption": "a photo of a dog",
                 "img_url": ["http://x/y.jpg"]},
            ],
        },
        "qa": [
            {"question": "What?", "answer": "dog", "evidence": ["D1:2; D2:1"], "category": 4},
            {"question": "Why?", "adversarial_answer": "trap", "evidence": ["D9:9"], "category": 5},
        ],
    }
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([item]), encoding="utf-8")
    (conv,) = load_locomo(path)
    assert [t.node_id for t in conv.corpus.turns] == ["locomo:conv-x:D1:1", "locomo:conv-x:D1:2", "locomo:conv-x:D2:1"]
    ann, bo = conv.corpus.turns[:2]
    assert ann.text == " Look! " and ann.index_text == "Ann: Look!"
    assert bo.index_text == "Bo: Nice a photo of a dog"
    assert bo.session_key == "conv-x:S1" and bo.order == ("2023-05-01T13:56:00", 1, 1)
    assert bo.metadata["img_url"] == ["http://x/y.jpg"] and bo.metadata["turn_index"] == 1
    q4, q5 = conv.questions
    assert (q4.qid, q4.category_name, q4.gold_evidence, q4.gold_sessions) == (
        "locomo:conv-x:0", "single-hop", ("D1:2", "D2:1"), ("conv-x:S1", "conv-x:S2"))
    assert q4.question_date == "2023-05-02T09:00:00" and q4.gold_user_evidence == ()
    assert (q5.answer, q5.abstention, q5.gold_evidence, q5.malformed_evidence) == ("trap", True, (), ("D9:9",))
    assert scoring_gold(q4, "locomo") == (("D1:2", "D2:1"), None)
    assert scoring_gold(q5, "locomo") == ((), "no_gold_evidence")
    assert scoring_gold(q5, "locomo_audited") == ((), "malformed_evidence")
    with pytest.raises(ValueError, match="protocol"):
        scoring_gold(q4, "official")
    del item["conversation"]["session_2_date_time"]
    path.write_text(json.dumps([item]), encoding="utf-8")
    with pytest.raises(ValueError, match="date_time"):
        load_locomo(path)


@pytest.mark.skipif(not LOCOMO.is_file(), reason="locomo10.json not present")
def test_locomo10_real_file_census() -> None:
    convs = load_locomo(LOCOMO)
    questions = [q for c in convs for q in c.questions]
    assert len(convs) == 10
    assert sum(len(c.corpus) for c in convs) == 5882
    assert len(questions) == 1986
    assert Counter(q.category for q in questions) == {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
    malformed = {q.qid: q.malformed_evidence for q in questions if q.malformed_evidence}
    assert malformed == {
        "locomo:conv-42:58": ("D10:19",), "locomo:conv-42:88": ("D",), "locomo:conv-43:18": ("D:11:26",),
        "locomo:conv-47:38": ("D4:36",), "locomo:conv-50:69": ("D30:05",)}
    assert sum(not q.gold_evidence for q in questions) == 5  # 4 unannotated + conv-50:69
    # E1 audited denominator: non-empty gold and no malformed/unresolved annotation.
    assert sum(bool(q.gold_evidence) and not q.malformed_evidence for q in questions) == 1977
    for protocol, scored in (("locomo", 1981), ("locomo_audited", 1977)):
        assert sum(scoring_gold(q, protocol)[1] is None for q in questions) == scored
    split = next(q for q in questions if q.qid == "locomo:conv-26:37")
    assert split.gold_evidence == ("D8:6", "D9:17")
    for conv in convs:
        assert all(g in {t.evidence_id for t in conv.corpus.turns} for q in conv.questions for g in q.gold_evidence)
    caption_turns = [t for c in convs for t in c.corpus.turns if t.metadata["blip_caption"]]
    assert len(caption_turns) == 1226
    t = caption_turns[0]
    assert t.index_text == f"{t.speaker}: {t.text.strip()} {t.metadata['blip_caption']}"
    identity = data.dataset_identity(LOCOMO)
    assert identity["known"] == "locomo10" and identity["bytes"] == data.LOCOMO10["bytes"]
    # Official recall with the whole conversation as context: only malformed strings miss.
    rows = [(q.category, list(raw), [t.evidence_id for t in c.corpus.turns])
            for c, item in zip(convs, json.loads(LOCOMO.read_text(encoding="utf-8")))
            for q, raw in zip(c.questions, (qa.get("evidence") or [] for qa in item["qa"]))]
    assert 0.99 < locomo_recall_official(rows)["overall"] < 1.0
    assert not math.isnan(locomo_recall_official(rows)["overall"])


@pytest.mark.skipif(not (LOCOMO.is_file() and EMG_LOCOMO.is_file()), reason="EMG LoCoMo copy not present")
def test_emg_locomo_copy_loads_identically() -> None:
    assert data.dataset_identity(EMG_LOCOMO)["known"] == "emg_locomo10"
    for ours, emg in zip(load_locomo(LOCOMO), load_locomo(EMG_LOCOMO)):
        assert ours.corpus.turns == emg.corpus.turns and ours.questions == emg.questions
