"""Score-based fusion ports and budgeted evidence assembly (zero provider calls)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.corpus import ScopedCorpus, TurnRecord
from engine.evidence import (
    assemble_evidence,
    pack,
    propagate_scores,
    rank_sessions,
    render_turn,
    session_units,
    token_count,
    window_units,
)
from engine.fusion import (
    EMG_LINEAR_WEIGHTS,
    dbsf_fuse,
    minmax_normalize,
    rrf_fuse,
    weighted_linear_fuse,
    zscore_fuse,
)

# ── DBSF (Qdrant score_fusion.rs) ─────────────────────────────────────────


def test_dbsf_matches_hand_computed_values():
    # list a: mean 2, sample variance 1 -> range [-1, 5]; list b is a singleton -> 0.5.
    fused = dbsf_fuse({"a": {"x": 1.0, "y": 2.0, "z": 3.0}, "b": {"y": 10.0}}, {"a": 1.0, "b": 2.0})
    assert fused == pytest.approx({"x": 1 / 3, "y": 0.5 + 2 * 0.5, "z": 2 / 3})


def test_dbsf_zero_variance_and_singleton_map_to_half():
    assert dbsf_fuse({"a": {"p": 4.0, "q": 4.0, "r": 4.0}}) == {"p": 0.5, "q": 0.5, "r": 0.5}
    assert dbsf_fuse({"a": {"p": -7.0}}) == {"p": 0.5}
    assert dbsf_fuse({"a": {}, "b": {"p": 1.0}}) == {"p": 0.5}


def test_dbsf_is_unclipped_and_missing_contributes_zero():
    values = [0.0] * 10 + [100.0]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    scores = {f"n{i:02d}": v for i, v in enumerate(values)}
    fused = dbsf_fuse({"a": scores, "b": {"n10": 1.0, "other": 3.0}})
    top = (100.0 - (mean - 3 * std)) / (6 * std)
    assert top > 1.0  # not clipped to [0, 1]
    b = dbsf_fuse({"b": {"n10": 1.0, "other": 3.0}})  # mean 2, sample std sqrt(2)
    assert b["n10"] == pytest.approx((1.0 - (2 - 3 * math.sqrt(2))) / (6 * math.sqrt(2)))
    assert fused["n10"] == pytest.approx(top + b["n10"])
    assert fused["n00"] == pytest.approx((0.0 - (mean - 3 * std)) / (6 * std))  # absent from b
    assert fused["other"] == pytest.approx(b["other"])  # absent from a


def test_dbsf_is_deterministic_and_validates():
    a = {"x": 1.0, "y": 5.0, "z": 2.5}
    b = {"z": 0.1, "w": 0.9}
    first = dbsf_fuse({"a": a, "b": b})
    second = dbsf_fuse({"b": dict(reversed(list(b.items()))), "a": dict(reversed(list(a.items())))})
    assert list(first.items()) == list(second.items())
    with pytest.raises(ValueError):
        dbsf_fuse({"a": a}, {"a": -1.0})
    with pytest.raises(ValueError):
        dbsf_fuse({"a": a}, {"a": float("nan")})
    with pytest.raises(ValueError):
        dbsf_fuse({"a": {"x": float("inf"), "y": 1.0}})


# ── z-score fusion (opsem) ────────────────────────────────────────────────


def test_zscore_matches_hand_computed_values_with_min_fill():
    fused = zscore_fuse(
        {"bm25": {"x": 1.0, "y": 2.0, "z": 3.0}, "dense": {"x": 3.0, "y": 1.0}},
        {"bm25": 0.4, "dense": 0.6},
    )
    # bm25 z = [-sqrt(1.5), 0, sqrt(1.5)]; dense filled [3, 1, 1] -> [sqrt 2, -1/sqrt 2, -1/sqrt 2].
    assert fused == pytest.approx({
        "x": -0.4 * math.sqrt(1.5) + 0.6 * math.sqrt(2),
        "y": -0.6 / math.sqrt(2),
        "z": 0.4 * math.sqrt(1.5) - 0.6 / math.sqrt(2),
    })


def test_zscore_matches_opsem_formula_on_complete_vectors():
    rng = np.random.default_rng(7)
    ids = [f"t{i}" for i in range(9)]
    bm, dense = rng.random(9) * 12, rng.random(9)
    fused = zscore_fuse({"sparse": dict(zip(ids, bm)), "dense": dict(zip(ids, dense))}, {"sparse": 0.4, "dense": 0.6})
    expected = 0.4 * (bm - bm.mean()) / bm.std() + 0.6 * (dense - dense.mean()) / dense.std()
    assert [fused[i] for i in ids] == pytest.approx(list(expected))


def test_zscore_constant_and_empty_channels_contribute_zero():
    assert zscore_fuse({"a": {"x": 5.0, "y": 5.0}}) == {"x": 0.0, "y": 0.0}
    fused = zscore_fuse({"a": {"x": 1.0, "y": 3.0}, "b": {}})
    assert fused == pytest.approx({"x": -1.0, "y": 1.0})
    with pytest.raises(ValueError):
        zscore_fuse({"a": {"x": 1.0}}, {"a": float("inf")})


# ── EMG min-max + linear ──────────────────────────────────────────────────


def test_minmax_normalize_matches_emg_query_local_minmax():
    assert minmax_normalize({"a": 1.0, "b": 3.0, "c": 2.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}
    assert minmax_normalize({"a": 0.4, "b": 0.4}) == {"a": 0.0, "b": 0.0}
    assert minmax_normalize({}) == {}
    with pytest.raises(ValueError):
        minmax_normalize({"a": float("nan"), "b": 1.0})


def test_weighted_linear_fuse_uses_emg_weights_and_zero_for_missing():
    fused = weighted_linear_fuse({"graph": {"x": 1.0}, "dense": {"x": 0.5, "y": 1.0}}, EMG_LINEAR_WEIGHTS)
    assert fused == pytest.approx({"x": 0.30 + 0.35, "y": 0.70})
    assert weighted_linear_fuse({"a": {"x": 0.2}, "b": {"x": 0.3}}) == pytest.approx({"x": 0.5})


def test_rrf_fuse_contract_unchanged():
    fused = rrf_fuse({"a": [("x", 3.0), ("y", 1.0)], "b": [("y", 2.0)]}, k=60)
    assert fused == pytest.approx({"x": 1 / 61, "y": 1 / 62 + 1 / 61})


# ── Evidence assembly ─────────────────────────────────────────────────────


def _turn(nid: str, session: str, pos: int, text: str, **meta) -> TurnRecord:
    return TurnRecord(node_id=nid, evidence_id=f"ev-{nid}", text=text, session_key=session,
                      order=(session, pos), speaker="Ann", metadata=meta)


def _small_corpus() -> ScopedCorpus:
    return ScopedCorpus([
        _turn("a", "s1", 0, "one two", date="d1"),
        _turn("b", "s1", 1, "three", date="d1"),
        _turn("c", "s1", 2, "four five six seven eight nine ten eleven twelve thirteen fourteen", date="d1"),
        _turn("d", "s2", 0, "eleven", date="d2"),
        _turn("e", "s2", 1, "twelve", date="d2"),
    ])


def test_token_count_and_render_match_harness_format():
    assert token_count("(d1) Ann: hi, there!") == 9
    turn = TurnRecord(node_id="n", evidence_id="D1:1", text="look", speaker="Bo",
                      timestamp="2023-05-08", metadata={"blip_caption": " a dog "})
    assert render_turn(turn) == "(2023-05-08) Bo: look [shares image: a dog]"
    dated = TurnRecord(node_id="n", evidence_id="n", text="x", speaker="Bo", timestamp="iso",
                       metadata={"date": "1:56 pm on 8 May, 2023"})
    assert render_turn(dated) == "(1:56 pm on 8 May, 2023) Bo: x"


def test_propagate_scores_is_same_session_one_hop():
    corpus = _small_corpus()
    ranked = propagate_scores(corpus, {"b": 2.0, "d": 1.0}, 0.5)
    # a, c inherit 0.5 * 2 from b; c does not leak into s2; e inherits 0.5 from d.
    assert ranked == [("b", 2.0), ("a", 1.0), ("c", 1.0), ("d", 1.0), ("e", 0.5)]
    with pytest.raises(ValueError):
        propagate_scores(corpus, {"b": -1.0}, 0.5)
    with pytest.raises(ValueError):
        propagate_scores(corpus, {"zz": 1.0}, 0.5)


def test_units_and_pack_skip_not_truncate():
    corpus = _small_corpus()
    assert window_units(corpus, ["c", "d"], 1) == [["b", "c"], ["d", "e"]]
    assert session_units(corpus, ["s2", "s1"]) == [["d", "e"], ["a", "b", "c"]]
    costs = {"a": 5, "b": 4, "c": 20, "d": 3, "e": 3}
    assert pack([["c"], ["a", "b"], ["d"], ["b", "e"]], costs, 13) == ["a", "b", "d"]
    assert pack([["c"], ["a"]], costs, None) == ["c", "a"]
    with pytest.raises(ValueError):
        session_units(corpus, ["missing"])


def test_rank_sessions_methods():
    corpus = _small_corpus()
    ranked = [("d", 3.0), ("a", 2.0), ("b", 2.0)]
    assert rank_sessions(corpus, ranked, "max") == [("s2", 3.0), ("s1", 2.0)]
    assert rank_sessions(corpus, ranked, "sum") == [("s1", 4.0), ("s2", 3.0)]
    dcg = rank_sessions(corpus, ranked, "dcg")
    assert dcg == [("s1", pytest.approx(1 / math.log2(3) + 1 / math.log2(4))), ("s2", 1.0)]


def test_assemble_without_budget_keeps_every_hit_and_labels_context():
    corpus = _small_corpus()
    ranked = [("b", 2.0), ("e", 1.0)]
    pack_ = assemble_evidence(corpus, ranked, strategy="window", window=1)
    assert [i.node_id for i in pack_.items] == ["a", "b", "c", "d", "e"]
    assert pack_.hit_ids == ("b", "e") and pack_.context_ids == ("a", "c", "d")
    assert pack_.dropped_hits == 0 and pack_.budget_tokens is None
    by_id = {i.node_id: i for i in pack_.items}
    assert (by_id["b"].role, by_id["b"].rank, by_id["b"].expanded_from) == ("hit", 1, None)
    assert (by_id["d"].role, by_id["d"].rank, by_id["d"].expanded_from) == ("context", None, "e")
    assert by_id["a"].evidence_id == "ev-a"
    assert pack_.packed_tokens == sum(token_count(render_turn(t)) for t in corpus.turns)


def test_assemble_budget_max_hits_and_order():
    corpus = _small_corpus()
    ranked = [("c", 3.0), ("a", 2.0), ("d", 1.0)]
    cost = {t.node_id: token_count(render_turn(t)) for t in corpus.turns}
    budget = cost["a"] + cost["d"]  # "c" is too large: skipped, later hits still packed
    out = assemble_evidence(corpus, ranked, strategy="turn", budget_tokens=budget, order="rank")
    assert [i.node_id for i in out.items] == ["a", "d"] and out.dropped_hits == 1
    capped = assemble_evidence(corpus, ranked, strategy="window", window=1, max_hits=1)
    assert capped.hit_ids == ("c",) and capped.context_ids == ("b",)  # a/d are outside max_hits
    sess = assemble_evidence(corpus, ranked, strategy="session", session_method="dcg", order="rank")
    assert [i.node_id for i in sess.items] == ["a", "b", "c", "d", "e"]
    assert {i.node_id: i.expanded_from for i in sess.items if i.role == "context"} == {"b": "c", "e": "d"}


def test_assemble_propagate_scores_context_and_never_fabricates_hits():
    corpus = _small_corpus()
    out = assemble_evidence(corpus, [("b", 2.0), ("d", 0.0)], strategy="propagate", lam=0.5, order="rank")
    assert [i.node_id for i in out.items] == ["b", "a", "c", "d"]  # zero-score hit d appended last
    assert out.hit_ids == ("b", "d")
    by_id = {i.node_id: i for i in out.items}
    assert (by_id["a"].role, by_id["a"].score, by_id["a"].expanded_from) == ("context", 1.0, "b")
    with pytest.raises(ValueError):
        assemble_evidence(corpus, [("b", 1.0), ("b", 0.5)])
    with pytest.raises(ValueError):
        assemble_evidence(corpus, [("zz", 1.0)])
    with pytest.raises(ValueError):
        assemble_evidence(corpus, [("b", 1.0)], strategy="bogus")
    with pytest.raises(ValueError):
        assemble_evidence(corpus, [("b", 1.0)], budget_tokens=0)


# ── Parity with scripts/offline_budgeted_evidence.py (E1-E4 harness) ──────

_SESSIONS = [
    (1, "1:56 pm on 8 May, 2023", [
        ("Caroline", "I went to the LGBTQ support group yesterday and it was so powerful.", ""),
        ("Melanie", "That sounds amazing! What happened at the support group?", ""),
        ("Caroline", "People shared their transgender stories. Here is the flyer.", "a poster for a support group"),
        ("Melanie", "I painted a sunrise over the lake last week.", "a painting of a sunrise"),
    ]),
    (2, "1:14 pm on 25 May, 2023", [
        ("Melanie", "We went camping at the lake with the kids.", ""),
        ("Caroline", "Camping sounds fun! Did the kids like the lake?", ""),
        ("Melanie", "They loved it, we roasted marshmallows.", ""),
    ]),
    (3, "7:55 pm on 9 June, 2023", [
        ("Caroline", "I am researching adoption agencies now.", ""),
        ("Melanie", "Adoption is wonderful. The support group must help.", ""),
        ("Caroline", "Yes, the group and my counseling classes both help.", ""),
        ("Melanie", "I signed up for a pottery class.", "a bowl made of clay"),
    ]),
]
_QUERIES = [
    "What did Caroline do at the support group?",
    "Where did Melanie go camping with the kids?",
    "What is Caroline researching?",
    "What did Melanie paint?",
    "pottery class clay",
]


def harness_conv() -> dict:
    turns = []
    for s, date, rows in _SESSIONS:
        for pos, (speaker, text, caption) in enumerate(rows):
            turns.append({"id": f"D{s}:{pos + 1}", "session": s, "pos": pos, "date": date,
                          "speaker": speaker, "text": text, "caption": caption, "obs": []})
    return {"sid": "synthetic", "turns": turns, "qs": []}


def corpus_from_harness(conv: dict) -> ScopedCorpus:
    return ScopedCorpus(
        TurnRecord(node_id=t["id"], evidence_id=t["id"], text=t["text"], session_key=str(t["session"]),
                   order=(t["session"], t["pos"]), speaker=t["speaker"],
                   metadata={"date": t["date"], "blip_caption": t["caption"]})
        for t in conv["turns"]
    )


def parity_mismatches(conv: dict, queries, budgets, rep: str = "raw", lams=(0.5, 0.7, 1.0), windows=(1, 2)) -> tuple:
    """Compare every selection against the harness; returns (checks, [mismatch descriptions])."""
    from scripts import offline_budgeted_evidence as h

    c = h.Conv(conv, rep)
    corpus = corpus_from_harness(conv)
    ids = [t["id"] for t in c.turns]
    assert [t.node_id for t in corpus.turns] == ids
    costs = {t.node_id: token_count(render_turn(t)) for t in corpus.turns}
    checks, bad = 1, []
    if [costs[i] for i in ids] != c.cost or [render_turn(t) for t in corpus.turns] != [h.render(t) for t in c.turns]:
        bad.append("render/token cost differs")

    def as_ids(units):
        return [[ids[i] for i in u] for u in units]

    for q in queries:
        sc = c.scores(q)
        ranked = [(ids[i], sc[i]) for i in c.ranked_turns(q)]
        ranked_ids = [n for n, _ in ranked]
        cases = [("turn", {"strategy": "turn"}, [[n] for n in ranked_ids])]
        for w in windows:
            mine = window_units(corpus, ranked_ids, w)
            checks += 1
            if mine != as_ids(h.strategy_units(f"nbr{w}", c, q)):
                bad.append(f"nbr{w} units differ for {q!r}")
            cases.append((f"nbr{w}", {"strategy": "window", "window": w}, mine))
        for lam in lams:
            mine = [n for n, _ in propagate_scores(corpus, dict(ranked), lam)]
            checks += 1
            if mine != [ids[j] for j in c.propagated(q, lam)]:
                bad.append(f"prop{lam} order differs for {q!r}")
            cases.append((f"prop{lam}", {"strategy": "propagate", "lam": lam}, [[n] for n in mine]))
        sessions = session_units(corpus, [str(s) for s in c.ranked_sessions(q)])
        checks += 1
        if sessions != as_ids(h.strategy_units("session", c, q)):
            bad.append(f"session units differ for {q!r}")
        for b in budgets:
            for name, kwargs, units in cases:
                expected = [ids[i] for i in h.pack(h.strategy_units(name, c, q), c.cost, b)]
                got = [i.node_id for i in assemble_evidence(corpus, ranked, budget_tokens=b, order="rank", **kwargs).items]
                checks += 2
                if got != expected:
                    bad.append(f"{name}@{b} assemble differs for {q!r}")
                if pack(units, costs, b) != expected:
                    bad.append(f"{name}@{b} pack differs for {q!r}")
            checks += 1
            if pack(sessions, costs, b) != [ids[i] for i in h.pack(h.strategy_units("session", c, q), c.cost, b)]:
                bad.append(f"session@{b} pack differs for {q!r}")
    return checks, bad


def test_parity_with_offline_budgeted_evidence_harness():
    checks, bad = parity_mismatches(harness_conv(), _QUERIES, budgets=[12, 30, 64, 128, 10**6])
    assert checks > 150
    assert bad == []
