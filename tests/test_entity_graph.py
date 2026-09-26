"""Entity-memory graph channel: EMG recall port, PPR, extractors, persistence.

Offline only: no provider calls. The upstream parity test runs only when the
EMG clone exists under tmp/upstream (gitignored) and is skipped otherwise.
"""

from __future__ import annotations

import json
import math
import random
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from engine.corpus import ScopedCorpus, TurnRecord
from engine.entity_extraction import (
    ENTITY_EXTRACTION_PROMPT,
    EntityMention,
    LexicalEntityExtractor,
    LLMEntityExtractor,
    NLTK_ENGLISH_STOPWORDS,
    normalize_entity_key,
    parse_entity_response,
    rake_phrase_scores,
)
from engine.entity_graph import (
    EntityMemoryGraph,
    memory_sort_key,
    normalize_semantic_scores,
    personalized_pagerank,
    tokenize_for_bm25,
)

ROOT = Path(__file__).resolve().parents[1]
EMG_ROOT = ROOT / "tmp" / "upstream" / "em_graph_memory"
L2, L4 = math.log(2.0), math.log(4.0)


def _bm25_ratio(n_docs: int, total_len: int) -> float:
    """Peak-normalised BM25Okapi score of a 2-token doc vs a 1-token doc (tf=1, k1=1.5, b=0.75)."""
    avgdl = total_len / n_docs
    norm = lambda dl: 1.0 + 1.5 * (0.25 + 0.75 * dl / avgdl)  # noqa: E731
    return norm(1) / norm(2)


# Entity docs: alice, alice cooper, bob, camping, camping trip, july, pottery,
# pottery class -> 8 docs, 11 tokens; every shared stem has df=2 (positive idf).
R = _bm25_ratio(8, 11)


def _mention(value: str, entity_type: str) -> EntityMention:
    return EntityMention(normalize_entity_key(value), value, entity_type)


def _corpus() -> ScopedCorpus:
    rows = [("t1", "D1:9", "Alice", "S1"), ("t2", "D1:10", "Bob", "S1"),
            ("t3", "D2:9", "Alice", "S2"), ("t4", "D2:10", "Bob", "S2")]
    return ScopedCorpus(
        [TurnRecord(nid, ev, f"text {nid}", session_key=s, order=(i,), speaker=spk)
         for i, (nid, ev, spk, s) in enumerate(rows)],
        scope_key="toy",
    )


MENTIONS = {
    "t1": [_mention("pottery class", "What"), _mention("July", "When")],
    "t2": [_mention("pottery", "What"), _mention("Alice Cooper", "What")],
    "t3": [_mention("camping trip", "What")],
    "t4": [_mention("camping", "What"), _mention("Alice", "Who")],
}


@pytest.fixture()
def graph() -> EntityMemoryGraph:
    return EntityMemoryGraph.from_corpus(_corpus(), MENTIONS)


def _approx(mapping):
    return pytest.approx(mapping, rel=1e-12, abs=1e-12)


def _assert_ranked(actual, expected):
    assert [n for n, _ in actual] == [n for n, _ in expected]
    assert [s for _, s in actual] == pytest.approx([s for _, s in expected], rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------- #
# Ported EMG functions
# --------------------------------------------------------------------------- #


def test_normalize_entity_key_matches_emg():
    assert normalize_entity_key("  Caroline's ") == "caroline"
    assert normalize_entity_key("LGBTQ   support\tgroup") == "lgbtq support group"
    # Upstream quirk kept: only a trailing 's is stripped and the result is not re-stripped.
    assert normalize_entity_key("Mel's 's") == "mel's "
    assert normalize_entity_key(None) == ""


def test_tokenize_for_bm25_stopwords_and_porter():
    assert tokenize_for_bm25("The runners were running 5 km in LGBTQ support-groups!") == [
        "runner", "run", "km", "lgbtq", "support", "group"]
    assert len(NLTK_ENGLISH_STOPWORDS) == 198 and "ourselves" in NLTK_ENGLISH_STOPWORDS


def test_graph_identity_merge_and_global_sequence(graph):
    assert graph.stats() == {
        "memories": 4, "entities": 8, "mention_edges": 11, "sequence_pairs": 3,
        "entity_types": {"What": 5, "When": 1, "Who": 2},
    }
    alice = graph.entities["entity:alice"]
    assert (alice.value, alice.type) == ("Alice", "Who")
    # Speaker added to t1/t3 (not extracted there); t4's extracted "Alice" is not a speaker duplicate.
    assert sorted(nid for eid, nid, _ in graph.mentions if eid == "entity:alice") == ["t1", "t3", "t4"]
    # The chain is global: t2 (session 1) links to t3 (session 2).
    assert graph.expand_sequence_neighbors({"t2": 1.0}) == {"t2": 1.0, "t1": 0.5, "t3": 0.5}


def test_from_corpus_merge_rules_and_validation():
    corpus = _corpus()
    graph = EntityMemoryGraph.from_corpus(corpus, {
        "t1": [EntityMention("mel", "Mel", "Who"), EntityMention("mel", "Mel", "Who")],
        "t2": [EntityMention("mel", "MEL ", "What"), EntityMention("mel", "Melly", "What")],
    }, add_speaker_as_entity=False)
    mel = graph.entities["entity:mel"]
    assert (mel.value, mel.type, mel.merged_from) == ("Melly", "Who", ["Mel", "MEL", "Melly"])
    assert graph.mentions == [("entity:mel", "t1", 1.0), ("entity:mel", "t2", 1.0)]
    with pytest.raises(ValueError, match="outside the corpus"):
        EntityMemoryGraph.from_corpus(corpus, {"ghost": []})


def test_entity_scores_degree_discount_and_bm25_soft_match(graph):
    assert graph.entity_scores({"Pottery"}) == _approx({"t2": 1.0 / L2, "t1": R / L2})
    assert graph.entity_scores({"pottery"}, degree_discount=False) == _approx({"t2": 1.0, "t1": R})
    # min_rel above the soft score keeps only the exact match; top_k_per_key=1 likewise.
    assert graph.entity_scores({"pottery"}, min_rel_score=0.9) == _approx({"t2": 1.0 / L2})
    assert graph.entity_scores({"pottery"}, top_k_per_key=1) == _approx({"t2": 1.0 / L2})
    assert graph.entity_scores(set()) == {} and graph.entity_scores({"zebra"}) == {}


def test_entity_scores_who_only_dampening(graph):
    # "alice cooper" (What) is matched only by a q-key that also matched a Who
    # entity, so it is Who-like and dampened too.
    who = 0.25 / L4
    assert graph.entity_scores({"alice"}) == _approx({"t1": who, "t3": who, "t4": who, "t2": 0.25 * R / L2})
    mixed = graph.entity_scores({"alice", "camping"})
    assert mixed == _approx({
        "t1": 0.25 * 0.5 / L4,
        "t2": 0.25 * (R / 2) / L2,
        "t3": (R / 2) / L2,     # content (camping trip) beats the Who score
        "t4": 0.5 / L2,         # content (camping)
    })
    assert graph.entity_scores({"alice"}, who_only_dampen=1.0)["t1"] == pytest.approx(1.0 / L4)


def test_expand_sequence_neighbors(graph):
    seeds = graph.entity_scores({"pottery"})
    assert graph.expand_sequence_neighbors(seeds) == _approx({"t2": 1 / L2, "t1": R / L2, "t3": 0.5 / L2})
    assert graph.expand_sequence_neighbors(seeds, secondary_scale=0.0) == seeds
    assert graph.expand_sequence_neighbors({}) == {}


def test_emg_rank_graph_only_positive_and_fill(graph):
    _assert_ranked(graph.emg_rank({"pottery"}), [("t2", 1 / L2), ("t1", R / L2), ("t3", 0.5 / L2)])
    _assert_ranked(graph.emg_rank({"pottery"}, expand_sequence=False, top_k=1), [("t2", 1 / L2)])
    assert graph.emg_rank({"pottery"}, fill=True, top_k=4)[-1] == ("t4", 0.0)
    assert graph.emg_rank({"zebra"}) == []
    assert [n for n, _ in graph.emg_rank({"zebra"}, fill=True, top_k=2)] == ["t1", "t2"]


def test_emg_rank_tie_break_chronological_vs_upstream(graph):
    ranked = graph.emg_rank({"alice"}, expand_sequence=False)
    assert [n for n, _ in ranked] == ["t2", "t1", "t3", "t4"]
    # EMG breaks ties by dia_id string: "D1:9" < "D2:10" < "D2:9".
    upstream = graph.emg_rank({"alice"}, expand_sequence=False, upstream_order=True)
    assert [n for n, _ in upstream] == ["t2", "t1", "t4", "t3"]


def test_emg_fused_rank_gate_fusion_and_fill(graph):
    sem = {"t1": 0.9, "t2": 0.1, "t3": 0.5, "t4": 0.3}
    calls = []

    def semantic(ids):
        calls.append(None if ids is None else frozenset(ids))
        return {k: v for k, v in sem.items() if ids is None or k in ids}

    ranked = graph.emg_fused_rank({"pottery"}, semantic, top_k=4)
    _assert_ranked(ranked, [
        ("t1", 0.3 * R / L2 + 0.7 * 0.9),
        ("t3", 0.3 * 0.5 / L2 + 0.7 * 0.5),
        ("t2", 0.3 / L2 + 0.7 * 0.1),
        ("t4", 0.7 * 0.3),  # dense fill outside the gate
    ])
    assert calls == [frozenset({"t1", "t2", "t3"}), None]
    assert len(graph.emg_fused_rank({"pottery"}, semantic, top_k=4, fill=False)) == 3

    calls.clear()  # empty gate: full-scope dense ranking
    assert [n for n, _ in graph.emg_fused_rank({"zebra"}, semantic, top_k=4)] == ["t1", "t3", "t4", "t2"]
    assert calls == [None]

    minmax = graph.emg_fused_rank({"zebra"}, semantic, top_k=1, semantic_normalization="query_local_minmax_v1")
    _assert_ranked(minmax, [("t1", 0.7)])
    with pytest.raises(ValueError, match="requires a semantic_scores"):
        graph.emg_fused_rank({"pottery"}, None, top_k=3)
    with pytest.raises(ValueError, match="finite"):
        graph.emg_fused_rank({"pottery"}, lambda ids: {"t1": float("nan")}, top_k=3)


def test_normalize_semantic_scores():
    assert normalize_semantic_scores({"a": 2.0, "b": 4.0}, "query_local_minmax_v1") == {"a": 0.0, "b": 1.0}
    assert normalize_semantic_scores({"a": 3.0, "b": 3.0}, "query_local_minmax_v1") == {"a": 0.0, "b": 0.0}
    with pytest.raises(ValueError):
        normalize_semantic_scores({}, "zscore")


def test_memory_sort_key_orders_by_session_datetime():
    early = memory_sort_key("D2:1", 2, "1:56 pm on 8 May, 2023")
    late = memory_sort_key("D1:3", 1, "10:00 am on 9 May, 2023")
    unparsed = memory_sort_key("D1:1", 1, "someday")
    assert early < late < unparsed and unparsed[0] == 1


def test_from_emg_json_maps_memories_to_node_ids():
    mem = lambda dia, n, dt, spk: {  # noqa: E731
        "id": f"memory:{dia}", "dia_id": dia, "session_num": n, "date_time": dt, "speaker": spk, "text": dia}
    data = {
        "sample_id": "conv-x",
        "memories": {"memory:D1:1": mem("D1:1", 1, "2:00 pm on 9 May, 2023", "A"),
                     "memory:D2:1": mem("D2:1", 2, "1:00 pm on 8 May, 2023", "B")},
        "entities": {"entity:tea": {"id": "entity:tea", "key": "tea", "value": "tea", "type": "What"}},
        "edges": [{"id": "e", "entity_id": "entity:tea", "memory_id": "memory:D1:1", "weight": 1.0}],
        "memory_edges": [
            {"id": "n", "src_memory_id": "memory:D2:1", "dst_memory_id": "memory:D1:1", "edge_type": "next"},
            {"id": "p", "src_memory_id": "memory:D1:1", "dst_memory_id": "memory:D2:1", "edge_type": "prev"}],
    }
    graph = EntityMemoryGraph.from_emg_json(data)
    assert [t.node_id for t in graph.corpus.turns] == ["locomo:conv-x:D2:1", "locomo:conv-x:D1:1"]
    assert graph.corpus.turn("locomo:conv-x:D1:1").evidence_id == "D1:1"
    _assert_ranked(graph.emg_rank({"tea"}), [("locomo:conv-x:D1:1", 1 / L2), ("locomo:conv-x:D2:1", 0.5 / L2)])
    bad = dict(data, edges=[{"id": "e", "entity_id": "entity:tea", "memory_id": "memory:D9:9"}])
    with pytest.raises(ValueError, match="Unknown memory endpoint"):
        EntityMemoryGraph.from_emg_json(bad)


# --------------------------------------------------------------------------- #
# PPR
# --------------------------------------------------------------------------- #


def _nx_ppr(n, edges, reset, damping):
    g = nx.MultiGraph()
    g.add_nodes_from(range(n))
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    pers = {i: float(r) for i, r in enumerate(reset)}
    pr = nx.pagerank(g, alpha=damping, personalization=pers, weight="weight", tol=1e-13, max_iter=10000)
    return np.array([pr[i] for i in range(n)])


@pytest.mark.parametrize("seed", range(6))
def test_personalized_pagerank_matches_networkx(seed):
    rng = random.Random(seed)
    n = rng.randint(5, 40)
    edges = []
    for _ in range(rng.randint(n, 3 * n)):
        u, v = rng.sample(range(n - 1), 2)  # node n-1 stays isolated (dangling)
        edges.append((u, v, rng.choice([1.0, 1.0, 2.5, 0.3])))
    edges.append(edges[0])  # a parallel edge: weights add
    reset = [rng.random() if rng.random() < 0.3 else 0.0 for _ in range(n)]
    reset[rng.randrange(n)] += 0.5
    damping = rng.choice([0.5, 0.85, 0.1])
    ours, iterations = personalized_pagerank(n, edges, reset, damping=damping)
    assert 0 < iterations < 1000
    assert ours.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.max(np.abs(ours - _nx_ppr(n, edges, reset, damping))) < 1e-8


def test_personalized_pagerank_input_validation():
    with pytest.raises(ValueError, match="positive finite mass"):
        personalized_pagerank(2, [(0, 1, 1.0)], [0.0, float("nan")])
    with pytest.raises(ValueError, match="self-loops"):
        personalized_pagerank(2, [(0, 0, 1.0)], [1.0, 0.0])
    with pytest.raises(RuntimeError, match="did not converge"):
        personalized_pagerank(3, [(0, 1, 1.0), (1, 2, 1.0)], [1.0, 0, 0], damping=0.99, max_iter=2)


def test_ppr_seeds_node_specificity_and_linking_top_k(graph):
    seeds = graph.ppr_seeds({"alice", "camping"})
    assert seeds == _approx({"entity:alice": 0.5 / 3, "entity:alice cooper": R / 2,
                             "entity:camping": 0.5, "entity:camping trip": R / 2})
    # Top-2 by strength; the 0.5 tie breaks by entity id.
    assert set(graph.ppr_seeds({"alice", "camping"}, linking_top_k=2)) == {"entity:alice", "entity:camping"}


@pytest.mark.parametrize("sequence", [True, False])
def test_ppr_rank_matches_networkx_on_the_same_graph(graph, sequence):
    passage = {"t1": 0.2, "t2": 0.8, "t3": 0.4, "t4": 0.2}
    ranked = graph.ppr_rank({"alice", "camping"}, include_sequence_edges=sequence, passage_scores=passage)
    g = nx.Graph()
    for eid, nid, w in graph.mentions:
        g.add_edge(eid, nid, weight=w)
    if sequence:
        g.add_edges_from([("t1", "t2"), ("t2", "t3"), ("t3", "t4")], weight=1.0)
    pers = dict(graph.ppr_seeds({"alice", "camping"}))
    for nid, s in passage.items():
        pers[nid] = pers.get(nid, 0.0) + (s - 0.2) / 0.6 * 0.05
    pr = nx.pagerank(g, alpha=0.5, personalization=pers, weight="weight", tol=1e-13, max_iter=10000)
    expected = sorted(((n, pr[n]) for n in ("t1", "t2", "t3", "t4")), key=lambda it: (-it[1], it[0]))
    assert [n for n, _ in ranked] == [n for n, _ in expected]
    assert max(abs(a[1] - b[1]) for a, b in zip(ranked, expected)) < 1e-8


def test_ppr_rank_empty_and_bad_passages(graph):
    assert graph.ppr_rank(set()) == []
    assert graph.ppr_rank({"zebra"}) == []
    with pytest.raises(ValueError, match="outside the corpus"):
        graph.ppr_rank({"pottery"}, passage_scores={"ghost": 1.0})
    assert len(graph.ppr_rank({"pottery"}, top_k=2)) == 2


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #


def test_lexical_extractor_is_deterministic_and_typed():
    extractor = LexicalEntityExtractor(speakers=["Caroline", "Melanie"])
    text = "I ran 5 km last week, on 7 May 2023, with Melanie at Lake Tahoe."
    got = [(m.key, m.type) for m in extractor.extract(text, speaker="Caroline")]
    assert got == [("caroline", "Who"), ("5 km", "How much"), ("last week", "When"),
                   ("7 may 2023", "When"), ("lake tahoe", "What"), ("melanie", "Who"), ("ran", "What")]
    assert extractor.extract(text, speaker="Caroline") == extractor.extract(text, speaker="Caroline")
    assert len(LexicalEntityExtractor(max_entities=2).extract(text, speaker="C")) == 3  # speaker + cap
    assert extractor.extract_query("When did Caroline go to the LGBTQ support group?") == {
        "caroline", "lgbtq", "support group"}
    assert LexicalEntityExtractor.version == "lexical-v1"


def test_rake_scores_degree_over_frequency():
    # deg(a)=2+1=3, freq(a)=2 -> 1.5; deg(b)=2, freq 1 -> 2; deg(c)=1 -> 1.
    assert rake_phrase_scores([["a", "b"], ["A"], ["c"]]) == [3.5, 1.5, 1.0]


def test_llm_extractor_parses_emg_responses_with_fake_completion():
    replies = iter([
        "not json at all",
        '```json\n[{"value": "Caroline\'s", "type": "Who"}, {"value": "support group", "type": "What"},'
        ' {"value": "Support Group.", "type": "What"}, {"value": "x", "type": "Bogus"},]\n```',
    ])
    seen = []

    def complete(messages):
        seen.append(messages)
        return next(replies)

    mentions = LLMEntityExtractor(complete).extract("Caroline went to a support group.")
    # Unknown type -> "What" (parse); same-length duplicate keeps the first; Who first, then longer.
    assert mentions == [EntityMention("caroline", "Caroline", "Who"),
                        EntityMention("support group", "support group", "What"),
                        EntityMention("x", "x", "What")]
    assert len(seen) == 2 and seen[0] == [{"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(
        text="Caroline went to a support group.")}]
    query = LLMEntityExtractor(lambda m: '[{"value": "Melanie", "type": "Who"}, {"value": "paint", "type": "What"}]')
    assert query.extract_query("What did Melanie paint?") == {"melanie", "paint"}
    with pytest.raises(ValueError):
        LLMEntityExtractor(lambda m: "nope", max_retries=2).extract("text")
    assert parse_entity_response('[]```json\n[]') == []


# --------------------------------------------------------------------------- #
# SQLite persistence
# --------------------------------------------------------------------------- #


def test_sqlite_entity_extraction_roundtrip_and_cascade(tmp_path):
    from storage.sqlite_store import SQLiteStore

    store = SQLiteStore(str(tmp_path / "store.db"))
    for nid in ("a", "b"):
        store.create_node(nid, f"memory {nid}", {}, np.zeros(4096, dtype=np.float32))
    generation = store.get_corpus_generation()
    mentions = [_mention("Caroline", "Who"), {"key": "tea", "value": "Tea", "type": "What"}]
    assert store.put_entity_extraction("a", "lexical-v1", mentions) == 2
    assert store.get_corpus_generation() > generation
    store.put_entity_extraction("b", "emg-llm-v4", [_mention("Mel", "Who")])
    assert store.get_entity_extractions(["a", "b", "missing"], "lexical-v1") == {"a": [
        {"key": "caroline", "value": "Caroline", "type": "Who"}, {"key": "tea", "value": "Tea", "type": "What"}]}
    store.put_entity_extraction("a", "lexical-v1", [])  # replace
    assert store.get_entity_extractions(["a"], "lexical-v1") == {"a": []}
    with pytest.raises(ValueError):
        store.put_entity_extraction("a", "", [])
    store.delete_node("b")
    assert store.get_entity_extractions(["b"], "emg-llm-v4")  # soft delete keeps derived rows
    store.hard_delete_soft_deleted_nodes()
    assert store.get_entity_extractions(["b"], "emg-llm-v4") == {}


# --------------------------------------------------------------------------- #
# Optional upstream parity (conv-26, shipped graph + question keys)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not (EMG_ROOT / "code" / "em_graph").is_dir(), reason="EMG clone not present")
def test_parity_with_upstream_emg_on_conv26(monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", sys.modules.get("fcntl") or types.ModuleType("fcntl"))
    monkeypatch.syspath_prepend(str(EMG_ROOT / "code"))
    from em_graph.build.models import EMGraph
    from em_graph.recall import retrieval as upstream
    from em_graph.recall import tokenize as upstream_tokenize
    from em_graph.recall.entity_bm25_index import EntityBM25Index

    # Never let upstream download NLTK data: hand it the list it would load.
    monkeypatch.setattr(upstream_tokenize, "_NLTK_STOPWORDS", set(NLTK_ENGLISH_STOPWORDS))
    base = EMG_ROOT / "outputs" / "em_graph"
    data = json.loads((base / "conv-26_em_graph_extract_v4_gpt35_tes.json").read_text(encoding="utf-8"))
    question_keys = json.loads((base / "conv-26_qkeys_gpt35_tes.json").read_text(encoding="utf-8"))
    up_graph = EMGraph.from_dict(data)
    up_index = EntityBM25Index.build(up_graph)
    graph = EntityMemoryGraph.from_emg_json(data)
    evidence = lambda nid: graph.corpus.turn(nid).evidence_id  # noqa: E731
    n = len(graph.corpus)
    sem = {t.node_id: math.sin(i * 1.7) for i, t in enumerate(graph.corpus.turns)}

    class _Dense:
        def scores(self, _question, memory_ids=None):
            ids = memory_ids if memory_ids is not None else up_graph.memories
            return {m: sem[f"locomo:conv-26:{up_graph.memories[m].dia_id}"] for m in ids}

    def ours_sem(ids):
        return {k: v for k, v in sem.items() if ids is None or k in ids}

    for keys in question_keys.values():
        seeds = upstream._entity_memory_scores(up_graph, set(keys), entity_bm25_index=up_index)
        assert {up_graph.memories[m].dia_id: s for m, s in seeds.items()} == {
            evidence(k): s for k, s in graph.entity_scores(keys).items()}
        entity_only = upstream.retrieve_dialog_ids(
            up_graph, "", top_k=n, embedding_index=None, entity_bm25_index=up_index,
            q_entity_keys=set(keys), entity_weight=1.0, semantic_weight=0.0)
        assert entity_only == [(evidence(k), s) for k, s in graph.emg_rank(keys, fill=True, upstream_order=True)]
        fused = upstream.retrieve_dialog_ids(
            up_graph, "", top_k=25, embedding_index=_Dense(), entity_bm25_index=up_index, q_entity_keys=set(keys))
        ours = graph.emg_fused_rank(keys, ours_sem, top_k=25, upstream_order=True)
        assert [d for d, _ in fused] == [evidence(k) for k, _ in ours]
        assert [s for _, s in fused] == pytest.approx([s for _, s in ours], abs=1e-12)
