"""Tri-signal retrieval: channel independence, fusion, rerank, evidence, API."""

from dataclasses import replace

import numpy as np
import pytest

from engine.corpus import ScopedCorpus, TurnRecord
from engine.fusion import rrf_fuse
from engine.trisignal import (
    ChannelUnavailable,
    RetrievalConfig,
    ScopeIndex,
    TriSignalRetriever,
    config_sha256,
)
from tests.embedding_double import Deterministic4096EmbeddingEngine

EMB = Deterministic4096EmbeddingEngine()

DIALOG = [
    # session 1
    ("Caroline", "I went to the LGBTQ support group yesterday and it was powerful."),
    ("Melanie", "That sounds amazing, how did it make you feel?"),
    ("Caroline", "Accepted. The transgender stories were so inspiring."),
    ("Melanie", "I painted a sunrise over the lake last week."),
    # session 2
    ("Caroline", "I am researching adoption agencies for my future family."),
    ("Melanie", "Adoption is a big step. Which agencies look good?"),
    ("Caroline", "One agency supports LGBTQ parents, which matters to me."),
    ("Melanie", "My kids loved the pottery class we took together."),
]


def _corpus() -> ScopedCorpus:
    turns = []
    for i, (speaker, text) in enumerate(DIALOG):
        session = 1 if i < 4 else 2
        turns.append(TurnRecord(
            node_id=f"n{i}",
            evidence_id=f"D{session}:{i % 4 + 1}",
            text=text,
            session_key=f"S{session}",
            order=(f"2023-05-0{session}", session, i),
            speaker=speaker,
            timestamp=f"2023-05-0{session}T10:00:00",
            index_text=f"{speaker}: {text}",
            metadata={"dia_id": f"D{session}:{i % 4 + 1}"},
        ))
    return ScopedCorpus(turns, scope_key="conv-test", generation=1)


def _scope() -> ScopeIndex:
    corpus = _corpus()
    return ScopeIndex(
        corpus,
        vectors=lambda: {t.node_id: EMB.embed(t.search_text) for t in corpus.turns},
        expected_dim=4096,
    )


def _retriever(**kwargs) -> TriSignalRetriever:
    return TriSignalRetriever(embed_query=EMB.embed, **kwargs)


def test_each_channel_runs_alone_and_only_itself():
    scope = _scope()
    retriever = _retriever()
    query = "Where did Caroline go for LGBTQ support?"
    for channel in ("dense", "sparse", "graph"):
        result = retriever.retrieve(scope, query, RetrievalConfig(channels=(channel,), top_k=5))
        executed = {name for name, run in result.channels.items() if run.executed}
        assert executed == {channel}
        assert result.trace["fusion"]["mode"] == "passthrough"
        assert result.hits, channel
        assert all(hit.sources == [channel] for hit in result.hits)
        assert result.fused == result.channels[channel].ranking


def test_graph_channel_is_independent_of_dense_vectors():
    corpus = _corpus()
    no_vectors = ScopeIndex(corpus, vectors=None)
    config = RetrievalConfig(channels=("graph",), top_k=8)
    query = "When did Caroline go to the LGBTQ support group?"
    without = _retriever().retrieve(no_vectors, query, config)
    with_vectors = _retriever().retrieve(_scope(), query, config)
    assert without.fused == with_vectors.fused
    assert without.trace["channels"]["graph"]["depends_on"] == []
    assert without.trace["channels"]["graph"]["query_keys"]


def test_graph_sequence_expansion_reaches_turns_without_query_terms():
    config = RetrievalConfig(channels=("graph",), top_k=8, sequence_expansion=True)
    # No query term appears in n1 ("That sounds amazing, how did it make you
    # feel?") after Lucene stopwording, so sparse cannot reach it.
    query = "When was Caroline at the LGBTQ support group?"
    ranked = [nid for nid, _ in _retriever().retrieve(_scope(), query, config).fused]
    sparse = RetrievalConfig(channels=("sparse",), top_k=8)
    sparse_ids = {nid for nid, _ in _retriever().retrieve(_scope(), query, sparse).fused}
    # n1 ("how did it make you feel?") shares no content term with the query;
    # it is reachable only through the chronological neighbour of an anchor.
    assert "n1" in ranked
    assert "n1" not in sparse_ids


def test_rrf_fusion_matches_rrf_fuse_over_channel_lists():
    config = RetrievalConfig(channels=("dense", "sparse", "graph"), top_k=8, fusion="rrf", rrf_k=60)
    result = _retriever().retrieve(_scope(), "adoption agencies LGBTQ parents", config)
    expected = rrf_fuse({n: r.ranking for n, r in result.channels.items()}, k=60)
    corpus = _corpus()
    assert result.fused == corpus.sort_scores(expected)
    top = result.hits[0]
    assert set(top.sources) <= {"dense", "sparse", "graph"}
    assert all(r is None or r >= 1 for r in top.channel_ranks.values())


@pytest.mark.parametrize("fusion", ["dbsf", "zscore", "minmax_linear"])
def test_score_fusions_execute_and_rank_every_candidate(fusion):
    config = RetrievalConfig(channels=("dense", "sparse", "graph"), top_k=5, fusion=fusion)
    result = _retriever().retrieve(_scope(), "painted sunrise lake", config)
    assert result.trace["fusion"]["mode"] == fusion
    assert result.hits[0].evidence_id == "D1:4"


def test_ppr_passage_seeds_must_be_requested_explicitly():
    config = RetrievalConfig(channels=("graph",), graph_method="ppr", ppr_passage_seed_channel="dense")
    with pytest.raises(ValueError, match="not requested"):
        config.validate()
    ok = RetrievalConfig(channels=("dense", "graph"), graph_method="ppr", ppr_passage_seed_channel="dense")
    result = _retriever().retrieve(_scope(), "adoption agencies", ok)
    assert result.trace["channels"]["graph"]["depends_on"] == ["dense"]


def test_ppr_graph_only_runs_without_other_channels():
    config = RetrievalConfig(channels=("graph",), graph_method="ppr", top_k=5)
    result = _retriever().retrieve(_scope(), "adoption agencies", config)
    assert result.hits
    assert {h.node_id for h in result.hits} & {"n4", "n5", "n6"}


def test_unavailable_stages_fail_closed():
    with pytest.raises(ChannelUnavailable, match="query embedder"):
        TriSignalRetriever().retrieve(_scope(), "q words", RetrievalConfig(channels=("dense",)))
    with pytest.raises(ChannelUnavailable, match="vector source"):
        _retriever().retrieve(ScopeIndex(_corpus()), "q words", RetrievalConfig(channels=("dense",)))
    with pytest.raises(ChannelUnavailable, match="reranker"):
        _retriever().retrieve(_scope(), "q words", RetrievalConfig(channels=("sparse",), rerank_pool=10))
    with pytest.raises(ChannelUnavailable, match="stored extractions"):
        _retriever().retrieve(_scope(), "q words", RetrievalConfig(channels=("graph",), graph_extractor="llm:x"))


def test_partial_stored_extraction_is_refused():
    corpus = _corpus()
    scope = ScopeIndex(corpus, stored_mentions=lambda extractor: {"n0": []})
    with pytest.raises(ChannelUnavailable, match="covers 1/8"):
        _retriever().retrieve(scope, "q words", RetrievalConfig(channels=("graph",), graph_extractor="llm:x"))


class _ReverseReranker:
    enabled = True

    def rerank(self, query, candidates, top_k=None):
        out = [dict(c) for c in candidates]
        for i, c in enumerate(out):
            c["rerank_score"] = float(i)  # reverse the fused order
        return out


def test_rerank_reorders_only_the_fixed_pool():
    config = RetrievalConfig(channels=("sparse", "graph"), top_k=3, rerank_pool=4)
    base = _retriever().retrieve(_scope(), "Caroline LGBTQ adoption", replace(config, rerank_pool=0, top_k=8))
    pool_ids = [nid for nid, _ in base.fused[:4]]
    result = _retriever(reranker=_ReverseReranker()).retrieve(_scope(), "Caroline LGBTQ adoption", config)
    assert [h.node_id for h in result.hits] == list(reversed(pool_ids))[:3]
    assert result.trace["rerank"]["applied"] is True


def test_evidence_budget_and_roles():
    config = RetrievalConfig(
        channels=("sparse",), top_k=3, evidence_strategy="window", evidence_window=1, evidence_budget_tokens=60,
    )
    result = _retriever().retrieve(_scope(), "painted sunrise lake", config)
    pack = result.evidence
    assert pack.packed_tokens <= 60
    roles = {item.role for item in pack.items}
    assert "hit" in roles
    hit_ids = {i.node_id for i in pack.items if i.role == "hit"}
    for item in pack.items:
        if item.role == "context":
            assert item.expanded_from in hit_ids
    order = [_corpus().index_of[i.node_id] for i in pack.items]
    assert order == sorted(order)


def test_config_hash_covers_ranking_knobs():
    base = RetrievalConfig()
    assert config_sha256(base) == config_sha256(RetrievalConfig())
    for change in (
        {"fusion": "dbsf"}, {"rrf_k": 10}, {"graph_method": "ppr"}, {"dense_mode": "hnsw"},
        {"weights": (("graph", 0.5),)}, {"sequence_expansion": False}, {"bm25_k1": 1.2},
    ):
        assert config_sha256(replace(base, **change)) != config_sha256(base)


def test_retrieve_endpoint_and_sdk(client):
    rows = []
    for i, (speaker, text) in enumerate(DIALOG):
        session = 1 if i < 4 else 2
        response = client.post("/nodes", json={
            "text": text,
            "metadata": {
                "containerTag": "trisignal-api-test",
                "session_id": f"tri-S{session}",
                "turn_index": i,
                "speaker": speaker,
                "dia_id": f"D{session}:{i % 4 + 1}",
                "timestamp": f"2023-05-0{session}T10:00:00+00:00",
            },
        })
        assert response.status_code == 201, response.text
        rows.append(response.json()["id"])

    payload = {
        "query": "Which agencies support LGBTQ parents?",
        "scope": {"containerTag": "trisignal-api-test"},
        "top_k": 3,
        "channels": ["dense", "sparse", "graph"],
        "evidence": {"strategy": "propagate", "budget_tokens": 200},
    }
    response = client.post("/retrieve", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hits"][0]["evidence_id"] == "D2:3"
    assert body["trace"]["channels"]["graph"]["executed"] is True
    assert body["trace"]["scope"]["turns"] == len(DIALOG)  # sentence chunks excluded
    assert isinstance(body["trace"]["corpus_generation"], int)
    assert len(body["trace"]["resolved_config_sha256"]) == 64
    assert body["evidence"]["packed_tokens"] <= 200

    bad = client.post("/retrieve", json={**payload, "unknown_field": 1})
    assert bad.status_code == 422

    from sdk.memory import HybridMemory

    sdk = HybridMemory(base_url="http://testserver", client=client, api_key="")
    out = sdk.retrieve("Which agencies support LGBTQ parents?", scope={"containerTag": "trisignal-api-test"},
                       top_k=2, channels=["sparse"])
    assert out["hits"][0]["evidence_id"] == "D2:3"

    for node_id in rows:
        client.delete(f"/nodes/{node_id}")
