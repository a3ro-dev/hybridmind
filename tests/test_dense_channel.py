"""Dense channel, ANN audit, embedding cache and query-format tests (offline).

Synthetic seeded 4096-d vectors only; zero provider calls.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine.corpus import ScopedCorpus, TurnRecord
from engine.dense_channel import (
    DenseMatrix,
    HNSWScope,
    ann_audit,
    ann_recall,
    exact_rank,
    exact_scores,
)
from engine.embedding import (
    RemoteEmbeddingEngine,
    TEIEmbeddingEngine,
    format_query_for_embedding,
)
from engine.embedding_cache import CacheItem, EmbeddingCache, EmbeddingCacheMiss, cache_key
from storage.vector_index import VectorIndex

DIM = 4096


def _vectors(n: int, seed: int = 7, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _corpus(n: int) -> ScopedCorpus:
    # Insert in reverse so alignment with the chronological order is exercised.
    return ScopedCorpus(
        TurnRecord(node_id=f"t{i:03d}", evidence_id=f"D1:{i}", text=f"turn {i}", order=(i,))
        for i in reversed(range(n))
    )


def _matrix(n: int = 64, seed: int = 7) -> tuple[DenseMatrix, np.ndarray]:
    corpus = _corpus(n)
    raw = _vectors(n, seed)
    return DenseMatrix.from_vectors(corpus, {f"t{i:03d}": raw[i] for i in range(n)}, DIM), raw


# ---------------------------------------------------------------- exact channel


def test_dense_matrix_is_aligned_normalized_float32():
    dm, raw = _matrix()
    assert dm.matrix.dtype == np.float32 and dm.matrix.shape == (64, DIM)
    assert [t.node_id for t in dm.corpus.turns] == dm.node_ids == [f"t{i:03d}" for i in range(64)]
    np.testing.assert_allclose(np.linalg.norm(dm.matrix, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(dm.matrix[5], raw[5] / np.linalg.norm(raw[5]), atol=1e-6)
    assert not dm.matrix.flags.writeable


def test_exact_scores_and_rank_match_bruteforce_numpy():
    dm, raw = _matrix()
    query = _vectors(1, seed=99)[0]
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    brute = unit.astype(np.float64) @ (query / np.linalg.norm(query)).astype(np.float64)

    scores = exact_scores(query, dm)
    np.testing.assert_allclose([scores[f"t{i:03d}"] for i in range(64)], brute, atol=1e-5)

    ranked = exact_rank(query, dm)
    assert [nid for nid, _ in ranked] == [f"t{i:03d}" for i in np.argsort(-brute, kind="stable")]
    assert ranked == dm.corpus.sort_scores(scores)
    assert exact_rank(query, dm, top_k=5) == ranked[:5]


def test_exact_rank_ties_break_by_chronological_index():
    corpus = _corpus(4)
    same = _vectors(1)[0]
    dm = DenseMatrix.from_vectors(corpus, {f"t{i:03d}": same for i in range(4)}, DIM)
    assert [nid for nid, _ in exact_rank(same, dm)] == ["t000", "t001", "t002", "t003"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda v: v.pop("t003"), "have no vector"),
        (lambda v: v.__setitem__("t003", np.full(DIM, np.nan, np.float32)), "non-finite"),
        (lambda v: v.__setitem__("t003", np.ones(1024, np.float32)), "expected \\(4096,\\)"),
        (lambda v: v.__setitem__("t003", np.zeros(DIM, np.float32)), "zero norm"),
    ],
)
def test_dense_matrix_fails_closed(mutate, message):
    corpus = _corpus(8)
    raw = _vectors(8)
    vectors = {f"t{i:03d}": raw[i] for i in range(8)}
    mutate(vectors)
    with pytest.raises(ValueError, match=message):
        DenseMatrix.from_vectors(corpus, vectors, DIM)


def test_query_vector_fails_closed():
    dm, _ = _matrix(8)
    with pytest.raises(ValueError, match="expected \\(4096,\\)"):
        exact_rank(np.ones(1536, np.float32), dm)
    with pytest.raises(ValueError, match="zero norm"):
        exact_scores(np.zeros(DIM, np.float32), dm)
    with pytest.raises(ValueError, match="top_k"):
        exact_rank(np.ones(DIM, np.float32), dm, top_k=0)


def test_inferred_dimension_rejects_mixed_widths():
    corpus = _corpus(2)
    dm = DenseMatrix.from_vectors(corpus, {"t000": np.ones(8), "t001": np.arange(1, 9)}, None)
    assert dm.dim == 8
    with pytest.raises(ValueError, match="expected \\(8,\\)"):
        DenseMatrix.from_vectors(corpus, {"t000": np.ones(8), "t001": np.ones(9)}, None)


# ------------------------------------------------------------------- HNSW audit


def test_ann_recall_definition():
    assert ann_recall([1, 2, 3, 4], [4, 3, 9, 8], 4) == 0.5
    assert ann_recall([1, 2], [2, 1, 7], 10) == 1.0  # denominator = min(k, |exact|)
    with pytest.raises(ValueError):
        ann_recall([], [1], 5)


def test_hnsw_scope_recall_against_exact_is_reported():
    dm, _ = _matrix(256)
    queries = _vectors(16, seed=3)
    scope = HNSWScope(dm, m=32, ef_construction=40, ef_search=64)
    recalls = []
    for q in queries:
        exact = [nid for nid, _ in exact_rank(q, dm, top_k=10)]
        ann = [nid for nid, _ in scope.rank(q, 10)]
        recalls.append(ann_recall(exact, ann, 10))
    recall = float(np.mean(recalls))
    print(f"synthetic 4096-d HNSW M=32 efC=40 efS=64 recall@10 = {recall:.3f}")
    assert recall >= 0.8
    assert scope.params()["ef_search"] == 64


def test_ann_audit_shape_and_k_exceeds_ef_flags():
    dm, _ = _matrix(128)
    result = ann_audit(dm, dm.matrix, ks=[10, 100], m=16, ef_construction=40,
                       ef_search_grid=[16, 128], exclude_rows=list(range(128)))
    assert result["leave_one_out"] and result["n_queries"] == 128 and result["dim"] == DIM
    rows = {row["ef_search"]: row for row in result["by_ef_search"]}
    assert rows[16]["k_exceeds_ef_search"] == {"10": False, "100": True}
    assert rows[128]["k_exceeds_ef_search"] == {"10": False, "100": False}
    for row in rows.values():
        assert all(0.0 <= v <= 1.0 for v in row["recall_at_k"].values())
        assert row["latency_ms"]["p50"] >= 0.0
    assert rows[128]["recall_at_k"]["100"] >= rows[16]["recall_at_k"]["100"]
    assert "efSearch" in result["caveat"]
    json.dumps(result)  # artifact-serialisable


def test_ann_audit_leave_one_out_never_counts_self():
    dm, _ = _matrix(20)
    result = ann_audit(dm, dm.matrix, ks=[50], ef_search_grid=[64], exclude_rows=list(range(20)))
    assert result["effective_k"] == {"50": 19}
    assert result["by_ef_search"][0]["recall_at_k"]["50"] == 1.0


# ------------------------------------------------------------ embedding cache


def test_cache_roundtrip_miss_and_replay(tmp_path):
    path = tmp_path / "cache.sqlite"
    items = [CacheItem("doc", "alpha"), CacheItem("query", "alpha", "Find it"), CacheItem("doc", "beta")]
    vecs = _vectors(3, seed=11)
    with EmbeddingCache(path, "qwen/qwen3-embedding-8b", DIM) as cache:
        assert cache.put_many(items[:2], vecs[:2]) == 2
        assert cache.put_many(items[:1], vecs[2:3]) == 0  # first write wins
        found, missing = cache.get_many(items)
        assert missing == [items[2]]
        np.testing.assert_array_equal(found[cache.key(items[0])], vecs[0])
        with pytest.raises(EmbeddingCacheMiss, match="1 of 3 items absent"):
            cache.require_all(items)
    with EmbeddingCache(path, "qwen/qwen3-embedding-8b", DIM) as reopened:
        replay = reopened.require_all(items[:2])
        np.testing.assert_array_equal(replay[reopened.key(items[1])], vecs[1])
        assert len(reopened) == 2


def test_cache_key_separates_model_kind_and_instruction():
    keys = {
        cache_key("m", "doc", "", "x"),
        cache_key("m", "query", "", "x"),
        cache_key("m", "query", "I", "x"),
        cache_key("m2", "doc", "", "x"),
    }
    assert len(keys) == 4
    with pytest.raises(ValueError):
        cache_key("m", "passage", "", "x")


def test_cache_rejects_invalid_vectors_atomically(tmp_path):
    with EmbeddingCache(tmp_path / "c.sqlite", "m", DIM) as cache:
        bad = _vectors(2)
        bad[1, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            cache.put_many([CacheItem("doc", "a"), CacheItem("doc", "b")], bad)
        with pytest.raises(ValueError, match="exactly \\(4096,\\)"):
            cache.put_many([CacheItem("doc", "a")], [np.ones(1536)])
        assert len(cache) == 0
    # A reader bound to another width refuses rows written at 4096.
    with EmbeddingCache(tmp_path / "c.sqlite", "m", DIM) as cache:
        cache.put_many([CacheItem("doc", "a")], _vectors(1))
    with EmbeddingCache(tmp_path / "c.sqlite", "m", 1536) as narrow:
        with pytest.raises(ValueError, match="dim=4096"):
            narrow.get_many([CacheItem("doc", "a")])


# ------------------------------------------------------- query instruction format


def test_query_format_strings():
    task = "Given a question, retrieve relevant conversation turns"
    assert format_query_for_embedding("When?", task, "qwen3") == f"Instruct: {task}\nQuery:When?"
    assert format_query_for_embedding("When?", task, "nv_embed") == f"Instruct: {task}\nQuery: When?"
    assert format_query_for_embedding("When?", None, "qwen3") == "When?"
    assert format_query_for_embedding("When?", "", "nv_embed") == "When?"
    assert format_query_for_embedding("When?", None, "none") == "When?"
    with pytest.raises(ValueError):
        format_query_for_embedding("When?", task, "none")
    with pytest.raises(ValueError):
        format_query_for_embedding("When?", task, "e5")


@pytest.mark.parametrize("engine_cls", [RemoteEmbeddingEngine, TEIEmbeddingEngine])
def test_embed_query_formats_then_embeds_without_network(engine_cls, monkeypatch):
    engine = engine_cls(base_url="https://example.invalid", api_key="test-key")
    sent = []

    def fake_call(texts):
        sent.extend(texts)
        return np.ones((len(texts), DIM), dtype=np.float32)

    monkeypatch.setattr(engine, "_call_api", fake_call)
    try:
        vec = engine.embed_query("When did Ann move?", instruction="Retrieve turns")
        assert vec.shape == (DIM,)
        engine.embed_query("plain", style="nv_embed")
        assert sent == ["Instruct: Retrieve turns\nQuery:When did Ann move?", "plain"]
    finally:
        engine.close()


# ------------------------------------------------------ VectorIndex.search_exact


def test_vector_index_search_exact_masks_tombstones_and_matches_bruteforce():
    index = VectorIndex(dimension=DIM, deletion_threshold=0.5)
    raw = _vectors(12, seed=21)
    index.add_batch([(f"n{i}", raw[i]) for i in range(10)])
    index.remove("n3")  # tombstoned row
    index.add("n5", raw[10])  # re-add: old row tombstoned, new row live
    assert len(index.deleted_ids) == 2

    query = raw[5] + 0.1 * raw[10]
    live = {f"n{i}": raw[i] for i in range(10) if i not in (3, 5)} | {"n5": raw[10]}
    q = query / np.linalg.norm(query)
    brute = sorted(
        ((nid, float(v @ q / np.linalg.norm(v))) for nid, v in live.items()),
        key=lambda item: -item[1],
    )

    exact = index.search_exact(query, top_k=20, min_score=-1.0)
    assert [nid for nid, _ in exact] == [nid for nid, _ in brute]
    np.testing.assert_allclose([s for _, s in exact], [s for _, s in brute], atol=1e-5)
    assert "n3" not in dict(exact) and len(exact) == 9
    assert exact[0][0] == "n5"
    assert all(s >= 0.0 for _, s in index.search_exact(query, top_k=20))
    assert index.search_exact(query, top_k=3) == exact[:3]
    with pytest.raises(ValueError):
        index.search_exact(query, top_k=0)

    stats = index.get_stats()
    assert stats["index_type"] == "IndexHNSWFlat" and stats["hnsw_m"] == 32


def test_vector_index_search_exact_ties_follow_insertion_order():
    index = VectorIndex(dimension=DIM)
    v = _vectors(1)[0]
    index.add_batch([("b", v), ("a", v), ("c", v)])
    assert [nid for nid, _ in index.search_exact(v, top_k=3)] == ["b", "a", "c"]
    assert VectorIndex(dimension=DIM).search_exact(v) == []


# ---------------------------------------------------------------- ann_audit CLI


def test_ann_audit_cli_query_indirection_and_leave_one_out(tmp_path):
    from scripts import ann_audit as cli

    rng = np.random.default_rng(5)
    docs = tmp_path / "conv-7_docs.npz"
    np.savez(docs, vectors=rng.standard_normal((40, 16)).astype(np.float32),
             memory_ids=np.array(["a"] * 40, dtype=object))
    queries = tmp_path / "queries.npz"
    np.savez(queries, vectors=rng.standard_normal((3, 16)).astype(np.float32),
             question_digests=np.array(["q0", "q1", "q2"]),
             qa_sample_ids=np.array(["conv-7", "conv-9", "conv-7", "conv-7"]),
             qa_question_digests=np.array(["q2", "q1", "q0", "q2"]))
    out = tmp_path / "audit.json"
    common = ["--docs-npz", str(docs), "--scope-regex", r"conv-\d+", "--expected-dim", "16",
              "--config", "8:4,64", "--k", "5", "30", "--label", "synthetic", "--output", str(out)]

    assert cli.main(common + ["--query-npz", str(queries), "--query-scope-key", "qa_sample_ids",
                              "--query-link-key", "qa_question_digests",
                              "--query-row-key", "question_digests"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    scope = payload["scopes"][0]
    assert scope["scope"] == "conv-7" and scope["n_queries"] == 2  # q2, q0 deduplicated
    assert payload["provider_calls"] == 0 and payload["provenance"]["faiss_threads"] == 1
    assert len(payload["provenance"]["inputs_sha256"]) == 2
    assert {(r["m"], r["ef_search"]) for r in payload["summary"]} == {(8, 4), (8, 64)}

    assert cli.main(common + ["--max-queries", "10"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["params"]["query_mode"] == "leave_one_out"
    assert payload["scopes"][0]["audits"][0]["effective_k"] == {"5": 5, "30": 30}

    # Object arrays are never unpickled.
    with pytest.raises(ValueError, match="allow_pickle"):
        cli.main(common + ["--ids-key", "memory_ids"])


def test_ann_audit_cli_reads_store_read_only_at_4096(tmp_path):
    from scripts import ann_audit as cli
    from storage.sqlite_store import SQLiteStore

    db = tmp_path / "store.db"
    store = SQLiteStore(str(db))
    raw = _vectors(12, seed=4)
    for i in range(12):
        store.create_node(f"n{i:02d}", f"text {i}", {}, embedding=raw[i])
    store.delete_node("n05")
    store._get_connection().close()

    ids, vectors = cli.load_store_vectors(db)
    assert ids == [f"n{i:02d}" for i in range(12) if i != 5] and vectors.shape == (11, DIM)
    out = tmp_path / "store_audit.json"
    assert cli.main(["--store", str(db), "--config", "16:32", "--k", "5",
                     "--label", "synthetic store", "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["params"]["expected_dim"] == DIM and payload["scopes"][0]["n_queries"] == 11
    with pytest.raises(SystemExit):
        cli.main(["--store", str(db), "--expected-dim", "1536", "--label", "x", "--output", str(out)])
