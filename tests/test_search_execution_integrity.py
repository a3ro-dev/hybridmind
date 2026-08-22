"""Adversarial tests for corpus generations and server-attested search."""

from storage.sqlite_store import SQLiteStore


def test_corpus_generation_is_persistent_and_transactional(tmp_path):
    path = tmp_path / "generation.db"
    store = SQLiteStore(path)
    initial = store.get_corpus_generation()

    store.create_node(node_id="a", text="alpha", metadata={})
    after_first = store.get_corpus_generation()
    assert after_first > initial

    try:
        with store.transaction():
            store.create_node(node_id="b", text="beta", metadata={})
            raise RuntimeError("rollback")
    except RuntimeError:
        pass
    assert store.get_corpus_generation() == after_first

    reopened = SQLiteStore(path)
    assert reopened.get_corpus_generation() == after_first


def test_hybrid_response_attests_actual_stages_and_corpus_generation(
    client, create_test_node,
):
    create_test_node(
        "Execution trace exact keyword",
        {"benchmark_sample_id": "trace-scope", "dia_id": "D1:1"},
    )

    response = client.post(
        "/search/hybrid",
        json={
            "query_text": "execution trace exact keyword",
            "top_k": 1,
            "rerank_pool": 0,
            "search_mode": "sparse_only",
            "route_weights": False,
            "filter_metadata": {"benchmark_sample_id": "trace-scope"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    trace = body["execution_trace"]
    assert body["search_type"] == "sparse_only"
    assert trace["schema_version"] == "hybridmind.search-execution/v1"
    assert trace["corpus_generation"] > 0
    assert trace["search_mode"] == "sparse_only"
    assert trace["resolved_config_sha256"]
    assert trace["stages"]["sparse"]["executed"] is True
    assert trace["stages"]["dense"]["executed"] is False
    assert trace["stages"]["graph"]["executed"] is False
    assert trace["stages"]["cross_encoder"]["attempted"] is False


def test_cache_is_scoped_to_corpus_generation(client, create_test_node):
    first_id = create_test_node(
        "generation cache sentinel",
        {"benchmark_sample_id": "generation-cache", "dia_id": "D1:1"},
    )
    payload = {
        "query_text": "generation cache sentinel",
        "top_k": 5,
        "rerank_pool": 0,
        "search_mode": "sparse_only",
        "route_weights": False,
        "filter_metadata": {"benchmark_sample_id": "generation-cache"},
    }
    first = client.post("/search/hybrid", json=payload).json()
    second = client.post("/search/hybrid", json=payload).json()
    assert first["execution_trace"]["cache_hit"] is False
    assert second["execution_trace"]["cache_hit"] is True
    old_generation = second["execution_trace"]["corpus_generation"]

    second_id = create_test_node(
        "generation cache sentinel newest",
        {"benchmark_sample_id": "generation-cache", "dia_id": "D1:2"},
    )
    assert second_id != first_id
    third = client.post("/search/hybrid", json=payload).json()
    assert third["execution_trace"]["cache_hit"] is False
    assert third["execution_trace"]["corpus_generation"] > old_generation


def test_api_rejects_naive_as_of_and_invalid_graph_direction(client):
    hybrid = client.post(
        "/search/hybrid",
        json={
            "query_text": "time",
            "top_k": 1,
            "rerank_pool": 0,
            "as_of": "2026-01-01T00:00:00",
        },
    )
    assert hybrid.status_code == 422

    graph = client.get(
        "/search/graph",
        params={"start_id": "missing", "direction": "sideways"},
    )
    assert graph.status_code == 422
