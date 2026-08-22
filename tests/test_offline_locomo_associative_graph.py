import copy
import json

import pytest

from scripts.offline_locomo_associative_graph import (
    CONDITIONS,
    _create_once,
    _metrics,
    build_graph,
    build_retrieval,
    degree_preserving_sham,
    evaluate_data,
    rank_question,
    _cluster_bootstrap,
    _bm25,
    _terms,
    _paths,
    _ppr,
    turn_records,
)


def _item(sample_id="synthetic"):
    return {
        "sample_id": sample_id,
        "conversation": {
            "session_1": [
                {"dia_id": "D1:0", "speaker": "Alice", "text": "Alice met Bob at the museum."},
                {"dia_id": "D1:1", "speaker": "Bob", "text": "Bob kept a cobalt compass in storage."},
            ],
            "session_1_date_time": "2024-01-01",
            "session_2": [
                {"dia_id": "D2:0", "speaker": "Alice", "text": "Alice asked where the cobalt compass was."},
                {"dia_id": "D2:1", "speaker": "Bob", "text": "The cobalt compass was in storage."},
            ],
            "session_2_date_time": "2024-01-02",
        },
        "qa": [
            {
                "question": "Where was the cobalt compass?",
                "answer": "storage",
                "evidence": ["D1:1"],
                "category": 3,
            }
        ],
    }


def test_gold_is_not_used_to_form_graph_or_rankings():
    item = _item()
    records = turn_records(item)
    original = rank_question(records, item["qa"][0]["question"], candidate_budget=3, sham_seed=7)
    changed = copy.deepcopy(item)
    changed["qa"][0]["evidence"] = ["D2:0"]
    changed["qa"][0]["answer"] = "not storage"
    changed_records = turn_records(changed)
    mutated = rank_question(changed_records, changed["qa"][0]["question"], candidate_budget=3, sham_seed=7)
    for condition in CONDITIONS:
        assert [row["evidence_id"] for row in original["rankings"][condition]] == [
            row["evidence_id"] for row in mutated["rankings"][condition]
        ]
    assert all("answer" not in node for node in original["graph"]["kind"])


def test_degree_preserving_sham_is_deterministic_and_preserves_degrees():
    records = turn_records(_item())
    graph = build_graph(records)
    first = degree_preserving_sham(graph, seed=17)
    second = degree_preserving_sham(graph, seed=17)
    assert first["edges"] == second["edges"]
    assert {node: len(graph["adj"].get(node, {})) for node in graph["kind"]} == {
        node: len(first["adj"].get(node, {})) for node in first["kind"]
    }
    retrieval = build_retrieval(records, sham_seed=17)
    assert retrieval["sham_effectiveness"]["changed_term_turn_edges"] > 0
    assert retrieval["sham_effectiveness"]["retained_term_turn_fraction"] < 1.0


def test_equal_candidate_budgets_exact_ids_and_zero_calls():
    data = [_item("a"), _item("b")]
    report = evaluate_data(data, candidate_budget=4, final_k=10, split_seed="test")
    assert report["execution"]["provider_calls"] == 0
    assert report["config"]["candidate_budget"] == 4
    assert report["config"]["final_k"] == 10
    assert report["metrics"]["graph_ppr"]["n"] == 2
    measured = [row for row in report["rows"] if row["status"] == "measured"]
    assert measured
    for row in measured:
        for condition in CONDITIONS:
            condition_row = row["conditions"][condition]
            assert condition_row["candidate_budget"] == 4
            assert len(condition_row["ranking"]) == 4
            assert all(entry["evidence_id"].startswith("locomo:") for entry in condition_row["ranking"])
    assert report["rows"][0]["category"] == "multi-hop"
    assert report["graph_index"]["all_shams_degree_preserving"] is True
    assert 0.0 <= report["graph_index"]["mean_sham_retained_term_turn_fraction"] < 1.0
    category_deltas = report["paired_deltas"]["held_out_by_category"]
    assert "multi-hop" in category_deltas
    assert "graph_ppr_vs_bm25s" in category_deltas["multi-hop"]


def test_bm25_zero_hit_is_deterministically_padded_to_budget():
    records = turn_records(_item())
    retrieval = build_retrieval(records)
    ranked = _bm25(retrieval["bm25_index"], "unseen-token-xyz", 3, retrieval["all_ids"])
    assert len(ranked) == 3
    assert [node for node, _ in ranked] == sorted(retrieval["all_ids"])[:3]
    assert all(score == 0.0 for _, score in ranked)


def test_metric_math_mrr_is_bounded_by_final_k_and_gold_subset():
    ranking = [{"evidence_id": f"e{i}"} for i in range(1, 6)]
    metrics = _metrics(ranking, {"e2", "e4"}, final_k=2, budget=5)
    assert metrics["exact_evidence_recall_at_10"] == 0.5
    assert metrics["mrr_first_exact_evidence_at_10"] == 0.5
    assert metrics["all_exact_evidence_hit_at_10"] == 0.0
    assert _metrics(ranking, {"e2"}, final_k=2, budget=5)["all_exact_evidence_hit_at_10"] == 1.0


def test_speaker_anchor_uses_token_boundaries():
    records = turn_records(_item())
    records[0] = {**records[0], "speaker": "Ann"}
    retrieval = build_retrieval(records, sham_seed=3)
    assert not any(node == "speaker:ann" for node in rank_question(retrieval, "Anniversary", candidate_budget=4)["anchors"])
    assert any(node == "speaker:ann" for node in rank_question(retrieval, "Ann", candidate_budget=4)["anchors"])
    assert not any(node == "speaker:ann" for node in rank_question(retrieval, "Anniversary", candidate_budget=4)["anchors"])


def test_graph_tokenizer_attests_stopword_removal():
    assert _terms("The cobalt and compass") == ["cobalt", "compass"]


def test_provenance_paths_use_one_bounded_multi_source_traversal():
    graph = build_graph(turn_records(_item()))
    anchors = ["term:cobalt", "term:storage"]
    targets = ["turn:locomo:synthetic:D1:1", "turn:locomo:synthetic:D2:1"]
    paths = _paths(graph, anchors, targets)
    assert set(paths) == set(targets)
    assert all(path and path[0] in anchors and path[-1] in targets for path in paths.values())
    assert all(len(path) <= 5 for path in paths.values())


def test_sparse_matrix_ppr_matches_reference_edge_iteration():
    graph = build_graph(turn_records(_item()))
    query = "Where was the cobalt compass?"
    actual, anchors = _ppr(graph, query)
    teleport = {node: (1.0 / len(anchors) if node in anchors else 0.0) for node in graph["kind"]}
    scores = dict(teleport)
    for _ in range(40):
        expected = {node: 0.15 * teleport[node] for node in graph["kind"]}
        dangling = 0.0
        for node in sorted(graph["kind"]):
            neighbors = graph["adj"].get(node, {})
            total = sum(neighbors.values())
            if not total:
                dangling += scores[node]
                continue
            for neighbor in sorted(neighbors):
                expected[neighbor] += 0.85 * scores[node] * neighbors[neighbor] / total
        for node in expected:
            expected[node] += 0.85 * dangling * teleport[node]
        scores = expected
    assert actual == pytest.approx({node: scores[node] for node in actual}, abs=1e-12)


def test_cluster_bootstrap_is_deterministic_and_clusters_conversations():
    values = {"conversation-a": [0.0, 1.0], "conversation-b": [1.0]}
    first = _cluster_bootstrap(values, seed=9, samples=100)
    second = _cluster_bootstrap(values, seed=9, samples=100)
    assert first == second
    assert first["clusters"] == 2


def test_malformed_questions_are_failure_ledger_rows():
    malformed = _item("malformed")
    malformed["qa"] = [None]
    report = evaluate_data([_item("valid"), malformed], candidate_budget=4, split_seed="test")
    failures = [row for row in report["failure_ledger"] if row["sample_id"] == "malformed"]
    assert failures and failures[0]["status"] == "failed_malformed_qa"
    assert failures[0]["question_id"] == "malformed:qa:0"


def test_create_once_does_not_overwrite(tmp_path):
    target = tmp_path / "result.json"
    _create_once(target, {"version": 1})
    with pytest.raises(FileExistsError):
        _create_once(target, {"version": 2})
    assert json.loads(target.read_text()) == {"version": 1}
