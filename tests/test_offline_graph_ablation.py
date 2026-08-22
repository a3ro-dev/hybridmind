from scripts.offline_graph_ablation import run_ablation


def test_graph_ablation_proves_bounded_gain_and_adversarial_invariants(tmp_path):
    result = run_ablation(tmp_path, cases=6, distractors_per_case=4, seed=11)

    assert result["provider_calls"] == 0
    assert result["design"]["metric_basis"] == "exact_evidence_id"
    assert result["conditions"]["graph_only"]["hit_at_2"] > result["conditions"]["vector_sparse"]["hit_at_2"]
    assert result["paired_effect"]["ci95_low"] > 0
    assert result["adversarial"]["cross_scope_leakage"] is False
    assert result["adversarial"]["historical_as_of_half_open_validity_pass"] is True
    assert result["adversarial"]["identical_text_distinct_provenance_pass"] is True
    assert result["hypothesis_h8_success"] is True
