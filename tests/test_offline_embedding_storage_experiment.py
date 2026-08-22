from scripts.offline_embedding_storage_experiment import run_experiment


def test_offline_storage_experiment_is_bit_exact_and_smaller(tmp_path):
    result = run_experiment(tmp_path, nodes=12, seed=7)

    assert result["provider_calls"] == 0
    assert result["bit_exact_logical_equivalence"] is True
    assert result["conditions"]["compact_override"]["physical_raw_embedding_rows"] == 0
    assert result["conditions"]["legacy_duplicated"]["physical_raw_embedding_rows"] == 12
    assert result["effect"]["database_bytes_saved"] > 0
    assert result["effect"]["database_reduction_fraction"] > 0
