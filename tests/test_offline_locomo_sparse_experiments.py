"""Scientific-contract tests for the offline sparse experiment harness."""

from scripts.offline_locomo_sparse_experiments import _select_variant


def _condition(*, recall: float, mrr: float, p95_ms: float, tokens: int = 100):
    return {
        "index_tokens_regex_proxy": tokens,
        "metrics": {
            "at_k": {"10": {"exact_evidence_recall": {"mean": recall}}},
            "mrr": {"mean": mrr},
            "query_latency_ms": {"p95": p95_ms},
            "by_category": {
                "single-hop": {"exact_evidence_recall_at_10": recall},
            },
        },
    }


def test_quality_selection_cannot_flip_from_one_shot_latency_noise():
    development = {
        "raw": _condition(recall=0.50, mrr=0.40, p95_ms=0.4),
        "speaker_prefix": _condition(recall=0.55, mrr=0.43, p95_ms=4.0),
    }

    winner, report = _select_variant(development)

    assert winner == "speaker_prefix"
    assert report["eligible"] == ["raw", "speaker_prefix"]
    assert report["latency_decision"]["applied"] is False
    assert (
        report["resource_observations"]["speaker_prefix"][
            "query_p95_ratio_vs_raw"
        ]
        == 10.0
    )


def test_deterministic_footprint_and_category_gates_still_reject_candidates():
    development = {
        "raw": _condition(recall=0.50, mrr=0.40, p95_ms=0.4),
        "oversized": _condition(
            recall=0.70, mrr=0.60, p95_ms=0.4, tokens=116,
        ),
        "regressed": _condition(recall=0.47, mrr=0.60, p95_ms=0.4),
    }

    winner, report = _select_variant(development)

    assert winner == "raw"
    assert "index token footprint" in report["rejected"]["oversized"][0]
    assert any("recall@10 regresses" in reason for reason in report["rejected"]["regressed"])
