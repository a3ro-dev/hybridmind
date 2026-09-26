"""Optional parity: HybridMind's EMG port vs the upstream EMG code (conv-26).

Runs ``scripts/reproduce_emg_locomo.py --port`` in a subprocess (the harness
seals the network process-wide, so it must not share the pytest process) on
the shipped conv-26 graph, question keys and cached 1536-d reference vectors.
Offline only: the harness raises on any provider/network attempt. Skipped when
the upstream clone or its NLTK data is absent (both gitignored under tmp/).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EMG_ROOT = ROOT / "tmp" / "upstream" / "em_graph_memory"
NLTK_DATA = ROOT / "tmp" / "nltk_data"

pytestmark = pytest.mark.skipif(
    not (EMG_ROOT / "code" / "em_graph").is_dir()
    or not (EMG_ROOT / "outputs" / "em_graph").is_dir()
    or not NLTK_DATA.is_dir(),
    reason="EMG clone / NLTK data not present",
)


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("emg_port") / "port.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce_emg_locomo.py"), "--port",
         "--samples", "conv-26", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_port_is_offline(payload):
    prov = payload["port_provenance"]
    assert prov["provider_calls"] == 0 and prov["blocked_network_attempts"] == []
    assert prov["query_artifact_usage"]["misses"] == 0


def test_port_rankings_identical_to_upstream(payload):
    cells = payload["port"]["cells"]
    assert {(v, int(k)) for v in cells for k in cells[v]} == {
        ("A", 5), ("A", 10), ("A", 25), ("A", 50), ("B", 5), ("B", 10), ("B", 25), ("B", 50),
        ("B_entity", 25), ("B_noseq", 25),
    }
    for variant, by_k in cells.items():
        for k, cell in by_k.items():
            ref = cell["reference_dense_upstream_order"]
            assert ref["compared"] == 199, (variant, k)
            assert ref["ordered_identical"] == 199, (variant, k)
            assert ref["metric_equal_to_upstream"], (variant, k)
            assert ref["max_abs_score_diff_shared_ids"] <= 1e-6, (variant, k)
            # HybridMind's own dense arithmetic and chronological tie-break may only
            # reorder float-level ties, never change the ranking logic.
            assert cell["hybridmind_dense_chronological"]["mismatch_other"] == 0, (variant, k)


def test_variants_reported_with_paired_cis(payload):
    variants = payload["variants"]
    assert "not reproductions" in variants["label"]
    assert set(variants["arms"]) == {
        "ppr_graph_only", "ppr_dense_passage_seeds", "lexical_graph_entity_only", "lexical_graph_fused",
    }
    for arm in variants["arms"].values():
        for k in ("5", "10", "25", "50"):
            cell = arm[k]
            assert 0.0 <= cell["overall"] <= 100.0 and 0.0 <= cell["cat1_4"] <= 100.0
            for against in ("ci_vs_A", "ci_vs_B"):
                ci = cell[against]["overall"]
                assert ci["ci95"][0] <= ci["mean_diff_points"] <= ci["ci95"][1]
                assert ci["rows"] == 199 and ci["clusters"] == 1
