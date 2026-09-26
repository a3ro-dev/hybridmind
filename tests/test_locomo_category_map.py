"""LoCoMo category IDs must follow the official mapping everywhere.

Five scripts once carried a permuted map (1=single-hop, 3=multi-hop, 4=world-knowledge),
which mislabelled published per-category findings. One canonical constant now exists.
"""

import json
from pathlib import Path

import pytest

import eval_locomo_retrieval
from scripts.offline_locomo_sparse_baseline import CATEGORY

OFFICIAL = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DATA = Path(__file__).resolve().parents[1] / "memorybench/data/benchmarks/locomo/locomo10.json"


def test_all_maps_are_the_official_one() -> None:
    assert CATEGORY == OFFICIAL
    assert eval_locomo_retrieval.CATEGORY_MAP == OFFICIAL


@pytest.mark.skipif(not DATA.is_file(), reason="locomo10.json not present")
def test_data_confirms_multi_hop_and_single_hop_ids() -> None:
    evidence = {1: [], 4: []}
    for sample in json.loads(DATA.read_text(encoding="utf-8")):
        for qa in sample["qa"]:
            if qa["category"] in evidence:
                evidence[qa["category"]].append(len(qa.get("evidence", [])))
    assert sum(n >= 2 for n in evidence[1]) / len(evidence[1]) > 0.9  # aggregation questions
    assert sum(evidence[4]) / len(evidence[4]) < 1.2  # single-fact questions
