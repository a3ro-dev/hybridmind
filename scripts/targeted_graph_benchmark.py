import json
import logging
import time
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We need cases for:
# 1. semantic paraphrase (vector should shine)
# 2. exact lexical lookup (BM25 should shine)
# 3. explicit edge-dependent multi-hop retrieval (graph should shine)
# 4. missing-anchor failure (graph fails)

DATASET = [
    # Semantic
    {"id": "sem1", "text": "Canines are known to be loyal companions.", "metadata": {"domain": "misc"}},
    {"id": "sem2", "text": "Felines often enjoy sleeping in sunny spots.", "metadata": {"domain": "misc"}},
    # Lexical
    {"id": "lex1", "text": "The patient was prescribed amoxapine 50mg for depression.", "metadata": {"domain": "medical"}},
    {"id": "lex2", "text": "The patient was prescribed fluoxetine 20mg for depression.", "metadata": {"domain": "medical"}},
    # Edge-dependent multi-hop (these don't have overlapping semantic words like 'chip' or 'engineering' with the query except for nodeB)
    {"id": "nodeA", "text": "Company X announced a breakthrough in quantum computing.", "metadata": {"domain": "news"}},
    {"id": "nodeB", "text": "Dr. Smith leads the team.", "metadata": {"domain": "news"}},
    {"id": "nodeC", "text": "The new processor operates at near absolute zero.", "metadata": {"domain": "news"}},
    # To make sure it doesn't get nodeC by chance, we add some other nodes that act as distractors
    {"id": "pad1", "text": "Who is the leader of the engineering team?", "metadata": {"domain": "misc"}},
    {"id": "pad2", "text": "Processors are becoming faster every year.", "metadata": {"domain": "misc"}},
]

EDGES = [
    # A -> B -> C
    ("nodeA", "nodeB", "supports", 1.0),
    ("nodeB", "nodeC", "supports", 1.0),
]

CASES = [
    {
        "name": "semantic_paraphrase",
        "query": "dogs make good pets",
        "ground_truth": ["sem1"],
        "anchors": []
    },
    {
        "name": "exact_lexical",
        "query": "amoxapine",
        "ground_truth": ["lex1"],
        "anchors": []
    },
    {
        "name": "edge_dependent_hop",
        "query": "What leads the engineering team",
        "ground_truth": ["nodeC"],
        "anchors": ["nodeA"]
    },
    {
        "name": "missing_anchor_failure",
        "query": "What leads the engineering team",
        "ground_truth": ["nodeC"],
        "anchors": []
    }
]

def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    return len(set(retrieved_k).intersection(set(relevant))) / len(relevant)

def run_benchmark():
    client = TestClient(app)
    client.delete("/bulk/clear")
    time.sleep(1)

    dataset_ids_map = {}
    for doc in DATASET:
        res = client.post("/nodes", json={"text": doc["text"], "metadata": doc["metadata"]})
        node_id = res.json()["id"]
        dataset_ids_map[doc["id"]] = node_id

    for source, target, edge_type, weight in EDGES:
        source_id = dataset_ids_map[source]
        target_id = dataset_ids_map[target]
        client.post("/edges", json={"source_id": source_id, "target_id": target_id, "type": edge_type, "weight": weight})

    results = {}
    for case in CASES:
        name = case["name"]
        query_text = case["query"]
        gt_ids = [dataset_ids_map[gt_id] for gt_id in case["ground_truth"]]
        anchors = [dataset_ids_map[a_id] for a_id in case["anchors"]]

        # Test vector
        res = client.post("/search/vector", json={"query_text": query_text, "top_k": 3})
        retrieved_ids = [r["node_id"].split("_")[0] for r in res.json()["results"]]
        vec_recall = recall_at_k(retrieved_ids, gt_ids, 3)

        # Test hybrid
        payload = {
            "query_text": query_text,
            "top_k": 3,
            "vector_weight": 0.1,  # heavily weight graph to ensure it surfaces nodeC
            "graph_weight": 0.9,
            "max_depth": 2
        }
        if anchors:
            payload["anchor_nodes"] = anchors
        res = client.post("/search/hybrid", json=payload)
        retrieved_ids = [r["node_id"].split("_")[0] for r in res.json()["results"]]
        hyb_recall = recall_at_k(retrieved_ids, gt_ids, 3)

        results[name] = {
            "vector_recall@3": vec_recall,
            "hybrid_recall@3": hyb_recall,
            "graph_helped": hyb_recall > vec_recall,
            "graph_hurt": hyb_recall < vec_recall
        }

    output_path = Path("benchmarks/results/targeted_graph_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Targeted benchmark results saved to {output_path}")
    logger.info(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_benchmark()
