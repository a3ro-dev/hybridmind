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

DATASET = [
    {"id": "doc1", "text": "Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.", "metadata": {"domain": "ml"}},
    {"id": "doc2", "text": "The Adam optimizer combines the advantages of AdaGrad and RMSProp, working well with sparse gradients.", "metadata": {"domain": "ml"}},
    {"id": "doc3", "text": "Backpropagation computes the gradient of the loss function with respect to the weights of the network.", "metadata": {"domain": "ml"}},
    {"id": "doc4", "text": "Transformer architectures rely entirely on self-attention mechanisms, dispensing with recurrence.", "metadata": {"domain": "ml"}},
    {"id": "doc5", "text": "BERT is a bidirectional transformer model that predicts masked words in a sentence.", "metadata": {"domain": "ml"}},
]

EDGES = [
    ("doc1", "doc2", "analogous_to", 1.0),
    ("doc1", "doc3", "depends_on", 1.0),
    ("doc4", "doc5", "supports", 1.0)
]

QUERIES = [
    {"query": "iterative optimization algorithm", "ground_truth": ["doc1"]},
    {"query": "Adam optimization", "ground_truth": ["doc2"]},
    {"query": "computing loss gradients", "ground_truth": ["doc3"]},
    {"query": "self-attention mechanism", "ground_truth": ["doc4"]},
    {"query": "bidirectional language model", "ground_truth": ["doc5"]},
    {"query": "gradient descent variants", "ground_truth": ["doc1", "doc2", "doc3"]},
    {"query": "attention based models", "ground_truth": ["doc4", "doc5"]}
]

def precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    relevant_retrieved = len(set(retrieved_k).intersection(set(relevant)))
    return relevant_retrieved / k

def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    relevant_retrieved = len(set(retrieved_k).intersection(set(relevant)))
    return relevant_retrieved / len(relevant)

def mrr(retrieved: list, relevant: list) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def run_ablation():
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

    modes = [
        {"name": "vector_only", "endpoint": "/search/vector", "params": {"top_k": 3}},
        {"name": "bm25_only", "endpoint": "/search/hybrid", "params": {"top_k": 3, "vector_weight": 1.0, "graph_weight": 0.0}},
        {"name": "hybrid", "endpoint": "/search/hybrid", "params": {"top_k": 3, "vector_weight": 0.6, "graph_weight": 0.4}},
        {"name": "hybrid_heavy_graph", "endpoint": "/search/hybrid", "params": {"top_k": 3, "vector_weight": 0.2, "graph_weight": 0.8}},
    ]

    results = {}

    for mode in modes:
        mode_name = mode["name"]
        results[mode_name] = {"precision@3": [], "recall@3": [], "mrr": []}

        for q in QUERIES:
            query_text = q["query"]
            gt = q["ground_truth"]
            gt_ids = [dataset_ids_map[gt_id] for gt_id in gt]

            payload = {"query_text": query_text, **mode["params"]}
            res = client.post(mode["endpoint"], json=payload)
            retrieved_ids = [r["node_id"] for r in res.json()["results"]]
            res_parents = []
            for rid in retrieved_ids:
                if "_" in rid:
                    res_parents.append(rid.split("_")[0])
                else:
                    res_parents.append(rid)

            results[mode_name]["precision@3"].append(precision_at_k(res_parents, gt_ids, 3))
            results[mode_name]["recall@3"].append(recall_at_k(res_parents, gt_ids, 3))
            results[mode_name]["mrr"].append(mrr(res_parents, gt_ids))

    summary = {}
    for mode, metrics in results.items():
        summary[mode] = {
            "mean_precision@3": sum(metrics["precision@3"]) / len(QUERIES),
            "mean_recall@3": sum(metrics["recall@3"]) / len(QUERIES),
            "mean_mrr": sum(metrics["mrr"]) / len(QUERIES)
        }

    output_path = Path("benchmarks/results/retrieval_ablation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Ablation results saved to {output_path}")
    logger.info(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_ablation()
