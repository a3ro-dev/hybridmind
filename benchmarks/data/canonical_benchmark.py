"""
Builds and loads the self-contained, deterministic canonical benchmark dataset.
Does NOT require any external API, network, or LLM call at dataset construction time.
"""

import json
from pathlib import Path

def load_benchmark() -> dict:
    """
    Load the benchmark dataset from the JSON file.
    Returns:
        dict: A dictionary containing 'documents', 'edges', and 'queries'.
    """
    json_path = Path(__file__).parent / 'canonical_benchmark.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    def build_dataset():
        docs = []
        edges = []
        queries = []

        doc_id_counter = 1
        query_id_counter = 1

        def add_doc(text, domain="general"):
            nonlocal doc_id_counter
            doc_id = f"doc_{doc_id_counter:03d}"
            docs.append({"id": doc_id, "text": text, "metadata": {"domain": domain}})
            doc_id_counter += 1
            return doc_id

        def add_edge(src, tgt, etype="relates_to"):
            edges.append({"source": src, "target": tgt, "type": etype, "weight": 1.0})

        # Family 1: Semantic
        for i in range(10):
            target = add_doc(f"The concept of distributed consensus algorithms like Raft and Paxos is essential for fault-tolerant systems in cloud computing part {i}.")
            queries.append({
                "query_id": f"sem_{query_id_counter:03d}",
                "query_text": f"How do multiple nodes agree on a single state when some might fail {i}?",
                "expected_node_ids": [target],
                "family": "SEMANTIC",
                "requires_graph_expansion": False,
                "hop_count": 0,
                "notes": "Semantic paraphrase"
            })
            query_id_counter += 1

        # Family 2: Lexical
        for i in range(10):
            target = add_doc(f"This document mentions a very specific term: xyzzyplugh{i} frobnicator.")
            queries.append({
                "query_id": f"lex_{query_id_counter:03d}",
                "query_text": f"xyzzyplugh{i} frobnicator usage",
                "expected_node_ids": [target],
                "family": "LEXICAL",
                "requires_graph_expansion": False,
                "hop_count": 0,
                "notes": "Lexical exact match"
            })
            query_id_counter += 1

        # Family 3: Single-hop graph
        for i in range(10):
            anchor = add_doc(f"Semantic anchor document about neural network backpropagation theory number {i}.")
            target = add_doc(f"Here is an arbitrary historical fact: Abraham Lincoln was born in a log cabin {i}.")
            add_edge(anchor, target)
            queries.append({
                "query_id": f"g1_{query_id_counter:03d}",
                "query_text": f"information about neural network backpropagation theory number {i}",
                "expected_node_ids": [target],
                "family": "GRAPH_SINGLE_HOP",
                "requires_graph_expansion": True,
                "hop_count": 1,
                "notes": "Edge-dependent single-hop"
            })
            query_id_counter += 1

        # Family 4: Multi-hop graph
        for i in range(10):
            anchor = add_doc(f"Anchor regarding climate change policy {i}.")
            intermediate = add_doc(f"Intermediate link on economics {i}.")
            target = add_doc(f"The capital of Australia is Canberra {i}.")
            add_edge(anchor, intermediate)
            add_edge(intermediate, target)
            queries.append({
                "query_id": f"g2_{query_id_counter:03d}",
                "query_text": f"climate change policy {i}",
                "expected_node_ids": [target],
                "family": "GRAPH_MULTI_HOP",
                "requires_graph_expansion": True,
                "hop_count": 2,
                "notes": "Edge-dependent multi-hop"
            })
            query_id_counter += 1

        # Family 5: Missing Anchor
        for i in range(10):
            target = add_doc(f"This document has no edges and talks about solitary confinement {i}.")
            queries.append({
                "query_id": f"miss_{query_id_counter:03d}",
                "query_text": f"solitary confinement {i}",
                "expected_node_ids": [target],
                "family": "MISSING_ANCHOR",
                "requires_graph_expansion": False,
                "hop_count": 0,
                "notes": "Missing anchor fallback"
            })
            query_id_counter += 1

        # Family 6: Oversmoothing
        for i in range(10):
            docA = add_doc(f"The apple pie recipe calls for cinnamon and nutmeg {i}.")
            docB = add_doc(f"The apple cake recipe calls for cinnamon and nutmeg {i}.")
            add_edge(docA, docB)
            add_edge(docB, docA)
            queries.append({
                "query_id": f"over_{query_id_counter:03d}",
                "query_text": f"apple pie recipe {i}",
                "expected_node_ids": [docA],
                "family": "OVERSMOOTHING",
                "requires_graph_expansion": False,
                "hop_count": 0,
                "notes": "Oversmoothing adversarial"
            })
            query_id_counter += 1

        # Add 140 noise documents
        for i in range(140):
            add_doc(f"This is a random noise document {i} designed to distract the vector search engine with overlapping words like network, policy, recipe, and algorithms.")

        return {
            "documents": docs,
            "edges": edges,
            "queries": queries
        }

    dataset = build_dataset()
    json_path = Path(__file__).parent / 'canonical_benchmark.json'
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
