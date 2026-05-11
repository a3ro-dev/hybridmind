"""
Runs all ablation conditions against the canonical benchmark dataset.
Does not use HTTP; instantiates engines directly.

Conditions:
A: VECTOR_ONLY
B: BM25_ONLY
C: VECTOR_PLUS_BM25
D: VECTOR_PLUS_GRAPH_RERANK_ONLY
E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION
F: FULL_PIPELINE
"""

import json
import os
from pathlib import Path

from engine.embedding import EmbeddingEngine
from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine
from engine.hybrid_ranker import HybridRanker
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from storage.bm25_index import BM25Index

def setup_engines(disable_averaging=False):
    db_path = ":memory:"
    sqlite_store = SQLiteStore(db_path)
    sqlite_store._init_schema()

    embedding_engine = EmbeddingEngine()
    embedding_engine.disable_neighborhood_averaging = disable_averaging

    vector_index = VectorIndex()

    graph_index = GraphIndex()
    bm25_index = BM25Index()

    vector_search = VectorSearchEngine(vector_index, sqlite_store, embedding_engine)
    graph_search = GraphSearchEngine(graph_index, sqlite_store)

    return sqlite_store, vector_index, graph_index, bm25_index, vector_search, graph_search, embedding_engine

def load_dataset():
    path = Path(__file__).parent / 'data' / 'canonical_benchmark.json'
    with open(path, 'r') as f:
        return json.load(f)

def run_ablations():
    dataset = load_dataset()
    docs = dataset['documents']
    edges = dataset['edges']
    queries = dataset['queries']

    results = {
        "A: VECTOR_ONLY": {},
        "B: BM25_ONLY": {},
        "C: VECTOR_PLUS_BM25": {},
        "D: VECTOR_PLUS_GRAPH_RERANK_ONLY": {},
        "E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION": {},
        "F: FULL_PIPELINE": {}
    }

    for condition_name in results.keys():
        print(f"Running {condition_name}...")
        disable_averaging = condition_name == "E: VECTOR_PLUS_BM25_PLUS_GRAPH_EXPANSION"
        disable_expansion = condition_name in ["A: VECTOR_ONLY", "B: BM25_ONLY", "C: VECTOR_PLUS_BM25", "D: VECTOR_PLUS_GRAPH_RERANK_ONLY"]

        vector_weight = 0.0 if condition_name == "B: BM25_ONLY" else 0.6
        graph_weight = 0.0 if condition_name in ["A: VECTOR_ONLY", "B: BM25_ONLY", "C: VECTOR_PLUS_BM25"] else 0.4
        bm25_boost = 0.0 if condition_name == "A: VECTOR_ONLY" else 0.25

        sqlite_store, vector_index, graph_index, bm25_index, vector_search, graph_search, embedding_engine = setup_engines(disable_averaging)

        hybrid_ranker = HybridRanker(
            vector_engine=vector_search,
            graph_engine=graph_search,
            bm25_index=bm25_index,
            disable_graph_expansion=disable_expansion
        )

        # Ingest docs
        print(f"  Ingesting {len(docs)} documents...")
        for doc in docs:
            embedding = embedding_engine.embed(doc['text'])
            node_id = doc['id']
            sqlite_store.create_node(node_id, doc['text'], doc.get('metadata', {}))
            vector_index.add(node_id, embedding)
            graph_index.add_node(node_id)
            bm25_index.add(node_id, doc['text'])

        # Ingest edges
        for edge in edges:
            edge_id = f"{edge['source']}_{edge['target']}"
            sqlite_store.create_edge(edge_id, edge['source'], edge['target'], edge['type'], edge.get('weight', 1.0))
            graph_index.add_edge(edge['source'], edge['target'], edge['type'], weight=edge.get('weight', 1.0))

        # Run queries
        condition_results = {family: {"recall@1": 0, "recall@3": 0, "recall@5": 0, "recall@10": 0, "precision@1": 0, "precision@3": 0, "mrr": 0, "hit_count": 0, "total": 0, "crash_count": 0} for family in ["SEMANTIC", "LEXICAL", "GRAPH_SINGLE_HOP", "GRAPH_MULTI_HOP", "MISSING_ANCHOR", "OVERSMOOTHING"]}

        for q in queries:
            family = q['family']
            condition_results[family]["total"] += 1

            try:
                if condition_name == "B: BM25_ONLY":
                    # Special handling for BM25 only to ensure vector is totally 0
                    res, _, _ = hybrid_ranker.search(
                        query_text=q['query_text'], top_k=10, vector_weight=0.0, graph_weight=0.0, bm25_boost_weight=1.0
                    )
                else:
                    res, _, _ = hybrid_ranker.search(
                        query_text=q['query_text'],
                        top_k=10,
                        vector_weight=vector_weight,
                        graph_weight=graph_weight,
                        bm25_boost_weight=bm25_boost
                    )

                result_ids = [r['node_id'] for r in res]
                expected = q['expected_node_ids']

                # metrics
                found = any(eid in result_ids for eid in expected)
                if found:
                    condition_results[family]["hit_count"] += 1

                if any(eid in result_ids[:1] for eid in expected):
                    condition_results[family]["recall@1"] += 1
                    condition_results[family]["precision@1"] += 1
                if any(eid in result_ids[:3] for eid in expected):
                    condition_results[family]["recall@3"] += 1
                if any(eid in result_ids[:5] for eid in expected):
                    condition_results[family]["recall@5"] += 1
                if any(eid in result_ids[:10] for eid in expected):
                    condition_results[family]["recall@10"] += 1

                prec3 = sum(1 for eid in expected if eid in result_ids[:3]) / 3.0
                condition_results[family]["precision@3"] += prec3

                for rank, rid in enumerate(result_ids):
                    if rid in expected:
                        condition_results[family]["mrr"] += 1.0 / (rank + 1)
                        break

            except Exception as e:
                print(f"Error on query {q['query_id']}: {e}")
                condition_results[family]["crash_count"] += 1

        # Normalize
        for family, mets in condition_results.items():
            tot = max(1, mets["total"])
            mets["recall@1"] /= tot
            mets["recall@3"] /= tot
            mets["recall@5"] /= tot
            mets["recall@10"] /= tot
            mets["precision@1"] /= tot
            mets["precision@3"] /= tot
            mets["mrr"] /= tot

        results[condition_name] = condition_results

    # Write outputs
    os.makedirs(Path(__file__).parent / 'results', exist_ok=True)
    with open(Path(__file__).parent / 'results' / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(Path(__file__).parent / 'results' / 'ablation_summary.md', 'w') as f:
        f.write("# Ablation Summary\n\n")
        f.write("| Condition | Family | Recall@3 | Delta vs VECTOR_ONLY | MRR |\n")
        f.write("|---|---|---|---|---|\n")
        for cond, fams in results.items():
            for fam, mets in fams.items():
                r3 = mets['recall@3']
                mrr = mets['mrr']
                delta = r3 - results["A: VECTOR_ONLY"][fam]['recall@3']
                f.write(f"| {cond} | {fam} | {r3:.3f} | {delta:+.3f} | {mrr:.3f} |\n")

if __name__ == "__main__":
    run_ablations()
