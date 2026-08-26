"""
Edge-threshold sweep (docs/EVALUATION.md section 3).

The graph channel's failure mode is missing edges, not misranked edges, so
the highest-leverage graph work is coverage, not ranking. This script rebuilds
auto-edges at each cosine threshold in {0.60..0.85 step 0.05}, reports the
resulting edge count, and — when a MuSiQue-style 2-hop dev set is available —
reports graph-reachability: the fraction of 2-hop dev questions where the
gold second-hop node is graph-reachable within 2 hops of a first-hop
retrieval hit. That reachability number is the ceiling on any future graph
learning (see docs/EVALUATION.md section 3, decomposition and auto-edge
experiments; GNN promotion remains gated on held-out evidence).

Reuses engine/edge_inference.py's infer_cosine_edges() for the actual edge
creation logic and eval_musique_retrieval.py's loader/relevance helpers for
the dev question set — does not duplicate either.

WARNING: this script rebuilds `similar_to` edges in place against whatever
.mind directory this process opens (HYBRIDMIND_* env / config.py paths) —
it mutates real, persisted edge state each iteration and saves the graph
index at the end holding the LAST threshold tested. Run it against a copy
of your corpus, not directly against a production `.mind` directory, unless
you intend to keep the final threshold's edges.

Usage:
  python scripts/sweep_edge_threshold.py [--min 0.60] [--max 0.85] [--step 0.05]
                                          [--musique-file musique_ans_v1.0_dev.jsonl]
                                          [--n 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.dependencies import get_db_manager
from engine.edge_inference import infer_cosine_edges
from eval_musique_retrieval import DATA_DIR as MUSIQUE_DATA_DIR
from eval_musique_retrieval import load_questions as load_musique_questions
from eval_musique_retrieval import is_relevant_by_id


def parse_args():
    p = argparse.ArgumentParser(description="Sweep auto-edge cosine threshold and report coverage")
    p.add_argument("--min", type=float, default=0.60)
    p.add_argument("--max", type=float, default=0.85)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--musique-file", type=str, default="musique_ans_v1.0_dev.jsonl")
    p.add_argument("--n", type=int, default=200, help="Max 2-hop dev questions to use for reachability")
    p.add_argument("--max-hops", type=int, default=2)
    return p.parse_args()


def _thresholds(lo: float, hi: float, step: float):
    n_steps = round((hi - lo) / step)
    return [round(lo + i * step, 2) for i in range(n_steps + 1)]


def _clear_similar_to_edges(sqlite_store, graph_index) -> int:
    """Remove every existing `similar_to` auto-edge so each threshold is measured cleanly."""
    to_remove = [
        (u, v, data.get("id"))
        for u, v, data in graph_index.graph.edges(data=True)
        if data.get("type") == "similar_to"
    ]
    for u, v, edge_id in to_remove:
        graph_index.remove_edge(u, v)
        if edge_id:
            sqlite_store.delete_edge(edge_id)
    return len(to_remove)


def _build_paragraph_id_index(sqlite_store, node_ids) -> dict:
    """node_id -> paragraph/musique/source id, for matching MuSiQue supporting_ids."""
    index = {}
    for node_id in node_ids:
        node = sqlite_store.get_node(node_id)
        if not node:
            continue
        meta = node.get("metadata", {}) or {}
        pid = meta.get("paragraph_id") or meta.get("musique_id") or meta.get("source_id")
        if pid:
            index[node_id] = str(pid)
    return index


def _load_two_hop_questions(musique_file: str, n: int):
    path = MUSIQUE_DATA_DIR / musique_file
    if not path.exists():
        print(f"  [reachability SKIPPED] {path} not found — download MuSiQue to measure this metric "
              f"(see eval_musique_retrieval.py header). Edge counts below are still valid.")
        return []
    return load_musique_questions(musique_file, n=n, n_hops_filter=2)


def compute_reachability(
    embedding_engine, vector_index, graph_index, pid_to_node: dict, questions: list, max_hops: int
) -> tuple:
    """
    Returns (reachable_fraction, n_evaluated). n_evaluated excludes questions
    where neither supporting paragraph could be matched to an ingested node
    (corpus doesn't contain that MuSiQue split) — those are not silently
    counted as failures, they're excluded and reported.
    """
    node_by_pid = {}
    for node_id, pid in pid_to_node.items():
        node_by_pid.setdefault(pid, node_id)

    hits = 0
    evaluated = 0
    for q in questions:
        supporting_ids = sorted(q["supporting_ids"])
        if len(supporting_ids) < 2:
            continue
        hop1_gold, hop2_gold = supporting_ids[0], supporting_ids[1]
        hop2_node = node_by_pid.get(str(hop2_gold))
        if hop2_node is None:
            continue

        emb = embedding_engine.embed(q["question"])
        candidates = vector_index.search(emb, top_k=5)
        if not candidates:
            continue

        hop1_node = None
        for node_id, _score in candidates:
            if node_by_pid.get(str(hop1_gold)) == node_id:
                hop1_node = node_id
                break
        if hop1_node is None:
            hop1_node = candidates[0][0]  # fall back to the top retrieval hit

        evaluated += 1
        dist = graph_index.get_shortest_path_length(hop1_node, hop2_node)
        if dist is not None and dist <= max_hops:
            hits += 1

    if evaluated == 0:
        return None, 0
    return hits / evaluated, evaluated


def main():
    args = parse_args()
    thresholds = _thresholds(args.min, args.max, args.step)
    print("WARNING: this rebuilds similar_to edges in place against the opened .mind "
          "directory and persists the LAST threshold tested. Use a corpus copy if unsure.\n")

    db_manager = get_db_manager()
    sqlite_store = db_manager.sqlite_store
    graph_index = db_manager.graph_index
    vector_index = db_manager.vector_index
    embedding_engine = db_manager.embedding_engine

    all_embeddings = sqlite_store.get_all_node_embeddings()
    node_ids = [nid for nid, _emb in all_embeddings]
    print(f"Corpus: {len(node_ids)} nodes")

    questions = _load_two_hop_questions(args.musique_file, args.n)
    pid_to_node = _build_paragraph_id_index(sqlite_store, node_ids) if questions else {}
    if questions and not pid_to_node:
        print("  [reachability SKIPPED] no ingested node carries a paragraph_id/musique_id/source_id "
              "matching the MuSiQue schema — ingest the MuSiQue corpus with that metadata first.")
        questions = []

    print(f"\n{'threshold':>9}  {'edges':>8}  {'reachability':>12}  {'n_eval':>7}")
    results = []
    for threshold in thresholds:
        removed = _clear_similar_to_edges(sqlite_store, graph_index)
        edge_count = 0
        for node_id, embedding in all_embeddings:
            edge_count += infer_cosine_edges(
                node_id, embedding, vector_index, sqlite_store, graph_index, threshold=threshold
            )
        reachability, n_eval = (None, 0)
        if questions:
            reachability, n_eval = compute_reachability(
                embedding_engine, vector_index, graph_index, pid_to_node, questions, args.max_hops
            )
        reach_str = f"{reachability:.1%}" if reachability is not None else "n/a"
        print(f"{threshold:>9.2f}  {edge_count:>8d}  {reach_str:>12}  {n_eval:>7d}  (removed {removed} stale edges first)")
        results.append({"threshold": threshold, "edge_count": edge_count, "reachability": reachability, "n_eval": n_eval})

    print("\nPick the knee (reachability gain per edge added), not the max threshold with the most edges --")
    print("an over-dense graph turns traversal into noise injection (docs/EVALUATION.md section 3).")

    db_manager.save_indexes()
    return results


if __name__ == "__main__":
    main()
