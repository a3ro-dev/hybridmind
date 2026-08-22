"""Offline LoCoMo associative-graph ablation.

This is a mechanism ablation inspired by associative-memory retrieval (and by
the propagation component of HippoRAG), not a reproduction of HippoRAG.  It
uses only corpus turns, terms, speakers, and sessions to build a graph.  Gold
answers/evidence are consulted only after a ranking has been formed.

The experiment is deliberately conventional and bounded: BM25S with a
speaker-prefixed document control, graph personalized PageRank (PPR), a
deterministic degree-preserving graph sham, and fixed-k RRF(BM25S+PPR).  Every
condition returns the same candidate budget before exact-evidence metrics are
computed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_locomo_sparse_baseline import (
    DEFAULT_DATASET,
    _canonical_id,
    _gold_evidence,
    _sha256,
)
from storage.bm25_index import BM25SBackend

SCHEMA = "hybridmind.offline-locomo-associative-graph/v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "results" / "offline-locomo-associative-graph-20260822.json"
TOKEN = re.compile(r"\b[\w]+\b", re.UNICODE)
STOPWORDS = frozenset(
    "a an and are as at be by for from had has have he her his i if in is it its me my of on or our she that the their them they this to was we were what when where which who will with you your".split()
)
TOKENIZER = "unicode_word_boundary_v2"
SESSION_KEY = re.compile(r"^session_(\d+)$")
CATEGORY = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "world-knowledge",
    5: "adversarial",
}
CONDITIONS = (
    "bm25s_speaker_prefix",
    "graph_ppr",
    "graph_sham_ppr",
    "rrf_bm25s_ppr",
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _terms(value: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in TOKEN.findall(value.lower()):
        if token in STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def turn_records(item: dict[str, Any]) -> list[dict[str, str]]:
    """Extract stable corpus records without touching QA or annotation fields."""
    sample_id = str(item.get("sample_id") or "").strip()
    conversation = item.get("conversation")
    if not sample_id or not isinstance(conversation, dict):
        raise ValueError("LoCoMo item is missing sample_id or conversation")
    keys = sorted(
        (key for key in conversation if SESSION_KEY.fullmatch(key)),
        key=lambda key: int(SESSION_KEY.fullmatch(key).group(1)),
    )
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for session_key in keys:
        session = session_key.removeprefix("session_")
        date = str(conversation.get(f"{session_key}_date_time") or "").strip()
        messages = conversation.get(session_key)
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            evidence_id = _canonical_id(
                sample_id, message.get("dia_id") or f"{session_key}:{index}"
            )
            if evidence_id in seen:
                raise ValueError(f"duplicate evidence ID: {evidence_id}")
            seen.add(evidence_id)
            records.append({
                "evidence_id": evidence_id,
                "text": text,
                "speaker": str(message.get("speaker") or "").strip(),
                "session": session,
                "date": date,
            })
    if not records:
        raise ValueError(f"LoCoMo sample {sample_id} has no indexable turns")
    return records


def _node(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _new_graph() -> dict[str, Any]:
    return {"adj": defaultdict(dict), "kind": {}, "turn": {}, "edges": []}


def _edge(graph: dict[str, Any], left: str, right: str, weight: float) -> None:
    if left == right:
        return
    old = graph["adj"][left].get(right)
    if old is None:
        graph["edges"].append((left, right, float(weight)))
        graph["adj"][left][right] = float(weight)
        graph["adj"][right][left] = float(weight)
    else:
        graph["adj"][left][right] = old + float(weight)
        graph["adj"][right][left] = old + float(weight)


def build_graph(records: list[dict[str, str]]) -> dict[str, Any]:
    """Build a source-linked turn/term/speaker/session graph from corpus only."""
    graph = _new_graph()
    for record in records:
        evidence_id = record["evidence_id"]
        turn = _node("turn", evidence_id)
        graph["kind"][turn] = "turn"
        graph["turn"][turn] = record
        terms = _terms(record["text"])
        for term in terms:
            term_node = _node("term", term)
            graph["kind"][term_node] = "term"
            _edge(graph, turn, term_node, 1.0)
        speaker = record.get("speaker", "")
        if speaker:
            speaker_node = _node("speaker", speaker.casefold())
            graph["kind"][speaker_node] = "speaker"
            _edge(graph, turn, speaker_node, 1.5)
        session_node = _node("session", record.get("session", ""))
        graph["kind"][session_node] = "session"
        _edge(graph, turn, session_node, 0.5)
    graph["records"] = records
    graph["edge_count"] = sum(len(neighbors) for neighbors in graph["adj"].values()) // 2
    graph["node_count"] = len(graph["kind"])
    graph["degree_sequence"] = sorted(
        Counter(graph["kind"][node] for node in graph["kind"]).items()
    )
    return graph


def degree_preserving_sham(graph: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Rewire only term-turn edges using deterministic double-edge swaps.

    The degree of every term and turn is preserved exactly; speaker/session
    links remain source-linked.  Thus popularity/degree alone cannot explain a
    sham-vs-real difference, while associative term adjacency is destroyed.
    """
    sham = _new_graph()
    sham["kind"] = dict(graph["kind"])
    sham["turn"] = dict(graph["turn"])
    # Copy all edges, then replace only the term-turn subset.
    term_turn = []
    fixed = []
    for left, right, weight in graph["edges"]:
        kinds = {graph["kind"].get(left), graph["kind"].get(right)}
        if kinds == {"term", "turn"}:
            term, turn = (left, right) if graph["kind"][left] == "term" else (right, left)
            term_turn.append((term, turn, weight))
        else:
            fixed.append((left, right, weight))
    rng = random.Random(seed)
    pairs = list(term_turn)
    used_pairs = {(term, turn) for term, turn, _ in pairs}
    # Deterministic swaps preserve both endpoint degree sequences.
    if len(pairs) > 1:
        for _ in range(max(1, len(pairs) * 8)):
            i, j = rng.sample(range(len(pairs)), 2)
            term_a, turn_a, weight_a = pairs[i]
            term_b, turn_b, weight_b = pairs[j]
            if term_a == term_b or turn_a == turn_b:
                continue
            old_a = (term_a, turn_a)
            old_b = (term_b, turn_b)
            cross_a = (term_a, turn_b)
            cross_b = (term_b, turn_a)
            used_pairs.remove(old_a)
            used_pairs.remove(old_b)
            if cross_a in used_pairs or cross_b in used_pairs:
                used_pairs.add(old_a)
                used_pairs.add(old_b)
                continue
            pairs[i] = (term_a, turn_b, weight_a)
            pairs[j] = (term_b, turn_a, weight_b)
            used_pairs.add(cross_a)
            used_pairs.add(cross_b)
    for left, right, weight in fixed:
        _edge(sham, left, right, weight)
    for term, turn, weight in pairs:
        _edge(sham, term, turn, weight)
    sham["records"] = graph["records"]
    sham["edge_count"] = sum(len(neighbors) for neighbors in sham["adj"].values()) // 2
    sham["node_count"] = len(sham["kind"])
    sham["degree_sequence"] = sorted(
        Counter(sham["kind"][node] for node in sham["kind"]).items()
    )
    return sham


def _degree_hash(graph: dict[str, Any]) -> str:
    payload = sorted(
        (node, graph["kind"].get(node), len(graph["adj"].get(node, {})))
        for node in graph["kind"]
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _prepare_ppr(graph: dict[str, Any]) -> None:
    """Prebuild the column-oriented random-walk operator for repeated PPR."""
    nodes = sorted(graph["kind"])
    positions = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    dangling: list[int] = []
    for source in nodes:
        neighbors = graph["adj"].get(source, {})
        total = float(sum(neighbors.values()))
        if not total:
            dangling.append(positions[source])
            continue
        for target, weight in neighbors.items():
            # Matrix times a column score vector: target rows, source columns.
            rows.append(positions[target])
            columns.append(positions[source])
            values.append(float(weight) / total)
    graph["ppr_nodes"] = nodes
    graph["ppr_positions"] = positions
    graph["ppr_transition"] = csr_matrix(
        (np.asarray(values, dtype=np.float64), (rows, columns)),
        shape=(len(nodes), len(nodes)),
        dtype=np.float64,
    )
    graph["ppr_dangling"] = np.asarray(dangling, dtype=np.int64)


def _term_turn_pairs(graph: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for left, right, _weight in graph["edges"]:
        kinds = {graph["kind"].get(left), graph["kind"].get(right)}
        if kinds != {"term", "turn"}:
            continue
        term, turn = (left, right) if graph["kind"][left] == "term" else (right, left)
        pairs.add((term, turn))
    return pairs


def build_retrieval(records: list[dict[str, str]], *, sham_seed: int = 0) -> dict[str, Any]:
    """Build each per-conversation retrieval structure exactly once."""
    graph = build_graph(records)
    sham = degree_preserving_sham(graph, sham_seed)
    _prepare_ppr(graph)
    _prepare_ppr(sham)
    bm_index = BM25SBackend()
    bm_index.add_batch([
        (record["evidence_id"], f"{record.get('speaker', '')}: {record['text']}".strip())
        for record in records
    ])
    real_pairs = _term_turn_pairs(graph)
    sham_pairs = _term_turn_pairs(sham)
    union = real_pairs | sham_pairs
    return {
        "records": records,
        "graph": graph,
        "sham_graph": sham,
        "bm25_index": bm_index,
        "all_ids": sorted(record["evidence_id"] for record in records),
        "degree_hash": _degree_hash(graph),
        "sham_degree_hash": _degree_hash(sham),
        "sham_effectiveness": {
            "term_turn_edges": len(real_pairs),
            "changed_term_turn_edges": len(real_pairs - sham_pairs),
            "retained_term_turn_fraction": (
                len(real_pairs & sham_pairs) / len(real_pairs) if real_pairs else 1.0
            ),
            "term_turn_edge_jaccard": (
                len(real_pairs & sham_pairs) / len(union) if union else 1.0
            ),
        },
    }


def _anchors(graph: dict[str, Any], query: str) -> list[str]:
    query_tokens = tuple(TOKEN.findall(query.casefold()))
    anchors: list[str] = []
    for term in _terms(query):
        node = _node("term", term)
        if node in graph["kind"]:
            anchors.append(node)
    # Speaker metadata is allowed only when the speaker string is literally
    # present in the query; no answer/evidence metadata is consulted.
    for node, kind in sorted(graph["kind"].items()):
        if kind != "speaker":
            continue
        speaker_tokens = tuple(TOKEN.findall(node.removeprefix("speaker:").casefold()))
        if speaker_tokens and any(
            query_tokens[index:index + len(speaker_tokens)] == speaker_tokens
            for index in range(len(query_tokens) - len(speaker_tokens) + 1)
        ):
            anchors.append(node)
    return list(dict.fromkeys(anchors))


def _ppr(graph: dict[str, Any], query: str, *, max_iter: int = 40) -> tuple[dict[str, float], list[str]]:
    anchors = _anchors(graph, query)
    turns = sorted(node for node, kind in graph["kind"].items() if kind == "turn")
    if not turns:
        return {}, anchors
    if not anchors:
        # No query anchor is a valid zero-hit result, not an oracle fallback.
        return {node: 0.0 for node in turns}, anchors
    if "ppr_transition" not in graph:
        _prepare_ppr(graph)
    nodes = graph["ppr_nodes"]
    positions = graph["ppr_positions"]
    teleport = np.zeros(len(nodes), dtype=np.float64)
    for node in anchors:
        teleport[positions[node]] = 1.0 / len(anchors)
    scores = teleport.copy()
    alpha = 0.15
    for _ in range(max_iter):
        dangling = float(scores[graph["ppr_dangling"]].sum()) if graph["ppr_dangling"].size else 0.0
        scores = (
            alpha * teleport
            + (1.0 - alpha) * (graph["ppr_transition"] @ scores)
            + (1.0 - alpha) * dangling * teleport
        )
    return {node: float(scores[positions[node]]) for node in turns}, anchors


def _paths(graph: dict[str, Any], anchors: list[str], targets: list[str]) -> dict[str, list[str]]:
    # One deterministic multi-source traversal supplies provenance for every
    # returned target.  Re-running BFS per anchor-target pair changes no
    # ranking and is needlessly O(A*T*(V+E)).
    queue: deque[tuple[str, list[str]]] = deque(
        (anchor, [anchor]) for anchor in sorted(set(anchors))
    )
    seen = set(anchors)
    discovered = {anchor: [anchor] for anchor in anchors}
    while queue:
        node, path = queue.popleft()
        if len(path) >= 5:
            continue
        for neighbor in sorted(graph["adj"].get(node, {})):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            neighbor_path = path + [neighbor]
            discovered[neighbor] = neighbor_path
            queue.append((neighbor, neighbor_path))
    return {target: discovered.get(target, []) for target in targets}


def _bm25(index: BM25SBackend, query: str, budget: int, all_ids: list[str]) -> list[tuple[str, float]]:
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stderr(sink):
            raw = [(str(node), float(score)) for node, score in index.search(query, top_k=budget)]
    seen: set[str] = set()
    result: list[tuple[str, float]] = []
    for node, score in raw:
        if node in all_ids and node not in seen:
            result.append((node, score))
            seen.add(node)
    # bm25s may emit fewer than top_k rows for a sparse query.  Padding is
    # deterministic and keeps the candidate budget comparable across channels.
    result.extend((node, 0.0) for node in all_ids if node not in seen)
    return result[:min(budget, len(all_ids))]


def rank_question(
    records_or_retrieval: list[dict[str, str]] | dict[str, Any], query: str, *, candidate_budget: int = 25,
    rrf_k: int = 60, sham_seed: int = 0,
) -> dict[str, Any]:
    """Form all rankings before any gold values are accepted by the caller."""
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    retrieval = (
        records_or_retrieval
        if isinstance(records_or_retrieval, dict)
        else build_retrieval(records_or_retrieval, sham_seed=sham_seed)
    )
    graph = retrieval["graph"]
    sham = retrieval["sham_graph"]
    all_ids = retrieval["all_ids"]
    target_budget = min(candidate_budget, len(all_ids))
    bm25 = _bm25(retrieval["bm25_index"], query, target_budget, all_ids)
    ppr_scores, anchors = _ppr(graph, query)
    sham_scores, sham_anchors = _ppr(sham, query)
    ppr = sorted(ppr_scores.items(), key=lambda item: (-item[1], item[0]))[:target_budget]
    sham_ppr = sorted(sham_scores.items(), key=lambda item: (-item[1], item[0]))[:target_budget]
    bm_positions = {node: index + 1 for index, (node, _) in enumerate(bm25)}
    ppr_positions = {node.removeprefix("turn:"): index + 1 for index, (node, _) in enumerate(ppr)}

    def rrf_score(node: str) -> float:
        return (
            (1.0 / (rrf_k + bm_positions[node]) if node in bm_positions else 0.0)
            + (1.0 / (rrf_k + ppr_positions[node]) if node in ppr_positions else 0.0)
        )

    rrf_candidates = sorted(
        set(bm_positions) | set(ppr_positions),
        key=lambda node: (
            -rrf_score(node),
            node,
        ),
    )[:target_budget]
    ppr_by_id = {node.removeprefix("turn:"): score for node, score in ppr}
    sham_by_id = {node.removeprefix("turn:"): score for node, score in sham_ppr}
    bm_by_id = dict(bm25)
    rankings: dict[str, list[dict[str, Any]]] = {}
    path_map = _paths(graph, anchors, [node for node, _ in ppr])
    sham_path_map = _paths(sham, sham_anchors, [node for node, _ in sham_ppr])

    def rows(ids: list[str], scores: dict[str, float], channel: str, paths: dict[str, list[str]],
             provenance: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        output = []
        for rank, evidence_id in enumerate(ids, 1):
            row = {
                "rank": rank,
                "evidence_id": evidence_id,
                "score": float(scores.get(evidence_id, 0.0)),
                "channel": channel,
                "anchor_nodes": anchors if channel == "ppr" else (sham_anchors if channel == "graph_sham_ppr" else []),
                "path": paths.get(_node("turn", evidence_id), []),
            }
            if provenance:
                row["provenance"] = provenance.get(evidence_id, {})
            output.append(row)
        return output

    rankings["bm25s_speaker_prefix"] = rows([node for node, _ in bm25], bm_by_id, "bm25s_speaker_prefix", {})
    rankings["graph_ppr"] = rows([node.removeprefix("turn:") for node, _ in ppr], ppr_by_id, "ppr", path_map)
    rankings["graph_sham_ppr"] = rows([node.removeprefix("turn:") for node, _ in sham_ppr], sham_by_id, "graph_sham_ppr", sham_path_map)
    rankings["rrf_bm25s_ppr"] = rows(
        rrf_candidates,
        {node: rrf_score(node) for node in rrf_candidates},
        "rrf_bm25s_ppr", {},
        {node: {"bm25_rank": bm_positions.get(node), "ppr_rank": ppr_positions.get(node)} for node in rrf_candidates},
    )
    return {
        "rankings": rankings,
        "candidate_budget": target_budget,
        "rrf_k": rrf_k,
        "anchors": anchors,
        "graph": graph,
        "sham_graph": sham,
    }


def _metrics(ranking: list[dict[str, Any]], gold: set[str], final_k: int, budget: int) -> dict[str, Any]:
    ids = [row["evidence_id"] for row in ranking]
    pool = set(ids[:budget])
    selected = set(ids[:final_k])
    first = next((index for index, evidence_id in enumerate(ids[:final_k], 1) if evidence_id in gold), None)
    return {
        "candidate_pool_oracle_recall": len(pool & gold) / len(gold) if gold else None,
        "exact_evidence_recall_at_10": len(selected & gold) / len(gold) if gold else None,
        "mrr_first_exact_evidence_at_10": 1.0 / first if first else 0.0,
        "mrr_first_exact_evidence": 1.0 / first if first else 0.0,
        "any_exact_evidence_hit_at_10": float(bool(selected & gold)),
        "all_exact_evidence_hit_at_10": float(gold.issubset(selected)),
        "gold_count": len(gold),
    }


def _split_ids(data: list[dict[str, Any]], seed: str) -> dict[str, list[str]]:
    ids = [str(item.get("sample_id") or "").strip() for item in data]
    if not ids or len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("LoCoMo sample IDs must be unique and non-empty")
    ranked = sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    midpoint = len(ranked) // 2
    if midpoint == 0 or midpoint == len(ranked):
        raise ValueError("at least two conversations are required for a split")
    return {"development": ranked[:midpoint], "held_out": ranked[midpoint:]}


def _cluster_bootstrap(values_by_cluster: dict[str, list[float]], *, seed: int, samples: int = 2000) -> dict[str, Any]:
    """Conversation-cluster bootstrap; questions from one conversation stay together."""
    clusters = sorted(values_by_cluster)
    if not clusters:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "clusters": 0}
    cluster_means = np.asarray(
        [float(np.mean(values_by_cluster[cluster])) for cluster in clusters], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = cluster_means[rng.integers(0, len(cluster_means), size=(samples, len(cluster_means)))].mean(axis=1)
    return {
        "mean": float(cluster_means.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": len(clusters),
        "bootstrap_samples": samples,
    }


def _paired_cluster_delta(
    rows: list[dict[str, Any]], left: str, right: str, metric: str, *,
    split: str, seed: int, category: str | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["status"] != "measured"
            or row["split"] != split
            or (category is not None and row["category"] != category)
        ):
            continue
        grouped[row["sample_id"]].append(
            row["conditions"][left]["metrics"][metric] - row["conditions"][right]["metrics"][metric]
        )
    return _cluster_bootstrap(grouped, seed=seed)


def evaluate_data(
    data: list[dict[str, Any]], *, dataset_path: Path | None = None, seed: int = 20260822,
    split_seed: str = "20260822", candidate_budget: int = 25, final_k: int = 10,
) -> dict[str, Any]:
    if final_k != 10:
        raise ValueError("this preregistered ablation reports final_k=10 only")
    if candidate_budget < 1 or final_k < 1:
        raise ValueError("candidate budget and final_k must be positive")
    split = _split_ids(data, split_seed)
    by_id = {str(item["sample_id"]): item for item in data}
    rows: list[dict[str, Any]] = []
    failure_ledger: list[dict[str, Any]] = []
    build_ms: list[float] = []
    query_ms: list[float] = []
    graph_footprints: list[dict[str, Any]] = []
    for split_name, sample_ids in split.items():
        for sample_id in sample_ids:
            item = by_id[sample_id]
            records = turn_records(item)
            build_start = time.perf_counter()
            retrieval = build_retrieval(records, sham_seed=seed)
            graph = retrieval["graph"]
            build_ms.append((time.perf_counter() - build_start) * 1000.0)
            graph_footprints.append({
                "sample_id": sample_id,
                "nodes": graph["node_count"],
                "edges": graph["edge_count"],
                "real_degree_sequence_sha256": retrieval["degree_hash"],
                "sham_degree_sequence_sha256": retrieval["sham_degree_hash"],
                "degree_preserved": retrieval["degree_hash"] == retrieval["sham_degree_hash"],
                "sham_effectiveness": retrieval["sham_effectiveness"],
            })
            questions = item.get("qa")
            if not isinstance(questions, list):
                failure = {
                    "sample_id": sample_id,
                    "split": split_name,
                    "question_id": f"{sample_id}:qa:missing",
                    "status": "failed_malformed_qa",
                    "failure": {"reason": "qa must be an array"},
                }
                rows.append(failure)
                failure_ledger.append(failure)
                continue
            for question_index, qa in enumerate(questions):
                question_id = f"{sample_id}:qa:{question_index}"
                if not isinstance(qa, dict):
                    failure = {
                        "sample_id": sample_id,
                        "split": split_name,
                        "question_id": question_id,
                        "question_index": question_index,
                        "status": "failed_malformed_qa",
                        "failure": {"reason": "question row must be an object"},
                    }
                    rows.append(failure)
                    failure_ledger.append(failure)
                    continue
                query = str(qa.get("question") or "").strip()
                gold, invalid = _gold_evidence(sample_id, qa.get("evidence", []))
                try:
                    category = CATEGORY.get(int(qa.get("category", 0)), "unknown")
                except (TypeError, ValueError):
                    category = "unknown"
                base = {
                    "sample_id": sample_id,
                    "split": split_name,
                    "question_id": question_id,
                    "question_index": question_index,
                    "question_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "category": category,
                    "gold_evidence_ids": sorted(gold),
                    "status": "measured",
                }
                if invalid or not query or not gold:
                    base["status"] = "failed_invalid_annotation" if invalid else ("failed_empty_question" if not query else "failed_no_gold")
                    base["failure"] = {"invalid_fragments": invalid, "reason": base["status"]}
                    rows.append(base)
                    failure_ledger.append(base)
                    continue
                corpus_ids = {record["evidence_id"] for record in records}
                missing = sorted(gold - corpus_ids)
                if missing:
                    base["status"] = "failed_unresolved_evidence"
                    base["failure"] = {"unresolved_evidence_ids": missing}
                    rows.append(base)
                    failure_ledger.append(base)
                    continue
                query_start = time.perf_counter()
                formed = rank_question(retrieval, query, candidate_budget=candidate_budget)
                query_ms.append((time.perf_counter() - query_start) * 1000.0)
                expected_budget = min(candidate_budget, len(records))
                base["conditions"] = {
                    condition: {
                        "candidate_budget": expected_budget,
                        "ranking": formed["rankings"][condition],
                        "metrics": _metrics(formed["rankings"][condition], gold, final_k, candidate_budget),
                    }
                    for condition in CONDITIONS
                }
                if any(len(base["conditions"][condition]["ranking"]) != expected_budget for condition in CONDITIONS):
                    raise AssertionError(f"candidate budget violation for {question_id}")
                candidate_sets = {
                    condition: {entry["evidence_id"] for entry in base["conditions"][condition]["ranking"]}
                    for condition in CONDITIONS
                }
                base["candidate_accounting"] = {
                    "incremental_gold_vs_bm25": {
                        condition: sorted((candidate_sets[condition] - candidate_sets["bm25s_speaker_prefix"]) & gold)
                        for condition in CONDITIONS if condition != "bm25s_speaker_prefix"
                    },
                    "pairwise_candidate_jaccard": {
                        f"{left}__{right}": len(candidate_sets[left] & candidate_sets[right]) / len(candidate_sets[left] | candidate_sets[right])
                        if candidate_sets[left] | candidate_sets[right] else 1.0
                        for index, left in enumerate(CONDITIONS)
                        for right in CONDITIONS[index + 1:]
                    },
                    "gold_hits_in_candidate_pool": {
                        condition: sorted(candidate_sets[condition] & gold) for condition in CONDITIONS
                    },
                }
                rows.append(base)
    measured = [row for row in rows if row["status"] == "measured"]
    metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [row for row in measured if condition in row["conditions"]]
        metric_names = (
            "candidate_pool_oracle_recall", "exact_evidence_recall_at_10",
            "mrr_first_exact_evidence_at_10", "any_exact_evidence_hit_at_10",
            "all_exact_evidence_hit_at_10",
        )
        metrics[condition] = {
            "n": len(selected),
            "overall": {
                key: float(np.mean([row["conditions"][condition]["metrics"][key] for row in selected])) if selected else None
                for key in metric_names
            },
            "conversation_cluster_bootstrap": {
                key: _cluster_bootstrap(
                    {sample_id: [row["conditions"][condition]["metrics"][key] for row in selected if row["sample_id"] == sample_id]
                     for sample_id in sorted({row["sample_id"] for row in selected})},
                    seed=seed + index + len(condition),
                )
                for index, key in enumerate(metric_names)
            },
            "by_split": {
                split_name: {
                    key: float(np.mean([row["conditions"][condition]["metrics"][key] for row in selected if row["split"] == split_name])) if any(row["split"] == split_name for row in selected) else None
                    for key in metric_names
                }
                for split_name in ("development", "held_out")
            },
            "by_category": {
                category: {
                    "n": sum(row["category"] == category for row in selected),
                    "exact_evidence_recall_at_10": float(np.mean([row["conditions"][condition]["metrics"]["exact_evidence_recall_at_10"] for row in selected if row["category"] == category])) if any(row["category"] == category for row in selected) else None,
                }
                for category in sorted({row["category"] for row in selected})
            },
        }
    delta_metric = "exact_evidence_recall_at_10"
    paired_deltas = {
        "held_out_vs_bm25s_speaker_prefix": {
            condition: _paired_cluster_delta(measured, condition, "bm25s_speaker_prefix", delta_metric, split="held_out", seed=seed + index + 100)
            for index, condition in enumerate(("graph_ppr", "graph_sham_ppr", "rrf_bm25s_ppr"))
        },
        "held_out_graph_ppr_vs_sham": _paired_cluster_delta(measured, "graph_ppr", "graph_sham_ppr", delta_metric, split="held_out", seed=seed + 110),
        "held_out_by_category": {
            category: {
                comparison: _paired_cluster_delta(
                    measured, left, right, delta_metric, split="held_out",
                    category=category, seed=seed + 200 + category_index * 10 + comparison_index,
                )
                for comparison_index, (comparison, left, right) in enumerate((
                    ("graph_ppr_vs_bm25s", "graph_ppr", "bm25s_speaker_prefix"),
                    ("rrf_vs_bm25s", "rrf_bm25s_ppr", "bm25s_speaker_prefix"),
                    ("graph_ppr_vs_sham", "graph_ppr", "graph_sham_ppr"),
                ))
            }
            for category_index, category in enumerate(sorted({
                row["category"] for row in measured if row["split"] == "held_out"
            }))
        },
    }
    overlap_pairs: dict[str, list[float]] = defaultdict(list)
    incremental: dict[str, list[float]] = defaultdict(list)
    for row in measured:
        for pair, value in row["candidate_accounting"]["pairwise_candidate_jaccard"].items():
            overlap_pairs[pair].append(value)
        for condition, ids in row["candidate_accounting"]["incremental_gold_vs_bm25"].items():
            incremental[condition].append(float(bool(ids)))
    source = {
        "dataset_path": str(dataset_path.resolve()) if dataset_path else None,
        "dataset_sha256": _sha256(dataset_path) if dataset_path else None,
        "samples": len(data),
        "split": split,
    }
    config = {
        "candidate_budget": candidate_budget,
        "final_k": final_k,
        "rrf_k": 60,
        "split_seed": split_seed,
        "sham_seed": seed,
        "graph": "turn-term-speaker-session weighted undirected graph; PPR alpha=0.15, 40 iterations",
        "tokenizer": TOKENIZER,
        "stopwords": sorted(STOPWORDS),
    }
    host = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    report = {
        "schema_version": SCHEMA,
        "classification": "exploratory_offline_mechanism_ablation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source": source,
        "execution": {
            "external_network_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
            "actual_external_cost_usd": 0.0,
            "provider_calls": 0,
        },
        "conditions": list(CONDITIONS),
        "metrics": metrics,
        "paired_deltas": paired_deltas,
        "candidate_accounting": {
            "mean_pairwise_candidate_jaccard": {pair: float(np.mean(values)) for pair, values in sorted(overlap_pairs.items())},
            "fraction_questions_with_incremental_gold_vs_bm25": {condition: float(np.mean(values)) for condition, values in sorted(incremental.items())},
        },
        "rows": rows,
        "failure_ledger": failure_ledger,
        "graph_index": {
            "footprints": graph_footprints,
            "build_ms_mean": float(np.mean(build_ms)) if build_ms else 0.0,
            "query_ms_mean": float(np.mean(query_ms)) if query_ms else 0.0,
            "all_shams_degree_preserving": all(
                footprint["degree_preserved"] for footprint in graph_footprints
            ),
            "mean_sham_retained_term_turn_fraction": float(np.mean([
                footprint["sham_effectiveness"]["retained_term_turn_fraction"]
                for footprint in graph_footprints
            ])) if graph_footprints else None,
        },
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("bm25s", "numpy", "scipy")
            if _package_available(package)
        },
        "host": host,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_status_sha256": hashlib.sha256((_git("status", "--porcelain=v1") or "").encode()).hexdigest(),
        "source_script_sha256": _sha256(Path(__file__).resolve()),
        "selection_or_promotion": False,
        "hipporag_reproduction": False,
        "interpretation_limits": [
            "This is an offline associative-graph mechanism ablation, not a HippoRAG reproduction or SOTA claim.",
            "Graph construction and query anchors use corpus/query text and known speaker metadata only; gold evidence is metric-only.",
            "LoCoMo conversations are indexed independently; results do not establish cross-user or persistent-memory behavior.",
            "Synthetic degree-preserving sham controls degree/popularity but does not prove causality; all conditions share a fixed candidate budget.",
            "No model, provider, reranker, reader, or external network call was used.",
            "This does not reproduce HippoRAG: no LLM entity extraction, trained retriever, or HippoRAG benchmark protocol is implemented.",
        ],
    }
    report["host"]["machine_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(host, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _package_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _create_once(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses to overwrite an existing
        # artifact, preserving the immutable experiment ledger.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--split-seed", default="20260822")
    parser.add_argument("--candidate-budget", type=int, default=25)
    parser.add_argument("--final-k", type=int, default=10)
    args = parser.parse_args(argv)
    dataset = args.dataset.resolve()
    data = json.loads(dataset.read_text(encoding="utf-8"))
    report = evaluate_data(
        data, dataset_path=dataset, seed=args.seed, split_seed=args.split_seed,
        candidate_budget=args.candidate_budget, final_k=args.final_k,
    )
    _create_once(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "provider_calls": 0, "conditions": report["conditions"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
