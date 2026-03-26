"""
Phase 2 sweep: weight sweep (β=0.4–0.8) × density sweep (75, 150, 375 edges).

Conventions match scripts/multi_domain_eval.py:
  - clear cache before every measured search call
  - read embeddings from SQLite with np.frombuffer(bytes, np.float32)
  - create/delete edges via REST API
  - append results to benchmarks/multi_domain_results.json

Run from repo root:  python scripts/phase2_sweep.py
"""
import json
import os
import random
import sqlite3
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_URL = "http://127.0.0.1:8000"
DB_PATH = ROOT / "data" / "hybridmind.mind" / "store.db"
RESULTS_FILE = ROOT / "benchmarks" / "multi_domain_results.json"
REPORT_FILE = ROOT / "docs" / "MULTI_DOMAIN_EVAL.md"

WEIGHT_BETAS = [0.4, 0.5, 0.6, 0.7, 0.8]
DENSITY_TARGETS = [75, 150, 375]           # edge counts = ~1%, 2%, 5% of 7510 nodes
DENSITY_TEST_BETAS = [0.5, 0.7, 0.8]      # betas to test at each density level
TOP_K = 12                                  # top-k for both vector + hybrid (extra 2 let us detect rank-11/12 promotions)
CONCEPT_QUERIES = [
    "optimization algorithms for convergence",
    "neural network architecture design",
    "statistical inference and uncertainty",
    "distributed systems and fault tolerance",
    "protein folding and molecular structure",
    "regulatory compliance and risk assessment",
    "gradient descent and loss functions",
    "natural language understanding",
    "clinical trials and treatment efficacy",
    "market dynamics and price prediction",
]

RNG = random.Random(42)


# ── helpers ──────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def short(text: str, n: int = 140) -> str:
    return text[:n] if text else ""


def count_domains(results: List[Dict]) -> Dict[str, int]:
    c: Counter = Counter(r.get("metadata", {}).get("domain", "?") for r in results)
    return dict(sorted(c.items()))


def score_of(r: Dict) -> float:
    return float(r.get("combined_score") or r.get("vector_score") or 0.0)


# ── DB embedding loader ───────────────────────────────────────────────────────

class EmbeddingStore:
    """Loads all node embeddings from SQLite once; provides cosine computations."""

    def __init__(self, db_path: Path):
        self.nodes: Dict[str, Dict] = {}  # id -> {text, metadata, emb}
        self.by_domain: Dict[str, List[str]] = defaultdict(list)
        self._load(db_path)

    def _load(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, text, metadata, embedding FROM nodes")
        for row in cur.fetchall():
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            emb = np.frombuffer(bytes(row["embedding"]), dtype=np.float32).copy() if row["embedding"] else None
            if emb is None:
                continue
            node_id = row["id"]
            self.nodes[node_id] = {"text": row["text"] or "", "metadata": meta, "emb": emb}
            domain = meta.get("domain", "unknown")
            self.by_domain[domain].append(node_id)
        conn.close()
        print(f"[EmbeddingStore] loaded {len(self.nodes)} nodes, domains: {dict((d, len(ids)) for d, ids in self.by_domain.items())}")

    def top_cross_domain_pairs(
        self,
        domain_a: str,
        domain_b: str,
        sample_a: int = 300,
        sample_b: int = 300,
    ) -> List[Tuple[str, str, float]]:
        """Return (id_a, id_b, cosine) sorted descending for cross-domain pairs."""
        ids_a = RNG.sample(self.by_domain[domain_a], k=min(sample_a, len(self.by_domain[domain_a])))
        ids_b = RNG.sample(self.by_domain[domain_b], k=min(sample_b, len(self.by_domain[domain_b])))
        mat_a = np.stack([self.nodes[i]["emb"] for i in ids_a])  # (A, D)
        mat_b = np.stack([self.nodes[i]["emb"] for i in ids_b])  # (B, D)
        # Normalise for cosine via dot product
        na = np.linalg.norm(mat_a, axis=1, keepdims=True)
        nb = np.linalg.norm(mat_b, axis=1, keepdims=True)
        mat_a_n = mat_a / np.where(na == 0, 1, na)
        mat_b_n = mat_b / np.where(nb == 0, 1, nb)
        sims = mat_a_n @ mat_b_n.T  # (A, B) cosine matrix
        pairs = []
        for i, id_a in enumerate(ids_a):
            for j, id_b in enumerate(ids_b):
                pairs.append((id_a, id_b, float(sims[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


# ── REST client ───────────────────────────────────────────────────────────────

class APIClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self.c = httpx.Client(timeout=timeout)

    def _j(self, r: httpx.Response) -> Any:
        r.raise_for_status()
        return r.json()

    def clear_cache(self) -> None:
        self.c.post(f"{self.base}/cache/clear")

    def vector_search(self, query: str, top_k: int = 12, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        self.clear_cache()
        payload: Dict[str, Any] = {"query_text": query, "top_k": top_k}
        if filter_metadata:
            payload["filter_metadata"] = filter_metadata
        return self._j(self.c.post(f"{self.base}/search/vector", json=payload)).get("results", [])

    def hybrid_search(self, query: str, top_k: int = 12, vector_weight: float = 0.6, graph_weight: float = 0.4) -> List[Dict]:
        self.clear_cache()
        payload = {
            "query_text": query,
            "top_k": top_k,
            "vector_weight": vector_weight,
            "graph_weight": graph_weight,
        }
        return self._j(self.c.post(f"{self.base}/search/hybrid", json=payload)).get("results", [])

    def create_edge(self, source_id: str, target_id: str, weight: float, etype: str = "analogous_to") -> str:
        r = self._j(self.c.post(f"{self.base}/edges", json={
            "source_id": source_id, "target_id": target_id, "type": etype, "weight": weight
        }))
        return r["id"]

    def delete_all_edges(self) -> int:
        deleted = 0
        skip = 0
        while True:
            batch = self._j(self.c.get(f"{self.base}/edges", params={"skip": skip, "limit": 1000}))
            if not batch:
                break
            for edge in batch:
                self.c.delete(f"{self.base}/edges/{edge['id']}").raise_for_status()
                deleted += 1
            skip += len(batch)
            if len(batch) < 1000:
                break
        self.clear_cache()
        return deleted

    def edge_count(self) -> int:
        stats = self._j(self.c.get(f"{self.base}/search/stats"))
        return int(stats.get("total_edges", 0))

    def health(self) -> Dict:
        return self._j(self.c.get(f"{self.base}/health"))


# ── sweep logic ───────────────────────────────────────────────────────────────

def run_query_at_weight(
    api: APIClient,
    query: str,
    beta: float,
    vector_results: List[Dict],  # pre-fetched top-12 vector results (cache already cleared)
) -> Dict[str, Any]:
    """Compare hybrid@beta against already-fetched vector results for one query."""
    alpha = round(1.0 - beta, 4)
    h_results = api.hybrid_search(query, top_k=TOP_K, vector_weight=alpha, graph_weight=beta)

    v_ids = [r["node_id"] for r in vector_results]
    h_ids = [r["node_id"] for r in h_results]
    v_top10 = set(v_ids[:10])
    h_top10 = set(h_ids[:10])

    set_diff = len(h_top10 - v_top10)              # nodes in hybrid top-10 but not vector top-10
    top1_changed = (v_ids[:1] != h_ids[:1])
    domain_dist_changed = (count_domains(vector_results[:10]) != count_domains(h_results[:10]))

    # rank promotion: which nodes moved from position 11–12 in vector into hybrid top-10?
    promoted = [nid for nid in h_ids[:10] if nid in v_ids[10:12]]
    # rank demotion: which nodes fell from vector top-10 to outside hybrid top-10?
    demoted = [nid for nid in v_ids[:10] if nid not in h_top10]

    return {
        "beta": beta,
        "alpha": alpha,
        "set_diff": set_diff,
        "top1_changed": top1_changed,
        "domain_dist_changed": domain_dist_changed,
        "promoted_nodes": promoted,        # entered top-10 in hybrid but outside in vector
        "demoted_nodes": demoted,           # fell out of top-10 in hybrid
        "hybrid_top1": {
            "node_id": h_results[0]["node_id"] if h_results else None,
            "domain": h_results[0].get("metadata", {}).get("domain") if h_results else None,
            "score": round(score_of(h_results[0]), 4) if h_results else 0.0,
        },
        "hybrid_domain_dist": count_domains(h_results[:10]),
    }


def run_weight_sweep(api: APIClient) -> Dict[str, Any]:
    """
    For each of the 10 concept queries: fetch vector results once, then run hybrid
    at each beta value. Records set differences, domain changes, rank promotions.
    """
    print("\n=== WEIGHT SWEEP ===")
    print(f"Betas: {WEIGHT_BETAS} | Queries: {len(CONCEPT_QUERIES)} | top_k={TOP_K}")

    results_by_query: List[Dict[str, Any]] = []

    for query in tqdm(CONCEPT_QUERIES, desc="WeightSweep"):
        # Fetch vector baseline once (top-12 to see rank-11/12 candidates)
        v_results = api.vector_search(query, top_k=TOP_K)
        v_top10_dist = count_domains(v_results[:10])
        v_top1 = {"node_id": v_results[0]["node_id"] if v_results else None,
                  "domain": v_results[0].get("metadata", {}).get("domain") if v_results else None,
                  "score": round(score_of(v_results[0]), 4) if v_results else 0.0}
        # Score gap between rank-10 and rank-11 (proxy for crossover difficulty)
        score_gap = None
        if len(v_results) >= 11:
            score_gap = round(score_of(v_results[9]) - score_of(v_results[10]), 5)

        per_beta = []
        for beta in WEIGHT_BETAS:
            row = run_query_at_weight(api, query, beta, v_results)
            per_beta.append(row)

        first_diff_beta = next((r["beta"] for r in per_beta if r["set_diff"] > 0), None)
        any_diff = any(r["set_diff"] > 0 for r in per_beta)

        results_by_query.append({
            "query": query,
            "vector_top1": v_top1,
            "vector_top10_domain_dist": v_top10_dist,
            "v_score_gap_rank10_11": score_gap,
            "per_beta": per_beta,
            "first_beta_with_diff": first_diff_beta,
            "any_diff_across_betas": any_diff,
        })
        diff_summary = {b["beta"]: b["set_diff"] for b in per_beta}
        tqdm.write(f"  {query[:50]:<50} | gap={score_gap} | diffs={diff_summary}")

    # Summary stats
    crossover_betas = [r["first_beta_with_diff"] for r in results_by_query if r["first_beta_with_diff"] is not None]
    n_any_diff = sum(1 for r in results_by_query if r["any_diff_across_betas"])
    n_per_beta = {b: sum(1 for r in results_by_query if any(pr["set_diff"] > 0 for pr in r["per_beta"] if pr["beta"] == b))
                  for b in WEIGHT_BETAS}

    return {
        "betas_tested": WEIGHT_BETAS,
        "queries_tested": len(CONCEPT_QUERIES),
        "top_k": TOP_K,
        "n_queries_with_any_diff": n_any_diff,
        "n_queries_with_diff_per_beta": n_per_beta,
        "crossover_betas": crossover_betas,
        "min_crossover_beta": min(crossover_betas) if crossover_betas else None,
        "queries": results_by_query,
    }


def build_graph_to_density(
    api: APIClient,
    store: EmbeddingStore,
    target_edges: int,
    domains: List[str],
    label: str = "",
) -> Dict[str, Any]:
    """
    Delete all existing edges, then build a cross-domain graph by taking the
    top-N highest-cosine cross-domain pairs globally (pool all domain-pair
    similarities, sort descending, pick top target_edges).

    Returns graph construction metadata.
    """
    print(f"\n  [Density] building graph for target={target_edges} edges {label}")
    deleted = api.delete_all_edges()
    print(f"  deleted {deleted} old edges")

    # Collect all cross-domain pairs across all domain pairs
    all_pairs: List[Tuple[str, str, float]] = []
    for domain_a, domain_b in combinations(sorted(domains), 2):
        pairs = store.top_cross_domain_pairs(domain_a, domain_b, sample_a=300, sample_b=300)
        # Deduplicate (source_id, target_id) — pairs are unique within a pair call
        all_pairs.extend(pairs)

    # Sort globally by cosine descending
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    # Dedup to avoid duplicate (A,B) or (B,A) edges for the same pair of nodes
    seen: set = set()
    deduped: List[Tuple[str, str, float]] = []
    for src, tgt, sim in all_pairs:
        key = (min(src, tgt), max(src, tgt))
        if key not in seen:
            seen.add(key)
            deduped.append((src, tgt, sim))

    # Take top-target_edges
    chosen = deduped[:target_edges]
    min_sim = round(chosen[-1][2], 5) if chosen else 0.0
    max_sim = round(chosen[0][2], 5) if chosen else 0.0

    # Create edges via API
    edge_ids = []
    domain_pair_counts: Counter = Counter()
    for src_id, tgt_id, sim in tqdm(chosen, desc=f"  CreateEdges@{target_edges}", leave=False):
        src_domain = store.nodes[src_id]["metadata"].get("domain", "?")
        tgt_domain = store.nodes[tgt_id]["metadata"].get("domain", "?")
        if src_domain == tgt_domain:
            continue  # skip same-domain (shouldn't happen but guard)
        pair_key = "-".join(sorted([src_domain, tgt_domain]))
        domain_pair_counts[pair_key] += 1
        eid = api.create_edge(src_id, tgt_id, sim)
        edge_ids.append(eid)

    actual = api.edge_count()
    print(f"  actual edges in index: {actual} (requested {target_edges}, created {len(edge_ids)})")

    return {
        "target_edges": target_edges,
        "actual_edges": actual,
        "created_edges": len(edge_ids),
        "min_cosine": min_sim,
        "max_cosine": max_sim,
        "domain_pair_counts": dict(sorted(domain_pair_counts.items())),
    }


def run_density_sweep(api: APIClient, store: EmbeddingStore) -> Dict[str, Any]:
    """
    For each target edge density: build graph, then test all CONCEPT_QUERIES
    with DENSITY_TEST_BETAS. Records whether hybrid diverges from vector.
    """
    print("\n=== DENSITY SWEEP ===")
    print(f"Targets: {DENSITY_TARGETS} edges | Betas: {DENSITY_TEST_BETAS}")

    domains = [d for d in store.by_domain if d not in ("probe",) and store.by_domain[d]]
    results_per_density: List[Dict[str, Any]] = []

    # Fetch vector baselines once (reuse across density/beta combos — same nodes, same embeddings)
    print("\n  Fetching vector baselines for all queries...")
    vector_baselines: Dict[str, List[Dict]] = {}
    for query in tqdm(CONCEPT_QUERIES, desc="VectorBaseline"):
        vector_baselines[query] = api.vector_search(query, top_k=TOP_K)

    for target in DENSITY_TARGETS:
        graph_info = build_graph_to_density(api, store, target, domains, label=f"~{round(target/7510*100,1)}%")

        per_beta_results: List[Dict[str, Any]] = []
        for beta in DENSITY_TEST_BETAS:
            alpha = round(1.0 - beta, 4)
            query_rows = []
            for query in tqdm(CONCEPT_QUERIES, desc=f"  β={beta}@{target}edges", leave=False):
                v_results = vector_baselines[query]
                row = run_query_at_weight(api, query, beta, v_results)
                query_rows.append({"query": query, **row})

            n_set_diff = sum(1 for r in query_rows if r["set_diff"] > 0)
            n_domain_diff = sum(1 for r in query_rows if r["domain_dist_changed"])
            n_top1_diff = sum(1 for r in query_rows if r["top1_changed"])
            per_beta_results.append({
                "beta": beta,
                "alpha": alpha,
                "n_queries_set_diff": n_set_diff,
                "n_queries_domain_dist_changed": n_domain_diff,
                "n_queries_top1_changed": n_top1_diff,
                "queries": query_rows,
            })
            print(f"    density={target} β={beta}: set_diff={n_set_diff}/10, domain_diff={n_domain_diff}/10, top1_diff={n_top1_diff}/10")

        results_per_density.append({
            "target_edges": target,
            "graph": graph_info,
            "per_beta": per_beta_results,
        })

    # Restore original 6-edge graph after sweep
    print("\n  Restoring original graph (6 edges at 0.45 threshold)...")
    api.delete_all_edges()
    # Re-create the 6 original edges from stored results
    try:
        prev = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        gc = prev.get("graph_construction", {})
        for ex in gc.get("pair_best_examples", {}).values():
            if ex and ex.get("source_id") and ex.get("target_id"):
                api.create_edge(ex["source_id"], ex["target_id"], ex.get("weight", 0.5))
        print(f"  restored {api.edge_count()} edges")
    except Exception as exc:
        print(f"  [warn] could not restore original edges: {exc}")

    # Build summary table
    summary: List[Dict] = []
    for d_result in results_per_density:
        target = d_result["target_edges"]
        for b_result in d_result["per_beta"]:
            summary.append({
                "edges": target,
                "density_pct": round(target / 7510 * 100, 2),
                "beta": b_result["beta"],
                "n_set_diff": b_result["n_queries_set_diff"],
                "n_domain_diff": b_result["n_queries_domain_dist_changed"],
                "n_top1_diff": b_result["n_queries_top1_changed"],
            })

    return {
        "density_targets": DENSITY_TARGETS,
        "betas_tested": DENSITY_TEST_BETAS,
        "queries_tested": len(CONCEPT_QUERIES),
        "top_k": TOP_K,
        "per_density": results_per_density,
        "summary_table": summary,
    }


# ── report generation ─────────────────────────────────────────────────────────

def build_weight_sweep_table(ws: Dict) -> str:
    lines = [
        "### Weight Sweep — top-10 set differences vs vector-only (10 concept queries, top_k=12)",
        "",
        "| Query | v score gap (r10–r11) | β=0.4 Δ | β=0.5 Δ | β=0.6 Δ | β=0.7 Δ | β=0.8 Δ | First crossover β |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ws["queries"]:
        q = r["query"][:42]
        gap = r["v_score_gap_rank10_11"]
        gap_str = f"{gap:.5f}" if gap is not None else "—"
        diffs = {pr["beta"]: pr["set_diff"] for pr in r["per_beta"]}
        first = r["first_beta_with_diff"]
        first_str = str(first) if first is not None else "none"
        row = f"| {q} | {gap_str} | {diffs.get(0.4,0)} | {diffs.get(0.5,0)} | {diffs.get(0.6,0)} | {diffs.get(0.7,0)} | {diffs.get(0.8,0)} | {first_str} |"
        lines.append(row)
    lines += [
        "",
        f"**Queries with any set diff across all betas:** {ws['n_queries_with_any_diff']}/{ws['queries_tested']}",
        "",
        "| β | Queries with ≥1 set diff |",
        "| --- | ---: |",
    ]
    for b in ws["betas_tested"]:
        n = ws["n_queries_with_diff_per_beta"].get(b, 0)
        lines.append(f"| {b} | {n}/{ws['queries_tested']} |")
    return "\n".join(lines)


def build_density_sweep_table(ds: Dict) -> str:
    lines = [
        "### Density Sweep — hybrid vs vector divergence at different edge densities",
        "",
        "| Edges | Density | β | Set-diff queries | Domain-dist changes | Top-1 changes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ds["summary_table"]:
        lines.append(
            f"| {row['edges']} | {row['density_pct']:.2f}% | {row['beta']} "
            f"| {row['n_set_diff']}/10 | {row['n_domain_diff']}/10 | {row['n_top1_diff']}/10 |"
        )
    lines += [
        "",
        "**Set-diff = nodes in hybrid top-10 not present in vector top-10 for the same query.**",
        "**A non-zero value means hybrid retrieved at least one node vector would have missed.**",
    ]
    return "\n".join(lines)


def build_promoted_nodes_table(ws: Dict) -> str:
    """Show which specific nodes were promoted into top-10 at each beta."""
    promoted_entries = []
    for r in ws["queries"]:
        for pr in r["per_beta"]:
            if pr["promoted_nodes"]:
                promoted_entries.append({
                    "query": r["query"][:50],
                    "beta": pr["beta"],
                    "count": len(pr["promoted_nodes"]),
                    "top1_changed": pr["top1_changed"],
                    "hybrid_top1_domain": pr["hybrid_top1"]["domain"],
                })
    if not promoted_entries:
        return "No rank promotions detected across any β value."
    lines = [
        "### Rank Promotions (nodes entering hybrid top-10 from position 11–12 in vector)",
        "",
        "| Query | β | Promoted | Top-1 changed | Hybrid top-1 domain |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for e in promoted_entries:
        lines.append(f"| {e['query']} | {e['beta']} | {e['count']} | {e['top1_changed']} | {e['hybrid_top1_domain']} |")
    return "\n".join(lines)


def append_to_report(ws: Dict, ds: Dict, edge_state_before: int) -> None:
    existing = REPORT_FILE.read_text(encoding="utf-8")
    marker = "\n\n---\n\n## Appendix: Phase 2 Sweep Results"
    existing = existing.split(marker)[0]  # strip old appendix if re-running

    weight_table = build_weight_sweep_table(ws)
    density_table = build_density_sweep_table(ds)
    promoted_table = build_promoted_nodes_table(ws)

    min_cross = ws.get("min_crossover_beta")
    cross_str = str(min_cross) if min_cross is not None else "none found (≥β=0.8 tested)"
    n_any = ws["n_queries_with_any_diff"]

    appendix = f"""

---

## Appendix: Phase 2 Sweep Results

*Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}. Server: {BASE_URL}. Nodes: 7510 (edge state before sweep: {edge_state_before}).*

### A.1 Weight Sweep

**Setup:** {ws['queries_tested']} concept queries × {len(ws['betas_tested'])} β values. Vector search run once per query (top_k={ws['top_k']}); hybrid run with matching α = 1−β. Cache cleared before every call.

**Crossover β (first weight where hybrid top-10 ≠ vector top-10):** {cross_str}

{weight_table}

{promoted_table}

### A.2 Density Sweep

**Setup:** Graph rebuilt at 3 edge-count targets ({', '.join(str(t) for t in ds['density_targets'])} edges ≈ {', '.join(f"{round(t/7510*100,1)}%" for t in ds['density_targets'])} node-density). Edges chosen as global top-N cross-domain pairs by cosine similarity (300 nodes sampled per domain per pair; deduplicated). β tested: {ds['betas_tested']}.

{density_table}

### A.3 Interpretation

- **Crossover β** is the minimum graph weight at which the graph component overrides vector ranking for ≥1 query.
- If no crossover found at β=0.8, the CRS graph score magnitude is insufficient to beat the vector score gap at this node count and edge density.
- The density sweep isolates whether the issue is edge sparsity (more edges = more nodes with non-zero graph score) or weight magnitude.
- **Key question:** if set_diff = 0 at all (density, β) combinations, the graph component's score contribution is structurally below the vector score gaps in this corpus, and the architecture needs recalibration.
"""
    REPORT_FILE.write_text(existing + appendix, encoding="utf-8")
    print(f"\nReport appended to {REPORT_FILE}")


def save_to_json(ws: Dict, ds: Dict) -> None:
    prev = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    prev["phase2_sweep"] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weight_sweep": ws,
        "density_sweep": ds,
    }
    RESULTS_FILE.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"Results saved to {RESULTS_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()
    api = APIClient()

    # Verify server
    health = api.health()
    nodes_total = health["components"]["database"]["nodes"]
    edges_before = health["components"]["graph_index"]["edges"]
    print(f"Server OK — nodes: {nodes_total}, edges: {edges_before}")
    if nodes_total < 100:
        sys.exit("ERROR: too few nodes. Run multi_domain_eval.py first.")

    # Load all embeddings from SQLite once
    store = EmbeddingStore(DB_PATH)

    # Phase 1: Weight sweep on current graph state (6 edges, 7510 nodes)
    ws_results = run_weight_sweep(api)

    # Phase 2: Density sweep (rebuilds + restores graph)
    ds_results = run_density_sweep(api, store)

    # Save
    save_to_json(ws_results, ds_results)
    append_to_report(ws_results, ds_results, edges_before)

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\nDone in {elapsed}s")

    # Print compact summary
    print("\n=== WEIGHT SWEEP SUMMARY ===")
    print(f"{'Query':<42} | {'gap':>8} | " + " | ".join(f"β={b}" for b in WEIGHT_BETAS))
    print("-" * 100)
    for r in ws_results["queries"]:
        gap = r["v_score_gap_rank10_11"]
        diffs = [str(next(pr["set_diff"] for pr in r["per_beta"] if pr["beta"] == b)) for b in WEIGHT_BETAS]
        print(f"{r['query'][:42]:<42} | {str(gap or '?'):>8} | " + " | ".join(f"{d:>5}" for d in diffs))

    print("\n=== DENSITY SWEEP SUMMARY ===")
    print(f"{'Edges':>6} | {'Density':>8} | {'β':>5} | {'SetDiff':>8} | {'DomDiff':>8} | {'Top1Diff':>9}")
    print("-" * 60)
    for row in ds_results["summary_table"]:
        print(f"{row['edges']:>6} | {row['density_pct']:>7.2f}% | {row['beta']:>5} | {row['n_set_diff']:>8}/10 | {row['n_domain_diff']:>8}/10 | {row['n_top1_diff']:>9}/10")


if __name__ == "__main__":
    main()
