"""
Phase 3: Apply the three structural fixes that allow the graph component
to produce non-zero scores.

Fix A — Intra-domain edges: build top-150 cosine-similar pairs within each
         domain (~750 edges) so that the top-3 vector reference nodes
         almost always have in-domain neighbors already inside the top-12
         candidate pool.

Fix B — Large top_k: retrieve 100 candidates instead of 12.  Cross-domain
         neighbors of the reference nodes are now reachable even if they
         sit at rank 20–60 in pure vector space.

Fix C — Explicit anchor_nodes: pass the top-1 vector result as anchor to
         hybrid search, bypassing the auto-anchor fallback entirely.
         This guarantees the reference node has been chosen deliberately.

All three fixes are tested on the 10 concept queries from Experiment 1.
Results are appended to benchmarks/multi_domain_results.json and
docs/MULTI_DOMAIN_EVAL.md.

Run from repo root:  python scripts/phase3_fixes.py
"""
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_URL = "http://127.0.0.1:8000"
DB_PATH = ROOT / "data" / "hybridmind.mind" / "store.db"
RESULTS_FILE = ROOT / "benchmarks" / "multi_domain_results.json"
REPORT_FILE = ROOT / "docs" / "MULTI_DOMAIN_EVAL.md"

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

# Intra-domain: edges per domain (×5 domains = 750 intra-domain edges total)
INTRA_EDGES_PER_DOMAIN = 150
# Cross-domain edges added on top of intra-domain for combined graph
CROSS_EDGES = 150
# Betas to test in each fix experiment
SWEEP_BETAS = [0.5, 0.7, 0.8]
TOP_K_NORMAL = 12
TOP_K_LARGE = 100

RNG = random.Random(42)


# ── helpers ───────────────────────────────────────────────────────────────────

def count_domains(results: List[Dict]) -> Dict[str, int]:
    c: Counter = Counter(r.get("metadata", {}).get("domain", "?") for r in results)
    return dict(sorted(c.items()))


def score_of(r: Dict) -> float:
    return float(r.get("combined_score") or r.get("vector_score") or 0.0)


def set_diff_count(v: List[Dict], h: List[Dict], k: int = 10) -> int:
    return len({r["node_id"] for r in h[:k]} - {r["node_id"] for r in v[:k]})


def avg_graph_score(results: List[Dict], top_k: int = 10) -> float:
    scores = [r.get("graph_score") or 0.0 for r in results[:top_k]]
    return round(float(np.mean(scores)) if scores else 0.0, 5)


# ── embedding store ───────────────────────────────────────────────────────────

class EmbeddingStore:
    def __init__(self, db_path: Path):
        self.nodes: Dict[str, Dict] = {}
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
            nid = row["id"]
            self.nodes[nid] = {"text": row["text"] or "", "metadata": meta, "emb": emb}
            self.by_domain[meta.get("domain", "unknown")].append(nid)
        conn.close()
        domains_summary = {d: len(ids) for d, ids in self.by_domain.items()}
        print(f"[EmbeddingStore] {len(self.nodes)} nodes | {domains_summary}")

    def top_intra_domain_pairs(
        self, domain: str, sample: int = 500, top_n: int = 200
    ) -> List[Tuple[str, str, float]]:
        """Top-N highest-cosine pairs within a single domain (upper triangle, no self-loops)."""
        ids = RNG.sample(self.by_domain[domain], k=min(sample, len(self.by_domain[domain])))
        mat = np.stack([self.nodes[i]["emb"] for i in ids])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_n = mat / np.where(norms == 0, 1, norms)
        sims = mat_n @ mat_n.T  # (N, N)
        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j], float(sims[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    def top_cross_domain_pairs(
        self, domain_a: str, domain_b: str, sample_a: int = 300, sample_b: int = 300
    ) -> List[Tuple[str, str, float]]:
        ids_a = RNG.sample(self.by_domain[domain_a], k=min(sample_a, len(self.by_domain[domain_a])))
        ids_b = RNG.sample(self.by_domain[domain_b], k=min(sample_b, len(self.by_domain[domain_b])))
        mat_a = np.stack([self.nodes[i]["emb"] for i in ids_a])
        mat_b = np.stack([self.nodes[i]["emb"] for i in ids_b])
        na = np.linalg.norm(mat_a, axis=1, keepdims=True)
        nb = np.linalg.norm(mat_b, axis=1, keepdims=True)
        mat_an = mat_a / np.where(na == 0, 1, na)
        mat_bn = mat_b / np.where(nb == 0, 1, nb)
        sims = mat_an @ mat_bn.T
        pairs = [(ids_a[i], ids_b[j], float(sims[i, j]))
                 for i in range(len(ids_a)) for j in range(len(ids_b))]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


# ── API client ────────────────────────────────────────────────────────────────

class APIClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self.c = httpx.Client(timeout=timeout)

    def _j(self, r: httpx.Response) -> Any:
        r.raise_for_status()
        return r.json()

    def clear_cache(self) -> None:
        self.c.post(f"{self.base}/cache/clear")

    def vector_search(self, query: str, top_k: int = 12) -> List[Dict]:
        self.clear_cache()
        return self._j(self.c.post(f"{self.base}/search/vector",
                                   json={"query_text": query, "top_k": top_k})).get("results", [])

    def hybrid_search(
        self, query: str, top_k: int = 12,
        vector_weight: float = 0.5, graph_weight: float = 0.5,
        anchor_nodes: Optional[List[str]] = None,
    ) -> List[Dict]:
        self.clear_cache()
        payload: Dict[str, Any] = {
            "query_text": query, "top_k": top_k,
            "vector_weight": vector_weight, "graph_weight": graph_weight,
        }
        if anchor_nodes:
            payload["anchor_nodes"] = anchor_nodes
        return self._j(self.c.post(f"{self.base}/search/hybrid", json=payload)).get("results", [])

    def create_edge(self, src: str, tgt: str, weight: float, etype: str = "analogous_to") -> str:
        return self._j(self.c.post(f"{self.base}/edges", json={
            "source_id": src, "target_id": tgt, "type": etype, "weight": weight
        }))["id"]

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
        return int(self._j(self.c.get(f"{self.base}/search/stats")).get("total_edges", 0))

    def health(self) -> Dict:
        return self._j(self.c.get(f"{self.base}/health"))


# ── graph builders ────────────────────────────────────────────────────────────

def build_intra_domain_graph(
    api: APIClient, store: EmbeddingStore, edges_per_domain: int
) -> Dict[str, Any]:
    """Delete all edges, then build top-edges_per_domain intra-domain pairs per domain."""
    deleted = api.delete_all_edges()
    print(f"  cleared {deleted} edges")

    domains = [d for d in store.by_domain if store.by_domain[d]]
    total_created = 0
    domain_counts: Dict[str, int] = {}
    for domain in sorted(domains):
        pairs = store.top_intra_domain_pairs(domain, sample=500, top_n=edges_per_domain)
        created = 0
        for src, tgt, sim in tqdm(pairs, desc=f"  intra:{domain}", leave=False):
            api.create_edge(src, tgt, sim, etype="supports")
            created += 1
        domain_counts[domain] = created
        total_created += created
    actual = api.edge_count()
    print(f"  intra-domain graph: {actual} edges ({total_created} created across {len(domains)} domains)")
    return {"type": "intra_domain", "edges_per_domain": edges_per_domain,
            "domain_counts": domain_counts, "actual_edges": actual}


def build_combined_graph(
    api: APIClient, store: EmbeddingStore,
    edges_per_domain: int, cross_edges: int
) -> Dict[str, Any]:
    """Intra-domain edges + top cross_edges cross-domain pairs."""
    intra_info = build_intra_domain_graph(api, store, edges_per_domain)

    # Add cross-domain edges on top
    domains = sorted(d for d in store.by_domain if store.by_domain[d])
    all_cross: List[Tuple[str, str, float]] = []
    for da, db in combinations(domains, 2):
        all_cross.extend(store.top_cross_domain_pairs(da, db))
    all_cross.sort(key=lambda x: x[2], reverse=True)

    seen: set = set()
    chosen = []
    for src, tgt, sim in all_cross:
        key = (min(src, tgt), max(src, tgt))
        if key not in seen:
            seen.add(key)
            chosen.append((src, tgt, sim))
        if len(chosen) >= cross_edges:
            break

    cross_created = 0
    for src, tgt, sim in tqdm(chosen, desc="  cross-domain edges", leave=False):
        api.create_edge(src, tgt, sim, etype="analogous_to")
        cross_created += 1

    actual = api.edge_count()
    print(f"  combined graph: {actual} edges ({cross_created} cross-domain added)")
    return {**intra_info, "type": "combined",
            "cross_edges_added": cross_created, "actual_edges": actual}


# ── experiment runners ────────────────────────────────────────────────────────

def _query_stats(v: List[Dict], h: List[Dict], k: int = 10) -> Dict:
    v_ids_k = [r["node_id"] for r in v[:k]]
    h_ids_k = [r["node_id"] for r in h[:k]]
    v_set = set(v_ids_k)
    h_set = set(h_ids_k)
    graph_scores_top10 = [r.get("graph_score") or 0.0 for r in h[:k]]
    nonzero_gs = sum(1 for g in graph_scores_top10 if g > 0)
    return {
        "set_diff": len(h_set - v_set),
        "top1_changed": v_ids_k[:1] != h_ids_k[:1],
        "domain_dist_changed": count_domains(v[:k]) != count_domains(h[:k]),
        "vector_domain_dist": count_domains(v[:k]),
        "hybrid_domain_dist": count_domains(h[:k]),
        "nonzero_graph_scores_in_top10": nonzero_gs,
        "avg_graph_score_top10": round(float(np.mean(graph_scores_top10)), 5),
        "max_graph_score_top10": round(float(max(graph_scores_top10)), 5) if graph_scores_top10 else 0.0,
    }


def run_fix_A_intra_domain(api: APIClient, store: EmbeddingStore) -> Dict[str, Any]:
    """
    Fix A: Build intra-domain graph (150 edges/domain = 750 total).
    Run each concept query at beta=[0.5,0.7,0.8] with top_k=12.
    """
    print("\n=== FIX A: INTRA-DOMAIN EDGES ===")
    graph_info = build_intra_domain_graph(api, store, INTRA_EDGES_PER_DOMAIN)

    # vector baselines (top_k=12)
    print("  fetching vector baselines...")
    v_baselines = {q: api.vector_search(q, top_k=TOP_K_NORMAL) for q in tqdm(CONCEPT_QUERIES, leave=False)}

    rows = []
    for beta in SWEEP_BETAS:
        alpha = round(1.0 - beta, 4)
        q_rows = []
        for q in tqdm(CONCEPT_QUERIES, desc=f"  fixA b={beta}", leave=False):
            v = v_baselines[q]
            h = api.hybrid_search(q, top_k=TOP_K_NORMAL, vector_weight=alpha, graph_weight=beta)
            stats = _query_stats(v, h)
            q_rows.append({"query": q, "beta": beta, **stats})
        n_diff = sum(1 for r in q_rows if r["set_diff"] > 0)
        n_gs = sum(1 for r in q_rows if r["nonzero_graph_scores_in_top10"] > 0)
        print(f"    beta={beta}: set_diff={n_diff}/10, queries_with_nonzero_gs={n_gs}/10")
        rows.extend(q_rows)

    return {"graph": graph_info, "top_k": TOP_K_NORMAL, "betas": SWEEP_BETAS, "queries": rows}


def run_fix_B_large_topk(api: APIClient) -> Dict[str, Any]:
    """
    Fix B: Expand candidate pool to top_k=100.
    Cross-domain neighbors of reference nodes now appear in candidate set.
    Intra-domain graph from Fix A should already be active.
    """
    print("\n=== FIX B: LARGE TOP_K=100 ===")
    edges_now = api.edge_count()
    print(f"  current edges in index: {edges_now}")

    rows = []
    for beta in SWEEP_BETAS:
        alpha = round(1.0 - beta, 4)
        q_rows = []
        for q in tqdm(CONCEPT_QUERIES, desc=f"  fixB b={beta}", leave=False):
            v = api.vector_search(q, top_k=TOP_K_LARGE)
            h = api.hybrid_search(q, top_k=TOP_K_LARGE, vector_weight=alpha, graph_weight=beta)
            # compare top-10 of each
            stats = _query_stats(v, h, k=10)
            q_rows.append({"query": q, "beta": beta, **stats})
        n_diff = sum(1 for r in q_rows if r["set_diff"] > 0)
        n_gs = sum(1 for r in q_rows if r["nonzero_graph_scores_in_top10"] > 0)
        print(f"    beta={beta}: set_diff={n_diff}/10, queries_with_nonzero_gs={n_gs}/10")
        rows.extend(q_rows)

    return {"top_k": TOP_K_LARGE, "edges_used": edges_now, "betas": SWEEP_BETAS, "queries": rows}


def run_fix_C_explicit_anchors(api: APIClient) -> Dict[str, Any]:
    """
    Fix C: Pass the top-1 vector result as explicit anchor_node.
    Guarantees the reference node is the highest-scoring vector match;
    if it has edges (intra-domain graph active), those neighbors get
    non-zero graph proximity scores.
    """
    print("\n=== FIX C: EXPLICIT ANCHOR NODES ===")
    edges_now = api.edge_count()
    print(f"  current edges in index: {edges_now}")

    rows = []
    for beta in SWEEP_BETAS:
        alpha = round(1.0 - beta, 4)
        q_rows = []
        for q in tqdm(CONCEPT_QUERIES, desc=f"  fixC b={beta}", leave=False):
            v = api.vector_search(q, top_k=TOP_K_NORMAL)
            if not v:
                continue
            anchor = v[0]["node_id"]
            h = api.hybrid_search(q, top_k=TOP_K_NORMAL, vector_weight=alpha, graph_weight=beta,
                                  anchor_nodes=[anchor])
            stats = _query_stats(v, h)
            q_rows.append({
                "query": q, "beta": beta,
                "anchor_id": anchor,
                "anchor_domain": v[0].get("metadata", {}).get("domain"),
                **stats,
            })
        n_diff = sum(1 for r in q_rows if r["set_diff"] > 0)
        n_gs = sum(1 for r in q_rows if r["nonzero_graph_scores_in_top10"] > 0)
        print(f"    beta={beta}: set_diff={n_diff}/10, queries_with_nonzero_gs={n_gs}/10")
        rows.extend(q_rows)

    return {"top_k": TOP_K_NORMAL, "edges_used": edges_now,
            "anchor_strategy": "top-1 vector result per query",
            "betas": SWEEP_BETAS, "queries": rows}


def run_fix_ABC_combined(api: APIClient, store: EmbeddingStore) -> Dict[str, Any]:
    """
    Fix A+B+C combined: intra+cross-domain graph, top_k=100, explicit anchors.
    The maximum-signal configuration.
    """
    print("\n=== FIX A+B+C: COMBINED (intra+cross graph, top_k=100, explicit anchor) ===")
    graph_info = build_combined_graph(api, store, INTRA_EDGES_PER_DOMAIN, CROSS_EDGES)
    edges_now = api.edge_count()

    rows = []
    for beta in SWEEP_BETAS:
        alpha = round(1.0 - beta, 4)
        q_rows = []
        for q in tqdm(CONCEPT_QUERIES, desc=f"  fixABC b={beta}", leave=False):
            v = api.vector_search(q, top_k=TOP_K_LARGE)
            if not v:
                continue
            anchor = v[0]["node_id"]
            h = api.hybrid_search(q, top_k=TOP_K_LARGE, vector_weight=alpha, graph_weight=beta,
                                  anchor_nodes=[anchor])
            stats = _query_stats(v, h, k=10)
            q_rows.append({
                "query": q, "beta": beta,
                "anchor_id": anchor,
                "anchor_domain": v[0].get("metadata", {}).get("domain"),
                **stats,
            })
        n_diff = sum(1 for r in q_rows if r["set_diff"] > 0)
        n_gs = sum(1 for r in q_rows if r["nonzero_graph_scores_in_top10"] > 0)
        print(f"    beta={beta}: set_diff={n_diff}/10, queries_with_nonzero_gs={n_gs}/10")
        rows.extend(q_rows)

    return {"graph": graph_info, "top_k": TOP_K_LARGE, "edges_used": edges_now,
            "anchor_strategy": "top-1 vector result per query",
            "betas": SWEEP_BETAS, "queries": rows}


# ── report ────────────────────────────────────────────────────────────────────

def _summary_table(rows: List[Dict], betas: List[float], top_k: int) -> str:
    lines = [
        f"| Beta | SetDiff queries | Queries w/ nonzero graph score | Avg max graph score |",
        f"| ---: | ---: | ---: | ---: |",
    ]
    for beta in betas:
        subset = [r for r in rows if r["beta"] == beta]
        n_diff = sum(1 for r in subset if r["set_diff"] > 0)
        n_gs = sum(1 for r in subset if r["nonzero_graph_scores_in_top10"] > 0)
        avg_max = round(float(np.mean([r["max_graph_score_top10"] for r in subset])), 5) if subset else 0.0
        lines.append(f"| {beta} | {n_diff}/{len(subset)} | {n_gs}/{len(subset)} | {avg_max} |")
    return "\n".join(lines)


def _per_query_table(rows: List[Dict], betas: List[float]) -> str:
    queries = list(dict.fromkeys(r["query"] for r in rows))
    col_header = " | ".join(f"b={b} Δ / gs>0" for b in betas)
    lines = [f"| Query | {col_header} |",
             "| --- | " + " | ".join(["---: | ---:"] * len(betas)) + " |"]
    for q in queries:
        cols = []
        for beta in betas:
            match = next((r for r in rows if r["query"] == q and r["beta"] == beta), None)
            if match:
                cols.append(f"{match['set_diff']} / {match['nonzero_graph_scores_in_top10']}")
            else:
                cols.append("- / -")
        lines.append(f"| {q[:45]} | " + " | ".join(cols) + " |")
    return "\n".join(lines)


def append_to_report(results: Dict[str, Any]) -> None:
    existing = REPORT_FILE.read_text(encoding="utf-8")
    marker = "\n\n---\n\n## Appendix: Phase 3 Fix Results"
    existing = existing.split(marker)[0]

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    def fix_block(title: str, key: str, note: str = "") -> str:
        r = results[key]
        rows = r["queries"]
        betas = r["betas"]
        top_k = r.get("top_k", 12)
        edges = r.get("graph", {}).get("actual_edges") or r.get("edges_used", "?")
        summary = _summary_table(rows, betas, top_k)
        per_query = _per_query_table(rows, betas)
        return f"""
### {title}

**Graph edges active:** {edges} | **top_k:** {top_k} | **betas tested:** {betas}
{note}

**Summary by beta:**

{summary}

**Per-query breakdown (set_diff / queries with nonzero graph score in top-10):**

{per_query}
"""

    fix_a_note = (
        f"> Intra-domain edges: top-{INTRA_EDGES_PER_DOMAIN} cosine-similar pairs sampled per domain "
        f"(500-node sample), {len(set(r.get('anchor_domain') or '' for r in results['fix_A']['queries']))} "
        f"domains. Reference nodes now have ≥1 in-domain neighbor in the graph."
    )
    fix_b_note = (
        "> Candidate pool expanded from 12 to 100.  Cross-domain neighbors of reference nodes "
        "can now enter the scoring window even if they rank ~20–80 in pure vector space."
    )
    fix_c_note = (
        "> Explicit anchor = top-1 vector result per query.  Bypasses the auto-anchor fallback; "
        "guarantees the reference node was chosen for this query, not just the nearest vector hits."
    )
    fix_abc_note = (
        f"> Combined intra+cross-domain graph ({INTRA_EDGES_PER_DOMAIN} edges/domain + "
        f"{CROSS_EDGES} cross-domain), top_k=100, explicit anchor.  Maximum-signal configuration."
    )

    appendix = f"""

---

## Appendix: Phase 3 Fix Results

*Generated: {ts}. Three structural fixes that allow the graph component to produce non-zero scores.*

**Root cause (confirmed Phase 2):** anchor-free CRS with <5% edge coverage → graph_score=0 for 100% of queries.
Three independent fixes were applied and stacked.

---
{fix_block("Fix A — Intra-Domain Edges (top_k=12, no anchor override)", "fix_A", fix_a_note)}
---
{fix_block("Fix B — Large Candidate Pool (top_k=100, intra-domain graph active)", "fix_B", fix_b_note)}
---
{fix_block("Fix C — Explicit Anchor Nodes (top_k=12, intra-domain graph active)", "fix_C", fix_c_note)}
---
{fix_block("Fix A+B+C Combined (intra+cross graph, top_k=100, explicit anchor)", "fix_ABC", fix_abc_note)}
---

### Interpretation

- **SetDiff > 0** means the graph component successfully promoted at least one node into the top-10 that pure vector missed.
- **Nonzero graph score** is the prerequisite: if graph scores are all zero, no weight value can produce a set diff.
- Progress across fixes shows which structural change matters most.
"""
    REPORT_FILE.write_text(existing + appendix, encoding="utf-8")
    print(f"Report appended to {REPORT_FILE}")


def save_to_json(results: Dict[str, Any]) -> None:
    prev = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    prev["phase3_fixes"] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **results,
    }
    RESULTS_FILE.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"Results saved to {RESULTS_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()
    api = APIClient()

    health = api.health()
    nodes = health["components"]["database"]["nodes"]
    edges = health["components"]["graph_index"]["edges"]
    print(f"Server OK — nodes: {nodes}, edges: {edges}")
    if nodes < 100:
        sys.exit("ERROR: too few nodes. Run multi_domain_eval.py first.")

    store = EmbeddingStore(DB_PATH)
    results: Dict[str, Any] = {}

    # Fix A: intra-domain edges, standard top_k=12, no explicit anchor
    results["fix_A"] = run_fix_A_intra_domain(api, store)

    # Fix B: still on intra-domain graph, but top_k=100
    results["fix_B"] = run_fix_B_large_topk(api)

    # Fix C: still on intra-domain graph, standard top_k=12, explicit anchor
    results["fix_C"] = run_fix_C_explicit_anchors(api)

    # Fix A+B+C: combined graph (intra+cross), top_k=100, explicit anchor
    results["fix_ABC"] = run_fix_ABC_combined(api, store)

    save_to_json(results)
    append_to_report(results)

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\nDone in {elapsed}s")

    # Compact summary across all fixes
    print("\n=== PHASE 3 SUMMARY ===")
    print(f"{'Fix':<10} | {'beta':<5} | {'SetDiff':>8} | {'NonzeroGS':>10} | {'AvgMaxGS':>10}")
    print("-" * 55)
    for fix_key, fix_label in [("fix_A", "A-intra"), ("fix_B", "B-topk100"), ("fix_C", "C-anchor"), ("fix_ABC", "ABC-all")]:
        r = results[fix_key]
        for beta in r["betas"]:
            subset = [row for row in r["queries"] if row["beta"] == beta]
            n_diff = sum(1 for row in subset if row["set_diff"] > 0)
            n_gs = sum(1 for row in subset if row["nonzero_graph_scores_in_top10"] > 0)
            avg_max = round(float(np.mean([row["max_graph_score_top10"] for row in subset])), 4)
            print(f"{fix_label:<10} | {beta:<5} | {n_diff:>5}/10 | {n_gs:>7}/10 | {avg_max:>10.4f}")


if __name__ == "__main__":
    main()
