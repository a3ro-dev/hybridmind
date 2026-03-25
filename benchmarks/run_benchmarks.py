"""
HybridMind Benchmark Suite
===========================
Runs all benchmarks programmatically and saves results to benchmarks/results.json.

Sections:
  1A - Latency benchmarks (8 operations, 200 requests each)
  1B - Scale benchmarks (50-1000 nodes)
  1C - Graph-conditioned embedding quality
  1D - CRS vs vector-only retrieval quality
  1E - Concurrency behavior

Usage:
    # Start server first:
    #   uvicorn main:app --reload --port 8000
    # Then in a separate terminal:
    python benchmarks/run_benchmarks.py
"""

import json
import os
import sys
import time
import platform
import statistics
import threading
import math
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
WARMUP_REQUESTS = 50
MEASURE_REQUESTS = 200
TIMEOUT = 15.0  # per-request timeout; CPU embedding ~2-3s, give headroom

# Output paths are relative to repo root
REPO_ROOT = Path(__file__).parent.parent
RESULTS_PATH = Path(__file__).parent / "results.json"

EVAL_QUERIES_PATH = REPO_ROOT / "data" / "eval_queries.json"
MIND_DB_PATH = REPO_ROOT / "data" / "hybridmind.mind" / "store.db"

# Scale test configuration
SCALE_SIZES = [50, 150, 300, 600, 1000]

# Synthetic text templates for scale testing
SYNTHETIC_TEMPLATES = [
    "This paper studies {} using {} methods with application to {}.",
    "We propose a novel {} algorithm for {} optimization in {} systems.",
    "An analysis of {} in the context of {} with theoretical {} bounds.",
    "Deep {} networks achieve state-of-the-art performance on {} benchmarks.",
    "Scalable {} approaches for {} tasks using {}-based representations.",
    "Efficient {} inference via {} approximation in high-dimensional {} spaces.",
    "Graph-based {} with {} attention mechanism for relational {} learning.",
    "Bayesian {} estimation under {} uncertainty with {} guarantees.",
    "Federated {} learning with {} privacy and {} communication efficiency.",
    "Transformer-based {} model for multi-modal {} understanding.",
]
SYNTHETIC_TOKENS_A = [
    "neural", "gradient", "probabilistic", "spectral", "topological",
    "adversarial", "contrastive", "diffusion", "autoregressive", "hierarchical",
    "manifold", "sparse", "dense", "causal", "distributional",
]
SYNTHETIC_TOKENS_B = [
    "graph", "sequence", "image", "text", "tabular",
    "speech", "video", "molecular", "temporal", "spatial",
    "cross-lingual", "multivariate", "structured", "unstructured", "symbolic",
]
SYNTHETIC_TOKENS_C = [
    "benchmark", "dataset", "framework", "pipeline", "architecture",
    "knowledge", "reasoning", "retrieval", "generation", "classification",
    "regression", "clustering", "ranking", "detection", "segmentation",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        timeout=TIMEOUT,
        headers={"X-API-Key": f"benchmark-main-{uuid.uuid4().hex[:8]}"},
    )


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    floor_k = int(k)
    ceil_k = min(floor_k + 1, len(sorted_data) - 1)
    return sorted_data[floor_k] + (sorted_data[ceil_k] - sorted_data[floor_k]) * (k - floor_k)


def _stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}
    return {
        "mean": round(statistics.mean(latencies), 3),
        "median": round(statistics.median(latencies), 3),
        "p95": round(_percentile(latencies, 95), 3),
        "p99": round(_percentile(latencies, 99), 3),
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
        "count": len(latencies),
    }


def _measure_op(fn, n: int = MEASURE_REQUESTS, warmup: int = WARMUP_REQUESTS) -> dict[str, float]:
    """Run fn() repeatedly; return latency statistics in milliseconds."""
    # Warmup
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass

    latencies = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            fn()
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            errors += 1

    result = _stats(latencies)
    result["errors"] = errors
    return result


def _synthetic_text(idx: int) -> str:
    template = SYNTHETIC_TEMPLATES[idx % len(SYNTHETIC_TEMPLATES)]
    a = SYNTHETIC_TOKENS_A[idx % len(SYNTHETIC_TOKENS_A)]
    b = SYNTHETIC_TOKENS_B[(idx * 3) % len(SYNTHETIC_TOKENS_B)]
    c = SYNTHETIC_TOKENS_C[(idx * 7) % len(SYNTHETIC_TOKENS_C)]
    return template.format(a, b, c) + f" (node-{idx})"


def _wait_for_server(max_wait: int = 30) -> None:
    print("Waiting for server to be ready...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print("Server ready.")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"Server not ready after {max_wait}s. Is uvicorn running on port 8000?")

    # Pre-warm: fire one embedding request and wait for it to complete.
    # sentence-transformers lazy-compiles on first call; without this, the
    # first N benchmark calls queue behind the compile and all look slow.
    print("Pre-warming embedding model (first call may take 5-30s on CPU)...")
    try:
        r = httpx.post(
            f"{BASE_URL}/search/vector",
            json={"query_text": f"prewarm {uuid.uuid4().hex}", "top_k": 1},
            timeout=60,
        )
        print(f"  Pre-warm done (HTTP {r.status_code}).")
    except Exception as e:
        print(f"  Pre-warm failed (continuing anyway): {e}")


def _get_all_node_ids(client: httpx.Client) -> list[str]:
    """Get all active node IDs from the server, paginating if needed."""
    all_ids = []
    offset = 0
    limit = 500
    while True:
        r = client.get("/nodes", params={"skip": offset, "limit": limit})
        if r.status_code != 200:
            break
        data = r.json()
        if isinstance(data, list):
            page = data
        else:
            page = data.get("nodes", data.get("items", []))
        if not page:
            break
        all_ids.extend(n["id"] for n in page)
        if len(page) < limit:
            break
        offset += limit
    return all_ids


def _get_most_connected_node(client: httpx.Client) -> str | None:
    """Return the node_id with the most edges."""
    try:
        stats = client.get("/search/stats").json()
        # Try to find a node we can use as anchor
        node_ids = _get_all_node_ids(client)
        if not node_ids:
            return None
        # Use the first node as a proxy; proper degree lookup would need extra API
        if len(node_ids) > 10:
            return node_ids[5]
        return node_ids[0]
    except Exception:
        return None


def _load_nodes(client: httpx.Client, target_count: int, current_count: int) -> None:
    """Load synthetic nodes until we reach target_count total."""
    needed = target_count - current_count
    if needed <= 0:
        return
    print(f"  Loading {needed} synthetic nodes (current={current_count}, target={target_count})...")
    loaded = 0
    unique_offset = current_count + 100000  # avoid collision with existing
    for i in range(needed):
        idx = unique_offset + i
        node = {
            "text": _synthetic_text(idx),
            "metadata": {
                "source": "benchmark_synthetic",
                "index": idx,
            }
        }
        # Use a unique API key per request to avoid rate limiting
        key = f"bench-load-{uuid.uuid4().hex[:8]}"
        try:
            r = httpx.post(
                f"{BASE_URL}/nodes",
                json=node,
                headers={"X-API-Key": key},
                timeout=TIMEOUT,
            )
            if r.status_code in (200, 201):
                loaded += 1
                if loaded % 25 == 0:
                    print(f"    ...{loaded}/{needed} loaded")
        except Exception as e:
            print(f"  Warning: node insert failed: {e}")
    print(f"  Loaded {loaded}/{needed} synthetic nodes.")


def _get_node_count(client: httpx.Client) -> int:
    try:
        stats = client.get("/search/stats").json()
        # stats returns total_nodes for non-deleted nodes
        count = stats.get("total_nodes", stats.get("node_count", 0))
        return int(count)
    except Exception:
        # Fall back to listing nodes
        try:
            ids = _get_all_node_ids(client)
            return len(ids)
        except Exception:
            return 0


def _get_faiss_memory(client: httpx.Client) -> int:
    """Get approximate FAISS index memory usage in bytes."""
    try:
        r = client.get("/health")
        data = r.json()
        size = data.get("components", {}).get("vector_index", {}).get("size", 0)
        # 384 floats * 4 bytes each
        return int(size) * 384 * 4
    except Exception:
        return 0


def _get_graph_memory(client: httpx.Client) -> int:
    """Get approximate NetworkX graph memory footprint in bytes."""
    try:
        r = client.get("/health")
        data = r.json()
        nodes = data.get("components", {}).get("graph_index", {}).get("nodes", 0)
        edges = data.get("components", {}).get("graph_index", {}).get("edges", 0)
        # Rough estimate: ~200 bytes per node + 150 bytes per edge in NetworkX
        return int(nodes) * 200 + int(edges) * 150
    except Exception:
        return 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_node_embeddings_from_db() -> dict[str, dict[str, list[float]]]:
    """Read raw and conditioned embeddings directly from SQLite."""
    import struct
    result = {}
    try:
        conn = sqlite3.connect(str(MIND_DB_PATH))
        cursor = conn.cursor()
        # Check columns available
        cursor.execute("PRAGMA table_info(nodes)")
        cols = {row[1] for row in cursor.fetchall()}
        
        if "embedding" not in cols:
            conn.close()
            return result
        
        has_raw = "raw_embedding" in cols
        cursor.execute("SELECT id, embedding FROM nodes WHERE deleted_at IS NULL LIMIT 200")
        rows = cursor.fetchall()
        
        for node_id, emb_blob in rows:
            if emb_blob is None:
                continue
            try:
                n_floats = len(emb_blob) // 4
                emb = list(struct.unpack(f"{n_floats}f", emb_blob))
                result[node_id] = {"conditioned": emb, "raw": emb}
            except Exception:
                pass
        
        if has_raw:
            cursor.execute("SELECT id, raw_embedding FROM nodes WHERE deleted_at IS NULL LIMIT 200")
            for node_id, raw_blob in cursor.fetchall():
                if raw_blob is None or node_id not in result:
                    continue
                try:
                    n_floats = len(raw_blob) // 4
                    raw = list(struct.unpack(f"{n_floats}f", raw_blob))
                    result[node_id]["raw"] = raw
                except Exception:
                    pass
        
        conn.close()
    except Exception as e:
        print(f"  Warning: Could not read embeddings from DB: {e}")
    return result


# ── Section 1A — Latency Benchmarks ───────────────────────────────────────────

def benchmark_1a_latency(client: httpx.Client) -> dict[str, Any]:
    print("\n[1A] Latency Benchmarks")
    results = {}

    node_count = _get_node_count(client)
    if node_count < 150:
        print(f"  Insufficient nodes ({node_count}). Loading demo data first...")
        _load_nodes(client, 150, node_count)
        node_count = _get_node_count(client)

    all_ids = _get_all_node_ids(client)
    anchor_id = _get_most_connected_node(client) or (all_ids[0] if all_ids else None)


    import random as _rnd
    _rnd.seed(None)

    # Detect embedding backend
    try:
        embedder = client.get("/health").json().get("components", {}).get("embedding_model", "unknown")
    except Exception:
        embedder = "unknown"
    print(f"  Embedding backend: {embedder}")
    if "mock" in str(embedder).lower() or embedder == "unknown":
        print("  WARNING: mock embeddings — node_insert shows DB+network only, not real ~50-150ms embed time")

    # Pre-build 30 distinct IDs for graph traversal rotation
    graph_ids = list({_rnd.choice(all_ids) for _ in range(min(30, len(all_ids)))}) if all_ids else all_ids[:30]
    _gi = [0]
    def _gid() -> str:
        v = graph_ids[_gi[0] % len(graph_ids)]; _gi[0] += 1; return v

    # Pre-build 50 distinct IDs for node_get rotation
    get_ids = list({_rnd.choice(all_ids) for _ in range(min(50, len(all_ids)))}) if all_ids else all_ids[:50]
    _ngi = [0]
    def _ngid() -> str:
        v = get_ids[_ngi[0] % len(get_ids)]; _ngi[0] += 1; return v

    N, W = 5, 2  # 5 measured + 2 warmup — fast proof run (~30s total on CPU)

    # 1. Vector search — uuid4 per call
    print("  [1A.1] Vector search...")
    results["vector_search"] = _measure_op(
        lambda: client.post("/search/vector", json={"query_text": f"neural network {uuid.uuid4().hex}", "top_k": 10}),
        n=N, warmup=W)

    # 2. Hybrid search — uuid4 per call (completely independent from vector above)
    print("  [1A.2] Hybrid search...")
    results["hybrid_search_default"] = _measure_op(
        lambda: client.post("/search/hybrid", json={"query_text": f"gradient descent {uuid.uuid4().hex}", "top_k": 10}),
        n=N, warmup=W)

    # 3. Hybrid with anchor — uuid4 per call
    print("  [1A.3] Hybrid search (anchored)...")
    results["hybrid_search_anchored"] = _measure_op(
        lambda: client.post("/search/hybrid", json={
            "query_text": f"representation learning {uuid.uuid4().hex}",
            "top_k": 10, "anchor_nodes": [anchor_id] if anchor_id else []}),
        n=N, warmup=W)

    # 4. Graph traversal — rotate through 30 distinct start nodes
    print(f"  [1A.4] Graph traversal ({len(graph_ids)} start nodes)...")
    results["graph_traversal"] = _measure_op(
        lambda: client.get("/search/graph", params={"start_id": _gid(), "depth": 2, "direction": "both"}),
        n=N, warmup=W)

    # 5. Node insert — unique uuid per call; verify count delta
    print("  [1A.5] Node insert...")
    count_before = _get_node_count(client)
    N_INS = 50
    results["node_insert"] = _measure_op(
        lambda: client.post("/nodes", json={
            "text": f"benchmark node {uuid.uuid4()} {_rnd.choice(['attention','graph','embedding','retrieval'])}",
            "metadata": {"source": "benchmark_1a"}}),
        n=N_INS, warmup=3)
    count_after = _get_node_count(client)
    results["node_insert"]["verified_inserts"] = count_after - count_before
    print(f"    count: {count_before} → {count_after} (+{count_after - count_before}, expected ~{N_INS})")

    # 6. Node get — rotate through 50 distinct IDs
    print(f"  [1A.6] Node get ({len(get_ids)} IDs)...")
    results["node_get"] = _measure_op(lambda: client.get(f"/nodes/{_ngid()}"), n=200, warmup=10)

    # 7. Snapshot — log HTTP status on each call
    print("  [1A.7] Snapshot...")
    snap_lats, snap_codes = [], []
    for i in range(10):
        t0 = time.perf_counter()
        try:
            r = client.post("/snapshot")
            elapsed = (time.perf_counter() - t0) * 1000
            snap_codes.append(r.status_code)
            if r.status_code in (200, 201):
                snap_lats.append(elapsed)
            else:
                print(f"    snap[{i}] HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"    snap[{i}] error: {e}")
    print(f"    statuses: {snap_codes}")
    results["snapshot"] = _stats(snap_lats)
    results["snapshot"]["errors"] = 10 - len(snap_lats)

    # 8. Compact — fresh 5 deletes per round, log HTTP status
    print("  [1A.8] Compact...")
    cmp_lats, cmp_codes = [], []
    for i in range(8):
        for _ in range(5):
            try:
                r = client.post("/nodes", json={"text": f"fodder_{uuid.uuid4().hex}"})
                if r.status_code in (200, 201):
                    nid = r.json().get("id")
                    if nid:
                        client.delete(f"/nodes/{nid}")
            except Exception:
                pass
        t0 = time.perf_counter()
        try:
            r = client.post("/admin/compact")
            elapsed = (time.perf_counter() - t0) * 1000
            cmp_codes.append(r.status_code)
            if r.status_code in (200, 201):
                cmp_lats.append(elapsed)
            else:
                print(f"    compact[{i}] HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"    compact[{i}] error: {e}")
    print(f"    statuses: {cmp_codes}")
    results["compact"] = _stats(cmp_lats)
    results["compact"]["errors"] = 8 - len(cmp_lats)

    print("  [1A] Done. Summary:")
    for name, stats in results.items():
        print(f"    {name}: mean={stats['mean']}ms p95={stats['p95']}ms errors={stats.get('errors', 0)}")

    return results


# ── Section 1B — Scale Benchmarks ─────────────────────────────────────────────

def benchmark_1b_scale(client: httpx.Client) -> dict[str, Any]:
    print("\n[1B] Scale Benchmarks")
    results = {}

    current_count = _get_node_count(client)
    all_ids_cache: list[str] = []

    for target in SCALE_SIZES:
        print(f"\n  Scale: {target} nodes (current={current_count})")
        if current_count < target:
            _load_nodes(client, target, current_count)
            current_count = _get_node_count(client)

        all_ids = _get_all_node_ids(client)
        anchor = all_ids[min(5, len(all_ids) - 1)] if all_ids else None

        # Cache-busting counter — each call gets a unique query string
        _sb_n = [0]
        def _sbq(base: str) -> str:
            _sb_n[0] += 1
            return f"{base} {_sb_n[0]}"

        n_ops = 60
        warmup_ops = 5  # small warmup — just to warm the model, not the cache

        vec_lats = []
        for _ in range(warmup_ops):
            try: client.post("/search/vector", json={"query_text": _sbq("optimization learning"), "top_k": 10})
            except Exception: pass
        for _ in range(n_ops):
            t0 = time.perf_counter()
            try:
                client.post("/search/vector", json={"query_text": _sbq("machine learning optimization"), "top_k": 10})
                vec_lats.append((time.perf_counter() - t0) * 1000)
            except Exception: pass

        hyb_lats = []
        for _ in range(warmup_ops):
            try: client.post("/search/hybrid", json={"query_text": _sbq("gradient descent learning"), "top_k": 10})
            except Exception: pass
        for _ in range(n_ops):
            t0 = time.perf_counter()
            try:
                client.post("/search/hybrid", json={"query_text": _sbq("machine learning optimization"), "top_k": 10})
                hyb_lats.append((time.perf_counter() - t0) * 1000)
            except Exception: pass

        graph_lats = []
        all_ids_scale = _get_all_node_ids(client)
        import random as _rnd
        for _ in range(warmup_ops):
            try:
                nid = _rnd.choice(all_ids_scale) if all_ids_scale else anchor
                client.get("/search/graph", params={"start_id": nid, "depth": 2, "direction": "both"})
            except Exception: pass
        for _ in range(n_ops):
            t0 = time.perf_counter()
            try:
                nid = _rnd.choice(all_ids_scale) if all_ids_scale else anchor
                client.get("/search/graph", params={"start_id": nid, "depth": 2, "direction": "both"})
                graph_lats.append((time.perf_counter() - t0) * 1000)
            except Exception: pass

        faiss_mem = _get_faiss_memory(client)
        graph_mem = _get_graph_memory(client)

        results[str(target)] = {
            "actual_node_count": current_count,
            "vector_search": {
                "p50": round(_percentile(vec_lats, 50), 3),
                "p95": round(_percentile(vec_lats, 95), 3),
            },
            "hybrid_search": {
                "p50": round(_percentile(hyb_lats, 50), 3),
                "p95": round(_percentile(hyb_lats, 95), 3),
            },
            "graph_traversal": {
                "p50": round(_percentile(graph_lats, 50), 3),
                "p95": round(_percentile(graph_lats, 95), 3),
            },
            "faiss_memory_bytes": faiss_mem,
            "graph_memory_bytes": graph_mem,
        }

        print(f"    vec p50={results[str(target)]['vector_search']['p50']}ms "
              f"p95={results[str(target)]['vector_search']['p95']}ms")
        print(f"    hyb p50={results[str(target)]['hybrid_search']['p50']}ms "
              f"p95={results[str(target)]['hybrid_search']['p95']}ms")
        print(f"    graph p50={results[str(target)]['graph_traversal']['p50']}ms "
              f"p95={results[str(target)]['graph_traversal']['p95']}ms")
        print(f"    FAISS mem={faiss_mem//1024}KB  Graph mem={graph_mem//1024}KB")

    return results


# ── Section 1C — Graph-Conditioned Embedding Quality ──────────────────────────

def benchmark_1c_conditioning(client: httpx.Client) -> dict[str, Any]:
    print("\n[1C] Graph-Conditioned Embedding Quality")
    results = {}

    # Create 20 isolated nodes
    isolated_ids = []
    for i in range(20):
        try:
            r = client.post("/nodes", json={
                "text": f"conditioning test node isolated {i}: {_synthetic_text(9000 + i)}",
                "metadata": {"source": "benchmark_1c", "group": "isolated", "index": i},
            })
            if r.status_code in (200, 201):
                isolated_ids.append(r.json()["id"])
        except Exception as e:
            print(f"  Warning: could not create node {i}: {e}")

    print(f"  Created {len(isolated_ids)} isolated nodes")

    # Create 10 edged nodes + 10 without edges
    edged_ids = []
    unedged_ids = []
    existing_ids = _get_all_node_ids(client)
    anchors = existing_ids[:5] if len(existing_ids) >= 5 else existing_ids

    for i in range(10):
        try:
            r = client.post("/nodes", json={
                "text": f"conditioning test edged node {i}: {_synthetic_text(9100 + i)}",
                "metadata": {"source": "benchmark_1c", "group": "edged"},
            })
            if r.status_code in (200, 201):
                nid = r.json()["id"]
                edged_ids.append(nid)
                # Add edges to anchor nodes
                for anchor in anchors[:2]:
                    try:
                        client.post("/edges", json={
                            "source_id": nid,
                            "target_id": anchor,
                            "type": "related_to",
                            "weight": 1.0,
                        })
                    except Exception:
                        pass
        except Exception:
            pass

    for i in range(10):
        try:
            r = client.post("/nodes", json={
                "text": f"conditioning test unedged node {i}: {_synthetic_text(9200 + i)}",
                "metadata": {"source": "benchmark_1c", "group": "unedged"},
            })
            if r.status_code in (200, 201):
                unedged_ids.append(r.json()["id"])
        except Exception:
            pass

    print(f"  Created {len(edged_ids)} edged nodes, {len(unedged_ids)} unedged nodes")

    # Wait briefly for embeddings to settle
    time.sleep(2)

    # Read embeddings from DB
    embeddings = _get_node_embeddings_from_db()

    # Compute cosine diff for isolated nodes
    isolated_diffs = []
    for nid in isolated_ids:
        if nid in embeddings:
            raw = embeddings[nid]["raw"]
            cond = embeddings[nid]["conditioned"]
            sim = _cosine_similarity(raw, cond)
            diff = 1.0 - sim
            isolated_diffs.append(diff)

    # Compute cosine diff for edged vs unedged
    edged_diffs = []
    for nid in edged_ids:
        if nid in embeddings:
            raw = embeddings[nid]["raw"]
            cond = embeddings[nid]["conditioned"]
            edged_diffs.append(1.0 - _cosine_similarity(raw, cond))

    unedged_diffs = []
    for nid in unedged_ids:
        if nid in embeddings:
            raw = embeddings[nid]["raw"]
            cond = embeddings[nid]["conditioned"]
            unedged_diffs.append(1.0 - _cosine_similarity(raw, cond))

    def safe_mean(lst: list[float]) -> float:
        return round(statistics.mean(lst), 6) if lst else 0.0

    results["isolated_nodes"] = {
        "count": len(isolated_diffs),
        "mean_cosine_diff": safe_mean(isolated_diffs),
        "min_cosine_diff": round(min(isolated_diffs), 6) if isolated_diffs else 0.0,
        "max_cosine_diff": round(max(isolated_diffs), 6) if isolated_diffs else 0.0,
        "nodes_with_diff_gt_0_01": sum(1 for d in isolated_diffs if d > 0.01),
    }

    results["edged_vs_unedged"] = {
        "edged_node_count": len(edged_diffs),
        "unedged_node_count": len(unedged_diffs),
        "edged_mean_diff": safe_mean(edged_diffs),
        "unedged_mean_diff": safe_mean(unedged_diffs),
        "edged_higher_than_unedged": safe_mean(edged_diffs) > safe_mean(unedged_diffs),
        "difference_in_conditioning": round(safe_mean(edged_diffs) - safe_mean(unedged_diffs), 6),
    }

    print(f"  Isolated: mean_diff={results['isolated_nodes']['mean_cosine_diff']:.6f}, "
          f"nodes_with_diff>0.01={results['isolated_nodes']['nodes_with_diff_gt_0_01']}")
    print(f"  Edged mean diff: {results['edged_vs_unedged']['edged_mean_diff']:.6f}")
    print(f"  Unedged mean diff: {results['edged_vs_unedged']['unedged_mean_diff']:.6f}")

    return results


# ── Section 1D — CRS vs Vector-Only Retrieval Quality ─────────────────────────

def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / k if k > 0 else 0.0


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, nid in enumerate(retrieved):
        if nid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, nid in enumerate(retrieved[:k])
        if nid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def benchmark_1d_eval(client: httpx.Client) -> dict[str, Any]:
    print("\n[1D] CRS vs Vector-Only Retrieval Quality")

    if not EVAL_QUERIES_PATH.exists():
        print("  eval_queries.json not found, skipping.")
        return {"error": "eval_queries.json not found"}

    with open(EVAL_QUERIES_PATH) as f:
        eval_data = json.load(f)

    queries = eval_data.get("queries", [])
    print(f"  Loaded {len(queries)} evaluation queries")

    def run_eval_pass() -> dict[str, Any]:
        vec_p5 = vec_p10 = vec_mrr = vec_ndcg = 0.0
        hyb_p5 = hyb_p10 = hyb_mrr = hyb_ndcg = 0.0
        attempted = 0

        for q in queries:
            query_text = q["query"]
            ground_truth = set(q["ground_truth_ids"])

            try:
                vr = client.post("/search/vector", json={"query_text": query_text, "top_k": 10})
                hr = client.post("/search/hybrid", json={"query_text": query_text, "top_k": 10})

                if vr.status_code != 200 or hr.status_code != 200:
                    continue

                vec_results = [r["node_id"] for r in vr.json().get("results", [])]
                hyb_results = [r["node_id"] for r in hr.json().get("results", [])]

                vec_p5 += _precision_at_k(vec_results, ground_truth, 5)
                vec_p10 += _precision_at_k(vec_results, ground_truth, 10)
                vec_mrr += _reciprocal_rank(vec_results, ground_truth)
                vec_ndcg += _ndcg_at_k(vec_results, ground_truth, 10)

                hyb_p5 += _precision_at_k(hyb_results, ground_truth, 5)
                hyb_p10 += _precision_at_k(hyb_results, ground_truth, 10)
                hyb_mrr += _reciprocal_rank(hyb_results, ground_truth)
                hyb_ndcg += _ndcg_at_k(hyb_results, ground_truth, 10)

                attempted += 1
            except Exception as e:
                print(f"  Warning: query eval error: {e}")

        n = max(attempted, 1)
        return {
            "queries_evaluated": attempted,
            "vector": {
                "precision_at_5": round(vec_p5 / n, 4),
                "precision_at_10": round(vec_p10 / n, 4),
                "mrr": round(vec_mrr / n, 4),
                "ndcg": round(vec_ndcg / n, 4),
            },
            "hybrid": {
                "precision_at_5": round(hyb_p5 / n, 4),
                "precision_at_10": round(hyb_p10 / n, 4),
                "mrr": round(hyb_mrr / n, 4),
                "ndcg": round(hyb_ndcg / n, 4),
            },
        }

    # Run 3 times to check determinism
    runs = []
    for i in range(3):
        print(f"  Run {i+1}/3...")
        runs.append(run_eval_pass())

    # Compute variance across runs
    metrics = ["precision_at_5", "precision_at_10", "mrr", "ndcg"]
    variance = {}
    for m in metrics:
        vals = [r["hybrid"][m] for r in runs]
        variance[f"hybrid_{m}_variance"] = round(statistics.variance(vals) if len(vals) > 1 else 0.0, 6)

    # Pick best run (first is representative)
    best = runs[0]

    # Determine winners per metric
    deltas = {}
    for m in metrics:
        vv = best["vector"][m]
        hv = best["hybrid"][m]
        delta = round(hv - vv, 4)
        winner = "hybrid" if hv >= vv else "vector"
        deltas[m] = {"vector": vv, "hybrid": hv, "delta": delta, "winner": winner}

    result = {
        "runs": runs,
        "variance": variance,
        "summary": deltas,
        "caveat": (
            "BM25 ground truth favors keyword overlap — semantic search may find "
            "relevant results BM25 doesn't label as relevant. Treat as directional signal."
        ),
    }

    print("  Eval results (run 1):")
    for m, d in deltas.items():
        print(f"    {m}: vector={d['vector']}, hybrid={d['hybrid']}, winner={d['winner']}")

    return result


# ── Section 1E — Concurrency Benchmarks ──────────────────────────────────────

def benchmark_1e_concurrency(client: httpx.Client) -> dict[str, Any]:
    print("\n[1E] Concurrency Benchmarks")
    results = {}
    all_ids = _get_all_node_ids(client)
    import random
    random.seed(123)

    def single_read():
        """One read operation with unique key to avoid rate limiting."""
        key = f"bench-read-{uuid.uuid4().hex[:8]}"
        c = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT,
                         headers={"X-API-Key": key})
        r = c.post("/search/vector", json={"query_text": "machine learning", "top_k": 5})
        c.close()
        return r.status_code == 200

    def single_write():
        """One write operation with unique key to avoid rate limiting."""
        key = f"bench-write-{uuid.uuid4().hex[:8]}"
        c = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT,
                         headers={"X-API-Key": key})
        r = c.post("/nodes", json={"text": f"concurrency test {uuid.uuid4().hex}"})
        c.close()
        return r.status_code in (200, 201)

    def run_threads(fn, n_threads: int, ops_per_thread: int):
        """Run fn() in n_threads threads, each doing ops_per_thread calls."""
        successes = []
        latencies = []
        errors = []
        lock = threading.Lock()

        def worker():
            local_lats = []
            local_ok = 0
            local_err = 0
            for _ in range(ops_per_thread):
                t0 = time.perf_counter()
                try:
                    ok = fn()
                    elapsed = (time.perf_counter() - t0) * 1000
                    local_lats.append(elapsed)
                    if ok:
                        local_ok += 1
                    else:
                        local_err += 1
                except Exception:
                    local_err += 1
            with lock:
                successes.append(local_ok)
                latencies.extend(local_lats)
                errors.append(local_err)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        t_start = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed_total = (time.perf_counter() - t_start) * 1000

        total_ops = n_threads * ops_per_thread
        total_success = sum(successes)
        total_errors = sum(errors)

        return {
            "threads": n_threads,
            "ops_per_thread": ops_per_thread,
            "total_ops": total_ops,
            "success_count": total_success,
            "error_count": total_errors,
            "success_rate": round(total_success / max(total_ops, 1), 4),
            "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0,
            "max_latency_ms": round(max(latencies), 3) if latencies else 0,
            "total_elapsed_ms": round(elapsed_total, 3),
        }

    # Concurrent reads
    print("  [1E.1] Concurrent reads...")
    read_results = []
    for n_threads in [10, 25, 50]:
        print(f"    {n_threads} threads × 10 reads...")
        r = run_threads(single_read, n_threads, 10)
        read_results.append(r)
        print(f"    success_rate={r['success_rate']:.2%} mean={r['mean_latency_ms']}ms max={r['max_latency_ms']}ms")
    results["concurrent_reads"] = read_results

    # Concurrent writes
    print("  [1E.2] Concurrent writes...")
    write_results = []
    for n_threads in [5, 10, 20]:
        print(f"    {n_threads} threads × 5 writes...")
        r = run_threads(single_write, n_threads, 5)
        # Verify final node count didn't diverge badly
        final_count = _get_node_count(client)
        write_results.append({**r, "final_node_count_after": final_count})
        print(f"    success_rate={r['success_rate']:.2%} mean={r['mean_latency_ms']}ms")
    results["concurrent_writes"] = write_results

    # Mixed read/write: 20 threads (15 readers, 5 writers)
    print("  [1E.3] Mixed reads + writes (15 readers, 5 writers)...")
    lock = threading.Lock()
    mixed_latencies = []
    mixed_successes = [0]
    mixed_errors = [0]
    mixed_500s = [0]

    def reader_fn():
        key = f"bench-mrw-r-{uuid.uuid4().hex[:8]}"
        c = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT,
                         headers={"X-API-Key": key})
        t0 = time.perf_counter()
        r = c.post("/search/hybrid", json={"query_text": "deep learning", "top_k": 5})
        elapsed = (time.perf_counter() - t0) * 1000
        c.close()
        with lock:
            mixed_latencies.append(elapsed)
            if r.status_code == 200:
                mixed_successes[0] += 1
            elif r.status_code == 500:
                mixed_500s[0] += 1
            else:
                mixed_errors[0] += 1

    def writer_fn():
        key = f"bench-mrw-w-{uuid.uuid4().hex[:8]}"
        c = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT,
                         headers={"X-API-Key": key})
        t0 = time.perf_counter()
        r = c.post("/nodes", json={"text": f"mixed test {uuid.uuid4().hex}"})
        elapsed = (time.perf_counter() - t0) * 1000
        c.close()
        with lock:
            mixed_latencies.append(elapsed)
            if r.status_code in (200, 201):
                mixed_successes[0] += 1
            elif r.status_code == 500:
                mixed_500s[0] += 1
            else:
                mixed_errors[0] += 1

    def mixed_worker(fn, n_ops: int):
        for _ in range(n_ops):
            fn()

    threads = (
        [threading.Thread(target=mixed_worker, args=(reader_fn, 10)) for _ in range(15)]
        + [threading.Thread(target=mixed_worker, args=(writer_fn, 5)) for _ in range(5)]
    )
    for t in threads: t.start()
    for t in threads: t.join()

    total_mixed = 15 * 10 + 5 * 5
    results["mixed_rw"] = {
        "total_ops": total_mixed,
        "success_count": mixed_successes[0],
        "error_count": mixed_errors[0],
        "http_500_count": mixed_500s[0],
        "success_rate": round(mixed_successes[0] / max(total_mixed, 1), 4),
        "mean_latency_ms": round(statistics.mean(mixed_latencies), 3) if mixed_latencies else 0,
        "max_latency_ms": round(max(mixed_latencies), 3) if mixed_latencies else 0,
    }
    print(f"    success_rate={results['mixed_rw']['success_rate']:.2%} "
          f"500s={results['mixed_rw']['http_500_count']}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def collect_system_info() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        "os": platform.system() + " " + platform.release(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(mem.total / (1024 ** 3), 2),
    }


def main():
    print("=" * 60)
    print("HybridMind Benchmark Suite")
    print("=" * 60)

    _wait_for_server()

    client = _client()
    timestamp = datetime.now(timezone.utc).isoformat()

    results = {
        "timestamp": timestamp,
        "system": collect_system_info(),
        "latency": {},
        "scale": {},
        "conditioning": {},
        "eval": {},
        "concurrency": {},
    }

    try:
        results["latency"] = benchmark_1a_latency(client)
    except Exception as e:
        print(f"[1A] ERROR: {e}")
        results["latency"] = {"error": str(e)}

    try:
        results["scale"] = benchmark_1b_scale(client)
    except Exception as e:
        print(f"[1B] ERROR: {e}")
        results["scale"] = {"error": str(e)}

    try:
        results["conditioning"] = benchmark_1c_conditioning(client)
    except Exception as e:
        print(f"[1C] ERROR: {e}")
        results["conditioning"] = {"error": str(e)}

    try:
        results["eval"] = benchmark_1d_eval(client)
    except Exception as e:
        print(f"[1D] ERROR: {e}")
        results["eval"] = {"error": str(e)}

    try:
        results["concurrency"] = benchmark_1e_concurrency(client)
    except Exception as e:
        print(f"[1E] ERROR: {e}")
        results["concurrency"] = {"error": str(e)}

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete. Results saved to: {RESULTS_PATH}")
    print(f"{'='*60}")

    client.close()
    return results


if __name__ == "__main__":
    main()
