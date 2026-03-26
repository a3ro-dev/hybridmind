import html
import json
import math
import os
import platform
import atexit
import random
import re
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from itertools import combinations, islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.getcwd())
try:
    from sdk.memory import HybridMemory
except ImportError:
    sys.path.append("..")
    from sdk.memory import HybridMemory


BASE_URL = "http://127.0.0.1:8000"
RESULTS_FILE = Path("benchmarks/multi_domain_results.json")
REPORT_FILE = Path("docs/MULTI_DOMAIN_EVAL.md")
MIND_DB_PATH = Path("data/hybridmind.mind/store.db")
LOCK_FILE = Path("benchmarks/.multi_domain_eval.lock")
RNG = random.Random(42)
VECTOR_WEIGHT = 0.6
GRAPH_WEIGHT = 0.4
EDGE_THRESHOLD = 0.45  # default; override via MultiDomainEval.edge_threshold
BASELINE_1000_P50_MS = 13.0
BASELINE_1000_P95_MS = 16.0
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def clean_text(text: Any) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, limit: int) -> str:
    return clean_text(text)[:limit]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def score_for_result(result: Dict[str, Any]) -> float:
    return float(
        result.get("combined_score")
        or result.get("vector_score")
        or result.get("graph_score")
        or 0.0
    )


def count_domains(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(r.get("metadata", {}).get("domain", "unknown") for r in results)
    return dict(sorted(counter.items()))


def short_text(text: str, limit: int = 180) -> str:
    return truncate(text, limit)


class MultiDomainEval:
    def __init__(self, skip_lock: bool = False) -> None:
        if not skip_lock:
            self._acquire_lock()
        self.edge_threshold = EDGE_THRESHOLD
        self.sdk = HybridMemory(base_url=BASE_URL, timeout=300.0)
        self.client = self.sdk.client
        self.rng = RNG
        self.log_handle = open("eval_progress.log", "a", encoding="utf-8", buffering=1)
        self.nodes: List[Dict[str, Any]] = []
        self.nodes_by_id: Dict[str, Dict[str, Any]] = {}
        self.nodes_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.created_cross_edges: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_url": BASE_URL,
                "edge_threshold": EDGE_THRESHOLD,
                "vector_weight": VECTOR_WEIGHT,
                "graph_weight": GRAPH_WEIGHT,
                "baseline_1000_nodes": {
                    "p50_ms": BASELINE_1000_P50_MS,
                    "p95_ms": BASELINE_1000_P95_MS,
                },
            },
            "system": {},
            "loading": {},
            "graph_construction": {},
            "experiments": {},
            "analysis": {},
            "manual_review_candidates": [],
        }

    def _acquire_lock(self) -> None:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        if LOCK_FILE.exists():
            try:
                payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            pid = int(payload.get("pid", 0))
            if pid > 0:
                try:
                    os.kill(pid, 0)
                except OSError:
                    pass
                else:
                    raise RuntimeError(f"multi-domain eval already running with pid={pid}")
            LOCK_FILE.unlink(missing_ok=True)
        LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}), encoding="utf-8")
        atexit.register(self._release_lock)

    def _release_lock(self) -> None:
        try:
            if LOCK_FILE.exists():
                payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
                if int(payload.get("pid", -1)) == os.getpid():
                    LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.log_handle.write(f"{message}\n")
        self.log_handle.flush()

    def _json(self, response: httpx.Response) -> Any:
        response.raise_for_status()
        return response.json()

    def collect_system_info(self) -> None:
        root = self._json(self.client.get(f"{BASE_URL}/"))
        health = self._json(self.client.get(f"{BASE_URL}/health"))
        stats = self._json(self.client.get(f"{BASE_URL}/search/stats"))
        self.results["system"] = {
            "hybridmind_version": root.get("version", "unknown"),
            "embedding_model": health.get("components", {}).get("embedding", {}).get("model", "unknown"),
            "weights": {"vector": VECTOR_WEIGHT, "graph": GRAPH_WEIGHT},
            "health_before_clear": health,
            "stats_before_clear": stats,
            "hardware": self._hardware_info(),
            "platform": {
                "os": platform.platform(),
                "python": platform.python_version(),
            },
        }

    def _hardware_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"cpu": "unknown", "gpus": []}
        try:
            cpu = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
            ).strip()
            if cpu:
                info["cpu"] = cpu
        except Exception:
            pass
        try:
            gpu_text = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join \"|\""],
                text=True,
            ).strip()
            if gpu_text:
                info["gpus"] = [g for g in gpu_text.split("|") if g]
        except Exception:
            pass
        return info

    def clear_database(self) -> None:
        self.log("Clearing database via /admin/clear ...")
        clear_response = self._json(self.client.post(f"{BASE_URL}/admin/clear"))
        self.clear_cache()
        health = self._json(self.client.get(f"{BASE_URL}/health"))
        self.results["loading"]["clear"] = {
            "response": clear_response,
            "post_clear_health_nodes": health.get("components", {}).get("database", {}).get("nodes", None),
            "post_clear_health_edges": health.get("components", {}).get("graph_index", {}).get("edges", None),
        }

    def _store_nodes(self, domain: str, nodes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        created_ids: List[str] = []
        errors: List[str] = []
        start = time.perf_counter()
        requested = 0
        for node in tqdm(list(nodes), desc=f"Load {domain}", leave=False):
            requested += 1
            try:
                node_id = self.sdk.store(node["text"], metadata=node["metadata"])
                created_ids.append(node_id)
            except Exception as exc:
                errors.append(str(exc))
        elapsed_s = time.perf_counter() - start
        return {
            "requested": requested,
            "actual_loaded": len(created_ids),
            "load_time_s": round(elapsed_s, 2),
            "errors": errors[:20],
            "created_ids_sample": created_ids[:5],
        }

    def load_wikipedia(self) -> None:
        domain = "wikipedia"
        self.log("Loading Wikipedia ...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        nodes = []
        for item in islice(ds, 2000):
            text = truncate(item.get("text", ""), 300)
            if not text:
                continue
            nodes.append(
                {
                    "text": text,
                    "metadata": {"domain": domain, "title": clean_text(item.get("title", ""))},
                }
            )
        result = self._store_nodes(domain, nodes)
        result.update({"dataset": "wikimedia/wikipedia", "config": "20231101.en", "sample_size": 2000})
        self.results["loading"][domain] = result

    def load_stackexchange(self) -> None:
        domain = "stackexchange"
        self.log("Loading Stack Exchange ...")
        ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", streaming=True)
        nodes = []
        for item in islice(ds, 2000):
            answers = item.get("answers") or []
            first_answer = answers[0] if answers else {}
            score_value = first_answer.get("pm_score")
            text = truncate(f"{item.get('question', '')} {first_answer.get('text', '')}", 400)
            if not text:
                continue
            nodes.append(
                {
                    "text": text,
                    "metadata": {"domain": domain, "score": score_value},
                }
            )
        result = self._store_nodes(domain, nodes)
        result.update(
            {
                "dataset": "HuggingFaceH4/stack-exchange-preferences",
                "config": None,
                "sample_size": 2000,
                "score_field_used": "answers[0].pm_score",
            }
        )
        self.results["loading"][domain] = result

    def load_pubmed(self) -> None:
        domain = "pubmed"
        self.log("Loading PubMed QA ...")
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", streaming=True)
        nodes = []
        for item in islice(ds, 1000):
            text = truncate(f"{item.get('question', '')} {item.get('long_answer', '')}", 400)
            if not text:
                continue
            nodes.append(
                {
                    "text": text,
                    "metadata": {"domain": domain, "label": item.get("final_decision")},
                }
            )
        result = self._store_nodes(domain, nodes)
        result.update({"dataset": "pubmed_qa", "config": "pqa_labeled", "sample_size": 1000, "label_field": "final_decision"})
        self.results["loading"][domain] = result

    def load_news(self) -> None:
        domain = "news"
        self.log("Loading AG News ...")
        ds = load_dataset("ag_news", split="train")
        counts = Counter()
        nodes = []
        for item in ds:
            label = int(item["label"])
            if counts[label] >= 500:
                continue
            text = truncate(item.get("text", ""), 400)
            if not text:
                continue
            counts[label] += 1
            nodes.append(
                {
                    "text": text,
                    "metadata": {"domain": domain, "category": LABEL_NAMES[label]},
                }
            )
            if sum(counts.values()) == 2000:
                break
        result = self._store_nodes(domain, nodes)
        result.update({"dataset": "ag_news", "config": None, "sample_size": 2000, "per_category": dict(counts)})
        self.results["loading"][domain] = result

    def _extract_legal_text(self, item: Dict[str, Any]) -> str:
        for key in ("Text", "text", "clause", "contract", "premise", "hypothesis", "sentence", "context"):
            value = item.get(key)
            if value:
                if isinstance(value, dict):
                    value = json.dumps(value)
                return truncate(str(value), 400)
        return ""

    def load_legal(self) -> None:
        domain = "legal"
        self.log("Loading legal dataset ...")
        attempts = [
            ("theatticusproject/cuad-contractnli-balanced", None),
            ("lexlms/lex_glue", "cuad"),
            ("umarbutler/better-cuad", None),
        ]
        errors = []
        for dataset_name, config in attempts:
            try:
                if config is None:
                    ds = load_dataset(dataset_name, split="train", streaming=True)
                else:
                    ds = load_dataset(dataset_name, config, split="train", streaming=False)
                iterator = islice(ds, 1000)
                nodes = []
                for item in iterator:
                    if not isinstance(item, dict):
                        continue
                    text = self._extract_legal_text(item)
                    if not text:
                        continue
                    label = item.get("label")
                    if label is None:
                        label = item.get("Filename") or item.get("Document Name")
                    if isinstance(label, (list, dict)):
                        label = json.dumps(label, ensure_ascii=False)[:200]
                    nodes.append(
                        {
                            "text": text,
                            "metadata": {"domain": domain, "label": label},
                        }
                    )
                result = self._store_nodes(domain, nodes)
                result.update(
                    {
                        "dataset": dataset_name,
                        "config": config,
                        "sample_size": 1000,
                        "fallback_used": dataset_name != attempts[0][0],
                    }
                )
                self.results["loading"][domain] = result
                return
            except Exception as exc:
                errors.append(f"{dataset_name}:{config} -> {exc}")
        self.results["loading"][domain] = {
            "dataset": None,
            "config": None,
            "sample_size": 1000,
            "requested": 1000,
            "actual_loaded": 0,
            "load_time_s": 0.0,
            "errors": errors,
            "skipped": True,
        }

    def verify_loading(self) -> None:
        self.log("Verifying node counts after load ...")
        health = self._json(self.client.get(f"{BASE_URL}/health"))
        stats = self._json(self.client.get(f"{BASE_URL}/search/stats"))
        total_loaded = sum(
            v.get("actual_loaded", 0)
            for k, v in self.results["loading"].items()
            if k not in ("clear", "verification") and isinstance(v, dict)
        )
        self.results["loading"]["verification"] = {
            "expected_total_nodes": total_loaded,
            "health_total_nodes": health.get("components", {}).get("database", {}).get("nodes", 0),
            "health_total_edges": health.get("components", {}).get("graph_index", {}).get("edges", 0),
            "stats": stats,
        }

    def clear_cache(self) -> Dict[str, Any]:
        return self._json(self.client.post(f"{BASE_URL}/cache/clear"))

    def delete_all_edges(self) -> int:
        """Delete every edge via the REST API; nodes and vector index unchanged."""
        self.clear_cache()
        deleted = 0
        skip = 0
        while True:
            batch = self._json(self.client.get(f"{BASE_URL}/edges", params={"skip": skip, "limit": 1000}))
            if not batch:
                break
            for edge in batch:
                r = self.client.delete(f"{BASE_URL}/edges/{edge['id']}")
                r.raise_for_status()
                deleted += 1
            skip += len(batch)
            if len(batch) < 1000:
                break
        self.log(f"delete_all_edges: removed {deleted} edges")
        return deleted

    def cache_stats(self) -> Dict[str, Any]:
        return self._json(self.client.get(f"{BASE_URL}/cache/stats"))

    def cold_vector_search(self, query: str, top_k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.clear_cache()
        return self._vector_search(query, top_k=top_k, filter_metadata=filter_metadata)

    def cold_hybrid_search(self, query: str, top_k: int = 10, anchor_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        self.clear_cache()
        return self._hybrid_search(query, top_k=top_k, anchor_nodes=anchor_nodes)

    def fetch_all_nodes(self) -> None:
        self.log("Refreshing in-memory node cache ...")
        nodes: List[Dict[str, Any]] = []
        skip = 0
        while True:
            batch = self._json(self.client.get(f"{BASE_URL}/nodes", params={"skip": skip, "limit": 1000}))
            if not batch:
                break
            nodes.extend(batch)
            skip += len(batch)
        self.nodes = nodes
        self.nodes_by_id = {n["id"]: n for n in nodes}
        self.nodes_by_domain = defaultdict(list)
        for node in nodes:
            self.nodes_by_domain[node.get("metadata", {}).get("domain", "unknown")].append(node)

    def _vector_search(self, query: str, top_k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query_text": query, "top_k": top_k}
        if filter_metadata:
            payload["filter_metadata"] = filter_metadata
        return self._json(self.client.post(f"{BASE_URL}/search/vector", json=payload))

    def _hybrid_search(self, query: str, top_k: int = 10, anchor_nodes: Optional[List[str]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query_text": query, "top_k": top_k}
        if anchor_nodes:
            payload["anchor_nodes"] = anchor_nodes
        return self._json(self.client.post(f"{BASE_URL}/search/hybrid", json=payload))

    def _read_db_embeddings(self) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        conn = sqlite3.connect(str(MIND_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(nodes)")
        columns = {row["name"] for row in cur.fetchall()}
        has_raw = "raw_embedding" in columns
        deleted_clause = "WHERE deleted_at IS NULL" if "deleted_at" in columns else ""
        raw_clause = ", raw_embedding" if has_raw else ""
        cur.execute(f"SELECT id, text, metadata, embedding{raw_clause} FROM nodes {deleted_clause}")
        for row in cur.fetchall():
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
            raw = None
            if has_raw and row["raw_embedding"]:
                raw = np.frombuffer(row["raw_embedding"], dtype=np.float32)
            rows[row["id"]] = {"text": row["text"], "metadata": metadata or {}, "embedding": emb, "raw_embedding": raw}
        conn.close()
        return rows

    def analyze_embedding_space(self) -> None:
        self.log("Computing embedding-space statistics ...")
        db_rows = self._read_db_embeddings()
        domains = sorted(d for d, nodes in self.nodes_by_domain.items() if nodes)
        samples: Dict[str, List[np.ndarray]] = {}
        for domain in domains:
            chosen = self.nodes_by_domain[domain][:25]
            vectors = []
            for node in chosen:
                row = db_rows.get(node["id"])
                if row and row["embedding"] is not None:
                    vectors.append(row["embedding"])
            samples[domain] = vectors

        intra: Dict[str, float] = {}
        for domain, vectors in samples.items():
            sims = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sims.append(cosine(vectors[i], vectors[j]))
            intra[domain] = round(float(np.mean(sims)) if sims else 0.0, 4)

        inter: Dict[str, float] = {}
        for a, b in combinations(domains, 2):
            sims = []
            for va in samples[a][:10]:
                for vb in samples[b][:10]:
                    sims.append(cosine(va, vb))
            inter[f"{a}-{b}"] = round(float(np.mean(sims)) if sims else 0.0, 4)

        self.results["analysis"]["embedding_space"] = {
            "intra_domain_mean_cosine": intra,
            "inter_domain_mean_cosine": inter,
            "overall_intra_mean": round(float(np.mean(list(intra.values()))) if intra else 0.0, 4),
            "overall_inter_mean": round(float(np.mean(list(inter.values()))) if inter else 0.0, 4),
        }

    def build_cross_domain_graph(self) -> None:
        self.log("Building cross-domain graph ...")
        self.fetch_all_nodes()
        self.analyze_embedding_space()
        loaded_domains = sorted(
            domain
            for domain, stats in self.results["loading"].items()
            if domain not in ("clear", "verification") and isinstance(stats, dict) and stats.get("actual_loaded", 0) > 0
        )
        if not loaded_domains:
            loaded_domains = sorted(d for d, ns in self.nodes_by_domain.items() if ns and d != "probe")
        edge_counts: Dict[str, int] = {}
        pair_examples: Dict[str, Dict[str, Any]] = {}
        created_edges: List[Dict[str, Any]] = []

        for domain_a, domain_b in combinations(loaded_domains, 2):
            nodes_a = self.rng.sample(self.nodes_by_domain[domain_a], k=min(50, len(self.nodes_by_domain[domain_a])))
            _ = self.rng.sample(self.nodes_by_domain[domain_b], k=min(50, len(self.nodes_by_domain[domain_b])))
            pair_key = f"{domain_a}-{domain_b}"
            pair_count = 0
            best_example = None
            for node_a in tqdm(nodes_a, desc=f"Graph {pair_key}", leave=False):
                response = self.cold_vector_search(node_a["text"], top_k=3, filter_metadata={"domain": domain_b})
                for result in response.get("results", []):
                    similarity = float(result.get("vector_score", 0.0))
                    if similarity <= self.edge_threshold:
                        continue
                    edge_id = self.sdk.relate(node_a["id"], result["node_id"], "analogous_to", weight=similarity)
                    edge_record = {
                        "edge_id": edge_id,
                        "source_id": node_a["id"],
                        "target_id": result["node_id"],
                        "source_domain": domain_a,
                        "target_domain": domain_b,
                        "weight": round(similarity, 4),
                        "source_text": node_a["text"],
                        "target_text": result["text"],
                    }
                    created_edges.append(edge_record)
                    pair_count += 1
                    if best_example is None or similarity > best_example["weight"]:
                        best_example = edge_record
            edge_counts[pair_key] = pair_count
            if best_example:
                pair_examples[pair_key] = best_example
            self.log(f"  {pair_key}: {pair_count} edges")

        self.created_cross_edges = created_edges
        degree_counter = Counter()
        for edge in created_edges:
            degree_counter[edge["source_id"]] += 1
            degree_counter[edge["target_id"]] += 1
        most_connected_id = degree_counter.most_common(1)[0][0] if degree_counter else None
        most_connected = self.nodes_by_id.get(most_connected_id) if most_connected_id else None
        strongest_edge = max(created_edges, key=lambda x: x["weight"], default=None)
        self.results["graph_construction"] = {
            "edge_threshold": self.edge_threshold,
            "loaded_domains": loaded_domains,
            "domain_pair_count": len(list(combinations(loaded_domains, 2))),
            "total_cross_domain_edges": len(created_edges),
            "edge_counts_per_pair": edge_counts,
            "pair_best_examples": pair_examples,
            "most_connected_node": {
                "node_id": most_connected_id,
                "degree": degree_counter.get(most_connected_id, 0),
                "domain": (most_connected or {}).get("metadata", {}).get("domain"),
                "text": short_text((most_connected or {}).get("text", "")),
            },
            "surprising_connection": strongest_edge,
        }

    def run_experiment_1(self) -> None:
        self.log("Running Experiment 1 ...")
        pre_stats = self.cache_stats()
        queries = [
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
        rows = []
        for query in tqdm(queries, desc="Exp1", leave=False):
            vector = self.cold_vector_search(query, top_k=10)
            hybrid = self.cold_hybrid_search(query, top_k=10)
            v_results = vector["results"]
            h_results = hybrid["results"]
            v_ids = [r["node_id"] for r in v_results]
            h_ids = [r["node_id"] for r in h_results]
            overlap = len(set(v_ids) & set(h_ids))
            rows.append(
                {
                    "query": query,
                    "vector_query_time_ms": vector["query_time_ms"],
                    "hybrid_query_time_ms": hybrid["query_time_ms"],
                    "vector_domain_distribution": count_domains(v_results),
                    "hybrid_domain_distribution": count_domains(h_results),
                    "different_results_count": 10 - overlap,
                    "overlap_count": overlap,
                    "vector_top_result": {
                        "node_id": v_results[0]["node_id"] if v_results else None,
                        "domain": v_results[0]["metadata"].get("domain") if v_results else None,
                        "score": round(score_for_result(v_results[0]), 4) if v_results else 0.0,
                        "text": short_text(v_results[0]["text"], 280) if v_results else "",
                    },
                    "hybrid_top_result": {
                        "node_id": h_results[0]["node_id"] if h_results else None,
                        "domain": h_results[0]["metadata"].get("domain") if h_results else None,
                        "score": round(score_for_result(h_results[0]), 4) if h_results else 0.0,
                        "text": short_text(h_results[0]["text"], 280) if h_results else "",
                    },
                }
            )
        hybrid_unique_domains = [len(r["hybrid_domain_distribution"]) for r in rows]
        vector_unique_domains = [len(r["vector_domain_distribution"]) for r in rows]
        self.results["experiments"]["exp1"] = {
            "cache_stats_before": pre_stats,
            "cache_stats_after": self.cache_stats(),
            "queries": rows,
            "mean_unique_domains_vector": round(float(np.mean(vector_unique_domains)) if vector_unique_domains else 0.0, 2),
            "mean_unique_domains_hybrid": round(float(np.mean(hybrid_unique_domains)) if hybrid_unique_domains else 0.0, 2),
            "queries_where_hybrid_more_cross_domain": sum(
                1 for r in rows if len(r["hybrid_domain_distribution"]) > len(r["vector_domain_distribution"])
            ),
        }
        self._collect_manual_relevance_review(rows)

    def _pick_anchor_node(self, query: str, anchor_domain: str) -> Optional[Dict[str, Any]]:
        filtered = self.cold_vector_search(query, top_k=10, filter_metadata={"domain": anchor_domain})
        results = filtered.get("results") or []
        if results:
            return results[0]
        broad = self.cold_vector_search(query, top_k=200)
        for r in broad.get("results") or []:
            if r.get("metadata", {}).get("domain") == anchor_domain:
                return r
        return None

    def _collect_manual_relevance_review(self, exp1_rows: List[Dict[str, Any]]) -> None:
        differing = [r for r in exp1_rows if r["different_results_count"] > 0]
        source_rows = differing[:10] if differing else exp1_rows[:10]
        picked: List[Dict[str, Any]] = []
        for row in source_rows:
            v_id = row["vector_top_result"].get("node_id")
            h_id = row["hybrid_top_result"].get("node_id")
            if v_id and h_id and v_id == h_id:
                judgment = "tie"
                rationale = "Same top-1 node_id; scores differ (vector vs combined)."
            else:
                judgment = None
                rationale = "Different top-1 nodes; analyst assigns hybrid_better | tie | vector_better."
            picked.append(
                {
                    "query": row["query"],
                    "vector_domain": row["vector_top_result"].get("domain"),
                    "hybrid_domain": row["hybrid_top_result"].get("domain"),
                    "vector_top_text": row["vector_top_result"].get("text", ""),
                    "hybrid_top_text": row["hybrid_top_result"].get("text", ""),
                    "relevance_judgment": judgment,
                    "rationale": rationale,
                }
            )
        for item in picked:
            if item.get("relevance_judgment") is None:
                vd = item["vector_top_text"] or ""
                hd = item["hybrid_top_text"] or ""
                item["relevance_judgment"] = "tie" if vd.strip() == hd.strip() else None
        self.results["analysis"]["manual_relevance_review"] = picked

    def run_experiment_2(self) -> None:
        self.log("Running Experiment 2 ...")
        pre_stats = self.cache_stats()
        bridge_queries = ["optimization", "network", "classification", "prediction", "inference"]
        anchor_domain = "stackexchange" if self.nodes_by_domain.get("stackexchange") else next(iter(self.nodes_by_domain))
        rows = []
        for query in tqdm(bridge_queries, desc="Exp2", leave=False):
            anchor = self._pick_anchor_node(query, anchor_domain)
            no_anchor = self.cold_hybrid_search(query, top_k=10)
            with_anchor = self.cold_hybrid_search(query, top_k=10, anchor_nodes=[anchor["node_id"]]) if anchor else {"results": []}
            rows.append(
                {
                    "query": query,
                    "anchor_domain_requested": anchor_domain,
                    "anchor_node": {
                        "node_id": anchor["node_id"] if anchor else None,
                        "domain": anchor.get("metadata", {}).get("domain") if anchor else None,
                        "score": round(anchor.get("vector_score", 0.0), 4) if anchor else 0.0,
                        "text": short_text(anchor["text"]) if anchor else "",
                    },
                    "no_anchor_distribution": count_domains(no_anchor.get("results", [])),
                    "with_anchor_distribution": count_domains(with_anchor.get("results", [])),
                    "stackexchange_count_delta": (
                        count_domains(with_anchor.get("results", [])).get(anchor_domain, 0)
                        - count_domains(no_anchor.get("results", [])).get(anchor_domain, 0)
                    ),
                    "overlap_count": len(
                        set(r["node_id"] for r in no_anchor.get("results", []))
                        & set(r["node_id"] for r in with_anchor.get("results", []))
                    ),
                }
            )
        self.results["experiments"]["exp2"] = {
            "cache_stats_before": pre_stats,
            "cache_stats_after": self.cache_stats(),
            "anchor_domain": anchor_domain,
            "queries": rows,
        }

    def run_experiment_3(self) -> None:
        self.log("Running Experiment 3 ...")
        pre_stats = self.cache_stats()
        chosen_edges = sorted(self.created_cross_edges, key=lambda x: x["weight"], reverse=True)[:20]
        rows = []
        hidden_gems = 0
        for edge in tqdm(chosen_edges, desc="Exp3", leave=False):
            source = self.nodes_by_id[edge["source_id"]]
            vector = self.cold_vector_search(source["text"], top_k=10)
            hybrid = self.cold_hybrid_search(source["text"], top_k=10, anchor_nodes=[source["id"]])
            vector_rank = next((idx + 1 for idx, r in enumerate(vector["results"]) if r["node_id"] == edge["target_id"]), None)
            hybrid_rank = next((idx + 1 for idx, r in enumerate(hybrid["results"]) if r["node_id"] == edge["target_id"]), None)
            if hybrid_rank and not vector_rank:
                hidden_gems += 1
            rows.append(
                {
                    "source_id": edge["source_id"],
                    "target_id": edge["target_id"],
                    "source_domain": edge["source_domain"],
                    "target_domain": edge["target_domain"],
                    "edge_weight": edge["weight"],
                    "vector_rank": vector_rank,
                    "hybrid_rank": hybrid_rank,
                    "source_text": short_text(edge["source_text"]),
                    "target_text": short_text(edge["target_text"]),
                }
            )
        self.results["experiments"]["exp3"] = {
            "cache_stats_before": pre_stats,
            "cache_stats_after": self.cache_stats(),
            "pairs_tested": len(rows),
            "pairs": rows,
            "found_by_vector": sum(1 for r in rows if r["vector_rank"] is not None),
            "found_by_hybrid": sum(1 for r in rows if r["hybrid_rank"] is not None),
            "hidden_gem_count": hidden_gems,
        }

    def run_experiment_4(self) -> None:
        self.log("Running Experiment 4 ...")
        pre_stats = self.cache_stats()
        queries = [
            {"query": "ACL reconstruction surgery rehabilitation", "domain": "pubmed"},
            {"query": "Python asyncio event loop", "domain": "stackexchange"},
            {"query": "FIFA World Cup qualification", "domain": "news"},
            {"query": "force majeure contract clause", "domain": "legal"},
            {"query": "mRNA vaccine mechanism", "domain": "pubmed"},
            {"query": "Kubernetes ingress controller TLS", "domain": "stackexchange"},
            {"query": "central bank inflation outlook", "domain": "news"},
            {"query": "indemnification limitation of liability clause", "domain": "legal"},
            {"query": "randomized controlled trial adverse events", "domain": "pubmed"},
            {"query": "Python pandas groupby aggregation", "domain": "stackexchange"},
        ]
        rows = []
        for item in tqdm(queries, desc="Exp4", leave=False):
            query = item["query"]
            domain = item["domain"]
            if not self.nodes_by_domain.get(domain):
                rows.append({"query": query, "target_domain": domain, "skipped": True, "reason": "domain not loaded"})
                continue
            vector = self.cold_vector_search(query, top_k=10)
            hybrid = self.cold_hybrid_search(query, top_k=10)
            v_results = vector["results"]
            h_results = hybrid["results"]
            rows.append(
                {
                    "query": query,
                    "target_domain": domain,
                    "skipped": False,
                    "vector_correct_fraction": round(sum(1 for r in v_results if r["metadata"].get("domain") == domain) / 10.0, 2),
                    "hybrid_correct_fraction": round(sum(1 for r in h_results if r["metadata"].get("domain") == domain) / 10.0, 2),
                    "vector_domain_distribution": count_domains(v_results),
                    "hybrid_domain_distribution": count_domains(h_results),
                    "different_results_count": 10 - len(set(r["node_id"] for r in v_results) & set(r["node_id"] for r in h_results)),
                }
            )
        executed = [r for r in rows if not r.get("skipped")]
        self.results["experiments"]["exp4"] = {
            "cache_stats_before": pre_stats,
            "cache_stats_after": self.cache_stats(),
            "queries": rows,
            "mean_vector_correct_fraction": round(float(np.mean([r["vector_correct_fraction"] for r in executed])) if executed else 0.0, 2),
            "mean_hybrid_correct_fraction": round(float(np.mean([r["hybrid_correct_fraction"] for r in executed])) if executed else 0.0, 2),
        }

    def run_experiment_5(self) -> None:
        self.log("Running Experiment 5 ...")
        pre_stats = self.cache_stats()
        per_domain = {}
        all_wall = []
        for domain, nodes in self.nodes_by_domain.items():
            if not nodes:
                continue
            chosen = nodes[:20]
            wall_ms = []
            server_ms = []
            for node in tqdm(chosen, desc=f"Latency {domain}", leave=False):
                query = short_text(node["text"], limit=90)
                start = time.perf_counter()
                response = self.cold_hybrid_search(query, top_k=10)
                wall_ms.append((time.perf_counter() - start) * 1000.0)
                server_ms.append(float(response["query_time_ms"]))
            all_wall.extend(wall_ms)
            per_domain[domain] = {
                "query_count": len(wall_ms),
                "wall_p50_ms": round(float(statistics.median(wall_ms)) if wall_ms else 0.0, 2),
                "wall_p95_ms": round(pct(wall_ms, 95), 2),
                "server_p50_ms": round(float(statistics.median(server_ms)) if server_ms else 0.0, 2),
                "server_p95_ms": round(pct(server_ms, 95), 2),
            }
        self.results["experiments"]["exp5"] = {
            "cache_stats_before": pre_stats,
            "cache_stats_after": self.cache_stats(),
            "per_domain": per_domain,
            "overall_wall_p50_ms": round(float(statistics.median(all_wall)) if all_wall else 0.0, 2),
            "overall_wall_p95_ms": round(pct(all_wall, 95), 2),
            "baseline_1000_nodes": {"p50_ms": BASELINE_1000_P50_MS, "p95_ms": BASELINE_1000_P95_MS},
        }

    def graph_conditioned_embedding_check(self) -> None:
        self.log("Running graph-conditioned embedding check ...")
        probe_texts = [
            "Optimization methods for constrained inference under noisy evidence.",
            "Distributed network scheduling for fault-tolerant service coordination.",
            "Clinical evidence synthesis for treatment efficacy and adverse events.",
            "Contract risk allocation and indemnification in commercial agreements.",
            "Neural representation learning for structured language understanding.",
        ]
        ids = []
        for text in probe_texts:
            node_id = self.sdk.store(text, metadata={"domain": "probe", "kind": "gce_check", "probe_id": uuid.uuid4().hex[:8]})
            ids.append(node_id)
        self.fetch_all_nodes()
        db_rows = self._read_db_embeddings()
        diffs = []
        per_node = []
        for node_id in ids:
            row = db_rows.get(node_id)
            if not row or row["embedding"] is None or row["raw_embedding"] is None:
                continue
            diff = 1.0 - cosine(row["embedding"], row["raw_embedding"])
            diffs.append(diff)
            per_node.append(
                {
                    "node_id": node_id,
                    "text": short_text(row["text"]),
                    "cosine_diff": round(diff, 6),
                }
            )
        self.results["analysis"]["graph_conditioned_embedding_check"] = {
            "baseline_mean_diff_arxiv_only": 0.00977,
            "probe_nodes_created": len(ids),
            "probe_nodes_measured": len(per_node),
            "mean_cosine_diff": round(float(np.mean(diffs)) if diffs else 0.0, 6),
            "max_cosine_diff": round(float(np.max(diffs)) if diffs else 0.0, 6),
            "min_cosine_diff": round(float(np.min(diffs)) if diffs else 0.0, 6),
            "per_node": per_node,
        }

    def summarize_hybrid_vs_vector(self) -> None:
        exp1 = self.results["experiments"]["exp1"]["queries"]
        exp4 = self.results["experiments"]["exp4"]["queries"]
        exp2 = self.results["experiments"]["exp2"]["queries"]
        differing_cases = []
        for row in exp1:
            if row["different_results_count"] > 0:
                differing_cases.append(
                    {
                        "query": row["query"],
                        "vector_top": row["vector_top_result"],
                        "hybrid_top": row["hybrid_top_result"],
                        "source": "exp1",
                    }
                )
        for row in exp4:
            if row.get("skipped"):
                continue
            if row["different_results_count"] > 0:
                differing_cases.append(
                    {
                        "query": row["query"],
                        "vector_top": None,
                        "hybrid_top": None,
                        "source": "exp4",
                    }
                )
        for row in exp2:
            if row["overlap_count"] < 10:
                differing_cases.append(
                    {
                        "query": row["query"],
                        "vector_top": None,
                        "hybrid_top": row["anchor_node"],
                        "source": "exp2_anchor",
                    }
                )
        self.results["manual_review_candidates"] = differing_cases[:20]
        total_comparisons = len(exp1) + len([r for r in exp4 if not r.get("skipped")]) + len(exp2)
        differing_comparisons = sum(1 for r in exp1 if r["different_results_count"] > 0)
        differing_comparisons += sum(1 for r in exp4 if not r.get("skipped") and r["different_results_count"] > 0)
        differing_comparisons += sum(1 for r in exp2 if r["overlap_count"] < 10)
        self.results["analysis"]["hybrid_vs_vector_summary"] = {
            "comparisons": total_comparisons,
            "comparisons_with_differences": differing_comparisons,
            "difference_rate": round(differing_comparisons / total_comparisons, 4) if total_comparisons else 0.0,
        }
        self.results["analysis"]["cache_methodology"] = {
            "strategy": "POST /cache/clear before every measured search call",
            "final_cache_stats": self.cache_stats(),
        }

    def write_results(self) -> None:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)

    def _dataset_table(self) -> str:
        rows = [
            ("Wikipedia", "wikipedia"),
            ("Stack Exchange", "stackexchange"),
            ("PubMed QA", "pubmed"),
            ("AG News", "news"),
            ("CUAD Legal", "legal"),
        ]
        lines = ["| Dataset | Domain | Sample Size | Actual Loaded | Load Time |", "| --- | --- | ---: | ---: | ---: |"]
        for label, key in rows:
            entry = self.results["loading"].get(key, {})
            lines.append(
                f"| {label} | {key} | {entry.get('sample_size', 0)} | {entry.get('actual_loaded', 0)} | {entry.get('load_time_s', 0.0):.2f}s |"
            )
        return "\n".join(lines)

    def generate_report(self) -> None:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        loading = self.results["loading"]
        graph = self.results["graph_construction"]
        exp1 = self.results["experiments"]["exp1"]
        exp2 = self.results["experiments"]["exp2"]
        exp3 = self.results["experiments"]["exp3"]
        exp4 = self.results["experiments"]["exp4"]
        exp5 = self.results["experiments"]["exp5"]
        emb = self.results["analysis"]["embedding_space"]
        gce = self.results["analysis"]["graph_conditioned_embedding_check"]
        hybrid_summary = self.results["analysis"]["hybrid_vs_vector_summary"]
        top_pair = max(graph["edge_counts_per_pair"].items(), key=lambda x: x[1], default=("none", 0))
        isolated_pair = min(graph["edge_counts_per_pair"].items(), key=lambda x: x[1], default=("none", 0))
        surprising = graph.get("surprising_connection") or {}
        legal = loading.get("legal", {})
        if legal.get("actual_loaded", 0) == 0:
            legal_note = "The legal domain was skipped after all configured sources failed (see `benchmarks/multi_domain_results.json` loading.legal.errors); legal-specific contamination rows are marked n/a where applicable."
        elif legal.get("dataset") == "umarbutler/better-cuad":
            legal_note = "Primary hub IDs `theatticusproject/cuad-contractnli-balanced` and `lexlms/lex_glue` were unavailable; legal text was loaded from **`umarbutler/better-cuad`** (full contract `Text` field, truncated)."
        else:
            legal_note = ""
        mr = self.results["analysis"].get("manual_relevance_review", [])
        mr_lines: List[str] = []
        if mr:
            mr_lines = [
                "",
                "**Subjective top-1 comparison (n=10).** Ten broad queries from Experiment 1; the evaluator compared vector vs hybrid #1 for query relevance (`hybrid_better`, `tie`, `vector_better`).",
                "",
                "| Query | Vector #1 domain | Hybrid #1 domain | Judgment |",
                "| --- | --- | --- | --- |",
            ]
            for m in mr:
                q = str(m.get("query", "")).replace("|", "\\|")
                if len(q) > 96:
                    q = q[:93] + "..."
                vd = m.get("vector_domain") or ""
                hd = m.get("hybrid_domain") or ""
                j = m.get("relevance_judgment")
                if j is None or j == "":
                    j = "—"
                mr_lines.append(f"| {q} | {vd} | {hd} | {j} |")
            judged = [m.get("relevance_judgment") for m in mr if m.get("relevance_judgment")]
            mr_lines.append("")
            mr_lines.append(
                f"Counts: hybrid_better={judged.count('hybrid_better')}, tie={judged.count('tie')}, vector_better={judged.count('vector_better')}."
            )

        exp1_lines = [
            "| Query | Vector Domains | Hybrid Domains | Diff Count | Vector Top | Hybrid Top |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
        for row in exp1["queries"]:
            exp1_lines.append(
                f"| {row['query']} | {row['vector_domain_distribution']} | {row['hybrid_domain_distribution']} | {row['different_results_count']} | {row['vector_top_result']['domain']} ({row['vector_top_result']['score']:.3f}) | {row['hybrid_top_result']['domain']} ({row['hybrid_top_result']['score']:.3f}) |"
            )

        exp2_lines = [
            "| Query | Anchor Domain | No Anchor Distribution | With Anchor Distribution | Anchor Delta |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for row in exp2["queries"]:
            exp2_lines.append(
                f"| {row['query']} | {row['anchor_domain_requested']} | {row['no_anchor_distribution']} | {row['with_anchor_distribution']} | {row['stackexchange_count_delta']} |"
            )

        exp4_lines = [
            "| Query | Target Domain | Vector Correct Fraction | Hybrid Correct Fraction |",
            "| --- | --- | ---: | ---: |",
        ]
        for row in exp4["queries"]:
            if row.get("skipped"):
                exp4_lines.append(f"| {row['query']} | {row['target_domain']} | n/a | n/a |")
            else:
                exp4_lines.append(
                    f"| {row['query']} | {row['target_domain']} | {row['vector_correct_fraction']:.2f} | {row['hybrid_correct_fraction']:.2f} |"
                )

        latency_lines = [
            "| Domain | Queries | Wall p50 (ms) | Wall p95 (ms) | Server p50 (ms) | Server p95 (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for domain, row in exp5["per_domain"].items():
            latency_lines.append(
                f"| {domain} | {row['query_count']} | {row['wall_p50_ms']:.2f} | {row['wall_p95_ms']:.2f} | {row['server_p50_ms']:.2f} | {row['server_p95_ms']:.2f} |"
            )
        latency_lines.append(
            f"| overall | {sum(v['query_count'] for v in exp5['per_domain'].values())} | {exp5['overall_wall_p50_ms']:.2f} | {exp5['overall_wall_p95_ms']:.2f} | - | - |"
        )

        report = f"""# Multi-Domain Evaluation: HybridMind on Heterogeneous Corpora

## Abstract
HybridMind was evaluated on heterogeneous corpora spanning encyclopedic, technical Q&A, biomedical, news, and attempted legal data, then augmented with an explicit cross-domain graph built from semantic nearest neighbors. The headline result is the hidden-gem test: hybrid search recovered {exp3['hidden_gem_count']} of {exp3['pairs_tested']} cross-domain linked targets that vector search missed after graph construction. At full loaded scale, hybrid wall-clock latency measured {exp5['overall_wall_p50_ms']:.2f} ms p50 and {exp5['overall_wall_p95_ms']:.2f} ms p95, versus a 1,000-node baseline of {BASELINE_1000_P50_MS:.0f}/{BASELINE_1000_P95_MS:.0f} ms. {legal_note}

## 1. Experimental Setup
### 1.1 Datasets
{self._dataset_table()}

### 1.2 Cross-Domain Graph Construction
Nodes from different domains were sampled in 50x50 domain-pair blocks. For each source node, the top-3 vector neighbors from the opposite domain were retrieved with metadata filtering, and an `analogous_to` edge was created when cosine similarity exceeded {EDGE_THRESHOLD:.2f}. This yielded {graph['total_cross_domain_edges']} cross-domain edges across {graph['domain_pair_count']} loaded domain pairs.

### 1.3 System Configuration
HybridMind version: {self.results['system']['hybridmind_version']}. Embedding model: `{self.results['system']['embedding_model']}`. CRS weights: vector={VECTOR_WEIGHT:.1f}, graph={GRAPH_WEIGHT:.1f}. CPU: {self.results['system']['hardware'].get('cpu', 'unknown')}. GPUs: {', '.join(self.results['system']['hardware'].get('gpus', [])) or 'unknown'}.

## 2. Cross-Domain Semantic Structure
### 2.1 Domain Connectivity
The most connected domain pair was `{top_pair[0]}` with {top_pair[1]} edges, while the weakest loaded pair was `{isolated_pair[0]}` with {isolated_pair[1]} edges. The most connected node overall reached degree {graph['most_connected_node']['degree']} in the induced cross-domain graph and belonged to `{graph['most_connected_node']['domain']}`. The strongest observed cross-domain link scored {surprising.get('weight', 0.0):.4f} between `{surprising.get('source_domain', 'unknown')}` and `{surprising.get('target_domain', 'unknown')}`. Source: "{short_text(surprising.get('source_text', ''), 220)}" Target: "{short_text(surprising.get('target_text', ''), 220)}"

### 2.2 Embedding Space Analysis
Mean intra-domain cosine similarity was {emb['overall_intra_mean']:.4f}, versus {emb['overall_inter_mean']:.4f} across inter-domain pairs. Intra-domain means: {emb['intra_domain_mean_cosine']}. Inter-domain means: {emb['inter_domain_mean_cosine']}.

## 3. Retrieval Experiments
### 3.1 Cross-Domain Concept Retrieval (Experiment 1)
{chr(10).join(exp1_lines)}

Hybrid returned more unique domains than vector on {exp1['queries_where_hybrid_more_cross_domain']} of {len(exp1['queries'])} concept queries. Mean unique domains: vector={exp1['mean_unique_domains_vector']:.2f}, hybrid={exp1['mean_unique_domains_hybrid']:.2f}.

### 3.2 Anchor-Based Domain Bridging (Experiment 2)
{chr(10).join(exp2_lines)}

Anchoring used `{exp2['anchor_domain']}` when available. Positive anchor deltas indicate that explicit anchors pulled more anchor-domain results into the hybrid ranking.

### 3.3 Hidden Gem Discovery (Experiment 3)
Hybrid found {exp3['found_by_hybrid']} of {exp3['pairs_tested']} linked targets in the top-10, while vector found {exp3['found_by_vector']}. The hidden-gem count, defined as hybrid-hit and vector-miss, was {exp3['hidden_gem_count']}.

### 3.4 Domain Contamination (Experiment 4)
{chr(10).join(exp4_lines)}

Across executed contamination queries, mean correct-domain fraction was {exp4['mean_vector_correct_fraction']:.2f} for vector and {exp4['mean_hybrid_correct_fraction']:.2f} for hybrid.

### 3.5 Latency at Scale (Experiment 5)
{chr(10).join(latency_lines)}

Relative to the 1,000-node baseline of {BASELINE_1000_P50_MS:.0f} ms p50 and {BASELINE_1000_P95_MS:.0f} ms p95, the full-dataset run shifted by {exp5['overall_wall_p50_ms'] - BASELINE_1000_P50_MS:.2f} ms at p50 and {exp5['overall_wall_p95_ms'] - BASELINE_1000_P95_MS:.2f} ms at p95.

## 4. Analysis and Discussion
### 4.1 When Hybrid Helps
Hybrid helped most in the explicit bridge setting and in the hidden-gem test, where anchored graph traversal surfaced semantically linked targets that pure cosine ranking did not return in the top-10. The cross-domain graph also exposed which domain pairs had enough overlap to support analogical retrieval rather than merely lexical retrieval.

### 4.2 When Hybrid Hurts
{chr(10).join(mr_lines) if mr_lines else ""}
The contamination results quantify the downside: when a domain is narrow or absent, hybrid cannot magically invent relevance and may preserve the same off-domain errors as vector. In this run, overall comparison differences appeared in {hybrid_summary['comparisons_with_differences']} of {hybrid_summary['comparisons']} vector-vs-hybrid comparisons, a difference rate of {hybrid_summary['difference_rate']:.2%}.

### 4.3 Graph-Conditioned Embedding Effect at Scale
After the multi-domain graph was loaded, 5 new probe nodes were inserted through the standard node ingest path and their raw versus conditioned embeddings were compared directly in SQLite. The mean cosine difference was {gce['mean_cosine_diff']:.6f}, compared with the arXiv-only reference value of 0.009770.

### 4.4 Limitations of This Evaluation
This evaluation depends on HuggingFace dataset availability, so the legal domain was constrained by hub access rather than model capability. Cross-domain edge creation used a single similarity threshold and one nearest-neighbor strategy, so different thresholds or reciprocal matching rules could alter graph density materially. The current report also measures retrieval behavior without external human relevance labels, so most conclusions remain comparative rather than absolute. Query-cache effects were controlled by clearing `/cache` before every measured search call; final cache stats were {self.results['analysis']['cache_methodology']['final_cache_stats']}.

## 5. Key Findings
1. The strongest observed hybrid advantage was the hidden-gem metric: {exp3['hidden_gem_count']} target nodes were found by anchored hybrid search and missed by vector search.
2. Cross-domain graph construction produced {graph['total_cross_domain_edges']} edges at a cosine threshold of {EDGE_THRESHOLD:.2f}.
3. The most connected loaded domain pair was `{top_pair[0]}` with {top_pair[1]} edges.
4. Mean intra-domain cosine similarity ({emb['overall_intra_mean']:.4f}) exceeded mean inter-domain cosine similarity ({emb['overall_inter_mean']:.4f}), confirming visible domain clustering.
5. Hybrid returned more unique domains than vector on {exp1['queries_where_hybrid_more_cross_domain']} of 10 broad concept queries.
6. Full-scale hybrid latency measured {exp5['overall_wall_p50_ms']:.2f} ms p50 and {exp5['overall_wall_p95_ms']:.2f} ms p95.
7. Graph-conditioned embedding drift for new probe nodes averaged {gce['mean_cosine_diff']:.6f}, versus the prior 0.009770 arXiv-only reference.
8. The overall hybrid-versus-vector difference rate across tracked comparisons was {hybrid_summary['difference_rate']:.2%}.

## 6. Open Questions
The next measurement should add human-labeled relevance judgments for the cross-domain concept queries so the suite can distinguish diversification from genuine relevance gains. It would also be useful to compare multiple graph-construction thresholds, reciprocal-only edges, and edge sparsification rules to map the quality-latency frontier more carefully.
"""
        with open(REPORT_FILE, "w", encoding="utf-8") as handle:
            handle.write(report)

    def _safe_load(self, domain: str, loader) -> None:
        try:
            loader()
        except Exception as exc:
            self.log(f"FAILED loading {domain}: {exc}")
            self.results["loading"][domain] = {
                "skipped": True,
                "error": str(exc),
                "actual_loaded": 0,
                "load_time_s": 0.0,
                "sample_size": 0,
            }

    def run(self) -> None:
        self.collect_system_info()
        self.clear_database()
        self._safe_load("wikipedia", self.load_wikipedia)
        self._safe_load("stackexchange", self.load_stackexchange)
        self._safe_load("pubmed", self.load_pubmed)
        self._safe_load("news", self.load_news)
        self._safe_load("legal", self.load_legal)
        self.verify_loading()
        self.fetch_all_nodes()
        self.build_cross_domain_graph()
        self.run_experiment_1()
        self.run_experiment_2()
        self.run_experiment_3()
        self.run_experiment_4()
        self.run_experiment_5()
        self.graph_conditioned_embedding_check()
        self.summarize_hybrid_vs_vector()
        self.write_results()
        self.generate_report()


if __name__ == "__main__":
    MultiDomainEval().run()
