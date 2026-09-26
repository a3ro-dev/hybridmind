#!/usr/bin/env python3
"""Offline reproduction of the EMG paper's LoCoMo retrieval recall (R1).

Runs the *upstream* Entity-Memory Graph retrieval code, unmodified, from a
read-only clone and scores it with the official LoCoMo ``recall_acc``.

Upstream: https://github.com/Sun668/em_graph_memory
  pinned commit f020e855be06ac9f33ec888945ff6b305d81cb07 (tag v1.0.8)
  files executed: code/em_graph/** (build, cache, recall),
  experiments/exp_2026_07_27_locomo_stack_refactor/run.py (VARIANTS,
  build_graph, identities); metric re-implemented from
  code/locomo_eval/vendor/locomo/task_eval/{evaluation,evaluation_stats}.py
  SPDX-License-Identifier: MIT -- Copyright (c) 2026 Sun668
Paper: arXiv 2608.27925 ("Entity-Memory graph retrieval", S. Sun).

Zero provider calls: OpenAI client constructors, socket connects and
``nltk.download`` are replaced with functions that raise before any upstream
module is imported, so a cache miss fails loudly instead of calling out.
Protocol: research/experiments/r1-emg-reproduction/protocol.md.

``--port`` runs HybridMind's own port (``engine/entity_graph.py``) on the same
shipped graphs, question keys and cached 1536-d reference vectors, reports
per-question ranking parity against the upstream code, then measures
HybridMind variants (PPR, lexical-extractor graph) on those artifacts. It adds
``port`` and ``variants`` keys to the existing result JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = REPO_ROOT / "tmp" / "upstream" / "em_graph_memory"
DEFAULT_NLTK_DATA = REPO_ROOT / "tmp" / "nltk_data"
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "repro-emg-locomo-20260926.json"
EXP_REL = Path("experiments") / "exp_2026_07_27_locomo_stack_refactor"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from engine.dense_channel import DenseMatrix, exact_scores  # noqa: E402
from engine.entity_extraction import LexicalEntityExtractor  # noqa: E402
from engine.entity_graph import DEVIATIONS as GRAPH_DEVIATIONS  # noqa: E402
from engine.entity_graph import EntityMemoryGraph  # noqa: E402
from eval_stats import DEFAULT_SEED  # noqa: E402
from scripts.offline_budgeted_evidence import cluster_bootstrap  # noqa: E402

UPSTREAM_REPO = "https://github.com/Sun668/em_graph_memory"
UPSTREAM_COMMIT = "f020e855be06ac9f33ec888945ff6b305d81cb07"
UPSTREAM_LICENSE = "MIT; Copyright (c) 2026 Sun668"
DATASET_SHA256 = "047d8e2528126afd02ab4b6fbade825ad390447e7a91861d35960c85752b4d74"
QUERY_ARTIFACT_REL = Path(
    "outputs/em_graph/query_embeddings/locomo10_047d8e25_text-embedding-3-small_v1.npz"
)
QUERY_ARTIFACT_SHA256 = "bef99a912383f201e18f2ccecfda0e28ae745b006016a6b38e6ce661eb986f9f"
EMBEDDING_MODEL = "text-embedding-3-small"
EXTRACT_MODEL = "gpt-3.5-turbo"
STACK_TAG = "gpt35_tes"
VECTOR_KINDS = ("memory_only", "extract_v4")
L2_NORMALIZATION = "l2_float32_v1"  # upstream embedding_index.L2_NORMALIZATION

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
METRIC_TOLERANCE = 1e-9
PASS_TOLERANCE_POINTS = 0.05
QUALITATIVE_B_MINUS_A_25 = 4.7374
QUALITATIVE_WINDOW_POINTS = 1.0

# Formal runs the published numbers come from. A@50 is the paper-valid retry;
# the first A@50 run (deb2ff2) was diagnostic only.
FORMAL_RUNS: Dict[Tuple[str, int], str] = {
    ("A", 5): "formal_all10_M4_A_top5_c4e3ad2_qfrozen_run01",
    ("A", 10): "formal_all10_M4_A_top10_0eee9b8_qfrozen_run01",
    ("A", 25): "formal_all10_M1_A_top25_76fcf5b_qfrozen_run01",
    ("A", 50): "formal_all10_M4_A_top50_retry_e7a5abf_qfrozen_run01",
    ("B", 5): "formal_all10_M4_B_top5_dff9df4_qfrozen_run01",
    ("B", 10): "formal_all10_M4_B_top10_54ae6a9_qfrozen_run01",
    ("B", 25): "formal_all10_M1_B_top25_41a7812_qfrozen_run01",
    ("B", 50): "formal_all10_M4_B_top50_6afa882_qfrozen_run01",
    ("B_embed", 25): "formal_all10_M2_B_embed_top25_1fe232f_qfrozen_run02",
    ("B_entity", 25): "formal_all10_M2_B_entity_top25_31d8c78_qfrozen_run01",
    ("B_noseq", 25): "formal_all10_M2_B_noseq_top25_cabad81_qfrozen_run01",
    ("B_gate", 25): "formal_all10_M2_B_gate_top25_faebb2f_qfrozen_run01",
    ("B_gate_seq", 25): "formal_all10_M2_B_gate_seq_top25_1bb800b_qfrozen_run01",
}

# Published recall (percent, all 1,986 rows) from the paper / results_tables.md.
PUBLISHED: Dict[Tuple[str, int], float] = {
    ("A", 5): 59.3584,
    ("A", 10): 68.9145,
    ("A", 25): 79.7468,
    ("A", 50): 86.7148,
    ("B", 5): 64.1667,
    ("B", 10): 74.4847,
    ("B", 25): 84.4842,
    ("B", 50): 90.3306,
    ("B_embed", 25): 79.7468,
    ("B_entity", 25): 67.0208,
    ("B_noseq", 25): 82.8114,
    ("B_gate", 25): 79.2853,
    ("B_gate_seq", 25): 79.7217,
}
PUBLISHED_CAT1_4_K25 = {"A": 82.8099, "B": 85.0232}

# Shipped pre-refactor answer checkpoints (legacy matched stack) per arm.
LEGACY_CHECKPOINTS = {
    "A": "all10_gpt35_tes_em_embed_fullpool_answers.checkpoint.json",
    "B_embed": "all10_gpt35_tes_em_embed_fullpool_answers.checkpoint.json",
    "B": "all10_gpt35_tes_em_full_0.3_0.7_answers.checkpoint.json",
    "B_noseq": "all10_gpt35_tes_em_full_noseq_answers.checkpoint.json",
    "B_entity": "all10_gpt35_tes_em_entity_only_answers.checkpoint.json",
}

DEVIATIONS: List[str] = [
    "Entity graphs: the shipped legacy outputs/em_graph/conv-*_em_graph_extract_v4_gpt35_tes.json "
    "are used; the formal runs read graphs/<identity-sha>.json, which are not published and whose "
    "recorded sha256 differ from the shipped files.",
    "Question entity keys: the shipped legacy conv-*_qkeys_gpt35_tes.json (keyed by 1-based QA "
    "index only) are served through a QuestionEntityCache stand-in; the formal runs read "
    "entities/question_entities.json (keyed by QA index + full question sha256), which is not "
    "published. A key miss raises instead of calling the LLM extractor.",
    "Memory vectors: the formal index (embedding_indexes/<identity-sha>.npz, shared by A and B) "
    "is not published. The shipped legacy conv-*_memory_emb_{memory_only|extract_v4}_gpt35_tes_"
    "text-embedding-3-small.npz (format v2, 32-hex digests of a pre-refactor memory text) back "
    "one shared index for every arm (default memory_only; --memory-vectors extract_v4 for the "
    "sensitivity run). Upstream MemoryEmbeddingIndex.matches cannot pass (digest agreement is "
    "reported per conversation), so vectors are aligned by exact memory id instead.",
    "Memory vectors are read with a restricted unpickler (ndarray/dtype/str only) instead of "
    "MemoryEmbeddingIndex.load(allow_pickle=True).",
    "Memory-only graphs are rebuilt with upstream run.build_graph into a temporary store (no LLM "
    "needed); byte comparison against the formal records normalizes CRLF to LF because Windows "
    "text mode translates newlines.",
    "No answer generation and no official evaluator run: retrieval calls EMGraphRecall.recall "
    "directly with the formal (question, qa_index, top_k); recall_acc/analyze_aggr_acc are "
    "re-implemented and validated by replaying every formal predictions.json against stats.json.",
    "Formal-run guards not executed: assert_formal_source_clean (author's private git tree), "
    "validate_formal_result, cost telemetry and the output-directory lifecycle.",
    "Formal run_config retrieval blocks predate the semantic_score_normalization key; it is "
    "defaulted to 'none' (the pinned VARIANTS value) before the equality check.",
    "Stubbed module: POSIX-only fcntl (imported by em_graph/cache/text_embeddings.py) is replaced "
    "on Windows by a stub whose flock raises; the text-embedding cache it locks is disabled "
    "(use_text_cache=False, strict frozen query artifact), so it is never called.",
    "Network sealed: openai client constructors, socket connect/create_connection and "
    "nltk.download raise; OPENAI_API_KEY is set to ''.",
    "Platform: Windows, this venv's Python/NumPy/rank_bm25/NLTK (versions recorded in "
    "provenance), NLTK stopwords from tmp/nltk_data (list size recorded).",
]


class ProviderCallAttempt(RuntimeError):
    """Raised when anything tries to reach a provider or the network."""


_BLOCKED_ATTEMPTS: List[str] = []


def _refuse(what: str):
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        _BLOCKED_ATTEMPTS.append(what)
        raise ProviderCallAttempt(f"offline reproduction refused: {what}")

    return _raise


def seal_network(nltk_data: Path) -> None:
    """Make every provider/network path raise before upstream code is imported."""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ["NLTK_DATA"] = str(nltk_data)
    socket.socket.connect = _refuse("socket.connect")  # type: ignore[method-assign]
    socket.create_connection = _refuse("socket.create_connection")  # type: ignore[assignment]
    import openai

    for name in ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"):
        if hasattr(openai, name):
            setattr(openai, name, type(name, (), {"__init__": _refuse(f"openai.{name}()")}))
    import nltk

    nltk.download = _refuse("nltk.download")
    if str(nltk_data) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- restricted .npz reader -------------------------------------------------

_NPZ_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _NPZ_GLOBALS:
            raise pickle.UnpicklingError(f"refusing global {module}.{name} in npz")
        return super().find_class(module, name)


def safe_npz(path: Path) -> Dict[str, np.ndarray]:
    """Read an .npz whose object arrays may hold only strings, without exec risk."""
    out: Dict[str, np.ndarray] = {}
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            raw = archive.read(member)
            key = member[:-4] if member.endswith(".npy") else member
            header = io.BytesIO(raw)
            version = np.lib.format.read_magic(header)
            _shape, _fortran, dtype = readers[version](header)
            if not dtype.hasobject:
                out[key] = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
                continue
            array = _ArrayOnlyUnpickler(header).load()
            if not isinstance(array, np.ndarray) or not all(
                isinstance(item, str) for item in array.ravel().tolist()
            ):
                raise ValueError(f"{path.name}:{key} object array is not all str")
            out[key] = array
    return out


# --- upstream loading ---------------------------------------------------------


@dataclass
class Upstream:
    root: Path
    runner: Any  # experiments/.../run.py module
    em: Any  # em_graph package
    samples: List[Dict[str, Any]]
    query_artifact: Any
    stopwords_size: int
    effective_stopwords_size: int
    artifact_hashes: Dict[str, str] = field(default_factory=dict)

    def artifact(self, rel: str | Path) -> Path:
        path = self.root / rel
        rel_key = Path(rel).as_posix()
        if rel_key not in self.artifact_hashes:
            self.artifact_hashes[rel_key] = _sha256(path)
        return path


def verify_commit(root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream HEAD {head} != pinned {UPSTREAM_COMMIT}")
    return head


@lru_cache(maxsize=None)
def upstream(root: Path = DEFAULT_UPSTREAM, nltk_data: Path = DEFAULT_NLTK_DATA) -> Upstream:
    """Seal the network, then import the pinned upstream code from the clone."""
    root = root.resolve()
    verify_commit(root)
    seal_network(nltk_data)
    if "fcntl" not in sys.modules and os.name == "nt":
        # POSIX-only import in em_graph/cache/text_embeddings.py; only the
        # text-embedding cache flush locks with it, and that cache is disabled.
        stub = type(sys)("fcntl")
        stub.LOCK_EX, stub.LOCK_UN = 2, 8
        stub.flock = _refuse("fcntl.flock (text-embedding cache write)")
        sys.modules["fcntl"] = stub
    sys.path.insert(0, str(root / "code"))
    spec = importlib.util.spec_from_file_location("emg_upstream_run", root / EXP_REL / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)  # type: ignore[union-attr]
    import em_graph
    from em_graph.recall import tokenize

    stops = tokenize._stopwords()
    if stops == set(tokenize._FALLBACK_STOPWORDS):
        raise RuntimeError("NLTK stopwords unavailable; upstream fell back silently")
    data_path = root / "data" / "locomo10.json"
    if _sha256(data_path) != DATASET_SHA256:
        raise RuntimeError("locomo10.json sha256 mismatch")
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    query_path = root / QUERY_ARTIFACT_REL
    if _sha256(query_path) != QUERY_ARTIFACT_SHA256:
        raise RuntimeError("query embedding artifact sha256 differs from the formal record")
    artifact = em_graph.QueryEmbeddingArtifact.load(query_path)
    artifact.validate_exact_dataset(
        samples,
        dataset_sha256=DATASET_SHA256,
        model_name=EMBEDDING_MODEL,
        role="context",
    )
    up = Upstream(
        root=root,
        runner=runner,
        em=em_graph,
        samples=samples,
        query_artifact=artifact,
        stopwords_size=len(stops),
        effective_stopwords_size=sum(1 for w in stops if re.fullmatch(r"[a-z0-9]+", w)),
    )
    up.artifact_hashes["data/locomo10.json"] = DATASET_SHA256
    up.artifact_hashes[QUERY_ARTIFACT_REL.as_posix()] = QUERY_ARTIFACT_SHA256
    return up


class RefusingExtractor:
    """Entity extractor stand-in: question keys must come from the shipped cache."""

    def extract(self, text: str) -> Any:
        raise ProviderCallAttempt(f"LLM entity extraction requested for {text[:60]!r}")


class ShippedQuestionKeys:
    """QuestionEntityCache stand-in serving shipped legacy keys (1-based QA index)."""

    def __init__(self, sample_id: str, keys: Mapping[int, Set[str]]):
        self.sample_id = sample_id
        self.keys = dict(keys)  # 0-based QA index

    def get(self, sample_id: str, qa_index: int, question: str) -> Set[str]:
        if sample_id != self.sample_id or qa_index not in self.keys:
            raise LookupError(f"no shipped question keys for {sample_id} qa{qa_index}")
        return set(self.keys[qa_index])

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProviderCallAttempt("question key cache write requested")


@dataclass
class ConvInputs:
    """Everything one LoCoMo conversation needs, from shipped/rebuilt artifacts."""

    sample: Dict[str, Any]
    memory_graph: Any  # upstream EMGraph rebuilt by run.build_graph (formal A input)
    entity_graph: Any  # shipped LLM EMGraph (extract_v4, gpt35_tes)
    index: Any  # upstream MemoryEmbeddingIndex over the shipped memory vectors
    qkeys: Dict[int, Set[str]]  # 0-based QA index -> shipped question entity keys
    query_vectors: np.ndarray  # (n_qa, d) frozen query vectors in QA order
    audit: Dict[str, Any]


def _search_digests(graph: Any) -> Tuple[List[str], List[str]]:
    """Upstream index identity for ``graph``: sorted memory ids + text digests."""
    from em_graph.cache.text_embeddings import text_digest
    from em_graph.recall.tokenize import memory_search_text

    memories = sorted(graph.memories.values(), key=lambda m: m.id)
    return [m.id for m in memories], [
        text_digest(memory_search_text(m) or m.dia_id) for m in memories
    ]


def _load_index(up: Upstream, rel: str, graphs: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Wrap shipped memory vectors in the upstream index, aligned by memory id.

    The shipped vectors predate the refactor and embed a legacy memory text, so
    upstream's digest identity check cannot pass; ids must still match exactly
    and the digest agreement is reported, never silently assumed.
    """
    data = safe_npz(up.artifact(rel))
    index = up.em.MemoryEmbeddingIndex(
        memory_ids=[str(x) for x in data["memory_ids"].tolist()],
        vectors=np.asarray(data["vectors"], dtype=np.float32),
        model_name=str(data["model_name"]),
        text_digests=[str(x) for x in data["text_digests"].tolist()],
        normalization=str(data.get("normalization", L2_NORMALIZATION)),
    )
    if index.model_name != EMBEDDING_MODEL or index.normalization != L2_NORMALIZATION:
        raise ValueError(f"{rel}: unexpected model/normalization")
    if not np.all(np.isfinite(index.vectors)):
        raise ValueError(f"{rel}: non-finite vectors")
    audit: Dict[str, Any] = {
        "file": rel,
        "format_version": str(data["format_version"]),
        "shape": list(index.vectors.shape),
    }
    for name, graph in graphs.items():
        ids, digests = _search_digests(graph)
        if ids != index.memory_ids:
            raise ValueError(f"{rel}: memory ids differ from the {name} graph")
        audit[f"digest_matches_{name}"] = sum(
            a == b for a, b in zip(digests, index.text_digests)
        )
        audit[f"upstream_identity_{name}"] = index.matches(
            model_name=EMBEDDING_MODEL,
            normalization=L2_NORMALIZATION,
            memory_ids=ids,
            text_digests=digests,
        )
    index._query_cache = up.query_artifact
    index.strict_query_cache = True
    index.use_text_cache = False
    return index, audit


def _graph_text_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@lru_cache(maxsize=None)
def load_inputs(conv: str, vectors: str = "memory_only") -> ConvInputs:
    """Load/rebuild every retrieval input for one conversation (no network).

    ``vectors`` picks the shipped memory-vector file (``memory_only`` or
    ``extract_v4``); the formal runs shared one index across A and B.
    """
    if vectors not in VECTOR_KINDS:
        raise ValueError(f"vectors must be one of {VECTOR_KINDS}")
    up = upstream()
    runner = up.runner
    sample = next(s for s in up.samples if s["sample_id"] == conv)
    audit: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        store = up.em.EMGraphArtifactStore(Path(tmp))
        path = runner.build_graph(
            sample,
            extract_model=EXTRACT_MODEL,
            store=store,
            memory_only=True,
            graph_profile=runner.VARIANTS["A"]["graph"],
        )
        audit["memory_graph_rebuilt"] = {"file": path.name, "sha256_lf": _graph_text_sha(path)}
        memory_graph = up.em.EMGraph.load_from_file(str(path))
    entity_rel = f"outputs/em_graph/{conv}_em_graph_extract_v4_{STACK_TAG}.json"
    entity_graph = up.em.EMGraph.load_from_file(str(up.artifact(entity_rel)))
    up.em.assert_bipartite(entity_graph)
    if entity_graph.sample_id != conv:
        raise ValueError(f"{entity_rel} holds {entity_graph.sample_id}")
    index, audit["vectors"] = _load_index(
        up,
        f"outputs/em_graph/{conv}_memory_emb_{vectors}_{STACK_TAG}_{EMBEDDING_MODEL}.npz",
        {"rebuilt_memory_graph": memory_graph, "shipped_entity_graph": entity_graph},
    )
    raw_keys = json.loads(
        up.artifact(f"outputs/em_graph/{conv}_qkeys_{STACK_TAG}.json").read_text(encoding="utf-8")
    )
    qa = sample["qa"]
    if sorted(int(k) for k in raw_keys) != list(range(1, len(qa) + 1)):
        raise ValueError(f"{conv} qkeys do not cover QA indices 1..{len(qa)}")
    qkeys = {int(k) - 1: {str(v) for v in values} for k, values in raw_keys.items()}
    query_vectors = np.stack(
        [up.query_artifact.get(str(row["question"]), EMBEDDING_MODEL, "context") for row in qa]
    )
    audit["memory_count"] = len(memory_graph.memories)
    audit["memory_text_normalized_equal_to_shipped_graph"] = sum(
        1
        for mid, mem in memory_graph.memories.items()
        if mem.text_normalized == entity_graph.memories[mid].text_normalized
    )
    audit["entity_graph_counts"] = {
        "memories": len(entity_graph.memories),
        "entities": len(entity_graph.entities),
        "mention_edges": len(entity_graph.edges),
        "memory_edges": len(entity_graph.memory_edges),
    }
    audit["qkeys_empty_rows"] = sum(1 for keys in qkeys.values() if not keys)
    return ConvInputs(
        sample=sample,
        memory_graph=memory_graph,
        entity_graph=entity_graph,
        index=index,
        qkeys=qkeys,
        query_vectors=query_vectors,
        audit=audit,
    )


# --- official metric ----------------------------------------------------------


def row_recall(evidence: Sequence[str], context_ids: Sequence[str]) -> Optional[float]:
    """Official per-row recall_acc (evaluation.py:228-235), rounded as serialized."""
    if not evidence:
        return None  # official row value is 1, but analyze_aggr_acc never sums it
    if context_ids and str(context_ids[0]).startswith("S"):
        raise ValueError("session-level contexts are not produced by dialog RAG")
    hits = sum(1 for ev in evidence if ev in context_ids)
    return round(float(hits) / len(evidence), 3)


def official_recall(rows: Sequence[Mapping[str, Any]], k: int) -> Dict[str, Any]:
    """Official LoCoMo recall over rows with ``evidence``, ``category``, ``context_ids``.

    Matches analyze_aggr_acc(rag=True): empty-evidence rows add 0 but count in
    every denominator, so overall = sum / len(rows) (1,986 on LoCoMo-10).
    """
    sums: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)
    per_row: List[Optional[float]] = []
    for row in rows:
        category = int(row["category"])
        counts[category] += 1
        value = row_recall(list(row["evidence"]), list(row["context_ids"])[:k])
        per_row.append(value)
        if value is not None:
            sums[category] += value
    nonempty = [v for v in per_row if v is not None]
    total = sum(counts.values())
    cat14 = sum(counts[c] for c in (1, 2, 3, 4))
    return {
        "k": int(k),
        "overall": 100.0 * sum(sums.values()) / total if total else 0.0,
        "by_category": {
            f"{c}:{CATEGORY_NAMES[c]}": 100.0 * sums[c] / counts[c] for c in sorted(counts)
        },
        "cat1_4": 100.0 * sum(sums[c] for c in (1, 2, 3, 4)) / cat14 if cat14 else 0.0,
        "nonempty_mean": 100.0 * sum(nonempty) / len(nonempty) if nonempty else 0.0,
        "denominators": {"all": total, "nonempty_evidence": len(nonempty), "cat1_4": cat14},
        "per_row": per_row,
    }


def _stats_overall(stats: Mapping[str, Any], model_key: str) -> Dict[str, float]:
    block = stats[model_key]
    counts = {int(c): int(n) for c, n in block["category_counts"].items()}
    ratios = {int(c): float(v) for c, v in block["recall_by_category"].items()}
    total = sum(counts.values())
    cat14 = sum(counts[c] for c in (1, 2, 3, 4))
    return {
        "overall": 100.0 * sum(ratios[c] * counts[c] for c in ratios) / total,
        "cat1_4": 100.0 * sum(ratios[c] * counts[c] for c in (1, 2, 3, 4)) / cat14,
        **{f"{c}:{CATEGORY_NAMES[c]}": 100.0 * ratios[c] for c in sorted(ratios)},
    }


def load_formal(up: Upstream, run: str) -> Dict[str, Any]:
    """Load one formal run and replay its contexts through ``official_recall``."""
    base = up.root / "outputs" / "locomo_formal" / run
    config = json.loads(up.artifact(base.relative_to(up.root) / "run_config.json").read_text("utf-8"))
    stats = json.loads(up.artifact(base.relative_to(up.root) / "stats.json").read_text("utf-8"))
    preds = json.loads(
        up.artifact(base.relative_to(up.root) / "predictions.json").read_text("utf-8")
    )
    model_key = config["model_key"]
    context_key = f"{config['prediction_key']}_context"
    rows: List[Dict[str, Any]] = []
    serialized: List[float] = []
    for sample, out in zip(up.samples, preds):
        if out["sample_id"] != sample["sample_id"] or len(out["qa"]) != len(sample["qa"]):
            raise ValueError(f"{run}: prediction rows misaligned with the dataset")
        for index, (gold, pred) in enumerate(zip(sample["qa"], out["qa"])):
            if gold["question"] != pred["question"]:
                raise ValueError(f"{run}: question mismatch at {sample['sample_id']} qa{index}")
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "qa_index": index,
                    "category": gold["category"],
                    "evidence": gold["evidence"],
                    "context_ids": list(pred[context_key]),
                }
            )
            serialized.append(float(pred[f"{model_key}_recall"]))
    k = int(config["retrieval"]["top_k"])
    replay = official_recall(rows, k)
    expected = _stats_overall(stats, model_key)
    ours = {"overall": replay["overall"], "cat1_4": replay["cat1_4"], **replay["by_category"]}
    diff = max(abs(ours[key] - expected[key]) for key in expected) / 100.0
    row_mismatch = sum(
        1
        for value, rec, row in zip(replay["per_row"], serialized, rows)
        if value is not None and value != rec
    )
    too_long = sum(1 for row in rows if len(row["context_ids"]) != k)
    if diff > METRIC_TOLERANCE or row_mismatch:
        raise RuntimeError(f"{run}: metric replay failed (diff={diff}, rows={row_mismatch})")
    return {
        "config": config,
        "rows": rows,
        "stats": expected,
        "metric_check": {
            "max_abs_diff_vs_stats": diff,
            "row_recall_mismatches": row_mismatch,
            "context_length_not_k": too_long,
            "pass": True,
        },
    }


# --- reproduction --------------------------------------------------------------


def _recall_parameters(up: Upstream, config: Mapping[str, Any]) -> Dict[str, Any]:
    block = dict(config["retrieval"])
    variant = block.pop("variant")
    for key in ("rag_mode", "top_k"):
        block.pop(key)
    block.setdefault("semantic_score_normalization", "none")
    expected = dict(up.runner.VARIANTS[variant]["recall"])
    if block != expected:
        raise ValueError(f"{variant}: run_config retrieval block != pinned VARIANTS: {block}")
    if config["models"]["embedding"] != EMBEDDING_MODEL:
        raise ValueError("formal run used a different embedding model")
    if config["graph_profile"] != up.runner.VARIANTS[variant]["graph"]:
        raise ValueError(f"{variant}: graph profile differs from pinned VARIANTS")
    return block


def _recaller(up: Upstream, inputs: ConvInputs, variant: str, params: Mapping[str, Any]) -> Any:
    memory_only = bool(up.runner.VARIANTS[variant]["graph"]["memory_only"])
    graph = inputs.memory_graph if memory_only else inputs.entity_graph
    full_pool = bool(params["force_full_pool"])
    return up.em.EMGraphRecall(
        graph,
        inputs.index,
        entity_bm25_index=None if full_pool else up.em.EntityBM25Index.build(graph),
        extractor=None if full_pool else RefusingExtractor(),
        question_cache=(
            None if full_pool else ShippedQuestionKeys(inputs.sample["sample_id"], inputs.qkeys)
        ),
        **params,
    )


def run_arm(
    up: Upstream,
    variant: str,
    k: int,
    config: Mapping[str, Any],
    convs: Sequence[str],
    vectors: str,
) -> List[Dict[str, Any]]:
    """Run upstream EMGraphRecall exactly as formal_graph.py wires it."""
    params = _recall_parameters(up, config)
    rows: List[Dict[str, Any]] = []
    for conv in convs:
        inputs = load_inputs(conv, vectors)
        recall = _recaller(up, inputs, variant, params)
        for index, qa in enumerate(inputs.sample["qa"]):
            result = recall.recall(inputs.sample, index, str(qa["question"]), k)
            rows.append(
                {
                    "sample_id": conv,
                    "qa_index": index,
                    "category": qa["category"],
                    "evidence": qa["evidence"],
                    "context_ids": list(result.context_ids),
                }
            )
    return rows


def _identity(ours: Sequence[Mapping[str, Any]], theirs: Sequence[Optional[Sequence[str]]], k: int) -> Dict[str, int]:
    compared = ordered = same_set = 0
    for row, other in zip(ours, theirs):
        if other is None:
            continue
        compared += 1
        mine = list(row["context_ids"])[:k]
        ref = list(other)[:k]
        ordered += int(mine == ref)
        same_set += int(set(mine) == set(ref))
    return {"compared": compared, "ordered_identical": ordered, "set_identical": same_set}


@lru_cache(maxsize=None)
def _legacy(file_name: str) -> Dict[Tuple[str, int], List[str]]:
    up = upstream()
    data = json.loads(up.artifact(f"outputs/em_graph/{file_name}").read_text(encoding="utf-8"))
    return {
        (str(entry["sample_id"]), int(entry["qa_index"]) - 1): list(entry["retrieved50"])
        for entry in data.values()
    }


def _verdict(results: Mapping[str, Mapping[str, Any]], full_scope: bool) -> Dict[str, Any]:
    if not full_scope:
        return {"status": "not_evaluated_subset_scope"}
    cells = [cell for arm in results.values() for cell in arm.values()]
    deltas = [abs(c["delta_vs_published"]) for c in cells]
    exact = all(
        c["identity_vs_formal"]["ordered_identical"] == c["identity_vs_formal"]["compared"]
        for c in cells
    )
    numeric = max(deltas) <= PASS_TOLERANCE_POINTS

    def r(arm: str, k: int) -> float:
        return results[arm][str(k)]["ours"]["overall"]

    b_a = {k: r("B", k) - r("A", k) for k in (5, 10, 25, 50)}
    ordering = (
        r("B", 25) > r("B_noseq", 25) > max(r("A", 25), r("B_gate", 25), r("B_gate_seq", 25))
        > r("B_entity", 25)
    )
    qualitative = (
        b_a[25] > 0
        and abs(b_a[25] - QUALITATIVE_B_MINUS_A_25) <= QUALITATIVE_WINDOW_POINTS
        and all(b_a[k] > 0 for k in (5, 10, 50))
        and ordering
    )
    if exact and numeric:
        status = "exact_reproduction"
    elif numeric:
        status = "numeric_reproduction"
    elif qualitative:
        status = "qualitative_reproduction"
    else:
        status = "not_reproduced"
    return {
        "status": status,
        "max_abs_delta_vs_published_points": max(deltas),
        "exact_ranking_identity_all_cells": exact,
        "b_minus_a": b_a,
        "k25_ordering_holds": ordering,
        "qualitative_criteria_hold": qualitative,
    }


def _versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("rank_bm25", "nltk", "openai"):
        try:
            out[name] = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def reproduce(
    arms: Sequence[str],
    convs: Optional[Sequence[str]],
    out: Path,
    vectors: str = "memory_only",
) -> Dict[str, Any]:
    started = time.perf_counter()
    up = upstream()
    all_convs = [s["sample_id"] for s in up.samples]
    scope = list(convs) if convs else all_convs
    full_scope = scope == all_convs
    cells = sorted(key for key in FORMAL_RUNS if key[0] in arms)
    results: Dict[str, Dict[str, Any]] = defaultdict(dict)
    metric_checks: Dict[str, Any] = {}
    for variant, k in cells:
        run = FORMAL_RUNS[(variant, k)]
        formal = load_formal(up, run)
        metric_checks[run] = formal["metric_check"]
        formal_rows = [row for row in formal["rows"] if row["sample_id"] in scope]
        t0 = time.perf_counter()
        rows = run_arm(up, variant, k, formal["config"], scope, vectors)
        elapsed = time.perf_counter() - t0
        ours = official_recall(rows, k)
        theirs = official_recall(formal_rows, k)
        legacy_file = LEGACY_CHECKPOINTS.get(variant)
        legacy = _legacy(legacy_file) if legacy_file else {}
        cell = {
            "formal_run": run,
            "published": PUBLISHED[(variant, k)],
            "formal": {key: v for key, v in theirs.items() if key != "per_row"},
            "ours": {key: v for key, v in ours.items() if key != "per_row"},
            "delta_vs_published": ours["overall"] - PUBLISHED[(variant, k)],
            "delta_vs_formal": ours["overall"] - theirs["overall"],
            "delta_cat1_4_vs_formal": ours["cat1_4"] - theirs["cat1_4"],
            "row_recall_equal_vs_formal": sum(
                1 for a, b in zip(ours["per_row"], theirs["per_row"]) if a == b
            ),
            "identity_vs_formal": _identity(rows, [r["context_ids"] for r in formal_rows], k),
            "identity_vs_legacy_checkpoint": (
                {
                    "file": legacy_file,
                    **_identity(
                        rows, [legacy.get((r["sample_id"], r["qa_index"])) for r in rows], k
                    ),
                }
                if legacy_file
                else None
            ),
            "seconds": round(elapsed, 2),
        }
        results[variant][str(k)] = cell
        print(
            f"{variant:>10} k={k:<2} ours={ours['overall']:.4f} formal={theirs['overall']:.4f} "
            f"published={PUBLISHED[(variant, k)]:.4f} identical="
            f"{cell['identity_vs_formal']['ordered_identical']}/{cell['identity_vs_formal']['compared']} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
    formal_a = json.loads(
        up.artifact(Path("outputs/locomo_formal") / FORMAL_RUNS[("A", 25)] / "run_config.json").read_text("utf-8")
    )
    formal_b = json.loads(
        up.artifact(Path("outputs/locomo_formal") / FORMAL_RUNS[("B", 25)] / "run_config.json").read_text("utf-8")
    )
    records_a = {r["sample_id"]: r for r in formal_a["cache_identity"]["records"]}
    records_b = {r["sample_id"]: r for r in formal_b["cache_identity"]["records"]}
    input_audit: Dict[str, Any] = {}
    for conv in scope:
        inputs = load_inputs(conv, vectors)
        rebuilt = inputs.audit["memory_graph_rebuilt"]
        shipped_b = up.root / f"outputs/em_graph/{conv}_em_graph_extract_v4_{STACK_TAG}.json"
        input_audit[conv] = {
            **inputs.audit,
            "memory_graph_matches_formal_A": {
                "identity_file": rebuilt["file"] == Path(records_a[conv]["graph"]["path"]).name,
                "sha256": rebuilt["sha256_lf"] == records_a[conv]["graph"]["sha256"],
            },
            "shipped_entity_graph_matches_formal_B_sha256": _graph_text_sha(shipped_b)
            == records_b[conv]["graph"]["sha256"],
            "formal_A_and_B_share_embedding_index": records_a[conv]["embedding_index"]["sha256"]
            == records_b[conv]["embedding_index"]["sha256"],
            "qkeys_rows": len(inputs.qkeys),
        }
    payload = {
        "schema": "hybridmind_emg_reproduction_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "protocol": "research/experiments/r1-emg-reproduction/protocol.md",
        "scope": {
            "conversations": scope,
            "full_locomo10": full_scope,
            "arms": list(arms),
            "memory_vectors": vectors,
        },
        "provenance": {
            "upstream_repo": UPSTREAM_REPO,
            "upstream_commit": verify_commit(up.root),
            "upstream_license": UPSTREAM_LICENSE,
            "upstream_clone": _display(up.root),
            "formal_run_source_commit": formal_b["source_commit"],
            "dataset_sha256": DATASET_SHA256,
            "query_artifact_sha256": QUERY_ARTIFACT_SHA256,
            "artifact_sha256": dict(sorted(up.artifact_hashes.items())),
            "versions": _versions(),
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "nltk_data": _display(Path(os.environ["NLTK_DATA"])),
            "nltk_stopwords_english_size": up.stopwords_size,
            "nltk_stopwords_effective_for_tokenizer": up.effective_stopwords_size,
            "provider_calls": len(_BLOCKED_ATTEMPTS),
            "blocked_network_attempts": list(_BLOCKED_ATTEMPTS),
            "query_artifact_usage": {
                "hits": up.query_artifact.hits,
                "misses": up.query_artifact.misses,
            },
        },
        "deviations": DEVIATIONS,
        "metric_check": metric_checks,
        "published_cat1_4_k25": PUBLISHED_CAT1_4_K25,
        "input_audit": input_audit,
        "results": results,
        "verdict": _verdict(results, full_scope and set(arms) >= set(PUBLISHED_ARMS)),
        "seconds": round(time.perf_counter() - started, 1),
    }
    if payload["provenance"]["provider_calls"]:
        raise RuntimeError("network/provider access was attempted")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return payload


# --- HybridMind port parity and variants (--port) ------------------------------

NODE_ID_FORMAT = "locomo:{sample_id}:{dia_id}"
PORT_CELLS: Tuple[Tuple[str, int], ...] = (
    ("A", 5), ("A", 10), ("A", 25), ("A", 50),
    ("B", 5), ("B", 10), ("B", 25), ("B", 50),
    ("B_entity", 25), ("B_noseq", 25),
)
VARIANT_KS = (5, 10, 25, 50)
BOOTSTRAP_RESAMPLES = 2000
FLOAT_NEAR_TIE = 1e-6  # score gap below which a swap is float rounding, not logic
PORT_CONFIGS = {
    "reference_dense_upstream_order": (
        "port fusion fed upstream MemoryEmbeddingIndex.scores (same float32 BLAS "
        "arithmetic) with EMG's dia_id tie-break: isolates engine/entity_graph.py"
    ),
    "hybridmind_dense_chronological": (
        "port fusion fed engine.dense_channel.exact_scores (renormalised rows, einsum) "
        "with the corpus chronological tie-break: what HybridMind runs"
    ),
}
PORT_NOTES: List[str] = [
    "Arm A runs the port on the shipped entity graph with no question keys (upstream "
    "force_full_pool): an empty gate ranks every memory by S alone, so only the memory ids "
    "and vectors matter; load_inputs checks those ids equal the rebuilt memory-only graph's.",
    "Recall parameters come from the formal run_config retrieval blocks (checked against the "
    "pinned VARIANTS by _recall_parameters) and are passed to EntityMemoryGraph.emg_fused_rank.",
    "Upstream scores the gated pool with a float32 BLAS product over the subset rows, so a "
    "memory's S differs by ulps with gate composition and set order, i.e. with PYTHONHASHSEED "
    "(measured 2.1e-7 to 2.5e-7 max across two full runs); a "
    "mismatch whose position-wise scores and shared-id scores agree within 1e-6 is counted as "
    "float_near_tie, anything else as other.",
]
VARIANT_NOTES: List[str] = [
    "HybridMind variants on EMG reference artifacts, not reproductions: shipped LLM graphs "
    "(extract_v4, gpt35_tes), shipped question keys, cached text-embedding-3-small vectors "
    "(1536-d reference track, not the 4096-d runtime), scored by the official recall_acc.",
    "Each variant ranks once at k=50 and is scored at every k by truncation; every variant "
    "ranking is prefix-consistent (sorted gate then sorted fill, or one sorted list).",
    "Graph-only variants return positive-evidence candidates only (no fill), so a question "
    "with no matched entity retrieves nothing and scores 0; counts are reported.",
    "Lexical graph: LexicalEntityExtractor(speakers = both LoCoMo speakers) over EMG's "
    "extraction text (text_normalized plus ' [Image: caption]'), speaker added as a Who "
    "entity; question keys from extract_query. Same memories, chronology and NEXT/PREV chain.",
    "Baseline arm A is the upstream code on the same artifacts; CIs are paired "
    "conversation-cluster percentile bootstraps of the per-row recall difference (empty-"
    "evidence rows contribute 0, as in the official sum/rows metric).",
]
_RECALL_ATTRS = (
    "entity_weight", "semantic_weight", "semantic_score_normalization", "expand_sequence",
    "sequence_secondary_scale", "entity_min_rel_score", "entity_top_k_per_key",
    "who_only_dampen", "degree_discount",
)

Scored = List[Tuple[str, float]]  # (dia_id, score)


@dataclass
class PortInputs:
    """HybridMind-side view of one conversation's reference artifacts."""

    graph: EntityMemoryGraph  # shipped LLM graph via from_emg_json
    node_of: Dict[str, str]  # upstream memory id -> node id
    dense: DenseMatrix  # cached reference vectors, rows aligned with graph.corpus
    lexical: EntityMemoryGraph  # same corpus, LexicalEntityExtractor mentions
    lexical_qkeys: Dict[int, Set[str]]


def _extraction_text(metadata: Mapping[str, Any]) -> str:
    """EMG builder._dialog_extraction_text over the stored memory fields."""
    parts = [str(metadata.get("text_normalized") or "")]
    caption = str(metadata.get("blip_caption") or "").strip()
    if caption:
        parts.append(f"[Image: {caption}]")
    return " ".join(p for p in parts if p).strip()


@lru_cache(maxsize=None)
def port_inputs(conv: str, vectors: str = "memory_only") -> PortInputs:
    up = upstream()
    inputs = load_inputs(conv, vectors)
    rel = f"outputs/em_graph/{conv}_em_graph_extract_v4_{STACK_TAG}.json"
    graph = EntityMemoryGraph.from_emg_json(up.artifact(rel), node_id_format=NODE_ID_FORMAT)
    node_of = {
        mid: NODE_ID_FORMAT.format(sample_id=conv, dia_id=inputs.entity_graph.memories[mid].dia_id)
        for mid in inputs.index.memory_ids
    }
    if set(node_of.values()) != set(graph.corpus.index_of):
        raise ValueError(f"{conv}: vector ids do not cover the port graph's memories")
    dense = DenseMatrix.from_vectors(
        graph.corpus,
        {node_of[mid]: vec for mid, vec in zip(inputs.index.memory_ids, inputs.index.vectors)},
        expected_dim=None,
    )
    speakers = [inputs.sample["conversation"][f"speaker_{s}"] for s in ("a", "b")]
    extractor = LexicalEntityExtractor(speakers=speakers)
    mentions = {
        t.node_id: extractor.extract(_extraction_text(t.metadata), speaker=t.speaker)
        for t in graph.corpus.turns
    }
    lexical = EntityMemoryGraph.from_corpus(graph.corpus, mentions)
    lexical_qkeys = {
        i: extractor.extract_query(str(qa["question"])) for i, qa in enumerate(inputs.sample["qa"])
    }
    return PortInputs(graph, node_of, dense, lexical, lexical_qkeys)


def _reference_scorer(inputs: ConvInputs, port: PortInputs, question: str):
    memory_of = {node: mid for mid, node in port.node_of.items()}

    def score(ids: Optional[Set[str]]) -> Dict[str, float]:
        wanted = None if ids is None else [memory_of[n] for n in ids]
        return {port.node_of[m]: s for m, s in inputs.index.scores(question, memory_ids=wanted).items()}

    return score


def _hybridmind_scorer(port: PortInputs, query_vec: np.ndarray):
    full = exact_scores(query_vec, port.dense)
    return lambda ids: full if ids is None else {n: full[n] for n in ids}


def _port_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "entity_weight": params["entity_weight"],
        "semantic_weight": params["semantic_weight"],
        "semantic_normalization": params["semantic_score_normalization"],
        "expand_sequence": params["expand_sequence"],
        "sequence_secondary_scale": params["sequence_secondary_scale"],
        "min_rel_score": params["entity_min_rel_score"],
        "top_k_per_key": params["entity_top_k_per_key"],
        "who_only_dampen": params["who_only_dampen"],
        "degree_discount": params["degree_discount"],
    }


def _upstream_scored(up: Upstream, inputs: ConvInputs, variant: str, params: Mapping[str, Any], k: int) -> List[Scored]:
    """Upstream retrieve_dialog_ids with scores, wired exactly as EMGraphRecall.recall."""
    from em_graph.recall.retrieval import retrieve_dialog_ids

    recall = _recaller(up, inputs, variant, params)
    conv = inputs.sample["sample_id"]
    kwargs = {name: getattr(recall, name) for name in _RECALL_ATTRS}
    out = []
    for index, qa in enumerate(inputs.sample["qa"]):
        question = str(qa["question"])
        out.append(retrieve_dialog_ids(
            recall.graph, question, top_k=k, embedding_index=recall.embedding_index,
            entity_bm25_index=recall.entity_bm25_index, extractor=recall.extractor,
            q_entity_keys=recall._question_keys(conv, index, question), **kwargs,
        ))
    return out


def _port_scored(
    inputs: ConvInputs, port: PortInputs, params: Mapping[str, Any], k: int, config: str
) -> List[Scored]:
    reference = config == "reference_dense_upstream_order"
    out = []
    for index, qa in enumerate(inputs.sample["qa"]):
        keys = () if params["force_full_pool"] else inputs.qkeys[index]
        scorer = (
            _reference_scorer(inputs, port, str(qa["question"]))
            if reference
            else _hybridmind_scorer(port, inputs.query_vectors[index])
        )
        ranked = port.graph.emg_fused_rank(
            keys, scorer, top_k=k, upstream_order=reference, **_port_kwargs(params)
        )
        out.append([(port.graph.corpus.turn(n).evidence_id, s) for n, s in ranked])
    return out


def _compare(theirs: Scored, ours: Scored) -> Tuple[bool, bool, float, bool]:
    """(ordered identical, set identical, max |score diff| on shared ids, float near-tie)."""
    identical = [d for d, _ in theirs] == [d for d, _ in ours]
    t, o = dict(theirs), dict(ours)
    shared = max((abs(t[d] - o[d]) for d in t.keys() & o.keys()), default=0.0)
    positional = (
        max((abs(a[1] - b[1]) for a, b in zip(theirs, ours)), default=0.0)
        if len(theirs) == len(ours)
        else math.inf
    )
    near_tie = not identical and max(shared, positional) <= FLOAT_NEAR_TIE
    return identical, set(t) == set(o), shared, near_tie


def _rows(sample_ids: Sequence[str], convs_qa: Mapping[str, Sequence[Mapping[str, Any]]], ranked: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
    rows, it = [], iter(ranked)
    for conv in sample_ids:
        for index, qa in enumerate(convs_qa[conv]):
            rows.append({
                "sample_id": conv, "qa_index": index, "category": qa["category"],
                "evidence": qa["evidence"], "context_ids": list(next(it)),
            })
    return rows


def _headline(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: result[key] for key in ("overall", "cat1_4", "by_category")}


def _port_cell(
    up: Upstream, variant: str, k: int, scope: Sequence[str], vectors: str, reproduced: Optional[float]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    formal = load_formal(up, FORMAL_RUNS[(variant, k)])
    params = _recall_parameters(up, formal["config"])
    qa_of = {conv: load_inputs(conv, vectors).sample["qa"] for conv in scope}
    theirs: List[Scored] = []
    for conv in scope:
        theirs += _upstream_scored(up, load_inputs(conv, vectors), variant, params, k)
    up_rows = _rows(scope, qa_of, [[d for d, _ in r] for r in theirs])
    up_metric = official_recall(up_rows, k)
    formal_ctx = {(r["sample_id"], r["qa_index"]): r["context_ids"] for r in formal["rows"]}
    legacy_file = LEGACY_CHECKPOINTS.get(variant)
    legacy = _legacy(legacy_file) if legacy_file else {}
    cell: Dict[str, Any] = {
        "formal_run": FORMAL_RUNS[(variant, k)],
        "upstream": {
            **_headline(up_metric),
            "equals_reproduction_result": None if reproduced is None else up_metric["overall"] == reproduced,
        },
    }
    for config in PORT_CONFIGS:
        ours: List[Scored] = []
        for conv in scope:
            ours += _port_scored(load_inputs(conv, vectors), port_inputs(conv, vectors), params, k, config)
        compared = [_compare(t, o) for t, o in zip(theirs, ours)]
        rows = _rows(scope, qa_of, [[d for d, _ in r] for r in ours])
        metric = official_recall(rows, k)
        cell[config] = {
            "compared": len(compared),
            "ordered_identical": sum(c[0] for c in compared),
            "set_identical": sum(c[1] for c in compared),
            "mismatch_float_near_tie": sum(c[3] for c in compared),
            "mismatch_other": sum(1 for c in compared if not c[0] and not c[3]),
            "max_abs_score_diff_shared_ids": max(c[2] for c in compared),
            **_headline(metric),
            "row_recall_equal_to_upstream": sum(
                a == b for a, b in zip(metric["per_row"], up_metric["per_row"])
            ),
            "metric_equal_to_upstream": metric["per_row"] == up_metric["per_row"]
            and metric["overall"] == up_metric["overall"],
            "identity_vs_formal_predictions": _identity(
                rows, [formal_ctx[(r["sample_id"], r["qa_index"])] for r in rows], k
            ),
            "identity_vs_legacy_checkpoint": (
                {"file": legacy_file, **_identity(rows, [legacy.get((r["sample_id"], r["qa_index"])) for r in rows], k)}
                if legacy_file
                else None
            ),
        }
    return cell, up_rows


def _paired_ci(rows: Sequence[Mapping[str, Any]], arm: Sequence[Optional[float]], base: Sequence[Optional[float]], cat1_4: bool) -> Dict[str, Any]:
    pairs = [
        (row["sample_id"], 100.0 * ((a or 0.0) - (b or 0.0)))
        for row, a, b in zip(rows, arm, base)
        if not cat1_4 or int(row["category"]) in (1, 2, 3, 4)
    ]
    ci = cluster_bootstrap(pairs, DEFAULT_SEED, samples=BOOTSTRAP_RESAMPLES)
    return {"mean_diff_points": ci["mean"], "ci95": [ci["lo"], ci["hi"]], "rows": ci["n"], "clusters": ci["clusters"]}


def _cis(rows: Sequence[Mapping[str, Any]], per_row: Sequence[Optional[float]], base_rows: Sequence[Mapping[str, Any]], k: int) -> Dict[str, Any]:
    """Paired cluster CIs (overall, cat 1-4) of an arm against a baseline at ``k``."""
    if [(r["sample_id"], r["qa_index"]) for r in rows] != [(r["sample_id"], r["qa_index"]) for r in base_rows]:
        raise ValueError("paired arms are not row-aligned")
    base = official_recall(base_rows, k)["per_row"]
    return {"overall": _paired_ci(rows, per_row, base, False), "cat1_4": _paired_ci(rows, per_row, base, True)}


def _variant_rankings(scope: Sequence[str], vectors: str) -> Dict[str, List[List[str]]]:
    top = max(VARIANT_KS)
    out: Dict[str, List[List[str]]] = defaultdict(list)
    for conv in scope:
        inputs, port = load_inputs(conv, vectors), port_inputs(conv, vectors)
        dia = lambda ranked: [port.graph.corpus.turn(n).evidence_id for n, _ in ranked]  # noqa: E731
        for index in range(len(inputs.sample["qa"])):
            keys, lexical_keys = inputs.qkeys[index], port.lexical_qkeys[index]
            dense = _hybridmind_scorer(port, inputs.query_vectors[index])
            out["ppr_graph_only"].append(dia(port.graph.ppr_rank(keys, top_k=top)))
            out["ppr_dense_passage_seeds"].append(
                dia(port.graph.ppr_rank(keys, passage_scores=dense(None), top_k=top))
            )
            out["lexical_graph_entity_only"].append(dia(port.lexical.emg_rank(lexical_keys, top_k=top)))
            out["lexical_graph_fused"].append(dia(port.lexical.emg_fused_rank(lexical_keys, dense, top_k=top)))
    return out


VARIANT_DESCRIPTIONS = {
    "ppr_graph_only": "HippoRAG-2 PPR over the shipped LLM graph: seeds = qkey match strength / "
    "|memories(e)| for the top-5 entities, damping 0.5, NEXT/PREV edges on, positive mass only",
    "ppr_dense_passage_seeds": "ppr_graph_only plus every memory seeded with min-max dense "
    "score x 0.05 (HippoRAG 2 passage_node_weight)",
    "lexical_graph_entity_only": "graph from LexicalEntityExtractor, question keys from "
    "extract_query; EMG E ranking alone (B_entity weights), positive candidates only",
    "lexical_graph_fused": "lexical graph and keys; EMG 0.30 E + 0.70 S gate with dense fill "
    "(B weights, expand_sequence on)",
}


def _key_stats(keys: Sequence[Set[str]]) -> Dict[str, Any]:
    return {
        "questions": len(keys),
        "mean_keys": sum(len(k) for k in keys) / len(keys) if keys else 0.0,
        "empty": sum(1 for k in keys if not k),
    }


def _graph_totals(graphs: Sequence[EntityMemoryGraph]) -> Dict[str, int]:
    stats = [g.stats() for g in graphs]
    return {key: sum(s[key] for s in stats) for key in ("memories", "entities", "mention_edges", "sequence_pairs")}


def port(convs: Optional[Sequence[str]], out: Path, vectors: str = "memory_only") -> Dict[str, Any]:
    """Port parity cells plus HybridMind variants; adds ``port``/``variants`` to ``out``."""
    started = time.perf_counter()
    up = upstream()
    all_convs = [s["sample_id"] for s in up.samples]
    scope = list(convs) if convs else all_convs
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    same_run = payload.get("scope", {}).get("conversations") == scope and payload.get("scope", {}).get(
        "memory_vectors"
    ) == vectors
    cells: Dict[str, Dict[str, Any]] = defaultdict(dict)
    baseline_rows: Dict[int, List[Dict[str, Any]]] = {}
    reference_rows: Dict[str, List[Dict[str, Any]]] = {}
    for variant, k in PORT_CELLS:
        t0 = time.perf_counter()
        reproduced = payload["results"][variant][str(k)]["ours"]["overall"] if same_run else None
        cell, up_rows = _port_cell(up, variant, k, scope, vectors, reproduced)
        cell["seconds"] = round(time.perf_counter() - t0, 2)
        cells[variant][str(k)] = cell
        if variant == "A":
            baseline_rows[k] = up_rows
        else:
            reference_rows[f"{variant}_upstream@{k}"] = up_rows
        summary = " ".join(
            f"{c.split('_')[0]}={cell[c]['ordered_identical']}/{cell[c]['compared']}"
            f"(tie={cell[c]['mismatch_float_near_tie']},other={cell[c]['mismatch_other']})"
            for c in PORT_CONFIGS
        )
        print(f"port {variant:>8} k={k:<2} upstream={cell['upstream']['overall']:.4f} {summary} ({cell['seconds']}s)", flush=True)

    t0 = time.perf_counter()
    rankings = _variant_rankings(scope, vectors)
    qa_of = {conv: load_inputs(conv, vectors).sample["qa"] for conv in scope}
    arms: Dict[str, Any] = {}
    for name, ranked in rankings.items():
        rows = _rows(scope, qa_of, ranked)
        arm: Dict[str, Any] = {
            "description": VARIANT_DESCRIPTIONS[name],
            "empty_rankings": sum(1 for r in ranked if not r),
            "rankings_shorter_than_50": sum(1 for r in ranked if len(r) < max(VARIANT_KS)),
        }
        for k in VARIANT_KS:
            metric = official_recall(rows, k)
            cell = arm[str(k)] = {
                **_headline(metric),
                "mean_context_len": sum(min(len(r), k) for r in ranked) / len(ranked),
                "ci_vs_A": _cis(rows, metric["per_row"], baseline_rows[k], k),
                "ci_vs_B": _cis(rows, metric["per_row"], reference_rows[f"B_upstream@{k}"], k),
            }
            print(f"variant {name:>26} k={k:<2} overall={metric['overall']:.4f} "
                  f"vs_A={cell['ci_vs_A']['overall']['mean_diff_points']:+.2f} "
                  f"vs_B={cell['ci_vs_B']['overall']['mean_diff_points']:+.2f}", flush=True)
        arms[name] = arm
    references: Dict[str, Any] = {}
    for label, rows in reference_rows.items():
        k = int(label.rsplit("@", 1)[1])
        metric = official_recall(rows, k)
        references[label] = {**_headline(metric), "ci_vs_A": _cis(rows, metric["per_row"], baseline_rows[k], k)}
    ports = [port_inputs(conv, vectors) for conv in scope]
    variants = {
        "label": "HybridMind variants on EMG reference artifacts (not reproductions)",
        "notes": VARIANT_NOTES,
        "baseline": "A (upstream EMG code, memory-only graph, same cached vectors); ci_vs_B "
        "compares against upstream B (LLM graph, 0.30/0.70) at the same k",
        "baseline_recall": {str(k): _headline(official_recall(baseline_rows[k], k)) for k in VARIANT_KS},
        "bootstrap": {
            "method": "paired conversation-cluster percentile bootstrap "
            "(scripts.offline_budgeted_evidence.cluster_bootstrap)",
            "seed": DEFAULT_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "official recall points; per-row difference, empty-evidence rows = 0",
        },
        "arms": arms,
        "upstream_references_vs_A": references,
        "graphs": {
            "llm_extract_v4": _graph_totals([p.graph for p in ports]),
            "lexical_v1": _graph_totals([p.lexical for p in ports]),
        },
        "question_keys": {
            "shipped_llm": _key_stats([k for c in scope for k in load_inputs(c, vectors).qkeys.values()]),
            "lexical_extract_query": _key_stats([k for p in ports for k in p.lexical_qkeys.values()]),
            "lexical_equal_to_shipped": sum(
                load_inputs(c, vectors).qkeys[i] == p.lexical_qkeys[i]
                for c, p in zip(scope, ports)
                for i in p.lexical_qkeys
            ),
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    configs = {
        c: {
            "cells": sum(1 for v in cells.values() for cell in v.values()),
            "all_ordered_identical": all(
                cell[c]["ordered_identical"] == cell[c]["compared"] for v in cells.values() for cell in v.values()
            ),
            "mismatch_float_near_tie": sum(cell[c]["mismatch_float_near_tie"] for v in cells.values() for cell in v.values()),
            "mismatch_other": sum(cell[c]["mismatch_other"] for v in cells.values() for cell in v.values()),
            "max_abs_score_diff": max(cell[c]["max_abs_score_diff_shared_ids"] for v in cells.values() for cell in v.values()),
            "all_metric_equal": all(cell[c]["metric_equal_to_upstream"] for v in cells.values() for cell in v.values()),
        }
        for c in PORT_CONFIGS
    }
    payload["port"] = {
        "label": "HybridMind engine/entity_graph.py port vs upstream EMG code on identical reference artifacts",
        "scope": {"conversations": scope, "full_locomo10": scope == all_convs, "memory_vectors": vectors},
        "node_id_format": NODE_ID_FORMAT,
        "configs": PORT_CONFIGS,
        "notes": PORT_NOTES,
        "entity_graph_deviations": GRAPH_DEVIATIONS,
        "float_near_tie_tolerance": FLOAT_NEAR_TIE,
        "summary": configs,
        "cells": cells,
    }
    payload["variants"] = variants
    payload["port_provenance"] = {
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": verify_commit(up.root),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "provider_calls": len(_BLOCKED_ATTEMPTS),
        "blocked_network_attempts": list(_BLOCKED_ATTEMPTS),
        "query_artifact_usage": {"hits": up.query_artifact.hits, "misses": up.query_artifact.misses},
        "seconds": round(time.perf_counter() - started, 1),
    }
    if _BLOCKED_ATTEMPTS:
        raise RuntimeError("network/provider access was attempted")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return payload


PUBLISHED_ARMS = sorted({arm for arm, _k in PUBLISHED})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="*", default=PUBLISHED_ARMS, choices=PUBLISHED_ARMS)
    parser.add_argument("--samples", nargs="*", help="conversation subset (smoke runs only)")
    parser.add_argument(
        "--memory-vectors",
        choices=VECTOR_KINDS,
        default="memory_only",
        help="which shipped memory-vector file backs the (shared) dense index",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--port",
        action="store_true",
        help="run HybridMind's port + variants and add 'port'/'variants' to --out "
        "(created if missing); --arms is ignored",
    )
    args = parser.parse_args(argv)
    if args.port:
        payload = port(args.samples, args.out, args.memory_vectors)
        print(json.dumps(payload["port"]["summary"], indent=2))
        print(f"wrote {_display(args.out)}")
        return 0
    payload = reproduce(args.arms, args.samples, args.out, args.memory_vectors)
    print(json.dumps(payload["verdict"], indent=2))
    print(f"wrote {_display(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
