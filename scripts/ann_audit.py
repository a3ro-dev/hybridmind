"""Offline HNSW-vs-exact recall audit over stored vectors. Zero provider calls.

Vectors come from ``.npz`` files (one scope per file; key names configurable)
or from a HybridMind SQLite store (read-only; 4096-d contract enforced by the
store's deserializer). Queries are either leave-one-out stored vectors or rows
of a query ``.npz``, optionally selected per scope through an EMG-style
QA -> question-digest indirection. ``.npz`` files are opened with
``allow_pickle=False``: object arrays (e.g. EMG ``memory_ids``) are never
unpickled, so documents are identified by row index.

Example (EMG reference vectors, per conversation):
  python scripts/ann_audit.py \\
    --docs-npz tmp/upstream/em_graph_memory/outputs/em_graph/conv-*_memory_emb_extract_v4_gpt35_tes_text-embedding-3-small.npz \\
    --query-npz tmp/upstream/em_graph_memory/outputs/em_graph/query_embeddings/locomo10_047d8e25_text-embedding-3-small_v1.npz \\
    --query-scope-key qa_sample_ids --query-link-key qa_question_digests --query-row-key question_digests \\
    --scope-regex "conv-\\d+" --expected-dim 1536 --label "..." --output experiments/results/ann-audit-....json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.corpus import ScopedCorpus, TurnRecord  # noqa: E402
from engine.dense_channel import DenseMatrix, ann_audit  # noqa: E402

MEMORY_CAP_BYTES = 512 * 1024 * 1024  # weak-laptop guard on the vector working set
DEFAULT_CONFIGS = ("32:16,32,64,128,256", "16:64", "48:64")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_config(text: str) -> Tuple[int, List[int]]:
    """``"32:16,64"`` -> (M=32, efSearch grid [16, 64])."""
    try:
        m, grid = text.split(":", 1)
        return int(m), [int(ef) for ef in grid.split(",") if ef]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"config must look like M:ef1,ef2 (got {text!r})") from exc


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path.name} has no key {key!r}; keys: {sorted(data.files)}")
        return np.asarray(data[key])


def load_store_vectors(db_path: Path) -> Tuple[List[str], np.ndarray]:
    """Retrievable node vectors from a HybridMind store, opened read-only."""
    from storage.sqlite_store import SQLiteStore

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL "
            "AND deleted_at IS NULL AND archived_at IS NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError(f"{db_path} has no retrievable embeddings")
    # _deserialize_embedding enforces the exact 4096-d finite contract.
    return [str(r[0]) for r in rows], np.vstack([SQLiteStore._deserialize_embedding(r[1]) for r in rows])


def select_queries(
    vectors: np.ndarray,
    scope_labels: Optional[np.ndarray],
    scope: str,
    link: Optional[np.ndarray],
    row_keys: Optional[np.ndarray],
) -> np.ndarray:
    """Query vectors for one scope; deduplicated, first-occurrence order."""
    if scope_labels is None:
        return vectors
    picked = np.flatnonzero(scope_labels.astype(str) == scope)
    if link is not None:
        if row_keys is None or len(row_keys) != len(vectors):
            raise ValueError("--query-row-key must align with query vectors")
        position = {str(key): i for i, key in enumerate(row_keys)}
        picked = np.asarray([position[str(link[i])] for i in picked], dtype=np.int64)
    elif len(scope_labels) != len(vectors):
        raise ValueError("--query-scope-key must align with query vectors when no link key is given")
    picked = np.asarray(list(dict.fromkeys(int(i) for i in picked)), dtype=np.int64)
    if len(picked) == 0:
        raise ValueError(f"no queries selected for scope {scope!r}")
    return vectors[picked]


def build_matrix(ids: Sequence[str], vectors: np.ndarray, expected_dim: Optional[int]) -> DenseMatrix:
    if vectors.ndim != 2 or len(ids) != len(vectors):
        raise ValueError("vectors must be 2-D and aligned with ids")
    if vectors.nbytes * 3 > MEMORY_CAP_BYTES:
        raise MemoryError(f"{vectors.nbytes * 3} working bytes exceed the {MEMORY_CAP_BYTES} cap")
    corpus = ScopedCorpus(TurnRecord(node_id=i, evidence_id=i, text="", order=(n,)) for n, i in enumerate(ids))
    return DenseMatrix.from_vectors(corpus, dict(zip(ids, vectors)), expected_dim)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import faiss  # type: ignore  # hard requirement; no substitute backend

    faiss.omp_set_num_threads(args.threads)
    configs = [parse_config(c) if isinstance(c, str) else c for c in (args.config or DEFAULT_CONFIGS)]
    inputs: Dict[str, str] = {}

    scopes: List[Tuple[str, List[str], np.ndarray]] = []
    if args.store:
        ids, vectors = load_store_vectors(Path(args.store))
        inputs[_display_path(Path(args.store))] = _sha256(Path(args.store))
        scopes.append((Path(args.store).stem, ids, vectors))
    for raw in args.docs_npz or []:
        path = Path(raw)
        vectors = load_npz_array(path, args.docs_key)
        ids = ([str(x) for x in load_npz_array(path, args.ids_key)] if args.ids_key
               else [f"row-{i:06d}" for i in range(len(vectors))])
        match = re.search(args.scope_regex, path.name) if args.scope_regex else None
        scopes.append((match.group(0) if match else path.stem, ids, vectors))
        inputs[_display_path(path)] = _sha256(path)
    if not scopes:
        raise SystemExit("provide --store or --docs-npz")

    query_vectors = scope_labels = link = row_keys = None
    if args.query_npz:
        qpath = Path(args.query_npz)
        inputs[_display_path(qpath)] = _sha256(qpath)
        query_vectors = load_npz_array(qpath, args.query_key)
        scope_labels = load_npz_array(qpath, args.query_scope_key) if args.query_scope_key else None
        link = load_npz_array(qpath, args.query_link_key) if args.query_link_key else None
        row_keys = load_npz_array(qpath, args.query_row_key) if args.query_row_key else None

    results = []
    for scope, ids, vectors in scopes:
        dm = build_matrix(ids, vectors, args.expected_dim)
        if query_vectors is None:
            queries, exclude = dm.matrix, np.arange(len(dm.matrix))
        else:
            queries, exclude = select_queries(query_vectors, scope_labels, scope, link, row_keys), None
        if args.max_queries and len(queries) > args.max_queries:
            # Deterministic, evenly spaced subsample (weak-laptop guard).
            keep = np.unique(np.linspace(0, len(queries) - 1, args.max_queries).astype(np.int64))
            queries = queries[keep]
            exclude = exclude[keep] if exclude is not None else None
        audits = [ann_audit(dm, queries, args.k, m=m, ef_construction=args.ef_construction,
                            ef_search_grid=grid, exclude_rows=exclude) for m, grid in configs]
        results.append({"scope": scope, "n_vectors": len(ids), "n_queries": len(queries), "audits": audits})
        print(f"{scope}: n={len(ids)} q={len(queries)} " + " ".join(
            f"M{a['hnsw']['m']}/ef{row['ef_search']}:R@{max(args.k)}={row['recall_at_k'][str(max(args.k))]:.3f}"
            for a in audits for row in a["by_ef_search"]))

    return {
        "schema": "hybridmind_ann_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "evidence_class": "measured_offline",
        "provider_calls": 0,
        "command": " ".join(sys.argv),
        "provenance": {
            "git_commit": _git("rev-parse", "HEAD") or None,
            # The audit code itself may be uncommitted; say so rather than imply HEAD.
            "git_dirty_paths": (_git("status", "--porcelain", "--", "scripts/ann_audit.py",
                                     "engine/dense_channel.py") or "").splitlines(),
            "inputs_sha256": inputs,
            "faiss_version": getattr(faiss, "__version__", None),
            "numpy_version": np.__version__,
            "faiss_cpu_distribution": _version("faiss-cpu"),
            "python": sys.version.split()[0],
            "faiss_threads": args.threads,
        },
        "params": {
            "configs": [{"m": m, "ef_search_grid": grid} for m, grid in configs],
            "ef_construction": args.ef_construction,
            "ks": sorted(set(args.k)),
            "query_mode": "leave_one_out" if query_vectors is None else "query_npz",
            "docs_key": args.docs_key,
            "ids": args.ids_key or "row_index",
            "query_keys": {"vectors": args.query_key, "scope": args.query_scope_key,
                           "link": args.query_link_key, "row": args.query_row_key},
            "expected_dim": args.expected_dim,
            "max_queries": args.max_queries,
        },
        "summary": summarize(results),
        "scopes": results,
    }


def summarize(scopes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Macro-average (and worst scope) recall@k per (M, efSearch) across scopes."""
    cells: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
    for scope in scopes:
        for audit in scope["audits"]:
            for row in audit["by_ef_search"]:
                cells.setdefault((audit["hnsw"]["m"], row["ef_search"]), []).append(row["recall_at_k"])
    return [
        {"m": m, "ef_search": ef, "n_scopes": len(rows),
         "macro_recall_at_k": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
         "worst_scope_recall_at_k": {k: float(np.min([r[k] for r in rows])) for k in rows[0]}}
        for (m, ef), rows in cells.items()
    ]


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    Path(handle.name).replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--docs-npz", nargs="+", help="one .npz per scope")
    src.add_argument("--store", help="HybridMind store.db (read-only, 4096-d)")
    p.add_argument("--docs-key", default="vectors")
    p.add_argument("--ids-key", default=None, help="string id array; default: row index")
    p.add_argument("--scope-regex", default=None, help="scope name = first match in the docs filename")
    p.add_argument("--query-npz", default=None, help="omit for leave-one-out queries")
    p.add_argument("--query-key", default="vectors")
    p.add_argument("--query-scope-key", default=None)
    p.add_argument("--query-link-key", default=None)
    p.add_argument("--query-row-key", default=None)
    p.add_argument("--expected-dim", type=int, default=None)
    p.add_argument("--config", action="append", type=parse_config, help="M:ef1,ef2 (repeatable)")
    p.add_argument("--ef-construction", type=int, default=40)
    p.add_argument("--k", type=int, nargs="+", default=[10, 25, 100])
    p.add_argument("--max-queries", type=int, default=None, help="evenly spaced query subsample")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--label", required=True, help="what these vectors are (and are not)")
    p.add_argument("--output", required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.store and args.expected_dim not in (None, 4096):
        raise SystemExit("store vectors are 4096-d by contract")
    if args.store:
        args.expected_dim = 4096
    payload = run(args)
    _write_atomic(Path(args.output), payload)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
