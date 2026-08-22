"""Measure lossless raw-embedding override storage without provider calls.

The legacy condition duplicates every native 4096-d float32 vector in both
``embedding`` and ``raw_embedding``. The compact condition stores NULL when
the values are bit-identical and reconstructs the same public pair on read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.sqlite_store import EMBEDDING_DIMENSION, SQLiteStore


SCHEMA = "hybridmind.offline-embedding-storage/v1"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _checkpoint(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")
        connection.commit()
    finally:
        connection.close()


def _logical_vector_digest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """SELECT embedding, COALESCE(raw_embedding, embedding),
                      raw_embedding IS NOT NULL
               FROM nodes ORDER BY id"""
        ).fetchall()
        overrides = 0
        for embedding, raw_embedding, has_override in rows:
            if embedding is None or raw_embedding is None:
                raise RuntimeError("experiment row is missing a native embedding")
            if len(embedding) != EMBEDDING_DIMENSION * 4:
                raise RuntimeError("experiment embedding width is invalid")
            digest.update(embedding)
            digest.update(raw_embedding)
            overrides += int(has_override)
        return digest.hexdigest(), len(rows), overrides
    finally:
        connection.close()


def run_experiment(work_dir: Path, *, nodes: int = 512, seed: int = 20260814) -> dict:
    if nodes < 1:
        raise ValueError("nodes must be positive")
    work_dir.mkdir(parents=True, exist_ok=True)
    compact_path = work_dir / "compact.db"
    legacy_path = work_dir / "legacy-duplicated.db"

    rng = np.random.default_rng(seed)
    store = SQLiteStore(compact_path)
    for index in range(nodes):
        vector = rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        store.create_node(
            node_id=f"node-{index:08d}",
            text=f"deterministic offline source {index}",
            metadata={"experiment_index": index},
            embedding=vector,
            raw_embedding=vector.copy(),
        )
    store.close()
    _checkpoint(compact_path)

    shutil.copy2(compact_path, legacy_path)
    legacy = sqlite3.connect(legacy_path)
    try:
        legacy.execute(
            "UPDATE nodes SET raw_embedding = embedding WHERE embedding IS NOT NULL"
        )
        legacy.commit()
    finally:
        legacy.close()
    _checkpoint(legacy_path)

    compact_digest, compact_rows, compact_overrides = _logical_vector_digest(
        compact_path
    )
    legacy_digest, legacy_rows, legacy_overrides = _logical_vector_digest(
        legacy_path
    )
    if compact_digest != legacy_digest or compact_rows != legacy_rows:
        raise RuntimeError("compact storage changed logical embedding bytes")
    if compact_overrides != 0 or legacy_overrides != nodes:
        raise RuntimeError("experiment conditions were not constructed correctly")

    compact_bytes = compact_path.stat().st_size
    legacy_bytes = legacy_path.stat().st_size
    saved = legacy_bytes - compact_bytes
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "provider_calls": 0,
        "seed": seed,
        "node_count": nodes,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "logical_embedding_pair_sha256": compact_digest,
        "bit_exact_logical_equivalence": True,
        "conditions": {
            "legacy_duplicated": {
                "database_bytes": legacy_bytes,
                "physical_raw_embedding_rows": legacy_overrides,
            },
            "compact_override": {
                "database_bytes": compact_bytes,
                "physical_raw_embedding_rows": compact_overrides,
            },
        },
        "effect": {
            "database_bytes_saved": saved,
            "database_reduction_fraction": saved / legacy_bytes,
            "database_reduction_percent": 100.0 * saved / legacy_bytes,
            "bytes_saved_per_node": saved / nodes,
            "raw_vector_bytes_per_node": EMBEDDING_DIMENSION * 4,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/offline-embedding-storage.json"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hybridmind_embedding_storage_") as temp:
        result = run_experiment(Path(temp), nodes=args.nodes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
