"""Lossless storage contracts for native 4096-d raw embeddings."""

import sqlite3

import numpy as np

from storage.sqlite_store import SQLiteStore


def _physical_raw_length(path, node_id: str):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            "SELECT length(raw_embedding) FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()[0]
    finally:
        connection.close()


def test_identical_raw_embedding_is_compact_but_read_is_bit_exact(tmp_path):
    path = tmp_path / "compact.db"
    store = SQLiteStore(path)
    vector = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    store.create_node(
        "same", "source", {}, embedding=vector, raw_embedding=vector.copy()
    )

    assert _physical_raw_length(path, "same") is None
    restored = store.get_node("same")
    assert np.array_equal(restored["embedding"], vector)
    assert np.array_equal(restored["raw_embedding"], vector)
    assert restored["embedding"] is not restored["raw_embedding"]
    assert not np.shares_memory(restored["embedding"], restored["raw_embedding"])


def test_distinct_graph_conditioned_and_raw_vectors_remain_distinct(tmp_path):
    path = tmp_path / "override.db"
    store = SQLiteStore(path)
    embedding = np.zeros(4096, dtype=np.float32)
    embedding[0] = 1.0
    raw = np.zeros(4096, dtype=np.float32)
    raw[1] = 1.0
    store.create_node(
        "different", "source", {}, embedding=embedding, raw_embedding=raw
    )

    assert _physical_raw_length(path, "different") == 4096 * 4
    restored = store.get_node("different")
    assert np.array_equal(restored["embedding"], embedding)
    assert np.array_equal(restored["raw_embedding"], raw)

    store.update_node(
        "different", embedding=raw.copy(), raw_embedding=raw.copy()
    )
    assert _physical_raw_length(path, "different") is None
    updated = store.get_node("different")
    assert np.array_equal(updated["embedding"], raw)
    assert np.array_equal(updated["raw_embedding"], raw)


def test_existing_duplicate_blob_is_losslessly_migrated_to_override_null(tmp_path):
    path = tmp_path / "legacy.db"
    vector = np.arange(4096, dtype=np.float32)
    store = SQLiteStore(path)
    store.create_node("legacy", "source", {}, embedding=vector)
    store.close()

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE nodes SET raw_embedding = embedding WHERE id = 'legacy'"
    )
    connection.commit()
    connection.close()
    assert _physical_raw_length(path, "legacy") == 4096 * 4

    reopened = SQLiteStore(path)
    assert _physical_raw_length(path, "legacy") is None
    restored = reopened.get_node("legacy")
    assert np.array_equal(restored["embedding"], vector)
    assert np.array_equal(restored["raw_embedding"], vector)
