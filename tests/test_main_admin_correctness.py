"""Behavioral regressions for destructive main-app administration paths."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import networkx as nx
import numpy as np

import main
from storage.sqlite_store import SQLiteStore


class _Projection:
    def __init__(self, *, fail_remove: bool = False):
        self.clear_calls = 0
        self.remove_calls = []
        self.fail_remove = fail_remove

    def clear(self):
        self.clear_calls += 1

    def remove(self, node_id):
        self.remove_calls.append(node_id)
        if self.fail_remove:
            raise RuntimeError("projection failure")
        return True


class _GraphProjection(_Projection):
    def __init__(self):
        super().__init__()
        self.graph = nx.MultiDiGraph()

    def remove_node(self, node_id):
        return self.remove(node_id)


class _Manager:
    def __init__(self, store, *, vector=None):
        self.sqlite_store = store
        self.vector_index = vector or _Projection()
        self.graph_index = _GraphProjection()
        self.bm25_index = _Projection()
        self.visual_store = None
        self.colbert_store = None
        self.rebuild_calls = 0
        self.guard_entries = 0

    @asynccontextmanager
    async def mutation_async(self):
        self.guard_entries += 1
        yield

    def _rebuild_indexes(self):
        self.rebuild_calls += 1


def _create_node(store: SQLiteStore, node_id: str = "node-1") -> None:
    vector = np.zeros(4096, dtype=np.float32)
    store.create_node(
        node_id=node_id,
        text="text that must be erased",
        metadata={},
        embedding=vector,
        raw_embedding=vector,
    )


def test_admin_clear_erases_version_history_and_primary_projections(tmp_path, monkeypatch):
    store = SQLiteStore(str(tmp_path / "clear.db"))
    _create_node(store)
    manager = _Manager(store)
    monkeypatch.setattr(main, "get_db_manager", lambda: manager)

    result = asyncio.run(main.clear_database())

    assert result == {"status": "success", "message": "Database cleared"}
    assert store.count_nodes() == 0
    with store._cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM node_versions")
        assert cursor.fetchone()[0] == 0
    assert manager.vector_index.clear_calls == 1
    assert manager.graph_index.clear_calls == 1
    assert manager.bm25_index.clear_calls == 1
    assert manager.guard_entries == 1


def test_admin_prune_rolls_sql_back_and_rebuilds_on_projection_failure(
    tmp_path, monkeypatch
):
    store = SQLiteStore(str(tmp_path / "prune.db"))
    _create_node(store)
    manager = _Manager(store, vector=_Projection(fail_remove=True))
    manager.graph_index.graph.add_node("node-1")
    monkeypatch.setattr(main, "get_db_manager", lambda: manager)
    monkeypatch.setattr("engine.consolidation.importance_score", lambda *a, **k: 0.0)

    response = asyncio.run(main.prune_low_importance(main.PruneRequest(threshold=0.5)))

    assert response.status_code == 500
    assert store.get_node("node-1") is not None
    with store._cursor() as cursor:
        cursor.execute("SELECT deleted_at FROM nodes WHERE id = ?", ("node-1",))
        assert cursor.fetchone()[0] is None
    assert manager.rebuild_calls == 1
    assert manager.guard_entries == 1
