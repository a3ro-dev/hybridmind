"""Startup must remain offline unless remote warm-up is explicitly enabled."""

import asyncio
from types import SimpleNamespace

import main
from config import settings


def test_default_startup_does_not_call_embedding_provider(monkeypatch):
    class Embedder:
        model = "remote-4096"

        def warmup(self, **_kwargs):
            raise AssertionError("startup made a remote warm-up call")

        def embed(self, *_args, **_kwargs):
            raise AssertionError("startup made a remote embedding call")

    db = SimpleNamespace(
        embedding_engine=Embedder(),
        vector_index=SimpleNamespace(size=0, dimension=4096),
        graph_index=SimpleNamespace(node_count=0),
        sqlite_store=SimpleNamespace(
            count_retrievable_nodes=lambda: 0,
            get_deleted_nodes_count=lambda: 0,
        ),
        mind_file=SimpleNamespace(read_manifest=lambda: {}),
        get_stats=lambda: {
            "total_nodes": 0,
            "total_edges": 0,
            "vector_index_size": 0,
            "graph_node_count": 0,
        },
        save_indexes=lambda: None,
        close=lambda save=False: None,
        _rebuild_indexes=lambda: None,
    )
    monkeypatch.setattr(settings, "startup_embedding_warmup_enabled", False)
    monkeypatch.setattr(settings, "memory_compression_enabled", False)
    monkeypatch.setattr(main, "_validate_api_security_configuration", lambda: None)
    monkeypatch.setattr(main, "verify_integrity", lambda _path: "PASSED")
    monkeypatch.setattr(main, "get_db_manager", lambda: db)

    async def run_lifespan():
        async with main.lifespan(main.app):
            pass

    asyncio.run(run_lifespan())
