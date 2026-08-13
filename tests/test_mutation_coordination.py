import asyncio
import inspect
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.params import Depends

from api import bulk, edges, nodes
from api.dependencies import ProcessMutationCoordinator, coordinate_mutation
from engine import consolidation
from main import detect_communities, ingest_session_facts


def test_async_mutations_remain_serialized_across_await_points():
    coordinator = ProcessMutationCoordinator()
    active = 0
    maximum_active = 0

    async def mutation():
        nonlocal active, maximum_active
        async with coordinator.async_():
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    async def run():
        await asyncio.gather(*(mutation() for _ in range(8)))

    asyncio.run(run())
    assert maximum_active == 1


def test_cancelled_async_waiter_cannot_orphan_process_lock():
    coordinator = ProcessMutationCoordinator()
    acquired = threading.Event()
    release = threading.Event()

    def hold_sync_lock():
        with coordinator.sync():
            acquired.set()
            assert release.wait(timeout=2)

    holder = threading.Thread(target=hold_sync_lock)
    holder.start()
    assert acquired.wait(timeout=2)

    async def run():
        async def wait_for_lock():
            async with coordinator.async_():
                raise AssertionError("cancelled waiter entered the mutation boundary")

        waiter = asyncio.create_task(wait_for_lock())
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await asyncio.to_thread(holder.join, 2)
        assert not holder.is_alive()
        async with coordinator.async_():
            pass

    asyncio.run(run())


def test_primary_mutation_routes_use_shared_coordination_dependency():
    endpoints = (
        nodes.create_node,
        nodes.update_node,
        nodes.delete_node,
        nodes.create_image_node,
        edges.create_edge,
        edges.update_edge,
        edges.delete_edge,
        bulk.bulk_create_nodes,
        bulk.bulk_create_edges,
        bulk.bulk_import,
        bulk.process_unstructured_data,
        bulk.clear_all_data,
        detect_communities,
        ingest_session_facts,
    )
    for endpoint in endpoints:
        parameter = inspect.signature(endpoint).parameters["mutation_guard"]
        assert isinstance(parameter.default, Depends)
        assert parameter.default.dependency is coordinate_mutation


def test_consolidation_enters_manager_mutation_boundary():
    events = []

    @contextmanager
    def mutation():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    manager = SimpleNamespace(sqlite_store=object(), mutation=mutation)
    with pytest.raises(ValueError, match="cannot replace exact source facts"):
        consolidation.consolidate_sessions(manager, archive_sources=True)
    assert events == ["enter", "exit"]
