"""Fact-ingest identity and bi-temporal conflict regression tests."""

import pytest

from main import (
    _fact_conflict_plan,
    _find_existing_fact_id,
    _stable_fact_node_id,
)
from storage.sqlite_store import SQLiteStore


def test_stable_fact_identity_is_retry_stable_and_scope_qualified():
    first = _stable_fact_node_id(
        session_id="session-a", container_tag="run-a", fact_text="  Alice LIKES tea. "
    )
    assert first == _stable_fact_node_id(
        session_id="session-a", container_tag="run-a", fact_text="alice likes  tea."
    )
    assert first != _stable_fact_node_id(
        session_id="session-b", container_tag="run-a", fact_text="Alice likes tea."
    )
    assert first != _stable_fact_node_id(
        session_id="session-a", container_tag="run-b", fact_text="Alice likes tea."
    )


def test_legacy_fact_lookup_never_crosses_container(tmp_path):
    store = SQLiteStore(str(tmp_path / "facts.db"))
    try:
        store.create_node(
            "legacy",
            "Alice likes tea.",
            {
                "type": "extracted_fact",
                "session_id": "session-a",
                "container_tag": "run-a",
            },
        )
        stable = _stable_fact_node_id(
            session_id="session-a", container_tag="run-a", fact_text="Alice likes tea."
        )
        assert _find_existing_fact_id(
            store,
            stable_id=stable,
            session_id="session-a",
            container_tag="run-a",
            fact_text="alice LIKES tea.",
        ) == "legacy"
        other = _stable_fact_node_id(
            session_id="session-a", container_tag="run-b", fact_text="Alice likes tea."
        )
        assert _find_existing_fact_id(
            store,
            stable_id=other,
            session_id="session-a",
            container_tag="run-b",
            fact_text="Alice likes tea.",
        ) is None
    finally:
        store.close()


@pytest.mark.parametrize(
    ("new_time", "prior_time", "edge_type", "source", "close_id"),
    [
        ("2025-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "supersedes", "new", "old"),
        ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "supersedes", "old", "new"),
        ("2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "contradicts", "new", None),
    ],
)
def test_conflict_plan_respects_effective_time(
    new_time, prior_time, edge_type, source, close_id
):
    plan = _fact_conflict_plan(
        node_id="new",
        new_valid_from=new_time,
        prior={"id": "old", "valid_from": prior_time},
    )
    assert plan["edge_type"] == edge_type
    assert plan["source_id"] == source
    assert plan["close_id"] == close_id
