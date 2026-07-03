"""
Verify fact contradiction / supersession check at ingest.
"""
import os
from config import settings

def test_fact_supersession_logic(client, db_manager):
    # Enable fact extraction (so ingest/session-facts triggers) and override threshold
    orig_fact_enabled = os.environ.get("FACT_EXTRACTION_ENABLED")
    os.environ["FACT_EXTRACTION_ENABLED"] = "true"
    orig_threshold = settings.fact_contradiction_threshold
    settings.fact_contradiction_threshold = 0.80

    try:
        # Clear database
        client.post("/admin/clear")

        # Mock out the fact extraction LLM call by populating fact cache directly
        # format of cache key: turns hash
        from main import _fact_cache, _fact_cache_key
        
        # Turn 1
        turns1 = [{"speaker": "User", "text": "My address is 123 Elm Street.", "date": "2026-07-01"}]
        key1 = _fact_cache_key(turns1)
        _fact_cache[key1] = [{"fact": "The user's address is 123 Elm Street.", "entities": ["user", "Elm Street"], "date": "2026-07-01"}]

        resp1 = client.post("/ingest/session-facts", json={
            "session_id": "sess_addr_1",
            "turns": turns1
        })
        assert resp1.status_code == 200
        n1_id = resp1.json()["node_ids"][0]

        # Turn 2 (contradicting)
        turns2 = [{"speaker": "User", "text": "Actually, my address is 456 Oak Avenue now.", "date": "2026-07-02"}]
        key2 = _fact_cache_key(turns2)
        _fact_cache[key2] = [{"fact": "The user's address is 456 Oak Avenue.", "entities": ["user", "Oak Avenue"], "date": "2026-07-02"}]

        resp2 = client.post("/ingest/session-facts", json={
            "session_id": "sess_addr_2",
            "turns": turns2
        })
        assert resp2.status_code == 200
        n2_id = resp2.json()["node_ids"][0]

        # Verify n2_id supersedes n1_id
        # Old node should have superseded_by in metadata
        node1 = db_manager.sqlite_store.get_node(n1_id)
        assert node1["metadata"].get("superseded_by") == n2_id

        # Verify a "supersedes" edge exists between new and old
        edges = db_manager.sqlite_store.get_node_edges(n2_id)
        assert any(e["type"] == "supersedes" and e["target_id"] == n1_id for e in edges)

    finally:
        if orig_fact_enabled is not None:
            os.environ["FACT_EXTRACTION_ENABLED"] = orig_fact_enabled
        else:
            del os.environ["FACT_EXTRACTION_ENABLED"]
        settings.fact_contradiction_threshold = orig_threshold
