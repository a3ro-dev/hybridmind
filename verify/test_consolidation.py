"""
Verify session memory consolidation.
"""
from datetime import datetime, timedelta
import os

def test_session_consolidation(client, db_manager):
    # Enable fact extraction
    orig_fact_enabled = os.environ.get("FACT_EXTRACTION_ENABLED")
    os.environ["FACT_EXTRACTION_ENABLED"] = "true"

    try:
        # Clear database
        client.post("/admin/clear")

        # Ingest 5 facts for a single session older than 24 hours (simulated by setting created_at back)
        session_id = "sess_consolidation_test"
        
        # Create facts directly using create_node with past timestamp
        store = db_manager.sqlite_store
        import uuid
        import numpy as np
        
        node_ids = []
        past_time = datetime.utcnow() - timedelta(hours=36)
        
        for i in range(5):
            nid = f"fact_node_{i}"
            embedding = np.zeros(db_manager.vector_index.dimension, dtype=np.float32)
            store.create_node(
                node_id=nid,
                text=f"Fact statement detail number {i}.",
                metadata={
                    "type": "extracted_fact",
                    "session_id": session_id,
                    "memory_pool": "raw"
                },
                embedding=embedding,
                raw_embedding=embedding
            )
            # manually set created_at back in sqlite
            with store._cursor() as cursor:
                cursor.execute(
                    "UPDATE nodes SET created_at = ? WHERE id = ?",
                    (past_time.isoformat(), nid)
                )
            db_manager.vector_index.add(nid, embedding)
            db_manager.graph_index.add_node(nid)
            node_ids.append(nid)

        # Run consolidation endpoint (mocking LLM summarize by ensuring it falls back safely to joined text)
        resp = client.post("/admin/consolidate", json={
            "min_facts": 5,
            "max_age_hours": 24
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["summaries_created"] == 1

        # Assert summary node exists
        with store._cursor() as cursor:
            cursor.execute(
                "SELECT id, text, metadata FROM nodes WHERE json_extract(metadata, '$.type') = 'session_summary'"
            )
            summary_node = cursor.fetchone()
        assert summary_node is not None
        assert "Fact statement detail number" in summary_node["text"]

    finally:
        if orig_fact_enabled is not None:
            os.environ["FACT_EXTRACTION_ENABLED"] = orig_fact_enabled
        else:
            del os.environ["FACT_EXTRACTION_ENABLED"]
