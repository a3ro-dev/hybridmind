"""
Verify session memory consolidation.
"""
from datetime import datetime, timedelta

def test_session_consolidation(client, db_manager, monkeypatch):
    # Consolidation must never invent a local fallback when the provider is
    # unavailable. Inject a deterministic explicit test double instead.
    monkeypatch.setattr(
        "engine.consolidation.llm_summarize",
        lambda facts, model=None: " | ".join(facts),
    )
    # Clear database
    client.post("/admin/clear")

    # Ingest 5 facts for a single session older than 24 hours (simulated by setting created_at back)
    session_id = "sess_consolidation_test"
        
    # Create facts directly using create_node with past timestamp
    store = db_manager.sqlite_store
    import numpy as np
        
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

    # Run consolidation endpoint with the explicit offline summary double.
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
