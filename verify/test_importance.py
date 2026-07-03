"""
Verify node importance scoring and pruning.
"""
from datetime import datetime, timedelta

def test_importance_pruning(client, db_manager):
    # Clear database
    client.post("/admin/clear")

    # Ingest 3 nodes:
    # 1. New node with high degree centrality (connected to many things) -> High importance
    # 2. Old node with no connections -> Low importance
    store = db_manager.sqlite_store
    import numpy as np

    now = datetime.utcnow()
    past_time = now - timedelta(days=90)
    emb = np.zeros(db_manager.vector_index.dimension, dtype=np.float32)

    # Node 1: new node
    store.create_node("new_node", "Fresh node content.", {"type": "fact"}, emb, emb)
    db_manager.vector_index.add("new_node", emb)
    db_manager.graph_index.add_node("new_node")

    # Node 2: old node
    store.create_node("old_node", "Old stale node content.", {"type": "fact"}, emb, emb)
    with store._cursor() as cursor:
        cursor.execute("UPDATE nodes SET created_at = ? WHERE id = ?", (past_time.isoformat(), "old_node"))
    db_manager.vector_index.add("old_node", emb)
    db_manager.graph_index.add_node("old_node")

    # Node 3: another node to link to Node 1 (centrality increase)
    store.create_node("other_node", "Other node content.", {"type": "fact"}, emb, emb)
    db_manager.vector_index.add("other_node", emb)
    db_manager.graph_index.add_node("other_node")

    store.create_edge("edge_1", "new_node", "other_node", "related_to", 1.0)
    db_manager.graph_index.add_edge(
        source_id="new_node",
        target_id="other_node",
        edge_type="related_to",
        weight=1.0,
        edge_id="edge_1",
    )

    # Compute importance scores
    from engine.consolidation import importance_score
    score_new = importance_score("new_node", db_manager)
    score_old = importance_score("old_node", db_manager)

    assert score_new > score_old

    # Run pruning endpoint with threshold between the two scores
    resp = client.post("/admin/prune-low-importance", json={"threshold": (score_new + score_old) / 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["nodes_pruned"] >= 1

    # Assert old node is soft-deleted, new node remains
    n_old = store.get_node("old_node")
    assert n_old is None  # get_node filters out deleted
