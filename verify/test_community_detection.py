"""
Verify Louvain community detection and community summary nodes.
"""
def test_community_detection(client, db_manager, monkeypatch):
    # Derived summaries are provider-backed and fail closed in production.
    # This explicit offline double keeps the verification focused on community
    # membership, provenance edges, and atomic summary persistence.
    monkeypatch.setattr(
        "engine.consolidation.llm_summarize",
        lambda facts, model=None: "Community: " + " | ".join(facts),
    )
    # Clear database
    client.post("/admin/clear")

    # Create a 6-node graph with 2 distinct clusters
    # Cluster A: nodes 0, 1, 2
    # Cluster B: nodes 3, 4, 5
    store = db_manager.sqlite_store
    import numpy as np
    emb = np.zeros(db_manager.vector_index.dimension, dtype=np.float32)

    for i in range(6):
        nid = f"c_node_{i}"
        store.create_node(nid, f"Member of cluster {i//3}.", {"type": "fact"}, emb, emb)
        db_manager.graph_index.add_node(nid)

    # Add edges within Cluster A
    store.create_edge("e_a1", "c_node_0", "c_node_1", "related_to", 1.0)
    store.create_edge("e_a2", "c_node_1", "c_node_2", "related_to", 1.0)
    store.create_edge("e_a3", "c_node_2", "c_node_0", "related_to", 1.0)

    # Add edges within Cluster B
    store.create_edge("e_b1", "c_node_3", "c_node_4", "related_to", 1.0)
    store.create_edge("e_b2", "c_node_4", "c_node_5", "related_to", 1.0)
    store.create_edge("e_b3", "c_node_5", "c_node_3", "related_to", 1.0)

    # Graph index rebuild
    edges = store.get_all_edges()
    db_manager.graph_index.rebuild_from_edges(edges)

    # Run community detection endpoint
    resp = client.post("/admin/detect-communities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["communities_found"] == 2
    assert data["summaries_created"] == 2

    # Verify summary nodes exist in SQLite
    with store._cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM nodes WHERE json_extract(metadata, '$.type') = 'community_summary'"
        )
        count = cursor.fetchone()[0]
    assert count == 2
