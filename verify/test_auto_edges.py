"""
Verify auto-edge creation on ingest.
"""
from config import settings

def test_auto_edge_creation(client, db_manager):
    # Enable auto edges
    original_enabled = settings.auto_edges_enabled
    settings.auto_edges_enabled = True
    settings.auto_edge_cosine_threshold = 0.70
    settings.auto_edge_max_per_node = 5

    try:
        # Clear database
        client.post("/admin/clear")

        # Create two semantically identical nodes
        resp1 = client.post("/nodes", json={
            "text": "The company Acme Corp manufactures heavy anvil equipment."
        })
        node1_id = resp1.json()["id"]

        resp2 = client.post("/nodes", json={
            "text": "Acme Corp is a manufacturer of heavy anvils."
        })
        node2_id = resp2.json()["id"]

        # Retrieve nodes and verify they are linked via similar_to edge
        resp_edges = client.get(f"/edges?source_id={node1_id}")
        assert resp_edges.status_code == 200
        edges = resp_edges.json()
        assert len(edges) > 0
        assert any(e["type"] == "similar_to" for e in edges)

    finally:
        settings.auto_edges_enabled = original_enabled
