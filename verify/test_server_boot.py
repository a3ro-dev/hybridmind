"""
Verify server boot and basic node CRUD operations.
"""
def test_boot_and_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "database" in data["components"]

def test_node_crud(client):
    # 1. Create a node
    resp = client.post("/nodes", json={
        "text": "Alice works at Acme Corp as a Staff Engineer.",
        "metadata": {"sessionId": "session_crud_test"}
    })
    assert resp.status_code == 201
    node = resp.json()
    assert node["text"] == "Alice works at Acme Corp as a Staff Engineer."
    assert node["metadata"]["sessionId"] == "session_crud_test"
    node_id = node["id"]

    # 2. Retrieve the node
    resp = client.get(f"/nodes/{node_id}")
    assert resp.status_code == 200
    retrieved = resp.json()
    assert retrieved["text"] == "Alice works at Acme Corp as a Staff Engineer."

    # 3. Update the node
    resp = client.put(f"/nodes/{node_id}", json={
        "text": "Alice works at Acme Corp as a Principal Engineer.",
        "metadata": {"sessionId": "session_crud_test_updated"}
    })
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["text"] == "Alice works at Acme Corp as a Principal Engineer."
    assert updated["metadata"]["sessionId"] == "session_crud_test_updated"

    # 4. List nodes
    resp = client.get("/nodes")
    assert resp.status_code == 200
    nodes = resp.json()
    assert any(n["id"] == node_id for n in nodes)

    # 5. Delete the node
    resp = client.delete(f"/nodes/{node_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
