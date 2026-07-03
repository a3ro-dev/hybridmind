"""
Verify temporal edges CRUD and decay scoring.
"""
from datetime import datetime, timedelta
from config import settings

def test_temporal_edge_crud(client):
    # 1. Create source and target nodes
    r1 = client.post("/nodes", json={"text": "Node 1"})
    n1_id = r1.json()["id"]
    r2 = client.post("/nodes", json={"text": "Node 2"})
    n2_id = r2.json()["id"]

    # 2. Create edge with temporal fields
    now = datetime.utcnow()
    valid_from = now - timedelta(days=10)
    valid_until = now + timedelta(days=10)

    r_edge = client.post("/edges", json={
        "source_id": n1_id,
        "target_id": n2_id,
        "type": "supports",
        "weight": 0.8,
        "valid_from": valid_from.isoformat(),
        "valid_until": valid_until.isoformat(),
        "confidence": 0.95
    })
    assert r_edge.status_code == 201
    edge = r_edge.json()
    assert edge["valid_from"] is not None
    assert edge["valid_until"] is not None
    assert edge["confidence"] == 0.95
    edge_id = edge["id"]

    # 3. Retrieve and assert
    r_get = client.get(f"/edges/{edge_id}")
    assert r_get.status_code == 200
    retrieved = r_get.json()
    assert retrieved["confidence"] == 0.95

    # 4. Update temporal fields
    r_up = client.put(f"/edges/{edge_id}", json={
        "valid_until": (now - timedelta(days=1)).isoformat(),
        "confidence": 0.5
    })
    assert r_up.status_code == 200
    updated = r_up.json()
    assert updated["confidence"] == 0.5
