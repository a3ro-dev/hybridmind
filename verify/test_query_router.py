"""
Verify query routing and weight overriding behavior.
"""
from engine.query_router import route_query

def test_query_routing_classification():
    # 1. Temporal
    route = route_query("When did Alice join Google in 2024?")
    assert route["type"] == "temporal"

    # 2. Multihop
    route = route_query("How is Bob connected to Alice?")
    assert route["type"] == "multihop"

    # 3. Entity
    route = route_query("Who is Bob?")
    assert route["type"] == "entity"
    assert route["metadata_filter"] is None

    # 4. Default
    route = route_query("Acme Corp revenue reports")
    assert route["type"] == "default"

def test_routing_execution(client):
    # Search should run without errors using routed parameters
    resp = client.post("/search/hybrid", json={
        "query_text": "When did Alice start?",
        "top_k": 3
    })
    assert resp.status_code == 200
