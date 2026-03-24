"""
Shared fixtures for HybridMind test suite.
Based on Test Case Markers.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ.setdefault("HYBRIDMIND_DATABASE_PATH", "test_data/hybridmind_test.db")
os.environ.setdefault("HYBRIDMIND_VECTOR_INDEX_PATH", "test_data/vector_test.index") 
os.environ.setdefault("HYBRIDMIND_GRAPH_INDEX_PATH", "test_data/graph_test.pkl")

from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    """Create test client for API tests."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def create_test_node(client):
    """Factory fixture to create test nodes."""
    created_ids = []
    
    def _create(text: str, metadata: dict = None):
        response = client.post("/nodes", json={
            "text": text,
            "metadata": metadata or {}
        })
        if response.status_code == 201:
            node_id = response.json()["id"]
            created_ids.append(node_id)
            return node_id
        return None
    
    yield _create
    
    # Cleanup created nodes
    for node_id in created_ids:
        try:
            client.delete(f"/nodes/{node_id}")
        except:
            pass


@pytest.fixture
def create_test_edge(client):
    """Factory fixture to create test edges."""
    created_ids = []
    
    def _create(source_id: str, target_id: str, edge_type: str = "related_to", weight: float = 1.0):
        response = client.post("/edges", json={
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "weight": weight
        })
        if response.status_code == 201:
            edge_id = response.json()["id"]
            created_ids.append(edge_id)
            return edge_id
        return None
    
    yield _create
    
    # Cleanup created edges
    for edge_id in created_ids:
        try:
            client.delete(f"/edges/{edge_id}")
        except:
            pass
