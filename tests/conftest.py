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
# Tests run offline and must NOT hit the RunPod/TEI or HackClub remotes. The
# embedding backend has no runtime fallback anymore (a down remote raises), so
# we explicitly force the LOCAL bge-m3 engine by clearing every remote-URL env
# BEFORE importing main. load_dotenv(override=False) then leaves these empty
# values in place, so get_embedding_engine() falls through to the local model.
for _var in ("RUNPOD_TEI_EMBEDDING_URL", "HC_EMBEDDING_URL", "RUNPOD_EMBEDDING_URL"):
    os.environ[_var] = ""
# Local bge-m3 is 1024-dim; match the FAISS index dimension so a fresh test
# index isn't born mismatched (see main.py's startup dimension guard, Phase 6 A2).
os.environ.setdefault("HYBRIDMIND_EMBEDDING_DIMENSION", "1024")

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
