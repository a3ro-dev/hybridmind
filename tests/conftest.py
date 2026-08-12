"""
Shared fixtures for HybridMind test suite.
Based on Test Case Markers.
"""

import pytest
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set an isolated .mind root before importing the application. The runtime uses
# mind_file_path as its authoritative storage root; overriding only the legacy
# component paths would make tests open and rebuild the real corpus.
_TEST_DIR = tempfile.mkdtemp(prefix="hybridmind_tests_")
_TEST_MIND = Path(_TEST_DIR) / "hybridmind_test.mind"
os.environ["HYBRIDMIND_MIND_FILE_PATH"] = str(_TEST_MIND)
os.environ["HYBRIDMIND_DATABASE_PATH"] = str(_TEST_MIND / "store.db")
os.environ["HYBRIDMIND_VECTOR_INDEX_PATH"] = str(_TEST_MIND / "vectors")
os.environ["HYBRIDMIND_GRAPH_INDEX_PATH"] = str(_TEST_MIND / "graph.nx")
# Tests run fully offline and must not hit remote endpoints. They inject a
# deterministic 4096-dimensional test double below; this is test dependency
# injection, not a runtime embedding fallback.
for _var in ("RUNPOD_TEI_EMBEDDING_URL", "HC_EMBEDDING_URL", "RUNPOD_EMBEDDING_URL"):
    os.environ[_var] = ""
# Tests preserve the production dimension invariant and inject an offline
# deterministic engine below; they never select a smaller local model.
os.environ["HYBRIDMIND_EMBEDDING_DIMENSION"] = "4096"
os.environ["RERANK_MODE"] = "off"
os.environ["HYBRIDMIND_AUTO_EDGES_ENABLED"] = "false"
os.environ["HYBRIDMIND_TEMPORAL_EDGES_ENABLED"] = "false"
os.environ["HYBRIDMIND_TEMPORAL_DECAY_ENABLED"] = "false"
os.environ["HYBRIDMIND_SALIENCE_ENABLED"] = "false"
os.environ["HYBRIDMIND_QUERY_DECOMPOSITION_ENABLED"] = "false"
os.environ["FACT_EXTRACTION_ENABLED"] = "false"

from tests.embedding_double import Deterministic4096EmbeddingEngine
import engine.embedding as _embedding_module

_TEST_EMBEDDER = Deterministic4096EmbeddingEngine()
_embedding_module.get_embedding_engine = lambda *args, **kwargs: _TEST_EMBEDDER

from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="session", autouse=True)
def isolated_test_store():
    yield
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


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
