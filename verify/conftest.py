"""
Shared fixtures for HybridMind verify tests.
Runs on an isolated temporary .mind store and test client.
"""
import os
import shutil
import tempfile
from pathlib import Path
import pytest

# Force isolated test environment paths
_TEMP_DIR = tempfile.mkdtemp(prefix="hybridmind_verify_")
os.environ["HYBRIDMIND_MIND_FILE_PATH"] = str(Path(_TEMP_DIR) / "test.mind")
os.environ["HYBRIDMIND_DATABASE_PATH"] = str(Path(_TEMP_DIR) / "test.mind" / "store.db")
os.environ["HYBRIDMIND_VECTOR_INDEX_PATH"] = str(Path(_TEMP_DIR) / "test.mind" / "vectors")
os.environ["HYBRIDMIND_GRAPH_INDEX_PATH"] = str(Path(_TEMP_DIR) / "test.mind" / "graph.nx")

# Verification must be offline and dimension-consistent: never let a test
# process load the production corpus, and never relax the 4096-dim invariant.
for _var in ("RUNPOD_TEI_EMBEDDING_URL", "HC_EMBEDDING_URL", "RUNPOD_EMBEDDING_URL"):
    os.environ[_var] = ""
for _var in ("RUNPOD_API_KEY", "RUNPOD_LLM_ENDPOINT_ID", "ZAI_API_KEY"):
    os.environ[_var] = ""
os.environ["HYBRIDMIND_EMBEDDING_DIMENSION"] = "4096"
os.environ["RERANK_MODE"] = "off"
os.environ["HYBRIDMIND_AUTO_EDGES_ENABLED"] = "false"
os.environ["HYBRIDMIND_TEMPORAL_EDGES_ENABLED"] = "false"
os.environ["HYBRIDMIND_TEMPORAL_DECAY_ENABLED"] = "false"
os.environ["HYBRIDMIND_SALIENCE_ENABLED"] = "false"
os.environ["HYBRIDMIND_QUERY_DECOMPOSITION_ENABLED"] = "false"
os.environ["FACT_EXTRACTION_ENABLED"] = "false"
os.environ["HYBRIDMIND_ALLOW_RESEARCH_PROXY"] = "false"

from tests.embedding_double import Deterministic4096EmbeddingEngine
import engine.embedding as _embedding_module

_VERIFY_EMBEDDER = Deterministic4096EmbeddingEngine()
_embedding_module.get_embedding_engine = lambda *args, **kwargs: _VERIFY_EMBEDDER

from fastapi.testclient import TestClient
from main import app
from api.dependencies import get_db_manager


@pytest.fixture(scope="session", autouse=True)
def test_env():
    """Ensure cleanup of temp database files after runs."""
    yield
    try:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(scope="session")
def client():
    """Provide isolated API TestClient."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def db_manager():
    """Provide database manager singleton for direct assertions."""
    return get_db_manager()
