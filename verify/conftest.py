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

# Setup mock or live keys if needed for tests
if not os.environ.get("HC_API_KEY"):
    os.environ["HC_API_KEY"] = "mock_key_for_testing"

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
