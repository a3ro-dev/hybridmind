"""MCP stdio server for HybridMind.

Expose a small memory tool surface to MCP-compatible clients. The server is a
thin adapter over the local HybridMind FastAPI process, configured via:

    HYBRIDMIND_API_URL=http://localhost:8000
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP


API_URL = os.getenv("HYBRIDMIND_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = float(os.getenv("HYBRIDMIND_MCP_TIMEOUT_SECONDS", "60"))

mcp = FastMCP("HybridMind")


def _edge_type(edge_type: str) -> str:
    """Map friendly aliases to the server's canonical edge taxonomy."""
    return "analogous_to" if edge_type == "related_to" else edge_type


def _request(method: str, path: str, **kwargs: Any) -> Any:
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        response = client.request(method, f"{API_URL}{path}", **kwargs)
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()


@mcp.tool()
def remember(text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Store a memory node in HybridMind."""
    return _request("POST", "/nodes", json={"text": text, "metadata": metadata or {}})


@mcp.tool()
def recall(query: str, top_k: int = 10, mode: str = "hybrid") -> List[Dict[str, Any]]:
    """Recall relevant memories with hybrid or vector search."""
    top_k = max(1, min(int(top_k), 200))
    if mode == "vector":
        payload = {"query_text": query, "top_k": top_k}
        data = _request("POST", "/search/vector", json=payload)
    else:
        payload = {"query_text": query, "top_k": top_k}
        data = _request("POST", "/search/hybrid", json=payload)
    return data.get("results", [])


@mcp.tool()
def relate(
    source_id: str,
    target_id: str,
    relationship: str = "related_to",
    weight: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a relationship edge between two memory nodes."""
    payload = {
        "source_id": source_id,
        "target_id": target_id,
        "type": _edge_type(relationship),
        "weight": max(0.0, min(float(weight), 1.0)),
        "metadata": metadata or {},
    }
    return _request("POST", "/edges", json=payload)


@mcp.tool()
def forget(node_id: str) -> Dict[str, Any]:
    """Soft-delete a memory node from HybridMind."""
    return _request("DELETE", f"/nodes/{node_id}")


@mcp.tool()
def health() -> Dict[str, Any]:
    """Return HybridMind server health information."""
    return _request("GET", "/health")


if __name__ == "__main__":
    mcp.run()
