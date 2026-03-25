"""
HybridMemory SDK — Python client for the HybridMind REST API.

BUG-7 fix:
  - recall() now uses POST /search/hybrid (or /search/vector) with correct payload
  - trace() now finds an anchor by semantic search, then graph-traverses from it
  - _get() added with query params support
"""
import httpx
from typing import List, Dict, Any, Optional


class HybridMemoryError(RuntimeError):
    pass


class HybridMemory:
    """
    Python SDK for the HybridMind agent memory REST API.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=self.timeout)

    # ==================== Internal HTTP helpers ====================

    def _check_response(self, response: httpx.Response):
        if not (200 <= response.status_code < 300):
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise HybridMemoryError(f"HTTP {response.status_code}: {detail}")

    def _post(self, path: str, payload: dict) -> dict:
        r = self.client.post(f"{self.base_url}{path}", json=payload)
        self._check_response(r)
        return r.json()

    def _get(self, path: str, params: dict = None) -> dict:
        """GET request with optional query parameters."""
        r = self.client.get(f"{self.base_url}{path}", params=params)
        self._check_response(r)
        return r.json()

    # ==================== Public API ====================

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new memory node.
        Returns the new node ID.
        """
        payload = {"text": text}
        if metadata is not None:
            payload["metadata"] = metadata
        response = self.client.post(f"{self.base_url}/nodes", json=payload)
        self._check_response(response)
        return response.json()["id"]

    def recall(self, query: str, top_k: int = 10, mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Retrieve relevant memory nodes using vector or hybrid search.

        BUG-7 fix: previously called GET /search which doesn't exist.
        Now correctly calls POST /search/hybrid or POST /search/vector.

        Args:
            query:  Natural-language query string
            top_k:  Maximum number of results
            mode:   'hybrid' (default), 'vector', or 'graph'
                    Note: 'graph' requires a start node — use trace() instead.
        """
        if mode == "hybrid":
            endpoint = "/search/hybrid"
            payload = {"query_text": query, "top_k": top_k}
        elif mode == "vector":
            endpoint = "/search/vector"
            payload = {"query_text": query, "top_k": top_k}
        elif mode == "graph":
            raise ValueError(
                "graph mode requires a start node id — use trace() instead"
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Use 'hybrid', 'vector', or 'graph'."
            )

        r = self._post(endpoint, payload)
        return r.get("results", [])

    def trace(self, concept: str, depth: int = 2) -> dict:
        """
        Explore the knowledge graph around a concept.

        BUG-7 fix: previously used the path-finder (/search/path), which is
        path-finding between two known nodes, not neighbourhood exploration.
        Now performs:
          1. Semantic vector search to find the closest anchor node for the concept.
          2. BFS graph traversal from that anchor to `depth` hops.

        Args:
            concept:  Natural-language concept to look up
            depth:    Traversal depth (hops from anchor)

        Returns:
            {"anchor": <result dict or None>, "neighbors": [<result dict>, ...]}
        """
        # Step 1: Find anchor node by semantic search
        results = self.recall(concept, top_k=1, mode="vector")
        if not results:
            return {"anchor": None, "neighbors": []}

        anchor = results[0]
        anchor_id = anchor["node_id"]

        # Step 2: Graph traversal from anchor
        r = self._get("/search/graph", params={
            "start_id": anchor_id,
            "depth": depth,
            "direction": "both",
        })

        neighbors = r.get("results", [])
        return {
            "anchor": anchor,
            "neighbors": neighbors,
        }

    def relate(self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0) -> str:
        """
        Create a directed edge relationship between two nodes.
        Returns the new edge ID.
        """
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type,
            "weight": weight,
        }
        response = self.client.post(f"{self.base_url}/edges", json=payload)
        self._check_response(response)
        return response.json()["id"]

    def forget(self, node_id: str) -> None:
        """Soft-delete a memory node."""
        response = self.client.delete(f"{self.base_url}/nodes/{node_id}")
        self._check_response(response)

    def compact(self) -> Dict[str, Any]:
        """Rebuild index and hard-delete soft-deleted nodes."""
        response = self.client.post(f"{self.base_url}/admin/compact")
        self._check_response(response)
        return response.json()

    def stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        response = self.client.get(f"{self.base_url}/search/stats")
        self._check_response(response)
        return response.json()
