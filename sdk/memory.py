import httpx
from typing import List, Dict, Any, Optional

class HybridMemoryError(RuntimeError):
    pass

class HybridMemory:
    """
    Python SDK for the HybridMind agent memory REST API.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)
        
    def _check_response(self, response: httpx.Response):
        if not (200 <= response.status_code < 300):
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise HybridMemoryError(f"HTTP {response.status_code}: {detail}")
            
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
        
    def recall(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memory nodes using hybrid search.
        """
        params = {"query": query, "top_k": top_k, "min_score": min_score}
        response = self.client.get(f"{self.base_url}/search", params=params)
        self._check_response(response)
        return response.json()
        
    def relate(self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0) -> str:
        """
        Create a directed edge relationship between two nodes.
        Returns the new edge ID.
        """
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type,
            "weight": weight
        }
        response = self.client.post(f"{self.base_url}/edges", json=payload)
        self._check_response(response)
        return response.json()["id"]
        
    def forget(self, node_id: str) -> None:
        """
        Soft-delete a memory node.
        """
        response = self.client.delete(f"{self.base_url}/nodes/{node_id}")
        self._check_response(response)
        
    def trace(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes.
        """
        response = self.client.get(f"{self.base_url}/search/path/{source_id}/{target_id}")
        if response.status_code == 404:
            return None
        self._check_response(response)
        return response.json()
        
    def compact(self) -> Dict[str, Any]:
        """
        Rebuild index and hard-delete soft-deleted nodes.
        """
        response = self.client.post(f"{self.base_url}/admin/compact")
        self._check_response(response)
        return response.json()
        
    def stats(self) -> Dict[str, Any]:
        """
        Get current system statistics.
        """
        response = self.client.get(f"{self.base_url}/search/stats")
        self._check_response(response)
        return response.json()
