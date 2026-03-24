"""
Graph Traversal Test Cases (TC-GRAPH-01 through TC-GRAPH-03)
Based on Test Case Markers.
"""

import pytest


class TestGraphTraversalBFS:
    """
    TC-GRAPH-01 (P0): GET /search/graph?start_id=X&depth=2 returns reachable nodes up to depth.
    """
    
    def test_depth_limited_traversal(self, client):
        """Test BFS traversal respects depth limit."""
        # Create chain: A -> B -> C -> D
        node_a = client.post("/nodes", json={"text": "Node A - Start", "metadata": {}}).json()
        node_b = client.post("/nodes", json={"text": "Node B - Depth 1", "metadata": {}}).json()
        node_c = client.post("/nodes", json={"text": "Node C - Depth 2", "metadata": {}}).json()
        node_d = client.post("/nodes", json={"text": "Node D - Depth 3", "metadata": {}}).json()
        
        # Create edges forming chain
        client.post("/edges", json={"source_id": node_a["id"], "target_id": node_b["id"], "type": "next"})
        client.post("/edges", json={"source_id": node_b["id"], "target_id": node_c["id"], "type": "next"})
        client.post("/edges", json={"source_id": node_c["id"], "target_id": node_d["id"], "type": "next"})
        
        # Query with depth=2 from A
        response = client.get("/search/graph", params={"start_id": node_a["id"], "depth": 2})
        
        assert response.status_code == 200
        results = response.json()["results"]
        result_ids = [r["node_id"] for r in results]
        
        # B and C should be found (depth 1 and 2)
        assert node_b["id"] in result_ids, "Node B (depth 1) should be in results"
        assert node_c["id"] in result_ids, "Node C (depth 2) should be in results"
        # D should NOT be found (depth 3 exceeds limit)
        assert node_d["id"] not in result_ids, "Node D (depth 3) should NOT be in results"
        
        # Cleanup
        for node in [node_a, node_b, node_c, node_d]:
            client.delete(f"/nodes/{node['id']}")


class TestGraphTraversalEdgeTypeFilter:
    """
    TC-GRAPH-02 (P1): Graph traversal respects relationship type filtering.
    """
    
    def test_filter_by_edge_type(self, client):
        """Test traversal filtered by specific edge type."""
        author = client.post("/nodes", json={"text": "Author node", "metadata": {}}).json()
        paper1 = client.post("/nodes", json={"text": "Paper written by author", "metadata": {}}).json()
        cited = client.post("/nodes", json={"text": "Paper cited by author", "metadata": {}}).json()
        
        client.post("/edges", json={"source_id": author["id"], "target_id": paper1["id"], "type": "author_of"})
        client.post("/edges", json={"source_id": author["id"], "target_id": cited["id"], "type": "cites"})
        
        response = client.get("/search/graph", params={
            "start_id": author["id"],
            "depth": 1,
            "edge_types": "author_of"
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        result_ids = [r["node_id"] for r in results]
        
        assert paper1["id"] in result_ids, "Paper (author_of) should be found"
        assert cited["id"] not in result_ids, "Cited paper should NOT be found with author_of filter"
        
        # Cleanup
        for node in [author, paper1, cited]:
            client.delete(f"/nodes/{node['id']}")


class TestGraphTraversalCycleHandling:
    """
    TC-GRAPH-03 (P1): Graph has cycles; traversal must not infinite-loop.
    """
    
    def test_cycle_does_not_cause_infinite_loop(self, client):
        """Test that cycles are handled without infinite loops."""
        # Create cycle: A -> B -> C -> A
        node_a = client.post("/nodes", json={"text": "Cycle Node A", "metadata": {}}).json()
        node_b = client.post("/nodes", json={"text": "Cycle Node B", "metadata": {}}).json()
        node_c = client.post("/nodes", json={"text": "Cycle Node C", "metadata": {}}).json()
        
        client.post("/edges", json={"source_id": node_a["id"], "target_id": node_b["id"], "type": "next"})
        client.post("/edges", json={"source_id": node_b["id"], "target_id": node_c["id"], "type": "next"})
        client.post("/edges", json={"source_id": node_c["id"], "target_id": node_a["id"], "type": "next"})
        
        # This should complete without hanging
        response = client.get("/search/graph", params={"start_id": node_a["id"], "depth": 5})
        
        assert response.status_code == 200
        results = response.json()["results"]
        result_ids = [r["node_id"] for r in results]
        
        # B and C should be found
        assert node_b["id"] in result_ids
        assert node_c["id"] in result_ids
        
        # Each should appear only once (no duplicates from cycle)
        assert result_ids.count(node_b["id"]) == 1
        assert result_ids.count(node_c["id"]) == 1
        
        # Cleanup
        for node in [node_a, node_b, node_c]:
            client.delete(f"/nodes/{node['id']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
