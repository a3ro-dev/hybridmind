"""
Hybrid Search Test Cases (TC-HYB-01 through TC-HYB-03)
Based on Test Case Markers.
"""

import pytest


class TestHybridWeightedMerge:
    """
    TC-HYB-01 (P0): POST /search/hybrid merges vector and graph scores using weights.
    """
    
    def test_hybrid_returns_combined_scores(self, client, create_test_node):
        """Test that hybrid search returns combined scores with breakdown."""
        create_test_node("Machine learning fundamentals and algorithms", {"topic": "ml"})
        create_test_node("Deep learning neural networks AI", {"topic": "dl"})
        create_test_node("Database administration SQL queries", {"topic": "db"})
        
        response = client.post("/search/hybrid", json={
            "query_text": "machine learning neural networks",
            "top_k": 10,
            "vector_weight": 0.7,
            "graph_weight": 0.3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        
        for result in data["results"]:
            assert "vector_score" in result
            assert "graph_score" in result
            assert "combined_score" in result
            assert 0 <= result["vector_score"] <= 1
            assert 0 <= result["graph_score"] <= 1
            assert 0 <= result["combined_score"] <= 1
    
    def test_combined_score_formula(self, client, create_test_node):
        """Verify combined_score = vector_weight * vector + graph_weight * graph."""
        create_test_node("Test node for score verification")
        
        vector_weight = 0.6
        graph_weight = 0.4
        
        response = client.post("/search/hybrid", json={
            "query_text": "test score verification",
            "top_k": 10,
            "vector_weight": vector_weight,
            "graph_weight": graph_weight
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        for result in results:
            v = result["vector_score"]
            g = result["graph_score"]
            expected = vector_weight * v + graph_weight * g
            actual = result["combined_score"]
            assert abs(actual - expected) < 0.01, f"Score mismatch: expected {expected:.4f}, got {actual:.4f}"


class TestHybridTuningExtremes:
    """
    TC-HYB-02 (P0): Test with vector_weight=1.0 and graph_weight=1.0.
    """
    
    def test_vector_weight_1_graph_weight_0(self, client, create_test_node):
        """With vector_weight=1.0, combined_score equals vector_score."""
        for topic in ["machine learning", "cooking recipes", "database systems"]:
            create_test_node(f"Document about {topic}")
        
        response = client.post("/search/hybrid", json={
            "query_text": "machine learning",
            "top_k": 5,
            "vector_weight": 1.0,
            "graph_weight": 0.0
        })
        
        assert response.status_code == 200
        for result in response.json()["results"]:
            assert abs(result["combined_score"] - result["vector_score"]) < 0.01
    
    def test_vector_weight_0_graph_weight_1(self, client, create_test_node):
        """With graph_weight=1.0, combined_score equals graph_score."""
        create_test_node("Center node for graph test")
        
        response = client.post("/search/hybrid", json={
            "query_text": "center graph test",
            "top_k": 10,
            "vector_weight": 0.0,
            "graph_weight": 1.0
        })
        
        assert response.status_code == 200
        for result in response.json()["results"]:
            assert abs(result["combined_score"] - result["graph_score"]) < 0.01


class TestHybridRelationshipWeighted:
    """
    TC-HYB-03 (P1): Edges with higher weight increase graph proximity score.
    """
    
    def test_edge_weight_affects_ranking(self, client):
        """Higher edge weight should improve graph proximity score."""
        hub = client.post("/nodes", json={"text": "Hub node central to graph", "metadata": {}}).json()
        high_weight = client.post("/nodes", json={"text": "Node with high weight connection", "metadata": {}}).json()
        low_weight = client.post("/nodes", json={"text": "Node with low weight connection", "metadata": {}}).json()
        
        client.post("/edges", json={"source_id": hub["id"], "target_id": high_weight["id"], "type": "relates", "weight": 0.9})
        client.post("/edges", json={"source_id": hub["id"], "target_id": low_weight["id"], "type": "relates", "weight": 0.1})
        
        response = client.post("/search/hybrid", json={
            "query_text": "node connection",
            "top_k": 10,
            "vector_weight": 0.3,
            "graph_weight": 0.7,
            "anchor_nodes": [hub["id"]]
        })
        
        assert response.status_code == 200
        
        # Cleanup
        for node in [hub, high_weight, low_weight]:
            client.delete(f"/nodes/{node['id']}")


class TestHybridSearchResponseFormat:
    """Additional tests for hybrid search response format."""
    
    def test_response_contains_required_fields(self, client, create_test_node):
        """Verify response contains all required fields."""
        create_test_node("Hybrid test node")
        
        response = client.post("/search/hybrid", json={
            "query_text": "hybrid test",
            "top_k": 5,
            "vector_weight": 0.6,
            "graph_weight": 0.4
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "query_time_ms" in data
        assert "search_type" in data
        assert data["search_type"] == "hybrid"
    
    def test_results_ordered_by_combined_score(self, client, create_test_node):
        """Verify results are ordered by combined_score descending."""
        for i in range(5):
            create_test_node(f"Hybrid ordering test node {i}")
        
        response = client.post("/search/hybrid", json={
            "query_text": "hybrid ordering test",
            "top_k": 10,
            "vector_weight": 0.6,
            "graph_weight": 0.4
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        scores = [r["combined_score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Results not ordered at index {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
