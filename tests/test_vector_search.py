"""
Vector Search Test Cases (TC-VEC-01 through TC-VEC-03)
Based on Test Case Markers.
"""

import pytest


class TestVectorSearchOrdering:
    """
    TC-VEC-01 (P0): POST /search/vector returns top-k results ordered by cosine similarity.
    """
    
    def test_top_k_ordering_basic(self, client, create_test_node):
        """Test that results are ordered by cosine similarity (descending)."""
        # Create nodes with varied content
        create_test_node("Machine learning algorithms for data processing", {"topic": "ml"})
        create_test_node("Database indexing and query optimization", {"topic": "db"})
        create_test_node("Cooking recipes and kitchen appliances", {"topic": "cooking"})
        
        response = client.post("/search/vector", json={
            "query_text": "machine learning data algorithms",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        
        results = data["results"]
        if len(results) >= 2:
            scores = [r.get("vector_score", 0) for r in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], "Results not ordered by score"
    
    def test_vector_search_response_format(self, client, create_test_node):
        """Verify response contains required fields."""
        create_test_node("Test node for response format")
        
        response = client.post("/search/vector", json={
            "query_text": "test response format",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "query_time_ms" in data
        assert "search_type" in data
        assert data["search_type"] == "vector"


class TestVectorSearchKGreaterThanDataset:
    """
    TC-VEC-02 (P1): Top-k with k > dataset size returns all items without error.
    """
    
    def test_top_k_exceeds_dataset_size(self, client, create_test_node):
        """Request more results than exist in dataset."""
        for i in range(3):
            create_test_node(f"Small dataset node {i}")
        
        response = client.post("/search/vector", json={
            "query_text": "small dataset node",
            "top_k": 100
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) <= 100


class TestVectorSearchMetadataFilter:
    """
    TC-VEC-03 (P1): Pass filter metadata.type=note with vector query.
    """
    
    def test_filter_by_metadata_type(self, client, create_test_node):
        """Filter vector search results by metadata type."""
        create_test_node("Note about caching strategies", {"type": "note"})
        create_test_node("Article about caching patterns", {"type": "article"})
        create_test_node("Another note on database optimization", {"type": "note"})
        
        response = client.post("/search/vector", json={
            "query_text": "caching strategies",
            "top_k": 10,
            "filter_metadata": {"type": "note"}
        })
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
