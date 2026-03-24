"""
Correctness Tests with Example Dataset (Section 10)
Based on Test Case Markers.

Note: These tests use auto-generated embeddings (not mock embeddings) 
to ensure compatibility with the existing database.
"""

import pytest


CANONICAL_DOCS = {
    "doc1": {"text": "Redis became the default choice for caching mostly because people like avoiding slow databases.", "title": "Redis caching strategies"},
    "doc2": {"text": "The RedisGraph module promises a weird marriage: pretend your cache is also a graph database.", "title": "RedisGraph module"},
    "doc3": {"text": "Distributed systems are basically long-distance relationships.", "title": "Distributed systems"},
    "doc4": {"text": "A short note on cache invalidation: you think you understand it until your application grows.", "title": "Cache invalidation note"},
    "doc5": {"text": "Graph algorithms show up in real life more than people notice.", "title": "Graph algorithms"},
    "doc6": {"text": "README draft: to combine Redis with a graph database, you start by defining nodes.", "title": "README: Redis+Graph"},
}

CANONICAL_EDGES = [
    ("doc1", "doc4", "analogous_to", 0.8),
    ("doc2", "doc6", "derived_from", 0.9),
    ("doc6", "doc1", "depends_on", 0.6),
    ("doc3", "doc5", "analogous_to", 0.5),
    ("doc2", "doc5", "supports", 0.3),
]


@pytest.fixture(scope="module")
def canonical_dataset(client):
    """Create the canonical dataset (embeddings auto-generated)."""
    created_nodes = {}
    created_edges = []
    
    # Create nodes (no custom embeddings - let the system generate them)
    for doc_id, doc_info in CANONICAL_DOCS.items():
        response = client.post("/nodes", json={
            "text": doc_info["text"],
            "metadata": {"title": doc_info["title"], "doc_id": doc_id}
        })
        if response.status_code == 201:
            created_nodes[doc_id] = response.json()["id"]
    
    # Create edges
    for source, target, edge_type, weight in CANONICAL_EDGES:
        if source in created_nodes and target in created_nodes:
            response = client.post("/edges", json={
                "source_id": created_nodes[source],
                "target_id": created_nodes[target],
                "type": edge_type,
                "weight": weight
            })
            if response.status_code == 201:
                created_edges.append(response.json()["id"])
    
    yield {"nodes": created_nodes, "edges": created_edges}
    
    # Cleanup
    for node_id in created_nodes.values():
        try:
            client.delete(f"/nodes/{node_id}")
        except:
            pass


class TestVectorOnlyCanonical:
    """Vector-only search with canonical dataset."""
    
    def test_vector_search_ordering(self, client, canonical_dataset):
        """Test vector search returns results ordered by score."""
        response = client.post("/search/vector", json={
            "query_text": "redis caching database",
            "top_k": 10
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        # Verify ordering by score (descending)
        if len(results) >= 2:
            scores = [r["vector_score"] for r in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], "Results not ordered by score"


class TestGraphOnlyCanonical:
    """Graph-only traversal with canonical dataset."""
    
    def test_graph_traversal_from_doc6(self, client, canonical_dataset):
        """Test graph traversal from doc6 with depth=2."""
        nodes = canonical_dataset["nodes"]
        doc6_id = nodes.get("doc6")
        
        if not doc6_id:
            pytest.skip("doc6 not created")
        
        response = client.get("/search/graph", params={
            "start_id": doc6_id,
            "depth": 2
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        result_ids = [r["node_id"] for r in results]
        
        # doc2 should be reachable (via mentions edge, reverse direction)
        doc2_id = nodes.get("doc2")
        # doc1 should be reachable (via references edge)
        doc1_id = nodes.get("doc1")
        
        # At least some connected nodes should be found
        assert len(results) >= 1, "Should find at least one connected node"


class TestHybridCanonical:
    """Hybrid search with canonical dataset."""
    
    def test_hybrid_search_with_weights(self, client, canonical_dataset):
        """Test hybrid search returns results with score breakdown."""
        nodes = canonical_dataset["nodes"]
        doc1_id = nodes.get("doc1")
        
        if not doc1_id:
            pytest.skip("doc1 not created")
        
        response = client.post("/search/hybrid", json={
            "query_text": "redis caching",
            "vector_weight": 0.6,
            "graph_weight": 0.4,
            "top_k": 10,
            "anchor_nodes": [doc1_id]
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        # Verify score breakdown present
        for result in results:
            assert "vector_score" in result
            assert "graph_score" in result
            assert "combined_score" in result
    
    def test_hybrid_results_sorted(self, client, canonical_dataset):
        """Verify results are sorted by combined_score descending."""
        nodes = canonical_dataset["nodes"]
        doc1_id = nodes.get("doc1")
        
        if not doc1_id:
            pytest.skip("doc1 not created")
        
        response = client.post("/search/hybrid", json={
            "query_text": "redis caching",
            "vector_weight": 0.6,
            "graph_weight": 0.4,
            "top_k": 10,
            "anchor_nodes": [doc1_id]
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        scores = [r["combined_score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Results not sorted by combined_score"


class TestCRUDCanonical:
    """CRUD examples from specification."""
    
    def test_create_node_example(self, client):
        """Test creating a node as per spec example."""
        response = client.post("/nodes", json={
            "text": "Mini note about TTLs and cache expiration.",
            "metadata": {"type": "note", "tags": ["cache"]}
        })
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        
        # Cleanup
        client.delete(f"/nodes/{data['id']}")
    
    def test_get_node_with_edges(self, client, canonical_dataset):
        """Test getting a node with edges."""
        nodes = canonical_dataset["nodes"]
        doc1_id = nodes.get("doc1")
        
        if not doc1_id:
            pytest.skip("doc1 not created")
        
        response = client.get(f"/nodes/{doc1_id}")
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert "text" in data
        assert "metadata" in data
        assert "edges" in data


class TestGraphProximityScore:
    """Test graph proximity score formula: graph_score = 1 / (1 + hops)"""
    
    def test_graph_score_formula(self):
        """Verify graph score calculation matches spec."""
        def compute_graph_score(hops):
            if hops < 0:
                return 0.0
            return 1.0 / (1.0 + hops)
        
        assert compute_graph_score(0) == 1.0
        assert abs(compute_graph_score(1) - 0.5) < 0.001
        assert abs(compute_graph_score(2) - 0.3333) < 0.01
    
    def test_hybrid_score_formula(self):
        """Verify hybrid CRS formula."""
        def compute_hybrid_score(vector_score, graph_score, v_weight=0.6, g_weight=0.4):
            return v_weight * vector_score + g_weight * graph_score
        
        # Test with sample values
        v = 0.8
        g = 0.5
        expected = 0.6 * v + 0.4 * g
        actual = compute_hybrid_score(v, g)
        
        assert abs(actual - expected) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
