"""
API & CRUD Test Cases (TC-API-01 through TC-API-05)
Based on Test Case Markers.
"""

import pytest


class TestCreateNode:
    """
    TC-API-01 (P0): POST /nodes creates a node with text, metadata and optional embedding.
    Expected: 201 Created; response contains id, stored text, metadata.
    """
    
    def test_create_node_basic(self, client):
        """Create node with text and metadata."""
        response = client.post("/nodes", json={
            "text": "Venkat's note on caching",
            "metadata": {"type": "note", "author": "v"}
        })
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["text"] == "Venkat's note on caching"
        assert data["metadata"]["type"] == "note"
        assert data["metadata"]["author"] == "v"
        
        # Cleanup
        client.delete(f"/nodes/{data['id']}")
    
    def test_create_node_verify_retrieval(self, client):
        """Create node and verify GET /nodes/{id} returns same content."""
        create_response = client.post("/nodes", json={
            "text": "Test content for retrieval verification",
            "metadata": {"test": True, "version": 1}
        })
        assert create_response.status_code == 201
        node_id = create_response.json()["id"]
        
        get_response = client.get(f"/nodes/{node_id}")
        assert get_response.status_code == 200
        retrieved = get_response.json()
        
        assert retrieved["id"] == node_id
        assert retrieved["text"] == "Test content for retrieval verification"
        assert retrieved["metadata"]["test"] == True
        
        # Cleanup
        client.delete(f"/nodes/{node_id}")


class TestReadNodeWithRelationships:
    """
    TC-API-02 (P0): GET /nodes/{id} returns node properties plus relationships.
    """
    
    def test_read_node_with_edges(self, client, create_test_node, create_test_edge):
        """Create nodes with edge and verify edge info is returned."""
        node_a_id = create_test_node("Node A - Source", {"role": "source"})
        node_b_id = create_test_node("Node B - Target", {"role": "target"})
        
        # Create edge A -> B
        client.post("/edges", json={
            "source_id": node_a_id,
            "target_id": node_b_id,
            "type": "references",
            "weight": 0.75
        })
        
        # GET node A and verify edges
        get_response = client.get(f"/nodes/{node_a_id}")
        assert get_response.status_code == 200
        node_data = get_response.json()
        
        assert "edges" in node_data
        edges = node_data["edges"]
        assert len(edges) >= 1
        
        found_edge = next((e for e in edges if e["target_id"] == node_b_id), None)
        assert found_edge is not None
        assert found_edge["type"] == "references"
        assert found_edge["weight"] == 0.75


class TestUpdateNode:
    """
    TC-API-03 (P0): PUT /nodes/{id} updates text and triggers embedding regeneration.
    """
    
    def test_update_node_text(self, client):
        """Update node text and verify change."""
        create_response = client.post("/nodes", json={
            "text": "Original text content",
            "metadata": {"version": 1}
        })
        node_id = create_response.json()["id"]
        
        update_response = client.put(f"/nodes/{node_id}", json={
            "text": "Updated text content - completely different",
            "metadata": {"version": 2},
            "regenerate_embedding": True
        })
        
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["text"] == "Updated text content - completely different"
        assert updated["metadata"]["version"] == 2
        
        # Cleanup
        client.delete(f"/nodes/{node_id}")
    
    def test_update_nonexistent_node_returns_404(self, client):
        """Updating non-existent node should return 404."""
        response = client.put("/nodes/nonexistent-node-id-12345", json={
            "text": "New text",
            "regenerate_embedding": False
        })
        assert response.status_code == 404


class TestDeleteNodeCascade:
    """
    TC-API-04 (P0): DELETE /nodes/{id} removes node and all associated edges.
    """
    
    def test_delete_node_cascades_edges(self, client):
        """Delete node and verify edges are also removed."""
        # Create nodes
        node_a = client.post("/nodes", json={"text": "Node A", "metadata": {}}).json()
        node_b = client.post("/nodes", json={"text": "Node B", "metadata": {}}).json()
        
        # Create edge
        edge = client.post("/edges", json={
            "source_id": node_a["id"],
            "target_id": node_b["id"],
            "type": "links_to",
            "weight": 0.5
        }).json()
        
        # Delete node A
        delete_response = client.delete(f"/nodes/{node_a['id']}")
        assert delete_response.status_code == 200
        delete_data = delete_response.json()
        assert delete_data["deleted"] == True
        assert delete_data["edges_removed"] >= 1
        
        # Verify node A is gone
        get_node_response = client.get(f"/nodes/{node_a['id']}")
        assert get_node_response.status_code == 404
        
        # Verify edge is gone
        get_edge_response = client.get(f"/edges/{edge['id']}")
        assert get_edge_response.status_code == 404
        
        # Cleanup remaining node
        client.delete(f"/nodes/{node_b['id']}")
    
    def test_delete_nonexistent_node_returns_404(self, client):
        """Deleting non-existent node should return 404."""
        response = client.delete("/nodes/nonexistent-node-id-12345")
        assert response.status_code == 404


class TestEdgeCRUD:
    """
    TC-API-05 (P1): POST /edges, GET /edges/{id}, update weight and delete.
    """
    
    def test_create_edge(self, client, create_test_node):
        """Create edge between two nodes."""
        node1_id = create_test_node("Source node")
        node2_id = create_test_node("Target node")
        
        response = client.post("/edges", json={
            "source_id": node1_id,
            "target_id": node2_id,
            "type": "cites",
            "weight": 0.9
        })
        
        assert response.status_code == 201
        edge = response.json()
        assert edge["source_id"] == node1_id
        assert edge["target_id"] == node2_id
        assert edge["type"] == "cites"
        assert edge["weight"] == 0.9
        
        # Cleanup
        client.delete(f"/edges/{edge['id']}")
    
    def test_get_edge(self, client, create_test_node):
        """Retrieve edge by ID."""
        node1_id = create_test_node("Node 1")
        node2_id = create_test_node("Node 2")
        
        create_response = client.post("/edges", json={
            "source_id": node1_id,
            "target_id": node2_id,
            "type": "mentions",
            "weight": 0.7
        })
        edge_id = create_response.json()["id"]
        
        get_response = client.get(f"/edges/{edge_id}")
        assert get_response.status_code == 200
        edge = get_response.json()
        assert edge["id"] == edge_id
        assert edge["type"] == "mentions"
        
        # Cleanup
        client.delete(f"/edges/{edge_id}")
    
    def test_update_edge_weight(self, client, create_test_node):
        """Update edge weight."""
        node1_id = create_test_node("Node 1")
        node2_id = create_test_node("Node 2")
        
        create_response = client.post("/edges", json={
            "source_id": node1_id,
            "target_id": node2_id,
            "type": "related_to",
            "weight": 0.5
        })
        edge_id = create_response.json()["id"]
        
        update_response = client.put(f"/edges/{edge_id}", json={"weight": 0.95})
        assert update_response.status_code == 200
        assert update_response.json()["weight"] == 0.95
        
        # Cleanup
        client.delete(f"/edges/{edge_id}")
    
    def test_delete_edge(self, client, create_test_node):
        """Delete edge."""
        node1_id = create_test_node("Node 1")
        node2_id = create_test_node("Node 2")
        
        create_response = client.post("/edges", json={
            "source_id": node1_id,
            "target_id": node2_id,
            "type": "test_edge"
        })
        edge_id = create_response.json()["id"]
        
        delete_response = client.delete(f"/edges/{edge_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"] == True
        
        get_response = client.get(f"/edges/{edge_id}")
        assert get_response.status_code == 404
    
    def test_create_edge_invalid_source_returns_404(self, client, create_test_node):
        """Creating edge with invalid source node should return 404."""
        node_id = create_test_node("Valid node")
        
        response = client.post("/edges", json={
            "source_id": "invalid-source-id",
            "target_id": node_id,
            "type": "test"
        })
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
