"""
Verify visual memory ingestion and MaxSim image search.
"""
from unittest.mock import MagicMock, patch
import numpy as np
from config import settings

@patch("engine.image_embedding.RemoteImageEmbeddingEngine")
def test_visual_memory_flow(mock_engine_cls, client, db_manager):
    # Setup mock image embedding engine
    mock_engine = MagicMock()
    # Mock embed_image returns 4 patches of 128 dimension
    mock_patches = np.random.randn(4, 128).tolist()
    mock_engine.embed_image.return_value = mock_patches
    # Mock embed_query returns 4 query patches of 128 dimension
    mock_query = np.random.randn(4, 128).tolist()
    mock_engine.embed_query.return_value = mock_query
    
    # Configure global mock instance
    mock_engine_cls.return_value = mock_engine
    
    # Enable image embedding URL in settings
    orig_url = settings.image_embedding_url
    settings.image_embedding_url = "http://localhost:8001"
    
    try:
        # Clear database
        client.post("/admin/clear")

        # 1. Ingest an image
        resp = client.post("/nodes/image", json={
            "image_b64": "base64_encoded_dummy_image_data_here",
            "caption": "A beautiful sunset over the mountains.",
            "metadata": {"tags": ["nature", "sunset"]}
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["modality"] == "image"
        node_id = data["id"]

        # Check visual store directly to see if patch vectors are present
        from api.dependencies import get_visual_store
        v_store = get_visual_store()
        saved_patches = v_store.get(node_id)
        assert saved_patches is not None
        assert saved_patches.shape == (4, 128)

        # 2. Search for the image using include_images=True
        resp_search = client.post("/search/hybrid", json={
            "query_text": "mountain sunset",
            "top_k": 5,
            "include_images": True
        })
        assert resp_search.status_code == 200
        search_data = resp_search.json()
        results = search_data["results"]
        
        # Verify the image result is present
        assert len(results) > 0
        assert any(r["node_id"] == node_id and r["reasoning"] == "visual_maxsim" for r in results)

    finally:
        settings.image_embedding_url = orig_url
