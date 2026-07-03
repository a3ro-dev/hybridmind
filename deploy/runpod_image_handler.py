"""
RunPod Serverless handler for ColQwen2.5 image + query embeddings.

Deploy to RunPod Serverless:
  1. Build Docker image with this file + deploy/requirements_image_server.txt
  2. Push to Docker Hub or RunPod registry
  3. Create a RunPod Serverless endpoint with your image
  4. Set HYBRIDMIND_IMAGE_EMBEDDING_URL in HybridMind .env

Handler input format:
  {"input": {"image_b64": "<base64>"}}
  {"input": {"query": "<text>"}}

Handler output format:
  {"patch_vectors": [[...], ...]}   # for image_b64
  {"query_vectors": [[...], ...]}   # for query

Budget note: ColQwen2.5 (7B) requires ~16GB VRAM. Use an A100 or H100 worker.
For $10-15 budget with occasional use, configure 0 min workers (scale to zero).
Cold start is ~60s; warm inference is ~2s/image.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Lazy-load ColPali to allow fast worker startup
_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return
    import torch
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    model_name = os.getenv("COLQWEN_MODEL", "vidore/colqwen2.5-v0.2")
    logger.info(f"Loading ColQwen2.5 model: {model_name}")

    _processor = ColQwen2_5_Processor.from_pretrained(model_name)
    _model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    ).eval()
    logger.info("ColQwen2.5 model loaded.")


def embed_image_b64(image_b64: str) -> list:
    """Embed a base64-encoded image. Returns list of patch vectors."""
    import torch
    from PIL import Image

    _load_model()
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    inputs = _processor.process_images([image]).to(_model.device)
    with torch.no_grad():
        embeddings = _model(**inputs)  # (1, seq_len, dim)

    patch_vectors = embeddings[0].float().cpu().numpy().tolist()
    return patch_vectors


def embed_query(query: str) -> list:
    """Embed a text query. Returns list of query patch vectors."""
    import torch

    _load_model()
    inputs = _processor.process_queries([query]).to(_model.device)
    with torch.no_grad():
        embeddings = _model(**inputs)  # (1, seq_len, dim)

    query_vectors = embeddings[0].float().cpu().numpy().tolist()
    return query_vectors


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless entry point.

    event["input"] must contain either:
      - "image_b64": str  → returns {"patch_vectors": [...]}
      - "query":    str  → returns {"query_vectors": [...]}
    """
    inp = event.get("input", {})

    if "image_b64" in inp:
        try:
            patch_vectors = embed_image_b64(inp["image_b64"])
            return {"patch_vectors": patch_vectors}
        except Exception as e:
            return {"error": f"image embedding failed: {e}"}

    elif "query" in inp:
        try:
            query_vectors = embed_query(inp["query"])
            return {"query_vectors": query_vectors}
        except Exception as e:
            return {"error": f"query embedding failed: {e}"}

    else:
        return {"error": "input must contain 'image_b64' or 'query'"}


if __name__ == "__main__":
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except ImportError:
        # Local FastAPI fallback for testing
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI(title="ColQwen2.5 Image Embedding Server")

        class ImageRequest(BaseModel):
            image_b64: str

        class QueryRequest(BaseModel):
            query: str

        @app.post("/embed_image")
        async def api_embed_image(req: ImageRequest):
            try:
                patch_vectors = embed_image_b64(req.image_b64)
                return {"patch_vectors": patch_vectors}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.post("/embed_query")
        async def api_embed_query(req: QueryRequest):
            try:
                query_vectors = embed_query(req.query)
                return {"query_vectors": query_vectors}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        uvicorn.run(app, host="0.0.0.0", port=8001)
