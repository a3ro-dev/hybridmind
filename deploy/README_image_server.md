# HybridMind Visual Memory — RunPod Serverless Deployment

This deploys the ColQwen2.5 image embedding server to RunPod Serverless.
Budget: **$10-15** for occasional use with scale-to-zero workers.

## Quick Start

### 1. Build Docker image

```bash
docker build -f deploy/Dockerfile.image_server -t hybridmind-image-server .
docker tag hybridmind-image-server <your-dockerhub>/hybridmind-image-server:latest
docker push <your-dockerhub>/hybridmind-image-server:latest
```

### 2. Create RunPod Serverless endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select your Docker image
4. GPU: **A100 SXM** (80GB) — ColQwen2.5 needs ~16GB VRAM
5. Set **Min Workers = 0** (scale to zero when idle → $0 idle cost)
6. Set **Max Workers = 1**
7. Note your **Endpoint ID**

### 3. Configure HybridMind

Add to your `.env`:

```bash
HYBRIDMIND_IMAGE_EMBEDDING_URL=https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync
HYBRIDMIND_IMAGE_RUNPOD_KEY=your_runpod_api_key
```

### 4. Test

```python
from engine.image_embedding import get_image_embedding_engine
import base64

engine = get_image_embedding_engine()
print(engine.health())  # True

with open("test.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
patches = engine.embed_image(b64)
print(f"Got {len(patches)} patch vectors of dim {len(patches[0])}")
```

## Cost Estimate

| Operation | Time | Cost (A100 @ ~$2/hr) |
|-----------|------|-------|
| Cold start | ~60s | ~$0.03 |
| Image embed (warm) | ~2s | ~$0.001 |
| 1000 images | ~35min | ~$1.17 |
| Monthly idle (0 workers) | — | $0 |

**For $10-15**: ~10,000 image embeddings with comfortable margin.

## Local Testing

To test locally without RunPod:

```bash
python -m venv .venv_image
.venv_image/Scripts/pip install -r deploy/requirements_image_server.txt
python deploy/runpod_image_handler.py  # starts local FastAPI on port 8001
```

Then set `HYBRIDMIND_IMAGE_EMBEDDING_URL=http://localhost:8001` in `.env`.
