# visual memory on RunPod serverless

this deploys the ColQwen2.5 image embedding server to RunPod serverless. the budget i planned around is **$10–15** for occasional use, with workers scaling to zero when idle.

it's an opt-in experiment. the visual path isn't part of the default retrieval stack.

## 1. build the image

```bash
docker build -f deploy/Dockerfile.image_server -t hybridmind-image-server .
docker tag hybridmind-image-server <your-dockerhub>/hybridmind-image-server:latest
docker push <your-dockerhub>/hybridmind-image-server:latest
```

## 2. create the endpoint

1. open [RunPod serverless](https://www.runpod.io/console/serverless)
2. click **New Endpoint**
3. pick your Docker image
4. GPU: A100 SXM (80GB). ColQwen2.5 needs about 16GB of VRAM
5. min workers = 0, so idle costs nothing
6. max workers = 1
7. copy the endpoint ID

## 3. point HybridMind at it

add to `.env`:

```bash
HYBRIDMIND_IMAGE_EMBEDDING_URL=https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync
HYBRIDMIND_IMAGE_RUNPOD_KEY=your_runpod_api_key
```

## 4. check it works

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

## rough cost

these are estimates at about $2/hr for an A100, not measured bills.

| operation | time | cost |
|-----------|------|-------|
| cold start | ~60s | ~$0.03 |
| one image, warm | ~2s | ~$0.001 |
| 1000 images | ~35min | ~$1.17 |
| a month idle at 0 workers | n/a | $0 |

so $10–15 covers roughly 10,000 image embeddings with room to spare.

## running it locally

```bash
python -m venv .venv_image
.venv_image/Scripts/pip install -r deploy/requirements_image_server.txt
python deploy/runpod_image_handler.py  # local FastAPI on port 8001
```

then set `HYBRIDMIND_IMAGE_EMBEDDING_URL=http://localhost:8001` in `.env`.
