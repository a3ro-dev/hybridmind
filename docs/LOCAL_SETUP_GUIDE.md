# HybridMind Local Database Setup Guide

This guide details how to install, configure, run, and test HybridMind locally on your machine.

---

## 1. Quick Local Installation

We provide an automated installation script that sets up a local virtual environment and installs dependencies:

```bash
# Run the installer script
python install.py
```

This script will automatically:
1. Create a `.venv` virtual environment and run `pip install -r requirements.txt`.
2. Generate a default `.env` file (if missing).
3. Connect the MCP server to **Claude Code**, **Claude Desktop**, and **Windsurf IDE**.

---

## 2. Environment Configuration (`.env`)

Configure parameters via `HYBRIDMIND_*` variables or `.env`:

```ini
# Embedding configuration
HYBRIDMIND_EMBEDDING_DIMENSION=4096

# A remote exact-4096 embedding backend is mandatory
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_TEI_EMBEDDING_URL=https://<endpoint>.api.runpod.ai
# Self-hosted vLLM is optional unless decomposition/extraction is enabled
RUNPOD_LLM_ENDPOINT_ID=your-vllm-endpoint-id

# Feature flags
HYBRIDMIND_AUTO_EDGES_ENABLED=false
HYBRIDMIND_RERANKER_MODEL=mixedbread-ai/mxbai-rerank-large-v2

# Optional cost-bounded research/testing provider; off by default
HYBRIDMIND_ALLOW_RESEARCH_PROXY=false
HYBRIDMIND_RESEARCH_PROXY_API_KEY=
```

---

## 3. Running the Server

Start the local server:

```bash
# Direct startup via main.py
python main.py

# Or via uvicorn directly
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Before any live provider check, create the offline resource report, prepare a
matching bounded live plan, and validate without provider calls:

```powershell
.\.venv\Scripts\python.exe scripts\offline_resource_frontier.py `
  --output benchmarks\results\offline_resource_frontier.json
.\.venv\Scripts\python.exe scripts\preflight.py `
  --plan path\to\live-plan.json --validate-only
```

Only then omit `--validate-only` to warm the providers selected by that plan.
A bare preflight command is rejected without making provider calls. Startup and
requests fail rather than selecting a local or lower-dimensional embedding model.

- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Swagger Docs: `http://127.0.0.1:8000/docs`

---

## 4. Running Tests and Verification

```bash
# Run full unit and integration test suite
python3 -m pytest tests/ -v

# Verify CLI commands
python -m cli.main health
python -m cli.main stats
```
