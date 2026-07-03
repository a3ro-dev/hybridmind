# HybridMind Local Database Setup Guide

This guide details how to run HybridMind locally on your machine and connect it to any AI agent or IDE (Claude Code, Claude Desktop, ChatGPT, Cursor, Windsurf).

---

## 1. Quick Local Installation

We provide an automated installation script that sets up a local virtual environment, installs dependencies, and configures MCP servers for your IDEs:

```bash
# Run the configurator script
python install.py
```

This script will automatically:
1. Create a `.venv` virtual environment and run `pip install -r requirements.txt`.
2. Generate a default `.env` file (if missing).
3. Connect the MCP server to **Claude Code**, **Claude Desktop**, and **Windsurf IDE**.
4. Generate the **ChatGPT OpenAPI Schema** for Custom GPT integration.

Alternatively, you can install the database as an editable pip package globally or inside your active environment:

```bash
# Install as editable pip package
pip install -e .

# Start the database server directly via CLI
hybridmind
```

---

## 2. Running the Local Database Server

To start the database server locally:

```bash
# Start the server directly via Python
python main.py

# Or via the installed package command line
hybridmind
```

The database runs entirely locally and starts up in less than 5 seconds.
- Interactive Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- REST API Base URL: `http://localhost:8000`

---

## 3. Connecting to AI Agents & IDEs

### Claude Code CLI
Claude Code is automatically configured by `install.py`.
Verify that the `hybridmind` server is listed and responsive:
```bash
claude mcp list
```

### Claude Desktop
Claude Desktop is configured via `claude_desktop_config.json`.
1. Make sure your local HybridMind server is running (`python main.py`).
2. Restart Claude Desktop.
3. Look for the "plug" icon in the input field to see the tools: `remember`, `recall`, `relate`, and `forget`.

### Cursor IDE
1. Open Cursor Settings -> Features -> MCP.
2. Click **Add New MCP Server**.
   - **Name**: `hybridmind`
   - **Type**: `command`
   - **Command**: `/path/to/hybridmind/.venv/Scripts/python mcp_server/main.py` (use the virtualenv python).
3. Click Save.

### Windsurf IDE
Windsurf is automatically configured by `install.py` in your global `mcp_config.json`.
Restart Windsurf to load the `hybridmind` tools.

### ChatGPT (Custom GPT / Custom Actions)
ChatGPT requires a public URL since it runs in the cloud and cannot access `localhost` directly.
1. Start an **ngrok** tunnel to forward your local server:
   ```bash
   ngrok http 8000
   ```
2. Copy the public HTTPS URL from ngrok (e.g., `https://your-subdomain.ngrok-free.app`).
3. Open ChatGPT and create a **Custom GPT**.
4. Under **Configure**, click **Create New Action**.
5. Paste the content of `docs/chatgpt_openapi_schema.json` into the schema box.
6. Replace the server URL in the schema with your ngrok HTTPS URL:
   ```json
   "servers": [
     {
       "url": "https://your-subdomain.ngrok-free.app"
     }
   ]
   ```
7. Custom GPT can now read and write directly to your local HybridMind memory database!

---

## 4. RunPod Remote Offloading

To prevent local heating/resource usage, the database is configured to offload heavy calculations:
- **Text Embeddings**: Offloaded to Hack Club proxy Qwen3-Embedding-8B (pre-configured in `.env`).
- **Visual Memory (ColQwen2.5)**: Offloaded to RunPod Serverless. Refer to [deploy/README_image_server.md](file:///d:/hybridmind/deploy/README_image_server.md) for deployment instructions.
- **Phase 6 Training**: Run on RunPod GPU instances (out of scope for local running).
