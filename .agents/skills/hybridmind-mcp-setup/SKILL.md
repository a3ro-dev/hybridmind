---
name: hybridmind-mcp-setup
description: >
  Set up HybridMind as an MCP (Model Context Protocol) memory server.
  Use when the user wants to connect HybridMind to Claude Code, Cursor,
  Windsurf, or any other MCP-compatible AI client. Covers installation,
  configuration, and troubleshooting.
---

# HybridMind MCP Server Setup

HybridMind exposes a plug-n-play MCP server at `mcp_server/main.py`.
Once connected, any MCP client gets four memory tools: **remember**, **recall**, **relate**, **forget**.

## Prerequisites

1. **HybridMind server must be running** at `http://localhost:8000` (or another URL you choose):
   ```bash
   cd /path/to/hybridmind
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Install the MCP SDK** in the HybridMind venv:
   ```bash
   pip install "mcp[cli]"
   ```
   (Already in `requirements.txt` — just run `pip install -r requirements.txt` if starting fresh.)

---

## Connecting to Claude Code

Add to `~/.claude/mcp.json` (create if it doesn't exist):

```json
{
  "mcpServers": {
    "hybridmind": {
      "command": "python",
      "args": ["/absolute/path/to/hybridmind/mcp_server/main.py"],
      "env": {
        "HYBRIDMIND_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

**Windows path example:**
```json
"args": ["D:\\hybridmind\\mcp_server\\main.py"]
```

Then restart Claude Code. You'll see **hybridmind** in the MCP servers panel.

---

## Connecting to Cursor

In `.cursor/mcp.json` (workspace) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "hybridmind": {
      "command": "python",
      "args": ["/absolute/path/to/hybridmind/mcp_server/main.py"],
      "env": {
        "HYBRIDMIND_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

---

## Connecting to Windsurf

In `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "hybridmind": {
      "command": "python",
      "args": ["/absolute/path/to/hybridmind/mcp_server/main.py"],
      "env": {
        "HYBRIDMIND_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

---

## Connecting to any other MCP client

The MCP server uses stdio transport. Start it with:
```bash
python /path/to/hybridmind/mcp_server/main.py
```

Use `mcp dev` to inspect the server tools locally:
```bash
mcp dev /path/to/hybridmind/mcp_server/main.py
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRIDMIND_API_URL` | `http://localhost:8000` | URL of the running HybridMind server |

---

## Verification

After connecting, ask your AI client:
> "Use the remember tool to store: 'My name is Alice and I like hiking.'"
> "Use the recall tool to find memories about Alice."

You should get back the stored memory. ✅

---

## Troubleshooting

**"Connection refused"**: HybridMind server is not running. Start it with `uvicorn main:app`.

**"module not found: mcp"**: Run `pip install "mcp[cli]"` in the HybridMind venv.

**Tools not appearing**: Restart the MCP client after editing mcp.json. Check the MCP server panel for error logs.

**Wrong port**: Set `HYBRIDMIND_API_URL=http://localhost:YOUR_PORT` in the `env` block.
