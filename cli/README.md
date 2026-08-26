# HybridMind Command-Line Interfaces

Three CLIs ship with the repository. All of them talk to the API server or
read `.mind` directories directly; none of them wake a remote provider.

1. **Admin CLI** (`cli/main.py`) — search, node/edge writes, stats, server launch.
2. **Interactive memory shell** (`cli/agent.py`) — an LLM chat loop backed by recall.
3. **`.mind` inspector** (`cli/mind.py`) — offline manifest/database tooling.

Start the API first (default `http://127.0.0.1:8000`); see the README Quick
Start for the preflight rules that govern live provider use.

---

## 1. Admin CLI (`cli/main.py`)

```bash
python -m cli.main <command> [options]
```

| Command | Purpose |
|---|---|
| `search "query" --mode hybrid --top-k 10 --json` | Run vector/graph/hybrid search (`--mode`, weights, `--json`). |
| `compare "query" [--anchor <node-id>]` | Side-by-side vector vs graph vs hybrid results. |
| `add_node "text" [--title T] [--tags a,b] [--json]` | Store a node via `POST /nodes`. |
| `get_node <id> [--json]` | Fetch one node. |
| `add_edge <src> <dst> [--type related_to] [--weight 1.0]` | Create an edge. |
| `stats` | Node/edge/index counts and health summary. |
| `serve [--host 127.0.0.1] [--port 8000] [--reload]` | Launch `main:app` with uvicorn. |
| `load_demo` | Ingest the bundled research-papers demo dataset. |

There are no delete/snapshot subcommands; destructive and snapshot operations
are API-only by design (`DELETE /nodes/{id}`, `POST /snapshot`), so they stay
behind the API security layer.

## 2. Interactive memory shell (`cli/agent.py`)

```bash
python cli/agent.py [--memory-url http://127.0.0.1:8000] [--session <id>]
```

A chat shell that recalls session-scoped and cross-session memories before each
turn (LLM provider follows `engine/llm_client.py` policy). Slash commands:

| Command | Action |
|---|---|
| `/memory` | Show memories recalled on the previous turn. |
| `/stats` | Node/edge counts from `memory.stats()`. |
| `/sessions` | List sessions. |
| `/archive` | Archive the current session, then exit. |
| `/forget <text>` | Recall the nearest node to `<text>`, confirm, soft-delete it by ID. |
| `/clear`, `/help`, `/exit` (`/quit`) | Terminal control. |

## 3. `.mind` inspector (`cli/mind.py`)

Offline tooling over storage directories — never contacts the API:

```bash
python cli/mind.py info     path/to/store.mind    # header/size summary
python cli/mind.py create   path/to/store.mind    # new empty database
python cli/mind.py export   path/to/store.mind -o out.mind.zip
python cli/mind.py import   archive.mind.zip target/
python cli/mind.py list     [directory]
python cli/mind.py delete   path/to/store.mind [-f]
python cli/mind.py manifest path/to/store.mind    # print manifest.json
```

`export` produces the checksummed v2 portable archive described in
`docs/ARCHITECTURE.md`; `import` performs the same path/checksum/semantic
validation as the API restore path.
