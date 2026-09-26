# command-line tools

three CLIs ship with the repo. they either talk to the API server or read `.mind` directories directly. none of them wake a remote provider on their own.

1. **admin CLI** (`cli/main.py`): search, node and edge writes, stats, server launch.
2. **memory shell** (`cli/agent.py`): an LLM chat loop that recalls memory before each turn.
3. **`.mind` inspector** (`cli/mind.py`): offline manifest and database tooling.

start the API first (default `http://127.0.0.1:8000`). the preflight rules for live provider use are in the README quick start.

## admin CLI

```bash
python -m cli.main <command> [options]
```

| command | what it does |
|---|---|
| `search "query" --mode hybrid --top-k 10 --json` | vector, graph, or hybrid search (`--mode`, weights, `--json`) |
| `compare "query" [--anchor <node-id>]` | vector vs graph vs hybrid, side by side |
| `add_node "text" [--title T] [--tags a,b] [--json]` | store a node via `POST /nodes` |
| `get_node <id> [--json]` | fetch one node |
| `add_edge <src> <dst> [--type related_to] [--weight 1.0]` | create an edge |
| `stats` | node, edge, and index counts plus health |
| `serve [--host 127.0.0.1] [--port 8000] [--reload]` | run `main:app` with uvicorn |
| `load_demo` | ingest the bundled research-papers demo dataset |

there are no delete or snapshot subcommands. those stay API-only (`DELETE /nodes/{id}`, `POST /snapshot`) so they sit behind the API's security layer.

## memory shell

```bash
python cli/agent.py [--memory-url http://127.0.0.1:8000] [--session <id>]
```

a chat shell that pulls session-scoped and cross-session memories before every turn. the LLM provider follows the policy in `engine/llm_client.py`.

| command | what it does |
|---|---|
| `/memory` | show what got recalled on the last turn |
| `/stats` | node and edge counts from `memory.stats()` |
| `/sessions` | list sessions |
| `/archive` | archive the current session, then exit |
| `/forget <text>` | find the nearest node to `<text>`, confirm, soft-delete it by ID |
| `/clear`, `/help`, `/exit` (`/quit`) | terminal control |

## `.mind` inspector

offline tooling over storage directories. it never contacts the API.

```bash
python cli/mind.py info     path/to/store.mind    # header and size summary
python cli/mind.py create   path/to/store.mind    # new empty database
python cli/mind.py export   path/to/store.mind -o out.mind.zip
python cli/mind.py import   archive.mind.zip target/
python cli/mind.py list     [directory]
python cli/mind.py delete   path/to/store.mind [-f]
python cli/mind.py manifest path/to/store.mind    # print manifest.json
```

`export` writes the checksummed v2 archive described in `docs/ARCHITECTURE.md`. `import` runs the same path, checksum, and semantic checks as the API restore path.
