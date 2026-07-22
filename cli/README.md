# HybridMind Command-Line Interfaces (CLI)

HybridMind provides three complementary command-line utilities for operating, managing, and inspecting the database:

1. **Administrative CLI** (`cli/main.py`) — Typer + Rich powered CLI for database management, search queries, bulk operations, and snapshot control.
2. **Interactive Memory Shell** (`cli/agent.py`) — Real-time interactive session shell for AI agents using slash commands.
3. **Snapshot Inspector** (`cli/mind.py`) — Direct offline inspector for `.mind` storage directories.

---

## 1. Administrative CLI (`cli/main.py`)

Execute administrative commands against a running HybridMind server:

```bash
python -m cli.main [COMMAND] [OPTIONS]
```

### Key Commands

- **Search**:
  ```bash
  python -m cli.main search "self-attention transformer" --mode hybrid --top-k 5
  python -m cli.main search "query text" --mode vector --top-k 10 --json
  ```
- **Node Operations**:
  ```bash
  python -m cli.main nodes list --limit 20
  python -m cli.main nodes get <node_uuid>
  python -m cli.main nodes add "Node content text" --metadata '{"domain": "ai"}'
  python -m cli.main nodes delete <node_uuid>
  ```
- **Edge Operations**:
  ```bash
  python -m cli.main edges add <source_uuid> <target_uuid> --type derived_from
  python -m cli.main edges list <node_uuid>
  ```
- **Snapshot & Maintenance**:
  ```bash
  python -m cli.main snapshot create --label "manual-backup"
  python -m cli.main stats
  python -m cli.main health
  ```

---

## 2. Interactive Memory Shell (`cli/agent.py`)

Launch an interactive terminal session with memory recall and session lifecycle support:

```bash
python cli/agent.py [options]
```

### Slash Commands

| Command | Action |
|---------|--------|
| `/memory` | Display last recalled memory nodes and scores |
| `/stats` | Call `memory.stats()` and pretty-print storage stats |
| `/sessions` | List all active and archived memory sessions |
| `/archive` | Archive the current session and exit |
| `/forget <text>` | Recall nearest node to `<text>` and soft-delete/forget it |
| `/clear` | Clear terminal output |
| `/help` | Display command list |
| `/exit` or `/quit` | Exit cleanly |

---

## 3. Snapshot Inspector (`cli/mind.py`)

Inspect `.mind` directory manifests, checksums, and SQLite store directly from disk:

```bash
python cli/mind.py inspect path/to/storage.mind
python cli/mind.py verify path/to/storage.mind
```
