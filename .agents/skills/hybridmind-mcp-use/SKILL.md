---
name: hybridmind-mcp-use
description: >
  Guide for AI agents/assistants on how to use the HybridMind MCP memory server tools.
  Teaches the AI how to use the remember, recall, relate, and forget tools to manage
  long-term conversational memory.
---

# Using HybridMind MCP Memory Server

As an AI assistant connected to HybridMind, you have access to four tools to maintain long-term memory across chat sessions.

## Available Tools

1. **`remember(text: str, metadata: dict = None)`**
   - **When to use**: Whenever the user shares a fact about themselves, their preferences, project details, or other important episodic memories.
   - **Metadata recommendation**: Always tag memories with `{"session_id": "current_session_id"}` and a category like `{"type": "user_preference"}` or `{"type": "project_fact"}` to make retrieval cleaner.
   - **Example**: `remember(text="Alice prefers Python over TypeScript for data science scripts", metadata={"session_id": "sess_123", "type": "user_preference"})`

2. **`recall(query: str, top_k: int = 10, mode: str = "hybrid")`**
   - **When to use**: At the beginning of a conversation or when the user asks a question about past interactions or stored facts.
   - **Modes**:
     - `hybrid` (default): Combined vector and graph proximity retrieval. Best for most queries.
     - `vector`: Pure semantic search.
   - **Example**: `recall(query="What are Alice's programming language preferences?")`

3. **`relate(source_id: str, target_id: str, relationship: str = "related_to", weight: float = 1.0)`**
   - **When to use**: When storing a new memory that is directly connected, caused by, or contradicts an existing memory.
   - **Common relationship types**:
     - `caused_by` / `led_to`: Causal chains of events.
     - `contradicts`: When a new memory invalidates or conflicts with an old one.
     - `supports`: Evidentiary links.
     - `supersedes`: When a new memory replaces an old one (e.g. updating an address or preference).
   - **Example**: `relate(source_id="new_node_id", target_id="old_node_id", relationship="supersedes")`

4. **`forget(node_id: str)`**
   - **When to use**: When the user explicitly requests to delete a specific memory, or when a memory is completely superseded and no longer relevant.
   - **Note**: This performs a soft-delete (the node is marked as deleted but stays in the DB until the next compact operation).

---

## Memory Guidelines for the AI

- **Be selective**: Do not store trivial conversational filler (e.g. "Hello", "Thanks"). Store actual entities, preferences, facts, and decisions.
- **Normalize facts**: Before storing, rephrase chat turns into standalone declarative sentences.
  - *Bad*: "He said he went there yesterday."
  - *Good*: "Bob visited the Seattle office on July 2nd, 2026."
- **Check before storing**: Run a quick `recall` query first to see if the fact or a conflicting fact already exists. If a conflict exists, store the new fact and call `relate` with `supersedes` to link it to the old fact.
