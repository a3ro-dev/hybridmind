"""HybridMemory SDK for HybridMind REST API.

Backward-compatible methods:
- store(), recall(), trace(), relate(), forget(), compact(), stats()

Production additions:
- session.create(), session.recall(), session.archive(), session.list()
- store_batch()
- store_with_auto_edges()
- recall_stream()
- tools.get_schema() for function-calling integrations
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence

import httpx


class HybridMemoryError(RuntimeError):
    """SDK-level error wrapper for API failures."""


@dataclass(frozen=True)
class AutoEdgeConfig:
    """Configuration for automatic edge creation during ingestion."""

    threshold: float = 0.7
    max_edges: int = 5
    allowed_edge_types: Sequence[str] = ("derived_from", "analogous_to")


class SessionAPI:
    """Session namespace: memory.session.*"""

    def __init__(self, memory: "HybridMemory"):
        self._m = memory

    def create(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a scoped session memory space."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        payload_metadata: Dict[str, Any] = {
            HybridMemory.META_SESSION_ID: session_id,
            HybridMemory.META_KIND: HybridMemory.KIND_SESSION_ROOT,
            "session_name": name,
            "session_status": "active",
            "created_via": "sdk.session.create",
        }
        if metadata:
            payload_metadata["session_metadata"] = metadata

        root_id = self._m.store(
            text=f"Session root: {name}",
            metadata=payload_metadata,
        )
        return {
            "session_id": session_id,
            "name": name,
            "root_node_id": root_id,
            "status": "active",
            "metadata": metadata or {},
        }

    def recall(self, query: str, session_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Recall strictly within session context via metadata filtering."""
        return self._m.recall(
            query=query,
            top_k=top_k,
            mode="vector",
            filter_metadata={HybridMemory.META_SESSION_ID: session_id},
        )

    def archive(self, session_id: str) -> Dict[str, Any]:
        """Archive a session by collapsing member nodes into one summary node."""
        nodes = self._m._list_nodes_filtered({HybridMemory.META_SESSION_ID: session_id})
        if not nodes:
            return {
                "session_id": session_id,
                "archived": False,
                "reason": "session not found",
                "archived_nodes": 0,
            }

        snippets: List[str] = []
        for node in nodes[:50]:
            text = (node.get("text") or "").strip()
            if text:
                snippets.append(text[:220])
        summary_text = "\n".join(f"- {s}" for s in snippets[:25]) or "- Session archived with no textual content."

        edges = self._m._list_edges_filtered(node_ids={n["id"] for n in nodes})
        archive_metadata = {
            HybridMemory.META_SESSION_ID: session_id,
            HybridMemory.META_KIND: HybridMemory.KIND_SESSION_SUMMARY,
            "session_status": "archived",
            "archived_at": datetime.utcnow().isoformat() + "Z",
            "archived_node_count": len(nodes),
            "archived_edge_count": len(edges),
            "session_node_ids": [n["id"] for n in nodes],
        }

        summary_id = self._m.store(
            text=(
                f"Archived session {session_id}\n"
                f"Node count: {len(nodes)}\n"
                f"Edge count: {len(edges)}\n"
                f"Summary:\n{summary_text}"
            ),
            metadata=archive_metadata,
        )

        archived_count = 0
        for node in nodes:
            if node["id"] == summary_id:
                continue
            self._m.forget(node["id"])
            archived_count += 1

        return {
            "session_id": session_id,
            "archived": True,
            "summary_node_id": summary_id,
            "archived_nodes": archived_count,
            "retained_nodes": 1,
        }

    def list(self) -> List[Dict[str, Any]]:
        """List all sessions with per-session stats."""
        roots = self._m._list_nodes_filtered(
            {HybridMemory.META_KIND: HybridMemory.KIND_SESSION_ROOT}
        )
        sessions: List[Dict[str, Any]] = []
        for root in roots:
            md = root.get("metadata") or {}
            session_id = md.get(HybridMemory.META_SESSION_ID)
            if not session_id:
                continue
            members = self._m._list_nodes_filtered({HybridMemory.META_SESSION_ID: session_id})
            member_ids = {n["id"] for n in members}
            edges = self._m._list_edges_filtered(node_ids=member_ids)
            sessions.append(
                {
                    "session_id": session_id,
                    "name": md.get("session_name", "unknown"),
                    "status": md.get("session_status", "active"),
                    "root_node_id": root["id"],
                    "node_count": len(members),
                    "edge_count": len(edges),
                    "created_at": root.get("created_at"),
                }
            )
        return sorted(sessions, key=lambda x: x.get("created_at") or "", reverse=True)


class ToolSchemaAPI:
    """Tool schema namespace: memory.tools.get_schema()."""

    def __init__(self, memory: "HybridMemory"):
        self._m = memory

    def get_schema(self) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling schemas for SDK methods."""
        return self._m._build_tool_schemas()


class HybridMemory:
    """Python SDK client for HybridMind API."""

    META_SESSION_ID = "_hm_session_id"
    META_KIND = "_hm_kind"
    KIND_SESSION_ROOT = "session_root"
    KIND_SESSION_SUMMARY = "session_summary"

    EDGE_RELATED_ALIAS = "related_to"
    EDGE_RELATED_CANONICAL = "analogous_to"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=self.timeout)
        self.session = SessionAPI(self)
        self.tools = ToolSchemaAPI(self)

    # ---------- lifecycle ----------

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "HybridMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- internal HTTP helpers ----------

    def _check_response(self, response: httpx.Response) -> None:
        if not (200 <= response.status_code < 300):
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise HybridMemoryError(f"HTTP {response.status_code}: {detail}")

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.client.post(f"{self.base_url}{path}", json=payload or {})
        self._check_response(response)
        return response.json()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        response = self.client.get(f"{self.base_url}{path}", params=params)
        self._check_response(response)
        return response.json()

    def _delete(self, path: str) -> Any:
        response = self.client.delete(f"{self.base_url}{path}")
        self._check_response(response)
        return response.json() if response.content else None

    # ---------- backward-compatible public API ----------

    def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Store a single memory node and return node ID."""
        payload_metadata = dict(metadata or {})
        if session_id:
            payload_metadata[self.META_SESSION_ID] = session_id
        payload = {"text": text, "metadata": payload_metadata}
        result = self._post("/nodes", payload)
        return result["id"]

    def recall(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        filter_metadata: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        anchor_nodes: Optional[List[str]] = None,
        max_depth: int = 2,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Recall relevant nodes via vector or hybrid search.

        If filter_metadata is provided with mode='hybrid', SDK applies
        client-side filtering because /search/hybrid does not expose
        filter_metadata in request schema.
        """
        if mode == "hybrid":
            payload = {
                "query_text": query,
                "top_k": min(max(top_k * 5, top_k), 200) if filter_metadata else min(top_k, 200),
                "vector_weight": vector_weight,
                "graph_weight": graph_weight,
                "anchor_nodes": anchor_nodes,
                "max_depth": max_depth,
                "min_score": min_score,
            }
            result = self._post("/search/hybrid", payload)
            rows = result.get("results", [])
            if filter_metadata:
                rows = [
                    r for r in rows
                    if self._metadata_matches(r.get("metadata") or {}, filter_metadata)
                ]
            return rows[:top_k]

        if mode == "vector":
            payload: Dict[str, Any] = {
                "query_text": query,
                "top_k": min(top_k, 200),
                "min_score": min_score,
            }
            if filter_metadata:
                payload["filter_metadata"] = filter_metadata
            result = self._post("/search/vector", payload)
            return result.get("results", [])

        if mode == "graph":
            raise ValueError("graph mode requires start node id; use trace()")

        raise ValueError(f"Unknown mode: {mode!r}. Use 'hybrid', 'vector', or 'graph'.")

    def trace(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """Find best semantic anchor for concept then graph traverse from it."""
        anchors = self.recall(concept, top_k=1, mode="vector")
        if not anchors:
            return {"anchor": None, "neighbors": []}

        anchor = anchors[0]
        neighbors = self._get(
            "/search/graph",
            params={
                "start_id": anchor["node_id"],
                "depth": depth,
                "direction": "both",
            },
        ).get("results", [])
        return {"anchor": anchor, "neighbors": neighbors}

    def relate(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create directed edge between nodes and return edge ID."""
        edge_type = self._normalize_edge_type(relation_type)
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "weight": weight,
            "metadata": metadata or {},
        }
        result = self._post("/edges", payload)
        return result["id"]

    def forget(self, node_id: str) -> None:
        """Soft-delete memory node."""
        self._delete(f"/nodes/{node_id}")

    def compact(self) -> Dict[str, Any]:
        """Rebuild indexes and hard-delete soft-deleted nodes."""
        return self._post("/admin/compact")

    def stats(self) -> Dict[str, Any]:
        """Return enriched memory stats."""
        base = self._get("/search/stats")
        nodes = self._list_nodes_paginated(limit=500)
        edges = self._list_edges_paginated(limit=1000)

        domain_distribution: Dict[str, int] = {}
        for node in nodes:
            md = node.get("metadata") or {}
            domain = md.get("domain", "unknown")
            domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        degree_by_node: Dict[str, int] = {node["id"]: 0 for node in nodes}
        for edge in edges:
            src = edge.get("source_id")
            tgt = edge.get("target_id")
            if src in degree_by_node:
                degree_by_node[src] += 1
            if tgt in degree_by_node:
                degree_by_node[tgt] += 1

        node_count = int(base.get("total_nodes", len(nodes)))
        edge_count = int(base.get("total_edges", len(edges)))
        avg_degree = (sum(degree_by_node.values()) / node_count) if node_count else 0.0

        degree_sorted = sorted(degree_by_node.items(), key=lambda kv: kv[1], reverse=True)
        top_nodes = []
        node_lookup = {n["id"]: n for n in nodes}
        for node_id, degree in degree_sorted[:10]:
            node = node_lookup.get(node_id, {})
            top_nodes.append(
                {
                    "node_id": node_id,
                    "degree": degree,
                    "text_preview": (node.get("text") or "")[:120],
                    "domain": (node.get("metadata") or {}).get("domain"),
                }
            )

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "edge_types": base.get("edge_types", {}),
            "domain_distribution": domain_distribution,
            "avg_degree": round(avg_degree, 3),
            "most_connected_nodes": top_nodes,
            "vector_index_size": base.get("vector_index_size"),
            "database_size_bytes": base.get("database_size_bytes"),
            "avg_edges_per_node": base.get("avg_edges_per_node"),
        }

    # ---------- new production features ----------

    def store_batch(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store multiple nodes in one call using /bulk/import."""
        if not nodes:
            return {"created_nodes": 0, "created_edges": 0, "node_ids": []}

        prepared_nodes: List[Dict[str, Any]] = []
        prepared_edges: List[Dict[str, Any]] = []
        node_ids: List[str] = []

        for node in nodes:
            if "text" not in node or not node["text"]:
                raise ValueError("Each batch node must include non-empty 'text'")
            node_id = node.get("id") or f"node_{uuid.uuid4().hex[:12]}"
            node_ids.append(node_id)
            prepared_nodes.append(
                {
                    "id": node_id,
                    "text": node["text"],
                    "metadata": node.get("metadata") or {},
                }
            )
            for edge in node.get("edges") or []:
                target_id = edge.get("target_id")
                if not target_id:
                    raise ValueError("Batch edge missing required 'target_id'")
                prepared_edges.append(
                    {
                        "source_id": node_id,
                        "target_id": target_id,
                        "type": self._normalize_edge_type(edge.get("type", "derived_from")),
                        "weight": float(edge.get("weight", 1.0)),
                        "metadata": edge.get("metadata") or {},
                    }
                )

        result = self._post(
            "/bulk/import",
            {
                "nodes": prepared_nodes,
                "edges": prepared_edges,
                "generate_embeddings": True,
            },
        )
        return {
            "node_ids": node_ids,
            "created_nodes": result.get("nodes", {}).get("created", 0),
            "failed_nodes": result.get("nodes", {}).get("failed", 0),
            "created_edges": result.get("edges", {}).get("created", 0),
            "failed_edges": result.get("edges", {}).get("failed", 0),
            "raw": result,
        }

    def store_with_auto_edges(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        auto_edge_config: Optional[AutoEdgeConfig] = None,
    ) -> Dict[str, Any]:
        """Store node and auto-link to nearest neighbors via vector similarity."""
        config = auto_edge_config or AutoEdgeConfig()
        allowed = [self._normalize_edge_type(t) for t in config.allowed_edge_types]
        if not allowed:
            raise ValueError("auto_edge_config.allowed_edge_types cannot be empty")

        node_id = self.store(text=text, metadata=metadata, session_id=session_id)
        candidates = self.recall(
            query=text,
            top_k=max(config.max_edges + 6, 10),
            mode="vector",
        )

        created_edges: List[Dict[str, Any]] = []
        for candidate in candidates:
            target_id = candidate.get("node_id")
            score = float(candidate.get("vector_score") or 0.0)
            if not target_id or target_id == node_id:
                continue
            if score < config.threshold:
                continue
            edge_type = self._select_auto_edge_type(score=score, allowed_edge_types=allowed)
            edge_id = self.relate(
                source_id=node_id,
                target_id=target_id,
                relation_type=edge_type,
                weight=min(1.0, max(0.0, score)),
                metadata={
                    "auto_created": True,
                    "auto_edge_score": score,
                },
            )
            created_edges.append(
                {
                    "edge_id": edge_id,
                    "source_id": node_id,
                    "target_id": target_id,
                    "type": edge_type,
                    "weight": min(1.0, max(0.0, score)),
                }
            )
            if len(created_edges) >= config.max_edges:
                break

        return {
            "node_id": node_id,
            "auto_edges_created": len(created_edges),
            "edges": created_edges,
            "config": {
                "threshold": config.threshold,
                "max_edges": config.max_edges,
                "allowed_edge_types": allowed,
            },
        }

    def recall_stream(
        self,
        query: str,
        top_k: int = 100,
        on_batch_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        batch_size: int = 20,
        mode: str = "hybrid",
        **recall_kwargs: Any,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream recall results in batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        rows = self.recall(query=query, top_k=top_k, mode=mode, **recall_kwargs)
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            if on_batch_callback:
                on_batch_callback(batch)
            yield batch

    # ---------- helper methods ----------

    def _list_nodes_paginated(self, limit: int = 500) -> List[Dict[str, Any]]:
        all_nodes: List[Dict[str, Any]] = []
        skip = 0
        while True:
            page = self._get("/nodes", params={"skip": skip, "limit": limit})
            if not page:
                break
            all_nodes.extend(page)
            if len(page) < limit:
                break
            skip += len(page)
        return all_nodes

    def _list_edges_paginated(self, limit: int = 1000) -> List[Dict[str, Any]]:
        all_edges: List[Dict[str, Any]] = []
        skip = 0
        while True:
            page = self._get("/edges", params={"skip": skip, "limit": limit})
            if not page:
                break
            all_edges.extend(page)
            if len(page) < limit:
                break
            skip += len(page)
        return all_edges

    def _list_nodes_filtered(self, metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = self._list_nodes_paginated()
        return [
            node for node in nodes
            if self._metadata_matches(node.get("metadata") or {}, metadata_filter)
        ]

    def _list_edges_filtered(self, node_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        edges = self._list_edges_paginated()
        if node_ids is None:
            return edges
        return [
            edge for edge in edges
            if edge.get("source_id") in node_ids and edge.get("target_id") in node_ids
        ]

    @staticmethod
    def _metadata_matches(metadata: Dict[str, Any], required: Dict[str, Any]) -> bool:
        for key, value in required.items():
            if metadata.get(key) != value:
                return False
        return True

    def _normalize_edge_type(self, edge_type: str) -> str:
        if edge_type == self.EDGE_RELATED_ALIAS:
            return self.EDGE_RELATED_CANONICAL
        return edge_type

    @staticmethod
    def _select_auto_edge_type(score: float, allowed_edge_types: Sequence[str]) -> str:
        if score >= 0.85 and "derived_from" in allowed_edge_types:
            return "derived_from"
        return allowed_edge_types[0]

    # ---------- tool schemas ----------

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            self._tool_schema(
                "store",
                "Store a single memory node.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "metadata": {"type": "object", "additionalProperties": True},
                        "session_id": {"type": "string"},
                    },
                    "required": ["text"],
                },
                "string (node_id)",
            ),
            self._tool_schema(
                "recall",
                "Recall relevant memory nodes with vector or hybrid search.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 200},
                        "mode": {"type": "string", "enum": ["hybrid", "vector"]},
                        "filter_metadata": {"type": "object", "additionalProperties": True},
                        "vector_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "graph_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "anchor_nodes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["query"],
                },
                "array[SearchResult]",
            ),
            self._tool_schema(
                "store_batch",
                "Store many nodes and optional edges in one bulk call.",
                {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "text": {"type": "string"},
                                    "metadata": {
                                        "type": "object",
                                        "additionalProperties": True,
                                    },
                                    "edges": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "target_id": {"type": "string"},
                                                "type": {"type": "string"},
                                                "weight": {"type": "number"},
                                                "metadata": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                },
                                            },
                                            "required": ["target_id"],
                                        },
                                    },
                                },
                                "required": ["text"],
                            },
                        },
                    },
                    "required": ["nodes"],
                },
                "object",
            ),
            self._tool_schema(
                "store_with_auto_edges",
                "Store one node and auto-create high-similarity edges.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "metadata": {"type": "object", "additionalProperties": True},
                        "session_id": {"type": "string"},
                        "auto_edge_config": {
                            "type": "object",
                            "properties": {
                                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                "max_edges": {"type": "integer", "minimum": 1, "maximum": 20},
                                "allowed_edge_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["text"],
                },
                "object",
            ),
            self._tool_schema(
                "session_create",
                "Create a scoped memory session.",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "metadata": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["name"],
                },
                "object",
            ),
            self._tool_schema(
                "session_recall",
                "Recall within a specific session.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "session_id": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    "required": ["query", "session_id"],
                },
                "array[SearchResult]",
            ),
            self._tool_schema(
                "session_archive",
                "Archive a session into one summary node.",
                {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                    },
                    "required": ["session_id"],
                },
                "object",
            ),
            self._tool_schema(
                "session_list",
                "List sessions and per-session stats.",
                {"type": "object", "properties": {}},
                "array[object]",
            ),
            self._tool_schema(
                "stats",
                "Get enriched memory statistics.",
                {"type": "object", "properties": {}},
                "object",
            ),
        ]

    @staticmethod
    def _tool_schema(
        name: str,
        description: str,
        parameters: Dict[str, Any],
        return_type: str,
    ) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
                "x-return-type": return_type,
            },
        }
