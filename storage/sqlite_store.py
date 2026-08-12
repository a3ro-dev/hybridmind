"""
SQLite persistence layer for HybridMind.
Stores nodes and edges with ACID guarantees.
"""

import json
import re
import sqlite3
import struct
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from contextlib import contextmanager
import threading


class SQLiteStore:
    """
    SQLite-based storage for nodes and edges.
    Thread-safe with connection pooling.
    """
    
    def __init__(self, db_path: str = "data/hybridmind.db"):
        """Initialize SQLite store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Optimize for performance
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection.execute("PRAGMA cache_size = -64000")  # 64MB
        return self._local.connection
    
    @contextmanager
    def _cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _init_schema(self):
        """Initialize database schema."""
        with self._cursor() as cursor:
            # Nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    embedding BLOB,
                    raw_embedding BLOB,
                    event_time TEXT DEFAULT NULL,
                    valid_from TEXT DEFAULT NULL,
                    valid_until TEXT DEFAULT NULL,
                    memory_kind TEXT DEFAULT NULL,
                    confidence REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TEXT DEFAULT NULL,
                    archived_at TEXT DEFAULT NULL,
                    archived_by TEXT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP DEFAULT NULL
                )
            """)
            
            # Migrations for existing databases
            try:
                cursor.execute("ALTER TABLE nodes ADD COLUMN deleted_at TIMESTAMP DEFAULT NULL")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE nodes ADD COLUMN raw_embedding BLOB")
            except sqlite3.OperationalError:
                pass
            for col_def in [
                "event_time TEXT DEFAULT NULL",
                "valid_from TEXT DEFAULT NULL",
                "valid_until TEXT DEFAULT NULL",
                "memory_kind TEXT DEFAULT NULL",
                "confidence REAL DEFAULT 1.0",
                "access_count INTEGER DEFAULT 0",
                "last_accessed_at TEXT DEFAULT NULL",
                "archived_at TEXT DEFAULT NULL",
                "archived_by TEXT DEFAULT NULL",
            ]:
                try:
                    cursor.execute(f"ALTER TABLE nodes ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass
            
            # Edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_updated ON nodes(updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_event_time ON nodes(event_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_memory_kind ON nodes(memory_kind)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_archived ON nodes(archived_at)")

            # Normalized entity mentions avoid full-table JSON scans and make
            # entity co-occurrence edges deterministic across ingest paths.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS node_entities (
                    node_id TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    PRIMARY KEY (node_id, entity_key),
                    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_entities_key ON node_entities(entity_key)"
            )

            # Temporal edge fields (Phase 3) — additive migrations safe for existing DBs
            for col_def in [
                "valid_from TEXT DEFAULT NULL",
                "valid_until TEXT DEFAULT NULL",
                "superseded_by TEXT DEFAULT NULL",
                "confidence REAL DEFAULT 1.0",
            ]:
                try:
                    cursor.execute(f"ALTER TABLE edges ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # column already exists
    
    # ==================== Embedding Serialization ====================
    
    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes."""
        if embedding is None:
            return None
        return embedding.astype(np.float32).tobytes()
    
    @staticmethod
    def _deserialize_embedding(data: bytes, dimension: int = 4096) -> Optional[np.ndarray]:
        """Deserialize bytes to numpy array."""
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32)
    
    # ==================== Node Operations ====================
    
    def create_node(
        self,
        node_id: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
        raw_embedding: Optional[np.ndarray] = None,
        event_time: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        memory_kind: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a new node."""
        now = datetime.utcnow()
        embedding_blob = self._serialize_embedding(embedding)
        raw_embedding_blob = self._serialize_embedding(raw_embedding) if raw_embedding is not None else embedding_blob
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO nodes (
                    id, text, metadata, embedding, raw_embedding, event_time,
                    valid_from, valid_until, memory_kind, confidence,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id,
                text,
                json.dumps(metadata),
                embedding_blob,
                raw_embedding_blob,
                event_time,
                valid_from,
                valid_until,
                memory_kind,
                confidence,
                now,
                now
            ))
        
        return {
            "id": node_id,
            "text": text,
            "metadata": metadata,
            "event_time": event_time,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "memory_kind": memory_kind,
            "confidence": confidence,
            "access_count": 0,
            "last_accessed_at": None,
            "archived_at": None,
            "archived_by": None,
            "created_at": now,
            "updated_at": now
        }
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID (ignores soft-deleted nodes)."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, text, metadata, embedding, raw_embedding, event_time,
                       valid_from, valid_until, memory_kind, confidence,
                       access_count, last_accessed_at, archived_at, archived_by,
                       created_at, updated_at
                FROM nodes WHERE id = ? AND deleted_at IS NULL
            """, (node_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "embedding": self._deserialize_embedding(row["embedding"]),
                "raw_embedding": self._deserialize_embedding(row["raw_embedding"]) if "raw_embedding" in row.keys() else None,
                "event_time": row["event_time"],
                "valid_from": row["valid_from"],
                "valid_until": row["valid_until"],
                "memory_kind": row["memory_kind"],
                "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                "access_count": row["access_count"] or 0,
                "last_accessed_at": row["last_accessed_at"],
                "archived_at": row["archived_at"],
                "archived_by": row["archived_by"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            
    def get_latest_node_by_session(
        self,
        session_id: str,
        exclude_node_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the most recently created node for a given session ID."""
        with self._cursor() as cursor:
            # Using JSON1 extension to filter by metadata field
            cursor.execute("""
                SELECT id, metadata, created_at 
                FROM nodes 
                WHERE (
                    json_extract(metadata, '$.sessionId') = ?
                    OR json_extract(metadata, '$.session_id') = ?
                )
                AND (? IS NULL OR id != ?)
                AND deleted_at IS NULL
                ORDER BY created_at DESC LIMIT 1
            """, (session_id, session_id, exclude_node_id, exclude_node_id))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"]
            }
    
    def update_node(
        self,
        node_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        raw_embedding: Optional[np.ndarray] = None,
        event_time: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        memory_kind: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a node."""
        # Get current node
        current = self.get_node(node_id)
        if current is None:
            return None
        
        # Apply updates
        new_text = text if text is not None else current["text"]
        new_metadata = metadata if metadata is not None else current["metadata"]
        new_embedding = embedding if embedding is not None else current["embedding"]
        new_raw_embedding = raw_embedding if raw_embedding is not None else current.get("raw_embedding")
        new_event_time = event_time if event_time is not None else current.get("event_time")
        new_valid_from = valid_from if valid_from is not None else current.get("valid_from")
        new_valid_until = valid_until if valid_until is not None else current.get("valid_until")
        new_memory_kind = memory_kind if memory_kind is not None else current.get("memory_kind")
        new_confidence = confidence if confidence is not None else current.get("confidence", 1.0)
        now = datetime.utcnow()
        
        embedding_blob = self._serialize_embedding(new_embedding)
        raw_embedding_blob = self._serialize_embedding(new_raw_embedding)
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE nodes
                SET text = ?, metadata = ?, embedding = ?, raw_embedding = ?,
                    event_time = ?, valid_from = ?, valid_until = ?,
                    memory_kind = ?, confidence = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
            """, (
                new_text,
                json.dumps(new_metadata),
                embedding_blob,
                raw_embedding_blob,
                new_event_time,
                new_valid_from,
                new_valid_until,
                new_memory_kind,
                new_confidence,
                now,
                node_id
            ))
        
        return {
            "id": node_id,
            "text": new_text,
            "metadata": new_metadata,
            "embedding": new_embedding,
            "event_time": new_event_time,
            "valid_from": new_valid_from,
            "valid_until": new_valid_until,
            "memory_kind": new_memory_kind,
            "confidence": new_confidence,
            "created_at": current["created_at"],
            "updated_at": now
        }
    
    def delete_node(self, node_id: str) -> Tuple[bool, int]:
        """
        Soft delete a node and remove its edges.
        Returns (success, edges_removed).
        """
        with self._cursor() as cursor:
            # Count edges to be removed
            cursor.execute("""
                SELECT COUNT(*) FROM edges
                WHERE source_id = ? OR target_id = ?
            """, (node_id, node_id))
            edges_count = cursor.fetchone()[0]
            
            # Delete edges immediately
            cursor.execute("""
                DELETE FROM edges WHERE source_id = ? OR target_id = ?
            """, (node_id, node_id))
            
            # Soft delete node
            cursor.execute("UPDATE nodes SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?", (node_id,))
            deleted = cursor.rowcount > 0

        return deleted, edges_count

    def soft_delete_node(self, node_id: str) -> bool:
        """Soft delete a node and its incident edges."""
        deleted, _ = self.delete_node(node_id)
        return deleted
        
    def get_deleted_nodes_count(self) -> int:
        """Get the number of soft-deleted nodes waiting for compaction."""
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM nodes WHERE deleted_at IS NOT NULL")
            return cursor.fetchone()[0]
            
    def hard_delete_soft_deleted_nodes(self) -> int:
        """Permanently delete all soft-deleted nodes."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM nodes WHERE deleted_at IS NOT NULL")
            return cursor.rowcount
    
    def list_nodes(
        self,
        skip: int = 0,
        limit: int = 100,
        include_embeddings: bool = False,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """List current active nodes with pagination."""
        with self._cursor() as cursor:
            if include_embeddings:
                cursor.execute("""
                    SELECT id, text, metadata, embedding, event_time, valid_from,
                           valid_until, memory_kind, confidence, access_count,
                           last_accessed_at, archived_at, archived_by,
                           created_at, updated_at
                    FROM nodes
                    WHERE deleted_at IS NULL AND (? OR archived_at IS NULL)
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (include_archived, limit, skip))
            else:
                cursor.execute("""
                    SELECT id, text, metadata, event_time, valid_from,
                           valid_until, memory_kind, confidence, access_count,
                           last_accessed_at, archived_at, archived_by,
                           created_at, updated_at
                    FROM nodes
                    WHERE deleted_at IS NULL AND (? OR archived_at IS NULL)
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (include_archived, limit, skip))
            
            nodes = []
            for row in cursor.fetchall():
                node = {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]),
                    "event_time": row["event_time"],
                    "valid_from": row["valid_from"],
                    "valid_until": row["valid_until"],
                    "memory_kind": row["memory_kind"],
                    "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                    "access_count": row["access_count"] or 0,
                    "last_accessed_at": row["last_accessed_at"],
                    "archived_at": row["archived_at"],
                    "archived_by": row["archived_by"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                if include_embeddings and "embedding" in row.keys():
                    node["embedding"] = self._deserialize_embedding(row["embedding"])
                nodes.append(node)
            
            return nodes
    
    def get_all_node_embeddings(self, include_archived: bool = False) -> List[Tuple[str, np.ndarray]]:
        """Get retrievable node IDs and embeddings for vector index rebuild."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, embedding FROM nodes
                WHERE embedding IS NOT NULL AND deleted_at IS NULL
                  AND (? OR archived_at IS NULL)
            """, (include_archived,))
            
            results = []
            for row in cursor.fetchall():
                embedding = self._deserialize_embedding(row["embedding"])
                if embedding is not None:
                    results.append((row["id"], embedding))
            
            return results
    
    def count_nodes(self) -> int:
        """Get total active node count."""
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM nodes WHERE deleted_at IS NULL")
            return cursor.fetchone()[0]

    def count_retrievable_nodes(self) -> int:
        """Return active nodes that participate in retrieval indexes."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM nodes WHERE deleted_at IS NULL AND archived_at IS NULL"
            )
            return cursor.fetchone()[0]

    def is_node_retrievable(self, node_id: str) -> bool:
        with self._cursor() as cursor:
            cursor.execute(
                """SELECT 1 FROM nodes
                   WHERE id = ? AND deleted_at IS NULL AND archived_at IS NULL""",
                (node_id,),
            )
            return cursor.fetchone() is not None

    # ==================== Structured Metadata ====================

    @staticmethod
    def canonicalize_entity(entity: str) -> str:
        value = unicodedata.normalize("NFKC", str(entity)).casefold().strip()
        value = re.sub(r"[^\w\s-]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def upsert_node_entities(
        self,
        node_id: str,
        entities: List[Any],
        confidence: float = 1.0,
    ) -> int:
        """Replace normalized entity mentions for a node."""
        normalized: Dict[str, str] = {}
        for item in entities:
            name = item.get("name", "") if isinstance(item, dict) else str(item)
            key = self.canonicalize_entity(name)
            if key:
                normalized.setdefault(key, name.strip())

        with self._cursor() as cursor:
            cursor.execute("DELETE FROM node_entities WHERE node_id = ?", (node_id,))
            cursor.executemany(
                """INSERT INTO node_entities (node_id, entity_key, entity_name, confidence)
                   VALUES (?, ?, ?, ?)""",
                [(node_id, key, name, confidence) for key, name in sorted(normalized.items())],
            )
        return len(normalized)

    def get_node_entities(self, node_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute(
                """SELECT entity_key, entity_name, confidence
                   FROM node_entities WHERE node_id = ? ORDER BY entity_key""",
                (node_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_nodes_by_entity(
        self,
        entity: str,
        exclude_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[str]:
        key = self.canonicalize_entity(entity)
        if not key:
            return []
        with self._cursor() as cursor:
            cursor.execute(
                """SELECT ne.node_id
                   FROM node_entities AS ne
                   JOIN nodes AS n ON n.id = ne.node_id
                   WHERE ne.entity_key = ?
                     AND (? IS NULL OR ne.node_id != ?)
                     AND n.deleted_at IS NULL AND n.archived_at IS NULL
                   ORDER BY n.created_at DESC, ne.node_id
                   LIMIT ?""",
                (key, exclude_id, exclude_id, limit),
            )
            return [row["node_id"] for row in cursor.fetchall()]

    def find_temporal_neighbors(
        self,
        event_time: str,
        exclude_id: Optional[str] = None,
        window_days: float = 30.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return retrievable nodes nearest to an ISO event timestamp."""
        with self._cursor() as cursor:
            cursor.execute(
                """SELECT id, event_time,
                          ABS(julianday(event_time) - julianday(?)) AS delta_days
                   FROM nodes
                   WHERE event_time IS NOT NULL
                     AND deleted_at IS NULL AND archived_at IS NULL
                     AND (? IS NULL OR id != ?)
                     AND ABS(julianday(event_time) - julianday(?)) <= ?
                   ORDER BY delta_days, id
                   LIMIT ?""",
                (event_time, exclude_id, exclude_id, event_time, window_days, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Access and Compression State ====================

    def record_access(self, node_ids: List[str], accessed_at: Optional[str] = None) -> int:
        """Atomically increment access statistics once per unique node."""
        unique_ids = list(dict.fromkeys(node_ids))
        if not unique_ids:
            return 0
        timestamp = accessed_at or datetime.now(timezone.utc).isoformat()
        updated = 0
        with self._cursor() as cursor:
            for offset in range(0, len(unique_ids), 500):
                batch = unique_ids[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                cursor.execute(
                    f"""UPDATE nodes
                        SET access_count = access_count + 1,
                            last_accessed_at = ?,
                            updated_at = updated_at
                        WHERE id IN ({placeholders})
                          AND deleted_at IS NULL AND archived_at IS NULL""",
                    (timestamp, *batch),
                )
                updated += cursor.rowcount
        return updated

    def archive_nodes(self, node_ids: List[str], archived_by: str) -> int:
        """Remove sources from retrieval without deleting provenance rows/edges."""
        unique_ids = [node_id for node_id in dict.fromkeys(node_ids) if node_id != archived_by]
        if not unique_ids:
            return 0
        timestamp = datetime.now(timezone.utc).isoformat()
        updated = 0
        with self._cursor() as cursor:
            for offset in range(0, len(unique_ids), 500):
                batch = unique_ids[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                cursor.execute(
                    f"""UPDATE nodes SET archived_at = ?, archived_by = ?
                        WHERE id IN ({placeholders}) AND deleted_at IS NULL""",
                    (timestamp, archived_by, *batch),
                )
                updated += cursor.rowcount
        return updated
    
    # ==================== Edge Operations ====================
    
    def create_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        superseded_by: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a new edge."""
        now = datetime.utcnow()
        metadata = metadata or {}

        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO edges
                  (id, source_id, target_id, type, weight, metadata, created_at,
                   valid_from, valid_until, superseded_by, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                source_id,
                target_id,
                edge_type,
                weight,
                json.dumps(metadata),
                now,
                valid_from,
                valid_until,
                superseded_by,
                confidence,
            ))

        return {
            "id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "weight": weight,
            "metadata": metadata,
            "created_at": now,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "superseded_by": superseded_by,
            "confidence": confidence,
        }
    
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by ID."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, source_id, target_id, type, weight, metadata, created_at,
                       valid_from, valid_until, superseded_by, confidence
                FROM edges WHERE id = ?
            """, (edge_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "id": row["id"],
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "type": row["type"],
                "weight": row["weight"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "valid_from": row["valid_from"],
                "valid_until": row["valid_until"],
                "superseded_by": row["superseded_by"],
                "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
            }
    
    def update_edge(
        self,
        edge_id: str,
        edge_type: Optional[str] = None,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_until: Optional[str] = None,
        superseded_by: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an edge."""
        current = self.get_edge(edge_id)
        if current is None:
            return None

        new_type = edge_type if edge_type is not None else current["type"]
        new_weight = weight if weight is not None else current["weight"]
        new_metadata = metadata if metadata is not None else current["metadata"]
        new_valid_until = valid_until if valid_until is not None else current.get("valid_until")
        new_superseded_by = superseded_by if superseded_by is not None else current.get("superseded_by")
        new_confidence = confidence if confidence is not None else current.get("confidence", 1.0)

        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE edges
                SET type = ?, weight = ?, metadata = ?,
                    valid_until = ?, superseded_by = ?, confidence = ?
                WHERE id = ?
            """, (new_type, new_weight, json.dumps(new_metadata),
                  new_valid_until, new_superseded_by, new_confidence, edge_id))

        return {
            "id": edge_id,
            "source_id": current["source_id"],
            "target_id": current["target_id"],
            "type": new_type,
            "weight": new_weight,
            "metadata": new_metadata,
            "created_at": current["created_at"],
            "valid_from": current.get("valid_from"),
            "valid_until": new_valid_until,
            "superseded_by": new_superseded_by,
            "confidence": new_confidence,
        }
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            return cursor.rowcount > 0
    
    def get_node_edges(
        self,
        node_id: str,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get all edges connected to a node."""
        with self._cursor() as cursor:
            if direction == "outgoing":
                cursor.execute("""
                    SELECT id, source_id, target_id, type, weight, metadata, created_at,
                           valid_from, valid_until, superseded_by, confidence
                    FROM edges WHERE source_id = ?
                """, (node_id,))
            elif direction == "incoming":
                cursor.execute("""
                    SELECT id, source_id, target_id, type, weight, metadata, created_at,
                           valid_from, valid_until, superseded_by, confidence
                    FROM edges WHERE target_id = ?
                """, (node_id,))
            else:  # both
                cursor.execute("""
                    SELECT id, source_id, target_id, type, weight, metadata, created_at,
                           valid_from, valid_until, superseded_by, confidence
                    FROM edges WHERE source_id = ? OR target_id = ?
                """, (node_id, node_id))

            edges = []
            for row in cursor.fetchall():
                edges.append({
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type": row["type"],
                    "weight": row["weight"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                    "valid_from": row["valid_from"],
                    "valid_until": row["valid_until"],
                    "superseded_by": row["superseded_by"],
                    "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                })

            return edges
    
    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges for graph index rebuild."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, source_id, target_id, type, weight, metadata, created_at,
                       valid_from, valid_until, superseded_by, confidence
                FROM edges
            """)

            edges = []
            for row in cursor.fetchall():
                edges.append({
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type": row["type"],
                    "weight": row["weight"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                    "valid_from": row["valid_from"],
                    "valid_until": row["valid_until"],
                    "superseded_by": row["superseded_by"],
                    "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                })

            return edges
    
    def count_edges(self) -> int:
        """Get total edge count."""
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM edges")
            return cursor.fetchone()[0]
    
    def get_edge_type_counts(self) -> Dict[str, int]:
        """Get counts by edge type."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT type, COUNT(*) as count
                FROM edges
                GROUP BY type
            """)
            return {row["type"]: row["count"] for row in cursor.fetchall()}
    
    # ==================== Utility Operations ====================
    
    def get_database_size(self) -> int:
        """Get database file size in bytes."""
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0
    
    def vacuum(self):
        """Optimize database by reclaiming space."""
        conn = self._get_connection()
        conn.execute("VACUUM")
    
    def close(self):
        """Close all connections."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

