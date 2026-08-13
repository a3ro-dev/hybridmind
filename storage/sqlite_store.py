"""
SQLite persistence layer for HybridMind.
Stores nodes and edges with ACID guarantees.
"""

import json
import math
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


EMBEDDING_DIMENSION = 4096


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
        managed_transaction = bool(getattr(self._local, "transaction_depth", 0))
        try:
            yield cursor
            if not managed_transaction:
                conn.commit()
        except Exception as e:
            if not managed_transaction:
                conn.rollback()
            raise
        finally:
            cursor.close()

    @contextmanager
    def transaction(self):
        """Run nested store operations in one SQLite transaction.

        Store methods normally commit independently for simple callers. API
        workflows that update several rows plus derived indexes use this outer
        boundary so a later failure can roll authoritative SQL state back and
        rebuild the derived indexes from the pre-operation state.
        """
        conn = self._get_connection()
        depth = int(getattr(self._local, "transaction_depth", 0))
        savepoint = f"hybridmind_nested_{depth}" if depth else None
        try:
            if depth == 0:
                conn.execute("BEGIN IMMEDIATE")
            else:
                conn.execute(f"SAVEPOINT {savepoint}")
            self._local.transaction_depth = depth + 1
            yield
            self._local.transaction_depth = depth
            if depth == 0:
                conn.commit()
            else:
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            self._local.transaction_depth = depth
            if depth == 0:
                conn.rollback()
            else:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
    
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
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_session_turn "
                "ON nodes(json_extract(metadata, '$.session_id'), "
                "CAST(json_extract(metadata, '$.turn_index') AS INTEGER))"
            )

            # Immutable bitemporal history. ``valid_*`` describes modeled
            # reality; ``asserted_*`` describes when HybridMind knew a version.
            # Embeddings remain in the current-state table because duplicating
            # 4096-float vectors per assertion would not improve temporal truth.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS node_versions (
                    node_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    event_time TEXT DEFAULT NULL,
                    valid_from TEXT DEFAULT NULL,
                    valid_until TEXT DEFAULT NULL,
                    memory_kind TEXT DEFAULT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    asserted_from TEXT NOT NULL,
                    asserted_until TEXT DEFAULT NULL,
                    operation TEXT NOT NULL,
                    PRIMARY KEY (node_id, version)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_versions_asserted "
                "ON node_versions(node_id, asserted_from, asserted_until)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_versions_valid "
                "ON node_versions(node_id, valid_from, valid_until)"
            )
            cursor.execute("""
                INSERT INTO node_versions (
                    node_id, version, text, metadata, event_time, valid_from,
                    valid_until, memory_kind, confidence, asserted_from,
                    asserted_until, operation
                )
                SELECT n.id, 1, n.text, n.metadata, n.event_time, n.valid_from,
                       n.valid_until, n.memory_kind, COALESCE(n.confidence, 1.0),
                       COALESCE(CAST(n.created_at AS TEXT), CURRENT_TIMESTAMP),
                       CAST(n.deleted_at AS TEXT), 'bootstrap'
                FROM nodes AS n
                WHERE NOT EXISTS (
                    SELECT 1 FROM node_versions AS v WHERE v.node_id = n.id
                )
            """)

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
    def _validate_embedding(embedding: np.ndarray, *, label: str) -> np.ndarray:
        """Validate the storage invariant before a vector crosses the SQL boundary."""
        if embedding is None:
            return None
        array = np.asarray(embedding, dtype=np.float32)
        if array.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"{label} must have shape ({EMBEDDING_DIMENSION},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{label} contains NaN or infinite values")
        return np.ascontiguousarray(array)

    @classmethod
    def _serialize_embedding(cls, embedding: np.ndarray) -> bytes:
        """Serialize one validated native-width float32 embedding."""
        if embedding is None:
            return None
        return cls._validate_embedding(embedding, label="embedding").tobytes()
    
    @classmethod
    def _deserialize_embedding(cls, data: bytes, dimension: int = EMBEDDING_DIMENSION) -> Optional[np.ndarray]:
        """Deserialize and validate persisted vector bytes; corruption fails loudly."""
        if data is None:
            return None
        if dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"HybridMind storage only supports {EMBEDDING_DIMENSION}-dimensional embeddings"
            )
        expected_bytes = EMBEDDING_DIMENSION * np.dtype(np.float32).itemsize
        if len(data) != expected_bytes:
            raise ValueError(
                f"persisted embedding has {len(data)} bytes; expected {expected_bytes}"
            )
        return cls._validate_embedding(
            np.frombuffer(data, dtype=np.float32).copy(),
            label="persisted embedding",
        )
    
    # ==================== Node Operations ====================

    @staticmethod
    def _validate_confidence(value: float, *, label: str) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{label} must be finite and in [0, 1]")
        return confidence

    @staticmethod
    def canonicalize_temporal_value(
        value: Optional[Any], *, label: str
    ) -> Optional[str]:
        """Validate one complete ISO-8601 value and store it canonically in UTC.

        Ingest-time storage must not use the query parser, which deliberately
        recognizes dates embedded in natural-language questions. Persisted
        temporal values are data, so trailing prose and partial matches fail.
        Naive ISO values are interpreted as UTC; date-only values denote UTC
        midnight.
        """
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            text = str(value).strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} must be a complete ISO-8601 value") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat()

    @classmethod
    def normalize_temporal_fields(
        cls,
        *,
        event_time: Optional[Any] = None,
        valid_from: Optional[Any] = None,
        valid_until: Optional[Any] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return canonical event/valid values after half-open interval checks."""
        event = cls.canonicalize_temporal_value(event_time, label="event_time")
        start = cls.canonicalize_temporal_value(valid_from, label="valid_from")
        end = cls.canonicalize_temporal_value(valid_until, label="valid_until")
        if start is not None and end is not None:
            if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
                raise ValueError("valid_from must be earlier than valid_until")
        return event, start, end

    @staticmethod
    def _assertion_time() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _record_node_version(
        cursor: sqlite3.Cursor,
        *,
        node_id: str,
        text: str,
        metadata: Dict[str, Any],
        event_time: Optional[str],
        valid_from: Optional[str],
        valid_until: Optional[str],
        memory_kind: Optional[str],
        confidence: float,
        asserted_from: str,
        operation: str,
    ) -> None:
        cursor.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM node_versions WHERE node_id = ?",
            (node_id,),
        )
        version = int(cursor.fetchone()[0])
        cursor.execute(
            """INSERT INTO node_versions (
                   node_id, version, text, metadata, event_time, valid_from,
                   valid_until, memory_kind, confidence, asserted_from,
                   asserted_until, operation
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
            (
                node_id,
                version,
                text,
                json.dumps(metadata),
                event_time,
                valid_from,
                valid_until,
                memory_kind,
                confidence,
                asserted_from,
                operation,
            ),
        )

    @staticmethod
    def _close_node_versions(
        cursor: sqlite3.Cursor, node_ids: List[str], asserted_until: str
    ) -> None:
        if not node_ids:
            return
        placeholders = ",".join("?" for _ in node_ids)
        cursor.execute(
            f"""UPDATE node_versions SET asserted_until = ?
                WHERE node_id IN ({placeholders}) AND asserted_until IS NULL""",
            (asserted_until, *node_ids),
        )
    
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
        asserted_from = self._assertion_time()
        confidence = self._validate_confidence(confidence, label="node confidence")
        event_time, valid_from, valid_until = self.normalize_temporal_fields(
            event_time=event_time,
            valid_from=valid_from,
            valid_until=valid_until,
        )
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
            self._record_node_version(
                cursor,
                node_id=node_id,
                text=text,
                metadata=metadata,
                event_time=event_time,
                valid_from=valid_from,
                valid_until=valid_until,
                memory_kind=memory_kind,
                confidence=confidence,
                asserted_from=asserted_from,
                operation="create",
            )
        
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

    def get_node_version(
        self,
        node_id: str,
        *,
        valid_at: Optional[str] = None,
        asserted_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the fact version valid and known at the requested times."""
        from engine.temporal import parse_datetime

        valid_reference = parse_datetime(valid_at) if valid_at is not None else None
        asserted_reference = (
            parse_datetime(asserted_at)
            if asserted_at is not None
            else datetime.now(timezone.utc)
        )
        if asserted_reference is None:
            raise ValueError("asserted_at must be an ISO-8601 timestamp")
        if valid_at is not None and valid_reference is None:
            raise ValueError("valid_at must be an ISO-8601 timestamp")
        asserted_iso = asserted_reference.isoformat()
        valid_iso = valid_reference.isoformat() if valid_reference is not None else None
        with self._cursor() as cursor:
            cursor.execute(
                """SELECT node_id, version, text, metadata, event_time,
                          valid_from, valid_until, memory_kind, confidence,
                          asserted_from, asserted_until, operation
                   FROM node_versions
                   WHERE node_id = ?
                     AND asserted_from <= ?
                     AND (asserted_until IS NULL OR ? < asserted_until)
                     AND (
                         ? IS NULL OR (
                             (valid_from IS NULL OR valid_from <= ?)
                             AND (valid_until IS NULL OR ? < valid_until)
                         )
                     )
                   ORDER BY version DESC
                   LIMIT 1""",
                (
                    node_id,
                    asserted_iso,
                    asserted_iso,
                    valid_iso,
                    valid_iso,
                    valid_iso,
                ),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        result["id"] = result.pop("node_id")
        result["metadata"] = json.loads(result["metadata"])
        return result
            
    def get_latest_node_by_session(
        self,
        session_id: str,
        exclude_node_id: Optional[str] = None,
        *,
        container_tag: Optional[str] = None,
        before_turn_index: Optional[int] = None,
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
                AND (
                    (
                        ? IS NULL
                        AND COALESCE(json_extract(metadata, '$.containerTag'), '') = ''
                        AND COALESCE(json_extract(metadata, '$.container_tag'), '') = ''
                    )
                    OR (
                        ? IS NOT NULL
                        AND (
                            json_extract(metadata, '$.containerTag') = ?
                            OR json_extract(metadata, '$.container_tag') = ?
                        )
                    )
                )
                AND (
                    ? IS NULL
                    OR CAST(json_extract(metadata, '$.turn_index') AS INTEGER) < ?
                )
                AND (? IS NULL OR id != ?)
                AND deleted_at IS NULL
                ORDER BY
                    CASE WHEN json_extract(metadata, '$.turn_index') IS NULL THEN 1 ELSE 0 END,
                    CAST(json_extract(metadata, '$.turn_index') AS INTEGER) DESC,
                    created_at DESC,
                    id DESC
                LIMIT 1
            """, (
                session_id,
                session_id,
                container_tag,
                container_tag,
                container_tag,
                container_tag,
                before_turn_index,
                before_turn_index,
                exclude_node_id,
                exclude_node_id,
            ))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"]
            }

    def get_sentence_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """Return live sentence chunks owned by a parent in stable chunk order."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT id, text, metadata, embedding, raw_embedding, event_time,
                       valid_from, valid_until, memory_kind, confidence,
                       created_at, updated_at
                FROM nodes
                WHERE json_extract(metadata, '$.parent_id') = ?
                  AND json_extract(metadata, '$.is_sentence_chunk') = 1
                  AND deleted_at IS NULL
                ORDER BY CAST(COALESCE(json_extract(metadata, '$.chunk_index'), 0) AS INTEGER), id
                """,
                (parent_id,),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "embedding": self._deserialize_embedding(row["embedding"]),
                "raw_embedding": self._deserialize_embedding(row["raw_embedding"]),
                "event_time": row["event_time"],
                "valid_from": row["valid_from"],
                "valid_until": row["valid_until"],
                "memory_kind": row["memory_kind"],
                "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def delete_sentence_children(self, parent_id: str) -> Tuple[List[str], int]:
        """Atomically replaceable-delete derived sentence chunks and their edges."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT id FROM nodes
                WHERE json_extract(metadata, '$.parent_id') = ?
                  AND json_extract(metadata, '$.is_sentence_chunk') = 1
                  AND deleted_at IS NULL
                """,
                (parent_id,),
            )
            node_ids = [row["id"] for row in cursor.fetchall()]
            if not node_ids:
                return [], 0
            self._close_node_versions(cursor, node_ids, self._assertion_time())
            placeholders = ",".join("?" for _ in node_ids)
            cursor.execute(
                f"SELECT COUNT(*) FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                (*node_ids, *node_ids),
            )
            edges_count = int(cursor.fetchone()[0])
            cursor.execute(
                f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                (*node_ids, *node_ids),
            )
            cursor.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", node_ids)
        return node_ids, edges_count

    def delete_node_family(self, node_id: str) -> Tuple[List[str], int]:
        """Atomically soft-delete a parent and its chunks and delete their SQL edges."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT id FROM nodes
                WHERE (
                    id = ? OR (
                        json_extract(metadata, '$.parent_id') = ?
                        AND json_extract(metadata, '$.is_sentence_chunk') = 1
                    )
                )
                AND deleted_at IS NULL
                """,
                (node_id, node_id),
            )
            node_ids = [row["id"] for row in cursor.fetchall()]
            if not node_ids:
                return [], 0
            asserted_until = self._assertion_time()
            self._close_node_versions(cursor, node_ids, asserted_until)
            placeholders = ",".join("?" for _ in node_ids)
            cursor.execute(
                f"SELECT COUNT(*) FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                (*node_ids, *node_ids),
            )
            edges_count = int(cursor.fetchone()[0])
            cursor.execute(
                f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                (*node_ids, *node_ids),
            )
            cursor.execute(
                f"UPDATE nodes SET deleted_at = ? WHERE id IN ({placeholders})",
                (asserted_until, *node_ids),
            )
        return node_ids, edges_count

    def erase_node_family(self, node_id: str) -> Tuple[List[str], int]:
        """Irreversibly erase a node, its chunks, edges, and version history.

        This is the user-facing forget policy. Lifecycle pruning uses
        ``soft_delete_node`` and may retain bitemporal audit history until a
        later hard-delete compaction.
        """
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT id FROM nodes
                WHERE (id = ? OR json_extract(metadata, '$.parent_id') = ?)
                  AND deleted_at IS NULL
                """,
                (node_id, node_id),
            )
            node_ids = [row["id"] for row in cursor.fetchall()]
            if not node_ids:
                return [], 0
            # A derived summary may contain the forgotten text. Without a
            # provider call it cannot be safely rewritten, so erase every
            # provenance descendant that was derived from the forgotten node,
            # along with any sentence chunks owned by those derived nodes.
            erase_ids = set(node_ids)
            while True:
                current = sorted(erase_ids)
                current_placeholders = ",".join("?" for _ in current)
                cursor.execute(
                    f"""SELECT DISTINCT source_id FROM edges
                        WHERE type = 'derived_from'
                          AND target_id IN ({current_placeholders})""",
                    current,
                )
                derived_ids = {str(row["source_id"]) for row in cursor.fetchall()}
                cursor.execute(
                    f"""SELECT id FROM nodes
                        WHERE json_extract(metadata, '$.parent_id')
                              IN ({current_placeholders})""",
                    current,
                )
                child_ids = {str(row["id"]) for row in cursor.fetchall()}
                expanded = erase_ids | derived_ids | child_ids
                if expanded == erase_ids:
                    break
                erase_ids = expanded
            node_ids = sorted(erase_ids)
            placeholders = ",".join("?" for _ in node_ids)
            cursor.execute(
                f"SELECT COUNT(*) FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                (*node_ids, *node_ids),
            )
            edges_count = int(cursor.fetchone()[0])
            cursor.execute(
                f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                (*node_ids, *node_ids),
            )
            cursor.execute(
                f"DELETE FROM node_versions WHERE node_id IN ({placeholders})",
                node_ids,
            )
            cursor.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", node_ids)
        return node_ids, edges_count
    
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
        new_confidence = self._validate_confidence(
            new_confidence, label="node confidence"
        )
        new_event_time, new_valid_from, new_valid_until = self.normalize_temporal_fields(
            event_time=new_event_time,
            valid_from=new_valid_from,
            valid_until=new_valid_until,
        )
        now = datetime.utcnow()
        asserted_from = self._assertion_time()
        
        embedding_blob = self._serialize_embedding(new_embedding)
        raw_embedding_blob = self._serialize_embedding(new_raw_embedding)
        
        with self._cursor() as cursor:
            self._close_node_versions(cursor, [node_id], asserted_from)
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
            if cursor.rowcount != 1:
                raise RuntimeError("Node changed concurrently during versioned update")
            self._record_node_version(
                cursor,
                node_id=node_id,
                text=new_text,
                metadata=new_metadata,
                event_time=new_event_time,
                valid_from=new_valid_from,
                valid_until=new_valid_until,
                memory_kind=new_memory_kind,
                confidence=new_confidence,
                asserted_from=asserted_from,
                operation="update",
            )
        
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
            asserted_until = self._assertion_time()
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
            self._close_node_versions(cursor, [node_id], asserted_until)
            cursor.execute(
                "UPDATE nodes SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                (asserted_until, node_id),
            )
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
        """Permanently erase soft-deleted nodes and their retained history."""
        with self._cursor() as cursor:
            cursor.execute("SELECT id FROM nodes WHERE deleted_at IS NOT NULL")
            node_ids = [str(row["id"]) for row in cursor.fetchall()]
            if not node_ids:
                return 0
            placeholders = ",".join("?" for _ in node_ids)
            cursor.execute(
                f"DELETE FROM node_versions WHERE node_id IN ({placeholders})",
                node_ids,
            )
            cursor.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", node_ids)
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
        if source_id == target_id:
            raise ValueError("Self-loop edges are not supported")
        if not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0:
            raise ValueError("Edge weight must be finite and in [0, 1]")
        confidence = self._validate_confidence(
            confidence, label="edge confidence"
        )
        _unused_event, valid_from, valid_until = self.normalize_temporal_fields(
            valid_from=valid_from,
            valid_until=valid_until,
        )

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
        if not math.isfinite(float(new_weight)) or not 0.0 <= float(new_weight) <= 1.0:
            raise ValueError("Edge weight must be finite and in [0, 1]")
        new_confidence = self._validate_confidence(
            new_confidence, label="edge confidence"
        )
        _unused_event, _normalized_from, new_valid_until = self.normalize_temporal_fields(
            valid_from=current.get("valid_from"),
            valid_until=new_valid_until,
        )

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

