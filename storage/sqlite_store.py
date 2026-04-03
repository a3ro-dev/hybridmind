"""
SQLite persistence layer for HybridMind.
Stores nodes and edges with ACID guarantees.
"""

import json
import sqlite3
import struct
from datetime import datetime
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
                detect_types=sqlite3.PARSE_DECLTYPES,
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
    
    # ==================== Embedding Serialization ====================
    
    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes."""
        if embedding is None:
            return None
        return embedding.astype(np.float32).tobytes()
    
    @staticmethod
    def _deserialize_embedding(data: bytes, dimension: int = 384) -> Optional[np.ndarray]:
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
        raw_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Create a new node."""
        now = datetime.utcnow()
        embedding_blob = self._serialize_embedding(embedding)
        raw_embedding_blob = self._serialize_embedding(raw_embedding) if raw_embedding is not None else embedding_blob
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO nodes (id, text, metadata, embedding, raw_embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                node_id,
                text,
                json.dumps(metadata),
                embedding_blob,
                raw_embedding_blob,
                now,
                now
            ))
        
        return {
            "id": node_id,
            "text": text,
            "metadata": metadata,
            "created_at": now,
            "updated_at": now
        }
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID (ignores soft-deleted nodes)."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, text, metadata, embedding, raw_embedding, created_at, updated_at
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
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            
    def get_latest_node_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recently created node for a given session ID."""
        with self._cursor() as cursor:
            # Using JSON1 extension to filter by metadata field
            cursor.execute("""
                SELECT id, metadata, created_at 
                FROM nodes 
                WHERE json_extract(metadata, '$.sessionId') = ? 
                AND deleted_at IS NULL
                ORDER BY created_at DESC LIMIT 1
            """, (session_id,))
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
        raw_embedding: Optional[np.ndarray] = None
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
        now = datetime.utcnow()
        
        embedding_blob = self._serialize_embedding(new_embedding)
        raw_embedding_blob = self._serialize_embedding(new_raw_embedding)
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE nodes
                SET text = ?, metadata = ?, embedding = ?, raw_embedding = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
            """, (
                new_text,
                json.dumps(new_metadata),
                embedding_blob,
                raw_embedding_blob,
                now,
                node_id
            ))
        
        return {
            "id": node_id,
            "text": new_text,
            "metadata": new_metadata,
            "embedding": new_embedding,
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
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """List current active nodes with pagination."""
        with self._cursor() as cursor:
            if include_embeddings:
                cursor.execute("""
                    SELECT id, text, metadata, embedding, created_at, updated_at
                    FROM nodes WHERE deleted_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, skip))
            else:
                cursor.execute("""
                    SELECT id, text, metadata, created_at, updated_at
                    FROM nodes WHERE deleted_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, skip))
            
            nodes = []
            for row in cursor.fetchall():
                node = {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                if include_embeddings and "embedding" in row.keys():
                    node["embedding"] = self._deserialize_embedding(row["embedding"])
                nodes.append(node)
            
            return nodes
    
    def get_all_node_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Get all active node IDs and embeddings for vector index rebuild."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL AND deleted_at IS NULL
            """)
            
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
    
    # ==================== Edge Operations ====================
    
    def create_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new edge."""
        now = datetime.utcnow()
        metadata = metadata or {}
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO edges (id, source_id, target_id, type, weight, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                source_id,
                target_id,
                edge_type,
                weight,
                json.dumps(metadata),
                now
            ))
        
        return {
            "id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "weight": weight,
            "metadata": metadata,
            "created_at": now
        }
    
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by ID."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, source_id, target_id, type, weight, metadata, created_at
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
                "created_at": row["created_at"]
            }
    
    def update_edge(
        self,
        edge_id: str,
        edge_type: Optional[str] = None,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update an edge."""
        current = self.get_edge(edge_id)
        if current is None:
            return None
        
        new_type = edge_type if edge_type is not None else current["type"]
        new_weight = weight if weight is not None else current["weight"]
        new_metadata = metadata if metadata is not None else current["metadata"]
        
        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE edges
                SET type = ?, weight = ?, metadata = ?
                WHERE id = ?
            """, (new_type, new_weight, json.dumps(new_metadata), edge_id))
        
        return {
            "id": edge_id,
            "source_id": current["source_id"],
            "target_id": current["target_id"],
            "type": new_type,
            "weight": new_weight,
            "metadata": new_metadata,
            "created_at": current["created_at"]
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
                    SELECT id, source_id, target_id, type, weight, metadata, created_at
                    FROM edges WHERE source_id = ?
                """, (node_id,))
            elif direction == "incoming":
                cursor.execute("""
                    SELECT id, source_id, target_id, type, weight, metadata, created_at
                    FROM edges WHERE target_id = ?
                """, (node_id,))
            else:  # both
                cursor.execute("""
                    SELECT id, source_id, target_id, type, weight, metadata, created_at
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
                    "created_at": row["created_at"]
                })
            
            return edges
    
    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges for graph index rebuild."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT id, source_id, target_id, type, weight, metadata, created_at
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
                    "created_at": row["created_at"]
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

