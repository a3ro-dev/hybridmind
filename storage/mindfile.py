"""
HybridMind Custom Database Format (.mind)

A .mind file is a directory-based database format that bundles:
- SQLite database (nodes, edges, metadata)
- FAISS vector index
- NetworkX graph index
- Manifest with metadata

Structure:
    database.mind/
    ├── manifest.json      # Version, stats, metadata
    ├── store.db           # SQLite database
    ├── vectors.faiss      # FAISS index
    ├── vectors.map        # ID mappings for FAISS
    └── graph.nx           # NetworkX graph (pickle)

This creates a portable, self-contained knowledge base.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import hashlib
import sqlite3

try:
    # Use fcntl for fsync on Unix, msvcrt on Windows
    import fcntl
    def fsync_file(file_obj):
        file_obj.flush()
        import os
        os.fsync(file_obj.fileno())
except ImportError:
    import msvcrt
    import os
    def fsync_file(file_obj):
        file_obj.flush()
        os.fsync(file_obj.fileno())

logger = logging.getLogger(__name__)

# File extension
MIND_EXTENSION = ".mind"
MANIFEST_FILE = "manifest.json"
SQLITE_FILE = "store.db"
VECTOR_INDEX_FILE = "vectors"  # Base name - VectorIndex adds .faiss extension
VECTOR_MAP_FILE = "vectors.map"
GRAPH_FILE = "graph.nx"


class MindFile:
    """
    HybridMind database file format (.mind).
    
    A .mind file is a directory containing all database components:
    - SQLite for persistent storage
    - FAISS for vector search
    - NetworkX for graph operations
    
    Usage:
        # Create new database
        db = MindFile("knowledge.mind")
        db.initialize()
        
        # Open existing
        db = MindFile("knowledge.mind")
        paths = db.get_paths()
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, path: str):
        """
        Initialize MindFile handler.
        
        Args:
            path: Path to .mind file (directory)
        """
        # Ensure .mind extension
        if not path.endswith(MIND_EXTENSION):
            path = path + MIND_EXTENSION
        
        self.path = Path(path)
        self.name = self.path.stem
        
    @property
    def exists(self) -> bool:
        """Check if the .mind file exists."""
        return self.path.exists() and self.path.is_dir()
    
    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_FILE
    
    @property
    def sqlite_path(self) -> Path:
        return self.path / SQLITE_FILE
    
    @property
    def vector_index_path(self) -> Path:
        return self.path / VECTOR_INDEX_FILE
    
    @property
    def vector_map_path(self) -> Path:
        return self.path / VECTOR_MAP_FILE
    
    @property
    def graph_path(self) -> Path:
        return self.path / GRAPH_FILE
    
    def get_paths(self) -> Dict[str, str]:
        """Get all component file paths."""
        return {
            "root": str(self.path),
            "manifest": str(self.manifest_path),
            "sqlite": str(self.sqlite_path),
            "vector_index": str(self.vector_index_path),
            "vector_map": str(self.vector_map_path),
            "graph": str(self.graph_path)
        }

    def _file_sha256(self, filepath: Path) -> str:
        """Compute SHA256 of a file."""
        if not filepath.exists():
            return ""
        sha_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha_hash.update(chunk)
        return sha_hash.hexdigest()

    def _checkpoint_wal(self, db_path: Path) -> None:
        """
        BUG-3 fix: Flush SQLite WAL to the main database file before checksumming.

        In WAL mode, SQLite may leave recent changes in the WAL file rather than
        immediately writing them to store.db. Checksumming store.db before a
        checkpoint yields a stale / incorrect hash. PRAGMA wal_checkpoint(FULL)
        ensures all committed WAL frames are written back to the database file.
        """
        try:
            import sqlite3 as _sqlite3
            with _sqlite3.connect(str(db_path)) as _db:
                _db.execute("PRAGMA wal_checkpoint(FULL)")
        except Exception as e:
            logger.warning(f"WAL checkpoint failed (non-fatal): {e}")

    def create_snapshot(
        self,
        sqlite_conn: sqlite3.Connection,
        vector_index,
        graph_index,
        nodes_count: int,
        edges_count: int
    ) -> bool:
        """
        Create atomic crash-safe snapshot of all three storage components.
        Writes to a temporary directory, fsyncs, computes SHA256, and then swaps.
        """
        import time
        from pathlib import Path
        temp_dir = self.path.with_suffix(".tmp_snapshot")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Write SQLite to temp dir using backup API
            temp_sqlite = temp_dir / SQLITE_FILE
            with sqlite3.connect(str(temp_sqlite)) as dst:
                sqlite_conn.backup(dst)
            dst.close()
            
            # 2. Save vector and graph indexes
            vector_index.save(str(temp_dir / VECTOR_INDEX_FILE))
            graph_index.save(str(temp_dir / GRAPH_FILE))
            
            # 3. Compute checksums
            # BUG-3 fix: checkpoint WAL so store.db contains all committed data
            self._checkpoint_wal(temp_sqlite)

            # FAISS adds .faiss to the vector_index path if it's FAISS
            temp_vector_faiss = temp_dir / (VECTOR_INDEX_FILE + ".faiss")
            if not temp_vector_faiss.exists():
                temp_vector_faiss = temp_dir / VECTOR_INDEX_FILE # fallback for numpy
                
            checksums = {
                SQLITE_FILE: self._file_sha256(temp_sqlite),
                GRAPH_FILE: self._file_sha256(temp_dir / GRAPH_FILE),
                VECTOR_INDEX_FILE: self._file_sha256(temp_vector_faiss)
            }
            if (temp_dir / VECTOR_MAP_FILE).exists():
                checksums[VECTOR_MAP_FILE] = self._file_sha256(temp_dir / VECTOR_MAP_FILE)

            # 4. Write new manifest.json
            manifest = self.read_manifest() or {}
            version = manifest.get("snapshot_version", 0) + 1
            
            new_manifest = {
                "format": "HybridMind",
                "version": self.VERSION,
                "name": self.name,
                "created": manifest.get("created", datetime.now(timezone.utc).isoformat()),
                "modified": datetime.now(timezone.utc).isoformat(),
                "snapshot_version": version,
                "checksums": checksums,
                "components": {
                    "sqlite": SQLITE_FILE,
                    "vector_index": VECTOR_INDEX_FILE,
                    "vector_map": VECTOR_MAP_FILE,
                    "graph": GRAPH_FILE
                },
                "stats": {
                    "nodes": nodes_count,
                    "edges": edges_count,
                    "vectors": vector_index.size
                },
                "metadata": manifest.get("metadata", {})
            }
            
            manifest_path = temp_dir / MANIFEST_FILE
            with open(manifest_path, 'w') as f:
                json.dump(new_manifest, f, indent=2)
                fsync_file(f)
            
            # Fsync all files in temp_dir
            for item in temp_dir.iterdir():
                if item.is_file():
                    with open(item, 'r+b') as f:
                        fsync_file(f)
                        
            # 5. Atomically rename/swap the files into the main directory.
            # SQLite's backup API already wrote store.db data atomically via the
            # connection backup above, so we only need to swap non-SQLite files.
            # On Windows, store.db / store.db-wal / store.db-shm are held open by
            # the live connection and cannot be replaced (WinError 32 / OSError).
            SQLITE_LOCKED_PREFIXES = (SQLITE_FILE,)
            for item in temp_dir.iterdir():
                if item.is_file():
                    # Skip SQLite files — they are managed by the live connection
                    if any(item.name.startswith(p) for p in SQLITE_LOCKED_PREFIXES):
                        continue
                    target = self.path / item.name
                    try:
                        os.replace(str(item), str(target))
                    except OSError as e:
                        # Broad OSError catches WinError 32 and PermissionError
                        logger.warning(f"Could not replace {item.name} (skipping): {e}")
                        raise
            
            # Cleanup temp dir
            shutil.rmtree(temp_dir, ignore_errors=True)
            return True
        except Exception as e:
            logger.error(f"Snapshot failed: {e}. Cleaning up temporary directory.")
            # On failure, clean up temp dir and leave previous snapshot intact
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    def initialize(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a new .mind database.
        
        Creates the directory structure and manifest file.
        
        Args:
            metadata: Optional metadata to include in manifest
            
        Returns:
            True if created successfully
        """
        if self.exists:
            logger.warning(f"MindFile already exists: {self.path}")
            return False
        
        try:
            # Create directory
            self.path.mkdir(parents=True, exist_ok=True)
            
            # Create manifest
            manifest = {
                "format": "HybridMind",
                "version": self.VERSION,
                "name": self.name,
                "created": datetime.now(timezone.utc).isoformat(),
                "modified": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "sqlite": SQLITE_FILE,
                    "vector_index": VECTOR_INDEX_FILE,
                    "vector_map": VECTOR_MAP_FILE,
                    "graph": GRAPH_FILE
                },
                "stats": {
                    "nodes": 0,
                    "edges": 0,
                    "vectors": 0
                },
                "metadata": metadata or {}
            }
            
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Created MindFile: {self.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create MindFile: {e}")
            return False
    
    def read_manifest(self) -> Optional[Dict[str, Any]]:
        """Read the manifest file."""
        if not self.manifest_path.exists():
            return None
        
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read manifest: {e}")
            return None
    
    def update_manifest(self, updates: Dict[str, Any]) -> bool:
        """Update manifest with new values."""
        manifest = self.read_manifest()
        if manifest is None:
            return False
        
        try:
            # Deep merge updates
            for key, value in updates.items():
                if isinstance(value, dict) and key in manifest:
                    manifest[key].update(value)
                else:
                    manifest[key] = value
            
            manifest["modified"] = datetime.now(timezone.utc).isoformat()
            
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update manifest: {e}")
            return False
    
    def update_stats(self, nodes: int = None, edges: int = None, vectors: int = None) -> bool:
        """Update database statistics in manifest."""
        stats = {}
        if nodes is not None:
            stats["nodes"] = nodes
        if edges is not None:
            stats["edges"] = edges
        if vectors is not None:
            stats["vectors"] = vectors
        
        if stats:
            return self.update_manifest({"stats": stats})
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get database info including size and stats."""
        manifest = self.read_manifest() or {}
        
        # Calculate size
        total_size = 0
        component_sizes = {}
        
        for name, path in [
            ("sqlite", self.sqlite_path),
            ("vector_index", self.vector_index_path),
            ("graph", self.graph_path)
        ]:
            if path.exists():
                size = path.stat().st_size
                component_sizes[name] = size
                total_size += size
        
        return {
            "path": str(self.path),
            "name": self.name,
            "exists": self.exists,
            "version": manifest.get("version", "unknown"),
            "created": manifest.get("created"),
            "modified": manifest.get("modified"),
            "stats": manifest.get("stats", {}),
            "size_bytes": total_size,
            "size_human": format_size(total_size),
            "component_sizes": component_sizes,
            "metadata": manifest.get("metadata", {})
        }
    
    def export(self, output_path: str, compress: bool = True) -> Optional[str]:
        """
        Export the .mind database to a portable archive.
        
        Args:
            output_path: Path for the exported file
            compress: Whether to compress (creates .mind.zip)
            
        Returns:
            Path to exported file, or None if failed
        """
        if not self.exists:
            logger.error("Cannot export: MindFile does not exist")
            return None
        
        try:
            if compress:
                # Create zip archive
                if not output_path.endswith('.zip'):
                    output_path = output_path + '.zip'
                
                shutil.make_archive(
                    output_path.replace('.zip', ''),
                    'zip',
                    self.path.parent,
                    self.path.name
                )
                logger.info(f"Exported to: {output_path}")
                return output_path
            else:
                # Copy directory
                shutil.copytree(self.path, output_path)
                logger.info(f"Exported to: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    @classmethod
    def import_from(cls, archive_path: str, target_path: str) -> Optional['MindFile']:
        """
        Import a .mind database from an archive.
        
        Args:
            archive_path: Path to .mind.zip or .mind directory
            target_path: Where to extract/copy
            
        Returns:
            MindFile instance, or None if failed
        """
        try:
            if archive_path.endswith('.zip'):
                # Extract zip
                shutil.unpack_archive(archive_path, target_path)
                # Find the .mind directory
                for item in Path(target_path).iterdir():
                    if item.suffix == MIND_EXTENSION:
                        return cls(str(item))
            else:
                # Copy directory
                if not target_path.endswith(MIND_EXTENSION):
                    target_path = target_path + MIND_EXTENSION
                shutil.copytree(archive_path, target_path)
                return cls(target_path)
                
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None
    
    def delete(self) -> bool:
        """Delete the .mind database."""
        if not self.exists:
            return True
        
        try:
            shutil.rmtree(self.path)
            logger.info(f"Deleted MindFile: {self.path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            return False


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def list_mind_files(directory: str = ".") -> list:
    """List all .mind files in a directory."""
    mind_files = []
    for item in Path(directory).iterdir():
        if item.is_dir() and item.suffix == MIND_EXTENSION:
            mf = MindFile(str(item))
            mind_files.append(mf.get_info())
    return mind_files


# Convenience function for creating default database
def create_default_mind(name: str = "hybridmind", data_dir: str = "data") -> MindFile:
    """Create the default HybridMind database."""
    path = os.path.join(data_dir, name)
    mind = MindFile(path)
    
    if not mind.exists:
        mind.initialize(metadata={
            "description": "HybridMind Vector + Graph Database",
            "author": "a3ro-dev"
        })
    
    return mind

