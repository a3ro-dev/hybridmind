"""
HybridMind Custom Database Format (.mind)

A .mind directory stores the authoritative SQLite database and manifest. A
portable ``.mind.zip`` snapshot additionally carries validated JSON/JSONL
representations of derived vector, graph, and sparse-index data. Runtime
indexes are rebuilt without deserializing executable pickle payloads.

Structure:
    database.mind/
    ├── manifest.json      # Version, stats, metadata
    ├── store.db           # SQLite database
    └── runtime-derived indexes are rebuilt from SQLite

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
import hmac
import sqlite3
import tempfile
import uuid
import zipfile
from contextlib import closing
from types import SimpleNamespace

def fsync_file(file_obj) -> None:
    """Flush Python and operating-system buffers for an opened file."""
    file_obj.flush()
    os.fsync(file_obj.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry update where the platform exposes directory fsync."""
    if os.name == "nt":
        # Python cannot portably open a Windows directory for FlushFileBuffers.
        # The archive itself is fsynced before ReplaceFile/MoveFileEx publishes it.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

logger = logging.getLogger(__name__)

# File extension
MIND_EXTENSION = ".mind"
MANIFEST_FILE = "manifest.json"
SQLITE_FILE = "store.db"
VECTOR_INDEX_FILE = "vectors"  # Base name - VectorIndex adds .faiss extension
VECTOR_MAP_FILE = "vectors.map"
GRAPH_FILE = "graph.nx"
SNAPSHOT_FORMAT_VERSION = 2
SAFE_SNAPSHOT_FILES = {
    MANIFEST_FILE,
    SQLITE_FILE,
    "vectors.json",
    "graph.jsonl",
    "bm25.jsonl",
}


class MindFile:
    """
    HybridMind database file format (.mind).
    
    A .mind directory contains authoritative SQLite state. Verified snapshot
    archives contain a fixed safe component set for portable recovery.
    
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

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        """Publish JSON without leaving a torn manifest after a crash."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as fh:
                temp_path = Path(fh.name)
                json.dump(payload, fh, indent=2)
                fsync_file(fh)
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def create_snapshot(
        self,
        sqlite_conn: sqlite3.Connection,
        vector_index,
        graph_index,
        nodes_count: int,
        edges_count: int,
        backup_dir: str = "data/backups",
    ) -> Path:
        """
        Create a consistent, immutable and safely serialised snapshot.

        SQLite is the source of truth for every derived index.  The bundle never
        contains Python pickles: graph/BM25/vector metadata are exported as JSON
        and are rebuilt from the verified SQLite database during startup.  The
        complete archive is written under a temporary name and published with
        os.replace(), so callers cannot observe a partial successful snapshot.
        """
        destination = Path(backup_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        archive_name = f"snapshot_{timestamp}.mind.zip"
        final_archive = destination / archive_name

        with tempfile.TemporaryDirectory(prefix=".hybridmind-snapshot-", dir=destination) as tmp:
            tmp_path = Path(tmp)
            bundle = tmp_path / f"{self.name}.mind"
            bundle.mkdir()

            with closing(sqlite3.connect(str(bundle / SQLITE_FILE))) as dst:
                sqlite_conn.backup(dst)
                dst.commit()
            self._validate_sqlite(bundle / SQLITE_FILE)

            # Safe, inspectable descriptions of the derived indexes.  Actual
            # embeddings remain in store.db and indexes are rebuilt from it.
            with closing(sqlite3.connect(str(bundle / SQLITE_FILE))) as snapshot_db:
                snapshot_db.row_factory = sqlite3.Row
                vector_rows = snapshot_db.execute(
                    "SELECT id FROM nodes WHERE deleted_at IS NULL "
                    "AND archived_at IS NULL AND embedding IS NOT NULL ORDER BY id"
                )
                vector_ids = [row["id"] for row in vector_rows]
                with open(bundle / "vectors.json", "w", encoding="utf-8") as fh:
                    json.dump({
                        "dimension": getattr(vector_index, "dimension", None),
                        "node_ids": vector_ids,
                        "rebuild_from": SQLITE_FILE,
                    }, fh, separators=(",", ":"))
                    fsync_file(fh)

                with open(bundle / "graph.jsonl", "w", encoding="utf-8") as fh:
                    graph_rows = list(snapshot_db.execute(
                        "SELECT e.id, e.source_id, e.target_id, e.type, e.weight, "
                        "e.metadata, e.created_at FROM edges e "
                        "JOIN nodes source ON source.id = e.source_id "
                        "JOIN nodes target ON target.id = e.target_id "
                        "WHERE source.deleted_at IS NULL AND source.archived_at IS NULL "
                        "AND target.deleted_at IS NULL AND target.archived_at IS NULL "
                        "ORDER BY e.id"
                    ))
                    for row in graph_rows:
                        fh.write(json.dumps(dict(row), separators=(",", ":"), default=str) + "\n")
                    fsync_file(fh)

                with open(bundle / "bm25.jsonl", "w", encoding="utf-8") as fh:
                    bm25_rows = list(snapshot_db.execute(
                        "SELECT id, text FROM nodes WHERE deleted_at IS NULL AND archived_at IS NULL ORDER BY id"
                    ))
                    for row in bm25_rows:
                        fh.write(json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n")
                    fsync_file(fh)

            checksums = {
                file.name: self._file_sha256(file)
                for file in sorted(bundle.iterdir())
                if file.is_file() and file.name != MANIFEST_FILE
            }
            previous = self.read_manifest() or {}
            manifest = {
                "format": "HybridMind",
                "version": self.VERSION,
                "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
                "name": self.name,
                "created": previous.get("created", datetime.now(timezone.utc).isoformat()),
                "modified": datetime.now(timezone.utc).isoformat(),
                "checksums": checksums,
                "components": {
                    "sqlite": SQLITE_FILE,
                    "vector_index": "vectors.json",
                    "graph": "graph.jsonl",
                    "bm25": "bm25.jsonl",
                },
                "stats": {
                    # Derive statistics from the backed-up transaction, never
                    # from mutable live counters supplied by the caller.
                    "nodes": len(bm25_rows),
                    "edges": len(graph_rows),
                    "vectors": len(vector_ids),
                },
                "metadata": previous.get("metadata", {}),
            }
            with open(bundle / MANIFEST_FILE, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
                fsync_file(fh)
            self.validate_snapshot_directory(bundle)

            temp_archive = destination / f".{archive_name}.{uuid.uuid4().hex}.tmp"
            try:
                with zipfile.ZipFile(temp_archive, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in sorted(bundle.iterdir()):
                        zf.write(file, arcname=f"{bundle.name}/{file.name}")
                with open(temp_archive, "r+b") as fh:
                    fsync_file(fh)
                os.replace(temp_archive, final_archive)
                _fsync_directory(destination)
            finally:
                temp_archive.unlink(missing_ok=True)

        # The live manifest is informational only; integrity belongs to the
        # immutable bundle whose exact files are checksummed above.
        self.update_manifest({
            "latest_snapshot": archive_name,
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "stats": manifest["stats"],
        })
        return final_archive

    @staticmethod
    def _validate_sqlite(db_path: Path) -> None:
        with closing(sqlite3.connect(str(db_path))) as db:
            result = db.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                raise ValueError("snapshot SQLite integrity check failed")
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if not {"nodes", "edges"}.issubset(tables):
                raise ValueError("snapshot SQLite schema is incomplete")
            invalid_embeddings = db.execute(
                "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL "
                "AND length(embedding) != ?",
                (4096 * 4,),
            ).fetchone()[0]
            if invalid_embeddings:
                raise ValueError("snapshot contains a non-4096-dimensional embedding")

    @classmethod
    def validate_snapshot_directory(cls, directory: Path) -> Dict[str, Any]:
        directory = Path(directory)
        if directory.is_symlink() or any(item.is_symlink() for item in directory.iterdir()):
            raise ValueError("snapshot directory cannot contain symbolic links")
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise ValueError("snapshot manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "HybridMind" or manifest.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
            raise ValueError("unsupported or legacy snapshot format")
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict) or set(checksums) != (SAFE_SNAPSHOT_FILES - {MANIFEST_FILE}):
            raise ValueError("snapshot manifest does not name the exact safe component set")
        actual_files = {p.name for p in directory.iterdir() if p.is_file()}
        if actual_files != SAFE_SNAPSHOT_FILES:
            raise ValueError("snapshot contains missing or unexpected files")
        helper = cls(str(directory))
        for name, expected in checksums.items():
            if not isinstance(expected, str) or not hmac.compare_digest(helper._file_sha256(directory / name), expected):
                raise ValueError(f"snapshot checksum mismatch: {name}")
        cls._validate_sqlite(directory / SQLITE_FILE)
        cls._validate_derived_descriptions(directory, manifest)
        return manifest

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL record in {path.name}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"invalid JSONL record in {path.name}:{line_number}")
                rows.append(value)
        return rows

    @classmethod
    def _validate_derived_descriptions(
        cls, directory: Path, manifest: Dict[str, Any]
    ) -> None:
        """Prove that every portable derived-index description matches SQLite."""
        vector_payload = json.loads((directory / "vectors.json").read_text(encoding="utf-8"))
        if not isinstance(vector_payload, dict) or vector_payload.get("dimension") != 4096:
            raise ValueError("snapshot vector description violates the 4096-dimension contract")

        with closing(sqlite3.connect(str(directory / SQLITE_FILE))) as db:
            db.row_factory = sqlite3.Row
            expected_vector_ids = [row["id"] for row in db.execute(
                "SELECT id FROM nodes WHERE deleted_at IS NULL AND archived_at IS NULL "
                "AND embedding IS NOT NULL ORDER BY id"
            )]
            if vector_payload.get("node_ids") != expected_vector_ids:
                raise ValueError("snapshot vector description does not match SQLite")

            expected_graph = [dict(row) for row in db.execute(
                "SELECT e.id, e.source_id, e.target_id, e.type, e.weight, "
                "e.metadata, e.created_at FROM edges e "
                "JOIN nodes source ON source.id = e.source_id "
                "JOIN nodes target ON target.id = e.target_id "
                "WHERE source.deleted_at IS NULL AND source.archived_at IS NULL "
                "AND target.deleted_at IS NULL AND target.archived_at IS NULL "
                "ORDER BY e.id"
            )]
            expected_bm25 = [dict(row) for row in db.execute(
                "SELECT id, text FROM nodes WHERE deleted_at IS NULL "
                "AND archived_at IS NULL ORDER BY id"
            )]

        if cls._read_jsonl(directory / "graph.jsonl") != expected_graph:
            raise ValueError("snapshot graph description does not match SQLite")
        if cls._read_jsonl(directory / "bm25.jsonl") != expected_bm25:
            raise ValueError("snapshot sparse description does not match SQLite")

        stats = manifest.get("stats")
        expected_stats = {
            "nodes": len(expected_bm25),
            "edges": len(expected_graph),
            "vectors": len(expected_vector_ids),
        }
        if stats != expected_stats:
            raise ValueError("snapshot statistics do not match SQLite")
    
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
            
            self._write_json_atomic(self.manifest_path, manifest)
            
            logger.info(f"Created MindFile: {self.path}")
            return True
            
        except Exception as exc:
            logger.error("Failed to create MindFile type=%s", type(exc).__name__)
            return False
    
    def read_manifest(self) -> Optional[Dict[str, Any]]:
        """Read the manifest file."""
        if not self.manifest_path.exists():
            return None
        
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to read manifest type=%s", type(exc).__name__)
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
            
            self._write_json_atomic(self.manifest_path, manifest)
            
            return True
            
        except Exception as exc:
            logger.error("Failed to update manifest type=%s", type(exc).__name__)
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
        if not compress:
            logger.error("Uncompressed live-directory export is disabled")
            return None
        if not self.sqlite_path.is_file():
            logger.error("Export refused: authoritative SQLite database is missing")
            return None

        target = Path(output_path).resolve()
        if not target.name.endswith(".mind.zip"):
            target = Path(f"{target}.mind.zip")
        if target.exists():
            logger.error("Export refused: target already exists")
            return None

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(str(self.sqlite_path))) as source:
                archive = self.create_snapshot(
                    sqlite_conn=source,
                    vector_index=SimpleNamespace(dimension=4096, size=0),
                    graph_index=None,
                    nodes_count=0,
                    edges_count=0,
                    backup_dir=str(target.parent),
                )
            if archive != target:
                os.replace(archive, target)
                _fsync_directory(target.parent)
            self.validate_archive(str(target))
            self.update_manifest({"latest_snapshot": target.name})
            return str(target)
        except Exception as exc:
            target.unlink(missing_ok=True)
            logger.error("Export failed type=%s", type(exc).__name__)
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
        source = Path(archive_path).resolve()
        target = Path(target_path)
        if target.suffix != MIND_EXTENSION:
            target = target.with_suffix(MIND_EXTENSION)
        target = target.resolve()
        if target.exists():
            logger.error("Import refused: target already exists")
            return None

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".hybridmind-import-", dir=target.parent) as tmp:
                stage_root = Path(tmp)
                if source.is_dir():
                    cls.validate_snapshot_directory(source)
                    staged = stage_root / target.name
                    shutil.copytree(source, staged)
                else:
                    with zipfile.ZipFile(source, "r") as zf:
                        members = zf.infolist()
                        files = [info for info in members if not info.is_dir()]
                        if len(files) != len(SAFE_SNAPSHOT_FILES):
                            raise ValueError("snapshot has an unexpected component count")
                        if sum(info.file_size for info in files) > 4 * 1024**3:
                            raise ValueError("snapshot exceeds the import size limit")
                        if any(info.file_size > 2 * 1024**3 for info in files):
                            raise ValueError("snapshot member exceeds size limit")
                        roots = set()
                        for info in members:
                            path = Path(info.filename)
                            if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
                                raise ValueError("unsafe snapshot archive path")
                            roots.add(path.parts[0])
                            if not info.is_dir() and path.parts[1] not in SAFE_SNAPSHOT_FILES:
                                raise ValueError("snapshot contains an unsafe serialized component")
                            unix_mode = info.external_attr >> 16
                            if unix_mode & 0o170000 == 0o120000:
                                raise ValueError("snapshot archive contains a symbolic link")
                        if len(roots) != 1:
                            raise ValueError("snapshot must contain exactly one .mind directory")
                        root = next(iter(roots))
                        if not root.endswith(MIND_EXTENSION):
                            raise ValueError("snapshot root must use the .mind extension")
                        zf.extractall(stage_root)
                        extracted = stage_root / root
                        cls.validate_snapshot_directory(extracted)
                        staged = stage_root / target.name
                        if extracted != staged:
                            os.replace(extracted, staged)
                cls.validate_snapshot_directory(staged)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
            return cls(str(target))
        except Exception as e:
            logger.error("Import rejected (%s)", type(e).__name__)
            return None

    @classmethod
    def validate_archive(cls, archive_path: str) -> Dict[str, Any]:
        """Validate an archive without mutating a live database path."""
        archive = Path(archive_path).resolve()
        with tempfile.TemporaryDirectory(prefix=".hybridmind-verify-", dir=archive.parent) as tmp:
            probe = Path(tmp) / "probe.mind"
            imported = cls.import_from(str(archive), str(probe))
            if imported is None:
                raise ValueError("snapshot archive validation failed")
            return cls.validate_snapshot_directory(imported.path)

    def restore_from_archive(self, archive_path: str) -> bool:
        """Stage and verify a backup before atomically replacing live SQLite.

        Derived indexes are intentionally not restored: startup rebuilds them
        from SQLite.  Replacing the database file inside the existing ``.mind``
        directory avoids the crash window created by renaming the whole live
        directory out of the way before installing its replacement.

        This method is a startup/recovery operation.  It must not be called
        while a live SQLite connection is open; Windows will reject the atomic
        replacement in that case and the original database remains untouched.
        """
        parent = self.path.resolve().parent
        stage = parent / f".{self.path.name}.restore-{uuid.uuid4().hex}.mind"
        imported = self.import_from(archive_path, str(stage))
        if imported is None:
            return False
        try:
            self.validate_snapshot_directory(imported.path)
            self.path.mkdir(parents=True, exist_ok=True)
            staged_db = self.path / f".{SQLITE_FILE}.restore-{uuid.uuid4().hex}.tmp"
            try:
                shutil.copyfile(imported.sqlite_path, staged_db)
                with staged_db.open("r+b") as fh:
                    fsync_file(fh)
                self._validate_sqlite(staged_db)
                os.replace(staged_db, self.sqlite_path)
                _fsync_directory(self.path)
            finally:
                staged_db.unlink(missing_ok=True)

            # A restored database must never be paired with sidecars from the
            # previous SQLite generation.  Recovery runs before connections are
            # opened, so these files are not in use in the supported path.
            for suffix in ("-wal", "-shm", "-journal"):
                Path(f"{self.sqlite_path}{suffix}").unlink(missing_ok=True)
            self.update_manifest({
                "restored_from": Path(archive_path).name,
                "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            })
            return True
        finally:
            if imported.path.exists():
                shutil.rmtree(imported.path, ignore_errors=True)
    
    def delete(self) -> bool:
        """Delete the .mind database."""
        if not self.exists:
            return True
        
        try:
            shutil.rmtree(self.path)
            logger.info(f"Deleted MindFile: {self.path}")
            return True
        except Exception as exc:
            logger.error("Failed to delete MindFile type=%s", type(exc).__name__)
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

