"""Focused offline tests for snapshot integrity and API security boundaries."""

import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
from config import settings
from api.dependencies import DatabaseManager
from storage.mindfile import MindFile, SAFE_SNAPSHOT_FILES


def _source_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, embedding BLOB,
            deleted_at TEXT, archived_at TEXT
        );
        CREATE TABLE edges (
            id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT, type TEXT,
            weight REAL, metadata TEXT, created_at TEXT
        );
        """
    )
    db.execute(
        "INSERT INTO nodes VALUES (?, ?, ?, NULL, NULL)",
        ("n1", "safe text", bytes(4096 * 4)),
    )
    db.commit()
    return db


def test_snapshot_is_atomic_verified_and_pickle_free(tmp_path):
    mind = MindFile(str(tmp_path / "source.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    try:
        archive = mind.create_snapshot(
            sqlite_conn=db,
            vector_index=SimpleNamespace(dimension=4096, size=1),
            graph_index=SimpleNamespace(),
            nodes_count=1,
            edges_count=0,
            backup_dir=str(tmp_path / "backups"),
        )
    finally:
        db.close()

    assert archive.name.startswith("snapshot_")
    assert archive.name.endswith(".mind.zip")
    manifest = MindFile.validate_archive(str(archive))
    assert set(manifest["checksums"]) == SAFE_SNAPSHOT_FILES - {"manifest.json"}
    assert manifest["components"]["bm25"] == "bm25.jsonl"
    with zipfile.ZipFile(archive) as zf:
        names = {Path(name).name for name in zf.namelist()}
    assert names == SAFE_SNAPSHOT_FILES
    assert not any(name.endswith((".pkl", ".nx")) for name in names)


def test_import_rejects_unlisted_pickle_and_preserves_target(tmp_path):
    archive = tmp_path / "unsafe.mind.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("unsafe.mind/manifest.json", json.dumps({}))
        zf.writestr("unsafe.mind/graph.pkl", b"not trusted")
    target = tmp_path / "target.mind"
    assert MindFile.import_from(str(archive), str(target)) is None
    assert not target.exists()


def test_restore_validates_before_replacing_live_directory(tmp_path):
    mind = MindFile(str(tmp_path / "live.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    try:
        valid = mind.create_snapshot(
            sqlite_conn=db,
            vector_index=SimpleNamespace(dimension=4096, size=1),
            graph_index=SimpleNamespace(),
            nodes_count=1,
            edges_count=0,
            backup_dir=str(tmp_path / "backups"),
        )
    finally:
        db.close()

    unsafe = tmp_path / "unsafe.mind.zip"
    with zipfile.ZipFile(unsafe, "w") as zf:
        zf.writestr("unsafe.mind/manifest.json", "{}")
        zf.writestr("unsafe.mind/graph.pkl", b"untrusted")
    before = mind.sqlite_path.read_bytes()
    assert mind.restore_from_archive(str(unsafe)) is False
    assert mind.sqlite_path.read_bytes() == before

    # A validated bundle is staged first, then swapped into place.
    mind.sqlite_path.write_bytes(b"corrupted")
    assert mind.restore_from_archive(str(valid)) is True
    with sqlite3.connect(mind.sqlite_path) as restored:
        assert restored.execute("SELECT text FROM nodes WHERE id='n1'").fetchone()[0] == "safe text"


def test_snapshot_rejects_semantically_forged_derived_index(tmp_path):
    mind = MindFile(str(tmp_path / "source.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    try:
        archive = mind.create_snapshot(
            sqlite_conn=db,
            vector_index=SimpleNamespace(dimension=4096, size=1),
            graph_index=SimpleNamespace(),
            nodes_count=999,
            edges_count=999,
            backup_dir=str(tmp_path / "backups"),
        )
    finally:
        db.close()

    forged = tmp_path / "forged.mind.zip"
    extract = tmp_path / "extract"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract)
    bundle = next(extract.iterdir())
    (bundle / "vectors.json").write_text(
        json.dumps({"dimension": 4096, "node_ids": ["attacker"], "rebuild_from": "store.db"}),
        encoding="utf-8",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib
    manifest["checksums"]["vectors.json"] = hashlib.sha256(
        (bundle / "vectors.json").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with zipfile.ZipFile(forged, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in bundle.iterdir():
            zf.write(path, arcname=f"{bundle.name}/{path.name}")

    # Recomputing a plain checksum is insufficient: the safe description must
    # agree with the authoritative SQLite snapshot.
    try:
        MindFile.validate_archive(str(forged))
    except ValueError as exc:
        assert "validation failed" in str(exc)
    else:
        raise AssertionError("semantically forged snapshot was accepted")


def test_snapshot_publish_failure_leaves_no_claimed_archive(tmp_path, monkeypatch):
    mind = MindFile(str(tmp_path / "source.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    backup_dir = tmp_path / "backups"
    original_replace = __import__("os").replace

    def fail_archive_publish(source, destination):
        if str(destination).endswith(".mind.zip"):
            raise OSError("simulated publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr("storage.mindfile.os.replace", fail_archive_publish)
    try:
        try:
            mind.create_snapshot(
                sqlite_conn=db,
                vector_index=SimpleNamespace(dimension=4096, size=1),
                graph_index=SimpleNamespace(),
                nodes_count=1,
                edges_count=0,
                backup_dir=str(backup_dir),
            )
        except OSError as exc:
            assert "publish failure" in str(exc)
        else:
            raise AssertionError("snapshot failure was swallowed")
    finally:
        db.close()

    assert not list(backup_dir.glob("snapshot_*.mind.zip"))


def test_restore_publish_failure_preserves_live_database(tmp_path, monkeypatch):
    mind = MindFile(str(tmp_path / "live.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    try:
        archive = mind.create_snapshot(
            sqlite_conn=db,
            vector_index=SimpleNamespace(dimension=4096, size=1),
            graph_index=SimpleNamespace(),
            nodes_count=1,
            edges_count=0,
            backup_dir=str(tmp_path / "backups"),
        )
    finally:
        db.close()

    before = mind.sqlite_path.read_bytes()
    original_replace = __import__("os").replace

    def fail_live_publish(source, destination):
        if Path(destination) == mind.sqlite_path:
            raise OSError("simulated restore publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr("storage.mindfile.os.replace", fail_live_publish)
    try:
        mind.restore_from_archive(str(archive))
    except OSError as exc:
        assert "restore publish failure" in str(exc)
    else:
        raise AssertionError("restore failure was swallowed")
    assert mind.sqlite_path.read_bytes() == before


def test_backup_rotation_uses_canonical_names(tmp_path, monkeypatch):
    mind = MindFile(str(tmp_path / "live.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    manager = object.__new__(DatabaseManager)
    manager.mind_file = mind
    manager.sqlite_store = SimpleNamespace(_get_connection=lambda: db)
    manager.vector_index = SimpleNamespace(dimension=4096, size=1)
    manager.graph_index = SimpleNamespace()
    manager.get_stats = lambda: {"total_nodes": 1, "total_edges": 0}
    monkeypatch.setattr(settings, "backup_dir", str(tmp_path / "backups"))
    monkeypatch.setattr(settings, "snapshot_retention", 3)
    try:
        for _ in range(4):
            manager.save_indexes()
    finally:
        db.close()

    backups = sorted((tmp_path / "backups").glob("snapshot_*.mind.zip"))
    assert len(backups) == 3
    assert all(path.name.startswith("snapshot_") for path in backups)


def test_integrity_recovery_skips_corrupt_newest_backup(tmp_path, monkeypatch):
    mind = MindFile(str(tmp_path / "live.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    backup_dir = tmp_path / "backups"
    try:
        valid = mind.create_snapshot(
            sqlite_conn=db,
            vector_index=SimpleNamespace(dimension=4096, size=1),
            graph_index=SimpleNamespace(),
            nodes_count=1,
            edges_count=0,
            backup_dir=str(backup_dir),
        )
    finally:
        db.close()
    (backup_dir / "snapshot_99999999_999999_999999.mind.zip").write_bytes(b"corrupt")
    mind.sqlite_path.write_bytes(b"corrupt live database")
    monkeypatch.setattr(settings, "backup_dir", str(backup_dir))

    assert main.verify_integrity(str(mind.path)) == "PASSED (restored from verified backup)"
    with sqlite3.connect(mind.sqlite_path) as restored:
        assert restored.execute("SELECT text FROM nodes WHERE id='n1'").fetchone()[0] == "safe text"
    assert valid.exists()


def test_direct_export_uses_verified_safe_snapshot_format(tmp_path):
    mind = MindFile(str(tmp_path / "live.mind"))
    assert mind.initialize()
    db = _source_db(mind.sqlite_path)
    db.close()

    exported = mind.export(str(tmp_path / "portable"), compress=True)
    assert exported == str((tmp_path / "portable.mind.zip").resolve())
    manifest = MindFile.validate_archive(exported)
    assert manifest["snapshot_format_version"] == 2


def test_api_key_is_required_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "unit-test-secret")
    client = TestClient(main.app)
    # Public liveness stays available to orchestrators.
    assert client.get("/live").status_code == 200
    assert client.get("/cache/stats").status_code == 401
    response = client.get(
        "/cache/stats", headers={"X-HybridMind-API-Key": "unit-test-secret"}
    )
    assert response.status_code == 200


def test_non_loopback_bind_requires_authentication(monkeypatch):
    monkeypatch.setattr(settings, "host", "0.0.0.0")
    monkeypatch.setattr(settings, "api_key", "")
    try:
        main._validate_api_security_configuration()
    except RuntimeError as exc:
        assert "requires HYBRIDMIND_API_KEY" in str(exc)
    else:
        raise AssertionError("unauthenticated network bind was accepted")


def test_missing_key_fails_closed_when_local_bypass_is_disabled(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "allow_unauthenticated_localhost", False)
    response = TestClient(main.app).get("/cache/stats")
    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication is not configured"}


def test_untrusted_host_and_origin_are_rejected(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    client = TestClient(main.app)
    assert client.get("/live", headers={"Host": "attacker.invalid"}).status_code == 400
    response = client.options(
        "/live",
        headers={
            "Origin": "https://attacker.invalid",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_snapshot_endpoint_does_not_claim_false_success(monkeypatch):
    class BrokenManager:
        def save_indexes(self):
            raise RuntimeError("secret provider response")

    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(main, "get_db_manager", lambda: BrokenManager())
    client = TestClient(main.app)
    response = client.post("/snapshot")
    assert response.status_code == 500
    assert response.json()["message"] == "Snapshot creation failed"
    assert "secret provider response" not in response.text


def test_disabled_fact_extraction_is_reported_not_silent(monkeypatch):
    monkeypatch.setattr(settings, "fact_extraction_enabled", False)
    monkeypatch.setattr(settings, "api_key", "")
    response = TestClient(main.app).post(
        "/ingest/session-facts",
        json={"session_id": "s1", "turns": [{"speaker": "a", "text": "hello"}]},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Fact extraction is disabled by configuration"
