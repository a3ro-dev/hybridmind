"""Focused offline tests for the immutable claim-ledger builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_claim_ledger import (
    ClaimLedgerValidationError,
    build_claim_ledger,
    render_markdown,
    write_ledger_outputs,
)


def _spec() -> dict:
    return {
        "schema_version": "hybridmind.claims-spec/v1",
        "claims": [
            {
                "claim_id": "fixture-measured",
                "title": "Fixture measured result",
                "status": "measured",
                "required_artifact": True,
                "artifact_schema": "fixture/v1",
                "schema_path": ["schema_version"],
                "classification": "fixture_result",
                "status_rule": {"kind": "absent", "path": ["status"]},
                "scope": "fixture corpus; held-out exact evidence IDs",
                "metric": "recall_at_10",
                "value_path": ["metrics", "recall_at_10"],
                "claim_boundary": ["Fixture only; not answer accuracy."],
            },
            {
                "claim_id": "fixture-invalidated",
                "title": "Fixture failed receipt",
                "status": "invalidated",
                "required_artifact": True,
                "artifact_schema": "fixture-failure/v1",
                "schema_path": ["schema_version"],
                "classification": "fixture_failure",
                "status_rule": {"kind": "equals", "path": ["status"], "value": "failed"},
                "failure_required": True,
                "scope": "fixture admission",
                "metric": None,
                "claim_boundary": ["Failed receipt is not a result."],
            },
            {
                "claim_id": "fixture-open",
                "title": "Fixture future experiment",
                "status": "open",
                "required_artifact": False,
                "artifact_schema": None,
                "scope": "future fixture corpus",
                "metric": "recall_at_10",
                "value_path": None,
                "claim_boundary": ["Open until an immutable artifact exists."],
            },
        ],
    }


def _write_artifact(path: Path, *, schema: str, classification: str, value: float | None = None, status: str | None = None) -> Path:
    document = {
        "schema_version": schema,
        "classification": classification,
        "execution": {
            "external_network_calls": 0,
            "provider_calls": 0,
            "embedding_calls": 0,
            "reranker_calls": 0,
            "reader_calls": 0,
        },
        "claim_boundary": ["Artifact-local boundary."],
    }
    if value is not None:
        document["metrics"] = {"recall_at_10": value}
    if status is not None:
        document["status"] = status
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def test_builder_is_deterministic_and_emits_open_claims(tmp_path: Path) -> None:
    measured = _write_artifact(tmp_path / "measured.json", schema="fixture/v1", classification="fixture_result", value=0.75)
    failed = _write_artifact(tmp_path / "receipt.json", schema="fixture-failure/v1", classification="fixture_failure", status="failed")

    first = build_claim_ledger(
        {"fixture-measured": measured, "fixture-invalidated": failed},
        spec=_spec(),
    )
    second = build_claim_ledger(
        {"fixture-measured": measured, "fixture-invalidated": failed},
        spec=_spec(),
    )

    assert first == second
    rows = {row["claim_id"]: row for row in first["claims"]}
    assert rows["fixture-measured"]["status"] == "measured"
    assert rows["fixture-measured"]["value"] == 0.75
    assert rows["fixture-measured"]["artifact_sha256"] == hashlib.sha256(measured.read_bytes()).hexdigest()
    assert rows["fixture-invalidated"]["status"] == "invalidated"
    assert rows["fixture-invalidated"]["execution"]["provider_calls"] == 0
    assert rows["fixture-open"]["artifact"] is None
    assert "fixture-open" in render_markdown(first)


def test_builder_validates_and_aggregates_explicit_metric_paths(tmp_path: Path) -> None:
    spec = _spec()
    measured_spec = spec["claims"][0]
    measured_spec.pop("value_path")
    measured_spec["value_paths"] = [["metrics", "seed_values", 0], ["metrics", "seed_values", 1]]
    measured_spec["aggregation"] = "mean"
    measured = _write_artifact(tmp_path / "measured.json", schema="fixture/v1", classification="fixture_result")
    document = json.loads(measured.read_text(encoding="utf-8"))
    document["metrics"] = {"seed_values": [0.5, 0.75]}
    measured.write_text(json.dumps(document), encoding="utf-8")
    failed = _write_artifact(tmp_path / "receipt.json", schema="fixture-failure/v1", classification="fixture_failure", status="failed")
    ledger = build_claim_ledger({"fixture-measured": measured, "fixture-invalidated": failed}, spec=spec)
    row = next(item for item in ledger["claims"] if item["claim_id"] == "fixture-measured")
    assert row["value"] == pytest.approx(0.625)
    assert row["component_values"] == [0.5, 0.75]


def test_builder_rejects_missing_or_malformed_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ClaimLedgerValidationError, match="required artifact missing"):
        build_claim_ledger({}, spec=_spec())

    measured = _write_artifact(tmp_path / "named-invalidated.json", schema="fixture/v1", classification="fixture_result", value=0.75, status="failed")
    failed = _write_artifact(tmp_path / "valid-receipt.json", schema="fixture-failure/v1", classification="fixture_failure", status="failed")
    with pytest.raises(ClaimLedgerValidationError, match="status rule failed"):
        build_claim_ledger({"fixture-measured": measured, "fixture-invalidated": failed}, spec=_spec())

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ClaimLedgerValidationError, match="not valid UTF-8 JSON"):
        build_claim_ledger({"fixture-measured": malformed, "fixture-invalidated": failed}, spec=_spec())


def test_outputs_are_create_once(tmp_path: Path) -> None:
    measured = _write_artifact(tmp_path / "measured.json", schema="fixture/v1", classification="fixture_result", value=0.75)
    failed = _write_artifact(tmp_path / "receipt.json", schema="fixture-failure/v1", classification="fixture_failure", status="failed")
    ledger = build_claim_ledger({"fixture-measured": measured, "fixture-invalidated": failed}, spec=_spec())
    json_path = tmp_path / "ledger.json"
    markdown_path = tmp_path / "ledger.md"
    write_ledger_outputs(ledger, json_path, markdown_path)
    original = json_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_ledger_outputs(ledger, json_path, markdown_path)
    assert json_path.read_bytes() == original
