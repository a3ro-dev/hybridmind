"""Build the checked-in retrieval-program claim ledger without side effects.

The builder only reads JSON artifacts named explicitly by the caller.  Claim
status, scope, and metric paths are maintainer declarations in
``claim_ledger_spec.json``; they are never inferred from a filename.  Both
published outputs are create-once files, so an existing ledger is never
silently rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path(__file__).with_name("claim_ledger_spec.json")
SCHEMA = "hybridmind.claim-ledger/v1"
SPEC_SCHEMA = "hybridmind.claims-spec/v1"
STATUSES = {"measured", "rejected", "invalidated", "open"}
CALL_FIELDS = (
    "external_network_calls",
    "provider_calls",
    "embedding_calls",
    "reranker_calls",
    "reader_calls",
)
COUNT_ALIASES = {
    "external_network_calls": ("external_network_calls", "network_calls", "external_calls"),
    "provider_calls": ("provider_calls",),
    "embedding_calls": ("embedding_calls", "remote_embedding_calls"),
    "reranker_calls": ("reranker_calls",),
    "reader_calls": ("reader_calls",),
}
CLAIM_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class ClaimLedgerValidationError(ValueError):
    """Raised when a spec, artifact, or output contract is not admissible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ClaimLedgerValidationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClaimLedgerValidationError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ClaimLedgerValidationError(f"{label} root must be an object: {path}")
    return value


def _at_path(document: Any, path: Sequence[str | int], label: str) -> Any:
    current = document
    for part in path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
        else:
            raise ClaimLedgerValidationError(f"{label} is missing at {path!r}")
    return current


def _optional_path(document: Any, path: Sequence[str | int]) -> tuple[bool, Any]:
    current = document
    for part in path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
        else:
            return False, None
    return True, current


def _finite_number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ClaimLedgerValidationError(f"{label} must be a finite number")
    return value


def _nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClaimLedgerValidationError(f"{label} must be a non-negative integer")
    return value


def _validate_spec_document(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ClaimLedgerValidationError(f"claims spec must use {SPEC_SCHEMA}")
    claims = spec.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ClaimLedgerValidationError("claims spec must contain a non-empty claims list")
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise ClaimLedgerValidationError("each claim spec entry must be an object")
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not CLAIM_ID_RE.fullmatch(claim_id) or claim_id in seen:
            raise ClaimLedgerValidationError(f"claim_id is missing, unstable, or duplicated: {claim_id!r}")
        seen.add(claim_id)
        if claim.get("status") not in STATUSES:
            raise ClaimLedgerValidationError(f"{claim_id}: invalid claim status")
        for field in ("title", "scope"):
            if not isinstance(claim.get(field), str) or not claim[field].strip():
                raise ClaimLedgerValidationError(f"{claim_id}: {field} must be non-empty")
        boundary = claim.get("claim_boundary")
        if not isinstance(boundary, list) or not boundary or any(not isinstance(item, str) or not item.strip() for item in boundary):
            raise ClaimLedgerValidationError(f"{claim_id}: claim_boundary must be non-empty strings")
        required = claim.get("required_artifact")
        if not isinstance(required, bool):
            raise ClaimLedgerValidationError(f"{claim_id}: required_artifact must be boolean")
        if required and claim["status"] == "open":
            raise ClaimLedgerValidationError(f"{claim_id}: open claims cannot require an artifact")
        if claim.get("value_path") is not None and not isinstance(claim["value_path"], list):
            raise ClaimLedgerValidationError(f"{claim_id}: value_path must be a list or null")
        value_paths = claim.get("value_paths")
        if value_paths is not None and (
            not isinstance(value_paths, list)
            or not value_paths
            or any(not isinstance(path, list) or not path for path in value_paths)
        ):
            raise ClaimLedgerValidationError(f"{claim_id}: value_paths must be a non-empty list of paths")
        if claim.get("value_path") is not None and value_paths is not None:
            raise ClaimLedgerValidationError(f"{claim_id}: use value_path or value_paths, not both")
        if value_paths is not None and claim.get("aggregation") != "mean":
            raise ClaimLedgerValidationError(f"{claim_id}: value_paths currently require aggregation=mean")
        expected_hash = claim.get("artifact_sha256")
        if expected_hash is not None and (not isinstance(expected_hash, str) or not HASH_RE.fullmatch(expected_hash)):
            raise ClaimLedgerValidationError(f"{claim_id}: artifact_sha256 must be lowercase SHA-256")
        defaults = claim.get("execution_defaults")
        if defaults is not None:
            if not isinstance(defaults, dict):
                raise ClaimLedgerValidationError(f"{claim_id}: execution_defaults must be an object")
            for field, value in defaults.items():
                if field not in CALL_FIELDS or (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ClaimLedgerValidationError(f"{claim_id}: invalid execution default {field}")
        if claim.get("status_rule") is not None and not isinstance(claim["status_rule"], dict):
            raise ClaimLedgerValidationError(f"{claim_id}: status_rule must be an object")
    return spec


def load_claims_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    return _validate_spec_document(_read_json(Path(path).resolve(), "claims spec"))


def _resolve_input_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _relative_artifact_path(path: Path) -> str:
    try:
        return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()
    except ValueError:
        # Unit fixtures may live on a different Windows volume. Keep the
        # recorded path relative (and deterministic) without pretending it is
        # inside the repository.
        drive = path.drive.rstrip(":").lower() or "unknown-drive"
        return (Path("external") / drive / path.relative_to(path.anchor)).as_posix()


def _validate_status_rule(document: dict[str, Any], claim: dict[str, Any], path: Path) -> None:
    rule = claim.get("status_rule")
    if not isinstance(rule, dict):
        raise ClaimLedgerValidationError(f"{claim['claim_id']}: status_rule is required")
    rule_path = rule.get("path")
    if not isinstance(rule_path, list) or not rule_path:
        raise ClaimLedgerValidationError(f"{claim['claim_id']}: status_rule.path is required")
    present, value = _optional_path(document, rule_path)
    kind = rule.get("kind")
    if kind == "absent":
        valid = not present
    elif kind == "equals":
        valid = present and value == rule.get("value")
    elif kind == "absent_or":
        values = rule.get("values")
        valid = (not present) or (isinstance(values, list) and value in values)
    else:
        raise ClaimLedgerValidationError(f"{claim['claim_id']}: unsupported status_rule kind")
    if not valid:
        raise ClaimLedgerValidationError(f"{path}: status rule failed for claim {claim['claim_id']}")
    if claim.get("failure_required") and (not present or value != "failed"):
        raise ClaimLedgerValidationError(f"{path}: failed status is required for {claim['claim_id']}")


def _extract_call_counts(document: dict[str, Any], claim: dict[str, Any], path: Path) -> dict[str, int | float]:
    execution = document.get("execution")
    if execution is not None and not isinstance(execution, dict):
        raise ClaimLedgerValidationError(f"{path}: execution must be an object")
    containers = [container for container in (execution, document) if isinstance(container, dict)]
    values: dict[str, int | float] = {}
    # Resolve subtype counters before provider_calls so the one explicitly
    # declared compatibility exception can prove all provider subtypes are
    # present and zero (rather than treating a missing field as zero).
    ordered_fields = (
        "external_network_calls",
        "embedding_calls",
        "reranker_calls",
        "reader_calls",
        "provider_calls",
    )
    for canonical in ordered_fields:
        aliases = COUNT_ALIASES[canonical]
        found: list[tuple[str, Any]] = []
        for container in containers:
            for alias in aliases:
                if alias in container:
                    found.append((alias, container[alias]))
        if not found:
            defaults = claim.get("execution_defaults", {})
            if isinstance(defaults, dict) and canonical in defaults:
                values[canonical] = _nonnegative_count(defaults[canonical], f"{path}: {canonical} execution default")
                continue
            if canonical == "provider_calls" and claim.get("allow_implicit_zero_provider_calls"):
                subtype_values = [values.get(field) for field in ("embedding_calls", "reranker_calls", "reader_calls")]
                if subtype_values == [0, 0, 0]:
                    values[canonical] = 0
                    continue
            raise ClaimLedgerValidationError(f"{path}: {canonical} execution count is missing")
        normalized = {_nonnegative_count(value, f"{path}: {canonical}") for _, value in found}
        if len(normalized) != 1:
            raise ClaimLedgerValidationError(f"{path}: conflicting {canonical} execution counts")
        values[canonical] = normalized.pop()
    cost_found: list[Any] = []
    for container in containers:
        if "actual_external_cost_usd" in container:
            cost_found.append(container["actual_external_cost_usd"])
    if cost_found:
        costs = {_finite_number(value, f"{path}: actual_external_cost_usd") for value in cost_found}
        if len(costs) != 1 or next(iter(costs)) < 0:
            raise ClaimLedgerValidationError(f"{path}: conflicting or negative external cost")
        values["actual_external_cost_usd"] = next(iter(costs))
    return values


def _validate_artifact(path: Path, claim: dict[str, Any]) -> dict[str, Any]:
    document = _read_json(path, "experiment artifact")
    schema_path = claim.get("schema_path", ["schema_version"])
    if not isinstance(schema_path, list) or not schema_path:
        raise ClaimLedgerValidationError(f"{claim['claim_id']}: schema_path is required")
    schema = _at_path(document, schema_path, f"{path} schema")
    expected_schema = claim.get("artifact_schema")
    if not isinstance(schema, str) or not schema.strip() or schema != expected_schema:
        raise ClaimLedgerValidationError(f"{path}: schema does not match claim {claim['claim_id']}")
    classification = claim.get("classification")
    if classification is not None:
        actual_classification = document.get("classification")
        if actual_classification != classification:
            raise ClaimLedgerValidationError(f"{path}: classification does not match claim {claim['claim_id']}")
    _validate_status_rule(document, claim, path)
    calls = _extract_call_counts(document, claim, path)
    value_path = claim.get("value_path")
    value_paths = claim.get("value_paths")
    value = None
    component_values: list[int | float] = []
    if value_path is not None:
        value = _finite_number(_at_path(document, value_path, f"{path} metric"), f"{path} metric")
        component_values = [value]
    elif value_paths is not None:
        component_values = [
            _finite_number(_at_path(document, item, f"{path} metric component"), f"{path} metric component")
            for item in value_paths
        ]
        value = sum(float(item) for item in component_values) / len(component_values)
    boundary = document.get("claim_boundary", document.get("claim_boundaries"))
    if boundary is not None:
        if isinstance(boundary, str):
            boundary = [boundary]
        if not isinstance(boundary, list) or any(not isinstance(item, str) or not item.strip() for item in boundary):
            raise ClaimLedgerValidationError(f"{path}: artifact claim boundary is malformed")
    actual_sha256 = sha256_file(path)
    expected_hash = claim.get("artifact_sha256")
    if expected_hash is not None and expected_hash != actual_sha256:
        raise ClaimLedgerValidationError(f"{path}: artifact SHA-256 does not match claim specification")
    return {
        "artifact": {"path": _relative_artifact_path(path), "sha256": actual_sha256},
        "execution": calls,
        "value": value,
        "component_values": component_values,
        "artifact_claim_boundary": boundary or [],
    }


def build_claim_ledger(
    artifacts: Mapping[str, str | os.PathLike[str]],
    *,
    spec: Mapping[str, Any] | None = None,
    spec_path: Path = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Validate explicit artifact paths and return a deterministic ledger."""
    loaded_spec = _validate_spec_document(dict(spec)) if spec is not None else load_claims_spec(spec_path)
    claim_specs = {claim["claim_id"]: claim for claim in loaded_spec["claims"] if isinstance(claim, dict) and "claim_id" in claim}
    unknown = set(artifacts) - set(claim_specs)
    if unknown:
        raise ClaimLedgerValidationError(f"artifact supplied for unknown claim(s): {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    for claim_id in sorted(claim_specs):
        claim = claim_specs[claim_id]
        supplied = artifacts.get(claim_id)
        if supplied is None:
            if claim.get("required_artifact"):
                raise ClaimLedgerValidationError(f"required artifact missing for claim {claim_id}")
            if claim.get("status") != "open":
                raise ClaimLedgerValidationError(f"non-open claim {claim_id} requires an artifact")
            rows.append({
                "claim_id": claim_id,
                "title": claim["title"],
                "status": claim["status"],
                "scope": claim["scope"],
                "metric": claim.get("metric"),
                "value": None,
                "component_values": [],
                "artifact": None,
                "artifact_relative_path": None,
                "artifact_sha256": None,
                "execution": None,
                "external_network_calls": None,
                "provider_calls": None,
                "claim_boundary": list(claim["claim_boundary"]),
            })
            continue
        path = _resolve_input_path(supplied)
        evidence = _validate_artifact(path, claim)
        boundaries = list(claim["claim_boundary"])
        boundaries.extend(item for item in evidence["artifact_claim_boundary"] if item not in boundaries)
        calls = evidence["execution"]
        rows.append({
            "claim_id": claim_id,
            "title": claim["title"],
            "status": claim["status"],
            "scope": claim["scope"],
            "metric": claim.get("metric"),
            "value": evidence["value"],
            "component_values": evidence["component_values"],
            "artifact": evidence["artifact"],
            "artifact_relative_path": evidence["artifact"]["path"],
            "artifact_sha256": evidence["artifact"]["sha256"],
            "execution": calls,
            "external_network_calls": calls["external_network_calls"],
            "provider_calls": calls["provider_calls"],
            "claim_boundary": boundaries,
        })
    return {
        "schema_version": SCHEMA,
        "builder": "scripts/build_claim_ledger.py",
        "strictly_offline": True,
        "spec_sha256": hashlib.sha256(canonical_json(loaded_spec)).hexdigest(),
        "claims": rows,
    }


def render_markdown(ledger: Mapping[str, Any]) -> str:
    lines = [
        "# Retrieval research claim ledger",
        "",
        "Generated by the deterministic offline claim-ledger builder. Missing or invalid evidence aborts publication.",
        "",
        "| Claim | Status | Scope | Metric | Value | Artifact | Calls |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in ledger["claims"]:
        artifact = row["artifact"]
        artifact_text = "—" if artifact is None else f"`{artifact['path']}` (`{artifact['sha256'][:12]}…`)"
        calls = "—" if row["execution"] is None else f"external={row['external_network_calls']}, provider={row['provider_calls']}"
        value = "—" if row["value"] is None else str(row["value"])
        lines.append(f"| `{row['claim_id']}` | {row['status']} | {row['scope']} | {row['metric'] or '—'} | {value} | {artifact_text} | {calls} |")
    lines.extend(["", "## Boundaries", ""])
    for row in ledger["claims"]:
        lines.append(f"- `{row['claim_id']}`: " + " ".join(row["claim_boundary"]))
    return "\n".join(lines) + "\n"


def write_create_once(path: Path, payload: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_ledger_outputs(ledger: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path = Path(json_path).resolve()
    markdown_path = Path(markdown_path).resolve()
    if json_path == markdown_path or json_path.exists() or markdown_path.exists():
        raise FileExistsError("claim-ledger outputs are create-once and must not already exist")
    json_text = json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    write_create_once(json_path, json_text)
    try:
        write_create_once(markdown_path, render_markdown(ledger))
    except Exception:
        # Keep an already published JSON receipt; callers can recover by using
        # new output paths. Never replace or truncate an immutable artifact.
        raise


def _parse_artifacts(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        claim_id, separator, path = value.partition("=")
        if not separator or not claim_id or not path:
            raise ClaimLedgerValidationError("--artifact requires CLAIM_ID=PATH")
        if claim_id in parsed:
            raise ClaimLedgerValidationError(f"duplicate artifact mapping for {claim_id}")
        parsed[claim_id] = path
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", default=[], metavar="CLAIM_ID=PATH")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--json-output", "--output-json", required=True, type=Path)
    parser.add_argument("--markdown-output", "--output-markdown", required=True, type=Path)
    args = parser.parse_args(argv)
    ledger = build_claim_ledger(_parse_artifacts(args.artifact), spec_path=args.spec)
    write_ledger_outputs(ledger, args.json_output, args.markdown_output)
    print(json.dumps({"json_output": str(args.json_output.resolve()), "markdown_output": str(args.markdown_output.resolve()), "claim_count": len(ledger["claims"]), "provider_calls": sum(row["provider_calls"] or 0 for row in ledger["claims"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
