"""Fail-closed aggregation of schema-v2 sparse held-out seed results.

The producer writes one result per conversation split seed.  This module does
not recompute retrieval metrics or pool rows from answers; it validates the
producer contract, verifies source/dataset hashes, and aggregates only the
locked held-out paired deltas.  Ten LoCoMo conversations are clustered and
therefore do not provide independent per-conversation replicates; the output
always carries an overlap warning and an explicit no-SOTA/no-answer-accuracy
claim boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hybridmind.offline-locomo-sparse-aggregate/v1"
INPUT_SCHEMA = "hybridmind.offline-locomo-sparse-experiment/v2"
HASH_RE = set("0123456789abcdef")


class AggregationValidationError(ValueError):
    """Raised when any input violates the schema/provenance contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("bm25s", "numpy", "PyStemmer"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _require_hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or set(value) - HASH_RE
    ):
        raise AggregationValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _resolve_declared_path(value: Any, artifact_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AggregationValidationError(f"{label} path is missing")
    candidate = Path(value)
    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        candidate = project_candidate if project_candidate.exists() else artifact_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise AggregationValidationError(f"{label} path does not exist: {candidate}")
    return candidate


def _at_path(mapping: Any, *keys: str, label: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise AggregationValidationError(f"missing required field {label}")
        current = current[key]
    return current


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AggregationValidationError(f"{label} must be a finite number")
    return float(value)


def _validate_zero_execution(document: dict[str, Any], artifact_path: Path) -> None:
    execution = document.get("execution")
    if not isinstance(execution, dict):
        raise AggregationValidationError(f"{artifact_path}: execution provenance is missing")
    required_zero = (
        "external_network_calls",
        "embedding_calls",
        "reranker_calls",
        "reader_calls",
    )
    for field in required_zero:
        value = execution.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
            raise AggregationValidationError(
                f"{artifact_path}: {field} must be exactly zero"
            )
    cost = execution.get("actual_external_cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or float(cost) != 0.0:
        raise AggregationValidationError(
            f"{artifact_path}: actual_external_cost_usd must be exactly zero"
        )
    # Keep the gate closed if a producer adds a generic call counter at either
    # level without changing the established v2 counters above.
    for container_name, container in (("artifact", document), ("execution", execution)):
        for field in ("provider_calls", "external_calls"):
            if field not in container:
                continue
            value = container[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
                raise AggregationValidationError(
                    f"{artifact_path}: {container_name}.{field} must be exactly zero"
                )


def _validate_input(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise AggregationValidationError(f"input artifact does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AggregationValidationError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(document, dict):
        raise AggregationValidationError(f"artifact root must be an object: {path}")
    if document.get("schema_version") != INPUT_SCHEMA:
        raise AggregationValidationError(f"{path}: schema-v2 artifact required")
    _validate_zero_execution(document, path)

    boundary = document.get("information_boundary")
    split_seed = _at_path(boundary, "split_seed", label="information_boundary.split_seed")
    if not isinstance(split_seed, (str, int)) or str(split_seed).strip() == "":
        raise AggregationValidationError(f"{path}: split seed must be non-empty")
    held_out = _at_path(boundary, "held_out", label="information_boundary.held_out")
    if not isinstance(held_out, list) or not held_out or any(not isinstance(item, str) for item in held_out):
        raise AggregationValidationError(f"{path}: held_out must be a non-empty string list")

    provenance = document.get("provenance")
    dataset_path = _resolve_declared_path(
        _at_path(provenance, "dataset", "path", label="provenance.dataset.path"),
        path,
        "dataset",
    )
    source_path = _resolve_declared_path(
        _at_path(provenance, "source", "path", label="provenance.source.path"),
        path,
        "source",
    )
    dataset_declared = _require_hash(
        _at_path(provenance, "dataset", "sha256", label="provenance.dataset.sha256"),
        "dataset SHA-256",
    )
    source_declared = _require_hash(
        _at_path(provenance, "source", "sha256", label="provenance.source.sha256"),
        "source SHA-256",
    )
    dataset_actual = _sha256(dataset_path)
    source_actual = _sha256(source_path)
    if dataset_actual != dataset_declared:
        raise AggregationValidationError(f"{path}: dataset SHA-256 does not match declared hash")
    if source_actual != source_declared:
        raise AggregationValidationError(f"{path}: source SHA-256 does not match declared hash")
    execution_environment = {
        "git_commit": _at_path(provenance, "git_commit", label="provenance.git_commit"),
        "python": _at_path(provenance, "python", label="provenance.python"),
        "platform": _at_path(provenance, "platform", label="provenance.platform"),
        "logical_cpu_count": _at_path(
            provenance,
            "logical_cpu_count",
            label="provenance.logical_cpu_count",
        ),
        "dependency_versions": _at_path(
            provenance,
            "dependency_versions",
            label="provenance.dependency_versions",
        ),
    }
    if not all(
        isinstance(execution_environment[field], str)
        and execution_environment[field].strip()
        for field in ("git_commit", "python", "platform")
    ):
        raise AggregationValidationError(
            f"{path}: execution-environment strings must be non-empty"
        )
    if (
        isinstance(execution_environment["logical_cpu_count"], bool)
        or not isinstance(execution_environment["logical_cpu_count"], int)
        or execution_environment["logical_cpu_count"] <= 0
    ):
        raise AggregationValidationError(
            f"{path}: logical_cpu_count must be a positive integer"
        )
    if not isinstance(execution_environment["dependency_versions"], dict):
        raise AggregationValidationError(
            f"{path}: dependency_versions must be an object"
        )

    selection = _at_path(
        document,
        "representation_experiment",
        "selection",
        label="representation_experiment.selection",
    )
    winner = selection.get("winner") if isinstance(selection, dict) else None
    if not isinstance(winner, str) or not winner:
        raise AggregationValidationError(f"{path}: locked winner is missing")
    representation_criteria = _at_path(
        document,
        "representation_experiment",
        "held_out",
        "success_criteria",
        label="representation_experiment.held_out.success_criteria",
    )
    adaptive_criteria = _at_path(
        document,
        "adaptive_k_experiment",
        "success_criteria",
        label="adaptive_k_experiment.success_criteria",
    )
    quality_criteria = selection.get("quality_selection_criteria")
    if not isinstance(quality_criteria, list) or not quality_criteria:
        raise AggregationValidationError(f"{path}: quality selection criteria are missing")

    held_out_result = _at_path(
        document,
        "representation_experiment",
        "held_out",
        label="representation_experiment.held_out",
    )
    recall_delta = _at_path(
        held_out_result,
        "paired_delta_winner_minus_raw",
        "exact_evidence_recall_at_10",
        label="held-out Recall@10 delta",
    )
    mrr_delta = _at_path(
        held_out_result,
        "paired_delta_winner_minus_raw",
        "mrr",
        label="held-out MRR delta",
    )
    metric_values: dict[str, dict[str, float | int | None]] = {}
    for name, metric in (("exact_evidence_recall_at_10", recall_delta), ("mrr", mrr_delta)):
        if not isinstance(metric, dict):
            raise AggregationValidationError(f"{path}: {name} interval is missing")
        mean_value = _finite_number(metric.get("mean"), f"{path}: {name}.mean")
        low = _finite_number(metric.get("ci95_low"), f"{path}: {name}.ci95_low")
        high = _finite_number(metric.get("ci95_high"), f"{path}: {name}.ci95_high")
        n = metric.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise AggregationValidationError(f"{path}: {name}.n must be a positive integer")
        if low > high:
            raise AggregationValidationError(f"{path}: {name} interval is inverted")
        metric_values[name] = {"mean": mean_value, "ci95_low": low, "ci95_high": high, "n": n}

    return {
        "path": path,
        "artifact_sha256": _sha256(path),
        "document": document,
        "split_seed": str(split_seed),
        "held_out": sorted(set(held_out)),
        "winner": winner,
        "criteria": {
            "quality_selection_criteria": quality_criteria,
            "representation_success_criteria": representation_criteria,
            "adaptive_success_criteria": adaptive_criteria,
        },
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_actual,
        "source_path": source_path,
        "source_sha256": source_actual,
        "execution_environment": execution_environment,
        "metric_values": metric_values,
        "quality_success": bool(held_out_result.get("quality_success", held_out_result.get("predeclared_success", False))),
        "representation_success": bool(held_out_result.get("predeclared_success", False)),
        "adaptive_success": bool(
            _at_path(
                document,
                "adaptive_k_experiment",
                "predeclared_success",
                label="adaptive_k_experiment.predeclared_success",
            )
        ),
    }


def _summary(values: Sequence[float], intervals: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        raise AggregationValidationError("cannot aggregate an empty metric sequence")
    dispersion = stdev(values) if len(values) > 1 else 0.0
    return {
        "per_seed": list(intervals),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "between_seed_stddev": float(dispersion),
        "between_seed_range": float(max(values) - min(values)),
        "n_seeds": len(values),
    }


def _overlap_warning(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            overlap = sorted(set(left["held_out"]) & set(right["held_out"]))
            pairs.append(
                {
                    "left_split_seed": left["split_seed"],
                    "right_split_seed": right["split_seed"],
                    "overlap_count": len(overlap),
                    "overlap_sample_ids": overlap,
                }
            )
    any_overlap = any(pair["overlap_count"] > 0 for pair in pairs)
    return {
        "warning": (
            "LoCoMo has ten conversation clusters; split-seed aggregates are not "
            "independent replicates and overlapping held-out conversations limit "
            "generalization."
        ),
        "any_held_out_overlap": any_overlap,
        "pairs": pairs,
    }


def aggregate(paths: Sequence[Path]) -> dict[str, Any]:
    """Validate and aggregate at least two distinct schema-v2 split artifacts."""
    if len(paths) < 2:
        raise AggregationValidationError("at least two seed artifacts are required")
    records = sorted(
        (_validate_input(Path(path)) for path in paths),
        key=lambda record: (record["split_seed"], str(record["path"])),
    )
    split_seeds = [record["split_seed"] for record in records]
    if len(set(split_seeds)) != len(split_seeds):
        raise AggregationValidationError("split seeds must be unique")

    dataset_hashes = {record["dataset_sha256"] for record in records}
    source_hashes = {record["source_sha256"] for record in records}
    if len(dataset_hashes) != 1:
        raise AggregationValidationError("dataset SHA-256 must match across artifacts")
    if len(source_hashes) != 1:
        raise AggregationValidationError("source SHA-256 must match across artifacts")
    winners = {record["winner"] for record in records}
    if len(winners) != 1:
        raise AggregationValidationError("locked winners must match across artifacts")
    criteria = {_canonical(record["criteria"]) for record in records}
    if len(criteria) != 1:
        raise AggregationValidationError("declared selection/success criteria must match across artifacts")
    environments = {
        _canonical(record["execution_environment"]) for record in records
    }
    if len(environments) != 1:
        raise AggregationValidationError(
            "execution environment must match across artifacts"
        )

    recall_values = [record["metric_values"]["exact_evidence_recall_at_10"]["mean"] for record in records]
    mrr_values = [record["metric_values"]["mrr"]["mean"] for record in records]
    recall_intervals = [
        {"split_seed": record["split_seed"], **record["metric_values"]["exact_evidence_recall_at_10"]}
        for record in records
    ]
    mrr_intervals = [
        {"split_seed": record["split_seed"], **record["metric_values"]["mrr"]}
        for record in records
    ]
    return {
        "schema_version": SCHEMA,
        "classification": "measured_offline_sparse_seed_aggregate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_calls": 0,
        "external_network_calls": 0,
        "artifact_count": len(records),
        "artifacts": [
            {
                "path": str(record["path"]),
                "sha256": record["artifact_sha256"],
                "split_seed": record["split_seed"],
                "held_out_sample_ids": record["held_out"],
            }
            for record in records
        ],
        "shared_provenance": {
            "dataset": {
                "path": str(records[0]["dataset_path"]),
                "sha256": records[0]["dataset_sha256"],
            },
            "source": {
                "path": str(records[0]["source_path"]),
                "sha256": records[0]["source_sha256"],
            },
            "locked_winner": records[0]["winner"],
            "declared_criteria": records[0]["criteria"],
            "execution_environment": records[0]["execution_environment"],
        },
        "held_out_delta_summary": {
            "exact_evidence_recall_at_10": _summary(recall_values, recall_intervals),
            "mrr": _summary(mrr_values, mrr_intervals),
        },
        "counts": {
            "positive_recall_at_10_delta": sum(value > 0 for value in recall_values),
            "positive_mrr_delta": sum(value > 0 for value in mrr_values),
            "quality_success": sum(record["quality_success"] for record in records),
            "representation_predeclared_success": sum(record["representation_success"] for record in records),
            "adaptive_predeclared_success": sum(record["adaptive_success"] for record in records),
        },
        "held_out_overlap": _overlap_warning(records),
        "claim_boundary": [
            "This aggregate is exact-evidence sparse retrieval, not answer accuracy.",
            "It is not an LLM judge result and does not establish downstream answer quality.",
            "It does not establish state of the art, near-state-of-the-art, or the 70-80% prompt-source substitution target.",
            "Ten-conversation split overlap means between-seed dispersion is descriptive, not an independent-sample confidence claim.",
        ],
        "aggregator_provenance": {
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_worktree_status_sha256": hashlib.sha256(
                (_git_output("status", "--porcelain=v1", "-z") or "").encode("utf-8")
            ).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
        },
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish a create-once artifact without overwriting a receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Linking the fully fsynced temporary file is atomic and fails if the
        # destination already exists. os.replace would silently rewrite an
        # experiment receipt and violate the immutable-artifact contract.
        os.link(temporary, path)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = aggregate(args.inputs)
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_count": report["artifact_count"],
                "provider_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
