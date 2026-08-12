"""Deterministic retrieval/lifecycle ablation plans and ledger provenance.

This module deliberately does not start services or call an embedding provider.
It creates a reproducible matrix of configurations for an already warmed,
4096-dimensional HybridMind deployment.  The resulting plan files are safe to
review before a remote evaluation.  Once an evaluator has produced a standard
``eval_ledger.LedgerWriter`` JSONL file, this script can copy it with immutable
ablation provenance attached to every row.

Examples
--------
List the controlled conditions without writing files::

    python scripts/ablation_matrix.py --list

Review the resolved plans (no network calls and no writes)::

    python scripts/ablation_matrix.py --dry-run --benchmark locomo

Write reviewable plans, then annotate a completed standard ledger::

    python scripts/ablation_matrix.py --benchmark locomo --output-dir out
    python scripts/ablation_matrix.py --benchmark locomo --output-dir out \
        --annotate-ledger hybrid=benchmarks/results/ledger_locomo_abc.jsonl

The harness intentionally does *not* claim that a plan was evaluated.  A
ledger is written only when a completed evaluator ledger is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# Direct script execution sets sys.path to scripts/, whereas tests and module
# execution already include the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings
from eval_ledger import DEFAULT_K_LIST, DEFAULT_SEED, config_hash


PLAN_SCHEMA = "hybridmind.ablation-plan/v1"
LEDGER_SCHEMA = "hybridmind.ablation-ledger/v1"
REQUIRED_LEDGER_FIELDS = {
    "question_id",
    "retrieved_ids_at_k",
    "gold_in_pool_pre_rerank",
    "gold_rank_post_rerank",
    "judged_correct",
    "judge_rationale",
}


@dataclass(frozen=True)
class AblationMode:
    """One controlled condition; weights are sent explicitly by an evaluator."""

    name: str
    description: str
    vector_weight: float
    sparse_weight: float
    graph_weight: float
    feature_overrides: Mapping[str, Any]


# Every condition pins the same retrieval controls.  Individual modes change
# exactly the signal weights or the one named lifecycle/temporal feature.
_CONTROLLED_BASE: dict[str, Any] = {
    "embedding_dimension": 4096,
    "query_routing_enabled": False,
    "query_time_expansion_enabled": False,
    "query_decomposition_enabled": False,
    "rerank_mode": "off",
    "local_lexical_rerank_enabled": False,
    "temporal_decay_enabled": False,
    "temporal_edges_enabled": False,
    "access_tracking_enabled": False,
    "salience_enabled": False,
    "fact_extraction_enabled": False,
    "memory_compression_enabled": False,
}


MODES: tuple[AblationMode, ...] = (
    AblationMode("vector_only", "Dense vector signal only.", 1.0, 0.0, 0.0, {}),
    AblationMode("sparse_only", "BM25/BM25S sparse signal only.", 0.0, 1.0, 0.0, {}),
    AblationMode("vector_sparse", "Dense and sparse signals, no graph.", 0.5, 0.5, 0.0, {}),
    AblationMode("graph_only", "Graph signal only; evaluator must supply anchors.", 0.0, 0.0, 1.0, {}),
    AblationMode("hybrid", "Dense, sparse, and graph RRF fusion baseline.", 0.50, 0.35, 0.15, {}),
    AblationMode(
        "temporal",
        "Hybrid baseline with temporal graph decay and temporal edges enabled.",
        0.50,
        0.35,
        0.15,
        {
            "query_time_expansion_enabled": True,
            "temporal_decay_enabled": True,
            "temporal_edges_enabled": True,
        },
    ),
    AblationMode(
        "salience",
        "Hybrid baseline with deterministic salience/access tracking enabled.",
        0.50,
        0.35,
        0.15,
        {"access_tracking_enabled": True, "salience_enabled": True},
    ),
    AblationMode(
        "structured_facts",
        "Hybrid baseline with structured fact extraction enabled during ingestion.",
        0.50,
        0.35,
        0.15,
        {"fact_extraction_enabled": True},
    ),
    AblationMode(
        "compression",
        "Hybrid baseline with memory compression enabled during lifecycle processing.",
        0.50,
        0.35,
        0.15,
        {"memory_compression_enabled": True},
    ),
)
MODE_BY_NAME = {mode.name: mode for mode in MODES}


def _redact_config(values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep plans auditable without serializing provider credentials."""
    return {
        key: "<redacted>" if key.endswith("_api_key") and value else value
        for key, value in values.items()
    }


def _environment_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_mode(mode: AblationMode, *, benchmark: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Resolve one mode against current Settings without changing process state."""
    resolved = Settings().model_dump()
    resolved.update(_CONTROLLED_BASE)
    resolved.update(mode.feature_overrides)
    if resolved["embedding_dimension"] != 4096:
        raise RuntimeError("Ablation matrix refuses any embedding dimension other than 4096")

    search_mode = {
        "vector_only": "vector_only",
        "sparse_only": "sparse_only",
        "vector_sparse": "vector_sparse",
        "graph_only": "graph_only",
    }.get(mode.name, "hybrid")
    request_parameters = {
        "vector_weight": mode.vector_weight,
        "graph_weight": mode.graph_weight,
        "bm25_boost_weight": mode.sparse_weight,
        "rerank_pool": 1,
        "search_mode": search_mode,
        "route_weights": False,
        "track_access": bool(resolved["access_tracking_enabled"]),
    }
    hashed_config = {
        "schema": PLAN_SCHEMA,
        "benchmark": benchmark,
        "mode": mode.name,
        "seed": seed,
        "request_parameters": request_parameters,
        "resolved_settings": _redact_config(resolved),
    }
    plan_hash = config_hash(hashed_config)
    output_ledger = (
        Path("ledgers") / benchmark / mode.name / f"ledger_{benchmark}_{plan_hash}.jsonl"
    )
    # The environment map is useful when booting a separately isolated API
    # process.  Evaluators must still pass request_parameters explicitly,
    # because current evaluation scripts submit their own request weights.
    environment_values = dict(_CONTROLLED_BASE)
    environment_values.update(mode.feature_overrides)
    environment = {
        f"HYBRIDMIND_{key.upper()}": _environment_value(value)
        for key, value in environment_values.items()
    }
    environment["RERANK_MODE"] = _environment_value(resolved["rerank_mode"])
    environment["HYBRIDMIND_ABLATION_MODE"] = mode.name
    environment["HYBRIDMIND_ABLATION_CONFIG_HASH"] = plan_hash
    return {
        **hashed_config,
        "plan_hash": plan_hash,
        "ledger_target": str(output_ledger),
        "evaluator_parameters": {
            "graph_anchor_strategy": "vector_top1" if mode.name == "graph_only" else "explicit",
        },
        "environment": dict(sorted(environment.items())),
        "protocol": {
            "preflight_required": "python scripts/preflight.py",
            "embedding_dimension_required": 4096,
            "random_seed": seed,
            "k_values": list(DEFAULT_K_LIST),
            "reranker": "off (controlled ablation)",
            "result_status": "planned; no benchmark result recorded",
        },
    }


def select_modes(names: str | None) -> tuple[AblationMode, ...]:
    if not names:
        return MODES
    selected: list[AblationMode] = []
    for name in (part.strip() for part in names.split(",")):
        if not name:
            continue
        try:
            selected.append(MODE_BY_NAME[name])
        except KeyError as exc:
            choices = ", ".join(MODE_BY_NAME)
            raise ValueError(f"Unknown mode {name!r}; choose from: {choices}") from exc
    if not selected:
        raise ValueError("At least one ablation mode is required")
    return tuple(selected)


def write_plans(plans: Iterable[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for plan in plans:
        path = output_dir / f"{plan['mode']}_{plan['plan_hash']}.plan.json"
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


def _validate_ledger_row(row: Mapping[str, Any], line_number: int) -> None:
    missing = sorted(REQUIRED_LEDGER_FIELDS - set(row))
    if missing:
        raise ValueError(f"Ledger line {line_number} is missing required fields: {', '.join(missing)}")
    if not isinstance(row["retrieved_ids_at_k"], Mapping):
        raise ValueError(f"Ledger line {line_number} retrieved_ids_at_k must be an object")


def annotate_ledger(source: Path, destination: Path, plan: Mapping[str, Any]) -> int:
    """Copy a completed native ledger with mode/config provenance.

    The retrieval and judgment values are not modified.  ``config_hash`` is
    replaced by the full controlled-plan hash, while the evaluator's original
    value remains available as ``source_config_hash``.
    """
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {source} line {line_number}: {exc}") from exc
        _validate_ledger_row(row, line_number)
        source_hash = row.get("config_hash")
        row["source_config_hash"] = source_hash
        row["config_hash"] = plan["plan_hash"]
        row["ablation"] = {
            "schema": LEDGER_SCHEMA,
            "mode": plan["mode"],
            "plan_hash": plan["plan_hash"],
            "resolved_settings_sha256": hashlib.sha256(
                json.dumps(plan["resolved_settings"], sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "request_parameters": plan["request_parameters"],
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"Ledger {source} contains no rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return len(rows)


def _parse_annotation(value: str) -> tuple[str, Path]:
    try:
        mode, source = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected MODE=PATH") from exc
    if mode not in MODE_BY_NAME:
        raise argparse.ArgumentTypeError(f"Unknown mode {mode!r}")
    return mode, Path(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic HybridMind ablation plans")
    parser.add_argument("--benchmark", default="locomo", help="Benchmark label recorded in plan/ledger names")
    parser.add_argument("--modes", help="Comma-separated modes (default: all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/ablation_matrix"))
    parser.add_argument("--list", action="store_true", help="List modes only; do not write")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plans; do not write")
    parser.add_argument(
        "--annotate-ledger",
        action="append",
        type=_parse_annotation,
        metavar="MODE=PATH",
        help="Copy a completed standard ledger with controlled mode provenance",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        modes = select_modes(args.modes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.list:
        for mode in modes:
            print(f"{mode.name:16} {mode.description}")
        return 0

    plans = [resolve_mode(mode, benchmark=args.benchmark, seed=args.seed) for mode in modes]
    if args.dry_run:
        print(json.dumps(plans, indent=2, sort_keys=True))
        return 0

    written = write_plans(plans, args.output_dir)
    for path in written:
        print(f"wrote plan: {path}")

    plan_by_mode = {plan["mode"]: plan for plan in plans}
    for mode_name, source in args.annotate_ledger or []:
        if mode_name not in plan_by_mode:
            raise SystemExit(f"--annotate-ledger mode {mode_name!r} is outside --modes")
        if not source.is_file():
            raise SystemExit(f"Completed ledger not found: {source}")
        plan = plan_by_mode[mode_name]
        destination = args.output_dir / plan["ledger_target"]
        count = annotate_ledger(source, destination, plan)
        print(f"wrote annotated ledger ({count} rows): {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
