"""Documentation integrity guards (AGENTS.md "Documentation map").

These tests make documentation rot fail the offline suite instead of being
discovered by a confused human or agent months later:

1. AGENTS.md is the single agent entry point (no duplicate twin).
2. Every expected document exists, and no unregistered `.md` appears in any
   documented area (root, docs/, benchmarks/, cli/, demos/, deploy/,
   experiments/reports/).
3. Relative markdown links in README/AGENTS/docs resolve to real files.
4. The rolling handoff (`docs/CURRENT_STATE.md`) keeps its required sections.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Registered documents, mirroring the Documentation Map table in AGENTS.md.
EXPECTED_DOCS: dict[str, str] = {
    "README.md": "human front door",
    "PHASE_IMPLEMENTATION_STATUS.md": "real-vs-scaffolded inventory",
    "AGENTS.md": "agent entry point / contract",
    "docs/ADVERSARIAL_AUDIT_REMEDIATION.md": "historical audit record",
    "docs/AGENT_INTEGRATION.md": "SDK/MCP integration contracts",
    "docs/ALGORITHM.md": "ranking/scoring algorithms",
    "docs/ARCHITECTURE.md": "request/data flow and storage",
    "docs/CURRENT_STATE.md": "rolling session handoff",
    "docs/DECISIONS.md": "append-only judgment-call log",
    "docs/EVALUATION.md": "evaluator usage and ledger schema",
    "docs/KV_CACHE_RESEARCH.md": "KV-hypothesis history",
    "docs/RESOURCE_SPEED_TOKENOMICS.md": "resource measurement protocol",
    "docs/RETRIEVAL_RESEARCH_PROTOCOL.md": "preregistered research gates",
    "docs/research/claim-ledger-20260822.md": "research claim ledger",
    "docs/research/design-space-experiment-program.md": "research program",
    "docs/research/hybridmind-retrieval-report-20260822.md": "research report",
    "docs/research/prior-art-mechanism-ledger.md": "prior-art ledger",
    "docs/research/sota-gap-analysis.md": "SOTA gap analysis record",
    "benchmarks/PERFORMANCE.md": "resource characterization",
    "benchmarks/results/BENCHMARK_REPORT.md": "valid vs deprecated results",
    "cli/README.md": "CLI command surfaces",
    "demos/techspec.md": "demo specification",
    "deploy/README_image_server.md": "visual backend deployment",
    "experiments/reports/baseline.md": "pre-change measurement snapshot",
}

# Areas whose markdown must all be registered above are enumerated by
# _doc_areas() below.

_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def _doc_areas() -> list[Path]:
    return [
        ROOT,
        ROOT / "docs",
        ROOT / "docs" / "research",
        ROOT / "benchmarks",
        ROOT / "benchmarks" / "results",
        ROOT / "cli",
        ROOT / "demos",
        ROOT / "deploy",
        ROOT / "experiments" / "reports",
    ]


def test_agents_md_is_the_single_entry_point() -> None:
    assert (ROOT / "AGENTS.md").is_file()
    assert not (ROOT / "AGENT.md").exists(), (
        "AGENT.md duplicates AGENTS.md; keep exactly one entry point"
    )


def test_registered_docs_exist() -> None:
    missing = [p for p in EXPECTED_DOCS if not (ROOT / p).is_file()]
    assert not missing, f"registered docs missing on disk: {missing}"


def test_no_unregistered_markdown_in_documented_areas() -> None:
    extra: list[str] = []
    for area in _doc_areas():
        if not area.is_dir():
            continue
        for path in area.glob("*.md"):
            rel = path.relative_to(ROOT).as_posix().replace("\\", "/")
            if rel not in EXPECTED_DOCS:
                extra.append(rel)
    assert not extra, (
        "unregistered markdown files found; register them in "
        f"tests/test_doc_integrity.py and the AGENTS.md doc map: {extra}"
    )


def test_relative_markdown_links_resolve() -> None:
    broken: list[str] = []
    for rel in EXPECTED_DOCS:
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in _LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            clean = target.split("#", 1)[0].strip()
            if not clean:
                continue
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                broken.append(f"{rel} -> {target}")
    assert not broken, f"broken relative links: {broken}"


def test_current_state_handoff_has_required_sections() -> None:
    text = (ROOT / "docs" / "CURRENT_STATE.md").read_text(encoding="utf-8")
    for required in ("Last updated", "Branch/commit", "Last verified",
                     "Active focus", "Gotchas"):
        assert required in text, (
            f"CURRENT_STATE.md lost its '{required}' entry; update the handoff"
        )
