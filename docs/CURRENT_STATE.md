# CURRENT STATE — rolling handoff

Read this first when arriving mid-stream; update it before ending a session.
Keep it short: this is a whiteboard, not a report. Historical narrative
belongs in `docs/DECISIONS.md` and `docs/research/`.

- **Last updated:** 2026-08-26
- **Branch/commit:** `main` @ `1ed5a68` + uncommitted repository-hygiene pass
  (deletions, doc consolidation, AGENTS.md restructure — summary in
  `docs/DECISIONS.md`, entry 2026-08-26)
- **Provider calls made this session:** 0 (offline work only)

## Last verified (2026-08-26, post-hygiene worktree)

- Full offline suite (`pytest tests/ -q`): **392 passed / 3 skipped** in ~48 s.
  Includes the new `tests/test_doc_integrity.py` guards (5 tests).
- Legacy `verify/` suite (`pytest verify/ -q`): **16 passed** — the four
  baseline-era failures were already repaired via explicit in-test feature
  opt-ins; no changes were needed this session.
- Compilation (`python -m compileall` over app/eval/scripts/tests/verify): pass.

## Active focus

Repository hygiene completed this session: dead code and one-off scripts
removed, docs consolidated under an enforced documentation map, AGENTS.md
restructured as the cold-start agent entry point. No retrieval, ranking,
storage, or eval-ledger logic was intentionally changed; comment-only
reference updates in eval files.

## Open questions / pending decisions

- None blocking. Next research gates are owned by
  `docs/research/design-space-experiment-program.md` §10 (independent memory
  corpus; priced native-4096-d Flat-vs-HNSW semantic test).
- Commit split suggestion for whoever commits: (1) code deletions + fixes,
  (2) doc restructure + guards, or a single hygiene commit; owner's choice.

## Gotchas for the next agent

- `.venv` (not system Python) for everything; `pnpm` inside `memorybench/`
  only if touching the benchmark harness.
- Never edit files via PowerShell string pipelines (`Get-Content |
  Set-Content`); they corrupt non-ASCII bytes (bitten twice now — see
  DECISIONS log).
- `memorybench/` is gitignored except two tracked provider files; its data
  includes a 347 MB checkpoint JSON — do not commit or "clean" blindly.
- Business material lives in gitignored `local/`; do not cite it as technical
  evidence.
