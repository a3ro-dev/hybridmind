# Judgment-call decision log

Append-only record of non-obvious decisions that a future maintainer (human or
agent) might want to revisit or reverse. One entry per decision; newest last.
Trivially reversible changes (typo fixes, comment updates) do not belong here.

Format: `## YYYY-MM-DD — short title`, then Context / Decision / Reversal notes.

---

## 2026-08-26 — Repository hygiene pass

**Context.** Multi-year accumulation from many agents/editors: dead modules,
stale docs, one-off scripts, and business material mixed into the tracked
tree. Goal: fewer, sharper files; docs that fail loudly when they rot.

1. **`AGENT.md` deleted; `AGENTS.md` is the single entry point.**
   Byte-identical duplicates invited silent divergence.

2. **Dead code removed** (verified zero inbound references before deletion):
   `middleware/` (rate limiting lives in `main.py`'s ASGI middleware),
   `engine/effectiveness.py` (`api/comparison.py` implements the endpoint via
   `engine/eval.py`), `ui/app.py` + the `streamlit` dependency (orphaned
   Streamlit dashboard; owner confirmed deletion), root `__init__.py`
   (nothing imports the repo root as a package), `run_tests.py`,
   `start_server.py` (hardcoded an absolute path and wrote logs into the
   repo root).

3. **One-off/era-specific scripts removed:** `scripts/test_wiki.py`,
   `scripts/fetch_context.py`, `scripts/check_cp.py`,
   `scripts/inspect_cp.py`, `scripts/parse_search_metrics.py`,
   `scripts/fetch_sample_20.py`, root `sample_20_raw.json` (the "sample_20"
   mini-eval era; its tombstone `scripts/score_sample_20.py` stays because
   `tests/test_eval_benchmark_integrity.py` asserts it refuses to run),
   `scripts/phase2_sweep.py`, `scripts/phase3_fixes.py`,
   `scripts/rerun_graph_exp1_low_threshold.py` (AG-news sweep era),
   `deploy/minecraft_maintenance.py` + its systemd unit (unrelated project's
   ops debris).
   `scripts/multi_domain_eval.py` was **kept**: it is now a quarantined
   plan-only stub whose live execution is asserted-dead by a test.

4. **Business trio untracked, not deleted.** `business-prop.pdf`,
   `deep-research-report (1).md`, `convert_to_pdf.py` moved to gitignored
   `local/`. The markdown makes marketing claims that contradict the repo's
   claim-discipline rules; it should never be cited as technical evidence.
   Recoverable in git history if ever needed again.

5. **Docs deleted as superseded/duplicated:**
   - `docs/MULTI_DOMAIN_EVAL.md` → content folded into `docs/EVALUATION.md`.
     No live writer remains (the quarantine stub cannot reach its report
     writer).
   - `docs/PHASE_6_REALISTIC.md` → its still-load-bearing conventions
     (versioned answer prompts, ~2.5-point noise floor, L1/L2/L3 loss
     decomposition) were folded into `docs/EVALUATION.md`; its punch-list
     status was already superseded by `PHASE_IMPLEMENTATION_STATUS.md`. All
     code-comment references repointed.
   - `docs/RESEARCH_HANDOFF.md` → point-in-time session handoff whose facts
     (worktree state, GPU status) were stale; live research state is owned by
     `docs/CURRENT_STATE.md` + `docs/research/design-space-experiment-program.md`.

6. **Untracked local debris archived, not destroyed:** March-era
   `stress_test*.py` + `STRESS_TEST_REPORT.md` moved to `tmp/archive/stress-2026-03/`;
   stray `server*.log`, `server.pid`, empty logs deleted (regenerable);
   empty `presentation/` directory removed; two 0-byte
   `benchmarks/results/ledger_*.jsonl` debris files deleted locally (ignored
   by git, no provenance value).

7. **Kept despite size:** `experiments/results/` (~250 MB tracked evidence
   JSONs, including uncited v1 variants of cited artifacts). Owner chose
   evidence integrity over clone size.

8. **Known debts documented rather than churned** (behavioral risk > benefit
   today): mixed error-response shapes in `main.py`
   (`{"status","message"}` vs `{"error"}` vs `HTTPException`), dual
   camel/snake metadata keys (`containerTag`/`container_tag`) kept for
   backward compatibility, and the TypeScript query-router mirror in
   `memorybench/src/providers/hybridmind/index.ts` that must be updated in
   lockstep with `engine/query_router.py`. See AGENTS.md "Known debts".

9. **Config-source-of-truth fix:** `engine/image_embedding.py` now reads its
   RunPod key from `config.Settings.image_runpod_key` (env
   `HYBRIDMIND_IMAGE_RUNPOD_KEY`) instead of raw `os.getenv`.

10. **pytest.ini markers pruned** (seven declared, zero used); Makefile
    rewritten to real targets (`test`, `verify`, `compile`, `check`) after
    its benchmark target pointed at a nonexistent file for months.
