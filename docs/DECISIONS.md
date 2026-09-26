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

## 2026-09-25 — LoCoMo category IDs corrected to the official mapping

**Context.** `eval_locomo_retrieval.py` and five offline scripts used
1=single-hop, 3=multi-hop, 4=world-knowledge. The data (cat 1: 98% of
questions cite ≥2 turns; cat 4: 1.07 turns on average; cat 3: "would …
likely" questions) and the official snap-research `task_eval` code (also used
by mem0) give 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop,
5=adversarial. The permuted labels made published findings wrong: the
"MiniLM hurts multi-hop by −0.091" and "PPR multi-hop Δ = 0" results are
open-domain (n≈42 per split); real multi-hop improved with MiniLM (+0.067).

**Decision.** One canonical `CATEGORY` in
`scripts/offline_locomo_sparse_baseline.py`; the evaluator keeps an equal
literal pinned by `tests/test_locomo_category_map.py`. Old artifacts are left
byte-identical (hashes are cited); their labels are corrected by errata in
the research docs rather than by rewriting evidence files.

**Reversal notes.** Only if an authoritative LoCoMo release documents a
different mapping; then change the one constant and the test together.

## 2026-09-25 — Live plans must price their usage ceiling

**Context.** `validate_live_plan` compared only *planned* usage cost to
`max_estimated_spend_usd`; a run may legally consume up to its ceiling.

**Decision.** Also reject plans whose ceiling cost exceeds the cap. Test:
`test_preflight_rejects_ceiling_that_can_outspend_the_cap`.

**Reversal notes.** None expected; plans that relied on the gap were unsafe.

## 2026-09-25 — Research workspace and answer-evaluation conventions

**Context.** The autoresearch workflow wants a state/log/findings workspace;
answer evaluation needs a judge and reader protocol the repo lacked.

**Decision.** (1) Workspace lives in `research/` (protocols, analyses,
literature notes); scripts stay in `scripts/`, raw results in
`experiments/results/`. Protocols are committed before their results.
(2) `scripts/budgeted_answer_eval.py` uses the official LongMemEval reader
(CoT, temperature 0, 800 tokens) and per-type judge prompts verbatim (MIT).
LoCoMo has no official LLM judge: we apply LongMemEval's strict templates
(temporal template for temporal, abstention template for adversarial with a
fixed explanation) and give the reader the last session date as "Current
Date". (3) Budgets are counted with a declared regex token proxy
(`\w+|[^\w\s]`), because no tokenizer download was admitted; ledgers also
store provider-reported usage.

**Reversal notes.** Swap the judge only with a stated meta-evaluation; a real
tokenizer can replace the proxy if budgets are re-derived for all arms.

## 2026-09-25 — Answer-level research runs use `scripts/budgeted_answer_eval.py`, not memorybench

**Context.** The owner asked to use memorybench. Its local copy (AI SDK 5) calls
the Responses API (Z.AI serves chat completions only), passes `maxTokens`
(ignored in v5, so outputs are uncapped), re-prompts every abstention into a
quoted answer (defeats `_abs`/adversarial scoring), reports LLM-judged hit rates
as "recall", never routes LongMemEval `_abs` ids to the abstention judge, uses a
permuted LoCoMo category map, and paraphrases the official judge prompts into
JSON output. The owner then approved building a better harness.

**Decision.** Research answer runs use the repo harness: official LongMemEval
reader/judge prompts verbatim, exact evidence IDs, plan-bound reader and judge
stages with cumulative per-plan spend, failure receipts, resume.

**Reversal notes.** memorybench remains useful for cross-provider demos. To use
it for research numbers, first fix the six issues above (and upstream them).

## 2026-09-26 — Tri-signal engine is a new `/retrieve` path, not a rewrite of `/search/hybrid`

**Context.** `HybridRanker.search` truncates the dense+sparse union on a
heuristic score before RRF and seeds its graph signal from the top three
dense/sparse hits, so its channels are neither independent nor separately
measurable (tmp survey: engine map). Rewriting it would break ~40 tested
behaviours callers rely on.

**Decision.** Add `engine/trisignal.py` behind `POST /retrieve` (SDK
`retrieve()`, MCP `retrieve`). Each channel ranks a scope-local corpus built
from SQLite (cached by `corpus_generation`): exact dense inner product
(HNSW optional, with a per-query recall audit), BM25S Lucene with scope-local
IDF, and the entity-memory graph with query-derived anchors. Cross-channel
seeding exists only when requested (`ppr_passage_seed_channel`) and is
recorded as `depends_on`. `/search/hybrid` is unchanged.

**Reversal notes.** Retire `/search/hybrid` once clients migrate and the
tri-signal defaults are measured on the 4096-d arm; keep the two paths from
sharing mutable state.

## 2026-09-26 — Graph channel = EMG entity-memory graph; PPR is an arm

**Context.** Of the graph methods with released code, only EMG
(arXiv 2608.27925, MIT) reports evidence-ID recall against a matched dense
control on conversational memory (+4.74 R@25 on LoCoMo). HippoRAG 2 (MIT) has
the proven PPR walk but no conversational evidence, and our earlier term-graph
PPR was null.

**Decision.** Port EMG's recall path with exact upstream parity (19,860/19,860
identical rankings on shipped artifacts, `scripts/reproduce_emg_locomo.py
--port`) as the default `graph_method="emg"`; port HippoRAG-2 seeding + PPR
(damping 0.5, top-5 entity seeds weighted by node specificity) as
`graph_method="ppr"`. PPR uses a scipy power iteration instead of igraph
(GPL); parity with networkx within 1e-8. The NLTK stopword list is vendored
(198 words) so no corpus download happens at runtime.

**Reversal notes.** Switch the default to PPR only if a preregistered arm on
the 4096-d track beats EMG; reference-track PPR graph-only was 62.6 vs 66.1.

## 2026-09-26 — Offline `lexical-v1` entity extractor is the default graph source

**Context.** EMG's extractor is an LLM call per turn (gpt-3.5-turbo); the
owner asked for zero spend. On EMG's own LoCoMo artifacts the deterministic
`lexical-v1` extractor (speakers, capitalized spans, time/number expressions,
RAKE-style phrases) gave graph-only recall@25 66.06 vs 66.09 for the LLM
graph, and +4.10 [3.14, 4.93] over dense when fused (LLM graph: +4.65).

**Decision.** Default `trisignal_graph_extractor="lexical-v1"`; LLM
extractions persist in `node_entity_extractions` and are selected by name.
Partial LLM coverage of a scope is refused, not mixed.

**Reversal notes.** Re-measure on the 4096-d track and on LongMemEval; the
parity is one dataset with 1536-d reference vectors.

## 2026-09-26 — Provisional `/retrieve` defaults: RRF k=60 over all three channels

**Context.** The dense arm cannot be measured on 4096-d vectors without spend;
reference-track (1536-d) results already show equal-weight RRF including
sparse is below dense+graph on LoCoMo.

**Decision.** Keep equal-weight RRF k=60 (the fusion contract) over all three
channels as the provisional server default, exposed per request, and let the
VPS 4096-d ablation set channels/weights/fusion. The reference-track numbers
are not used to tune defaults.

**Reversal notes.** Change `trisignal_channels`/`trisignal_fusion` after the
4096-d run on both datasets, with the paired CIs recorded in
`benchmarks/results/BENCHMARK_REPORT.md`.

## 2026-09-26 — Reranker default, TEI reranker and keyless loopback TEI

**Context.** The default `mixedbread-ai/mxbai-rerank-large-v2` needs
sentence-transformers >= 5.4 (the venv has 5.3) and its accuracy comment had
no source. The VPS plan runs TEI next to the API, where no credential exists.

**Decision.** Default `reranker_model` is `BAAI/bge-reranker-v2-m3`
(Apache-2.0; used by LazyMem). New `rerank_mode="tei"` calls TEI `/rerank`
(schema checked against TEI 1.9.4 OpenAPI) and fails closed on missing or
duplicate indices. `LOCAL_TEI_EMBEDDING_URL` and loopback `reranker_tei_url`
accept only loopback hosts and never carry a credential; RunPod URLs stay
bound to `RUNPOD_API_KEY`.

**Reversal notes.** mxbai-v2 can return after the dependency is upgraded and
its load path is tested.

## 2026-09-26 — Evaluation hygiene: backups, licences, reference vectors

**Context.** `tests/conftest.py` did not isolate `HYBRIDMIND_BACKUP_DIR`, so
every suite run wrote shutdown snapshots into `data/backups/` and pruned the
operator's `.mind.zip` backups to three. The LoCoMo recall code is
CC-BY-NC-4.0. EMG's shipped vectors are 1536-d.

**Decision.** Tests write backups to their temp dir. LoCoMo recall is
reimplemented from its definition (no code copied). EMG's 1536-d vectors are
used only by the reference-track harness (`--dense emg-reference`), never by
the runtime engine, and every such result is labelled "reference track".

**Reversal notes.** None for the first two. Drop the reference track once the
4096-d LoCoMo cache exists.
