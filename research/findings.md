# Research findings (living synthesis)

## Research question

At a fixed evidence-token and inference budget, which changes recover missing evidence, preserve
all evidence needed for multi-hop answers, and turn that evidence into correct, supported answers?
Working form: **which published conversational-memory mechanisms still help once rendered tokens
are held fixed — and which gains were just more text?**

## Current understanding (2026-09-25, after audit + E1–E3)

1. **The previous paper's per-category story was mislabelled.** Every offline LoCoMo script used
   a permuted category map. "MiniLM hurts multi-hop" and "PPR multi-hop Δ0" are open-domain
   (n≈42). Real multi-hop (cat 1) is cross-session aggregation and improved under MiniLM.
   The field has the same problem: ≥4 incompatible LoCoMo mappings circulate (official = mem0
   harness; the Mem0 *paper*, Memobase, Backboard, Hindsight, Memanto rotate; Zep swaps 2/3;
   supermemory memorybench uses its own), verified by re-weighting published per-category scores.
2. **Evidence locality splits LoCoMo in two.** Single-hop evidence is local (reply/adjacent
   turns); multi-hop evidence spans sessions (38% ≥3 sessions). They want opposite budget
   allocations: depth around a hit vs breadth across hits.
3. **At equal rendered tokens, neighbourhood beats more hits** (E1, exploratory): ±1 windows
   +0.08–0.11 complete coverage; window@1k (0.675) > top-k@2k (0.641). This is the equal-token
   control the neighbour-expansion literature (LazyMem, CueMem, ReFind) does not report.
4. **Cross-session aggregation is the unsolved retrieval stage.** Complete coverage 0.12/0.19/0.29
   at 1k/2k/4k with lexical units; the miss is a vocabulary gap between third-person questions and
   first-person turns. Derived fact keys (dataset observations, diagnostic) add +0.05–0.07 (E2) —
   consistent with LongMemEval's key expansion (R@10 0.692→0.784).
5. **Propagation is a refinement, not the mechanism** (E3): λ=0.7 one-hop score diffusion removes
   the fixed window's temporal/multi-hop cost but only ties the window at 2k on production keys.
6. **Published "SOTA" numbers are mostly not comparable**: the 100% claims are session-level
   recall_any hit rates (Engram 60 memories on 470 q; MemPalace top-50 > sessions per convo);
   answer accuracy swings 20+ points with harness/judge (Hindsight 94.6 own vs 72.2 independent);
   the reader dominates (Gemini 3 Pro full-context 92.0% on LME-S); LoCoMo is saturated by full
   context (92.6% vs a 93.6% annotation ceiling), so LoCoMo gains only matter at budgets ≪ history.

7. **Graph retrieval is dataset-dependent, not universally additive** (R1 + tri-signal runs,
   2026-09-26). EMG's entity-memory graph reproduces its LoCoMo fusion gain (+4.65 vs +4.74 R@25,
   exact port parity) and PPR-seeded tri-signal fusion is +3.7 R@25 over dense on the reference
   track; on LongMemEval-S with the offline lexical graph, adding the graph to BM25 costs −0.081
   recall_all@10. A free lexical extractor matched the LLM extractor on LoCoMo (66.06 vs 66.09).

## Lessons and constraints

- BM25S tie order depends on k (47/1,977 LoCoMo top-10 sets differ) — break ties explicitly.
- On Windows, `Path.write_text` converts LF→CRLF; the repo is LF. Write bytes or use Edit.
- Do not trust per-category labels in any artifact produced before commit 27d4e92.
- The local `longmemeval_s.json` is the oracle file; confirmation needs the official cleaned S.
- LongMemEval turns are long and evidence sits in user turns: fixed neighbour windows are
  expensive there (oracle smoke test: nbr1 −0.23 at 2k) — the locality mechanism may not transfer.
- Laptop is weak: no local neural encoders beyond seconds-scale; live calls need a priced plan.

## Open questions

- Does the neighbourhood/propagation result transfer to official LongMemEval-S (E4, locked)?
- Does complete coverage at a fixed budget convert into answer accuracy, and how much of the
  remaining error is reader failure with complete evidence? (needs admitted live budget)
- Does our own fact extraction reproduce E2's aggregation gain without dataset leakage?
- Dense (native 4096-d) + sparse complementarity at equal budget.
