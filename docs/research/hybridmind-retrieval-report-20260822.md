# hybridmind, after contact with evidence

22 august 2026

## the answer first

this is not a SOTA claim. it is a quantified offline ceiling and a much more
honest architecture.

i treated every component as disposable. the conventional baseline survived.
speaker-prefixed BM25S improved held-out exact-evidence Recall@10 by 0.02966
across five conversation splits. a fixed-pool MiniLM selector added 0.04262 on
its held-out split, but hurt multi-hop by 0.09127 and therefore failed the
unconditional-default gate. BGE-M3 MaxSim also improved a fixed BM25 pool by
0.02922, but required 837.6 MB of token vectors for 5,881 turns. both selectors
stay gated and experimental.

the mechanisms that did not survive are just as important. BGE-M3 learned
sparse lost 0.09070 Recall@10 to BM25S at equal budget. graph PPR proved that
real term associations matter beyond node degree, but failed to beat BM25S
reliably and added exactly zero held-out multi-hop lift. Turbovec compressed the
raw vector matrix by about 5.2x at four bits, but retained only 0.85344 mean
synthetic Recall@10. the speaker router and two-field sparse RRF were dominated.

## the research tree

```text
                    HYBRIDMIND
                        |
             +----------+----------+
             |                     |
        RETRIEVAL              MEMORY MODEL
             |                     |
     +-------+-------+       +-----+-----+
     |       |       |       |     |     |
   dense   sparse  graph   episodic semantic temporal
     |       |       |       |     |     |
     +-------+-------+-------+-----+-----+
                        |
                  candidate fusion
                        |
                 reranking / selection
                        |
                 evidence grounding
                        |
                  answer generation
```

the architecture rule is simple: a component earns its place only by improving
the measured objective at a defensible resource point. novelty has no positive
weight.

## what survived

| mechanism | held-out evidence | decision |
|---|---|---|
| speaker-prefixed BM25S | +0.02966 mean Recall@10 over raw across five reused conversation splits | keep as the conventional candidate baseline |
| MiniLM fixed-pool reranking | +0.04262 Recall@10, CI [0.01299, 0.06643] | keep behind a category/resource gate; multi-hop regressed |
| BGE-M3 MaxSim on BM25 pool | +0.02922 Recall@10, CI [0.01419, 0.04085] | retain as an experimental selector; storage and encoding block promotion |
| associative PPR vs degree sham | +0.56447 Recall@10, CI [0.55794, 0.57030] | association is real signal, but graph is not a general winner |
| FAISS HNSW control | synthetic Recall@10 0.66523 at efSearch 64; 1.0 at 1024 | expose the control; do not call efSearch 64 harmless |
| compact SQLite embedding storage | 48.50% database reduction on 512 native 4096-d vectors | keep; bit-exact logical behavior was preserved |

## what was eliminated

| mechanism | falsifying evidence | decision |
|---|---|---|
| BGE-M3 learned sparse | -0.09070 pre-rerank Recall@10 vs BM25S; CI entirely below zero | reject this configuration on LoCoMo |
| general PPR promotion | +0.00859 vs BM25S; CI crosses zero; multi-hop delta 0 | reject as a default; temporal follow-up remains open |
| unconditional MiniLM | multi-hop delta -0.09127; CI [-0.13158, -0.03101] | reject; gating or different training required |
| speaker router | same recall as unconditional prefix while retaining two indexes | reject |
| two-field sparse RRF | 0.56056 vs 0.57136 for one prefix index | reject on the measured split |
| Turbovec 4-bit default | 0.85344 synthetic Recall@10 at about 5.2x compression | reject until native semantic evidence changes the frontier |
| local LongMemEval score | every haystack session was gold and there were no distractors | invalidate the score and fail dataset admission |

## the ceiling we can defend

on the MiniLM split, speaker-prefixed BM25S reaches 0.58806 Recall@10 before
selection, MiniLM reaches 0.63068, and the fixed top-25 candidate oracle is
0.67334. the remaining selector-oracle gap is 0.04266. on the separate BGE-M3
split, BM25S reaches 0.56799 before MaxSim, 0.59721 after it, and 0.67037 at the
candidate oracle. these are split-specific ceilings and should not be compared
as if they were one leaderboard.

the best evidence says candidate generation is still the larger unsolved
problem. rerankers can recover some evidence already in the pool. they cannot
repair missing evidence.

## resource truth

- every retained experiment artifact records zero provider calls and zero
  external experiment cost.
- BGE-M3 emitted 163,804 learned-sparse postings and 837,558,272 token-vector
  bytes for 5,881 turns. local encoding took 13.38 minutes on this CPU.
- MiniLM took 287.06 ms on average per 25-document pool on the same class of
  local CPU run.
- native 4096-d HNSW mechanics required much higher search effort than the
  previous default to approach exact neighbors.
- compacting duplicate raw embeddings saved 16,384 bytes per node and 48.50%
  of the measured SQLite file without changing logical vectors.

## code and measurement changes

the service now exposes executed retrieval stages, candidate counts, graph
anchors, reranker evidence, corpus generation, and resolved configuration.
`as_of` reaches dense, sparse, graph, cache, and final filtering. deduplication
preserves evidence identity. enabled optional stages fail closed. SQLite avoids
storing the same native vector twice when no distinct raw embedding exists.
HNSW controls are explicit. the LongMemEval runner refuses oracle-context
subsets. sparse documents can carry a source-derived speaker prefix without
changing evidence IDs.

verification on the final code state: 387 Python tests passed with 3 skipped;
the legacy verification suite passed 16; compilation and dependency integrity
passed; 4 TypeScript tests and Prettier passed. the frontend dependency repair
fetched registry tarballs but did not change `pnpm-lock.yaml`; it is not counted
as an experiment or provider call.

## what is still not true

- there is no defensible SOTA claim.
- there is no validated 10M-100M semantic retrieval result.
- there is no end-to-end grounded answer result.
- there is no transformer KV-cache replacement result.
- there is no independent LongMemEval retrieval corpus in the local checkout.
- there is no external-backend Pareto winner yet.

the next gates are narrow: acquire an independent exact-source memory corpus;
run the priced, preflight-bound native 4096-d semantic Flat-versus-HNSW test;
and test a compact category-aware selector against the current conventional
stack. until then, the simpler architecture wins.

## evidence and primary sources

local claims are bound to `experiments/results/claim-ledger-20260822.json`
(SHA-256 `6a12c941f10a538fdb5fd1d35a76385e9ea8a3a177d370d6abda94efc634cca3`).
the full 39-system prior-art ledger is
`docs/research/prior-art-mechanism-ledger.md`.

primary sources: TurboQuant (https://arxiv.org/abs/2402.18096), Vespa ranking
(https://docs.vespa.ai/en/ranking.html), ColBERTv2
(https://arxiv.org/abs/2112.01488), SPLADE++
(https://doi.org/10.1145/3477495.3531857), DiskANN
(https://github.com/microsoft/DiskANN), HippoRAG
(https://arxiv.org/abs/2405.14831), Graphiti
(https://github.com/getzep/graphiti), Matryoshka Representation Learning
(https://arxiv.org/abs/2205.13147), BGE-M3
(https://arxiv.org/abs/2402.03216), and LongMemEval
(https://arxiv.org/abs/2410.10813).
