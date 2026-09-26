# R1 — analysis: EMG LoCoMo retrieval reproduction (offline)

Status: **qualitative reproduction** under the locked protocol (`protocol.md`). The numbers are not reproduced exactly or numerically. The single cause of the dense-arm gap is identified, and the direction and size of the entity-fusion gain are reproduced.

- Harness: `scripts/reproduce_emg_locomo.py` (defaults).
- Results: `experiments/results/repro-emg-locomo-20260926.json`.
- Sensitivity run: `--memory-vectors extract_v4` → `experiments/results/repro-emg-locomo-20260926-vectors-extract_v4.json`.
- Upstream: `Sun668/em_graph_memory` @ `f020e855be06ac9f33ec888945ff6b305d81cb07` (checked with `git rev-parse HEAD`), MIT.
- Environment: Windows 11, Python 3.13.5, NumPy 2.4.3, rank_bm25 0.2.2, NLTK 3.9.4.
- Provider calls: **0**.
  - The network was sealed: OpenAI constructors, socket connect and `nltk.download` all raise.
  - Frozen query artifact: 25,884 hits, 0 misses.
  - 75 artifacts are sha256-hashed in the result file.
- Runtime: about 2.5 minutes for the full 13-cell run.

## 1. Metric check (passed before any comparison)

We replayed each of the 13 formal `predictions.json` context lists through `official_recall`. Every overall and per-category value matched the shipped `stats.json` to within 1.4e-16. All 1,986 per-row `…_recall` fields matched, and every context held exactly k ids.

Denominators:
- 1,986 rows in total.
- 1,982 rows with non-empty evidence.
- 1,540 rows in categories 1–4.

The published numbers are **Σ row recall / 1,986**. The four empty-evidence rows add 0 but still count in the denominator (`analyze_aggr_acc`). The mean over the 1,982 non-empty rows is a different number (formal A@25: 79.9078, not 79.7468). The survey note (`tmp/survey/conv-graph.md` §2) says "averaged over the 1,982 rows"; that is the wrong denominator for the published values. The formal replay equals the published table in every cell.

## 2. Measured vs published

The metric is overall recall_acc in percent, over all 1,986 rows. "Formal" is the shipped formal run replayed, and it equals the published value in every cell.

Column definitions:
- **Δ pub:** ours − published.
- **ordered = / set =:** rows (out of 1,986) whose top-k context matches the formal run in the same order / as the same set.
- **legacy ordered =:** rows (out of 1,974 unique questions) whose ordered top-k matches the shipped pre-refactor answer checkpoint for that arm.

| Arm | k | Ours | Published | Δ pub | ordered = | set = | legacy ordered = |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 5 | 56.7468 | 59.3584 | −2.6116 | 22 | 127 | 1,961 |
| A | 10 | 67.0605 | 68.9145 | −1.8540 | 0 | 7 | 1,918 |
| A | 25 | 78.3294 | 79.7468 | −1.4174 | 0 | 0 | 1,687 |
| A | 50 | 85.4047 | 86.7148 | −1.3101 | 0 | 0 | 1,239 |
| B | 5 | 62.5235 | 64.1667 | −1.6432 | 26 | 164 | 612 |
| B | 10 | 73.3068 | 74.4847 | −1.1779 | 1 | 11 | 366 |
| B | 25 | 82.9763 | 84.4842 | −1.5079 | 0 | 0 | 182 |
| B | 50 | 89.2695 | 90.3306 | −1.0611 | 0 | 0 | 68 |
| B_embed | 25 | 78.3294 | 79.7468 | −1.4174 | 0 | 0 | 1,687 |
| B_entity | 25 | 66.0890 | 67.0208 | −0.9318 | 7 | 14 | 143 |
| B_noseq | 25 | 81.3190 | 82.8114 | −1.4924 | 0 | 0 | 223 |
| B_gate | 25 | 77.8426 | 79.2853 | −1.4427 | 0 | 0 | — |
| B_gate_seq | 25 | 78.3294 | 79.7217 | −1.3923 | 0 | 0 | — |

**B − A.**

| k | Ours | Published | Difference |
|---|---:|---:|---:|
| 5 | +5.78 | +4.81 | +0.97 |
| 10 | +6.25 | +5.57 | +0.68 |
| 25 | +4.65 | +4.74 | −0.09 |
| 50 | +3.86 | +3.62 | +0.24 |

**Categories 1–4 subset at k=25.**

| | A | B | B − A |
|---|---:|---:|---:|
| Ours | 80.2352 | 83.0460 | +2.81 |
| Published | 82.8099 | 85.0232 | +2.21 |

**Per category at k=25, ours (formal in parentheses).**

| Category | A | B |
|---|---|---|
| 1 multi-hop | 60.96 (65.16) | 62.12 (67.09) |
| 2 temporal | 88.89 (89.67) | 90.76 (89.62) |
| 3 open-domain | 46.75 (48.28) | 51.62 (55.19) |
| 4 single-hop | 87.22 (90.05) | 90.71 (92.69) |
| 5 adversarial | 71.75 (69.17) | 82.74 (82.62) |

**Verdict against the protocol's criteria.**
- **Exact reproduction:** fails. Ordered identity is 0–26 of 1,986 rows per cell.
- **Numeric reproduction:** fails. |Δ| is 0.93–2.61 points, against a limit of 0.05.
- **Qualitative reproduction:** holds, on all three conditions.
  - B − A at k=25 is +4.65, inside +4.74 ± 1.0.
  - B − A is positive at k = 5, 10 and 50.
  - The k=25 ordering holds: B (82.98) > B_noseq (81.32) > max(A 78.33, B_gate 77.84, B_gate_seq 78.33) > B_entity (66.09).

B_gate_seq equals A to four decimals. This is not a bug: in both our run and the formal runs, 1,968 of 1,986 rows have identical A and B_gate_seq rankings. The ±1 sequence-expanded gate almost always contains the dense top 25.

## 3. Gap diagnosis

Every input to arm A is identical to the formal run except the memory vectors. That makes the dense-arm gap attributable to a single cause.

| Input | Status vs the formal run |
|---|---|
| Code | Pinned upstream `run.VARIANTS`, `EMGraphRecall`, `retrieve_dialog_ids`, `MemoryEmbeddingIndex.scores`. Every formal `run_config` retrieval block equals the pinned `VARIANTS` (it is missing `semantic_score_normalization`, which defaults to `none`). |
| Memory-only graph (A) | **Identical.** The upstream `build_graph` rebuild matches the formal identity file name and the recorded sha256 (LF-normalized) for 10/10 conversations. |
| Query vectors | **Identical.** The same frozen artifact (sha256 `bef99a91…`, which equals the formal record) passes the upstream `validate_exact_dataset` with a strict lookup. |
| Memory vectors | **Different.** The formal `embedding_indexes/<sha>.npz` (one index per conversation, shared by A and B: 10/10) is not published. The shipped legacy v2 npz digests agree with the upstream `memory_search_text` digests on **0/5,882** memories. The shipped legacy graphs' `text_normalized` agrees with the post-refactor normalization on only 1,010/5,882 memories. The legacy text rewrote pronouns and time words ("Caroline went to a LGBTQ support group 7 May 2023"). The formal text keeps the verbatim turn and annotates it ("I went to a LGBTQ support group yesterday [7 May 2023]"). |
| Entity graph (B arms) | **Different.** The shipped `extract_v4_gpt35_tes` graph sha256 matches the formal record for 0/10 conversations. |
| Question keys (B arms) | **Different source.** The shipped legacy index-keyed `qkeys` are used; the formal `question_entities.json` is not published. All 1,986 rows have keys. |
| Metric | Identical (§1). |

Attribution:
1. **The A, B_embed, B_gate and B_gate_seq gaps (−1.3 to −2.6) come from the memory vectors.** Graph, queries, code and metric are verified identical for A. Our dense ranking also reproduces the pre-refactor stack's dense checkpoint on 1,961/1,974 rows at k=5 and 1,687 at k=25 (1,949 as a set). The residual swaps fit the legacy stack's live query embeddings versus the frozen artifact. So the shipped vectors are the legacy stack's vectors, and they score lower than the formal (post-refactor text) index. Which of the two shipped vector files is used does not matter: the extract_v4 sensitivity run is identical everywhere except A@50 (85.3544) and B@50 (89.2947).
2. **The B_entity gap (−0.93) comes from the entity graph and question keys, not the vectors.** With weights 1.0/0.0, both the gate ranking and the fill order are independent of the vectors. Its ordered identity with the formal run is 7/1,986, which confirms that the shipped legacy graph and keys differ from the formal ones.
3. **The B-family gaps combine both causes.** They largely cancel in the contrast: B − A at k=25 is off by only −0.09 points. At k=5 and k=10 our contrast is larger than the paper's (+0.97, +0.68) because our dense baseline is weaker.
4. **The legacy B checkpoints are not a formal target.** Identity of B-family arms with those checkpoints is low (182/1,974 for B at k=25). They were produced by the pre-refactor retrieval code (v01 `em_graph/retrieval.py`), which is not in the public repo, and before `RETRIEVAL_SCORE_VERSION` became `protocol_bound_semantic_gated_fill_topk_v3`.

Ruled out:
- **NLTK stopword-list version.** Our list has 198 words, but only 153 of them are `[a-z0-9]+` tokens. The other 45 are all apostrophe contractions, which the upstream tokenizer can never emit, so a different list version cannot change the BM25 tokens.
- **Platform and PYTHONHASHSEED.** Every upstream sort has an explicit tie-break (score, then `dia_id`; entity id for BM25).

## 4. What this establishes, and the baseline going forward

- **Reproduced:** The mechanism claim holds on this executable stack. Entity-score fusion adds +4.65 recall points at k=25 (published +4.74), and the gain is positive at every k. The gate alone adds nothing: B_gate is at or below A. Entity-only is about 12 points below dense.
- **Not reproduced:** The absolute published values. Reproducing them needs the formal post-refactor memory index, graphs and question keys. Rebuilding those would take text-embedding-3-small and gpt-3.5-turbo calls, which are out of scope under the zero-provider rule.
- **Best executable baseline for the `--port` comparison:** the upstream code on the shipped inputs, which is this run.
  - k=25: A 78.3294, B 82.9763.
  - k=5/10/50: A 56.7468/67.0605/85.4047; B 62.5235/73.3068/89.2695.
  - A HybridMind port of `engine/entity_graph.py` should be judged against these numbers and against per-row ranking identity with this run, via `load_inputs(conv)` and `official_recall(rows, k)`. The published numbers are not the right yardstick, because the port will see the same legacy vectors, graphs and keys.

## Port parity and HybridMind variants (`--port`)

- Command: `python scripts/reproduce_emg_locomo.py --port` (all 10 LoCoMo conversations, `memory_only` vectors).
- Output: `port`, `variants` and `port_provenance` keys in `experiments/results/repro-emg-locomo-20260926.json`. The existing keys are unchanged.
- Provider calls: **0**. Frozen query artifact: 37,818 hits, 0 misses. Runtime about 4.5 min.
- Inputs are the same reference artifacts as the reproduction:
  - shipped LLM graphs (extract_v4, gpt35_tes), loaded with `EntityMemoryGraph.from_emg_json`;
  - shipped question keys;
  - cached text-embedding-3-small memory vectors. This is the 1536-d reference track, not the 4096-d runtime.

### Port parity (`engine/entity_graph.py` vs upstream EMG code)

Upstream rankings come from `retrieve_dialog_ids`, wired as in `EMGraphRecall.recall`. Their recall equals the reproduction's `results` cell in all 10 cells.

| Configuration | Question rankings ordered-identical | Float-tie mismatches | Other mismatches | Max abs score diff | Official metric equal |
|---|---|---|---|---|---|
| Port fed upstream `MemoryEmbeddingIndex.scores`, dia_id tie-break | **19,860 / 19,860** (all 10 cells) | 0 | 0 | 2.1e-7 to 2.5e-7 | all 10 cells, every row |
| Port fed HybridMind `dense_channel.exact_scores`, chronological tie-break | 17,876 / 19,860 | 1,984 | **0** | 6.0e-7 | 9/10 cells (not B_entity@25) |

- Cells: A@5/10/25/50, B@5/10/25/50, B_entity@25, B_noseq@25.
- A mismatch counts as a float tie when the position-by-position scores and the scores of shared ids agree within 1e-6.
- The port's logic is exact, and no change to `engine/entity_graph.py` was needed.

The two remaining kinds of difference are both deliberate:

1. **Dense arithmetic (12 swaps).** These are in A@25/50, B@25/50 and B_noseq@25, and recall is unchanged.
   - Upstream scores the gated pool with a float32 BLAS product over the subset rows, so a memory's S changes by ulps depending on the gate's members and set order. The max diff moves with PYTHONHASHSEED (2.1e-7 vs 2.5e-7 across runs).
   - HybridMind renormalises the rows and uses einsum, which is deterministic and does not depend on the subset.
2. **Tie-break order (1,972 questions, all in B_entity@25).**
   - EMG's degree-discounted E gives every memory that mentions an entity the same score, so ties are everywhere.
   - dia_id string order and chronological order therefore choose different memories at the k cutoff.
   - Top-25 sets are identical for 796/1,986 questions and row recall is equal for 1,950/1,986.
   - Overall recall is 66.09 vs 65.90 (-0.19 points); categories 1-4 are 64.00 vs 63.84.
   - `upstream_order=True` reproduces EMG exactly.

Agreement with the shipped formal predictions stays low, for example 0/1,986 ordered-identical for A@25 and B@25. This is expected: the formal runs used vectors and graphs that were not published (see the reproduction section). Against the legacy checkpoints, A@5 is ordered-identical for 1,961 of 1,986 questions.

### HybridMind variants on EMG reference artifacts (not reproductions; exploratory, not preregistered)

Metric: official recall, overall / categories 1-4. The CI is a paired conversation-cluster percentile bootstrap (10 clusters, seed 42, 2,000 resamples) of the per-row difference, with empty-evidence rows counted as 0.

References, both upstream code on the same artifacts:
- A: 56.75/58.73 (k=5), 67.06/69.08 (k=10), 78.33/80.24 (k=25), 85.40/86.89 (k=50).
- B (vs A): 62.52/62.68 (+5.78 [4.04, 7.43]), 73.31/73.08 (+6.25 [5.01, 7.49]), 82.98/83.05 (+4.65 [3.52, 5.74]), 89.27/89.08 (+3.86 [3.03, 4.76]).

| Variant | k | Recall (all / cat 1-4) | vs A overall [95% CI] | vs B overall [95% CI] | vs B cat 1-4 [95% CI] |
|---|---|---|---|---|---|
| PPR graph-only | 5 | 39.33 / 37.92 | -17.42 [-20.76, -14.67] | -23.20 [-26.21, -20.49] | -24.75 [-27.67, -22.18] |
| | 10 | 49.30 / 47.67 | -17.76 [-21.53, -14.36] | -24.00 [-27.24, -21.21] | -25.41 [-28.29, -22.91] |
| | 25 | 62.62 / 60.78 | -15.71 [-20.16, -11.68] | -20.36 [-24.37, -16.95] | -22.26 [-25.59, -19.72] |
| | 50 | 69.66 / 67.65 | -15.75 [-20.57, -11.32] | -19.61 [-24.23, -15.45] | -21.43 [-25.87, -17.62] |
| PPR + dense passage seeds x0.05 | 5 | 52.85 / 51.57 | -3.89 [-5.98, -1.84] | -9.67 [-11.63, -8.08] | -11.11 [-13.43, -9.34] |
| | 10 | 71.03 / 69.85 | +3.97 [2.44, 5.77] | -2.27 [-3.52, -1.14] | -3.22 [-4.30, -2.28] |
| | 25 | 83.50 / 83.17 | +5.17 [4.05, 6.14] | +0.53 [-0.57, 1.32] | +0.13 [-0.72, 0.76] |
| | 50 | 89.99 / 89.59 | +4.58 [3.39, 5.94] | +0.72 [-0.20, 1.70] | +0.50 [-0.51, 1.39] |
| Lexical graph, E only | 5 | 38.66 / 37.19 | -18.09 [-20.41, -15.34] | -23.87 [-25.43, -21.90] | -25.49 [-26.95, -23.68] |
| | 10 | 49.97 / 47.43 | -17.09 [-20.11, -13.60] | -23.33 [-25.50, -20.81] | -25.64 [-28.05, -22.91] |
| | 25 | 66.06 / 63.60 | -12.27 [-17.10, -8.30] | -16.92 [-21.12, -13.39] | -19.44 [-23.27, -16.39] |
| | 50 | 74.62 / 72.08 | -10.78 [-15.30, -6.65] | -14.65 [-18.58, -11.12] | -17.01 [-20.80, -13.74] |
| Lexical graph, 0.30 E + 0.70 S | 5 | 61.17 / 61.19 | +4.43 [3.02, 6.02] | -1.35 [-2.93, 0.18] | -1.48 [-3.25, 0.22] |
| | 10 | 71.73 / 71.41 | +4.67 [3.53, 5.87] | -1.57 [-2.51, -0.66] | -1.67 [-2.74, -0.60] |
| | 25 | 82.43 / 82.11 | +4.10 [3.14, 4.93] | -0.55 [-1.45, 0.34] | -0.94 [-1.71, -0.25] |
| | 50 | 88.89 / 88.50 | +3.49 [2.84, 4.19] | -0.38 [-1.20, 0.42] | -0.59 [-1.48, 0.35] |

How the variants were set up:
- **PPR.** HippoRAG-2 seeds: qkey match strength divided by the number of memories mentioning the entity, for the top 5 entities. Damping 0.5, NEXT/PREV edges on, positive mass only.
- **Passage seeds.** min-max dense score x 0.05 on every memory.
- **Lexical graph.** `LexicalEntityExtractor` (both speakers known) runs over EMG's extraction text (`text_normalized` plus ` [Image: caption]`). Question keys come from `extract_query`.
  - Totals: 14,993 entities and 35,937 mention edges, vs 12,748 and 34,402 for the LLM graph.
  - Query keys: mean 3.95 per question vs 3.70; the key set equals the shipped LLM key set for 594/1,986 questions.
- **Empty rankings.** Graph-only variants return no fill, so questions with nothing matched retrieve nothing: 1 for PPR graph-only, 2 for lexical E.
- **Prefix scoring.** Each variant ranks once at k=50 and is scored at smaller k by truncation; the rankings are prefix-consistent.

Reading:
1. **PPR on EMG's bipartite graph alone is a poor retriever.** It is 15.7 to 17.8 points below dense A at every k, and at k=25 it is also below EMG's own entity-only E (62.62 vs 66.09).
2. **PPR with a small dense passage prior matches EMG B at k>=25** (+0.53 [-0.57, 1.32] at k=25) but is clearly worse at k<=10. The 0.05 passage weight is too weak to order the head of the list.
3. **A zero-LLM lexical extractor recovers about 88% of EMG's k=25 gain over dense** (+4.10 of +4.65). Overall it is not distinguishable from B at k=25 and k=50, but it is slightly worse on categories 1-4 at k=25 (-0.94 [-1.71, -0.25]) and at k=10 (-1.57 overall).

Caveats:
- The vectors are the legacy 1536-d reference track.
- There are 10 clusters, so the CIs are wide.
- The variants were exploratory and not in `protocol.md`, so none of this is a runtime (4096-d) result.
