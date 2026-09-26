# R1 — Reproducing EMG's LoCoMo retrieval numbers from the upstream code (offline)

Status: **REPRODUCTION** (not a new-method experiment). Locked before the full run.
Harness: `scripts/reproduce_emg_locomo.py`. Zero provider calls. No answer generation.

## Target

The paper is "Entity–Memory graph retrieval" (arXiv 2608.27925). The code is
github.com/Sun668/em_graph_memory, MIT, pinned at `f020e855be06ac9f33ec888945ff6b305d81cb07`
(tag v1.0.8). The paper's retrieval metric is the official LoCoMo `recall_acc` on LoCoMo-10
(1,986 QA rows). All the numbers below are percentages.

| Arm | What it retrieves | k=5 | k=10 | k=25 | k=50 |
|---|---|---:|---:|---:|---:|
| A | Dense only, memory-only graph, full pool | 59.3584 | 68.9145 | 79.7468 | 86.7148 |
| B | Entity gate + ±1 sequence, 0.30·E + 0.70·S | 64.1667 | 74.4847 | 84.4842 | 90.3306 |
| B_embed | Dense over B's Memory nodes (control) | | | 79.7468 | |
| B_entity | Entity score only (1.0/0.0) + sequence | | | 67.0208 | |
| B_noseq | B without sequence expansion | | | 82.8114 | |
| B_gate | Dense ranking inside the entity gate | | | 79.2853 | |
| B_gate_seq | B_gate + sequence expansion | | | 79.7217 | |

- B−A at k=5/10/25/50 is +4.8083/+5.5702/+4.7374/+3.6159.
- On the categories 1–4 subset, recall at k=25 goes from 82.8099 (A) to 85.0232 (B).
- The published values match the shipped `outputs/locomo_formal/<run>/stats.json`. The harness
  re-derives them by replay (see Metric).

Arms and flags are read from the retrieval block in each formal run's `run_config.json`. The
harness requires that block to equal `run.py:VARIANTS[variant]["recall"]` at the pinned commit.
For A@50 the formal run is `M4_A_top50_retry_e7a5abf`, which is the paper-valid rerun. v58 was
diagnostic only.

## Inputs (all shipped, all hashed)

- `data/locomo10.json`: sha256 must be `047d8e25…4d74`.
- Query vectors: `outputs/em_graph/query_embeddings/locomo10_047d8e25_text-embedding-3-small_v1.npz`.
  - Its sha256 must equal the one recorded in the formal run_config (`bef99a91…`).
  - Strict lookup: a miss raises.
- Per conversation, the shipped `gpt35_tes` caches:
  - `conv-*_em_graph_extract_v4_gpt35_tes.json`
  - `conv-*_em_graph_memory_only_gpt35_tes.json`
  - `conv-*_memory_emb_{extract_v4,memory_only}_gpt35_tes_text-embedding-3-small.npz`
  - `conv-*_qkeys_gpt35_tes.json`

Known before running: these are **not** byte-identical to the caches the formal runs used.
- run_config records `graphs/<identity-sha>.json`, `embedding_indexes/<identity-sha>.npz` and
  `entities/question_entities.json`. None of these are in the public repo. The shipped graph
  file hashes differ from the recorded ones.
- The shipped files carry the name of the pre-refactor "matched publish stack" (`gpt35_tes`).
  The author's PLAN/TODO record that stack as invalidated and rebuilt on 2026-07-27.
- The harness therefore measures three things:
  1. How close the upstream code gets on the shipped caches.
  2. Whether the upstream graph builder reproduces the formal memory-only graphs byte-for-byte.
     This build needs no LLM.
  3. Ranking identity against both the formal predictions and the shipped legacy answer
     checkpoints.

## Metric (official `recall_acc`)

- Per row, the metric is the fraction of the raw gold `evidence` strings that appear in the
  top-k `dia_id` context, rounded to 3 decimals. This follows `vendor/locomo/task_eval/
  evaluation.py` and the upstream evaluator.
- A row with empty evidence contributes 0 (`evaluation_stats.analyze_aggr_acc`).
- Overall = Σ contributions / **1,986**, which counts every row. By category = Σ / category
  count. The categories 1–4 subset = Σ / 1,540.
- The mean over the 1,982 non-empty rows is also reported, but only as a secondary figure; the
  published numbers use the 1,986 denominator.
- The category mapping is the official one: 1 = multi-hop, 2 = temporal, 3 = open-domain,
  4 = single-hop, 5 = adversarial.
- Metric check (must pass before any comparison):
  - Replay the shipped formal `predictions.json` contexts through our implementation.
  - The overall and per-category values must equal the shipped `stats.json` within 1e-9.
  - Every row must equal the serialized `…_recall` field.

## Pass criteria (fixed now)

- **Exact reproduction:** for every (arm, k), ordered top-k context identity with the formal
  predictions is 1,986/1,986, and |ours − formal| ≤ 0.05 recall points (percentage points,
  about one question).
- **Numeric reproduction:** |Δ| ≤ 0.05 points for every (arm, k), with identity below 100%.
- **Qualitative reproduction:** all three of the following hold.
  - B − A at k=25 is > 0 and within ±1.0 point of +4.7374.
  - B − A > 0 at k = 5, 10 and 50.
  - The k=25 ordering B > B_noseq > max(A, B_gate, B_gate_seq) > B_entity holds.
- Otherwise the result is **not reproduced**. The gap must be attributed to named causes such as
  inputs, code or metric.

## What counts as a deviation

Any difference from `formal_graph.py`'s path counts. Each one is listed in `DEVIATIONS` in the
harness:
- different input caches;
- bypassed identity checks;
- a replaced loader;
- stubbed modules;
- how question keys are served.

Platform differences also count and are recorded: Windows, Python/NumPy/rank_bm25 versions,
the size of the NLTK stopword list, and `PYTHONHASHSEED`.
