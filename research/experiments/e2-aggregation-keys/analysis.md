# E2 analysis — DIAGNOSTIC (dataset-provided observation keys; leakage-flagged)

Artifact: `experiments/results/offline-locomo-budgeted-evidence-e2-obskeys-20260925.json`.
All 10 conversations, n=1,977, 0 calls.

At 2,048 tokens, complete coverage (all / multi-hop / single-hop):
spk_cap|turn 0.678/0.247/0.771; spk_cap|nbr1 0.758/0.222/0.895;
spk_cap_obs|turn 0.748/0.315/0.860; spk_cap_obs|nbr1 0.790/0.272/0.925.

- Prediction (≥ +0.05 multi-hop for obs|nbr1 vs spk_cap|nbr1 at 2k): +0.050 — met exactly;
  with turn packing +0.068. Derived third-person fact keys attack the vocabulary gap.
- New observation: a depth/breadth budget conflict — multi-hop prefers distinct hits (turn),
  single-hop prefers neighbourhoods (nbr1). Motivates E3.
- Caveat stands: observations are LoCoMo-author LLM outputs; this licenses (does not replace)
  our own priced extraction run.
