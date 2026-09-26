# E3 analysis — EXPLORATORY (LoCoMo; λ tuned on dev half, frozen, scored once on held-out)

Artifacts: `experiments/results/offline-locomo-e3-propagation-{dev,heldout}-20260925.json`.
Dev = conv-41/49/42/43/50 (n=1,089); held-out = conv-30/47/26/44/48 (n=888). 0 calls.

Dev selection: prop0.7 best mean answerable complete coverage for both key types
(spk_cap 0.7069 vs nbr1 0.6918, turn 0.6511).

Held-out, answerable (cats 1–4) complete coverage:

| keys | B | turn | nbr1 | prop0.7 |
|---|---|---|---|---|
| spk_cap | 512 | 0.516 | 0.565 | 0.586 |
| spk_cap | 1024 | 0.576 | 0.648 | 0.660 |
| spk_cap | 2048 | 0.638 | 0.722 | 0.720 |
| spk_cap | 4096 | 0.705 | 0.772 | 0.785 |
| spk_cap_obs | 2048 | 0.720 | 0.759 | 0.762 |

Locked criterion at 2,048: spk_cap **fails narrowly** (ties nbr1 −0.002; single-hop 0.881 vs
0.894 exceeds the 0.01 tolerance by 0.003); spk_cap_obs **passes**. Propagation removes the
fixed window's temporal regression (0.742 → 0.761) and multi-hop cost (0.230 → 0.257) and wins
at 512/1024/4096, but does not strictly dominate at 2,048.

Interpretation: the conversational neighbourhood is the mechanism (+0.08 over top-k at every
budget); propagation is a budget-allocation refinement worth carrying to confirmation as a
secondary arm, not a headline. λ=0.7 is frozen for LongMemEval.
