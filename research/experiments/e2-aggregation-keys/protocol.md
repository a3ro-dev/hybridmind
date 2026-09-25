# E2 — Can derived fact keys recover cross-session aggregation evidence? (LoCoMo, diagnostic)

Status: **EXPLORATORY / DIAGNOSTIC**. Locked before running. Harness:
`scripts/offline_budgeted_evidence.py` with a new `spk_cap_obs` representation. Zero provider calls.

## Failure being targeted

E1: multi-hop (LoCoMo cat. 1, cross-session aggregation) complete-evidence coverage is
0.12 / 0.19 / 0.29 at 1k / 2k / 4k tokens under every unit; neighbour expansion does not help.
Audit + inspection: 80.5% of missed gold turns rank below 25; the dominant miss is vocabulary
mismatch between third-person questions ("What does Melanie do to destress?") and first-person
turns ("I've been running farther to de-stress").

## Intervention (established: LongMemEval fact-augmented key expansion; doc2query)

Index key = speaker + turn text + caption + the dataset's `observation` facts linked to that turn
(third-person, speaker-named paraphrases). The *rendered value* and its token charge are
unchanged (the reader never sees the observation), so any gain is purely a key/recall effect.

**Leakage caveat (why this is only diagnostic):** the observations were produced by the
LoCoMo authors' LLM pipeline from the same sessions; if QA annotation drew on them, lexical
overlap is inflated. A positive result licenses paying for our own extraction; it is not a
HybridMind result.

## Prediction

`spk_cap_obs|nbr1` improves multi-hop complete coverage over `spk_cap|nbr1` by ≥ +0.05 at 2,048
tokens, with smaller gains elsewhere. If < +0.02, derived fact keys are not the lever and dense
semantic matching or query-side expansion become the next candidates.

## Controls

Same budgets, same packing, same render charge; baseline `raw|turn` (as E1) and the direct
comparator `spk_cap|nbr1`. Conversation-cluster bootstrap, descriptive.
