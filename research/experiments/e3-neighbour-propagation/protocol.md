# E3 — Neighbour score propagation: letting the budget choose depth vs breadth (LoCoMo)

Status: **EXPLORATORY** (LoCoMo inspected). λ is tuned on a 5-conversation dev half and frozen
before the other half is scored; the frozen λ is then carried unchanged to LongMemEval (E4,
confirmatory). Locked before running. Zero provider calls.

## Failure being targeted

E1/E2 show a budget conflict: fixed ±1 windows help local evidence (single-hop +0.14–0.16) but
spend tokens that cross-session aggregation needs (multi-hop coverage: turn 0.315 vs window
0.272 at 2,048 with fact keys). Runtime routing may not use gold categories.

## Intervention

`prop{λ}`: score'(j) = max(s(j), λ · max(s(j−1), s(j+1))) within a session, where s is the
BM25S score; pack single turns by score' (chronological tie-break). A neighbour enters only when
its inherited score beats the next distinct hit. This is one-hop score diffusion along the
conversation chain — the role HybridMind's `next_turn` edges are meant to play.

Grid λ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}; also `top3nbr` (±1 windows for the three best hits,
then single turns) as a simpler rule-based alternative.

## Selection rule (dev half only)

Maximise mean complete-evidence coverage over answerable categories (1–4) averaged across
B ∈ {1024, 2048, 4096} on the dev half; ties → smaller λ.

## Success criterion (held-out half, still exploratory)

The selected strategy must (a) match or beat both `turn` and `nbr1` on answerable complete
coverage at 2,048, and (b) be within 0.01 of the better of the two on *each* of single-hop and
multi-hop — i.e. it removes the depth/breadth trade-off rather than averaging it. Reported for
`spk_cap` keys (production-realisable) and `spk_cap_obs` keys (diagnostic).

## Split

Conversations sorted by sha256("20260925-e3:" + sample_id); first five = dev, last five =
held-out (same scheme as the existing sparse scripts, new seed).
