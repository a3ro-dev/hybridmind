"""Token-budgeted evidence assembly over a ``ScopedCorpus``.

Turns a fused ranking into the context a reader receives: optional
same-session expansion (neighbour windows, one-hop score propagation, whole
sessions), greedy packing under a rendered-token budget, and explicit roles so
evidence recall is never inflated. A ranked turn is a ``hit`` with its rank; a
turn admitted only because a hit pulled it in is ``context`` and names the hit
it was ``expanded_from``. Nothing becomes a hit unless the caller ranked it.

Ported from HybridMind's offline E1-E4 harness
``scripts/offline_budgeted_evidence.py`` (``ntok``, ``render``,
``Conv.propagated``, ``pack``, ``strategy_units`` for turn / nbr{w} / prop{lam}
/ session) - the code behind research/experiments/e1-budgeted-units and
e3-neighbour-propagation. Same repository and license; no third-party code.
``rank_sessions(method="dcg")`` is a HybridMind reimplementation of the session
grouping described for Emergence "Simple"
(https://github.com/EmergenceAI/emergence_simple_fast has no license, so no
code was copied).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from engine.corpus import ScopedCorpus, TurnRecord

STRATEGIES = ("turn", "window", "propagate", "session")
SESSION_METHODS = ("max", "sum", "dcg")
ORDERS = ("chronological", "rank")

# render_turn reads these metadata keys (first non-empty wins).
DATE_KEYS = ("session_date", "date")  # falls back to TurnRecord.timestamp
CAPTION_KEYS = ("blip_caption", "caption")

DEVIATIONS: List[str] = [
    "propagate: signed score scales (dense cosine, z-score, DBSF; flagged by the caller or detected by a "
    "negative score) are min-max mapped to [0, 1] before propagation; BM25/RRF scores are used unchanged, as in E3.",
    "Units and costs are keyed by node_id instead of harness list positions; selections are identical "
    "(tests/test_fusion_evidence.py parity test).",
    "render_turn takes the date from metadata session_date/date, else TurnRecord.timestamp, and the caption "
    "from metadata blip_caption/caption; the harness read the LoCoMo/LongMemEval fields directly.",
    "Strategy 'session' ranks sessions from the fused turn ranking (rank_sessions max/sum/dcg); the harness "
    "ranked sessions with a separate session-level BM25S index. session_units + pack are faithful given the "
    "same session order.",
    "propagate_scores rejects negative or non-finite scores (the harness only ever saw positive BM25 scores, "
    "and one-hop max-propagation is not meaningful for negative scores).",
    "Strategy 'propagate' appends hits whose propagated score is 0 after the propagated units, in rank order, "
    "so an unbudgeted pack still contains every ranked hit (a no-op for positive scores).",
    "pack accepts budget=None meaning unlimited.",
    "Roles (hit/context), expanded_from, max_hits and rank/chronological item order are HybridMind additions; "
    "the harness returned only the selected index set.",
]

# Token proxy copied from scripts/offline_budgeted_evidence.py (_TOK / ntok):
# a declared regex proxy, no tokenizer download.
_TOK = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class EvidenceItem:
    """One packed turn. ``text`` is the rendered string the tokens were charged for."""

    node_id: str
    evidence_id: str
    role: str  # "hit" | "context"
    rank: Optional[int]  # 1-based hit rank; None for context
    score: Optional[float]  # hit: ranked score; propagated context: inherited score
    expanded_from: Optional[str]  # context only: the hit that admitted it
    tokens: int
    text: str


@dataclass(frozen=True)
class EvidencePack:
    items: Tuple[EvidenceItem, ...]
    packed_tokens: int
    budget_tokens: Optional[int]
    strategy: str
    dropped_hits: int  # ranked hits (within max_hits) that did not fit
    hit_ids: Tuple[str, ...]  # in item order
    context_ids: Tuple[str, ...]  # in item order


def token_count(text: str) -> int:
    """Same proxy as ``offline_budgeted_evidence.ntok``: words and punctuation marks."""
    return len(_TOK.findall(text))


def _meta(turn: TurnRecord, keys: Sequence[str]) -> str:
    for key in keys:
        value = turn.metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def render_turn(turn: TurnRecord) -> str:
    """``(date) speaker: text [shares image: caption]`` as in ``offline_budgeted_evidence.render``."""
    date = _meta(turn, DATE_KEYS) or (turn.timestamp or "")
    caption = _meta(turn, CAPTION_KEYS).strip()
    suffix = f" [shares image: {caption}]" if caption else ""
    return f"({date}) {turn.speaker}: {turn.text}{suffix}"


def _check_scores(corpus: ScopedCorpus, scores: Mapping[str, float]) -> None:
    for node_id, value in scores.items():
        if node_id not in corpus:
            raise ValueError(f"unknown node_id for this scope: {node_id!r}")
        if not math.isfinite(value):
            raise ValueError(f"score for {node_id!r} must be finite")


def _propagate(
    corpus: ScopedCorpus, scores: Mapping[str, float], lam: float
) -> Dict[str, Tuple[float, Optional[str]]]:
    """``{node_id: (score', source)}``; source is the neighbour that set score' (None = own score)."""
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("propagation lambda must be finite and non-negative")
    _check_scores(corpus, scores)
    if any(value < 0.0 for value in scores.values()):
        raise ValueError("score propagation requires non-negative scores")
    turns = corpus.turns
    out: Dict[str, Tuple[float, Optional[str]]] = {}
    for j, turn in enumerate(turns):
        own = scores.get(turn.node_id, 0.0)
        best, source = 0.0, None  # max(nb, default=0.0); ties keep the earlier neighbour
        for k in (j - 1, j + 1):
            if 0 <= k < len(turns) and turns[k].session_key == turn.session_key:
                value = scores.get(turns[k].node_id, 0.0)
                if value > best:
                    best, source = value, turns[k].node_id
        inherited = lam * best
        v = max(own, inherited)
        if v > 0:
            out[turn.node_id] = (v, None if own >= inherited else source)
    return out


def propagate_scores(corpus: ScopedCorpus, scores: Mapping[str, float], lam: float) -> List[Tuple[str, float]]:
    """One-hop same-session diffusion ``v = max(s, lam * max(neighbour s))`` (``Conv.propagated``).

    Returns every turn with ``v > 0`` sorted by ``ScopedCorpus.sort_scores``.
    """
    return corpus.sort_scores({node_id: v for node_id, (v, _) in _propagate(corpus, scores, lam).items()})


def window_units(corpus: ScopedCorpus, ranked_ids: Sequence[str], w: int) -> List[List[str]]:
    """``nbr{w}``: each hit plus up to ``w`` same-session turns on each side, chronological."""
    if not isinstance(w, int) or w < 0:
        raise ValueError("window must be a non-negative integer")
    return [corpus.same_session_neighbors(node_id, w) for node_id in ranked_ids]


def _session_members(corpus: ScopedCorpus) -> Dict[str, List[str]]:
    members: Dict[str, List[str]] = {}
    for turn in corpus.turns:
        members.setdefault(turn.session_key, []).append(turn.node_id)
    return members


def session_units(corpus: ScopedCorpus, ranked_sessions: Sequence[str]) -> List[List[str]]:
    """Every turn of each ranked session, chronological, in session rank order."""
    members = _session_members(corpus)
    unknown = [key for key in ranked_sessions if key not in members]
    if unknown:
        raise ValueError(f"unknown session keys for this scope: {unknown!r}")
    return [list(members[key]) for key in ranked_sessions]


def _admit(units: Sequence[Sequence[str]], costs: Mapping[str, int], budget: Optional[int]) -> Iterator[Tuple[int, List[str]]]:
    """Yield ``(unit_index, newly_admitted_ids)`` for each unit ``pack`` takes."""
    seen: set = set()
    used = 0
    for u, unit in enumerate(units):
        new = [node_id for node_id in unit if node_id not in seen]
        cost = sum(costs[node_id] for node_id in new)
        if not new or (budget is not None and used + cost > budget):
            continue
        seen.update(new)
        used += cost
        yield u, new


def pack(units: Sequence[Sequence[str]], costs: Mapping[str, int], budget: Optional[int]) -> List[str]:
    """Greedy in rank order; a unit that does not fit is skipped, not truncated."""
    return [node_id for _, new in _admit(units, costs, budget) for node_id in new]


def _check_ranked(corpus: ScopedCorpus, ranked: Sequence[Tuple[str, float]]) -> None:
    ids = [node_id for node_id, _ in ranked]
    if len(set(ids)) != len(ids):
        raise ValueError("ranked list contains duplicate node IDs")
    _check_scores(corpus, dict(ranked))


def rank_sessions(
    corpus: ScopedCorpus, ranked: Sequence[Tuple[str, float]], method: str = "max"
) -> List[Tuple[str, float]]:
    """Group ranked turns by session; ties break on the session's first chronological turn.

    ``max``/``sum`` aggregate member scores; ``dcg`` sums ``1 / log2(rank + 1)``
    over member hits (1-based ranks), a HybridMind reimplementation of the
    Emergence "Simple" session grouping.
    """
    if method not in SESSION_METHODS:
        raise ValueError(f"session method must be one of {SESSION_METHODS}")
    _check_ranked(corpus, ranked)
    agg: Dict[str, float] = {}
    for rank, (node_id, score) in enumerate(ranked, start=1):
        key = corpus.turn(node_id).session_key
        if method == "max":
            agg[key] = max(agg.get(key, -math.inf), score)
        elif method == "sum":
            agg[key] = agg.get(key, 0.0) + score
        else:
            agg[key] = agg.get(key, 0.0) + 1.0 / math.log2(rank + 1)
    first: Dict[str, int] = {}
    for index, turn in enumerate(corpus.turns):
        first.setdefault(turn.session_key, index)
    return sorted(agg.items(), key=lambda item: (-item[1], first[item[0]]))


def _positive_int(name: str, value: Optional[int]) -> None:
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer when set")


def assemble_evidence(
    corpus: ScopedCorpus,
    ranked: List[Tuple[str, float]],
    *,
    strategy: str = "turn",
    window: int = 1,
    lam: float = 0.7,
    budget_tokens: Optional[int] = None,
    max_hits: Optional[int] = None,
    order: str = "chronological",
    session_method: str = "max",
    signed_scores: bool = False,
) -> EvidencePack:
    """Expand and pack a ranking into an ``EvidencePack``.

    ``ranked`` is ``[(node_id, score)]`` best first; its first ``max_hits``
    entries (all when None) are the hits. Strategies build packing units:
    ``turn`` = each hit; ``window`` = hit +/- ``window`` same-session turns;
    ``propagate`` = single turns ordered by one-hop propagated score
    (``lam``); ``session`` = whole sessions ordered by ``rank_sessions``.
    Units are packed greedily under ``budget_tokens`` (rendered-token proxy);
    without a budget every hit is included. ``order`` sets item order:
    chronological, or admission (rank) order.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    _positive_int("budget_tokens", budget_tokens)
    _positive_int("max_hits", max_hits)
    _check_ranked(corpus, ranked)
    hits = list(ranked if max_hits is None else ranked[:max_hits])
    hit_rank = {node_id: rank for rank, (node_id, _) in enumerate(hits, start=1)}
    hit_score = {node_id: float(score) for node_id, score in hits}
    context_score: Dict[str, float] = {}

    if strategy == "turn":
        units = [[node_id] for node_id, _ in hits]
        heads: List[Optional[str]] = [node_id for node_id, _ in hits]
    elif strategy == "window":
        units = window_units(corpus, [node_id for node_id, _ in hits], window)
        heads = [node_id for node_id, _ in hits]
    elif strategy == "propagate":
        # Propagation is scale-invariant, so BM25/RRF scores (>= 0, zero = no
        # evidence) propagate exactly as in E3.
        # ``signed_scores`` marks score scales whose zero is not "no evidence"
        # (cosine, z-score, DBSF); those are always min-max mapped.
        low = min(hit_score.values(), default=0.0)
        if signed_scores or low < 0.0:
            span = max(hit_score.values()) - low
            hit_score = {n: ((v - low) / span if span > 0 else 0.0) for n, v in hit_score.items()}
        propagated = _propagate(corpus, hit_score, lam)
        ordered = [node_id for node_id, _ in corpus.sort_scores({n: v for n, (v, _) in propagated.items()})]
        ordered += [node_id for node_id, _ in hits if node_id not in propagated]
        units = [[node_id] for node_id in ordered]
        heads = [propagated[node_id][1] if node_id in propagated else None for node_id in ordered]
        context_score = {n: v for n, (v, _) in propagated.items() if n not in hit_rank}
    else:
        sessions = [key for key, _ in rank_sessions(corpus, hits, session_method)]
        units = session_units(corpus, sessions)
        best_hit: Dict[str, str] = {}
        for node_id, _ in hits:
            best_hit.setdefault(corpus.turn(node_id).session_key, node_id)
        heads = [best_hit[key] for key in sessions]

    rendered = {node_id: render_turn(corpus.turn(node_id)) for unit in units for node_id in unit}
    costs = {node_id: token_count(text) for node_id, text in rendered.items()}
    items: List[EvidenceItem] = []
    for u, new in _admit(units, costs, budget_tokens):
        for node_id in new:
            is_hit = node_id in hit_rank
            if not is_hit and heads[u] is None:
                raise RuntimeError(f"context turn {node_id!r} has no admitting hit")
            items.append(EvidenceItem(
                node_id=node_id,
                evidence_id=corpus.turn(node_id).evidence_id,
                role="hit" if is_hit else "context",
                rank=hit_rank.get(node_id),
                score=hit_score[node_id] if is_hit else context_score.get(node_id),
                expanded_from=None if is_hit else heads[u],
                tokens=costs[node_id],
                text=rendered[node_id],
            ))
    if order == "chronological":
        items.sort(key=lambda item: corpus.rank_key(item.node_id))
    hit_ids = tuple(item.node_id for item in items if item.role == "hit")
    return EvidencePack(
        items=tuple(items),
        packed_tokens=sum(item.tokens for item in items),
        budget_tokens=budget_tokens,
        strategy=strategy,
        dropped_hits=len(hits) - len(hit_ids),
        hit_ids=hit_ids,
        context_ids=tuple(item.node_id for item in items if item.role == "context"),
    )
