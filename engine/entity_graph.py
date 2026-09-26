"""Entity-memory graph channel: EMG recall port plus HippoRAG-2-style PPR.

Graph (EMG, arXiv 2608.27925): one memory node per corpus turn, one entity
node per normalised key (``entity:{key}``), entity--memory ``mentions`` edges
(weight 1.0) and a GLOBAL chronological NEXT/PREV chain over memories. There
are no entity--entity edges.

Two rankers run over the same graph:

* EMG recall (``entity_scores`` -> ``expand_sequence_neighbors`` ->
  ``emg_rank`` / ``emg_fused_rank``): BM25 soft-match of query entity keys
  against entity documents, degree-discounted entity strength, Who-only
  dampening, +/-1 chronological expansion, and the reference
  ``0.30 * E + 0.70 * S`` gated fusion with dense fill.
* ``ppr_rank``: personalised PageRank with HippoRAG 2 seed weighting (node
  specificity, ``linking_top_k`` entities, optional passage seeds scaled by
  ``passage_node_weight``) and igraph damping semantics (continue prob. 0.5).

Port attribution (``tokenize_for_bm25``, ``EntityBM25Index``,
``normalize_semantic_scores``, entity scoring, sequence expansion, fused
ranking, entity identity/merge rules, ``memory_sort_key``, EMG JSON loading):
EMG, https://github.com/Sun668/em_graph_memory, commit
f020e855be06ac9f33ec888945ff6b305d81cb07, files code/em_graph/recall/retrieval.py,
code/em_graph/recall/entity_bm25_index.py, code/em_graph/recall/tokenize.py,
code/em_graph/build/builder.py, code/em_graph/build/models.py.
SPDX-License-Identifier: MIT. Copyright (c) 2026 Sun668.

PPR seed weighting and parameters follow HippoRAG 2,
https://github.com/OSU-NLP-Group/HippoRAG, paper code commit
191f281122e2437743daf1dd2dae23059a14c088, src/hipporag/HippoRAG.py
(``graph_search_with_fact_entities``, ``get_top_k_weights``, ``run_ppr``) and
src/hipporag/utils/misc_utils.py (``min_max_normalize``).
SPDX-License-Identifier: MIT. Copyright (c) 2025 OSU Natural Language
Processing (LICENSE of HippoRAG HEAD 1438aba; the 191f281 snapshot here has
only src/). The PageRank solver itself is an original scipy power iteration
(igraph/PRPACK is GPL and is not used).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from engine.corpus import ScopedCorpus, TurnRecord
from engine.entity_extraction import NLTK_ENGLISH_STOPWORDS, normalize_entity_key

DEVIATIONS = [
    "Memories are keyed by corpus node_id (from_emg_json maps 'memory:{dia_id}' "
    "through node_id_format) and rankings return node_ids, not dia_ids; the "
    "dia_id stays the TurnRecord.evidence_id.",
    "Ranking ties break by corpus chronological index (engine.corpus contract); "
    "EMG breaks ties by dia_id string. upstream_order=True restores EMG's key.",
    "Per-entity match strength sums query keys in sorted key order; EMG sums "
    "in Python set order, which depends on PYTHONHASHSEED.",
    "from_corpus attaches mentions in chronological turn order; EMG attaches "
    "in ThreadPoolExecutor completion order, so which type a merged entity "
    "keeps (first seen wins) is nondeterministic upstream.",
    "from_corpus includes every corpus turn as a memory and uses "
    "EntityMention.key (normalize_entity_key(value) in both extractors) as the "
    "entity identity instead of recomputing it in the builder.",
    "tokenize_for_bm25 uses the vendored NLTK English list (198 words) and "
    "always stems; EMG downloads the list at runtime and silently degrades to "
    "a 33-word list / no stemming when NLTK is unavailable.",
    "Semantic scores must be finite (raise otherwise); EMG checks finiteness "
    "only under query-local min-max calibration.",
    "emg_rank (graph-only, E alone) returns only positive candidates unless "
    "fill=True, which reproduces EMG's entity-only (1.0/0.0) fill to top_k.",
    "PPR seeds come from BM25 entity-key match strength (EMG strength) instead "
    "of HippoRAG 2 fact-embedding scores and an LLM recognition filter; no "
    "synonymy edges. Seed = strength / |memories mentioning e|; the top "
    "linking_top_k entities are selected by strength (HippoRAG: mean fact score).",
    "PPR optionally includes the NEXT/PREV chain as ONE undirected weight-1.0 "
    "edge per adjacent pair (HippoRAG graphs have no sequence edges; adding "
    "both stored NEXT and PREV to an undirected igraph graph would weigh 2.0).",
    "PPR is solved by scipy power iteration (L1 tol 1e-10, max_iter 1000, "
    "raises if not converged) instead of igraph PRPACK; dangling nodes "
    "teleport by the reset vector (networkx default; igraph passes the reset "
    "vector as PRPACK's dangling distribution). A query with zero seed mass "
    "returns [] instead of HippoRAG's assertion error.",
]

WHO_ONLY_DAMPEN = 0.25
SEQUENCE_SECONDARY_SCALE = 0.5
DEFAULT_ENTITY_WEIGHT = 0.30
DEFAULT_SEMANTIC_WEIGHT = 0.70
DEFAULT_ENTITY_MIN_REL_SCORE = 0.5
DEFAULT_ENTITY_TOP_K_PER_KEY = 20
SEMANTIC_SCORE_NONE = "none"
SEMANTIC_SCORE_QUERY_MINMAX = "query_local_minmax_v1"
SEMANTIC_SCORE_NORMALIZATIONS = {SEMANTIC_SCORE_NONE, SEMANTIC_SCORE_QUERY_MINMAX}

PPR_DAMPING = 0.5
PPR_LINKING_TOP_K = 5
PPR_PASSAGE_NODE_WEIGHT = 0.05
PPR_TOL = 1e-10
PPR_MAX_ITER = 1000

SemanticScoreFn = Callable[[Optional[Set[str]]], Mapping[str, float]]
Ranking = List[Tuple[str, float]]

# --------------------------------------------------------------------------- #
# Tokenisation (EMG recall/tokenize.py)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_STEMMER = PorterStemmer()


@lru_cache(maxsize=65536)
def _stem(token: str) -> str:
    return _STEMMER.stem(token)


def tokenize_for_bm25(text: str) -> List[str]:
    """Stopword-filtered Porter stems for BM25 (list, keeps duplicates for TF)."""
    out: List[str] = []
    for tok in _TOKEN_RE.findall(str(text or "").lower()):
        if len(tok) <= 1 or tok in NLTK_ENGLISH_STOPWORDS:
            continue
        out.append(_stem(tok))
    return out


# --------------------------------------------------------------------------- #
# Entity BM25 index (EMG recall/entity_bm25_index.py)
# --------------------------------------------------------------------------- #


def _entity_doc_text(key: str, value: str) -> str:
    key = str(key or "").strip()
    value = str(value or "").strip()
    if key and value and key.lower() != value.lower():
        return f"{key} {value}"
    return key or value


class EntityBM25Index:
    """Short-document BM25Okapi (k1=1.5, b=0.75) over entity nodes."""

    def __init__(self, entities: Iterable["EntityNode"]):
        ordered = sorted(entities, key=lambda e: e.id)
        self.entity_ids = [e.id for e in ordered]
        self.entity_keys = [str(e.key or "").strip() or str(e.value or "").strip().lower() for e in ordered]
        corpus = [tokenize_for_bm25(_entity_doc_text(e.key, e.value)) or ["_empty"] for e in ordered]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._key_to_ids: Dict[str, List[str]] = {}
        for eid, ek in zip(self.entity_ids, self.entity_keys):
            nk = normalize_entity_key(ek)
            if nk:
                self._key_to_ids.setdefault(nk, []).append(eid)

    def scores_for_query(self, query: str) -> Dict[str, float]:
        """Peak-normalized BM25 scores for one query string over all entities."""
        q_tokens = tokenize_for_bm25(query)
        if not q_tokens or self._bm25 is None:
            return {eid: 0.0 for eid in self.entity_ids}
        scored = {eid: float(s) for eid, s in zip(self.entity_ids, self._bm25.get_scores(q_tokens))}
        peak = max(scored.values())
        if peak <= 0.0:
            return {eid: 0.0 for eid in scored}
        return {eid: score / peak for eid, score in scored.items()}

    def match_q_keys(
        self,
        q_keys: Iterable[str],
        *,
        min_rel_score: float = DEFAULT_ENTITY_MIN_REL_SCORE,
        top_k_per_key: Optional[int] = DEFAULT_ENTITY_TOP_K_PER_KEY,
    ) -> Dict[str, Dict[str, float]]:
        """``entity_id -> {q_key: match}``; exact key = 1.0, keep >= min_rel, top-k per key."""
        out: Dict[str, Dict[str, float]] = {}
        threshold = float(min_rel_score)
        for raw_qk in q_keys:
            qk = normalize_entity_key(raw_qk)
            if not qk:
                continue
            scored = self.scores_for_query(qk)
            for eid in self._key_to_ids.get(qk, []):
                scored[eid] = max(float(scored.get(eid, 0.0)), 1.0)
            pairs = [(eid, sc) for eid, sc in scored.items() if sc >= threshold]
            pairs.sort(key=lambda item: (-item[1], item[0]))
            if top_k_per_key is not None:
                pairs = pairs[: max(int(top_k_per_key), 0)]
            for eid, sc in pairs:
                bucket = out.setdefault(eid, {})
                bucket[qk] = max(float(bucket.get(qk, 0.0)), float(sc))
        return out


def normalize_semantic_scores(scores: Mapping[str, float], mode: str) -> Dict[str, float]:
    """Query-local calibration without changing candidate identity (EMG)."""
    if mode not in SEMANTIC_SCORE_NORMALIZATIONS:
        raise ValueError(f"unsupported semantic score normalization {mode!r}")
    clean = {str(k): float(v) for k, v in scores.items()}
    if not all(math.isfinite(v) for v in clean.values()):
        raise ValueError("semantic scores must be finite")
    if mode == SEMANTIC_SCORE_NONE or not clean:
        return clean
    low, high = min(clean.values()), max(clean.values())
    span = high - low
    if span <= 0.0:
        return {k: 0.0 for k in clean}
    return {k: (v - low) / span for k, v in clean.items()}


def _normalize_q_keys(q_keys: Iterable[str]) -> List[str]:
    return sorted({k for k in (normalize_entity_key(q) for q in q_keys) if k})


def _degree_weight(degree: int, *, enabled: bool = True) -> float:
    return 1.0 / math.log1p(float(max(int(degree), 1))) if enabled else 1.0


def _is_who(entity_type: str) -> bool:
    return str(entity_type or "").strip().lower() == "who"


# --------------------------------------------------------------------------- #
# EMG JSON helpers (build/builder.py)
# --------------------------------------------------------------------------- #

_DIA_RE = re.compile(r"^D(\d+):(\d+)$", re.I)
_LOCOMO_DATETIME_FORMAT = "%I:%M %p on %d %B, %Y"


def _parse_memory_datetime(value: str) -> Optional[datetime]:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None
    try:
        return datetime.strptime(text, _LOCOMO_DATETIME_FORMAT)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def memory_sort_key(dia_id: str, session_num: int, date_time: str) -> Tuple[int, datetime, int, int, str]:
    """EMG chronological key: parsed session datetime, then session/turn order."""
    match = _DIA_RE.match(str(dia_id or "").strip())
    session, turn = (int(match.group(1)), int(match.group(2))) if match else (int(session_num), 0)
    parsed = _parse_memory_datetime(date_time)
    if parsed is None:
        return (1, datetime.max, session, turn, dia_id)
    return (0, parsed, session, turn, dia_id)


# --------------------------------------------------------------------------- #
# Personalised PageRank
# --------------------------------------------------------------------------- #


def personalized_pagerank(
    n: int,
    edges: Iterable[Tuple[int, int, float]],
    reset: Sequence[float],
    *,
    damping: float = PPR_DAMPING,
    tol: float = PPR_TOL,
    max_iter: int = PPR_MAX_ITER,
) -> Tuple[np.ndarray, int]:
    """Undirected weighted PPR; returns ``(scores summing to 1, iterations)``.

    ``damping`` is the walk-continuation probability (igraph semantics; equal
    to networkx ``alpha``). Parallel edges add. NaN/negative reset entries are
    zeroed like HippoRAG ``run_ppr``; the reset vector is then normalised.
    Dangling (degree-0) nodes jump according to the reset vector.
    """
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")
    r = np.asarray(reset, dtype=float)
    if r.shape != (n,):
        raise ValueError(f"reset must have shape ({n},)")
    r = np.where(np.isnan(r) | (r < 0), 0.0, r)
    total = r.sum()
    if not total > 0.0 or not math.isfinite(total):
        raise ValueError("reset vector needs positive finite mass")
    r = r / total
    rows, cols, weights = [], [], []
    for u, v, w in edges:
        if u == v:
            raise ValueError("self-loops are not supported")
        if not (math.isfinite(w) and w > 0.0):
            raise ValueError("edge weights must be positive and finite")
        rows += [u, v]
        cols += [v, u]
        weights += [w, w]
    W = sp.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=float)
    out_weight = np.asarray(W.sum(axis=1)).ravel()
    dangling = out_weight == 0.0
    inv = np.divide(1.0, out_weight, out=np.zeros(n), where=~dangling)
    transition_t = (sp.diags(inv) @ W).T.tocsr()
    x = r.copy()
    for iteration in range(1, int(max_iter) + 1):
        nxt = damping * (transition_t @ x + x[dangling].sum() * r) + (1.0 - damping) * r
        err = float(np.abs(nxt - x).sum())
        x = nxt
        if err < tol:
            return x, iteration
    raise RuntimeError(f"PPR did not converge in {max_iter} iterations (L1 change {err:.3e})")


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #


@dataclass
class EntityNode:
    """Merged entity: EMG identity ``entity:{key}``; longest surface value wins."""

    id: str
    key: str
    value: str
    type: str
    merged_from: List[str] = field(default_factory=list)


class EntityMemoryGraph:
    """EMG entity--memory graph over one ``ScopedCorpus`` (immutable once built)."""

    def __init__(
        self,
        corpus: ScopedCorpus,
        entities: Mapping[str, EntityNode],
        mentions: Sequence[Tuple[str, str, float]],
        sequence: Sequence[Tuple[str, str]],
    ):
        for eid, nid, _w in mentions:
            if eid not in entities:
                raise ValueError(f"Unknown entity endpoint: {eid}")
            if nid not in corpus:
                raise ValueError(f"Unknown memory endpoint: {nid}")
        for src, dst in sequence:
            if src not in corpus or dst not in corpus:
                raise ValueError(f"Unknown memory endpoint in sequence edge {src}->{dst}")
        self.corpus = corpus
        self.entities: Dict[str, EntityNode] = dict(entities)
        self.mentions: List[Tuple[str, str, float]] = list(mentions)
        # Directed NEXT/PREV list, as EMG stores it (two per adjacent pair).
        self.sequence: List[Tuple[str, str]] = list(sequence)
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        for src, dst in self.sequence:
            self._adjacency[src].append(dst)
        self._degrees = Counter(eid for eid, _nid, _w in self.mentions)
        memories_of: Dict[str, Set[str]] = defaultdict(set)
        for eid, nid, _w in self.mentions:
            memories_of[eid].add(nid)
        self._memories_of = dict(memories_of)
        self._bm25: Optional[EntityBM25Index] = None
        self._ppr_cache: Dict[bool, Tuple[List[str], Dict[str, int], List[Tuple[int, int, float]]]] = {}

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        corpus: ScopedCorpus,
        mentions: Mapping[str, Sequence[Any]],
        add_speaker_as_entity: bool = True,
    ) -> "EntityMemoryGraph":
        """Build from per-turn ``EntityMention``s; sequence edges follow corpus order."""
        unknown = sorted(set(mentions) - set(corpus.index_of))
        if unknown:
            raise ValueError(f"mentions reference {len(unknown)} node(s) outside the corpus, e.g. {unknown[0]!r}")
        entities: Dict[str, EntityNode] = {}
        edges: List[Tuple[str, str, float]] = []
        edge_keys: Set[Tuple[str, str]] = set()
        for turn in corpus.turns:
            values: List[Tuple[str, str, str]] = []
            for m in mentions.get(turn.node_id, ()):
                value = str(m.value).strip()
                if value and m.key:
                    values.append((m.key, value, m.type))
            speaker = str(turn.speaker or "").strip()
            if add_speaker_as_entity and speaker:
                speaker_key = normalize_entity_key(speaker)
                if speaker_key and speaker_key not in {k for k, _, _ in values}:
                    values.append((speaker_key, speaker, "Who"))
            for key, value, entity_type in values:
                eid = f"entity:{key}"
                node = entities.get(eid)
                if node is None:
                    entities[eid] = EntityNode(eid, key, value, entity_type, [value])
                else:
                    if value not in node.merged_from:
                        node.merged_from.append(value)
                    if len(value) > len(node.value):
                        node.value = value
                    if not node.type and entity_type:
                        node.type = entity_type
                if (eid, turn.node_id) not in edge_keys:
                    edge_keys.add((eid, turn.node_id))
                    edges.append((eid, turn.node_id, 1.0))
        ids = [t.node_id for t in corpus.turns]
        sequence: List[Tuple[str, str]] = []
        for left, right in zip(ids, ids[1:]):
            sequence += [(left, right), (right, left)]
        return cls(corpus, entities, edges, sequence)

    @classmethod
    def from_emg_json(
        cls,
        path_or_dict: Union[str, Path, Mapping[str, Any]],
        *,
        node_id_format: str = "locomo:{sample_id}:{dia_id}",
        corpus: Optional[ScopedCorpus] = None,
    ) -> "EntityMemoryGraph":
        """Load an upstream ``conv-*_em_graph_*.json`` exactly (entities, edges, NEXT/PREV).

        Memory ``memory:{dia_id}`` becomes ``node_id_format.format(sample_id, dia_id)``
        with ``evidence_id = dia_id``. Pass ``corpus`` to reuse an existing scope;
        its node ids must equal the graph's memories exactly.
        """
        if isinstance(path_or_dict, Mapping):
            data = path_or_dict
        else:
            data = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
        sample_id = str(data.get("sample_id", ""))
        node_of: Dict[str, str] = {}
        turns: List[TurnRecord] = []
        for payload in (data.get("memories") or {}).values():
            dia_id = str(payload.get("dia_id", ""))
            session_num = int(payload.get("session_num", 0) or 0)
            date_time = str(payload.get("date_time", ""))
            node_id = node_id_format.format(sample_id=sample_id, dia_id=dia_id)
            node_of[str(payload["id"])] = node_id
            parsed = _parse_memory_datetime(date_time)
            turns.append(TurnRecord(
                node_id=node_id,
                evidence_id=dia_id,
                text=str(payload.get("text", "")),
                session_key=f"{sample_id}:S{session_num}",
                order=memory_sort_key(dia_id, session_num, date_time),
                speaker=str(payload.get("speaker", "")),
                timestamp=parsed.isoformat() if parsed else None,
                metadata={
                    "dia_id": dia_id, "session_number": session_num, "session_date_time": date_time,
                    "text_normalized": str(payload.get("text_normalized", "")),
                    "blip_caption": str(payload.get("blip_caption", "")), "sample_id": sample_id,
                },
            ))
        if corpus is None:
            corpus = ScopedCorpus(turns, scope_key=f"emg:{sample_id}")
        elif set(corpus.index_of) != set(node_of.values()):
            raise ValueError("corpus node ids do not match the EMG graph memories")
        entities = {
            str(p["id"]): EntityNode(
                str(p["id"]), str(p.get("key", "")), str(p.get("value", "")), str(p.get("type", "")),
                [str(x) for x in (p.get("merged_from") or [])],
            )
            for p in (data.get("entities") or {}).values()
        }
        edges = []
        for p in data.get("edges") or []:
            if p.get("edge_type", "mentions") != "mentions":
                raise ValueError(f"Mentions list only accepts MENTIONS edges, got {p.get('edge_type')}")
            memory_id = str(p["memory_id"])
            if memory_id not in node_of:
                raise ValueError(f"Unknown memory endpoint: {memory_id}")
            edges.append((str(p["entity_id"]), node_of[memory_id], float(p.get("weight", 1.0) or 1.0)))
        sequence = []
        for p in data.get("memory_edges") or []:
            if p.get("edge_type", "next") not in ("next", "prev"):
                raise ValueError(f"Memory edges only accept NEXT/PREV, got {p.get('edge_type')}")
            src, dst = str(p["src_memory_id"]), str(p["dst_memory_id"])
            if src not in node_of or dst not in node_of:
                raise ValueError(f"Unknown memory endpoint in {src}->{dst}")
            sequence.append((node_of[src], node_of[dst]))
        return cls(corpus, entities, edges, sequence)

    def stats(self) -> Dict[str, Any]:
        return {
            "memories": len(self.corpus),
            "entities": len(self.entities),
            "mention_edges": len(self.mentions),
            "sequence_pairs": len(self._sequence_pairs()),
            "entity_types": dict(sorted(Counter(e.type for e in self.entities.values()).items())),
        }

    @property
    def bm25(self) -> EntityBM25Index:
        if self._bm25 is None:
            self._bm25 = EntityBM25Index(self.entities.values())
        return self._bm25

    # -- EMG recall -----------------------------------------------------------

    def _match(self, q_keys: Iterable[str], min_rel_score: float, top_k_per_key: Optional[int]):
        keys = _normalize_q_keys(q_keys)
        if not keys or not self.entities:
            return {}, {}
        entity_to_q = self.bm25.match_q_keys(keys, min_rel_score=min_rel_score, top_k_per_key=top_k_per_key)
        effective = {qk for scores in entity_to_q.values() for qk in scores}
        if not effective:
            return {}, {}
        denom = float(len(effective))
        strength = {eid: sum(qs[k] for k in sorted(qs)) / denom for eid, qs in entity_to_q.items()}
        return entity_to_q, strength

    def entity_scores(
        self,
        q_keys: Iterable[str],
        *,
        min_rel_score: float = DEFAULT_ENTITY_MIN_REL_SCORE,
        top_k_per_key: Optional[int] = DEFAULT_ENTITY_TOP_K_PER_KEY,
        who_only_dampen: float = WHO_ONLY_DAMPEN,
        degree_discount: bool = True,
    ) -> Dict[str, float]:
        """EMG ``_entity_memory_scores``: ``{node_id: seed score}`` (all > 0)."""
        entity_to_q, strength = self._match(q_keys, min_rel_score, top_k_per_key)
        if not strength:
            return {}
        entity_raw = {
            eid: s * _degree_weight(self._degrees.get(eid, 1), enabled=bool(degree_discount))
            for eid, s in strength.items()
        }
        who_q_keys: Set[str] = set()
        for eid, q_scores in entity_to_q.items():
            ent = self.entities.get(eid)
            if ent is not None and _is_who(ent.type):
                who_q_keys |= set(q_scores)
        who_scores: Dict[str, float] = defaultdict(float)
        content_scores: Dict[str, float] = defaultdict(float)
        for eid, nid, weight in self.mentions:
            e_score = entity_raw.get(eid)
            if not e_score:
                continue
            scored = float(e_score) * float(weight or 1.0)
            ent = self.entities.get(eid)
            matched = set(entity_to_q.get(eid, {}))
            if (ent is not None and _is_who(ent.type)) or (bool(matched) and matched <= who_q_keys):
                who_scores[nid] = max(who_scores[nid], scored)
            else:
                content_scores[nid] = max(content_scores[nid], scored)
        out: Dict[str, float] = {}
        for nid in set(who_scores) | set(content_scores):
            content_e = content_scores.get(nid, 0.0)
            who_e = who_scores.get(nid, 0.0)
            out[nid] = max(content_e, who_e) if content_e > 0.0 else who_e * float(who_only_dampen)
        return out

    def expand_sequence_neighbors(
        self, seed_scores: Mapping[str, float], *, secondary_scale: float = SEQUENCE_SECONDARY_SCALE
    ) -> Dict[str, float]:
        """Give +/-1 chronological neighbours ``max(existing, scale * seed)`` (one hop)."""
        if not seed_scores:
            return {}
        scale = float(secondary_scale)
        if scale <= 0.0:
            return dict(seed_scores)
        out = dict(seed_scores)
        for nid, score in seed_scores.items():
            if score <= 0.0:
                continue
            for neigh in self._adjacency.get(nid, ()):
                out[neigh] = max(out.get(neigh, 0.0), float(score) * scale)
        return out

    def _sort(self, scores: Mapping[str, float], upstream_order: bool) -> Ranking:
        if upstream_order:
            return sorted(scores.items(), key=lambda it: (-it[1], self.corpus.turn(it[0]).evidence_id))
        return self.corpus.sort_scores(scores)

    def emg_fused_rank(
        self,
        q_keys: Iterable[str],
        semantic_scores: Optional[SemanticScoreFn],
        *,
        top_k: int,
        entity_weight: float = DEFAULT_ENTITY_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        semantic_normalization: str = SEMANTIC_SCORE_NONE,
        expand_sequence: bool = True,
        sequence_secondary_scale: float = SEQUENCE_SECONDARY_SCALE,
        min_rel_score: float = DEFAULT_ENTITY_MIN_REL_SCORE,
        top_k_per_key: Optional[int] = DEFAULT_ENTITY_TOP_K_PER_KEY,
        who_only_dampen: float = WHO_ONLY_DAMPEN,
        degree_discount: bool = True,
        fill: bool = True,
        upstream_order: bool = False,
    ) -> Ranking:
        """EMG ``retrieve_dialog_ids``: gate, ``ew * E + sw * S`` in the gate, dense fill.

        ``semantic_scores(ids)`` returns ``{node_id: score}`` for ``ids`` (a set)
        or for the whole scope when ``ids`` is None; it is not called when
        ``semantic_weight == 0``. An empty gate ranks the full scope by
        ``sw * S``. ``fill=False`` stops after the gated pool.
        """
        if semantic_normalization not in SEMANTIC_SCORE_NORMALIZATIONS:
            raise ValueError(f"unsupported semantic score normalization {semantic_normalization!r}")
        use_semantic = float(semantic_weight) != 0.0
        if use_semantic and semantic_scores is None:
            raise ValueError("semantic_weight != 0 requires a semantic_scores callable")
        requested = max(int(top_k), 0)
        if not self.corpus.turns or requested == 0:
            return []
        seeds = self.entity_scores(
            q_keys, min_rel_score=min_rel_score, top_k_per_key=top_k_per_key,
            who_only_dampen=who_only_dampen, degree_discount=degree_discount,
        )
        e_scores = (
            self.expand_sequence_neighbors(seeds, secondary_scale=sequence_secondary_scale)
            if expand_sequence else seeds
        )
        candidates = set(e_scores)
        gated = bool(candidates)

        def semantic(ids: Optional[Set[str]]) -> Dict[str, float]:
            if not use_semantic:
                return {}
            return normalize_semantic_scores(semantic_scores(ids), semantic_normalization)

        s_scores = semantic(candidates if gated else None)
        pool = candidates if gated else set(self.corpus.index_of)
        ranked = self._sort(
            {nid: float(entity_weight) * e_scores.get(nid, 0.0) + float(semantic_weight) * s_scores.get(nid, 0.0)
             for nid in pool},
            upstream_order,
        )
        target = min(requested, len(self.corpus))
        if fill and gated and len(ranked) < target:
            full = semantic(None)
            rest = {nid: float(semantic_weight) * full.get(nid, 0.0) for nid in self.corpus.index_of if nid not in candidates}
            ranked.extend(self._sort(rest, upstream_order)[: target - len(ranked)])
        return ranked[:target]

    def emg_rank(
        self,
        q_keys: Iterable[str],
        *,
        top_k: Optional[int] = None,
        expand_sequence: bool = True,
        sequence_secondary_scale: float = SEQUENCE_SECONDARY_SCALE,
        min_rel_score: float = DEFAULT_ENTITY_MIN_REL_SCORE,
        top_k_per_key: Optional[int] = DEFAULT_ENTITY_TOP_K_PER_KEY,
        who_only_dampen: float = WHO_ONLY_DAMPEN,
        degree_discount: bool = True,
        fill: bool = False,
        upstream_order: bool = False,
    ) -> Ranking:
        """Graph-only ranking by E alone (EMG ``B_entity``: weights 1.0/0.0).

        Default returns only positive candidates. ``fill=True`` reproduces
        EMG's entity-only output exactly (zero-score fill up to ``top_k``,
        default the whole scope); combine with ``upstream_order=True`` for
        EMG's dia_id tie-break.
        """
        kwargs = dict(
            expand_sequence=expand_sequence, sequence_secondary_scale=sequence_secondary_scale,
            min_rel_score=min_rel_score, top_k_per_key=top_k_per_key,
            who_only_dampen=who_only_dampen, degree_discount=degree_discount, upstream_order=upstream_order,
        )
        limit = len(self.corpus) if top_k is None else top_k
        if fill:
            return self.emg_fused_rank(
                q_keys, None, top_k=limit, entity_weight=1.0, semantic_weight=0.0, fill=True, **kwargs
            )
        ranked = self.emg_fused_rank(q_keys, None, top_k=len(self.corpus), entity_weight=1.0,
                                     semantic_weight=0.0, fill=False, **kwargs)
        return [(nid, s) for nid, s in ranked if s > 0.0][: max(int(limit), 0)]

    # -- HippoRAG-2-style PPR -------------------------------------------------

    def _sequence_pairs(self) -> List[Tuple[str, str]]:
        index = self.corpus.index_of
        pairs = {tuple(sorted((a, b), key=index.__getitem__)) for a, b in self.sequence if a != b}
        return sorted(pairs, key=lambda p: (index[p[0]], index[p[1]]))

    def _ppr_structure(self, include_sequence_edges: bool):
        cached = self._ppr_cache.get(include_sequence_edges)
        if cached is None:
            nodes = sorted(self.entities) + [t.node_id for t in self.corpus.turns]
            position = {node: i for i, node in enumerate(nodes)}
            edges = [(position[eid], position[nid], float(w)) for eid, nid, w in self.mentions]
            if include_sequence_edges:
                edges += [(position[a], position[b], 1.0) for a, b in self._sequence_pairs()]
            cached = self._ppr_cache[include_sequence_edges] = (nodes, position, edges)
        return cached

    def ppr_seeds(
        self,
        q_keys: Iterable[str],
        *,
        linking_top_k: Optional[int] = PPR_LINKING_TOP_K,
        min_rel_score: float = DEFAULT_ENTITY_MIN_REL_SCORE,
        top_k_per_key: Optional[int] = DEFAULT_ENTITY_TOP_K_PER_KEY,
    ) -> Dict[str, float]:
        """Entity reset weights: ``strength(e) / |memories(e)|`` for the top linking_top_k."""
        _entity_to_q, strength = self._match(q_keys, min_rel_score, top_k_per_key)
        ranked = sorted(strength.items(), key=lambda it: (-it[1], it[0]))
        if linking_top_k:
            ranked = ranked[: int(linking_top_k)]
        out = {}
        for eid, s in ranked:
            n_memories = len(self._memories_of.get(eid, ()))
            out[eid] = s / n_memories if n_memories else s
        return out

    def ppr_rank(
        self,
        q_keys: Iterable[str],
        *,
        damping: float = PPR_DAMPING,
        linking_top_k: Optional[int] = PPR_LINKING_TOP_K,
        include_sequence_edges: bool = True,
        passage_scores: Optional[Mapping[str, float]] = None,
        passage_node_weight: float = PPR_PASSAGE_NODE_WEIGHT,
        min_rel_score: float = DEFAULT_ENTITY_MIN_REL_SCORE,
        top_k_per_key: Optional[int] = DEFAULT_ENTITY_TOP_K_PER_KEY,
        top_k: Optional[int] = None,
        tol: float = PPR_TOL,
        max_iter: int = PPR_MAX_ITER,
    ) -> Ranking:
        """Rank memories by PPR mass from entity (and optional passage) seeds.

        ``passage_scores`` (``{node_id: score}``, normally the whole scope) are
        min-max normalised and scaled by ``passage_node_weight`` (HippoRAG 2).
        Returns positive-mass memories only; no seed mass returns ``[]``.
        """
        nodes, position, edges = self._ppr_structure(bool(include_sequence_edges))
        reset = np.zeros(len(nodes))
        for eid, weight in self.ppr_seeds(
            q_keys, linking_top_k=linking_top_k, min_rel_score=min_rel_score, top_k_per_key=top_k_per_key
        ).items():
            reset[position[eid]] += weight
        if passage_scores:
            unknown = [nid for nid in passage_scores if nid not in self.corpus]
            if unknown:
                raise ValueError(f"passage_scores reference nodes outside the corpus, e.g. {unknown[0]!r}")
            values = np.array([float(passage_scores[nid]) for nid in passage_scores])
            if not np.isfinite(values).all():
                raise ValueError("passage_scores must be finite")
            span = values.max() - values.min()
            if span > 0.0:
                for nid, v in zip(passage_scores, values):
                    reset[position[nid]] += (v - values.min()) / span * float(passage_node_weight)
        if not reset.sum() > 0.0:
            return []
        scores, _iterations = personalized_pagerank(
            len(nodes), edges, reset, damping=damping, tol=tol, max_iter=max_iter
        )
        memory_scores = {
            t.node_id: float(scores[position[t.node_id]])
            for t in self.corpus.turns
            if scores[position[t.node_id]] > 0.0
        }
        ranked = self.corpus.sort_scores(memory_scores)
        return ranked if top_k is None else ranked[: max(int(top_k), 0)]
